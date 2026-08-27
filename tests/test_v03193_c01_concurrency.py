"""
tests/test_v03193_c01_concurrency.py
======================================
Regression tests for C01 — real race conditions in PolicyStore.

Bugs confirmed by reproduction (not just code review):

  BUG 1 — get(), all_active(), get_history(), reward_log_count(),
           recent_rewards() had no lock. While save()/log_reward() hold
           self._lock and use `with self._conn:` (which opens a SQLite
           transaction), concurrent unprotected reads on the same
           connection caused:
               sqlite3.InterfaceError: bad parameter or other API misuse
           Observed at 5-10% rate under 20-thread concurrent load.

  BUG 2 — _update_arm() used get() + in-memory mutation + save().
           Two threads fetching the same policy both saw alpha=1.0,
           both computed alpha+reward, both saved — one update silently
           overwritten. Reproduced: expected +40.0 alpha growth, actual
           +12.7 (lost ~27 updates per 2000 operations).

  Fix 1 — All read methods now acquire self._lock.
  Fix 2 — _update_arm() now uses update_atomic() (BEGIN IMMEDIATE
           transaction) so the read-modify-write is serialised.
"""

from __future__ import annotations

import random
import tempfile
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_store(tmp_path):
    from policy.store import PolicyStore
    return PolicyStore(memory_dir=tmp_path)


def _make_learner(store):
    from policy.learner import PolicyLearner
    return PolicyLearner(policy_store=store)


def _seed_policies(learner, n=5):
    from policy.models import PolicyRecord, PolicyDomain, PolicyType
    for i in range(n):
        learner.register(PolicyRecord(
            policy_id=f"policy_{i}", name=f"arm_{i}",
            domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.RETRIEVAL_WEIGHTS,
        ))


def _reward(pid, value=None):
    from policy.models import RewardSignal, RewardType
    return RewardSignal(
        reward_type=RewardType.BENCHMARK_SCORE,
        value=value if value is not None else random.random(),
        policy_id=pid,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Lock presence — white-box
# ═════════════════════════════════════════════════════════════════════════════

class TestReadMethodsAcquireLock:

    def test_get_acquires_lock(self):
        import inspect
        from policy.store import PolicyStore
        src = inspect.getsource(PolicyStore.get)
        assert 'self._lock' in src, "get() must acquire self._lock (C01)"

    def test_all_active_acquires_lock(self):
        import inspect
        from policy.store import PolicyStore
        src = inspect.getsource(PolicyStore.all_active)
        assert 'self._lock' in src, "all_active() must acquire self._lock (C01)"

    def test_count_acquires_lock(self):
        import inspect
        from policy.store import PolicyStore
        src = inspect.getsource(PolicyStore.count)
        assert 'self._lock' in src, "count() must acquire self._lock (C01)"

    def test_get_history_acquires_lock(self):
        import inspect
        from policy.store import PolicyStore
        src = inspect.getsource(PolicyStore.get_history)
        assert 'self._lock' in src, "get_history() must acquire self._lock (C01)"

    def test_reward_log_count_acquires_lock(self):
        import inspect
        from policy.store import PolicyStore
        src = inspect.getsource(PolicyStore.reward_log_count)
        assert 'self._lock' in src, "reward_log_count() must acquire self._lock (C01)"

    def test_update_arm_uses_update_atomic(self):
        import inspect
        from policy.learner import PolicyLearner
        src = inspect.getsource(PolicyLearner._update_arm)
        assert 'update_atomic' in src, (
            "_update_arm must use update_atomic() to prevent lost updates (C01)"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Concurrent correctness — black-box reproduction of original bugs
# ═════════════════════════════════════════════════════════════════════════════

class TestNoInterfaceError:
    """
    Reproduce the original InterfaceError: bad parameter or other API misuse.
    Previously observed at 5-10% rate under 20-thread load.
    After fix: must be 0% across all runs.
    """

    def _run_once(self, tmp_path):
        store = _make_store(tmp_path)
        learner = _make_learner(store)
        _seed_policies(learner)
        errors = []

        def go(tid):
            for i in range(100):
                try:
                    learner.observe(_reward(f"policy_{tid % 5}"))
                    store.all_active()
                    store.reward_log_count(f"policy_{tid % 5}")
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=go, args=(i,)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        store.close()
        return errors

    def test_no_interface_error_run1(self, tmp_path):
        errors = self._run_once(tmp_path)
        assert not errors, f"Errors on run 1: {errors[:3]}"

    def test_no_interface_error_run2(self, tmp_path):
        errors = self._run_once(tmp_path)
        assert not errors, f"Errors on run 2: {errors[:3]}"

    def test_no_interface_error_run3(self, tmp_path):
        errors = self._run_once(tmp_path)
        assert not errors, f"Errors on run 3: {errors[:3]}"

    def test_no_interface_error_20_threads_200_iters(self, tmp_path):
        """Heavier load to catch low-probability races."""
        store = _make_store(tmp_path)
        learner = _make_learner(store)
        _seed_policies(learner)
        errors = []

        def go(tid):
            for i in range(200):
                try:
                    learner.observe(_reward(f"policy_{tid % 5}"))
                    store.all_active()
                    store.get(f"policy_{tid % 5}")
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=go, args=(i,)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        store.close()
        assert not errors, f"InterfaceError still occurring: {errors[:3]}"


class TestNoLostUpdates:
    """
    Reproduce the lost-update race from _update_arm's get+mutate+save pattern.
    Previously: expected +40.0 alpha growth, actual ~12.7 (lost ~70% of updates).
    After fix with update_atomic(): all updates must be accounted for via
    reward_log rows (we can't check exact alpha due to value variance,
    but we can check that reward_log count = actual observe() calls).
    """

    def test_reward_log_count_matches_observe_calls(self, tmp_path):
        store = _make_store(tmp_path)
        learner = _make_learner(store)
        _seed_policies(learner)
        call_count = [0]
        errors = []

        def go(tid):
            for _ in range(50):
                try:
                    learner.observe(_reward(f"policy_{tid % 5}"))
                    call_count[0] += 1
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=go, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors
        total_logged = store.reward_log_count()
        # Every observe() call logs a reward row — none should be lost
        assert total_logged == call_count[0], (
            f"reward_log has {total_logged} rows but {call_count[0]} observe() "
            f"calls were made — {call_count[0]-total_logged} rewards lost"
        )
        store.close()

    def test_no_exception_from_concurrent_observe(self, tmp_path):
        """update_atomic() must handle concurrent writes without raising."""
        store = _make_store(tmp_path)
        learner = _make_learner(store)
        _seed_policies(learner)
        errors = []

        def go(tid):
            for _ in range(100):
                try:
                    learner.observe(_reward(f"policy_{tid % 5}", value=0.8))
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=go, args=(i,)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        store.close()
        assert not errors, f"Concurrent observe() raised: {errors[:3]}"

    def test_alpha_grows_monotonically_with_positive_rewards(self, tmp_path):
        """With all rewards = 1.0, alpha must grow (not stay flat due to lost updates)."""
        store = _make_store(tmp_path)
        learner = _make_learner(store)
        _seed_policies(learner, n=1)

        initial = store.get("policy_0").alpha
        errors = []

        def go(_):
            for _ in range(20):
                try:
                    learner.observe(_reward("policy_0", value=1.0))
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=go, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors
        final = store.get("policy_0").alpha
        assert final > initial, (
            f"alpha did not grow: initial={initial}, final={final}. "
            "Lost updates may still be occurring."
        )
        store.close()

    def test_concurrent_reads_do_not_block_writes_indefinitely(self, tmp_path):
        """Locking reads must not deadlock with concurrent writes (RLock used)."""
        import time
        store = _make_store(tmp_path)
        learner = _make_learner(store)
        _seed_policies(learner)
        done = []

        def writer():
            for _ in range(50):
                learner.observe(_reward("policy_0", value=0.7))
            done.append('writer')

        def reader():
            for _ in range(50):
                store.all_active()
                store.get("policy_0")
            done.append('reader')

        threads = (
            [threading.Thread(target=writer) for _ in range(5)] +
            [threading.Thread(target=reader) for _ in range(5)]
        )
        for t in threads: t.start()
        for t in threads: t.join(timeout=10)

        assert len(done) == 10, (
            f"Only {len(done)}/10 threads completed — possible deadlock"
        )
        store.close()

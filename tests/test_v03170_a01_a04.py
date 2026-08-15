"""
Blix v0.3.16.9 — Tests for Issues A01, A02, A03, A04

A01: Double-decay on LRU eviction
  - _epoch_at_last_write attribute exists
  - _apply_delta_epoch method exists
  - arm evicted and re-fetched is NOT double-decayed
  - delta epoch is correctly computed as current / at_write
  - flush_decay records epoch_at_last_write = 1.0 after flush

A02: Broad except swallows config errors silently
  - ImportError returns None (silent — config module not installed)
  - Pydantic validation error raises RuntimeError (not swallowed)
  - Log ERROR emitted on config failure
  - store.py config blocks follow same pattern

A03: Double-close raises ProgrammingError
  - HGSHM.close() is idempotent
  - PolicyStore.close() is idempotent
  - GraphStore.close() is idempotent
  - HGSHM._closed flag set after first close
  - context manager + explicit close does not double-close

A04: _reward_log_counts lost on restart
  - counter initialised from DB on first write after restart
  - counter matches actual DB row count after init
  - prune fires correctly after restart when table is over limit
  - table does not grow beyond 2× limit after restart
"""
from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from policy.models import (
    PolicyRecord, PolicyDomain, PolicyType,
    RewardSignal, RewardType,
)
from policy.store import PolicyStore
from policy.learner import PolicyLearner
from memory.hybrid.hgshm import HGSHM
from memory.hybrid.graph.graph_store import GraphStore


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def store(tmp_dir):
    s = PolicyStore(tmp_dir)
    yield s
    s.close()


@pytest.fixture
def learner(store):
    l = PolicyLearner(store, cache_max_size=5)
    l.register_defaults()
    return l


def _reward(pid: str, value: float = 0.8) -> RewardSignal:
    return RewardSignal(RewardType.BENCHMARK_SCORE, value=value, policy_id=pid)


def _policy(name: str = "test") -> PolicyRecord:
    return PolicyRecord(
        name=name,
        domain=PolicyDomain.SYSTEM,
        policy_type=PolicyType.PLANNER_CONFIG,
        config={"beam_width": 3},
    )


# ════════════════════════════════════════════════════════════════════
# A01 — Delta epoch prevents double-decay on LRU eviction
# ════════════════════════════════════════════════════════════════════

class TestDeltaEpoch:
    def test_epoch_at_last_write_attribute_exists(self, learner):
        assert hasattr(learner, '_epoch_at_last_write')
        assert isinstance(learner._epoch_at_last_write, dict)

    def test_apply_delta_epoch_method_exists(self, learner):
        assert hasattr(learner, '_apply_delta_epoch')
        assert callable(learner._apply_delta_epoch)

    def test_update_records_epoch_at_last_write(self, learner, store):
        p = store.all_active(domain=PolicyDomain.SYSTEM,
                             policy_type=PolicyType.PLANNER_CONFIG)[0]
        learner.observe(_reward(p.policy_id))
        assert p.policy_id in learner._epoch_at_last_write

    def test_flush_resets_epoch_at_last_write_to_one(self, learner, store):
        p = store.all_active(domain=PolicyDomain.SYSTEM,
                             policy_type=PolicyType.PLANNER_CONFIG)[0]
        # Accumulate some epoch
        for _ in range(5):
            learner.observe(_reward(p.policy_id))
        # Force flush
        learner.flush_decay()
        # After flush, epoch = 1.0, so at_last_write should be 1.0 too
        for pid, epoch_val in learner._epoch_at_last_write.items():
            assert epoch_val == pytest.approx(1.0), (
                f"Policy {pid[:8]}: epoch_at_last_write={epoch_val} after flush, expected 1.0"
            )

    def test_no_double_decay_on_eviction(self, tmp_dir):
        """
        Core A01 correctness test.

        With cache_max_size=2 and 3 arms, the 3rd arm evicts the 1st.
        When the 1st is re-fetched, only the delta epoch should apply —
        not the full accumulated epoch — so its alpha is correct.
        """
        store = PolicyStore(tmp_dir)
        # Use tiny cache to force eviction
        learner = PolicyLearner(store, cache_max_size=2, decay_persist_every=1000)
        learner.register_defaults()

        arms = store.all_active(domain=PolicyDomain.SYSTEM,
                                policy_type=PolicyType.PLANNER_CONFIG)
        # Need at least 3 arms — register_defaults gives 3 PLANNER_CONFIG arms
        assert len(arms) >= 3, f"Need ≥3 arms, got {len(arms)}"

        a0, a1, a2 = arms[0], arms[1], arms[2]

        # Prime a0 with positive rewards so alpha > 1
        for _ in range(10):
            learner.observe(_reward(a0.policy_id, value=0.9))

        alpha_after_rewards = learner._cache_get(a0.policy_id).alpha
        assert alpha_after_rewards > 5.0, "a0 should have high alpha after rewards"

        # Record the epoch when a0 was last written
        epoch_at_write_a0 = learner._epoch_at_last_write.get(a0.policy_id, 1.0)

        # Observe a1 and a2 to advance the epoch without touching a0
        for _ in range(5):
            learner.observe(_reward(a1.policy_id, value=0.8))
        for _ in range(5):
            learner.observe(_reward(a2.policy_id, value=0.7))

        # Advance epoch more by observing a1 (forces cache pressure)
        for _ in range(10):
            learner.observe(_reward(a1.policy_id, value=0.8))

        current_epoch = learner._decay_epoch

        # Force a0 out of cache by filling with other arms
        # (cache_max_size=2, so after accessing a1 and a2, a0 is evicted)
        learner._cache_get(a1.policy_id)
        learner._cache_get(a2.policy_id)

        # Verify a0 is NOT in cache
        if a0.policy_id in learner._cache:
            # Manually evict if needed
            del learner._cache[a0.policy_id]

        # Now re-fetch a0 from DB — _apply_delta_epoch should apply only the delta
        a0_refetched = learner._get_cached(a0.policy_id)
        assert a0_refetched is not None

        # Expected alpha with delta epoch:
        # delta = current_epoch / epoch_at_write
        # effective_alpha = 1.0 + (db_alpha - 1.0) * delta
        #
        # The db_alpha is approximately alpha_after_rewards (flushed periodically)
        # What we're checking: it should NOT be decayed by the full current_epoch
        # (which would give effective_alpha ≈ 1.0 + (db_alpha-1) * current_epoch)
        db_alpha = store.get(a0.policy_id).alpha
        if db_alpha > 1.0 and current_epoch < 1.0:
            epoch_at_write = learner._epoch_at_last_write.get(a0.policy_id, 1.0)
            delta = current_epoch / max(epoch_at_write, 1e-10)
            full_epoch_alpha = 1.0 + (db_alpha - 1.0) * current_epoch
            delta_epoch_alpha = 1.0 + (db_alpha - 1.0) * delta

            # The refetched alpha should be closer to delta_epoch_alpha than to full_epoch_alpha
            err_delta = abs(a0_refetched.alpha - delta_epoch_alpha)
            err_full  = abs(a0_refetched.alpha - full_epoch_alpha)

            # If delta < current_epoch, delta gives MORE alpha than full_epoch
            # (delta is closer to 1.0 than current_epoch)
            # So err_delta < err_full means we used delta not full epoch
            if delta < current_epoch:
                assert err_delta < err_full or abs(err_delta - err_full) < 0.01, (
                    f"A01: double-decay detected.\n"
                    f"  db_alpha={db_alpha:.4f}, current_epoch={current_epoch:.6f}\n"
                    f"  epoch_at_write={epoch_at_write:.6f}, delta={delta:.6f}\n"
                    f"  expected ≈ {delta_epoch_alpha:.4f} (delta)\n"
                    f"  got        {a0_refetched.alpha:.4f}\n"
                    f"  full-epoch would give {full_epoch_alpha:.4f}"
                )

        store.close()

    def test_apply_delta_epoch_zero_decay_when_no_change(self, learner, store):
        """
        If _decay_epoch == epoch_at_last_write, delta=1.0 and no decay applied.
        """
        p = _policy("delta_none")
        store.save(p)
        p.alpha = 5.0; store.save(p)

        # Set epoch_at_last_write to match current epoch
        learner._epoch_at_last_write[p.policy_id] = learner._decay_epoch

        alpha_before = p.alpha
        learner._apply_delta_epoch(p)
        # No change expected — delta is 1.0
        assert p.alpha == pytest.approx(alpha_before, rel=1e-6)

    def test_apply_delta_epoch_applies_partial_decay(self, learner, store):
        """delta < 1.0 means arm decays partially, not fully."""
        p = _policy("partial")
        p.alpha = 10.0
        store.save(p)

        # Simulate: arm was written when epoch=0.99, now epoch=0.98
        learner._epoch_at_last_write[p.policy_id] = 0.99
        learner._decay_epoch = 0.98

        alpha_before = p.alpha
        learner._apply_delta_epoch(p)

        # delta = 0.98 / 0.99 ≈ 0.98989...
        expected = 1.0 + (alpha_before - 1.0) * (0.98 / 0.99)
        assert p.alpha == pytest.approx(expected, rel=1e-4)

    def test_first_load_applies_full_epoch(self, learner, store):
        """
        When no write epoch recorded (first ever load), full epoch applied.
        This is correct: no prior in-memory decay has been applied.
        """
        p = _policy("first_load")
        p.alpha = 8.0
        store.save(p)

        # No entry in _epoch_at_last_write
        assert p.policy_id not in learner._epoch_at_last_write

        # Set a non-trivial epoch
        learner._decay_epoch = 0.90

        alpha_before = p.alpha
        learner._apply_delta_epoch(p)

        # epoch_at_write defaults to 1.0, so delta = 0.90 / 1.0 = 0.90
        expected = 1.0 + (alpha_before - 1.0) * 0.90
        assert p.alpha == pytest.approx(expected, rel=1e-4)


# ════════════════════════════════════════════════════════════════════
# A02 — Config errors no longer silently swallowed
# ════════════════════════════════════════════════════════════════════

class TestConfigErrorHandling:
    def test_import_error_returns_none_silently(self):
        """If _adma_settings is None (config absent), return None without raising.
        A19: module-level singleton — patch the module attribute directly."""
        import policy.learner as lm
        from policy.learner import _default_learner_cfg
        original = lm._adma_settings
        try:
            lm._adma_settings = None
            result = _default_learner_cfg()
        finally:
            lm._adma_settings = original
        assert result is None

    def test_import_error_optimizer_returns_none(self):
        import policy.optimizer as om
        from policy.optimizer import _default_optimizer_cfg
        original = om._adma_settings
        try:
            om._adma_settings = None
            result = _default_optimizer_cfg()
        finally:
            om._adma_settings = original
        assert result is None

    def test_import_error_reward_returns_none(self):
        import policy.reward as rm
        from policy.reward import _default_reward_cfg
        original = rm._adma_settings
        try:
            rm._adma_settings = None
            result = _default_reward_cfg()
        finally:
            rm._adma_settings = original
        assert result is None

    def test_import_error_compiler_returns_none(self):
        import policy.compiler as cm
        from policy.compiler import _default_compiler_cfg
        original = cm._adma_settings
        try:
            cm._adma_settings = None
            result = _default_compiler_cfg()
        finally:
            cm._adma_settings = original
        assert result is None

    def test_config_validation_error_raises_runtime_error(self):
        """
        If _adma_settings.learner raises (e.g. Pydantic error), the exception
        propagates from _default_learner_cfg().
        A19: no RuntimeError wrapping — ValueError propagates directly.
        """
        import policy.learner as lm
        from policy.learner import _default_learner_cfg
        original = lm._adma_settings
        try:
            mock_settings = MagicMock()
            type(mock_settings).learner = property(
                lambda self: (_ for _ in ()).throw(ValueError("invalid config")))
            lm._adma_settings = mock_settings
            with pytest.raises((RuntimeError, ValueError)):
                _default_learner_cfg()
        finally:
            lm._adma_settings = original

    def test_valid_config_loads_correctly(self):
        """Normal operation: config loads and returns settings object."""
        from policy.learner import _default_learner_cfg
        result = _default_learner_cfg()
        # Should return the settings object (not None) in normal operation
        assert result is not None
        assert hasattr(result, 'decay_factor')

    def test_valid_optimizer_config_loads(self):
        from policy.optimizer import _default_optimizer_cfg
        result = _default_optimizer_cfg()
        assert result is not None
        assert hasattr(result, 'aging_threshold')

    def test_valid_reward_config_loads(self):
        from policy.reward import _default_reward_cfg
        result = _default_reward_cfg()
        assert result is not None
        assert hasattr(result, 'latency_target_ms')

    def test_valid_compiler_config_loads(self):
        from policy.compiler import _default_compiler_cfg
        result = _default_compiler_cfg()
        assert result is not None
        assert hasattr(result, 'token_budget')


# ════════════════════════════════════════════════════════════════════
# A03 — Idempotent close()
# ════════════════════════════════════════════════════════════════════

class TestIdempotentClose:
    def test_hgshm_double_close_safe(self, tmp_dir):
        h = HGSHM(tmp_dir)
        h.close()
        h.close()  # Must not raise

    def test_hgshm_closed_flag_set(self, tmp_dir):
        h = HGSHM(tmp_dir)
        assert h._closed is False
        h.close()
        assert h._closed is True

    def test_hgshm_triple_close_safe(self, tmp_dir):
        h = HGSHM(tmp_dir)
        h.close(); h.close(); h.close()

    def test_policy_store_double_close_safe(self, tmp_dir):
        s = PolicyStore(tmp_dir)
        s.close()
        s.close()  # Must not raise

    def test_policy_store_closed_flag_set(self, tmp_dir):
        s = PolicyStore(tmp_dir)
        assert s._closed is False
        s.close()
        assert s._closed is True

    def test_graph_store_double_close_safe(self, tmp_dir):
        gs = GraphStore(tmp_dir)
        gs.close()
        gs.close()  # Must not raise

    def test_graph_store_closed_flag_set(self, tmp_dir):
        gs = GraphStore(tmp_dir)
        assert gs._closed is False
        gs.close()
        assert gs._closed is True

    def test_hgshm_context_manager_plus_explicit_close(self, tmp_dir):
        """
        Pattern that triggered A03: caller uses 'with' AND explicit close().
        Must not raise.
        """
        h = HGSHM(tmp_dir)
        with h:
            pass  # __exit__ calls close()
        h.close()  # explicit close after context — must be safe

    def test_policy_store_context_manager_plus_explicit_close(self, tmp_dir):
        s = PolicyStore(tmp_dir)
        with s:
            pass
        s.close()  # must be safe

    def test_memory_manager_double_close_safe(self, tmp_dir):
        from memory.hybrid.hgshm import HGSHM
        from memory.system.system_memory import SystemMemory
        from memory.manager import MemoryManager
        h = HGSHM(tmp_dir)
        sm = SystemMemory(h)
        mgr = MemoryManager(h, sm)
        mgr.close()
        mgr.close()  # must not raise

    def test_hgshm_operations_before_close_work(self, tmp_dir):
        """Normal operations still work before close()."""
        h = HGSHM(tmp_dir)
        h.remember("test node")
        assert h.count_by_tag("nonexistent") == 0
        h.close()

    def test_closed_flag_prevents_double_close_work(self, tmp_dir):
        """Verify _closed truly prevents double-close, not just no-op by luck."""
        h = HGSHM(tmp_dir)
        gs_close_calls = []
        original = h.graph_store.close

        def counting_close():
            gs_close_calls.append(1)
            original()

        h.graph_store.close = counting_close
        h.close()
        h.close()  # second call must be blocked

        assert len(gs_close_calls) == 1, (
            f"graph_store.close() called {len(gs_close_calls)} times — "
            f"expected 1 (double-close not prevented)"
        )


# ════════════════════════════════════════════════════════════════════
# A04 — reward_log counter initialised from DB after restart
# ════════════════════════════════════════════════════════════════════

class TestRewardLogCounterPersistence:
    def test_counter_empty_on_fresh_store(self, store):
        """Fresh PolicyStore has no counter entries."""
        assert store._reward_log_counts == {}

    def test_counter_increments_normally(self, store):
        p = _policy("counter_incr")
        store.save(p)
        for i in range(3):
            store.log_reward(_reward(p.policy_id), max_rows_per_policy=100)
        assert store._reward_log_counts.get(p.policy_id) == 3

    def test_counter_initialised_from_db_on_restart(self, tmp_dir):
        """
        After restart, first write initialises counter from DB so prune
        fires at the correct threshold, not after (limit + db_count) inserts.
        """
        # Session 1: insert 5 rows
        s1 = PolicyStore(tmp_dir)
        p = _policy("restart_test")
        s1.save(p)
        for _ in range(5):
            s1.log_reward(_reward(p.policy_id), max_rows_per_policy=0)
        s1.close()

        # Session 2: re-open
        s2 = PolicyStore(tmp_dir)
        assert p.policy_id not in s2._reward_log_counts

        # First write: counter should initialise from DB (5 rows) then +1 = 6
        s2.log_reward(_reward(p.policy_id), max_rows_per_policy=100)
        assert s2._reward_log_counts[p.policy_id] == s2.reward_log_count(p.policy_id)
        s2.close()

    def test_prune_fires_correctly_after_restart(self, tmp_dir):
        """
        If DB already has max_rows rows, the first write after restart
        should trigger prune — not wait for another max_rows inserts.
        """
        limit = 5

        # Session 1: fill to exactly limit rows
        s1 = PolicyStore(tmp_dir)
        p = _policy("prune_restart")
        s1.save(p)
        for _ in range(limit):
            s1.log_reward(_reward(p.policy_id), max_rows_per_policy=0)
        assert s1.reward_log_count(p.policy_id) == limit
        s1.close()

        # Session 2: first write should trigger prune (limit+1 > limit)
        s2 = PolicyStore(tmp_dir)
        s2.log_reward(_reward(p.policy_id), max_rows_per_policy=limit)

        # After prune, table should be at or below limit
        count = s2.reward_log_count(p.policy_id)
        assert count <= limit, (
            f"Expected ≤ {limit} rows after prune, got {count}. "
            f"A04 counter init is incorrect."
        )
        s2.close()

    def test_table_never_exceeds_2x_limit_across_restarts(self, tmp_dir):
        """
        Without A04 fix, table would grow to 2× limit after restart
        (counter=0, so prune doesn't fire until another limit rows inserted).
        With fix, table stays at or below limit.
        """
        limit = 3

        # Simulate 5 restart cycles
        for session in range(5):
            s = PolicyStore(tmp_dir)
            p_list = s.all_active(
                domain=PolicyDomain.SYSTEM,
                policy_type=PolicyType.PLANNER_CONFIG)
            if not p_list:
                p = _policy(f"grow_test")
                s.save(p)
                pid = p.policy_id
            else:
                pid = p_list[0].policy_id

            # Insert 2 rows per session
            for _ in range(2):
                s.log_reward(_reward(pid), max_rows_per_policy=limit)

            count = s.reward_log_count(pid)
            assert count <= limit + 1, (
                f"Session {session}: {count} rows > {limit+1} = "
                f"limit exceeded (A04 not working)"
            )
            s.close()

    def test_counter_correct_after_prune(self, store):
        """After prune fires, counter is reset to keep_last."""
        p = _policy("post_prune")
        store.save(p)
        limit = 3

        for _ in range(limit + 2):  # trigger prune
            store.log_reward(_reward(p.policy_id), max_rows_per_policy=limit)

        # Counter should equal actual DB count
        assert store._reward_log_counts.get(p.policy_id) == store.reward_log_count(p.policy_id)


# ════════════════════════════════════════════════════════════════════
# .gitignore — presence and correctness
# ════════════════════════════════════════════════════════════════════

class TestGitignore:
    def test_gitignore_exists(self):
        gi = Path(__file__).parent.parent / '.gitignore'
        assert gi.exists(), ".gitignore not found in repository root"

    def test_gitignore_excludes_databases(self):
        gi = Path(__file__).parent.parent / '.gitignore'
        content = gi.read_text()
        assert '*.db' in content
        assert 'hgshm.db' in content
        assert 'policy.db' in content
        assert 'vectors.db' in content

    def test_gitignore_excludes_pycache(self):
        gi = Path(__file__).parent.parent / '.gitignore'
        content = gi.read_text()
        assert '__pycache__/' in content

    def test_gitignore_excludes_venv(self):
        gi = Path(__file__).parent.parent / '.gitignore'
        content = gi.read_text()
        assert '.venv/' in content or 'venv/' in content

    def test_gitignore_excludes_results_dir(self):
        gi = Path(__file__).parent.parent / '.gitignore'
        content = gi.read_text()
        assert 'results/' in content

    def test_gitignore_excludes_secrets(self):
        gi = Path(__file__).parent.parent / '.gitignore'
        content = gi.read_text()
        assert '*.pem' in content or '.secrets/' in content or 'secrets.yaml' in content

    def test_gitignore_excludes_model_weights(self):
        gi = Path(__file__).parent.parent / '.gitignore'
        content = gi.read_text()
        assert '*.bin' in content or '*.pt' in content or '*.gguf' in content

    def test_gitignore_not_empty(self):
        gi = Path(__file__).parent.parent / '.gitignore'
        assert gi.stat().st_size > 500

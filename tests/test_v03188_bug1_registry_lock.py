"""
tests/test_v03188_bug1_registry_lock.py
=========================================
Regression tests for the _HGSHM_REGISTRY race condition fix.

Bug: _get_hgshm() had no lock around its check-then-create sequence.
Two threads simultaneously checking "key not in registry" would both
pass, both create a HGSHM instance, and the first would be orphaned
(leaked SQLite connection; any writes to it silently lost).

Fix: Added _REGISTRY_LOCK (threading.Lock) with double-checked locking:
  - Fast path: atomic dict read without the lock (CPython GIL safe)
  - Slow path: re-check inside the lock before creating the instance
  - close_registry() also holds the lock during teardown
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRegistryLockExists:

    def test_registry_lock_present(self):
        import memory.hybrid.shims as shims
        assert hasattr(shims, '_REGISTRY_LOCK'), (
            "_REGISTRY_LOCK must exist on the shims module"
        )

    def test_registry_lock_is_threading_lock(self):
        import threading
        import memory.hybrid.shims as shims
        assert isinstance(shims._REGISTRY_LOCK, type(threading.Lock())), (
            "_REGISTRY_LOCK must be a threading.Lock"
        )

    def test_lock_in_get_hgshm_source(self):
        import inspect
        from memory.hybrid.shims import _get_hgshm
        src = inspect.getsource(_get_hgshm)
        assert '_REGISTRY_LOCK' in src, (
            "_get_hgshm must acquire _REGISTRY_LOCK"
        )

    def test_double_checked_locking_pattern(self):
        """Both a fast-path check and an in-lock re-check must be present."""
        import inspect
        from memory.hybrid.shims import _get_hgshm
        src = inspect.getsource(_get_hgshm)
        # Two 'if key in _HGSHM_REGISTRY' checks: one before lock, one inside
        assert src.count('if key in _HGSHM_REGISTRY') >= 2, (
            "Double-checked locking requires two 'key in registry' checks"
        )

    def test_close_registry_lock_in_source(self):
        import inspect
        from memory.hybrid.shims import close_registry
        src = inspect.getsource(close_registry)
        assert '_REGISTRY_LOCK' in src, (
            "close_registry must hold _REGISTRY_LOCK during teardown"
        )


class TestRegistryConcurrency:

    def setup_method(self):
        from memory.hybrid.shims import close_registry
        close_registry()

    def teardown_method(self):
        from memory.hybrid.shims import close_registry
        close_registry()

    def test_20_threads_same_key_one_instance(self, tmp_path):
        """20 concurrent calls for the same key must produce exactly 1 instance."""
        from memory.hybrid.shims import _get_hgshm
        instances = []
        errors = []

        def get():
            try:
                instances.append(_get_hgshm(tmp_path, user_id="alice"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=get) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, f"Thread errors: {errors}"
        unique = set(id(i) for i in instances)
        assert len(unique) == 1, (
            f"Race condition: {len(unique)} distinct HGSHM instances created "
            f"(expected 1). Orphaned connections = leaked SQLite handles."
        )

    def test_50_threads_same_key_one_instance(self, tmp_path):
        """Stress test: 50 threads."""
        from memory.hybrid.shims import _get_hgshm
        instances = []
        errors = []

        def get():
            try:
                instances.append(_get_hgshm(tmp_path, user_id="stress"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=get) for _ in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors
        assert len(set(id(i) for i in instances)) == 1, (
            "50-thread stress: still producing multiple instances"
        )

    def test_same_key_repeated_calls_same_instance(self, tmp_path):
        """Sequential same-key calls must always return the cached instance."""
        from memory.hybrid.shims import _get_hgshm
        h1 = _get_hgshm(tmp_path, user_id="bob")
        h2 = _get_hgshm(tmp_path, user_id="bob")
        h3 = _get_hgshm(tmp_path, user_id="bob")
        assert h1 is h2 is h3

    def test_different_keys_different_instances(self, tmp_path):
        """Different user_ids for the same path must yield different instances."""
        from memory.hybrid.shims import _get_hgshm
        h1 = _get_hgshm(tmp_path, user_id="alice")
        h2 = _get_hgshm(tmp_path, user_id="bob")
        assert h1 is not h2

    def test_concurrent_different_keys_no_interference(self, tmp_path):
        """Concurrent creation of different keys must not interfere."""
        from memory.hybrid.shims import _get_hgshm, _HGSHM_REGISTRY
        results = {}
        errors = []

        def get(uid):
            try:
                results[uid] = _get_hgshm(tmp_path / uid, user_id=uid)
            except Exception as exc:
                errors.append(exc)

        users = [f"user_{i}" for i in range(10)]
        threads = [threading.Thread(target=get, args=(u,)) for u in users]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors
        assert len(results) == 10
        # All instances must be distinct
        unique = set(id(v) for v in results.values())
        assert len(unique) == 10

    def test_registry_not_grown_beyond_one_per_key(self, tmp_path):
        """After 50 concurrent calls for same key, registry has exactly 1 entry."""
        from memory.hybrid.shims import _get_hgshm, _HGSHM_REGISTRY
        errors = []

        def get():
            try:
                _get_hgshm(tmp_path, user_id="count_test")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=get) for _ in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors
        matching_keys = [k for k in _HGSHM_REGISTRY if "count_test" in k]
        assert len(matching_keys) == 1, (
            f"Expected 1 registry entry for the key, found {len(matching_keys)}"
        )

    def test_close_registry_clears_all(self, tmp_path):
        from memory.hybrid.shims import _get_hgshm, close_registry, _HGSHM_REGISTRY
        _get_hgshm(tmp_path / "a", user_id="x")
        _get_hgshm(tmp_path / "b", user_id="y")
        assert len(_HGSHM_REGISTRY) >= 2
        closed = close_registry()
        assert closed >= 2
        assert len(_HGSHM_REGISTRY) == 0

    def test_no_user_id_path_unaffected(self, tmp_path):
        """Legacy no-user_id path still works correctly after lock addition."""
        from memory.hybrid.shims import _get_hgshm
        h1 = _get_hgshm(tmp_path)
        h2 = _get_hgshm(tmp_path)
        assert h1 is h2

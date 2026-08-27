"""
tests/test_v03181_a24_a25_a26.py
===================================
Regression tests for:

  A24 — EmbeddingManager.close() added; HGSHM.close() calls it
  A25 — memory/ package hierarchy populated with __all__ and re-exports
  A26 — policy_versions INSERT OR IGNORE → INSERT (errors are no longer silent)
"""

from __future__ import annotations

import sqlite3
import tempfile
import uuid
from pathlib import Path

import pytest


# ═════════════════════════════════════════════════════════════════════════════
# A24 — EmbeddingManager.close() / context manager
# ═════════════════════════════════════════════════════════════════════════════

class TestEmbeddingManagerClose:

    def test_close_method_exists(self):
        from memory.hybrid.vector.embedding_manager import EmbeddingManager
        em = EmbeddingManager()
        assert hasattr(em, 'close') and callable(em.close)

    def test_close_clears_cache(self):
        from memory.hybrid.vector.embedding_manager import EmbeddingManager
        em = EmbeddingManager()
        em.embed("hello world")
        assert em.cache_size >= 1
        em.close()
        assert em.cache_size == 0

    def test_close_idempotent(self):
        """Calling close() multiple times must not raise."""
        from memory.hybrid.vector.embedding_manager import EmbeddingManager
        em = EmbeddingManager()
        em.embed("test text")
        em.close()
        em.close()  # must not raise
        assert em.cache_size == 0

    def test_close_on_empty_cache(self):
        """close() on a fresh (empty) instance must not raise."""
        from memory.hybrid.vector.embedding_manager import EmbeddingManager
        em = EmbeddingManager()
        assert em.cache_size == 0
        em.close()  # must not raise
        assert em.cache_size == 0

    def test_embed_still_works_after_close(self):
        """embed() after close() must still produce valid vectors (cache just refills)."""
        from memory.hybrid.vector.embedding_manager import EmbeddingManager
        em = EmbeddingManager()
        v1 = em.embed("hello")
        em.close()
        v2 = em.embed("hello")
        assert v1 == v2, "embed must return same vector after cache clear"
        assert em.cache_size == 1

    def test_context_manager_protocol_exists(self):
        from memory.hybrid.vector.embedding_manager import EmbeddingManager
        em = EmbeddingManager()
        assert hasattr(em, '__enter__') and hasattr(em, '__exit__')

    def test_context_manager_clears_cache_on_exit(self):
        from memory.hybrid.vector.embedding_manager import EmbeddingManager
        with EmbeddingManager() as em:
            em.embed("context manager test")
            assert em.cache_size >= 1
        assert em.cache_size == 0

    def test_context_manager_returns_self(self):
        from memory.hybrid.vector.embedding_manager import EmbeddingManager
        em = EmbeddingManager()
        with em as em2:
            assert em2 is em


class TestHGSHMCloseCallsEmbeddingManagerClose:
    """HGSHM.close() must clear the embedding cache via embedding_manager.close()."""

    def test_hgshm_close_clears_embedding_cache(self, tmp_path):
        from memory.hybrid.hgshm import HGSHM
        h = HGSHM(tmp_path)
        h.remember("some important fact")
        assert h.embedding_manager.cache_size >= 1
        h.close()
        assert h.embedding_manager.cache_size == 0, (
            "HGSHM.close() must clear embedding_manager cache"
        )

    def test_hgshm_context_manager_clears_embedding_cache(self, tmp_path):
        from memory.hybrid.hgshm import HGSHM
        with HGSHM(tmp_path) as h:
            h.remember("fact one")
            h.remember("fact two")
            size_before = h.embedding_manager.cache_size
        assert h.embedding_manager.cache_size == 0, (
            "Embedding cache must be cleared when HGSHM context manager exits"
        )

    def test_hgshm_double_close_safe(self, tmp_path):
        from memory.hybrid.hgshm import HGSHM
        h = HGSHM(tmp_path)
        h.remember("test")
        h.close()
        h.close()  # must not raise
        assert h.embedding_manager.cache_size == 0

    def test_hgshm_stats_include_embedding_cache(self, tmp_path):
        """stats() must report embedding_cache size."""
        from memory.hybrid.hgshm import HGSHM
        h = HGSHM(tmp_path)
        h.remember("cached node")
        stats = h.stats()
        assert 'embedding_cache' in stats, "stats() must include embedding_cache"
        h.close()


# ═════════════════════════════════════════════════════════════════════════════
# A25 — memory/ package __all__ declarations
# ═════════════════════════════════════════════════════════════════════════════

class TestMemoryPackageAll:
    """All four memory __init__.py files must declare __all__."""

    def test_memory_package_has_all(self):
        import memory
        assert hasattr(memory, '__all__'), "memory/__init__.py must define __all__"
        assert isinstance(memory.__all__, list)
        assert len(memory.__all__) > 0

    def test_memory_hybrid_has_all(self):
        import memory.hybrid as mh
        assert hasattr(mh, '__all__')
        assert len(mh.__all__) > 0

    def test_memory_system_has_all(self):
        import memory.system as ms
        assert hasattr(ms, '__all__')
        assert len(ms.__all__) > 0

    def test_memory_user_has_all(self):
        import memory.user as mu
        assert hasattr(mu, '__all__')
        assert len(mu.__all__) > 0


class TestMemoryPackageReExports:
    """Canonical symbols must be importable from the package level."""

    def test_hgshm_from_memory(self):
        from memory import HGSHM
        from memory.hybrid.hgshm import HGSHM as _HGSHM
        assert HGSHM is _HGSHM

    def test_memory_manager_from_memory(self):
        from memory import MemoryManager
        from memory.manager import MemoryManager as _MM
        assert MemoryManager is _MM

    def test_routed_context_from_memory(self):
        from memory import RoutedContext
        from memory.manager import RoutedContext as _RC
        assert RoutedContext is _RC

    def test_system_memory_from_memory(self):
        from memory import SystemMemory
        from memory.system.system_memory import SystemMemory as _SM
        assert SystemMemory is _SM

    def test_user_memory_from_memory(self):
        from memory import UserMemory
        from memory.user.user_memory import UserMemory as _UM
        assert UserMemory is _UM

    def test_validate_user_id_from_memory(self):
        from memory import validate_user_id
        from memory.user.user_memory import validate_user_id as _vui
        assert validate_user_id is _vui

    def test_memory_node_from_memory(self):
        from memory import MemoryNode
        from memory.hybrid.models.memory_node import MemoryNode as _MN
        assert MemoryNode is _MN

    def test_memory_type_from_memory(self):
        from memory import MemoryType
        from memory.hybrid.models.memory_node import MemoryType as _MT
        assert MemoryType is _MT

    def test_hgshm_from_memory_hybrid(self):
        from memory.hybrid import HGSHM
        from memory.hybrid.hgshm import HGSHM as _HGSHM
        assert HGSHM is _HGSHM

    def test_memory_node_from_memory_hybrid(self):
        from memory.hybrid import MemoryNode
        assert MemoryNode is not None

    def test_memory_type_from_memory_hybrid(self):
        from memory.hybrid import MemoryType
        assert MemoryType is not None

    def test_hierarchy_level_from_memory_hybrid(self):
        from memory.hybrid import HierarchyLevel
        assert HierarchyLevel is not None

    def test_epistemic_status_from_memory_hybrid(self):
        from memory.hybrid import EpistemicStatus
        assert EpistemicStatus is not None

    def test_system_memory_from_memory_system(self):
        from memory.system import SystemMemory
        from memory.system.system_memory import SystemMemory as _SM
        assert SystemMemory is _SM

    def test_user_memory_from_memory_user(self):
        from memory.user import UserMemory
        from memory.user.user_memory import UserMemory as _UM
        assert UserMemory is _UM

    def test_validate_user_id_from_memory_user(self):
        from memory.user import validate_user_id
        assert callable(validate_user_id)


class TestMemoryAllContents:
    """__all__ contents must match what is actually exported."""

    def test_memory_all_symbols_importable(self):
        import memory
        for name in memory.__all__:
            assert hasattr(memory, name), (
                f"memory.__all__ lists '{name}' but it is not importable from memory"
            )

    def test_memory_hybrid_all_symbols_importable(self):
        import memory.hybrid as mh
        for name in mh.__all__:
            assert hasattr(mh, name), (
                f"memory.hybrid.__all__ lists '{name}' but it is not importable"
            )

    def test_memory_system_all_symbols_importable(self):
        import memory.system as ms
        for name in ms.__all__:
            assert hasattr(ms, name)

    def test_memory_user_all_symbols_importable(self):
        import memory.user as mu
        for name in mu.__all__:
            assert hasattr(mu, name)


# ═════════════════════════════════════════════════════════════════════════════
# A26 — policy_versions INSERT OR IGNORE → INSERT
# ═════════════════════════════════════════════════════════════════════════════

class TestPolicyVersionsInsert:
    """save_version must use INSERT (not INSERT OR IGNORE) so errors surface."""

    def test_no_insert_or_ignore_in_save_version(self):
        import inspect
        from policy.store import PolicyStore
        src = inspect.getsource(PolicyStore.save_version)
        assert "INSERT OR IGNORE" not in src, (
            "save_version() must not use INSERT OR IGNORE (A26)"
        )
        assert "INSERT INTO policy_versions" in src

    def test_duplicate_version_id_raises(self, tmp_path):
        """Saving the same PolicyVersion twice must raise, not silently drop."""
        from policy.store import PolicyStore
        from policy.models import PolicyRecord, PolicyVersion, PolicyType, PolicyDomain

        store = PolicyStore(memory_dir=tmp_path)
        p = PolicyRecord(
            policy_id=str(uuid.uuid4()),
            name="dup-test",
            policy_type=PolicyType.PLANNER_CONFIG,
            domain=PolicyDomain.SYSTEM,
            config={},
        )
        store.save(p)

        pv = PolicyVersion(
            policy_id=p.policy_id, version=1, config=p.config,
            alpha=p.alpha, beta_=p.beta_, mean_reward=p.confidence,
            reason="first save",
        )
        store.save_version(pv)

        with pytest.raises((sqlite3.IntegrityError, sqlite3.Error, RuntimeError)):
            store.save_version(pv)  # same version_id → must raise

        store.close()

    def test_unique_version_ids_both_saved(self, tmp_path):
        """Two distinct version snapshots must both be saved."""
        from policy.store import PolicyStore
        from policy.models import PolicyRecord, PolicyVersion, PolicyType, PolicyDomain

        store = PolicyStore(memory_dir=tmp_path)
        p = PolicyRecord(
            policy_id=str(uuid.uuid4()),
            name="multi-snap",
            policy_type=PolicyType.PLANNER_CONFIG,
            domain=PolicyDomain.SYSTEM,
            config={},
        )
        store.save(p)

        pv1 = PolicyVersion(policy_id=p.policy_id, version=1, config=p.config,
                            alpha=1.0, beta_=1.0, mean_reward=0.5, reason="v1")
        pv2 = PolicyVersion(policy_id=p.policy_id, version=2, config=p.config,
                            alpha=2.0, beta_=1.0, mean_reward=0.6, reason="v2")

        store.save_version(pv1)
        store.save_version(pv2)

        history = store.get_history(p.policy_id)
        assert len(history) == 2, f"Expected 2 snapshots, got {len(history)}"
        assert {h.version for h in history} == {1, 2}
        store.close()

    def test_get_history_no_select_star(self):
        """get_history() must not use SELECT *."""
        import inspect
        from policy.store import PolicyStore
        src = inspect.getsource(PolicyStore.get_history)
        assert "SELECT *" not in src, (
            "get_history() must use explicit column list (A21/A26)"
        )

    def test_version_data_preserved_after_insert(self, tmp_path):
        """All PolicyVersion fields must survive save_version → get_history."""
        from policy.store import PolicyStore
        from policy.models import PolicyRecord, PolicyVersion, PolicyType, PolicyDomain

        store = PolicyStore(memory_dir=tmp_path)
        p = PolicyRecord(
            policy_id=str(uuid.uuid4()),
            name="field-check",
            policy_type=PolicyType.PLANNER_CONFIG,
            domain=PolicyDomain.SYSTEM,
            config={"k": "v"},
            alpha=3.5, beta_=2.1,
        )
        store.save(p)

        pv = PolicyVersion(
            policy_id=p.policy_id, version=1,
            config=p.config, alpha=3.5, beta_=2.1,
            mean_reward=0.72, reason="field-check",
        )
        store.save_version(pv)
        history = store.get_history(p.policy_id)
        h = history[0]

        assert h.alpha == 3.5
        assert h.beta_ == 2.1
        assert abs(h.mean_reward - 0.72) < 1e-9
        assert h.reason == "field-check"
        assert h.config == {"k": "v"}
        store.close()

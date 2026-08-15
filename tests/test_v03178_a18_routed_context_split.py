"""
tests/test_v03178_a18_routed_context_split.py
===============================================
Regression tests for A18:
  RoutedContext.to_memory_context() previously hard-coded a 10/10 split,
  silently discarding memories beyond index 20 when top_k > 20.

Fix:
  - RoutedContext gains a ``primary_split`` field (default=10, backward-compat).
  - MemoryManager.query() sets ``primary_split=top_k`` on the returned context.
  - to_memory_context() uses primary_split as the boundary; supporting_memories
    receives ALL remaining memories (no upper cap).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memory.manager import RoutedContext


# ─── helpers ─────────────────────────────────────────────────────────────────

def _fake_memories(n: int):
    """Return n lightweight RetrievedMemory stubs ranked by descending score."""
    from memory.hybrid.models.memory_context import RetrievedMemory
    mems = []
    for i in range(n):
        rm = MagicMock(spec=RetrievedMemory)
        rm.final_score = 1.0 - i * 0.01
        rm.node = MagicMock()
        rm.node.node_id = f"node-{i:03d}"
        mems.append(rm)
    return mems


# ═════════════════════════════════════════════════════════════════════════════
# RoutedContext — primary_split field
# ═════════════════════════════════════════════════════════════════════════════

class TestRoutedContextPrimarySplit:
    """RoutedContext must expose primary_split and use it in to_memory_context()."""

    def test_primary_split_field_exists(self):
        rc = RoutedContext()
        assert hasattr(rc, 'primary_split'), (
            "RoutedContext must have a primary_split field (A18)"
        )

    def test_primary_split_default_is_ten(self):
        """Default backward-compatible value must be 10."""
        rc = RoutedContext()
        assert rc.primary_split == 10

    def test_primary_split_configurable(self):
        rc = RoutedContext(primary_split=25)
        assert rc.primary_split == 25

    def test_to_memory_context_uses_primary_split(self):
        """primary_memories must contain exactly primary_split items (or fewer if short)."""
        mems = _fake_memories(30)
        rc = RoutedContext(query="q", merged_memories=mems, primary_split=15)
        ctx = rc.to_memory_context()
        assert len(ctx.primary_memories) == 15

    def test_supporting_gets_all_remaining(self):
        """supporting_memories must contain everything after primary_split, no upper cap."""
        mems = _fake_memories(30)
        rc = RoutedContext(query="q", merged_memories=mems, primary_split=15)
        ctx = rc.to_memory_context()
        assert len(ctx.supporting_memories) == 15

    def test_no_memories_discarded(self):
        """primary + supporting must account for all merged_memories."""
        mems = _fake_memories(50)
        rc = RoutedContext(query="q", merged_memories=mems, primary_split=20)
        ctx = rc.to_memory_context()
        total = len(ctx.primary_memories) + len(ctx.supporting_memories)
        assert total == 50, (
            f"All 50 memories must be preserved; got primary={len(ctx.primary_memories)} "
            f"supporting={len(ctx.supporting_memories)}"
        )

    def test_hardcoded_cap_of_20_is_gone(self):
        """Memories beyond index 20 must NOT be silently dropped (A18 regression)."""
        mems = _fake_memories(25)
        rc = RoutedContext(query="q", merged_memories=mems, primary_split=10)
        ctx = rc.to_memory_context()
        # Old code: supporting_memories = merged_memories[10:20] → 10 items, 5 dropped
        # New code: supporting_memories = merged_memories[10:]   → 15 items, none dropped
        assert len(ctx.supporting_memories) == 15, (
            f"Memories beyond index 20 must not be discarded; "
            f"expected 15 supporting, got {len(ctx.supporting_memories)}"
        )

    def test_split_with_fewer_memories_than_split(self):
        """When len(merged) < primary_split, all go to primary, supporting is empty."""
        mems = _fake_memories(5)
        rc = RoutedContext(query="q", merged_memories=mems, primary_split=10)
        ctx = rc.to_memory_context()
        assert len(ctx.primary_memories) == 5
        assert len(ctx.supporting_memories) == 0

    def test_split_with_exact_boundary(self):
        """When len(merged) == primary_split, all go to primary, supporting is empty."""
        mems = _fake_memories(10)
        rc = RoutedContext(query="q", merged_memories=mems, primary_split=10)
        ctx = rc.to_memory_context()
        assert len(ctx.primary_memories) == 10
        assert len(ctx.supporting_memories) == 0

    def test_split_zero_primary(self):
        """primary_split=0 → all memories go to supporting."""
        mems = _fake_memories(8)
        rc = RoutedContext(query="q", merged_memories=mems, primary_split=0)
        ctx = rc.to_memory_context()
        assert len(ctx.primary_memories) == 0
        assert len(ctx.supporting_memories) == 8

    def test_empty_merged_memories(self):
        """Empty merged_memories must not crash to_memory_context()."""
        rc = RoutedContext(query="q", primary_split=10)
        ctx = rc.to_memory_context()
        assert ctx.primary_memories == []
        assert ctx.supporting_memories == []

    def test_order_preserved_in_primary(self):
        """Primary memories must be the first primary_split items by rank order."""
        mems = _fake_memories(20)
        rc = RoutedContext(query="q", merged_memories=mems, primary_split=5)
        ctx = rc.to_memory_context()
        expected_ids = [f"node-{i:03d}" for i in range(5)]
        actual_ids = [rm.node.node_id for rm in ctx.primary_memories]
        assert actual_ids == expected_ids

    def test_order_preserved_in_supporting(self):
        """Supporting memories must follow primary in rank order."""
        mems = _fake_memories(20)
        rc = RoutedContext(query="q", merged_memories=mems, primary_split=5)
        ctx = rc.to_memory_context()
        expected_ids = [f"node-{i:03d}" for i in range(5, 20)]
        actual_ids = [rm.node.node_id for rm in ctx.supporting_memories]
        assert actual_ids == expected_ids

    def test_other_fields_unaffected(self):
        """to_memory_context() must still copy principle_nodes, belief_nodes etc."""
        from memory.hybrid.models.memory_context import MemoryContext
        system_ctx = MagicMock(spec=MemoryContext)
        system_ctx.principle_nodes = ["p1", "p2"]
        user_ctx = MagicMock(spec=MemoryContext)
        user_ctx.belief_nodes = ["b1"]
        user_ctx.knowledge_gaps = ["gap1"]

        rc = RoutedContext(
            query="q",
            merged_memories=_fake_memories(5),
            primary_split=3,
            system_context=system_ctx,
            user_context=user_ctx,
            routing_latency_ms=42.0,
        )
        ctx = rc.to_memory_context()
        assert ctx.principle_nodes == ["p1", "p2"]
        assert ctx.belief_nodes == ["b1"]
        assert ctx.knowledge_gaps == ["gap1"]
        assert ctx.retrieval_latency_ms == 42.0


# ═════════════════════════════════════════════════════════════════════════════
# MemoryManager.query() — sets primary_split from top_k
# ═════════════════════════════════════════════════════════════════════════════

class TestMemoryManagerQuerySetsTopK:
    """MemoryManager.query() must set primary_split=top_k on the returned RoutedContext."""

    def _make_manager(self, tmp_path):
        from memory.hybrid.hgshm import HGSHM
        from memory.system.system_memory import SystemMemory
        from memory.manager import MemoryManager
        hgshm = HGSHM(tmp_path)
        system = SystemMemory(hgshm)
        return MemoryManager(hgshm=hgshm, system_memory=system), hgshm

    def test_default_top_k_sets_primary_split_10(self, tmp_path):
        mgr, hgshm = self._make_manager(tmp_path)
        rc = mgr.query("test query")
        assert rc.primary_split == 10, (
            f"Default top_k=10 must set primary_split=10, got {rc.primary_split}"
        )
        hgshm.close()

    def test_custom_top_k_sets_primary_split(self, tmp_path):
        mgr, hgshm = self._make_manager(tmp_path)
        rc = mgr.query("test query", top_k=25)
        assert rc.primary_split == 25, (
            f"top_k=25 must set primary_split=25, got {rc.primary_split}"
        )
        hgshm.close()

    def test_top_k_5_sets_primary_split_5(self, tmp_path):
        mgr, hgshm = self._make_manager(tmp_path)
        rc = mgr.query("test query", top_k=5)
        assert rc.primary_split == 5
        hgshm.close()

    def test_primary_split_reflected_in_to_memory_context(self, tmp_path):
        """to_memory_context() on a query() result must honour primary_split."""
        mgr, hgshm = self._make_manager(tmp_path)

        # Add enough memories to cross the split boundary
        for i in range(15):
            hgshm.remember(f"fact number {i}", confidence=0.8, importance=0.7)

        rc = mgr.query("fact", top_k=5)
        ctx = rc.to_memory_context()
        # primary_split=5 → primary_memories has at most 5 entries
        assert len(ctx.primary_memories) <= 5, (
            f"primary_memories must respect primary_split=5; "
            f"got {len(ctx.primary_memories)}"
        )
        hgshm.close()

    def test_no_memories_discarded_in_to_memory_context(self, tmp_path):
        """primary + supporting must account for all merged_memories."""
        mgr, hgshm = self._make_manager(tmp_path)

        for i in range(20):
            hgshm.remember(f"memory {i}", confidence=0.8, importance=0.7)

        rc = mgr.query("memory", top_k=15)
        ctx = rc.to_memory_context()
        total = len(ctx.primary_memories) + len(ctx.supporting_memories)
        assert total == len(rc.merged_memories), (
            f"No memories may be discarded: {len(rc.merged_memories)} merged, "
            f"but primary+supporting={total}"
        )
        hgshm.close()


# ═════════════════════════════════════════════════════════════════════════════
# Backward compatibility
# ═════════════════════════════════════════════════════════════════════════════

class TestA18BackwardCompat:
    """Existing callers with top_k=10 (default) must be unaffected."""

    def test_default_split_same_as_before(self):
        """With 20 memories and default primary_split=10, behaviour is identical."""
        mems = _fake_memories(20)
        rc = RoutedContext(query="q", merged_memories=mems)  # default primary_split=10
        ctx = rc.to_memory_context()
        assert len(ctx.primary_memories) == 10
        assert len(ctx.supporting_memories) == 10

    def test_routed_context_construction_unchanged(self):
        """RoutedContext() with no primary_split must still work."""
        rc = RoutedContext(query="hello", merged_memories=[])
        assert rc.query == "hello"
        assert rc.primary_split == 10  # default

    def test_to_memory_context_returns_memory_context(self):
        from memory.hybrid.models.memory_context import MemoryContext
        rc = RoutedContext(query="q", merged_memories=_fake_memories(5))
        ctx = rc.to_memory_context()
        assert isinstance(ctx, MemoryContext)

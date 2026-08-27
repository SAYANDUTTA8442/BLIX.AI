"""
tests/test_v03189_bug2_topk_floor.py
======================================
Regression tests for the MemoryManager.query() top_k // 2 floor bug.

Bug: system and user memory sub-queries used `top_k // 2` directly.
  top_k=1 → sub_k=0 → both recall() calls asked for 0 results → always
  returned empty lists, silently dropping all system/user memory results
  even though the caller explicitly requested them.

Fix: sub_k = max(1, top_k // 2) guarantees at least one result per
sub-domain for any top_k >= 1.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestSubKCalculation:
    """White-box: verify the max(1, ...) guard is present and correct."""

    def test_sub_k_formula_in_source(self):
        import memory.manager as mm
        src = inspect.getsource(mm.MemoryManager.query)
        assert 'max(1, top_k // 2)' in src, (
            "query() must use max(1, top_k // 2), not bare top_k // 2"
        )

    def test_sub_k_used_for_system_recall(self):
        import memory.manager as mm
        src = inspect.getsource(mm.MemoryManager.query)
        # sub_k must be the variable passed, not the raw expression
        assert 'recall(query, top_k=sub_k)' in src or 'top_k=sub_k' in src

    def test_sub_k_never_zero_for_positive_top_k(self):
        for top_k in range(1, 20):
            sub_k = max(1, top_k // 2)
            assert sub_k >= 1, f"sub_k={sub_k} for top_k={top_k}"

    def test_sub_k_values(self):
        cases = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 10: 5, 11: 5, 20: 10}
        for top_k, expected in cases.items():
            assert max(1, top_k // 2) == expected, f"top_k={top_k}"


class TestQuerySubKPassthrough:
    """Black-box: system and user recall() must receive sub_k, not 0."""

    def _make_manager(self):
        from memory.manager import MemoryManager
        hgshm    = MagicMock()
        system   = MagicMock()
        manager  = MemoryManager.__new__(MemoryManager)
        manager._hgshm   = hgshm
        manager._system  = system
        manager._user_memories = {}
        manager._lock    = __import__('threading').Lock()
        # recall returns an empty MemoryContext-like object
        from memory.hybrid.models.memory_context import MemoryContext
        system.recall.return_value  = MemoryContext()
        hgshm.recall.return_value   = MemoryContext()
        return manager, system, hgshm

    def _make_user_mem(self):
        from memory.hybrid.models.memory_context import MemoryContext
        um = MagicMock()
        um.recall.return_value = MemoryContext()
        return um

    def test_top_k_1_system_gets_sub_k_1(self):
        manager, system, hgshm = self._make_manager()
        manager.get_user_memory = MagicMock(return_value=self._make_user_mem())
        manager.query("q", top_k=1)
        args, kwargs = system.recall.call_args
        passed_top_k = kwargs.get('top_k', args[1] if len(args) > 1 else None)
        assert passed_top_k == 1, (
            f"top_k=1 → system recall must get sub_k=1, got {passed_top_k}"
        )

    def test_top_k_1_user_gets_sub_k_1(self):
        manager, system, hgshm = self._make_manager()
        user_mem = self._make_user_mem()
        manager.get_user_memory = MagicMock(return_value=user_mem)
        manager.query("q", top_k=1)
        args, kwargs = user_mem.recall.call_args
        passed_top_k = kwargs.get('top_k', args[1] if len(args) > 1 else None)
        assert passed_top_k == 1, (
            f"top_k=1 → user recall must get sub_k=1, got {passed_top_k}"
        )

    def test_top_k_2_system_gets_sub_k_1(self):
        manager, system, hgshm = self._make_manager()
        manager.get_user_memory = MagicMock(return_value=self._make_user_mem())
        manager.query("q", top_k=2)
        _, kwargs = system.recall.call_args
        assert kwargs.get('top_k') == 1

    def test_top_k_10_system_gets_sub_k_5(self):
        manager, system, hgshm = self._make_manager()
        manager.get_user_memory = MagicMock(return_value=self._make_user_mem())
        manager.query("q", top_k=10)
        _, kwargs = system.recall.call_args
        assert kwargs.get('top_k') == 5

    def test_top_k_1_system_recall_called_once(self):
        """system.recall must be called (not skipped) even with top_k=1."""
        manager, system, hgshm = self._make_manager()
        manager.get_user_memory = MagicMock(return_value=self._make_user_mem())
        manager.query("q", top_k=1)
        system.recall.assert_called_once()

    def test_top_k_1_user_recall_called_once(self):
        manager, system, hgshm = self._make_manager()
        user_mem = self._make_user_mem()
        manager.get_user_memory = MagicMock(return_value=user_mem)
        manager.query("q", top_k=1)
        user_mem.recall.assert_called_once()

    def test_general_always_gets_full_top_k(self):
        """General HGSHM always receives the full top_k, not sub_k."""
        manager, system, hgshm = self._make_manager()
        manager.get_user_memory = MagicMock(return_value=self._make_user_mem())
        manager.query("q", top_k=3)
        _, kwargs = hgshm.recall.call_args
        assert kwargs.get('top_k') == 3, (
            "General HGSHM must receive full top_k, not sub_k"
        )

    def test_system_never_receives_zero(self):
        manager, system, hgshm = self._make_manager()
        manager.get_user_memory = MagicMock(return_value=self._make_user_mem())
        for top_k in [1, 2, 3]:
            system.recall.reset_mock()
            manager.query("q", top_k=top_k)
            _, kwargs = system.recall.call_args
            assert kwargs.get('top_k', -1) >= 1, (
                f"system recall received top_k=0 for top_k={top_k}"
            )

    def test_user_never_receives_zero(self):
        manager, system, hgshm = self._make_manager()
        for top_k in [1, 2, 3]:
            user_mem = self._make_user_mem()
            manager.get_user_memory = MagicMock(return_value=user_mem)
            manager.query("q", top_k=top_k)
            _, kwargs = user_mem.recall.call_args
            assert kwargs.get('top_k', -1) >= 1, (
                f"user recall received top_k=0 for top_k={top_k}"
            )

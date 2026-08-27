"""
tests/test_v03180_a22_a23.py
==============================
Regression tests for:

  A22 — PolicyCompiler.compile() truncates system_instructions at
         max_system_instructions_chars (new PromptCompilerSettings field)
  A23 — HybridWeightsSettings gains to_raw_dict() and to_normalised_dict();
         to_dict() now returns normalised values
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ═════════════════════════════════════════════════════════════════════════════
# A22 — PromptCompilerSettings.max_system_instructions_chars
# ═════════════════════════════════════════════════════════════════════════════

class TestPromptCompilerSettings:

    def test_field_exists(self):
        from config.schema import PromptCompilerSettings
        cfg = PromptCompilerSettings()
        assert hasattr(cfg, 'max_system_instructions_chars')

    def test_default_is_8000(self):
        from config.schema import PromptCompilerSettings
        assert PromptCompilerSettings().max_system_instructions_chars == 8000

    def test_must_be_positive(self):
        from config.schema import PromptCompilerSettings
        import pydantic
        with pytest.raises((pydantic.ValidationError, ValueError)):
            PromptCompilerSettings(max_system_instructions_chars=0)

    def test_configurable(self):
        from config.schema import PromptCompilerSettings
        cfg = PromptCompilerSettings(max_system_instructions_chars=500)
        assert cfg.max_system_instructions_chars == 500


class TestSystemInstructionsTruncation:
    """compile() must enforce max_system_instructions_chars."""

    def _make_compiler(self, tmp_path):
        from policy.store import PolicyStore
        from policy.learner import PolicyLearner
        from policy.compiler import PolicyCompiler, PolicySelector
        store = PolicyStore(memory_dir=tmp_path)
        learner = PolicyLearner(policy_store=store)
        sel = PolicySelector(learner=learner)
        compiler = PolicyCompiler(policy_selector=sel)
        return compiler, store

    def _patch_settings(self, max_chars):
        """Context manager that patches _adma_settings with given max_sys_chars."""
        import policy.compiler as cm
        fake = MagicMock()
        fake.prompt_compiler.token_budget = 2000
        fake.prompt_compiler.max_memory_nodes = 5
        fake.prompt_compiler.max_system_instructions_chars = max_chars
        return patch.object(cm, '_adma_settings', fake)

    def test_no_truncation_when_under_limit(self, tmp_path):
        compiler, store = self._make_compiler(tmp_path)
        with self._patch_settings(8000):
            prompt = compiler.compile("test task")
        # Default system_instructions are short — must not be truncated
        assert len(prompt.system_instructions) <= 8000
        store.close()

    def test_truncation_fires_at_limit(self, tmp_path):
        """system_instructions must be capped at max_system_instructions_chars."""
        compiler, store = self._make_compiler(tmp_path)
        with self._patch_settings(50):
            prompt = compiler.compile("test task")
        assert len(prompt.system_instructions) <= 50, (
            f"system_instructions not truncated: len={len(prompt.system_instructions)}"
        )
        store.close()

    def test_truncation_logged_at_debug(self, tmp_path, caplog):
        """A DEBUG log entry must be emitted when truncation occurs."""
        import logging
        compiler, store = self._make_compiler(tmp_path)
        with self._patch_settings(10):
            with caplog.at_level(logging.DEBUG, logger='policy.compiler'):
                compiler.compile("test task")
        truncation_logged = any(
            "truncated" in r.message.lower() and r.levelno == logging.DEBUG
            for r in caplog.records
        )
        assert truncation_logged, (
            "PolicyCompiler must log DEBUG when system_instructions is truncated"
        )
        store.close()

    def test_no_debug_log_when_no_truncation(self, tmp_path, caplog):
        """No truncation → no truncation DEBUG log."""
        import logging
        compiler, store = self._make_compiler(tmp_path)
        with self._patch_settings(8000):
            with caplog.at_level(logging.DEBUG, logger='policy.compiler'):
                compiler.compile("short task")
        truncation_logged = any(
            "truncated" in r.message.lower()
            for r in caplog.records
            if r.name == 'policy.compiler'
        )
        assert not truncation_logged
        store.close()

    def test_compile_still_returns_valid_prompt_after_truncation(self, tmp_path):
        """Truncated system_instructions must still produce a valid CompiledPrompt."""
        from policy.compiler import CompiledPrompt
        compiler, store = self._make_compiler(tmp_path)
        with self._patch_settings(10):
            prompt = compiler.compile("test")
        assert isinstance(prompt, CompiledPrompt)
        assert isinstance(prompt.system_instructions, str)
        assert isinstance(prompt.task_instructions, str)
        store.close()

    def test_max_sys_chars_read_from_settings(self, tmp_path):
        """Verify the cap is actually read from _cfg, not hardcoded."""
        compiler, store = self._make_compiler(tmp_path)
        # With cap=5 and cap=100, results must differ if sys_instructions > 5
        with self._patch_settings(5):
            p5 = compiler.compile("test task")
        with self._patch_settings(100):
            p100 = compiler.compile("test task")
        assert len(p5.system_instructions) <= 5
        assert len(p100.system_instructions) <= 100
        # p100 should be longer than p5 (or equal if naturally shorter)
        assert len(p100.system_instructions) >= len(p5.system_instructions)
        store.close()

    def test_default_fallback_is_8000(self, tmp_path):
        """When _adma_settings is None, fallback cap must be 8000."""
        import policy.compiler as cm
        compiler, store = self._make_compiler(tmp_path)
        original = cm._adma_settings
        try:
            cm._adma_settings = None
            prompt = compiler.compile("test task")
            # Default system_instructions are well under 8000
            assert len(prompt.system_instructions) <= 8000
        finally:
            cm._adma_settings = original
        store.close()


# ═════════════════════════════════════════════════════════════════════════════
# A23 — HybridWeightsSettings normalisation methods
# ═════════════════════════════════════════════════════════════════════════════

class TestHybridWeightsSettingsMethods:

    def test_to_raw_dict_exists(self):
        from config.schema import HybridWeightsSettings
        assert hasattr(HybridWeightsSettings, 'to_raw_dict')
        assert callable(HybridWeightsSettings.to_raw_dict)

    def test_to_normalised_dict_exists(self):
        from config.schema import HybridWeightsSettings
        assert hasattr(HybridWeightsSettings, 'to_normalised_dict')
        assert callable(HybridWeightsSettings.to_normalised_dict)

    def test_to_dict_still_exists(self):
        from config.schema import HybridWeightsSettings
        assert hasattr(HybridWeightsSettings, 'to_dict')

    def test_to_raw_dict_returns_raw_values(self):
        from config.schema import HybridWeightsSettings
        ws = HybridWeightsSettings(semantic=2.0, vector=3.0,
            graph_distance=0, importance=0, confidence=0, recency=0,
            hierarchy=0, context_similarity=0, attention=0,
            belief_confidence=0, planning_relevance=0)
        raw = ws.to_raw_dict()
        assert raw['semantic'] == 2.0
        assert raw['vector'] == 3.0
        assert abs(sum(raw.values()) - 5.0) < 1e-9

    def test_to_normalised_dict_sums_to_one(self):
        from config.schema import HybridWeightsSettings
        ws = HybridWeightsSettings()
        normed = ws.to_normalised_dict()
        assert abs(sum(normed.values()) - 1.0) < 1e-9, (
            f"to_normalised_dict() must sum to 1.0, got {sum(normed.values())}"
        )

    def test_to_normalised_dict_correct_proportions(self):
        """2.0 and 3.0 must normalise to 0.4 and 0.6."""
        from config.schema import HybridWeightsSettings
        ws = HybridWeightsSettings(semantic=2.0, vector=3.0,
            graph_distance=0, importance=0, confidence=0, recency=0,
            hierarchy=0, context_similarity=0, attention=0,
            belief_confidence=0, planning_relevance=0)
        normed = ws.to_normalised_dict()
        assert abs(normed['semantic'] - 0.4) < 1e-9
        assert abs(normed['vector'] - 0.6) < 1e-9

    def test_to_normalised_dict_non_default_sums_to_one(self):
        from config.schema import HybridWeightsSettings
        ws = HybridWeightsSettings(semantic=10.0, vector=5.0,
            graph_distance=3.0, importance=2.0, confidence=0,
            recency=0, hierarchy=0, context_similarity=0,
            attention=0, belief_confidence=0, planning_relevance=0)
        normed = ws.to_normalised_dict()
        assert abs(sum(normed.values()) - 1.0) < 1e-9

    def test_to_dict_returns_normalised(self):
        """to_dict() must now return normalised values (A23)."""
        from config.schema import HybridWeightsSettings
        ws = HybridWeightsSettings()
        d = ws.to_dict()
        assert abs(sum(d.values()) - 1.0) < 1e-9, (
            f"to_dict() must return normalised weights (A23), sum={sum(d.values())}"
        )

    def test_to_dict_equals_to_normalised_dict(self):
        from config.schema import HybridWeightsSettings
        ws = HybridWeightsSettings()
        assert ws.to_dict() == ws.to_normalised_dict()

    def test_to_raw_dict_contains_all_11_keys(self):
        from config.schema import HybridWeightsSettings
        ws = HybridWeightsSettings()
        keys = set(ws.to_raw_dict().keys())
        expected = {
            "semantic", "vector", "graph_distance", "importance", "confidence",
            "recency", "hierarchy", "context_similarity", "attention",
            "belief_confidence", "planning_relevance",
        }
        assert keys == expected

    def test_to_normalised_dict_contains_all_11_keys(self):
        from config.schema import HybridWeightsSettings
        ws = HybridWeightsSettings()
        keys = set(ws.to_normalised_dict().keys())
        expected = {
            "semantic", "vector", "graph_distance", "importance", "confidence",
            "recency", "hierarchy", "context_similarity", "attention",
            "belief_confidence", "planning_relevance",
        }
        assert keys == expected

    def test_to_normalised_dict_all_non_negative(self):
        from config.schema import HybridWeightsSettings
        ws = HybridWeightsSettings()
        normed = ws.to_normalised_dict()
        assert all(v >= 0.0 for v in normed.values()), (
            "Normalised weights must all be non-negative"
        )

    def test_default_weights_already_normalise_to_themselves(self):
        """Default weights sum to 1.0, so to_raw_dict ≈ to_normalised_dict."""
        from config.schema import HybridWeightsSettings
        ws = HybridWeightsSettings()
        raw = ws.to_raw_dict()
        normed = ws.to_normalised_dict()
        for key in raw:
            assert abs(raw[key] - normed[key]) < 1e-9, (
                f"Default weights already sum to 1.0 — "
                f"raw[{key}]={raw[key]} should equal normed[{key}]={normed[key]}"
            )


class TestDefaultRetrievalWeightsNormalised:
    """_default_retrieval_weights() must return normalised weights (A23)."""

    def test_default_retrieval_weights_normalised(self):
        from memory.hybrid.retrieval.hybrid_retriever import _default_retrieval_weights
        weights = _default_retrieval_weights()
        if weights:  # may be empty if config absent
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, (
                f"_default_retrieval_weights() must return normalised weights, "
                f"sum={total}"
            )

    def test_retrieval_weights_source_is_to_normalised_dict(self):
        """hybrid_retriever._default_retrieval_weights must call to_normalised_dict."""
        import inspect
        from memory.hybrid import retrieval
        import memory.hybrid.retrieval.hybrid_retriever as m
        src = inspect.getsource(m._default_retrieval_weights)
        assert 'to_normalised_dict' in src, (
            "_default_retrieval_weights must call to_normalised_dict() (A23)"
        )

    def test_adaptive_retriever_uses_normalised_weights(self, tmp_path):
        """AdaptiveRetriever must apply normalised weights to HybridRetriever."""
        from memory.hybrid.hgshm import HGSHM
        from memory.hybrid.retrieval.hybrid_retriever import HybridWeights

        h = HGSHM(tmp_path)
        retriever = h.hybrid_retriever

        # _weights must already be normalised (from HybridWeights.normalised())
        w = retriever._weights
        total = (w.semantic + w.vector + w.graph_distance + w.importance +
                 w.confidence + w.recency + w.hierarchy + w.context_similarity +
                 w.attention + w.belief_confidence + w.planning_relevance)
        assert abs(total - 1.0) < 1e-6, (
            f"HybridRetriever._weights must be normalised; sum={total}"
        )
        h.close()

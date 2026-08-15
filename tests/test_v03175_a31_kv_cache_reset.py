"""
tests/test_v03175_a31_kv_cache_reset.py
=========================================
Regression tests for A31: unbounded KV cache / state isolation between
queries in LLM inference providers.

Reality check
-------------
Neither ``OllamaProvider`` nor ``TransformersProvider`` accumulates
``past_key_values`` or ``conversation_history`` across independent
``generate()`` calls — the HuggingFace pipeline API and the Ollama
single-turn message list are inherently stateless at the Python level.

The fix therefore:
  1. Adds ``reset()`` to ``LLMProvider`` base as a documented no-op hook.
  2. Overrides it in both concrete providers with accurate docstrings.
  3. Provides a stable call-site contract for evaluation harnesses so
     they can call ``llm.reset()`` uniformly without special-casing the
     backend.

These tests verify the contract, not imaginary state.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from llm.base import LLMProvider
from llm.ollama_provider import OllamaProvider
from llm.transformers_provider import TransformersProvider


# ═════════════════════════════════════════════════════════════════════════════
# Base class contract
# ═════════════════════════════════════════════════════════════════════════════

class TestLLMProviderBaseReset:
    """LLMProvider.reset() must be part of the public interface."""

    def test_base_has_reset_method(self):
        assert hasattr(LLMProvider, "reset"), (
            "LLMProvider base must define reset() as part of the public interface"
        )

    def test_reset_is_callable(self):
        assert callable(LLMProvider.reset)

    def test_reset_signature(self):
        sig = inspect.signature(LLMProvider.reset)
        # Only 'self' — no required args
        params = [p for p in sig.parameters if p != "self"]
        assert params == [], f"reset() must take no args beyond self, got: {params}"

    def test_reset_returns_none(self):
        """reset() must return None (no state object or token)."""
        # Use a minimal concrete subclass to call the base no-op
        class _Stub(LLMProvider):
            def generate(self, prompt: str) -> str:
                return ""

        stub = _Stub()
        result = stub.reset()
        assert result is None

    def test_base_reset_is_not_abstract(self):
        """reset() must be a concrete no-op, not abstract — subclasses need not override."""
        # If it were abstract, instantiating _Stub above would fail.
        # Verify directly:
        assert not getattr(LLMProvider.reset, "__isabstractmethod__", False), (
            "reset() must be concrete (default no-op), not abstract"
        )

    def test_reset_docstring_present(self):
        doc = LLMProvider.reset.__doc__
        assert doc and len(doc.strip()) > 20, (
            "reset() must have a meaningful docstring explaining its purpose"
        )


# ═════════════════════════════════════════════════════════════════════════════
# OllamaProvider
# ═════════════════════════════════════════════════════════════════════════════

class TestOllamaProviderReset:
    """OllamaProvider.reset() — stateless client, documented no-op."""

    def test_reset_callable(self):
        provider = OllamaProvider()
        assert callable(provider.reset)

    def test_reset_returns_none(self):
        provider = OllamaProvider()
        assert provider.reset() is None

    def test_reset_idempotent(self):
        """Calling reset() multiple times must not raise."""
        provider = OllamaProvider()
        provider.reset()
        provider.reset()
        provider.reset()

    def test_no_past_key_values(self):
        """OllamaProvider must not hold past_key_values — it's stateless."""
        provider = OllamaProvider()
        assert not hasattr(provider, "past_key_values"), (
            "OllamaProvider must not store past_key_values"
        )

    def test_no_conversation_history(self):
        """OllamaProvider must not accumulate conversation_history."""
        provider = OllamaProvider()
        assert not hasattr(provider, "conversation_history"), (
            "OllamaProvider must not store conversation_history — "
            "each generate() call sends a fresh single-turn message"
        )

    def test_reset_does_not_break_subsequent_generate(self):
        """After reset(), generate() must still work (via mock)."""
        import sys, types
        fake_ollama = types.ModuleType("ollama")
        fake_ollama.chat = MagicMock(return_value={"message": {"content": "Paris"}})
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            provider = OllamaProvider()
            provider.reset()
            result = provider.generate("What is the capital of France?")
        assert result == "Paris"

    def test_reset_overrides_base(self):
        """OllamaProvider must override reset() (not just inherit the base no-op)."""
        # The override has a provider-specific docstring — verify it's distinct
        assert OllamaProvider.reset is not LLMProvider.reset, (
            "OllamaProvider should override reset() with a provider-specific implementation"
        )

    def test_each_generate_is_stateless(self):
        """Two generate() calls must each produce independent results (no shared state)."""
        import sys, types
        responses = [
            {"message": {"content": "answer_1"}},
            {"message": {"content": "answer_2"}},
        ]
        fake_ollama = types.ModuleType("ollama")
        fake_ollama.chat = MagicMock(side_effect=responses)
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            provider = OllamaProvider()
            r1 = provider.generate("Q1")
            provider.reset()
            r2 = provider.generate("Q2")
        assert r1 == "answer_1"
        assert r2 == "answer_2"


# ═════════════════════════════════════════════════════════════════════════════
# TransformersProvider
# ═════════════════════════════════════════════════════════════════════════════

class TestTransformersProviderReset:
    """TransformersProvider.reset() — pipeline is stateless, documented no-op."""

    def test_reset_callable(self):
        provider = TransformersProvider()
        assert callable(provider.reset)

    def test_reset_returns_none(self):
        provider = TransformersProvider()
        assert provider.reset() is None

    def test_reset_idempotent(self):
        provider = TransformersProvider()
        provider.reset()
        provider.reset()
        provider.reset()

    def test_no_past_key_values(self):
        """TransformersProvider must not store past_key_values between calls."""
        provider = TransformersProvider()
        assert not hasattr(provider, "past_key_values"), (
            "TransformersProvider must not store past_key_values — "
            "the HuggingFace pipeline discards KV state after each call"
        )

    def test_no_conversation_history(self):
        provider = TransformersProvider()
        assert not hasattr(provider, "conversation_history")

    def test_reset_before_load_does_not_crash(self):
        """reset() called before _load() must not raise."""
        provider = TransformersProvider()
        assert provider.is_loaded is False
        provider.reset()  # must not raise
        assert provider.is_loaded is False

    def test_reset_overrides_base(self):
        assert TransformersProvider.reset is not LLMProvider.reset, (
            "TransformersProvider should override reset() with a provider-specific docstring"
        )

    def test_is_loaded_false_initially(self):
        provider = TransformersProvider()
        assert provider.is_loaded is False

    def test_reset_after_mock_load_does_not_affect_is_loaded(self):
        """
        reset() is a state-isolation hook, not an unload trigger.
        is_loaded must remain True after reset() if the model is loaded —
        we do not want to force a reload between every benchmark query.
        """
        provider = TransformersProvider()
        # Simulate a loaded state by setting _pipe directly
        provider._pipe = MagicMock()
        assert provider.is_loaded is True

        provider.reset()
        # Pipeline should still be loaded — reset() does not unload the model
        assert provider.is_loaded is True

    def test_reset_does_not_break_subsequent_generate(self):
        """After reset(), generate() must still invoke the pipeline correctly."""
        provider = TransformersProvider()

        mock_pipe_output = [{"generated_text": "  mocked answer  "}]
        mock_pipe = MagicMock(return_value=mock_pipe_output)

        # Pre-load with the mock
        provider._pipe = mock_pipe
        provider._tokenizer = MagicMock()
        provider._tokenizer.chat_template = None  # skip apply_chat_template path
        provider._tokenizer.eos_token_id = 0

        provider.reset()
        result = provider.generate("test prompt")
        assert result == "mocked answer"
        mock_pipe.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
# Harness contract — uniform reset() across providers
# ═════════════════════════════════════════════════════════════════════════════

class TestHarnessContract:
    """
    Verify that a benchmark harness can call llm.reset() uniformly on
    any LLMProvider without special-casing the backend.
    """

    def test_all_providers_have_reset(self):
        """Every concrete provider class must expose reset()."""
        for cls in [OllamaProvider, TransformersProvider]:
            assert hasattr(cls, "reset") and callable(cls.reset), (
                f"{cls.__name__} must have a callable reset() method"
            )

    def test_harness_loop_pattern(self):
        """
        Simulate the evaluation harness loop:
            for sample in dataset:
                llm.reset()
                answer = llm.generate(sample["question"])
        Must complete without errors for both providers.
        """
        dataset = [{"question": f"Q{i}"} for i in range(5)]

        # OllamaProvider
        import sys, types
        fake_ollama = types.ModuleType("ollama")
        fake_ollama.chat = MagicMock(return_value={"message": {"content": "ok"}})
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            op = OllamaProvider()
            for sample in dataset:
                op.reset()
                op.generate(sample["question"])

        # TransformersProvider (mocked pipeline)
        tp = TransformersProvider()
        tp._pipe = MagicMock(return_value=[{"generated_text": "ok"}])
        tp._tokenizer = MagicMock()
        tp._tokenizer.chat_template = None
        tp._tokenizer.eos_token_id = 0
        for sample in dataset:
            tp.reset()
            tp.generate(sample["question"])

    def test_reset_return_value_ignored_safely(self):
        """
        Harness code written as `llm.reset()` (ignoring return value)
        must work — reset() must return None, not a generator or awaitable.
        """
        for provider in [OllamaProvider(), TransformersProvider()]:
            ret = provider.reset()
            assert ret is None, (
                f"{type(provider).__name__}.reset() must return None, got {type(ret)}"
            )

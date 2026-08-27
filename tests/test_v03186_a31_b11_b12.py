"""
tests/test_v03186_a31_b11_b12.py
==================================
Regression tests for:

  A31 — LLMProvider.reset() hook verified across all providers (no-op where
         stateless, hook always callable, harness pattern documented).

  B11 — GlobalWorkspace isolation model documented and verified:
         separate instances share no state; class has no singletons.

  B12 — Silent except Exception passes improved:
         · memory/future_memory.py: corrupt JSON now logs WARNING + typed catch
         · api/context.py: curiosity engine callback now logs WARNING
         · 9 remaining silent passes (shutdown/teardown/rollback) left as-is
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════════════
# A31 — LLMProvider.reset() hook
# ═════════════════════════════════════════════════════════════════════════════

class TestLLMProviderResetHook:

    def test_base_class_has_reset(self):
        from llm.base import LLMProvider
        assert hasattr(LLMProvider, 'reset')
        assert callable(LLMProvider.reset)

    def test_base_reset_is_not_abstract(self):
        """reset() must be a concrete no-op default, not abstract."""
        import inspect
        from llm.base import LLMProvider
        # If it were abstract it would appear in __abstractmethods__
        assert 'reset' not in getattr(LLMProvider, '__abstractmethods__', set())

    def test_transformers_provider_has_reset(self):
        from llm.transformers_provider import TransformersProvider
        assert hasattr(TransformersProvider, 'reset')
        assert callable(TransformersProvider.reset)

    def test_ollama_provider_has_reset(self):
        from llm.ollama_provider import OllamaProvider
        assert hasattr(OllamaProvider, 'reset')
        assert callable(OllamaProvider.reset)

    def test_transformers_reset_does_not_raise_before_load(self):
        """reset() must be safe to call before the model is loaded."""
        from llm.transformers_provider import TransformersProvider
        tp = TransformersProvider.__new__(TransformersProvider)
        tp._pipe = None
        tp._tokenizer = None
        tp._model = None
        tp.reset()  # must not raise

    def test_ollama_reset_does_not_raise(self):
        from llm.ollama_provider import OllamaProvider
        op = OllamaProvider.__new__(OllamaProvider)
        op.reset()  # must not raise

    def test_reset_callable_on_base_instance_via_subclass(self):
        """Concrete subclass must inherit reset() if it does not override."""
        from llm.base import LLMProvider

        class MinimalProvider(LLMProvider):
            def generate(self, prompt: str) -> str:
                return ""

        mp = MinimalProvider()
        mp.reset()  # must not raise — inherited no-op

    def test_all_providers_covered(self):
        """Every .py in llm/ that defines an LLMProvider subclass must have reset()."""
        import importlib, inspect
        from llm.base import LLMProvider
        llm_dir = PROJECT_ROOT / 'llm'
        for py in llm_dir.glob('*.py'):
            if py.name in ('__init__.py', 'base.py', 'provider_factory.py'):
                continue
            mod = importlib.import_module(f'llm.{py.stem}')
            for name, cls in inspect.getmembers(mod, inspect.isclass):
                if issubclass(cls, LLMProvider) and cls is not LLMProvider:
                    assert hasattr(cls, 'reset'), (
                        f"{name} in {py.name} must have a reset() method (A31)"
                    )

    def test_harness_pattern_documented_in_base(self):
        """base.py docstring must show llm.reset() harness usage example."""
        from llm.base import LLMProvider
        doc = LLMProvider.reset.__doc__ or ''
        assert 'reset' in doc.lower()
        assert 'harness' in doc.lower() or 'benchmark' in doc.lower() or 'isolat' in doc.lower()


# ═════════════════════════════════════════════════════════════════════════════
# B11 — GlobalWorkspace isolation model
# ═════════════════════════════════════════════════════════════════════════════

class TestGlobalWorkspaceIsolation:

    def test_no_class_level_shared_state(self):
        """GlobalWorkspace must not have mutable class-level attributes."""
        from workspace.global_workspace import GlobalWorkspace
        # Only __dict__ of instances should hold mutable state
        class_vars = {
            k: v for k, v in vars(GlobalWorkspace).items()
            if not k.startswith('_') or k in ('__dict__', '__weakref__')
        }
        # None of the class-level items should be mutable containers
        for k, v in class_vars.items():
            assert not isinstance(v, (list, dict, set)), (
                f"GlobalWorkspace.{k} is a mutable class-level {type(v).__name__} — "
                f"shared state risk (B11)"
            )

    def test_no_module_level_singleton(self):
        """There must be no global GlobalWorkspace instance in the module."""
        import workspace.global_workspace as gw_mod
        for name in dir(gw_mod):
            val = getattr(gw_mod, name)
            from workspace.global_workspace import GlobalWorkspace
            assert not isinstance(val, GlobalWorkspace), (
                f"Module-level GlobalWorkspace singleton '{name}' found — "
                f"cross-session leak risk (B11)"
            )

    def test_two_instances_have_independent_memory(self):
        from workspace.global_workspace import GlobalWorkspace
        gw1 = GlobalWorkspace()
        gw2 = GlobalWorkspace()
        assert gw1._memory is not gw2._memory

    def test_two_instances_have_independent_attention(self):
        from workspace.global_workspace import GlobalWorkspace
        gw1 = GlobalWorkspace()
        gw2 = GlobalWorkspace()
        assert gw1._attention is not gw2._attention

    def test_two_instances_have_independent_pending_queues(self):
        from workspace.global_workspace import GlobalWorkspace
        gw1 = GlobalWorkspace()
        gw2 = GlobalWorkspace()
        assert gw1._pending is not gw2._pending

    def test_two_instances_have_independent_broadcast_bus(self):
        from workspace.global_workspace import GlobalWorkspace
        gw1 = GlobalWorkspace()
        gw2 = GlobalWorkspace()
        assert gw1._broadcast is not gw2._broadcast

    def test_pending_queue_not_shared(self):
        """Items submitted to one workspace must not appear in another."""
        from workspace.global_workspace import GlobalWorkspace
        from workspace.attention_manager import AttentionCandidate
        gw1 = GlobalWorkspace()
        gw2 = GlobalWorkspace()
        candidate = AttentionCandidate(ref_id="r1", source="test", content_summary="test item")
        gw1.submit_candidate(candidate)
        assert gw1.pending_count == 1
        assert gw2.pending_count == 0

    def test_cycle_count_not_shared(self):
        from workspace.global_workspace import GlobalWorkspace
        gw1 = GlobalWorkspace()
        gw2 = GlobalWorkspace()
        gw1._cycle_count = 42
        assert gw2._cycle_count == 0

    def test_docstring_documents_isolation(self):
        from workspace.global_workspace import GlobalWorkspace
        doc = GlobalWorkspace.__doc__ or ''
        assert 'isolation' in doc.lower() or 'isolat' in doc.lower(), (
            "GlobalWorkspace docstring must document the isolation model (B11)"
        )

    def test_injected_dependencies_honoured(self):
        """Custom injected WorkspaceMemory must be used, not replaced."""
        from workspace.global_workspace import GlobalWorkspace
        from workspace.workspace_memory import WorkspaceMemory
        custom_mem = WorkspaceMemory()
        gw = GlobalWorkspace(workspace_memory=custom_mem)
        assert gw._memory is custom_mem


# ═════════════════════════════════════════════════════════════════════════════
# B12 — Silent except Exception handlers improved
# ═════════════════════════════════════════════════════════════════════════════

class TestFutureMemoryCorruptFile:

    def test_corrupt_json_does_not_raise(self, tmp_path):
        from memory.future_memory import FutureMemoryStore
        p = tmp_path / "future.json"
        p.write_text("{{not valid json}}")
        fm = FutureMemoryStore(future_memory_file=p)
        fm._load()  # must not raise
        assert len(fm._states) == 0

    def test_corrupt_json_logs_warning(self, tmp_path, caplog):
        from memory.future_memory import FutureMemoryStore
        p = tmp_path / "future.json"
        p.write_text("{{bad}}")
        fm = FutureMemoryStore(future_memory_file=p)
        with caplog.at_level(logging.WARNING, logger='memory.future_memory'):
            fm._load()
        assert any('future_memory' in r.message.lower() or
                   'could not load' in r.message.lower() or
                   'state' in r.message.lower()
                   for r in caplog.records), (
            "Corrupt JSON load must emit a WARNING log (B12)"
        )

    def test_truncated_file_does_not_raise(self, tmp_path):
        from memory.future_memory import FutureMemoryStore
        p = tmp_path / "future.json"
        p.write_bytes(b'[{"expected_state_id": "abc"')  # truncated
        fm = FutureMemoryStore(future_memory_file=p)
        fm._load()  # must not raise

    def test_wrong_type_in_file_does_not_raise(self, tmp_path):
        from memory.future_memory import FutureMemoryStore
        p = tmp_path / "future.json"
        p.write_text(json.dumps([{"unexpected_key": 1}]))
        fm = FutureMemoryStore(future_memory_file=p)
        fm._load()  # must not raise

    def test_missing_file_still_works(self, tmp_path):
        from memory.future_memory import FutureMemoryStore
        p = tmp_path / "nonexistent.json"
        fm = FutureMemoryStore(future_memory_file=p)
        fm._load()  # must not raise — file does not exist
        assert len(fm._states) == 0

    def test_exception_is_not_broad_anymore(self):
        """The except clause must name specific types, not bare Exception (B12)."""
        import inspect
        from memory.future_memory import FutureMemoryStore
        src = inspect.getsource(FutureMemoryStore._load)
        # Must NOT have a bare 'except Exception:' (with no named types) followed by pass
        import re
        bare_silent = re.search(r'except\s+Exception\s*:\s*\n\s*pass', src)
        assert bare_silent is None, (
            "_load() must not have a bare 'except Exception: pass' (B12)"
        )

    def test_future_memory_has_logger(self):
        import memory.future_memory as fm_mod
        assert hasattr(fm_mod, 'log'), "future_memory module must have a logger (B12)"


class TestCuriosityEngineCallback:

    def test_callback_logs_on_generate_signals_failure(self, tmp_path):
        """curiosity_engine.generate_signals failure must log WARNING, not silently pass."""
        src = (PROJECT_ROOT / 'api' / 'context.py').read_text()
        assert 'curiosity_engine.generate_signals failed' in src or \
               'generate_signals' in src and 'log.warning' in src, (
            "curiosity_engine callback must log on failure (B12)"
        )

    def test_callback_silent_pass_removed(self):
        """The raw 'except Exception: pass' in the curiosity callback must be gone."""
        import re
        src = (PROJECT_ROOT / 'api' / 'context.py').read_text()
        # Find the block around generate_signals
        idx = src.find('curiosity_engine.generate_signals')
        assert idx != -1
        block = src[idx:idx + 300]
        bare = re.search(r'except\s+Exception\s*:\s*\n\s*pass', block)
        assert bare is None, (
            "Bare 'except Exception: pass' must be replaced with a log.warning call (B12)"
        )

    def test_justified_silent_passes_remain(self):
        """Shutdown-sequence silent passes in context.py must still be present
        (they are intentionally silent — best-effort cleanup must not crash)."""
        src = (PROJECT_ROOT / 'api' / 'context.py').read_text()
        # The shutdown block contains multiple try/except: pass — count them
        import re
        silent_count = len(re.findall(r'except\s+Exception\s*:\s*\n\s*pass', src))
        # There should be at least 3 remaining (persist, expire_stale, agent.shutdown,
        # background_processor.shutdown — all best-effort cleanup on teardown)
        assert silent_count >= 3, (
            f"Expected ≥3 justified silent passes in shutdown sequence, got {silent_count}"
        )

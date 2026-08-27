"""
tests/test_v03198_module_mismatches.py
========================================
Regression tests for cross-module path/name mismatches found in audit:

  MISMATCH 1 — eval_harness.py: `from core.retriever import SemanticRetriever`
               module core.retriever does not exist.
               Fix: `from core.semantic_retriever import SemanticRetriever`

  MISMATCH 2 — eval_harness.py: `create_provider(model=llm_model)`
               function is actually `build_provider(cfg: LLMSettings)`.
               Fix: use `build_provider` from `llm.provider_factory`.

  MISMATCH 3 — eval_harness.py: `agent.retrieve_memory(q).all_memories()`
               retrieve_memory() returns list[MemoryEntry], not MemoryContext.
               MemoryEntry has no .text attribute — use .output instead.
               Fix: iterate list directly, access .output/.input.

  MISMATCH 4 — pyproject.toml: no package exclusion config meant auto-discovery
               installed tests/, experiments/, hypothesis/, learning/ into wheel.
               Fix: [tool.setuptools.packages.find] exclude list added.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# MISMATCH 1 — SemanticRetriever import path
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticRetrieverImport:

    def test_core_retriever_module_does_not_exist(self):
        """core.retriever must not exist — it never did."""
        import importlib
        spec = importlib.util.find_spec("core.retriever")
        assert spec is None, (
            "core.retriever should not exist — the module is core.semantic_retriever"
        )

    def test_core_semantic_retriever_exists(self):
        from core.semantic_retriever import SemanticRetriever
        assert SemanticRetriever is not None

    def test_semantic_retriever_correct_constructor(self):
        """SemanticRetriever takes embedding_store + legacy_retriever, not hgshm."""
        import inspect
        from core.semantic_retriever import SemanticRetriever
        sig = inspect.signature(SemanticRetriever.__init__)
        params = set(sig.parameters.keys())
        assert 'embedding_store' in params, (
            "SemanticRetriever.__init__ must have 'embedding_store'"
        )
        assert 'legacy_retriever' in params, (
            "SemanticRetriever.__init__ must have 'legacy_retriever'"
        )
        assert 'hgshm' not in params, (
            "SemanticRetriever does not accept 'hgshm' — that was the wrong constructor call"
        )

    def test_eval_harness_uses_correct_import(self):
        src = (PROJECT_ROOT / 'eval_harness.py').read_text()
        assert 'from core.semantic_retriever import SemanticRetriever' in src, (
            "eval_harness must import SemanticRetriever from core.semantic_retriever"
        )
        assert 'from core.retriever import SemanticRetriever' not in src, (
            "eval_harness must not use the non-existent core.retriever module"
        )

    def test_eval_harness_no_hgshm_constructor(self):
        src = (PROJECT_ROOT / 'eval_harness.py').read_text()
        assert 'SemanticRetriever(hgshm=' not in src, (
            "eval_harness must not call SemanticRetriever(hgshm=...) — wrong constructor"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MISMATCH 2 — create_provider vs build_provider
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderFactory:

    def test_create_provider_does_not_exist(self):
        """create_provider was never defined — the function is build_provider."""
        import importlib
        mod = importlib.import_module('llm.provider_factory')
        assert not hasattr(mod, 'create_provider'), (
            "create_provider should not exist in llm.provider_factory"
        )

    def test_build_provider_exists(self):
        from llm.provider_factory import build_provider
        assert callable(build_provider)

    def test_build_provider_takes_llmsettings(self):
        import inspect
        from llm.provider_factory import build_provider
        sig = inspect.signature(build_provider)
        params = list(sig.parameters.keys())
        assert 'cfg' in params, (
            f"build_provider must accept 'cfg', got {params}"
        )

    def test_llmsettings_importable_from_config_settings(self):
        from config.settings import LLMSettings
        assert LLMSettings is not None

    def test_eval_harness_uses_build_provider(self):
        src = (PROJECT_ROOT / 'eval_harness.py').read_text()
        assert 'build_provider' in src, (
            "eval_harness must use build_provider, not create_provider"
        )
        # Count raw occurrences — only docstring comment references are acceptable
        occurrences = src.count('create_provider')
        assert occurrences == 0, (
            "eval_harness must not reference the non-existent create_provider"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MISMATCH 3 — retrieve_memory() return type
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieveMemoryReturnType:

    def test_retrieve_memory_returns_list(self):
        """TutorAgent.retrieve_memory() returns list[MemoryEntry], not MemoryContext."""
        import inspect
        from core.tutor_agent import TutorAgent
        sig = inspect.signature(TutorAgent.retrieve_memory)
        # Check return annotation if present
        ret = sig.return_annotation
        if ret != inspect.Parameter.empty:
            assert 'MemoryContext' not in str(ret), (
                "retrieve_memory return type must not be MemoryContext"
            )

    def test_memory_entry_has_no_text_attr(self):
        """MemoryEntry has no .text — harness must use .output or .input."""
        from schemas.memory_entry import MemoryEntry
        fields = set(MemoryEntry.model_fields.keys())
        assert 'text' not in fields, (
            f"MemoryEntry has no 'text' field — available: {sorted(fields)}"
        )
        assert 'output' in fields or 'input' in fields, (
            f"MemoryEntry must have 'output' or 'input' — got {sorted(fields)}"
        )

    def test_eval_harness_no_all_memories_call(self):
        src = (PROJECT_ROOT / 'eval_harness.py').read_text()
        assert '.all_memories()' not in src, (
            "eval_harness must not call .all_memories() on retrieve_memory() result — "
            "it returns a list, not MemoryContext"
        )

    def test_eval_harness_no_r_text(self):
        """r.text was the wrong field access — should be r.output or getattr fallback."""
        src = (PROJECT_ROOT / 'eval_harness.py').read_text()
        import re
        # Look for 'r.text' in the retrieved comprehension context
        bad_pattern = re.search(r'for r in.*retrieve_memory.*\br\.text\b', src, re.DOTALL)
        assert bad_pattern is None, (
            "eval_harness must not use r.text from retrieve_memory() results"
        )

    def test_eval_harness_uses_output_attribute(self):
        src = (PROJECT_ROOT / 'eval_harness.py').read_text()
        # Should use getattr(r, 'output', ...) pattern
        assert "'output'" in src or '"output"' in src, (
            "eval_harness must access 'output' attribute from MemoryEntry"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MISMATCH 4 — pyproject.toml dev package exclusions
# ─────────────────────────────────────────────────────────────────────────────

class TestPackageExclusions:

    def test_find_packages_config_present(self):
        src = (PROJECT_ROOT / 'pyproject.toml').read_text()
        assert '[tool.setuptools.packages.find]' in src, (
            "pyproject.toml must have [tool.setuptools.packages.find] to control discovery"
        )

    def test_tests_excluded(self):
        src = (PROJECT_ROOT / 'pyproject.toml').read_text()
        assert '"tests"' in src or "'tests'" in src, (
            "tests/ must be excluded from wheel — end users shouldn't get test code"
        )

    def test_experiments_excluded(self):
        src = (PROJECT_ROOT / 'pyproject.toml').read_text()
        assert 'experiments' in src.split('[tool.setuptools.packages.find]')[1].split('[project]')[0], (
            "experiments/ must be excluded from wheel"
        )

    def test_exclude_covers_subpackages(self):
        """Exclusions must cover subpackages (e.g. tests.* not just tests)."""
        src = (PROJECT_ROOT / 'pyproject.toml').read_text()
        find_block = src.split('[tool.setuptools.packages.find]')[1].split('[project]')[0]
        assert 'tests.*' in find_block, (
            "Exclusion must include 'tests.*' to cover subpackages"
        )

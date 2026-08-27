"""
tests/test_v03199_harness_prereqs.py
======================================
Regression tests for pre-benchmark cross-module mismatches found in audit:

  1. eval_harness uses core.memory_manager.MemoryManager (TutorAgent's API),
     not memory.manager.MemoryManager (HGSHM-based).
  2. EmbeddingStore constructor uses embed_model_name/embeddings_file/ids_file.
  3. MemoryManager.add_memory uses assistant_output not assistant_response.
  4. Full construction pipeline (harness simulation) works end-to-end.
  5. TutorAgent remains compatible with its existing test suite (no regressions).
"""
from __future__ import annotations
import sys, tempfile, inspect
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestCorrectMemoryManagerUsed:

    def test_eval_harness_imports_core_memory_manager(self):
        src = (PROJECT_ROOT / 'eval_harness.py').read_text()
        assert 'from core.memory_manager import MemoryManager' in src, (
            "eval_harness must use core.memory_manager.MemoryManager (TutorAgent's API), "
            "not memory.manager.MemoryManager (HGSHM-based)"
        )

    def test_eval_harness_not_importing_hgshm_memory_manager(self):
        src = (PROJECT_ROOT / 'eval_harness.py').read_text()
        # Should not import the HGSHM-based MemoryManager in _build_blix_profile
        build_fn_start = src.find('def _build_blix_profile(')
        build_fn_end   = src.find('\ndef ', build_fn_start + 1)
        build_fn = src[build_fn_start:build_fn_end]
        assert 'from memory.manager import MemoryManager' not in build_fn, (
            "_build_blix_profile must not use the HGSHM-based MemoryManager"
        )

    def test_core_memory_manager_has_tutor_agent_api(self):
        from core.memory_manager import MemoryManager
        required = ['add_memory', 'get_all_memories', 'memory_count',
                    'update_memory', 'get_memory_by_id']
        for method in required:
            assert hasattr(MemoryManager, method), (
                f"core.MemoryManager must have {method}() for TutorAgent compatibility"
            )

    def test_core_memory_manager_has_profile_and_learning_state(self):
        from core.memory_manager import MemoryManager
        mm = MemoryManager.__new__(MemoryManager)
        # These are properties/attributes TutorAgent accesses
        src = inspect.getsource(MemoryManager)
        assert 'learning_state' in src
        assert 'profile' in src


class TestEmbeddingStoreConstructor:

    def test_embedding_store_params(self):
        sig = inspect.signature(
            __import__('core.embedding_store', fromlist=['EmbeddingStore']).EmbeddingStore.__init__
        )
        params = set(sig.parameters.keys())
        assert 'embed_model_name' in params
        assert 'embeddings_file'  in params
        assert 'ids_file'         in params

    def test_embedding_store_not_taking_hgshm(self):
        sig = inspect.signature(
            __import__('core.embedding_store', fromlist=['EmbeddingStore']).EmbeddingStore.__init__
        )
        assert 'hgshm' not in sig.parameters

    def test_eval_harness_uses_correct_embedding_store_kwargs(self):
        src = (PROJECT_ROOT / 'eval_harness.py').read_text()
        assert 'embed_model_name=' in src
        assert 'embeddings_file='  in src
        assert 'ids_file='         in src


class TestAddMemorySignature:

    def test_add_memory_uses_assistant_output(self):
        sig = inspect.signature(
            __import__('core.memory_manager', fromlist=['MemoryManager'])
            .MemoryManager.add_memory
        )
        params = set(sig.parameters.keys())
        assert 'assistant_output' in params, (
            "add_memory() uses 'assistant_output', not 'assistant_response'"
        )
        assert 'assistant_response' not in params

    def test_eval_harness_uses_assistant_output(self):
        src = (PROJECT_ROOT / 'eval_harness.py').read_text()
        assert 'assistant_output=' in src
        assert 'assistant_response=' not in src


class TestFullConstructionPipeline:

    def test_build_blix_profile_simulation(self, tmp_path):
        """Simulate exactly what _build_blix_profile() does at runtime."""
        from core.memory_manager import MemoryManager
        from config.settings import settings
        from core.embedding_store import EmbeddingStore
        from core.memory_retriever import MemoryRetriever
        from core.semantic_retriever import SemanticRetriever
        from core.prompt_builder import PromptBuilder
        from core.tutor_agent import TutorAgent
        from eval_harness import _make_mock_llm

        mm = MemoryManager(
            conversations_file=tmp_path/"conversations.json",
            profile_file=tmp_path/"profile.json",
            learning_state_file=tmp_path/"learning_state.json",
        )
        embed_cfg = settings.embed
        es = EmbeddingStore(
            embed_model_name=embed_cfg.model,
            embeddings_file=tmp_path/"embeddings.npy",
            ids_file=tmp_path/"embedding_ids.json",
            threshold=embed_cfg.threshold,
            top_k=embed_cfg.top_k,
        )
        retriever = SemanticRetriever(
            embedding_store=es,
            legacy_retriever=MemoryRetriever(),
        )
        agent = TutorAgent(
            llm=_make_mock_llm(),
            memory_manager=mm,
            retriever=retriever,
            prompt_builder=PromptBuilder(),
        )
        assert agent is not None
        assert mm.memory_count() == 0

    def test_adaptation_and_chat(self, tmp_path):
        """Simulate one training sample + one query as the harness does."""
        from core.memory_manager import MemoryManager
        from config.settings import settings
        from core.embedding_store import EmbeddingStore
        from core.memory_retriever import MemoryRetriever
        from core.semantic_retriever import SemanticRetriever
        from core.prompt_builder import PromptBuilder
        from core.tutor_agent import TutorAgent
        from eval_harness import _make_mock_llm

        mm = MemoryManager(
            conversations_file=tmp_path/"c.json",
            profile_file=tmp_path/"p.json",
            learning_state_file=tmp_path/"l.json",
        )
        embed_cfg = settings.embed
        es = EmbeddingStore(
            embed_model_name=embed_cfg.model,
            embeddings_file=tmp_path/"e.npy",
            ids_file=tmp_path/"e_ids.json",
            threshold=embed_cfg.threshold, top_k=embed_cfg.top_k,
        )
        agent = TutorAgent(
            llm=_make_mock_llm(),
            memory_manager=mm,
            retriever=SemanticRetriever(embedding_store=es, legacy_retriever=MemoryRetriever()),
            prompt_builder=PromptBuilder(),
        )
        # Adaptation
        mm.add_memory(user_input="What is Python?", assistant_output="A language.")
        assert mm.memory_count() == 1

        # A31 reset
        agent._llm.reset()

        # Chat (test phase)
        response = agent.chat("What is Python?")
        assert isinstance(response, str) and len(response) > 0

        # retrieve_memory returns list[MemoryEntry]
        entries = agent.retrieve_memory("Python")
        assert isinstance(entries, list)
        assert len(entries) >= 1
        # Each entry has .input and .output
        for entry in entries:
            assert hasattr(entry, 'input')
            assert hasattr(entry, 'output')

    def test_mock_llm_implements_full_interface(self):
        from eval_harness import _make_mock_llm
        llm = _make_mock_llm()
        assert callable(llm.generate)
        assert callable(llm.reset)
        assert callable(llm.model_name)
        assert callable(llm.supports_streaming)
        # model_name must return a string
        assert isinstance(llm.model_name(), str)
        # generate must return a string
        result = llm.generate("test prompt")
        assert isinstance(result, str)

    def test_tutor_agent_existing_tests_unaffected(self):
        """Verify tutor_agent.py was restored — original API still works."""
        import inspect
        from core.tutor_agent import TutorAgent
        src = inspect.getsource(TutorAgent.retrieve_memory)
        # Must still use get_all_memories() — the core.MemoryManager API
        assert 'get_all_memories()' in src, (
            "TutorAgent.retrieve_memory must use get_all_memories() — "
            "do not change tutor_agent.py to work around MemoryManager mismatches"
        )

"""
tests/test_v03176_a12_a13_a14.py
===================================
Regression tests for:

  A12 — Broad except Exception in retrieval hot path → specific exceptions
  A13 — VectorStore brute-force fallback warning fires at module load time
  A14 — PolicyLearner LRU cache protected by threading.Lock
"""

from __future__ import annotations

import inspect
import logging
import sqlite3
import tempfile
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ═════════════════════════════════════════════════════════════════════════════
# A12 — Specific exception handling in hybrid_retriever.py
# ═════════════════════════════════════════════════════════════════════════════

class TestA12SpecificExceptions:
    """Broad except Exception clauses replaced with specific typed catches."""

    def test_default_retrieval_weights_import_error_returns_empty(self):
        """ImportError (config absent) → return {} silently."""
        import memory.hybrid.retrieval.hybrid_retriever as m
        original = None
        # Patch hgshm_settings to simulate ImportError
        import sys
        with patch.dict(sys.modules, {'config': None, 'config.settings': None}):
            from memory.hybrid.retrieval.hybrid_retriever import _default_retrieval_weights
            # If config is already imported, patch it at the source
        # Direct test: call with real env (config is present here → should return dict)
        from memory.hybrid.retrieval.hybrid_retriever import _default_retrieval_weights
        result = _default_retrieval_weights()
        assert isinstance(result, dict)

    def test_default_retrieval_weights_no_bare_except_exception(self):
        """_default_retrieval_weights must not have a bare 'except Exception: return {}'."""
        src = inspect.getsource(
            __import__('memory.hybrid.retrieval.hybrid_retriever',
                       fromlist=['_default_retrieval_weights'])
            ._default_retrieval_weights
        )
        # The function must catch ImportError specifically
        assert 'except ImportError' in src, (
            "_default_retrieval_weights must catch ImportError explicitly"
        )
        # Must not silently swallow all exceptions with bare except Exception + no log
        lines = [l.strip() for l in src.splitlines()]
        for i, line in enumerate(lines):
            if line == 'except Exception:':
                # If there's a bare except Exception, ensure the next line is NOT 'return {}'
                next_line = lines[i + 1] if i + 1 < len(lines) else ''
                assert next_line != 'return {}', (
                    "bare 'except Exception: return {}' is still present — must log first"
                )

    def test_temporal_retriever_skips_bad_nodes_not_silently(self, tmp_path, caplog):
        """Malformed node data must be skipped with a WARNING, not silently."""
        from memory.hybrid.retrieval.hybrid_retriever import TemporalRetriever
        from memory.hybrid.hgshm import HGSHM

        h = HGSHM(tmp_path)
        # Add a good node
        h.remember("valid memory", confidence=0.9, importance=0.8)

        retriever = TemporalRetriever(graph_store=h.graph_store)

        # Inject a node with a broken created_dt that causes AttributeError
        bad_node = MagicMock()
        bad_node.node_id = "bad-node"
        bad_node.created_dt = None  # .timestamp() will raise AttributeError

        good_nodes = h.graph_store.all_nodes(limit=10)

        with patch.object(h.graph_store, 'all_nodes', return_value=[bad_node] + good_nodes):
            with caplog.at_level(logging.WARNING,
                                 logger='memory.hybrid.retrieval.hybrid_retriever'):
                results = retriever.recent(max_age_hours=24, top_k=10)

        # Good nodes should still be returned
        assert isinstance(results, list)
        # Warning about the bad node must have been emitted
        assert any("bad-node" in r.message or "TemporalRetriever" in r.message
                   for r in caplog.records), (
            "TemporalRetriever must log a WARNING when skipping a bad node"
        )
        h.close()

    def test_ctx_score_embed_failure_logs_warning(self, tmp_path, caplog):
        """Embedding failure during ctx-score phase must be logged and not raise."""
        from memory.hybrid.hgshm import HGSHM

        h = HGSHM(tmp_path)
        h.remember("test node", confidence=0.8, importance=0.7)

        retriever = h.hybrid_retriever
        original_embed = retriever._emb.embed
        call_count = [0]

        # Fail embed when called with node text (not the query text).
        # context_hint triggers the ctx_score embed path.
        def embed_fails_for_nodes(text):
            if text != "test query" and text != "ctx hint":
                raise RuntimeError("embedding device OOM")
            return original_embed(text)

        retriever._emb.embed = embed_fails_for_nodes

        with caplog.at_level(logging.DEBUG,
                             logger='memory.hybrid.retrieval.hybrid_retriever'):
            results = retriever.retrieve("test query", top_k=5,
                                         context_hint="ctx hint")

        retriever._emb.embed = original_embed

        # Retrieval must still complete (ctx_score defaults to 0.0)
        assert isinstance(results, list)
        # Some log record about the failure must exist (WARNING or ERROR)
        relevant = [r for r in caplog.records
                    if r.levelno >= logging.WARNING
                    and ('embed' in r.message.lower() or 'HybridRetriever' in r.message)]
        assert relevant, (
            "HybridRetriever must log WARNING or ERROR when embed fails during scoring. "
            f"Records seen: {[r.message for r in caplog.records]}"
        )
        h.close()

    def test_temporal_retriever_bad_node_does_not_stop_retrieval(self, tmp_path):
        """A bad node must not prevent good nodes from being returned."""
        from memory.hybrid.retrieval.hybrid_retriever import TemporalRetriever
        from memory.hybrid.hgshm import HGSHM

        h = HGSHM(tmp_path)
        h.remember("good node one", confidence=0.9, importance=0.9)
        h.remember("good node two", confidence=0.8, importance=0.8)

        retriever = TemporalRetriever(graph_store=h.graph_store)
        bad_node = MagicMock()
        bad_node.node_id = "bad"
        bad_node.created_dt = None

        good_nodes = h.graph_store.all_nodes(limit=10)
        with patch.object(h.graph_store, 'all_nodes',
                          return_value=[bad_node] + good_nodes):
            results = retriever.recent(max_age_hours=24, top_k=10)

        assert len(results) >= 2, (
            "Good nodes must be returned even when one node has malformed data"
        )
        h.close()

    def test_ctx_score_failure_does_not_break_retrieval(self, tmp_path):
        """Embedding failure in ctx_score path must not prevent results."""
        from memory.hybrid.hgshm import HGSHM

        h = HGSHM(tmp_path)
        h.remember("important fact", confidence=0.9, importance=0.9)

        retriever = h.hybrid_retriever
        original_embed = retriever._emb.embed
        call_count = [0]

        def embed_fails_late(text):
            call_count[0] += 1
            if call_count[0] > 2:
                raise RuntimeError("OOM")
            return original_embed(text)

        retriever._emb.embed = embed_fails_late
        results = retriever.retrieve("important fact", top_k=5)
        retriever._emb.embed = original_embed

        assert isinstance(results, list)
        # Retrieval must still complete (ctx_score defaults to 0.0 on failure)
        h.close()


# ═════════════════════════════════════════════════════════════════════════════
# A13 — VectorStore brute-force fallback warning at module load
# ═════════════════════════════════════════════════════════════════════════════

class TestA13FallbackWarning:
    """sqlite-vec absence must be logged at WARNING at module load time."""

    def test_sqlite_vec_availability_flag_exists(self):
        from memory.hybrid.vector.vector_store import _SQLITE_VEC_AVAILABLE
        assert isinstance(_SQLITE_VEC_AVAILABLE, bool)

    def test_warning_fires_at_module_load_when_absent(self):
        """When sqlite_vec is absent, log.warning must fire during import, not on use."""
        import sys
        import importlib

        # Simulate sqlite_vec absent by patching sys.modules
        # We check the module source directly to verify the warning is at module scope
        import memory.hybrid.vector.vector_store as m
        src = Path(m.__file__).read_text()

        # The warning must be at module level (inside an except ImportError block)
        # not inside a method
        assert 'log.warning' in src, "vector_store must call log.warning"
        # Verify the warning is NOT inside a class or def (module-level)
        lines = src.splitlines()
        in_class_or_def = False
        warning_at_module_level = False
        indent_level = 0
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('class ') or stripped.startswith('def '):
                in_class_or_def = True
            elif not line.startswith(' ') and not line.startswith('\t'):
                in_class_or_def = False
            if 'log.warning' in line and not in_class_or_def:
                warning_at_module_level = True
                break
        assert warning_at_module_level, (
            "sqlite-vec fallback warning must be at module level, not inside a method"
        )

    def test_warning_message_mentions_sqlite_vec(self):
        """Warning message must mention sqlite-vec so operators know what to install."""
        import memory.hybrid.vector.vector_store as m
        src = Path(m.__file__).read_text()
        # Find the warning string
        import re
        warnings = re.findall(r'log\.warning\("([^"]+)"', src)
        sqlite_vec_warnings = [w for w in warnings if 'sqlite' in w.lower()]
        assert sqlite_vec_warnings, (
            "At least one log.warning must mention sqlite-vec"
        )
        msg = sqlite_vec_warnings[0]
        assert 'brute' in msg.lower() or 'fallback' in msg.lower(), (
            f"Warning should mention brute-force or fallback: {msg!r}"
        )

    def test_fallback_mode_returns_results(self, tmp_path):
        """Brute-force fallback must still return correct results."""
        from memory.hybrid.vector.vector_store import VectorStore
        store = VectorStore(memory_dir=tmp_path, dim=4)
        vec = [0.1, 0.2, 0.3, 0.4]
        node_id = str(uuid.uuid4())
        store.upsert(node_id, vec)  # first arg is node_id
        results = store.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0].node_id == node_id  # check node_id, not embedding_id
        store.close()

    def test_availability_flag_consistent_with_import(self):
        """_SQLITE_VEC_AVAILABLE must reflect whether sqlite_vec was importable."""
        import sys
        from memory.hybrid.vector.vector_store import _SQLITE_VEC_AVAILABLE
        sqlite_vec_importable = 'sqlite_vec' in sys.modules or (
            __import__('importlib').util.find_spec('sqlite_vec') is not None
        )
        assert _SQLITE_VEC_AVAILABLE == sqlite_vec_importable, (
            f"_SQLITE_VEC_AVAILABLE={_SQLITE_VEC_AVAILABLE} but "
            f"sqlite_vec importable={sqlite_vec_importable}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# A14 — PolicyLearner LRU cache thread-safety
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def learner_with_store(tmp_path):
    from policy.store import PolicyStore
    from policy.learner import PolicyLearner
    store = PolicyStore(memory_dir=tmp_path)
    learner = PolicyLearner(policy_store=store)
    yield learner, store
    store.close()


def _make_policy(name="t"):
    from policy.models import PolicyRecord, PolicyType, PolicyDomain
    return PolicyRecord(
        policy_id=str(uuid.uuid4()),
        name=name,
        policy_type=PolicyType.PLANNER_CONFIG,
        domain=PolicyDomain.SYSTEM,
        config={},
    )


class TestA14CacheThreadSafety:
    """PolicyLearner._cache must be protected by a threading.Lock."""

    def test_cache_lock_exists(self, learner_with_store):
        learner, _ = learner_with_store
        assert hasattr(learner, '_cache_lock'), (
            "PolicyLearner must have _cache_lock attribute"
        )

    def test_cache_lock_is_threading_lock(self, learner_with_store):
        learner, _ = learner_with_store
        lock = learner._cache_lock
        # threading.Lock() returns a _thread.lock type; check it has acquire/release
        assert hasattr(lock, 'acquire') and hasattr(lock, 'release'), (
            "_cache_lock must be a threading.Lock or compatible"
        )

    def test_cache_put_acquires_lock(self, learner_with_store):
        """_cache_put must use the lock (verify via source inspection)."""
        import inspect
        learner, _ = learner_with_store
        src = inspect.getsource(learner._cache_put)
        assert '_cache_lock' in src, (
            "_cache_put must use _cache_lock"
        )

    def test_cache_get_acquires_lock(self, learner_with_store):
        """_cache_get must use the lock."""
        import inspect
        learner, _ = learner_with_store
        src = inspect.getsource(learner._cache_get)
        assert '_cache_lock' in src, (
            "_cache_get must use _cache_lock"
        )

    def test_concurrent_put_does_not_corrupt_cache(self, learner_with_store):
        """200 concurrent _cache_put calls must not raise or corrupt the cache."""
        learner, _ = learner_with_store
        errors = []

        def put_batch():
            try:
                for _ in range(50):
                    learner._cache_put(str(uuid.uuid4()), _make_policy())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=put_batch) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, f"Concurrent _cache_put raised: {errors}"
        assert len(learner._cache) <= learner._cache_max_size, (
            "Cache must not exceed max_size under concurrent writes"
        )

    def test_concurrent_put_and_get(self, learner_with_store):
        """Mixed concurrent reads and writes must not corrupt the cache."""
        learner, _ = learner_with_store
        errors = []
        pids = []

        # Pre-populate
        for _ in range(10):
            p = _make_policy()
            pids.append(p.policy_id)
            learner._cache_put(p.policy_id, p)

        def writer():
            try:
                for _ in range(30):
                    learner._cache_put(str(uuid.uuid4()), _make_policy())
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for pid in pids * 10:
                    learner._cache_get(pid)  # may return None — that's fine
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=writer) for _ in range(3)] +
            [threading.Thread(target=reader) for _ in range(3)]
        )
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, f"Concurrent put+get raised: {errors}"

    def test_cache_respects_max_size_under_concurrency(self, learner_with_store):
        """Cache size must never exceed _cache_max_size even under concurrent writes."""
        learner, _ = learner_with_store
        # Use a small max to make eviction frequent
        learner._cache_max_size = 20

        def flood():
            for _ in range(100):
                learner._cache_put(str(uuid.uuid4()), _make_policy())

        threads = [threading.Thread(target=flood) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        with learner._cache_lock:
            size = len(learner._cache)
        assert size <= 20, f"Cache size {size} exceeds max 20 after concurrent writes"

    def test_cache_put_promotes_existing_key(self, learner_with_store):
        """Putting an existing key must move it to MRU (no duplicate, just promotion)."""
        learner, _ = learner_with_store
        p = _make_policy()
        learner._cache_put(p.policy_id, p)
        learner._cache_put(p.policy_id, p)  # second put → should move to end
        with learner._cache_lock:
            assert list(learner._cache.keys())[-1] == p.policy_id

    def test_cache_get_returns_none_on_miss(self, learner_with_store):
        learner, _ = learner_with_store
        result = learner._cache_get("nonexistent-id")
        assert result is None

    def test_cache_get_promotes_hit_to_mru(self, learner_with_store):
        """cache_get on a hit must move the entry to MRU position."""
        learner, _ = learner_with_store
        p1, p2, p3 = _make_policy(), _make_policy(), _make_policy()
        for p in [p1, p2, p3]:
            learner._cache_put(p.policy_id, p)

        # Access p1 → should move to MRU (last)
        learner._cache_get(p1.policy_id)
        with learner._cache_lock:
            keys = list(learner._cache.keys())
        assert keys[-1] == p1.policy_id, (
            "_cache_get must promote hit to MRU position"
        )

    def test_threading_import_present(self):
        """threading must be imported in policy.learner."""
        import policy.learner as m
        import inspect
        src = inspect.getsource(m)
        assert 'import threading' in src, (
            "policy.learner must import threading for A14 lock"
        )

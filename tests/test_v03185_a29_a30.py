"""
tests/test_v03185_a29_a30.py
==============================
Regression tests for:

  A29 — PolicyLearner.register() check-then-act race condition made atomic
         by holding self._store._lock for the entire read-check-write sequence.

  A30 — ContextBuilder 11-step pipeline limits exposed via ContextBuilderSettings
         sub-schema on HGSHMSettings, replacing all hardcoded magic numbers.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════════════
# A29 — PolicyLearner.register() atomicity
# ═════════════════════════════════════════════════════════════════════════════

class TestRegisterThreadSafety:

    def _make_learner(self, tmp_path):
        from policy.store import PolicyStore
        from policy.learner import PolicyLearner
        store = PolicyStore(memory_dir=tmp_path)
        return PolicyLearner(policy_store=store), store

    def _make_policy(self, name="default"):
        from policy.models import PolicyRecord, PolicyDomain, PolicyType
        return PolicyRecord(
            name=name,
            domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.RETRIEVAL_WEIGHTS,
        )

    def test_register_returns_policy(self, tmp_path):
        learner, store = self._make_learner(tmp_path)
        p = self._make_policy()
        result = learner.register(p)
        assert result.name == "default"
        store.close()

    def test_register_idempotent_no_overwrite(self, tmp_path):
        """Registering the same name twice without overwrite returns the first."""
        learner, store = self._make_learner(tmp_path)
        p1 = self._make_policy()
        p2 = self._make_policy()
        r1 = learner.register(p1)
        r2 = learner.register(p2, overwrite=False)
        assert r1.policy_id == r2.policy_id, "Second register must return the existing policy"
        store.close()

    def test_register_overwrite_updates_config(self, tmp_path):
        from policy.models import PolicyRecord, PolicyDomain, PolicyType
        learner, store = self._make_learner(tmp_path)
        p1 = self._make_policy()
        learner.register(p1)
        p2 = PolicyRecord(
            name="default",
            domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.RETRIEVAL_WEIGHTS,
            config={"updated": True},
        )
        result = learner.register(p2, overwrite=True)
        assert result.config.get("updated") is True
        store.close()

    def test_concurrent_register_no_duplicates_20_threads(self, tmp_path):
        """20 threads simultaneously registering same policy → exactly 1 row."""
        learner, store = self._make_learner(tmp_path)
        from policy.models import PolicyDomain, PolicyType
        errors = []

        def register_once():
            try:
                learner.register(self._make_policy())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register_once) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        from policy.models import PolicyDomain, PolicyType
        all_policies = store.all_active(
            domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.RETRIEVAL_WEIGHTS,
        )
        assert not errors, f"Thread errors: {errors}"
        assert len(all_policies) == 1, (
            f"Expected exactly 1 policy after concurrent registration, "
            f"got {len(all_policies)} — race condition not fixed (A29)"
        )
        store.close()

    def test_concurrent_register_50_threads(self, tmp_path):
        """Stress test: 50 threads, same assertion."""
        learner, store = self._make_learner(tmp_path)
        errors = []

        def register_once():
            try:
                learner.register(self._make_policy())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register_once) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        from policy.models import PolicyDomain, PolicyType
        all_policies = store.all_active(
            domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.RETRIEVAL_WEIGHTS,
        )
        assert not errors, f"Thread errors: {errors}"
        assert len(all_policies) == 1, (
            f"Race still present: got {len(all_policies)} policies after 50 threads"
        )
        store.close()

    def test_concurrent_register_distinct_names_no_collision(self, tmp_path):
        """10 threads each registering a distinct policy → 10 policies, no data loss."""
        learner, store = self._make_learner(tmp_path)
        errors = []

        def register_named(i):
            try:
                learner.register(self._make_policy(name=f"policy_{i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register_named, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        from policy.models import PolicyDomain, PolicyType
        all_policies = store.all_active(
            domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.RETRIEVAL_WEIGHTS,
        )
        assert not errors
        assert len(all_policies) == 10, (
            f"Expected 10 distinct policies, got {len(all_policies)}"
        )
        store.close()

    def test_register_acquires_store_lock(self):
        """register() must enter self._store._lock (white-box)."""
        import inspect
        src = inspect.getsource(
            __import__('policy.learner', fromlist=['PolicyLearner']).PolicyLearner.register
        )
        assert 'self._store._lock' in src, (
            "register() must acquire self._store._lock for A29 atomicity"
        )

    def test_register_defaults_still_works(self, tmp_path):
        """register_defaults() must complete without error after A29 change."""
        learner, store = self._make_learner(tmp_path)
        learner.register_defaults()  # must not raise
        store.close()


# ═════════════════════════════════════════════════════════════════════════════
# A30 — ContextBuilderSettings schema + ContextBuilder wiring
# ═════════════════════════════════════════════════════════════════════════════

class TestContextBuilderSettingsSchema:

    def test_class_exists(self):
        from config.schema import ContextBuilderSettings
        assert ContextBuilderSettings is not None

    def test_default_values_match_original_hardcodes(self):
        from config.schema import ContextBuilderSettings
        cbs = ContextBuilderSettings()
        assert cbs.max_gap_nodes == 5
        assert cbs.max_neighbourhood_seeds == 3
        assert cbs.max_neighbourhood_edges == 50
        assert cbs.max_expansion_seeds == 3
        assert cbs.max_causal_seeds == 3
        assert cbs.max_causal_chains == 3
        assert cbs.max_causal_depth == 3
        assert abs(cbs.graph_score_decay - 0.7) < 1e-9

    def test_custom_values_accepted(self):
        from config.schema import ContextBuilderSettings
        cbs = ContextBuilderSettings(
            max_gap_nodes=20,
            max_neighbourhood_edges=100,
            graph_score_decay=0.5,
        )
        assert cbs.max_gap_nodes == 20
        assert cbs.max_neighbourhood_edges == 100
        assert abs(cbs.graph_score_decay - 0.5) < 1e-9

    def test_invalid_decay_rejected(self):
        from config.schema import ContextBuilderSettings
        with pytest.raises(Exception):
            ContextBuilderSettings(graph_score_decay=1.5)  # > 1.0
        with pytest.raises(Exception):
            ContextBuilderSettings(graph_score_decay=0.0)  # must be > 0

    def test_negative_limits_rejected(self):
        from config.schema import ContextBuilderSettings
        with pytest.raises(Exception):
            ContextBuilderSettings(max_gap_nodes=-1)
        with pytest.raises(Exception):
            ContextBuilderSettings(max_causal_depth=0)  # ge=1

    def test_hgshm_settings_has_context_builder_field(self):
        from config.schema import HGSHMSettings
        s = HGSHMSettings()
        assert hasattr(s, 'context_builder')

    def test_hgshm_context_builder_is_default_instance(self):
        from config.schema import HGSHMSettings, ContextBuilderSettings
        s = HGSHMSettings()
        assert isinstance(s.context_builder, ContextBuilderSettings)
        assert s.context_builder.max_gap_nodes == 5

    def test_hgshm_context_builder_overridable(self):
        from config.schema import HGSHMSettings, ContextBuilderSettings
        custom = ContextBuilderSettings(max_gap_nodes=15)
        s = HGSHMSettings(context_builder=custom)
        assert s.context_builder.max_gap_nodes == 15

    def test_round_trip_json(self):
        from config.schema import HGSHMSettings
        s = HGSHMSettings()
        data = s.model_dump()
        s2 = HGSHMSettings(**data)
        assert s2.context_builder.max_gap_nodes == s.context_builder.max_gap_nodes


class TestContextBuilderWiring:

    def _make_cb(self, settings=None):
        from memory.hybrid.context.context_builder import ContextBuilder
        return ContextBuilder(
            hybrid_retriever=MagicMock(),
            graph_store=MagicMock(),
            context_builder_settings=settings,
        )

    def test_accepts_settings_param(self):
        from config.schema import ContextBuilderSettings
        cb = self._make_cb(ContextBuilderSettings(max_gap_nodes=12))
        assert cb._cbs.max_gap_nodes == 12

    def test_defaults_applied_when_none_passed(self):
        cb = self._make_cb(None)
        assert cb._cbs is not None
        assert cb._cbs.max_gap_nodes == 5

    def test_custom_gap_nodes_respected(self):
        from config.schema import ContextBuilderSettings
        cb = self._make_cb(ContextBuilderSettings(max_gap_nodes=0))
        assert cb._cbs.max_gap_nodes == 0

    def test_custom_neighbourhood_edges_respected(self):
        from config.schema import ContextBuilderSettings
        cb = self._make_cb(ContextBuilderSettings(max_neighbourhood_edges=10))
        assert cb._cbs.max_neighbourhood_edges == 10

    def test_custom_graph_decay_respected(self):
        from config.schema import ContextBuilderSettings
        cb = self._make_cb(ContextBuilderSettings(graph_score_decay=0.9))
        assert abs(cb._cbs.graph_score_decay - 0.9) < 1e-9

    def test_no_hardcoded_magic_numbers_in_build(self):
        """Confirm the key magic numbers no longer appear literally in build()."""
        import inspect
        from memory.hybrid.context.context_builder import ContextBuilder
        src = inspect.getsource(ContextBuilder.build)
        # These specific slices must not appear literally anymore
        assert '[:5]' not in src, "gap_ids[:5] must be replaced by settings"
        assert '[:3]' not in src, "primary_ids[:3] must be replaced by settings"

    def test_no_hardcoded_magic_numbers_in_helpers(self):
        import inspect
        from memory.hybrid.context.context_builder import ContextBuilder
        expand_src = inspect.getsource(ContextBuilder._expand_supporting)
        causal_src = inspect.getsource(ContextBuilder._extract_causal_chains)
        neighbourhood_src = inspect.getsource(ContextBuilder._get_neighbourhood_edges)
        assert '[:3]' not in expand_src, "expansion seed [:3] must be replaced by settings"
        assert '[:3]' not in causal_src, "causal seed [:3] must be replaced by settings"
        assert '[:50]' not in neighbourhood_src, "edges[:50] must be replaced by settings"
        assert 'self._cbs.graph_score_decay' in expand_src, "graph decay must read from settings"

    def test_context_builder_settings_in_schema_all(self):
        """ContextBuilderSettings must be accessible from config.schema."""
        from config import schema
        assert hasattr(schema, 'ContextBuilderSettings')

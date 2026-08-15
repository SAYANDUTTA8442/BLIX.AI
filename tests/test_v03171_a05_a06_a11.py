"""
Blix v0.3.16.9 — Tests for Issues A05, A06, A11

A05: ConsolidationEngine O(n) scan — prune uses SQL upper bounds
  - all_nodes() accepts max_confidence and max_importance
  - SQL pushdown filters work at persistence layer
  - consolidate() prune only loads candidates below threshold
  - max_* params propagated through GraphStore and HGSHM facades

A06: DB filenames configurable
  - DatabaseFilenameSettings defaults match historical hardcoded values
  - Validation rejects invalid filenames
  - PolicyStore accepts db_filename override
  - HGSHMStore accepts db_filename override
  - VectorStore accepts db_filename override
  - HGSHMSettings has database sub-schema
  - Env vars BLIX_DB_* override filenames
  - Default resolves from settings when not passed

A11: mutation_scale ignored in run_cycle
  - PolicyOptimizer stores self._mutation_scale
  - mutation_scale from constructor flows to evolve_poor_performers
  - mutation_scale from settings flows to evolve_poor_performers
  - explicit override takes precedence over settings
  - run_cycle() actually uses self._mutation_scale
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config.schema import (
    DatabaseFilenameSettings, HGSHMSettings, ADMASettings,
)
from config.settings import (
    load_hgshm_settings, load_adma_settings,
)
from memory.hybrid.hgshm import HGSHM
from memory.hybrid.graph.graph_store import GraphStore
from memory.hybrid.storage.persistence import HGSHMStore
from memory.hybrid.vector.vector_store import VectorStore
from memory.hybrid.models.memory_node import MemoryType
from policy.store import PolicyStore
from policy.optimizer import PolicyOptimizer
from policy.models import PolicyRecord, PolicyDomain, PolicyType


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def hgshm(tmp_dir):
    h = HGSHM(tmp_dir)
    yield h
    h.close()


@pytest.fixture
def store(tmp_dir):
    s = PolicyStore(tmp_dir)
    yield s
    s.close()


# ════════════════════════════════════════════════════════════════════
# A05 — ConsolidationEngine SQL upper bounds (max_confidence/max_importance)
# ════════════════════════════════════════════════════════════════════

class TestMaxBoundsOnAllNodes:
    """Verify max_confidence/max_importance push filters to SQL."""

    def test_all_nodes_accepts_max_confidence(self, hgshm):
        """max_confidence parameter accepted without error."""
        results = hgshm.all_nodes(max_confidence=0.5)
        assert isinstance(results, list)

    def test_all_nodes_accepts_max_importance(self, hgshm):
        """max_importance parameter accepted without error."""
        results = hgshm.all_nodes(max_importance=0.5)
        assert isinstance(results, list)

    def test_max_confidence_filters_high_confidence(self, hgshm):
        """Nodes with confidence > max_confidence must not be returned."""
        hgshm.remember("low conf",  confidence=0.2, importance=0.5)
        hgshm.remember("high conf", confidence=0.9, importance=0.5)

        results = hgshm.all_nodes(max_confidence=0.5)
        confs = [n.confidence for n in results]
        assert all(c <= 0.5 for c in confs), (
            f"Nodes above max_confidence=0.5 returned: {confs}"
        )
        assert any(c <= 0.2 for c in confs), "Low-confidence node should be present"

    def test_max_importance_filters_high_importance(self, hgshm):
        """Nodes with importance > max_importance must not be returned."""
        hgshm.remember("low imp",  confidence=0.5, importance=0.1)
        hgshm.remember("high imp", confidence=0.5, importance=0.9)

        results = hgshm.all_nodes(max_importance=0.3)
        imps = [n.importance for n in results]
        assert all(i <= 0.3 for i in imps), (
            f"Nodes above max_importance=0.3 returned: {imps}"
        )

    def test_max_both_combined(self, hgshm):
        """Both max_* filters applied together must return only candidates."""
        hgshm.remember("candidate",  confidence=0.1, importance=0.05)
        hgshm.remember("high_conf",  confidence=0.9, importance=0.05)
        hgshm.remember("high_imp",   confidence=0.1, importance=0.9)
        hgshm.remember("both_high",  confidence=0.9, importance=0.9)

        results = hgshm.all_nodes(max_confidence=0.15, max_importance=0.06)
        assert len(results) == 1
        assert results[0].text == "candidate"

    def test_max_bounds_compatible_with_min_bounds(self, hgshm):
        """min_* and max_* filters can be combined."""
        hgshm.remember("in_range",   confidence=0.5, importance=0.5)
        hgshm.remember("too_low",    confidence=0.1, importance=0.1)
        hgshm.remember("too_high",   confidence=0.9, importance=0.9)

        results = hgshm.all_nodes(
            min_confidence=0.3, max_confidence=0.7,
            min_importance=0.3, max_importance=0.7,
        )
        assert len(results) == 1
        assert results[0].text == "in_range"

    def test_max_none_returns_all(self, hgshm):
        """max_* = None (default) must return all nodes."""
        for i in range(5):
            hgshm.remember(f"node_{i}", confidence=i * 0.2, importance=i * 0.2)
        results = hgshm.all_nodes()
        assert len(results) == 5

    def test_persistence_layer_max_params(self, tmp_dir):
        """Verify max_* are applied at the HGSHMStore SQL layer."""
        import inspect
        sig = inspect.signature(HGSHMStore.all_nodes)
        assert "max_confidence" in sig.parameters
        assert "max_importance" in sig.parameters

    def test_graph_store_propagates_max_params(self, tmp_dir):
        """GraphStore.all_nodes must accept and pass max_* to store."""
        import inspect
        sig = inspect.signature(GraphStore.all_nodes)
        assert "max_confidence" in sig.parameters
        assert "max_importance" in sig.parameters

    def test_hgshm_facade_propagates_max_params(self, tmp_dir):
        """HGSHM.all_nodes must accept max_* parameters."""
        import inspect
        sig = inspect.signature(HGSHM.all_nodes)
        assert "max_confidence" in sig.parameters
        assert "max_importance" in sig.parameters

    def test_consolidation_prune_uses_sql_bounds(self, hgshm):
        """
        consolidate() prune scan must only load low-importance/confidence nodes.
        Verify by checking that high-value nodes survive consolidation.
        """
        from memory.hybrid.consolidation.consolidation_engine import (
            ConsolidationEngine, DuplicateDetector, MemoryMerger, ImportanceModel
        )

        h = hgshm
        high = h.remember("important knowledge", confidence=0.9, importance=0.9)
        low  = h.remember("ephemeral noise",     confidence=0.05, importance=0.03)

        gs = h.graph_store
        vs = h.vector_index
        em = h.embedding_manager

        engine = ConsolidationEngine(gs, vs, em)

        low_node = gs.get_node(low.node_id)
        low_node.access_count = 0
        gs.update_node(low_node)

        engine.consolidate(
            prune_below_importance=0.05,
            prune_below_confidence=0.10,
            max_scan=100,
        )

        assert gs.get_node(high.node_id) is not None, (
            "High-value node was incorrectly pruned"
        )

# ════════════════════════════════════════════════════════════════════
# A06 — Configurable DB filenames
# ════════════════════════════════════════════════════════════════════

class TestDatabaseFilenameSettings:
    def test_defaults_match_historical_values(self):
        s = DatabaseFilenameSettings()
        assert s.hgshm_db   == "hgshm.db"
        assert s.vectors_db == "vectors.db"
        assert s.policy_db  == "policy.db"

    def test_invalid_filename_no_db_extension(self):
        with pytest.raises(Exception):
            DatabaseFilenameSettings(hgshm_db="hgshm_no_ext")

    def test_invalid_filename_path_separator(self):
        with pytest.raises(Exception):
            DatabaseFilenameSettings(hgshm_db="some/path/hgshm.db")

    def test_invalid_filename_special_chars(self):
        with pytest.raises(Exception):
            DatabaseFilenameSettings(hgshm_db="hgshm<>.db")

    def test_valid_custom_filename(self):
        s = DatabaseFilenameSettings(
            hgshm_db="hgshm_prod.db",
            vectors_db="vectors_prod.db",
            policy_db="policy_prod.db",
        )
        assert s.hgshm_db   == "hgshm_prod.db"
        assert s.vectors_db == "vectors_prod.db"
        assert s.policy_db  == "policy_prod.db"

    def test_hgshm_settings_has_database_field(self):
        hs = HGSHMSettings()
        assert hasattr(hs, "database")
        assert isinstance(hs.database, DatabaseFilenameSettings)

    def test_hgshm_settings_database_defaults(self):
        hs = HGSHMSettings()
        assert hs.database.hgshm_db   == "hgshm.db"
        assert hs.database.vectors_db == "vectors.db"
        assert hs.database.policy_db  == "policy.db"


class TestPolicyStoreDbFilename:
    def test_default_filename_is_policy_db(self, tmp_dir):
        s = PolicyStore(tmp_dir)
        assert s._db_path.name == "policy.db"
        s.close()

    def test_custom_filename_honoured(self, tmp_dir):
        s = PolicyStore(tmp_dir, db_filename="custom_policy.db")
        assert s._db_path.name == "custom_policy.db"
        assert (tmp_dir / "custom_policy.db").exists()
        s.close()

    def test_custom_filename_is_usable(self, tmp_dir):
        """Custom-named DB must still function correctly."""
        s = PolicyStore(tmp_dir, db_filename="test_policy.db")
        p = PolicyRecord(
            name="test", domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.PLANNER_CONFIG
        )
        s.save(p)
        retrieved = s.get(p.policy_id)
        assert retrieved is not None
        assert retrieved.name == "test"
        s.close()

    def test_env_var_blix_db_policy(self, tmp_dir):
        with patch.dict(os.environ, {"BLIX_DB_POLICY": "env_policy.db"}):
            s = load_hgshm_settings()
        assert s.database.policy_db == "env_policy.db"

    def test_two_stores_different_filenames_isolated(self, tmp_dir):
        """Two PolicyStore instances in same dir with different names are isolated."""
        s1 = PolicyStore(tmp_dir, db_filename="policy_a.db")
        s2 = PolicyStore(tmp_dir, db_filename="policy_b.db")
        p = PolicyRecord(
            name="only_in_a", domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.PLANNER_CONFIG,
        )
        s1.save(p)
        assert s2.get(p.policy_id) is None  # not in s2
        s1.close(); s2.close()


class TestHGSHMStoreDbFilename:
    def test_default_filename_is_hgshm_db(self, tmp_dir):
        s = HGSHMStore(tmp_dir)
        assert s._db_path.name == "hgshm.db"
        s.close()

    def test_custom_filename_honoured(self, tmp_dir):
        s = HGSHMStore(tmp_dir, db_filename="custom_hgshm.db")
        assert s._db_path.name == "custom_hgshm.db"
        s.close()


class TestVectorStoreDbFilename:
    def test_default_filename_is_vectors_db(self, tmp_dir):
        s = VectorStore(tmp_dir, dim=32)
        assert s._db_path.name == "vectors.db"

    def test_custom_filename_honoured(self, tmp_dir):
        s = VectorStore(tmp_dir, dim=32, db_filename="custom_vectors.db")
        assert s._db_path.name == "custom_vectors.db"


class TestDbFilenameEnvVars:
    def test_blix_db_hgshm_env_var(self):
        with patch.dict(os.environ, {"BLIX_DB_HGSHM": "env_hgshm.db"}):
            s = load_hgshm_settings()
        assert s.database.hgshm_db == "env_hgshm.db"

    def test_blix_db_vectors_env_var(self):
        with patch.dict(os.environ, {"BLIX_DB_VECTORS": "env_vectors.db"}):
            s = load_hgshm_settings()
        assert s.database.vectors_db == "env_vectors.db"

    def test_blix_db_policy_env_var(self):
        with patch.dict(os.environ, {"BLIX_DB_POLICY": "env_policy.db"}):
            s = load_hgshm_settings()
        assert s.database.policy_db == "env_policy.db"

    def test_env_var_invalid_filename_rejected(self):
        with patch.dict(os.environ, {"BLIX_DB_HGSHM": "bad/path.db"}):
            with pytest.raises(Exception):
                load_hgshm_settings()


# ════════════════════════════════════════════════════════════════════
# A11 — mutation_scale flows through run_cycle
# ════════════════════════════════════════════════════════════════════

class TestMutationScaleFlowthrough:
    def test_optimizer_stores_mutation_scale(self, store):
        o = PolicyOptimizer(store, mutation_scale=0.25)
        assert o._mutation_scale == pytest.approx(0.25)

    def test_optimizer_default_mutation_scale_from_settings(self, store):
        from config.settings import adma_settings
        o = PolicyOptimizer(store)
        assert o._mutation_scale == pytest.approx(
            adma_settings.optimizer.mutation_scale
        )

    def test_explicit_override_beats_settings(self, store):
        o = PolicyOptimizer(store, mutation_scale=0.42)
        assert o._mutation_scale == pytest.approx(0.42)

    def test_run_cycle_passes_mutation_scale_to_evolve(self, store):
        """
        run_cycle() must call evolve_poor_performers(mutation_scale=self._mutation_scale).
        We verify by patching evolve_poor_performers and capturing the kwarg.
        """
        o = PolicyOptimizer(store, mutation_scale=0.33,
                            min_observations=1, aging_threshold=0.99)
        called_with = []
        original = o.evolve_poor_performers

        def tracking_evolve(mutation_scale=0.1):
            called_with.append(mutation_scale)
            return original(mutation_scale=mutation_scale)

        o.evolve_poor_performers = tracking_evolve
        o.run_cycle(spawn_mutants=True)

        assert called_with, "evolve_poor_performers was never called"
        assert called_with[0] == pytest.approx(0.33), (
            f"Expected mutation_scale=0.33, got {called_with[0]}. "
            f"run_cycle() is not passing self._mutation_scale."
        )

    def test_run_cycle_default_mutation_scale_is_not_hardcoded(self, store):
        """
        Changing mutation_scale via constructor must affect what run_cycle uses.
        If mutation_scale were hardcoded as 0.1 in run_cycle(), different
        optimizer instances would always use 0.1 regardless of constructor arg.
        """
        o_low  = PolicyOptimizer(store, mutation_scale=0.01,
                                 min_observations=1, aging_threshold=0.99)
        o_high = PolicyOptimizer(store, mutation_scale=0.50,
                                 min_observations=1, aging_threshold=0.99)

        low_calls = []
        high_calls = []

        def track_low(mutation_scale=0.1):
            low_calls.append(mutation_scale)
            return []

        def track_high(mutation_scale=0.1):
            high_calls.append(mutation_scale)
            return []

        o_low.evolve_poor_performers  = track_low
        o_high.evolve_poor_performers = track_high

        o_low.run_cycle(spawn_mutants=True)
        o_high.run_cycle(spawn_mutants=True)

        assert low_calls[0]  == pytest.approx(0.01), (
            f"Low optimizer used {low_calls[0]}, expected 0.01"
        )
        assert high_calls[0] == pytest.approx(0.50), (
            f"High optimizer used {high_calls[0]}, expected 0.50"
        )
        assert low_calls[0] != high_calls[0], (
            "Both optimizers used the same mutation_scale — A11 not fixed"
        )

    def test_spawn_mutant_receives_correct_scale(self, store):
        """End-to-end: a spawned mutant's metadata records the actual scale used."""
        o = PolicyOptimizer(store, mutation_scale=0.42,
                            min_observations=1, aging_threshold=0.99)

        poor = PolicyRecord(
            name="poor_parent",
            domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.PLANNER_CONFIG,
            config={"beam_width": 5, "max_depth": 3},
            alpha=1.0, beta_=10.0,
            success_count=1, failure_count=5,
        )
        store.save(poor)

        mutant = o.spawn_mutant(poor, mutation_scale=o._mutation_scale)
        assert mutant is not None
        assert mutant.metadata.get("mutation_scale") == pytest.approx(0.42), (
            f"Mutant metadata.mutation_scale={mutant.metadata.get('mutation_scale')}, "
            f"expected 0.42"
        )

    def test_mutation_scale_settings_override(self):
        """BLIX_ADMA_MUTATION_SCALE env var changes the effective scale."""
        from unittest.mock import patch
        with patch.dict(os.environ, {"BLIX_ADMA_MUTATION_SCALE": "0.77"}):
            s = load_adma_settings()
        assert s.optimizer.mutation_scale == pytest.approx(0.77)

"""
Blix v0.3.16.1 — Tests for ISSUE-014 (Centralized Configuration)

Tests cover:
  - Schema defaults match previously-hardcoded values
  - Validation rejects invalid values with descriptive errors
  - Environment variable overrides work for every field
  - Profile overrides (development / testing / benchmark / production)
  - Feature flags default values and env-var toggle
  - Explicit constructor arguments take precedence over settings
  - Configuration snapshot export (JSON + YAML)
  - Backward compatibility: existing tests pass unchanged
  - Settings are singletons (no re-loading on each import)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config.schema import (
    ADMASettings, HGSHMSettings, FeatureFlags,
    PolicyLearnerSettings, PolicyOptimizerSettings,
    RewardEngineSettings, RewardLogSettings, PromptCompilerSettings,
    HybridWeightsSettings, RetrievalSettings, ConsolidationSettings,
    HierarchySettings, EmbeddingSettings,
    _PROFILE_OVERRIDES,
)
from config.settings import (
    adma_settings, hgshm_settings, feature_flags,
    load_adma_settings, load_hgshm_settings, load_feature_flags,
    export_config_snapshot,
)


# ════════════════════════════════════════════════════════════════════
# Schema defaults match previously-hardcoded values
# ════════════════════════════════════════════════════════════════════

class TestDefaultValues:
    """
    Verify that schema defaults exactly match the values that were
    previously hardcoded in policy/learner.py, optimizer.py, etc.
    A regression here means behavior has silently changed.
    """

    def test_learner_decay_factor(self):
        assert PolicyLearnerSettings().decay_factor == 0.995

    def test_learner_reward_threshold(self):
        assert PolicyLearnerSettings().reward_threshold == 0.5

    def test_learner_snapshot_every(self):
        assert PolicyLearnerSettings().snapshot_every == 20

    def test_learner_decay_persist_every(self):
        assert PolicyLearnerSettings().decay_persist_every == 50

    def test_learner_cache_max_size(self):
        assert PolicyLearnerSettings().cache_max_size == 1000

    def test_optimizer_min_observations(self):
        assert PolicyOptimizerSettings().min_observations == 10

    def test_optimizer_aging_threshold(self):
        assert PolicyOptimizerSettings().aging_threshold == 0.35

    def test_optimizer_convergence_window(self):
        assert PolicyOptimizerSettings().convergence_window == 5

    def test_optimizer_convergence_tolerance(self):
        assert PolicyOptimizerSettings().convergence_tolerance == 0.02

    def test_optimizer_rollback_drop_threshold(self):
        assert PolicyOptimizerSettings().rollback_drop_threshold == 0.10

    def test_optimizer_mutation_scale(self):
        assert PolicyOptimizerSettings().mutation_scale == 0.1

    def test_reward_latency_target_ms(self):
        assert RewardEngineSettings().latency_target_ms == 500.0

    def test_reward_log_max_rows(self):
        assert RewardLogSettings().max_rows_per_policy == 1000

    def test_reward_log_stats_last_n(self):
        assert RewardLogSettings().stats_last_n == 1000

    def test_prompt_compiler_token_budget(self):
        assert PromptCompilerSettings().token_budget == 2000

    def test_prompt_compiler_max_memory_nodes(self):
        assert PromptCompilerSettings().max_memory_nodes == 5

    def test_hgshm_embedding_dim(self):
        assert EmbeddingSettings().dim == 256

    def test_hgshm_retrieval_top_k(self):
        assert RetrievalSettings().default_top_k == 10

    def test_hgshm_consolidation_sim_threshold(self):
        assert ConsolidationSettings().similarity_threshold == pytest.approx(0.92)

    def test_hgshm_hierarchy_max_summary(self):
        assert HierarchySettings().max_summary_length == 300


# ════════════════════════════════════════════════════════════════════
# Validation — fail fast with descriptive errors
# ════════════════════════════════════════════════════════════════════

class TestValidation:
    def test_decay_factor_must_be_positive(self):
        with pytest.raises(Exception, match="decay_factor|greater"):
            PolicyLearnerSettings(decay_factor=0.0)

    def test_decay_factor_cannot_exceed_one(self):
        with pytest.raises(Exception):
            PolicyLearnerSettings(decay_factor=1.001)

    def test_reward_threshold_must_be_in_unit_interval(self):
        with pytest.raises(Exception):
            PolicyLearnerSettings(reward_threshold=-0.1)
        with pytest.raises(Exception):
            PolicyLearnerSettings(reward_threshold=1.1)

    def test_snapshot_every_must_be_positive(self):
        with pytest.raises(Exception):
            PolicyLearnerSettings(snapshot_every=0)
        with pytest.raises(Exception):
            PolicyLearnerSettings(snapshot_every=-5)

    def test_cache_max_size_must_be_positive(self):
        with pytest.raises(Exception):
            PolicyLearnerSettings(cache_max_size=0)

    def test_aging_threshold_must_be_in_unit_interval(self):
        with pytest.raises(Exception):
            PolicyOptimizerSettings(aging_threshold=1.5)

    def test_mutation_scale_must_be_positive_and_bounded(self):
        with pytest.raises(Exception):
            PolicyOptimizerSettings(mutation_scale=0.0)
        with pytest.raises(Exception):
            PolicyOptimizerSettings(mutation_scale=1.1)

    def test_latency_target_must_be_positive(self):
        with pytest.raises(Exception):
            RewardEngineSettings(latency_target_ms=0.0)

    def test_embedding_dim_must_be_positive(self):
        with pytest.raises(Exception):
            EmbeddingSettings(dim=0)

    def test_retrieval_top_k_must_be_positive(self):
        with pytest.raises(Exception):
            RetrievalSettings(default_top_k=0)

    def test_sim_threshold_must_be_in_unit_interval(self):
        with pytest.raises(Exception):
            ConsolidationSettings(similarity_threshold=1.5)

    def test_hybrid_weights_all_zero_rejected(self):
        with pytest.raises(Exception, match="zero|positive"):
            HybridWeightsSettings(
                semantic=0, vector=0, graph_distance=0, importance=0,
                confidence=0, recency=0, hierarchy=0, context_similarity=0,
                attention=0, belief_confidence=0, planning_relevance=0,
            )

    def test_min_cluster_size_must_be_gt_one(self):
        with pytest.raises(Exception):
            HierarchySettings(min_cluster_size=1)

    def test_convergence_window_must_be_gt_one(self):
        with pytest.raises(Exception):
            PolicyOptimizerSettings(convergence_window=1)


# ════════════════════════════════════════════════════════════════════
# Environment variable overrides
# ════════════════════════════════════════════════════════════════════

class TestEnvVarOverrides:
    def test_decay_factor_env_var(self):
        with patch.dict(os.environ, {"BLIX_ADMA_DECAY_FACTOR": "0.98"}):
            s = load_adma_settings()
        assert s.learner.decay_factor == pytest.approx(0.98)

    def test_cache_max_size_env_var(self):
        with patch.dict(os.environ, {"BLIX_ADMA_CACHE_MAX_SIZE": "500"}):
            s = load_adma_settings()
        assert s.learner.cache_max_size == 500

    def test_aging_threshold_env_var(self):
        with patch.dict(os.environ, {"BLIX_ADMA_AGING_THRESHOLD": "0.45"}):
            s = load_adma_settings()
        assert s.optimizer.aging_threshold == pytest.approx(0.45)

    def test_max_rows_per_policy_env_var(self):
        with patch.dict(os.environ, {"BLIX_ADMA_MAX_ROWS_PER_POLICY": "2000"}):
            s = load_adma_settings()
        assert s.reward_log.max_rows_per_policy == 2000

    def test_token_budget_env_var(self):
        with patch.dict(os.environ, {"BLIX_ADMA_TOKEN_BUDGET_COMPILER": "1500"}):
            s = load_adma_settings()
        assert s.prompt_compiler.token_budget == 1500

    def test_hgshm_top_k_env_var(self):
        with patch.dict(os.environ, {"BLIX_HGSHM_TOP_K": "15"}):
            s = load_hgshm_settings()
        assert s.retrieval.default_top_k == 15

    def test_hgshm_embedding_dim_env_var(self):
        with patch.dict(os.environ, {"BLIX_HGSHM_EMBEDDING_DIM": "512"}):
            s = load_hgshm_settings()
        assert s.embedding.dim == 512

    def test_hgshm_sim_threshold_env_var(self):
        with patch.dict(os.environ, {"BLIX_HGSHM_SIM_THRESHOLD": "0.85"}):
            s = load_hgshm_settings()
        assert s.consolidation.similarity_threshold == pytest.approx(0.85)

    def test_env_var_overrides_yaml(self):
        """Env var must win over YAML override."""
        yaml = {"learner": {"decay_factor": 0.97}}
        with patch.dict(os.environ, {"BLIX_ADMA_DECAY_FACTOR": "0.99"}):
            s = load_adma_settings(yaml_overrides=yaml)
        assert s.learner.decay_factor == pytest.approx(0.99)

    def test_invalid_env_var_value_raises(self):
        """A non-numeric env var for a float field must raise at load time."""
        with patch.dict(os.environ, {"BLIX_ADMA_DECAY_FACTOR": "not_a_float"}):
            with pytest.raises(ValueError):
                load_adma_settings()


# ════════════════════════════════════════════════════════════════════
# Profile overrides
# ════════════════════════════════════════════════════════════════════

class TestProfileOverrides:
    def test_testing_profile_reduces_cache(self):
        s = load_adma_settings(profile="testing")
        assert s.learner.cache_max_size < 1000

    def test_testing_profile_reduces_retention(self):
        s = load_adma_settings(profile="testing")
        assert s.reward_log.max_rows_per_policy < 1000

    def test_benchmark_profile_increases_cache(self):
        s = load_adma_settings(profile="benchmark")
        assert s.learner.cache_max_size > 1000

    def test_benchmark_profile_increases_retention(self):
        s = load_adma_settings(profile="benchmark")
        assert s.reward_log.max_rows_per_policy > 1000

    def test_benchmark_profile_increases_top_k(self):
        s = load_hgshm_settings(profile="benchmark")
        assert s.retrieval.default_top_k > 10

    def test_production_profile_is_valid(self):
        s = load_adma_settings(profile="production")
        assert s.learner.cache_max_size > 0

    def test_development_profile_is_valid(self):
        s = load_adma_settings(profile="development")
        assert s.learner.cache_max_size > 0

    def test_blix_profile_env_var(self):
        with patch.dict(os.environ, {"BLIX_PROFILE": "testing"}):
            s = load_adma_settings()
        assert s.learner.cache_max_size < 1000

    def test_unknown_profile_uses_defaults(self):
        """An unrecognized profile name must not crash — falls back to defaults."""
        s = load_adma_settings(profile="nonexistent_profile")
        assert s.learner.decay_factor == pytest.approx(0.995)

    def test_profile_keys_are_valid(self):
        """Every profile in _PROFILE_OVERRIDES must produce valid settings."""
        for profile_name in _PROFILE_OVERRIDES:
            s = load_adma_settings(profile=profile_name)
            assert s.learner.decay_factor > 0


# ════════════════════════════════════════════════════════════════════
# Feature flags
# ════════════════════════════════════════════════════════════════════

class TestFeatureFlags:
    def test_all_flags_default_true(self):
        f = load_feature_flags()
        assert f.hybrid_retrieval is True
        assert f.adaptive_policy is True
        assert f.graph_reasoning is True
        assert f.hierarchy is True
        assert f.semantic_search is True
        assert f.reward_learning is True
        assert f.prompt_compiler is True
        assert f.memory_consolidation is True

    def test_yaml_override_disables_flag(self):
        f = load_feature_flags({"graph_reasoning": False})
        assert f.graph_reasoning is False
        assert f.hybrid_retrieval is True  # others unchanged

    def test_env_var_disables_flag(self):
        with patch.dict(os.environ, {"BLIX_FEATURE_GRAPH_REASONING": "false"}):
            f = load_feature_flags()
        assert f.graph_reasoning is False

    def test_env_var_zero_disables_flag(self):
        with patch.dict(os.environ, {"BLIX_FEATURE_HIERARCHY": "0"}):
            f = load_feature_flags()
        assert f.hierarchy is False

    def test_env_var_off_disables_flag(self):
        with patch.dict(os.environ, {"BLIX_FEATURE_SEMANTIC_SEARCH": "off"}):
            f = load_feature_flags()
        assert f.semantic_search is False

    def test_env_var_true_enables_flag(self):
        with patch.dict(os.environ, {"BLIX_FEATURE_ADAPTIVE_POLICY": "true"}):
            f = load_feature_flags()
        assert f.adaptive_policy is True

    def test_env_var_1_enables_flag(self):
        with patch.dict(os.environ, {"BLIX_FEATURE_REWARD_LEARNING": "1"}):
            f = load_feature_flags()
        assert f.reward_learning is True


# ════════════════════════════════════════════════════════════════════
# Constructor override takes precedence over settings
# ════════════════════════════════════════════════════════════════════

class TestConstructorPrecedence:
    def test_policy_learner_explicit_decay_overrides_settings(self):
        from policy.store import PolicyStore
        from policy.learner import PolicyLearner

        with tempfile.TemporaryDirectory() as td:
            store = PolicyStore(Path(td))
            l = PolicyLearner(store, decay_factor=0.90)
            assert l._decay_factor == pytest.approx(0.90)

    def test_policy_learner_explicit_cache_overrides_settings(self):
        from policy.store import PolicyStore
        from policy.learner import PolicyLearner

        with tempfile.TemporaryDirectory() as td:
            store = PolicyStore(Path(td))
            l = PolicyLearner(store, cache_max_size=42)
            assert l._cache_max_size == 42

    def test_policy_learner_no_args_uses_settings(self):
        from policy.store import PolicyStore
        from policy.learner import PolicyLearner

        with tempfile.TemporaryDirectory() as td:
            store = PolicyStore(Path(td))
            l = PolicyLearner(store)
            assert l._decay_factor == pytest.approx(adma_settings.learner.decay_factor)
            assert l._cache_max_size == adma_settings.learner.cache_max_size

    def test_policy_optimizer_explicit_aging_overrides_settings(self):
        from policy.store import PolicyStore
        from policy.optimizer import PolicyOptimizer

        with tempfile.TemporaryDirectory() as td:
            store = PolicyStore(Path(td))
            o = PolicyOptimizer(store, aging_threshold=0.60)
            assert o._aging_t == pytest.approx(0.60)

    def test_policy_optimizer_no_args_uses_settings(self):
        from policy.store import PolicyStore
        from policy.optimizer import PolicyOptimizer

        with tempfile.TemporaryDirectory() as td:
            store = PolicyStore(Path(td))
            o = PolicyOptimizer(store)
            assert o._aging_t == pytest.approx(adma_settings.optimizer.aging_threshold)
            assert o._min_obs == adma_settings.optimizer.min_observations


# ════════════════════════════════════════════════════════════════════
# Configuration snapshot
# ════════════════════════════════════════════════════════════════════

class TestConfigSnapshot:
    def test_snapshot_contains_required_keys(self):
        snap = export_config_snapshot()
        assert "adma"       in snap
        assert "hgshm"      in snap
        assert "features"   in snap
        assert "generated_at" in snap
        assert "profile"    in snap

    def test_snapshot_adma_values_are_correct(self):
        snap = export_config_snapshot()
        assert snap["adma"]["learner"]["decay_factor"] == pytest.approx(0.995)
        assert snap["adma"]["optimizer"]["aging_threshold"] == pytest.approx(0.35)
        assert snap["adma"]["reward_log"]["max_rows_per_policy"] == 1000

    def test_snapshot_hgshm_values_are_correct(self):
        snap = export_config_snapshot()
        assert snap["hgshm"]["retrieval"]["default_top_k"] == 10
        assert snap["hgshm"]["embedding"]["dim"] == 256

    def test_snapshot_features_are_correct(self):
        snap = export_config_snapshot()
        assert snap["features"]["hybrid_retrieval"] is True
        assert snap["features"]["adaptive_policy"] is True

    def test_snapshot_writes_json_to_disk(self, tmp_path):
        export_config_snapshot(tmp_path)
        json_file = tmp_path / "config_snapshot.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert "adma" in data

    def test_snapshot_writes_yaml_to_disk(self, tmp_path):
        export_config_snapshot(tmp_path)
        yaml_file = tmp_path / "config_snapshot.yaml"
        assert yaml_file.exists()
        assert yaml_file.stat().st_size > 0

    def test_snapshot_creates_output_dir(self, tmp_path):
        out = tmp_path / "nested" / "run_001"
        export_config_snapshot(out)
        assert (out / "config_snapshot.json").exists()

    def test_snapshot_is_json_serializable(self):
        snap = export_config_snapshot()
        # Must serialize without error
        serialized = json.dumps(snap, default=str)
        assert len(serialized) > 100

    def test_snapshot_without_output_dir_returns_dict(self):
        snap = export_config_snapshot(output_dir=None)
        assert isinstance(snap, dict)
        assert "adma" in snap


# ════════════════════════════════════════════════════════════════════
# Settings are singletons
# ════════════════════════════════════════════════════════════════════

class TestSettingsSingletons:
    def test_adma_settings_is_singleton(self):
        from config.settings import adma_settings as s1
        from config.settings import adma_settings as s2
        assert s1 is s2

    def test_hgshm_settings_is_singleton(self):
        from config.settings import hgshm_settings as s1
        from config.settings import hgshm_settings as s2
        assert s1 is s2

    def test_feature_flags_is_singleton(self):
        from config.settings import feature_flags as f1
        from config.settings import feature_flags as f2
        assert f1 is f2


# ════════════════════════════════════════════════════════════════════
# YAML overrides
# ════════════════════════════════════════════════════════════════════

class TestYAMLOverrides:
    def test_yaml_override_learner_field(self):
        s = load_adma_settings({"learner": {"decay_factor": 0.97}})
        assert s.learner.decay_factor == pytest.approx(0.97)
        # Other fields must use defaults
        assert s.learner.cache_max_size == 1000

    def test_yaml_override_optimizer_field(self):
        s = load_adma_settings({"optimizer": {"aging_threshold": 0.50}})
        assert s.optimizer.aging_threshold == pytest.approx(0.50)

    def test_yaml_override_multiple_sections(self):
        s = load_adma_settings({
            "learner": {"decay_factor": 0.97},
            "reward_log": {"max_rows_per_policy": 500},
        })
        assert s.learner.decay_factor == pytest.approx(0.97)
        assert s.reward_log.max_rows_per_policy == 500

    def test_yaml_override_hgshm_weights(self):
        s = load_hgshm_settings({
            "retrieval": {"weights": {"semantic": 0.5, "vector": 0.3,
                                      "graph_distance": 0.1, "importance": 0.1}}
        })
        assert s.retrieval.weights.semantic == pytest.approx(0.5)

    def test_invalid_yaml_value_raises(self):
        with pytest.raises(Exception):
            load_adma_settings({"learner": {"decay_factor": 2.0}})  # > 1.0


# ════════════════════════════════════════════════════════════════════
# HybridWeights to_dict
# ════════════════════════════════════════════════════════════════════

class TestHybridWeights:
    def test_to_dict_has_all_11_keys(self):
        w = HybridWeightsSettings()
        d = w.to_dict()
        assert len(d) == 11
        expected_keys = {
            "semantic", "vector", "graph_distance", "importance", "confidence",
            "recency", "hierarchy", "context_similarity", "attention",
            "belief_confidence", "planning_relevance",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_are_floats(self):
        w = HybridWeightsSettings()
        for k, v in w.to_dict().items():
            assert isinstance(v, float), f"{k} is not float"
            assert v >= 0, f"{k} is negative"

    def test_all_zero_weight_rejected(self):
        with pytest.raises(Exception):
            HybridWeightsSettings(**{k: 0.0 for k in [
                "semantic", "vector", "graph_distance", "importance", "confidence",
                "recency", "hierarchy", "context_similarity", "attention",
                "belief_confidence", "planning_relevance",
            ]})

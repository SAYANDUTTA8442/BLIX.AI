"""
tests/test_v03174_a19_module_level_settings.py
================================================
Regression tests for A19: lazy settings imports replaced with module-level
imports in all five policy modules.

Covers:
  - Each of the 5 modules exposes _adma_settings at module level
  - store.py additionally exposes _hgshm_settings at module level
  - _default_*_cfg() helpers return correct sub-configs
  - When _adma_settings is None (simulated absent config), helpers return None
  - Inline lazy imports in store.py (db_filename, log_reward, reward_stats)
    use the module-level singletons instead of per-call imports
  - Full instantiation still works with module-level settings
"""

from __future__ import annotations

import tempfile
import types
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ═════════════════════════════════════════════════════════════════════════════
# A19 — module-level _adma_settings present in all 5 modules
# ═════════════════════════════════════════════════════════════════════════════

class TestModuleLevelSettingsPresent:
    """Each policy module must expose _adma_settings at module scope."""

    def test_learner_has_module_level_adma_settings(self):
        import policy.learner as m
        assert hasattr(m, "_adma_settings"), (
            "policy.learner must have module-level _adma_settings"
        )

    def test_optimizer_has_module_level_adma_settings(self):
        import policy.optimizer as m
        assert hasattr(m, "_adma_settings")

    def test_reward_has_module_level_adma_settings(self):
        import policy.reward as m
        assert hasattr(m, "_adma_settings")

    def test_compiler_has_module_level_adma_settings(self):
        import policy.compiler as m
        assert hasattr(m, "_adma_settings")

    def test_store_has_module_level_adma_settings(self):
        import policy.store as m
        assert hasattr(m, "_adma_settings")

    def test_store_has_module_level_hgshm_settings(self):
        import policy.store as m
        assert hasattr(m, "_hgshm_settings"), (
            "policy.store must have module-level _hgshm_settings"
        )

    def test_adma_settings_is_not_none_when_config_present(self):
        """With config/ installed, _adma_settings must be the live singleton."""
        import policy.learner as m
        # config/ is present in the test env — should be non-None
        assert m._adma_settings is not None, (
            "_adma_settings must be populated when config.settings is importable"
        )

    def test_hgshm_settings_is_not_none_when_config_present(self):
        import policy.store as m
        assert m._hgshm_settings is not None


# ═════════════════════════════════════════════════════════════════════════════
# A19 — _default_*_cfg() functions return correct sub-configs
# ═════════════════════════════════════════════════════════════════════════════

class TestDefaultCfgFunctions:
    """_default_*_cfg() must delegate to module-level singleton, not re-import."""

    def test_learner_cfg_returns_learner_sub_config(self):
        from policy.learner import _default_learner_cfg
        cfg = _default_learner_cfg()
        assert cfg is not None
        assert hasattr(cfg, "decay_factor")
        assert hasattr(cfg, "reward_threshold")
        assert hasattr(cfg, "cache_max_size")

    def test_optimizer_cfg_returns_optimizer_sub_config(self):
        from policy.optimizer import _default_optimizer_cfg
        cfg = _default_optimizer_cfg()
        assert cfg is not None
        assert hasattr(cfg, "min_observations")
        assert hasattr(cfg, "mutation_scale")

    def test_reward_cfg_returns_reward_engine_sub_config(self):
        from policy.reward import _default_reward_cfg
        cfg = _default_reward_cfg()
        assert cfg is not None
        assert hasattr(cfg, "latency_target_ms")

    def test_compiler_cfg_returns_prompt_compiler_sub_config(self):
        from policy.compiler import _default_compiler_cfg
        cfg = _default_compiler_cfg()
        assert cfg is not None
        assert hasattr(cfg, "token_budget")
        assert hasattr(cfg, "max_memory_nodes")

    def test_learner_cfg_returns_none_when_settings_none(self):
        """When _adma_settings is None, helpers must return None gracefully."""
        import policy.learner as m
        original = m._adma_settings
        try:
            m._adma_settings = None
            from policy.learner import _default_learner_cfg
            assert _default_learner_cfg() is None
        finally:
            m._adma_settings = original

    def test_optimizer_cfg_returns_none_when_settings_none(self):
        import policy.optimizer as m
        original = m._adma_settings
        try:
            m._adma_settings = None
            from policy.optimizer import _default_optimizer_cfg
            assert _default_optimizer_cfg() is None
        finally:
            m._adma_settings = original

    def test_reward_cfg_returns_none_when_settings_none(self):
        import policy.reward as m
        original = m._adma_settings
        try:
            m._adma_settings = None
            from policy.reward import _default_reward_cfg
            assert _default_reward_cfg() is None
        finally:
            m._adma_settings = original

    def test_compiler_cfg_returns_none_when_settings_none(self):
        import policy.compiler as m
        original = m._adma_settings
        try:
            m._adma_settings = None
            from policy.compiler import _default_compiler_cfg
            assert _default_compiler_cfg() is None
        finally:
            m._adma_settings = original

    def test_cfg_functions_do_not_contain_lazy_import(self):
        """_default_*_cfg function bodies must not contain inline 'from config' imports."""
        import inspect
        import policy.learner as lm
        import policy.optimizer as om
        import policy.reward as rm
        import policy.compiler as cm

        for fn in [lm._default_learner_cfg, om._default_optimizer_cfg,
                   rm._default_reward_cfg, cm._default_compiler_cfg]:
            src = inspect.getsource(fn)
            assert "from config" not in src, (
                f"{fn.__name__} still contains a lazy 'from config' import"
            )
            assert "import adma_settings" not in src, (
                f"{fn.__name__} still contains an inline import of adma_settings"
            )


# ═════════════════════════════════════════════════════════════════════════════
# A19 — store.py inline lazy imports eliminated
# ═════════════════════════════════════════════════════════════════════════════

class TestStoreLazyImportsEliminated:
    """store.py method bodies must not contain inline config imports."""

    def test_init_no_lazy_hgshm_import(self):
        import inspect
        from policy.store import PolicyStore
        src = inspect.getsource(PolicyStore.__init__)
        assert "from config.settings import hgshm_settings" not in src, (
            "PolicyStore.__init__ must not contain a lazy hgshm_settings import"
        )

    def test_log_reward_no_lazy_adma_import(self):
        import inspect
        from policy.store import PolicyStore
        src = inspect.getsource(PolicyStore.log_reward)
        assert "from config.settings import adma_settings" not in src, (
            "PolicyStore.log_reward must not contain a lazy adma_settings import"
        )

    def test_reward_stats_no_lazy_adma_import(self):
        import inspect
        from policy.store import PolicyStore
        src = inspect.getsource(PolicyStore.reward_stats)
        assert "from config.settings import adma_settings" not in src, (
            "PolicyStore.reward_stats must not contain a lazy adma_settings import"
        )

    def test_store_uses_module_level_hgshm_settings_for_db_path(self, tmp_path):
        """PolicyStore must use _hgshm_settings for db filename, not a lazy import."""
        import policy.store as m
        original = m._hgshm_settings
        try:
            # Patch _hgshm_settings to supply a custom db filename
            fake = MagicMock()
            fake.database.policy_db = "custom_a19.db"
            m._hgshm_settings = fake
            from policy.store import PolicyStore
            store = PolicyStore(memory_dir=tmp_path)
            assert store._db_path.name == "custom_a19.db", (
                f"Expected custom_a19.db, got {store._db_path.name}"
            )
            store.close()
        finally:
            m._hgshm_settings = original

    def test_store_falls_back_to_default_when_hgshm_settings_none(self, tmp_path):
        """When _hgshm_settings is None, store must use 'policy.db'."""
        import policy.store as m
        original = m._hgshm_settings
        try:
            m._hgshm_settings = None
            from policy.store import PolicyStore
            store = PolicyStore(memory_dir=tmp_path)
            assert store._db_path.name == "policy.db"
            store.close()
        finally:
            m._hgshm_settings = original

    def test_log_reward_uses_module_level_settings(self, tmp_path):
        """log_reward must use _adma_settings.reward_log.max_rows_per_policy."""
        import policy.store as m
        from policy.store import PolicyStore
        from policy.models import PolicyRecord, PolicyType, PolicyDomain, RewardSignal, RewardType
        import uuid

        original = m._adma_settings
        try:
            fake = MagicMock()
            fake.reward_log.max_rows_per_policy = 5
            fake.reward_log.stats_last_n = 1000
            m._adma_settings = fake

            store = PolicyStore(memory_dir=tmp_path)
            p = PolicyRecord(
                policy_id=str(uuid.uuid4()),
                name="a19-test",
                policy_type=PolicyType.PLANNER_CONFIG,
                domain=PolicyDomain.SYSTEM,
                config={},
            )
            store.save(p)

            # Log 7 rewards; cap is 5 — oldest 2 should be pruned
            for i in range(7):
                sig = RewardSignal(
                    policy_id=p.policy_id,
                    reward_type=RewardType.TASK_COMPLETED,
                    value=float(i) / 6,
                    context={},
                )
                store.log_reward(sig)

            count = store.reward_log_count(p.policy_id)
            assert count <= 5, (
                f"log_reward must honour _adma_settings cap of 5, got {count}"
            )
            store.close()
        finally:
            m._adma_settings = original

    def test_reward_stats_uses_module_level_settings(self, tmp_path):
        """reward_stats must use _adma_settings.reward_log.stats_last_n."""
        import policy.store as m
        from policy.store import PolicyStore
        from policy.models import PolicyRecord, PolicyType, PolicyDomain, RewardSignal, RewardType
        import uuid

        original = m._adma_settings
        try:
            fake = MagicMock()
            fake.reward_log.max_rows_per_policy = 1000
            fake.reward_log.stats_last_n = 3
            m._adma_settings = fake

            store = PolicyStore(memory_dir=tmp_path)
            p = PolicyRecord(
                policy_id=str(uuid.uuid4()),
                name="a19-stats",
                policy_type=PolicyType.PLANNER_CONFIG,
                domain=PolicyDomain.SYSTEM,
                config={},
            )
            store.save(p)

            values = [0.1, 0.2, 0.3, 0.4, 0.5]
            for v in values:
                sig = RewardSignal(
                    policy_id=p.policy_id,
                    reward_type=RewardType.TASK_COMPLETED,
                    value=v,
                    context={},
                )
                store.log_reward(sig)

            stats = store.reward_stats(p.policy_id)
            # With last_n=3, mean should be from [0.3, 0.4, 0.5] = 0.4
            assert abs(stats["mean"] - 0.4) < 1e-6, (
                f"reward_stats mean should be 0.4 (last 3 of 5), got {stats['mean']}"
            )
            store.close()
        finally:
            m._adma_settings = original


# ═════════════════════════════════════════════════════════════════════════════
# A19 — end-to-end: constructors still work correctly
# ═════════════════════════════════════════════════════════════════════════════

class TestConstructorsWorkWithModuleLevelSettings:
    """Full instantiation must still work after the lazy→module-level change."""

    def test_policy_learner_instantiates(self, tmp_path):
        from policy.store import PolicyStore
        from policy.learner import PolicyLearner
        store = PolicyStore(memory_dir=tmp_path)
        learner = PolicyLearner(policy_store=store)
        assert learner._decay_factor == pytest.approx(0.995)
        store.close()

    def test_policy_optimizer_instantiates(self, tmp_path):
        from policy.store import PolicyStore
        from policy.optimizer import PolicyOptimizer
        store = PolicyStore(memory_dir=tmp_path)
        opt = PolicyOptimizer(policy_store=store)
        assert opt._min_obs == 10
        store.close()

    def test_reward_engine_instantiates(self, tmp_path):
        from policy.store import PolicyStore
        from policy.learner import PolicyLearner
        from policy.reward import RewardEngine
        store = PolicyStore(memory_dir=tmp_path)
        learner = PolicyLearner(policy_store=store)
        engine = RewardEngine(learner=learner)
        assert engine._learner is learner
        store.close()

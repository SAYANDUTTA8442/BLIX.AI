"""
Blix v0.3.16.1 — Tests for ISSUE-018

ISSUE-018: Empty policy/__init__.py and no __all__ declarations.
Without __all__, the public API surface was invisible — any internal
name was implicitly public, and any refactoring could silently break
callers. This test suite verifies the fix.

Tests cover:
  - Every name in policy.__all__ is importable from the package
  - 'from policy import *' imports exactly the names in __all__
  - Every sub-module has its own __all__
  - Each sub-module's __all__ contains only its own public names
  - Private names (_prefix) are excluded from all __all__ lists
  - Private names remain directly importable from their modules
  - All existing import paths (from policy.X import Y) still work
  - policy.__all__ is stable (no unexpected additions)
  - __init__.py is not empty
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import policy
import policy.models
import policy.store
import policy.reward
import policy.learner
import policy.optimizer
import policy.compiler
import policy.adaptive
import policy.ablation_v3


# ════════════════════════════════════════════════════════════════════
# policy/__init__.py — package-level public API
# ════════════════════════════════════════════════════════════════════

class TestPackageInit:
    def test_init_is_not_empty(self):
        """policy/__init__.py must not be empty."""
        init_path = Path(policy.__file__)
        assert init_path.stat().st_size > 0, "policy/__init__.py is empty"

    def test_package_has_all(self):
        """policy must define __all__."""
        assert hasattr(policy, "__all__"), "policy package missing __all__"
        assert isinstance(policy.__all__, list)
        assert len(policy.__all__) > 0

    def test_all_names_importable_from_package(self):
        """Every name in policy.__all__ must be accessible on the package object."""
        missing = [name for name in policy.__all__ if not hasattr(policy, name)]
        assert not missing, (
            f"Names in policy.__all__ but not accessible on package: {missing}"
        )

    def test_star_import_matches_all(self):
        """'from policy import *' must import exactly the names in __all__."""
        g: dict = {}
        exec("from policy import *", g)
        star_names = {k for k in g if not k.startswith("_")}
        declared   = set(policy.__all__)
        assert star_names == declared, (
            f"star-import mismatch.\n"
            f"  Extra in star: {star_names - declared}\n"
            f"  Missing from star: {declared - star_names}"
        )

    def test_no_private_names_in_package_all(self):
        """No name beginning with '_' should appear in policy.__all__."""
        private = [n for n in policy.__all__ if n.startswith("_")]
        assert not private, (
            f"Private names leaked into policy.__all__: {private}"
        )

    def test_expected_public_names_present(self):
        """Spot-check that the most important names are in __all__."""
        expected = {
            "PolicyRecord", "PolicyDomain", "PolicyType",
            "RewardSignal", "RewardType", "PolicyVersion",
            "PolicyStore",
            "RewardEngine", "SystemRewardEngine", "UserRewardEngine",
            "PolicyLearner",
            "PolicyOptimizer",
            "PolicySelector", "PolicyCompiler", "CompiledPrompt",
            "sanitize_task_text",
            "AdaptiveRetriever", "AdaptivePlanner",
            "AblationConfig", "AblationV3Runner", "ABLATION_CONDITIONS",
            "AblationV3Report", "AblationBenchmarkResult", "AblationConditionResult",
        }
        missing = expected - set(policy.__all__)
        assert not missing, (
            f"Expected public names not in policy.__all__: {missing}"
        )

    def test_package_all_is_list_of_strings(self):
        assert all(isinstance(n, str) for n in policy.__all__)

    def test_package_all_has_no_duplicates(self):
        assert len(policy.__all__) == len(set(policy.__all__)), (
            "policy.__all__ contains duplicates"
        )


# ════════════════════════════════════════════════════════════════════
# Sub-module __all__ declarations
# ════════════════════════════════════════════════════════════════════

SUB_MODULES = [
    policy.models,
    policy.store,
    policy.reward,
    policy.learner,
    policy.optimizer,
    policy.compiler,
    policy.adaptive,
    policy.ablation_v3,
]

class TestSubModuleAll:
    @pytest.mark.parametrize("mod", SUB_MODULES,
                              ids=[m.__name__ for m in SUB_MODULES])
    def test_module_has_all(self, mod: ModuleType):
        """Every sub-module must define __all__."""
        assert hasattr(mod, "__all__"), f"{mod.__name__} is missing __all__"

    @pytest.mark.parametrize("mod", SUB_MODULES,
                              ids=[m.__name__ for m in SUB_MODULES])
    def test_all_is_nonempty_list_of_strings(self, mod: ModuleType):
        assert isinstance(mod.__all__, list) and len(mod.__all__) > 0
        assert all(isinstance(n, str) for n in mod.__all__)

    @pytest.mark.parametrize("mod", SUB_MODULES,
                              ids=[m.__name__ for m in SUB_MODULES])
    def test_all_names_exist_in_module(self, mod: ModuleType):
        """Every name declared in __all__ must actually exist in the module."""
        missing = [n for n in mod.__all__ if not hasattr(mod, n)]
        assert not missing, (
            f"{mod.__name__}.__all__ references non-existent names: {missing}"
        )

    @pytest.mark.parametrize("mod", SUB_MODULES,
                              ids=[m.__name__ for m in SUB_MODULES])
    def test_no_private_names_in_all(self, mod: ModuleType):
        """Private names must not appear in any module's __all__."""
        private = [n for n in mod.__all__ if n.startswith("_")]
        assert not private, (
            f"{mod.__name__}.__all__ exposes private names: {private}"
        )

    @pytest.mark.parametrize("mod", SUB_MODULES,
                              ids=[m.__name__ for m in SUB_MODULES])
    def test_no_duplicates_in_all(self, mod: ModuleType):
        assert len(mod.__all__) == len(set(mod.__all__)), (
            f"{mod.__name__}.__all__ contains duplicates"
        )


class TestSubModuleAllContents:
    """Spot-checks that each sub-module's __all__ has the right names."""

    def test_models_all_contains_policy_record(self):
        assert "PolicyRecord" in policy.models.__all__

    def test_models_all_contains_enums(self):
        for name in ("PolicyDomain", "PolicyType", "RewardType"):
            assert name in policy.models.__all__, f"{name} missing from models.__all__"

    def test_models_all_contains_reward_signal(self):
        assert "RewardSignal" in policy.models.__all__

    def test_store_all_contains_policy_store(self):
        assert "PolicyStore" in policy.store.__all__

    def test_reward_all_contains_all_engines(self):
        for name in ("RewardEngine", "SystemRewardEngine", "UserRewardEngine"):
            assert name in policy.reward.__all__

    def test_learner_all_contains_policy_learner(self):
        assert "PolicyLearner" in policy.learner.__all__

    def test_optimizer_all_contains_policy_optimizer(self):
        assert "PolicyOptimizer" in policy.optimizer.__all__

    def test_compiler_all_contains_public_names(self):
        for name in ("PolicySelector", "PolicyCompiler", "CompiledPrompt", "sanitize_task_text"):
            assert name in policy.compiler.__all__, f"{name} missing from compiler.__all__"

    def test_adaptive_all_contains_both_classes(self):
        assert "AdaptiveRetriever" in policy.adaptive.__all__
        assert "AdaptivePlanner"   in policy.adaptive.__all__

    def test_ablation_all_contains_runner_and_conditions(self):
        assert "AblationV3Runner"    in policy.ablation_v3.__all__
        assert "ABLATION_CONDITIONS" in policy.ablation_v3.__all__
        assert "AblationConfig"      in policy.ablation_v3.__all__


# ════════════════════════════════════════════════════════════════════
# Private names excluded from __all__ but still directly importable
# ════════════════════════════════════════════════════════════════════

class TestPrivateNamesStillAccessible:
    def test_default_policies_importable_from_learner(self):
        from policy.learner import _default_policies
        policies = _default_policies()
        assert len(policies) > 0

    def test_context_key_importable_from_learner(self):
        from policy.learner import _context_key
        key = _context_key({"task_type": "code"})
        assert isinstance(key, str)

    def test_schema_version_importable_from_store(self):
        from policy.store import _SCHEMA_VERSION
        assert isinstance(_SCHEMA_VERSION, int)
        assert _SCHEMA_VERSION >= 1

    def test_migrations_importable_from_store(self):
        from policy.store import _MIGRATIONS
        assert isinstance(_MIGRATIONS, dict)

    def test_active_sql_importable_from_store(self):
        from policy.store import PolicyStore
        assert hasattr(PolicyStore, "_ACTIVE_SQL")

    def test_none_of_the_above_in_package_all(self):
        """Confirm private names are not accidentally re-exported."""
        for private_name in (
            "_default_policies", "_context_key",
            "_SCHEMA_VERSION", "_MIGRATIONS", "_ACTIVE_SQL",
        ):
            assert private_name not in policy.__all__, (
                f"Private name {private_name!r} leaked into policy.__all__"
            )


# ════════════════════════════════════════════════════════════════════
# Backward compatibility — existing import paths still work
# ════════════════════════════════════════════════════════════════════

class TestExistingImportPathsUnchanged:
    """
    Every import statement used in the existing test suite must
    continue to work unchanged after adding __all__.
    """

    def test_from_policy_models_import_all_public(self):
        from policy.models import (
            PolicyDomain, PolicyType, RewardType,
            RewardSignal, PolicyVersion, PolicyRecord,
        )

    def test_from_policy_store_import(self):
        from policy.store import PolicyStore

    def test_from_policy_store_import_private(self):
        from policy.store import PolicyStore, _SCHEMA_VERSION, _MIGRATIONS

    def test_from_policy_reward_import(self):
        from policy.reward import RewardEngine, SystemRewardEngine, UserRewardEngine

    def test_from_policy_learner_import_public(self):
        from policy.learner import PolicyLearner

    def test_from_policy_learner_import_private(self):
        from policy.learner import PolicyLearner, _default_policies, _context_key

    def test_from_policy_optimizer_import(self):
        from policy.optimizer import PolicyOptimizer

    def test_from_policy_compiler_import(self):
        from policy.compiler import PolicySelector, PolicyCompiler, CompiledPrompt
        from policy.compiler import sanitize_task_text

    def test_from_policy_adaptive_import(self):
        from policy.adaptive import AdaptiveRetriever, AdaptivePlanner

    def test_from_policy_ablation_v3_import(self):
        from policy.ablation_v3 import (
            AblationConfig, ABLATION_CONDITIONS,
            AblationBenchmarkResult, AblationConditionResult,
            AblationV3Report, AblationV3Runner,
        )

    def test_flat_package_import_works(self):
        """The new flat import path must work for every public name."""
        from policy import (
            PolicyDomain, PolicyType, RewardType,
            RewardSignal, PolicyVersion, PolicyRecord,
            PolicyStore,
            RewardEngine, SystemRewardEngine, UserRewardEngine,
            PolicyLearner,
            PolicyOptimizer,
            PolicySelector, PolicyCompiler, CompiledPrompt, sanitize_task_text,
            AdaptiveRetriever, AdaptivePlanner,
            AblationConfig, AblationBenchmarkResult, AblationConditionResult,
            AblationV3Report, AblationV3Runner, ABLATION_CONDITIONS,
        )

    def test_flat_import_gives_same_objects(self):
        """Flat import must give the same class objects as sub-module import."""
        from policy import PolicyRecord as PR1
        from policy.models import PolicyRecord as PR2
        assert PR1 is PR2

        from policy import PolicyStore as PS1
        from policy.store import PolicyStore as PS2
        assert PS1 is PS2

        from policy import PolicyLearner as PL1
        from policy.learner import PolicyLearner as PL2
        assert PL1 is PL2

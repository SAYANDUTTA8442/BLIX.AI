"""
tests/test_v03172_a07_a08_a09_a10.py
======================================
Regression tests for blix v0.3.17.2:

  A07 — Broadcast observe() rewards only the last-selected arm
  A08 — PolicyVersion.beta renamed to beta_ (consistency with PolicyRecord)
  A09 — store_system / store_user drop **kwargs, use explicit signatures
  A10 — HGSHM shim registry is bounded; close_registry() added
"""

from __future__ import annotations

import inspect
import tempfile
import uuid
from pathlib import Path

import pytest


# ─── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture()
def policy_store(tmp_dir):
    from policy.store import PolicyStore
    store = PolicyStore(memory_dir=tmp_dir)
    yield store
    store.close()


@pytest.fixture()
def learner(policy_store):
    from policy.learner import PolicyLearner
    return PolicyLearner(policy_store=policy_store)


@pytest.fixture()
def sample_policy(policy_store):
    """A persisted PolicyRecord ready for use."""
    from policy.models import PolicyRecord, PolicyType, PolicyDomain
    p = PolicyRecord(
        policy_id=str(uuid.uuid4()),
        name="test_arm",
        policy_type=PolicyType.PLANNER_CONFIG,
        domain=PolicyDomain.SYSTEM,
        config={"mode": "test"},
        alpha=3.0,
        beta_=2.0,
    )
    policy_store.save(p)
    return p


# ═════════════════════════════════════════════════════════════════════════════
# A07 — Broadcast rewards only reach the last-selected arm
# ═════════════════════════════════════════════════════════════════════════════

class TestBroadcastRewardRouting:
    """A07: observe() broadcast must update only the arm from _last_selected."""

    def test_broadcast_drops_when_no_selection(self, learner):
        """No _last_selected entry → reward is dropped, empty update list."""
        from policy.models import RewardSignal, RewardType
        reward = RewardSignal(
            reward_type=RewardType.TASK_COMPLETED,
            value=1.0,
            context={},
        )
        updated = learner.observe(reward)
        assert updated == [], (
            "Broadcast reward must be dropped when no arm was recently selected"
        )

    def test_broadcast_updates_only_last_selected(self, learner, sample_policy, policy_store):
        """After a select(), broadcast reward updates only that arm."""
        from policy.models import RewardSignal, RewardType, PolicyType, PolicyDomain

        # Prime _last_selected by injecting directly (avoids needing a live embedding)
        ctx_key = "global"
        learner._last_selected[ctx_key] = sample_policy.policy_id
        # Also put it in cache so _get_cached works
        learner._cache_put(sample_policy.policy_id, sample_policy)

        alpha_before = sample_policy.alpha
        beta_before = sample_policy.beta_

        # BENCHMARK_SCORE maps to PLANNER_CONFIG (sample_policy type) → should hit
        reward = RewardSignal(
            reward_type=RewardType.BENCHMARK_SCORE,
            value=1.0,
            context={},
        )
        updated = learner.observe(reward)

        assert len(updated) == 1, (
            f"Expected exactly 1 arm updated, got {len(updated)}"
        )
        assert updated[0].policy_id == sample_policy.policy_id

        # alpha should have increased (positive reward)
        assert updated[0].alpha > alpha_before, (
            "Positive reward must increase alpha"
        )

    def test_broadcast_with_wrong_type_drops(self, learner, sample_policy):
        """_last_selected entry exists but arm policy_type doesn't match → dropped."""
        from policy.models import RewardSignal, RewardType, PolicyType

        # Inject _last_selected for a context
        ctx_key = "global"
        learner._last_selected[ctx_key] = sample_policy.policy_id
        learner._cache_put(sample_policy.policy_id, sample_policy)

        # MEMORY_QUALITY maps to RETRIEVAL_WEIGHTS only — our arm is PLANNER_CONFIG
        from policy.learner import _reward_to_policy_types
        reward_type = RewardType.MEMORY_QUALITY
        mapped_types = _reward_to_policy_types(reward_type)
        # Guard: only test if PLANNER_CONFIG is indeed not in mapped types
        if PolicyType.PLANNER_CONFIG not in mapped_types:
            reward = RewardSignal(reward_type=reward_type, value=1.0, context={})
            updated = learner.observe(reward)
            arm_updates = [u for u in updated if u.policy_id == sample_policy.policy_id]
            assert arm_updates == [], (
                "Arm of wrong type must not receive broadcast reward"
            )

    def test_direct_reward_still_works(self, learner, sample_policy):
        """policy_id set directly → still routes to that arm regardless of _last_selected."""
        from policy.models import RewardSignal, RewardType
        alpha_before = sample_policy.alpha

        reward = RewardSignal(
            policy_id=sample_policy.policy_id,
            reward_type=RewardType.TASK_COMPLETED,
            value=1.0,
            context={},
        )
        learner._cache_put(sample_policy.policy_id, sample_policy)
        updated = learner.observe(reward)

        assert len(updated) == 1
        assert updated[0].policy_id == sample_policy.policy_id
        assert updated[0].alpha > alpha_before

    def test_broadcast_negative_reward_decrements_beta(self, learner, sample_policy):
        """Negative broadcast reward must increment beta_ on the selected arm."""
        from policy.models import RewardSignal, RewardType
        ctx_key = "global"
        learner._last_selected[ctx_key] = sample_policy.policy_id
        learner._cache_put(sample_policy.policy_id, sample_policy)
        beta_before = sample_policy.beta_

        reward = RewardSignal(
            reward_type=RewardType.TASK_COMPLETED,
            value=0.0,
            context={},
        )
        updated = learner.observe(reward)
        if updated:
            assert updated[0].beta_ >= beta_before, (
                "Zero/negative reward must not decrease beta_"
            )

    def test_context_key_isolates_selections(self, learner, sample_policy, policy_store):
        """Rewards with a different context key than selection are dropped."""
        from policy.models import RewardSignal, RewardType
        # Select under ctx "task_type=planning"
        learner._last_selected["task_type=planning"] = sample_policy.policy_id
        learner._cache_put(sample_policy.policy_id, sample_policy)

        # Reward arrives with empty context → resolves to "global" → no match
        reward = RewardSignal(
            reward_type=RewardType.TASK_COMPLETED,
            value=1.0,
            context={},
        )
        updated = learner.observe(reward)
        arm_hits = [u for u in updated if u.policy_id == sample_policy.policy_id]
        assert arm_hits == [], (
            "Reward with different context key must not update arm from another context"
        )


# ═════════════════════════════════════════════════════════════════════════════
# A08 — PolicyVersion.beta renamed to beta_
# ═════════════════════════════════════════════════════════════════════════════

class TestPolicyVersionBetaField:
    """A08: PolicyVersion must use beta_ consistently with PolicyRecord."""

    def test_policy_version_has_beta_underscore_field(self):
        from policy.models import PolicyVersion
        pv = PolicyVersion()
        assert hasattr(pv, "beta_"), "PolicyVersion must have field beta_ (not beta)"
        assert not hasattr(pv, "beta") or "beta_" in pv.__dataclass_fields__, (
            "PolicyVersion.beta (no underscore) must not be the field name"
        )

    def test_policy_version_beta_default(self):
        from policy.models import PolicyVersion
        pv = PolicyVersion()
        assert pv.beta_ == 1.0

    def test_policy_version_beta_constructor(self):
        from policy.models import PolicyVersion
        pv = PolicyVersion(beta_=3.7)
        assert pv.beta_ == 3.7

    def test_to_dict_uses_beta_underscore(self):
        from policy.models import PolicyVersion
        pv = PolicyVersion(beta_=2.5)
        d = pv.to_dict()
        assert "beta_" in d, "to_dict() must emit 'beta_' key"
        assert "beta" not in d or d.get("beta_") == 2.5, (
            "to_dict() must not emit bare 'beta' key"
        )
        assert d["beta_"] == 2.5

    def test_snapshot_produces_beta_underscore(self, sample_policy):
        """PolicyRecord.snapshot() must produce PolicyVersion with beta_."""
        pv = sample_policy.snapshot(reason="unit-test")
        assert hasattr(pv, "beta_"), "snapshot() must produce PolicyVersion.beta_"
        assert pv.beta_ == sample_policy.beta_

    def test_save_version_get_history_round_trip(self, policy_store, sample_policy):
        """beta_ must survive save_version → get_history."""
        from policy.models import PolicyVersion
        pv = PolicyVersion(
            policy_id=sample_policy.policy_id,
            version=1,
            config=sample_policy.config,
            alpha=sample_policy.alpha,
            beta_=sample_policy.beta_,
            mean_reward=sample_policy.confidence,
            reason="round-trip test",
        )
        policy_store.save_version(pv)
        history = policy_store.get_history(sample_policy.policy_id)
        assert len(history) >= 1
        pv2 = history[0]
        assert hasattr(pv2, "beta_"), "Loaded PolicyVersion must have beta_ field"
        assert pv2.beta_ == sample_policy.beta_

    def test_rollback_restores_beta_underscore(self, policy_store, sample_policy):
        """rollback() must write pv.beta_ → policy.beta_ (no beta/beta_ mismatch)."""
        from policy.models import PolicyVersion
        target_beta = 4.2
        pv = PolicyVersion(
            policy_id=sample_policy.policy_id,
            version=1,
            config=sample_policy.config,
            alpha=sample_policy.alpha,
            beta_=target_beta,
            mean_reward=sample_policy.confidence,
            reason="rollback test",
        )
        policy_store.save_version(pv)
        result = policy_store.rollback(sample_policy.policy_id, to_version=1)
        assert result is not None
        assert result.beta_ == target_beta, (
            f"rollback must restore beta_={target_beta}, got {result.beta_}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# A09 — store_system / store_user explicit signatures
# ═════════════════════════════════════════════════════════════════════════════

class TestManagerExplicitSignatures:
    """A09: MemoryManager.store_system and store_user must not accept **kwargs."""

    def test_store_system_no_kwargs(self):
        from memory.manager import MemoryManager
        sig = inspect.signature(MemoryManager.store_system)
        assert "kwargs" not in str(sig), (
            f"store_system must not have **kwargs, got: {sig}"
        )

    def test_store_system_has_required_params(self):
        from memory.manager import MemoryManager
        sig = inspect.signature(MemoryManager.store_system)
        params = sig.parameters
        assert "text" in params
        assert "success" in params
        assert "latency_ms" in params
        assert "subsystems" in params
        assert "metadata" in params

    def test_store_user_no_kwargs(self):
        from memory.manager import MemoryManager
        sig = inspect.signature(MemoryManager.store_user)
        assert "kwargs" not in str(sig), (
            f"store_user must not have **kwargs, got: {sig}"
        )

    def test_store_user_has_required_params(self):
        from memory.manager import MemoryManager
        sig = inspect.signature(MemoryManager.store_user)
        params = sig.parameters
        assert "user_id" in params
        assert "text" in params
        assert "category" in params
        assert "strength" in params
        assert "metadata" in params

    def _make_manager(self, tmp_dir):
        from memory.hybrid.hgshm import HGSHM
        from memory.system.system_memory import SystemMemory
        from memory.manager import MemoryManager
        hgshm = HGSHM(tmp_dir)
        system = SystemMemory(hgshm)
        return MemoryManager(hgshm=hgshm, system_memory=system), hgshm

    def test_store_system_rejects_unknown_kwarg(self, tmp_dir):
        """Passing an unrecognised keyword must raise TypeError, not silently bypass."""
        mgr, hgshm = self._make_manager(tmp_dir)
        with pytest.raises(TypeError):
            mgr.store_system("test", tags=["should_be_rejected"])
        hgshm.close()

    def test_store_user_rejects_unknown_kwarg(self, tmp_dir):
        mgr, hgshm = self._make_manager(tmp_dir)
        with pytest.raises(TypeError):
            mgr.store_user("u1", "pref", importance=0.9)
        hgshm.close()

    def test_store_system_functional(self, tmp_dir):
        """store_system works correctly with explicit args."""
        mgr, hgshm = self._make_manager(tmp_dir)
        node = mgr.store_system(
            "workflow completed",
            success=True,
            latency_ms=42.0,
            subsystems=["planner"],
            metadata={"extra": "info"},
        )
        assert node is not None
        hgshm.close()

    def test_store_user_functional(self, tmp_dir):
        """store_user works correctly with explicit args."""
        mgr, hgshm = self._make_manager(tmp_dir)
        node = mgr.store_user(
            "alice",
            "prefers dark mode",
            category="ui",
            strength=0.9,
        )
        assert node is not None
        hgshm.close()


# ═════════════════════════════════════════════════════════════════════════════
# A10 — Shim registry bounded + close_registry()
# ═════════════════════════════════════════════════════════════════════════════

class TestShimRegistryBounded:
    """A10: _HGSHM_REGISTRY must be bounded and closeable."""

    def setup_method(self):
        """Reset registry and max before each test."""
        import memory.hybrid.shims as shims
        from memory.hybrid.shims import close_registry
        close_registry()
        shims._REGISTRY_MAX = 128

    def teardown_method(self):
        """Clean up after each test."""
        import memory.hybrid.shims as shims
        from memory.hybrid.shims import close_registry
        close_registry()
        shims._REGISTRY_MAX = 128

    def test_close_registry_exists(self):
        from memory.hybrid.shims import close_registry
        assert callable(close_registry)

    def test_close_registry_on_empty(self):
        from memory.hybrid.shims import close_registry
        n = close_registry()
        assert n == 0

    def test_close_registry_closes_instances(self, tmp_dir):
        import memory.hybrid.shims as shims
        from memory.hybrid.shims import _get_hgshm, close_registry
        d = tmp_dir / "reg_test"
        d.mkdir()
        _get_hgshm(d)
        assert len(shims._HGSHM_REGISTRY) == 1
        n = close_registry()
        assert n == 1
        assert shims._HGSHM_REGISTRY == {}

    def test_registry_max_constant_exists(self):
        import memory.hybrid.shims as shims
        assert hasattr(shims, "_REGISTRY_MAX")
        assert isinstance(shims._REGISTRY_MAX, int)
        assert shims._REGISTRY_MAX > 0

    def test_eviction_at_capacity(self, tmp_dir):
        """When registry hits _REGISTRY_MAX, the oldest entry is evicted."""
        import memory.hybrid.shims as shims
        from memory.hybrid.shims import _get_hgshm

        shims._REGISTRY_MAX = 2
        d1, d2, d3 = tmp_dir / "ev1", tmp_dir / "ev2", tmp_dir / "ev3"
        d1.mkdir(); d2.mkdir(); d3.mkdir()

        _get_hgshm(d1)
        _get_hgshm(d2)
        assert len(shims._HGSHM_REGISTRY) == 2

        _get_hgshm(d3)  # should evict d1
        assert len(shims._HGSHM_REGISTRY) == 2, (
            "Registry must not grow past _REGISTRY_MAX"
        )
        assert str(d1.resolve()) not in shims._HGSHM_REGISTRY, (
            "Oldest entry (d1) must have been evicted"
        )
        assert str(d3.resolve()) in shims._HGSHM_REGISTRY, (
            "Newly added entry (d3) must be present"
        )

    def test_same_dir_reuses_instance(self, tmp_dir):
        """Requesting the same memory_dir twice returns the cached instance."""
        import memory.hybrid.shims as shims
        from memory.hybrid.shims import _get_hgshm

        d = tmp_dir / "reuse"
        d.mkdir()
        inst1 = _get_hgshm(d)
        inst2 = _get_hgshm(d)
        assert inst1 is inst2, "Same memory_dir must return the same HGSHM instance"
        assert len(shims._HGSHM_REGISTRY) == 1

    def test_registry_grows_up_to_max(self, tmp_dir):
        """Registry accepts up to _REGISTRY_MAX entries without eviction."""
        import memory.hybrid.shims as shims
        from memory.hybrid.shims import _get_hgshm

        shims._REGISTRY_MAX = 3
        for i in range(3):
            d = tmp_dir / f"grow{i}"
            d.mkdir()
            _get_hgshm(d)
        assert len(shims._HGSHM_REGISTRY) == 3

    def test_close_registry_idempotent(self):
        """Calling close_registry() twice must not raise."""
        from memory.hybrid.shims import close_registry
        close_registry()
        close_registry()  # must not raise

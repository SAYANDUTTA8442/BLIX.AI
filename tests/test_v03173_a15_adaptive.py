"""
tests/test_v03173_a15_adaptive.py
===================================
Tests for A15: AdaptiveRetriever and AdaptivePlanner had zero test coverage.

Covers:
  - AdaptiveRetriever.retrieve():
      weight application to HybridRetriever._weights
      fallback when PolicySelector returns no policy (uniform weights)
      reward dispatch after retrieval with results
      reward dispatch skipped when no results returned
      no crash when reward_engine is None
      weight-application exception swallowed, retrieval still proceeds
      _current_policy_id updated when select_one returns a policy
      current_weights property
  - AdaptivePlanner.search():
      policy config applied (beam_width / max_depth)
      fallback defaults when no policy config
      reward dispatched after search
      no crash when reward_engine is None
      _current_policy_id updated when select_one returns a policy
      current_config property
      deferred BeamSearchPlanner import is resolvable
"""

from __future__ import annotations

import types
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from policy.adaptive import AdaptiveRetriever, AdaptivePlanner


# ─── helpers / lightweight stubs ─────────────────────────────────────────────

def _make_policy(policy_type_val="retrieval_weights", config: dict | None = None):
    """Minimal PolicyRecord-like stub."""
    p = MagicMock()
    p.policy_id = str(uuid.uuid4())
    p.policy_type.value = policy_type_val
    p.config = config or {}
    p.is_active = True
    return p


def _uniform_weights() -> dict[str, float]:
    keys = ["semantic", "vector", "graph_distance", "importance", "confidence",
            "recency", "hierarchy", "context_similarity", "attention",
            "belief_confidence", "planning_relevance"]
    return {k: 1.0 / len(keys) for k in keys}


def _make_selector(weights_cfg: dict | None = None,
                   policy: Any = None,
                   planner_cfg: dict | None = None):
    """Stub PolicySelector."""
    sel = MagicMock()
    sel.get_retrieval_weights.return_value = weights_cfg or _uniform_weights()
    sel.get_planner_config.return_value = planner_cfg or {"beam_width": 3, "max_depth": 2}
    sel._learner.select_one.return_value = policy
    return sel


def _make_retrieved_memory(score: float = 0.8):
    """Stub RetrievedMemory with final_score."""
    rm = MagicMock()
    rm.final_score = score
    return rm


def _make_hgshm(results: list | None = None):
    """Stub HGSHM with hybrid_retriever."""
    from memory.hybrid.retrieval.hybrid_retriever import HybridWeights
    hgshm = MagicMock()
    hgshm.hybrid_retriever._weights = HybridWeights().normalised()
    hgshm.hybrid_retriever.retrieve.return_value = results if results is not None else []
    return hgshm


# ═════════════════════════════════════════════════════════════════════════════
# AdaptiveRetriever
# ═════════════════════════════════════════════════════════════════════════════

class TestAdaptiveRetriever:

    def test_instantiation(self):
        sel = _make_selector()
        hgshm = _make_hgshm()
        ar = AdaptiveRetriever(hgshm=hgshm, policy_selector=sel)
        assert ar is not None
        assert ar._selector is sel
        assert ar._hgshm is hgshm
        assert ar._reward is None

    def test_weights_applied_to_hybrid_retriever(self):
        """retrieve() must write chosen weights to hgshm.hybrid_retriever._weights."""
        from memory.hybrid.retrieval.hybrid_retriever import HybridWeights
        weights_cfg = {
            "semantic": 0.5, "vector": 0.5,
            "graph_distance": 0.0, "importance": 0.0, "confidence": 0.0,
            "recency": 0.0, "hierarchy": 0.0, "context_similarity": 0.0,
            "attention": 0.0, "belief_confidence": 0.0, "planning_relevance": 0.0,
        }
        hgshm = _make_hgshm()
        sel = _make_selector(weights_cfg=weights_cfg, policy=_make_policy())
        ar = AdaptiveRetriever(hgshm=hgshm, policy_selector=sel)
        ar.retrieve("test query")

        applied: HybridWeights = hgshm.hybrid_retriever._weights
        # After normalise, semantic and vector should each be ~0.5
        assert abs(applied.semantic - 0.5) < 1e-6, (
            f"semantic weight not applied correctly: {applied.semantic}"
        )
        assert abs(applied.vector - 0.5) < 1e-6, (
            f"vector weight not applied correctly: {applied.vector}"
        )

    def test_weights_applied_when_no_policy_returned(self):
        """Even with no active policy, uniform fallback weights are applied."""
        from memory.hybrid.retrieval.hybrid_retriever import HybridWeights
        hgshm = _make_hgshm()
        sel = _make_selector(policy=None)  # no policy arm
        ar = AdaptiveRetriever(hgshm=hgshm, policy_selector=sel)
        ar.retrieve("anything")

        applied = hgshm.hybrid_retriever._weights
        assert isinstance(applied, HybridWeights), (
            "Fallback must still produce a HybridWeights object on _weights"
        )

    def test_retrieve_calls_underlying_retriever(self):
        """retrieve() must call hgshm.hybrid_retriever.retrieve with query + top_k."""
        results = [_make_retrieved_memory(0.9)]
        hgshm = _make_hgshm(results=results)
        sel = _make_selector()
        ar = AdaptiveRetriever(hgshm=hgshm, policy_selector=sel)
        returned = ar.retrieve("my query", top_k=5)

        hgshm.hybrid_retriever.retrieve.assert_called_once()
        call_args = hgshm.hybrid_retriever.retrieve.call_args
        assert call_args[0][0] == "my query" or call_args[1].get("query") == "my query" or \
               "my query" in str(call_args)
        assert returned == results

    def test_reward_dispatched_after_retrieval_with_results(self):
        """on_retrieval must be called when results are non-empty."""
        results = [_make_retrieved_memory(0.8), _make_retrieved_memory(0.6)]
        hgshm = _make_hgshm(results=results)
        reward_engine = MagicMock()
        policy = _make_policy()
        sel = _make_selector(policy=policy)
        ar = AdaptiveRetriever(hgshm=hgshm, policy_selector=sel, reward_engine=reward_engine)
        ar.retrieve("query")

        reward_engine.on_retrieval.assert_called_once()
        call_kwargs = reward_engine.on_retrieval.call_args
        # mean_score should be (0.8 + 0.6) / 2 = 0.7
        score_arg = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("score") or call_kwargs[1].get("mean_score")
        assert abs(score_arg - 0.7) < 1e-6, f"Wrong mean_score: {score_arg}"

    def test_reward_not_dispatched_when_no_results(self):
        """on_retrieval must NOT be called when retriever returns empty list."""
        hgshm = _make_hgshm(results=[])
        reward_engine = MagicMock()
        sel = _make_selector()
        ar = AdaptiveRetriever(hgshm=hgshm, policy_selector=sel, reward_engine=reward_engine)
        ar.retrieve("empty query")

        reward_engine.on_retrieval.assert_not_called()

    def test_no_crash_when_reward_engine_none(self):
        """reward_engine=None must not raise."""
        hgshm = _make_hgshm(results=[_make_retrieved_memory(0.9)])
        sel = _make_selector()
        ar = AdaptiveRetriever(hgshm=hgshm, policy_selector=sel, reward_engine=None)
        # Must not raise
        ar.retrieve("query")

    def test_weight_application_failure_is_swallowed(self):
        """If HybridWeights construction fails, retrieve() continues without raising."""
        hgshm = _make_hgshm(results=[_make_retrieved_memory(0.5)])
        sel = _make_selector(weights_cfg={"bad_key": 999})  # will cause HybridWeights to skip
        ar = AdaptiveRetriever(hgshm=hgshm, policy_selector=sel)
        # Must not raise — weight failure is caught internally
        result = ar.retrieve("query")
        assert result is not None  # retrieval still proceeds

    def test_current_policy_id_set_when_policy_selected(self):
        """_current_policy_id must be updated after select_one returns a policy."""
        policy = _make_policy()
        sel = _make_selector(policy=policy)
        hgshm = _make_hgshm()
        ar = AdaptiveRetriever(hgshm=hgshm, policy_selector=sel)
        assert ar._current_policy_id is None
        ar.retrieve("query")
        assert ar._current_policy_id == policy.policy_id

    def test_current_policy_id_not_set_when_no_policy(self):
        """_current_policy_id stays None when select_one returns None."""
        sel = _make_selector(policy=None)
        hgshm = _make_hgshm()
        ar = AdaptiveRetriever(hgshm=hgshm, policy_selector=sel)
        ar.retrieve("query")
        assert ar._current_policy_id is None

    def test_current_weights_property(self):
        """current_weights must return a dict."""
        sel = _make_selector()
        ar = AdaptiveRetriever(hgshm=_make_hgshm(), policy_selector=sel)
        w = ar.current_weights
        assert isinstance(w, dict)
        assert all(isinstance(v, float) for v in w.values())

    def test_reward_policy_id_passed_through(self):
        """Reward dispatch must include the selected policy_id."""
        policy = _make_policy()
        results = [_make_retrieved_memory(0.75)]
        hgshm = _make_hgshm(results=results)
        reward_engine = MagicMock()
        sel = _make_selector(policy=policy)
        ar = AdaptiveRetriever(hgshm=hgshm, policy_selector=sel, reward_engine=reward_engine)
        ar.retrieve("query")

        call_kwargs = reward_engine.on_retrieval.call_args[1]
        assert call_kwargs.get("policy_id") == policy.policy_id, (
            "policy_id must be forwarded to RewardEngine.on_retrieval"
        )


# ═════════════════════════════════════════════════════════════════════════════
# AdaptivePlanner
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _FakeBeamResult:
    """Stub BeamSearchResult."""
    goal: str = "test"
    best_value: float = 0.85
    runner_up_trajectories: list = field(default_factory=list)
    runner_up_values: list = field(default_factory=list)


class TestAdaptivePlanner:

    def _make_planner(self, planner_cfg=None, policy=None, reward_engine=None,
                      beam_result=None):
        """Build AdaptivePlanner with all dependencies stubbed."""
        vn = MagicMock()
        sel = _make_selector(planner_cfg=planner_cfg, policy=policy)
        ap = AdaptivePlanner(value_network=vn, policy_selector=sel,
                             reward_engine=reward_engine)
        result = beam_result or _FakeBeamResult()
        return ap, sel, result

    def test_instantiation(self):
        vn = MagicMock()
        sel = _make_selector()
        ap = AdaptivePlanner(value_network=vn, policy_selector=sel)
        assert ap._vn is vn
        assert ap._selector is sel
        assert ap._reward is None

    def test_planner_config_applied_to_beam_search(self):
        """search() must instantiate BeamSearchPlanner with policy-selected config."""
        cfg = {"beam_width": 5, "max_depth": 4}
        ap, sel, result = self._make_planner(planner_cfg=cfg)

        with patch("planning.beam_search.BeamSearchPlanner") as MockBSP:
            MockBSP.return_value.search.return_value = result
            ap.search("goal", start_state={}, action_generator=lambda s: [])

        MockBSP.assert_called_once_with(ap._vn, beam_width=5, max_depth=4)

    def test_fallback_defaults_when_no_policy(self):
        """search() uses beam_width=3, max_depth=2 when no policy config available."""
        ap, sel, result = self._make_planner(
            planner_cfg={"beam_width": 3, "max_depth": 2},
            policy=None,
        )
        with patch("planning.beam_search.BeamSearchPlanner") as MockBSP:
            MockBSP.return_value.search.return_value = result
            ap.search("goal", start_state={}, action_generator=lambda s: [])

        MockBSP.assert_called_once_with(ap._vn, beam_width=3, max_depth=2)

    def test_reward_dispatched_after_search(self):
        """on_planner must be called with best_value when search succeeds."""
        reward_engine = MagicMock()
        policy = _make_policy("planner_config")
        result = _FakeBeamResult(best_value=0.92, runner_up_trajectories=["a", "b"])
        ap, sel, _ = self._make_planner(policy=policy, reward_engine=reward_engine,
                                        beam_result=result)

        with patch("planning.beam_search.BeamSearchPlanner") as MockBSP:
            MockBSP.return_value.search.return_value = result
            ap.search("goal", start_state={}, action_generator=lambda s: [])

        reward_engine.on_planner.assert_called_once()
        call_args = reward_engine.on_planner.call_args
        best_val = call_args[0][0] if call_args[0] else call_args[1].get("best_value")
        assert abs(best_val - 0.92) < 1e-9

    def test_reward_not_dispatched_when_best_value_none(self):
        """on_planner must NOT be called when result.best_value is None."""
        reward_engine = MagicMock()
        result = _FakeBeamResult(best_value=None)
        ap, sel, _ = self._make_planner(reward_engine=reward_engine, beam_result=result)

        with patch("planning.beam_search.BeamSearchPlanner") as MockBSP:
            MockBSP.return_value.search.return_value = result
            ap.search("goal", start_state={}, action_generator=lambda s: [])

        reward_engine.on_planner.assert_not_called()

    def test_no_crash_when_reward_engine_none(self):
        """reward_engine=None must not raise during search."""
        ap, sel, result = self._make_planner(reward_engine=None)
        with patch("planning.beam_search.BeamSearchPlanner") as MockBSP:
            MockBSP.return_value.search.return_value = result
            ap.search("goal", start_state={}, action_generator=lambda s: [])  # must not raise

    def test_current_policy_id_set_when_policy_selected(self):
        """_current_policy_id must be updated after select_one returns a policy."""
        policy = _make_policy("planner_config")
        ap, sel, result = self._make_planner(policy=policy)
        assert ap._current_policy_id is None

        with patch("planning.beam_search.BeamSearchPlanner") as MockBSP:
            MockBSP.return_value.search.return_value = result
            ap.search("goal", start_state={}, action_generator=lambda s: [])

        assert ap._current_policy_id == policy.policy_id

    def test_current_policy_id_stays_none_when_no_policy(self):
        ap, sel, result = self._make_planner(policy=None)
        with patch("planning.beam_search.BeamSearchPlanner") as MockBSP:
            MockBSP.return_value.search.return_value = result
            ap.search("goal", start_state={}, action_generator=lambda s: [])
        assert ap._current_policy_id is None

    def test_current_config_property(self):
        """current_config must return a dict."""
        ap, _, _ = self._make_planner()
        cfg = ap.current_config
        assert isinstance(cfg, dict)

    def test_reward_n_traj_counts_runner_ups(self):
        """on_planner n_traj arg must be len(runner_ups) + 1."""
        reward_engine = MagicMock()
        result = _FakeBeamResult(best_value=0.8, runner_up_trajectories=["x", "y", "z"])
        ap, sel, _ = self._make_planner(reward_engine=reward_engine, beam_result=result)

        with patch("planning.beam_search.BeamSearchPlanner") as MockBSP:
            MockBSP.return_value.search.return_value = result
            ap.search("goal", start_state={}, action_generator=lambda s: [])

        call_args = reward_engine.on_planner.call_args
        n_traj = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("n_traj")
        assert n_traj == 4, f"Expected 4 (3 runner-ups + 1), got {n_traj}"

    def test_beam_search_import_is_resolvable(self):
        """BeamSearchPlanner must be importable from the deferred import path."""
        from planning.beam_search import BeamSearchPlanner
        assert callable(BeamSearchPlanner)

    def test_reward_policy_id_passed_through(self):
        """on_planner must include the selected policy_id as kwarg."""
        policy = _make_policy("planner_config")
        reward_engine = MagicMock()
        result = _FakeBeamResult(best_value=0.7)
        ap, sel, _ = self._make_planner(policy=policy, reward_engine=reward_engine,
                                        beam_result=result)

        with patch("planning.beam_search.BeamSearchPlanner") as MockBSP:
            MockBSP.return_value.search.return_value = result
            ap.search("goal", start_state={}, action_generator=lambda s: [])

        call_kwargs = reward_engine.on_planner.call_args[1]
        assert call_kwargs.get("policy_id") == policy.policy_id, (
            "policy_id must be forwarded to RewardEngine.on_planner"
        )

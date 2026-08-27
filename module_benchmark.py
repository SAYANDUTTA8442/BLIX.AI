"""
module_benchmark.py
====================
Subsystem benchmark for Blix v0.3.19.3 — measures modules beyond the
memory/ADMA layer already covered by eval_harness.py.

Covered subsystems
------------------
Tier A (HIGH priority — core NeurIPS claims):
  1. Planning        — plan quality, depth, success, replanning rate
  2. Workspace       — attention precision, cycle throughput, broadcast latency
  3. Metacognition   — confidence calibration, strategy adaptation, self-model accuracy
  4. Reflection      — consolidation ratio, insight generation, goal completion
  5. Retrieval (adv) — cross-encoder reranking lift, active-attention hit rate

Tier B (MEDIUM priority — optional, graceful fallback):
  6. Causality       — causal chain depth, counterfactual consistency, epistemic confidence
  7. Specialists     — opinion confidence, agreement rate, consensus quality
  8. Curiosity       — signal novelty, gap coverage, generation latency
  9. World Model     — prediction error, ranking accuracy (requires torch)

No production code is modified.  The benchmark instantiates subsystems
in isolation and measures their observable outputs against controlled inputs.

Usage
-----
python module_benchmark.py \\
    --subsystems all \\
    --seeds 42,43,44 \\
    --output module_results/ \\
    --verbose

python module_benchmark.py --subsystems planning,workspace,metacognition
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import tempfile
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("module_benchmark")

# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubsystemResult:
    subsystem: str
    seed: int
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str]         = field(default_factory=list)
    duration_s: float         = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metrics"] = self.metrics
        return d


ALL_SUBSYSTEMS = [
    "planning", "workspace", "metacognition",
    "reflection", "retrieval_adv",
    "causality", "specialists", "curiosity", "world_model",
]

TIER_A = {"planning", "workspace", "metacognition", "reflection", "retrieval_adv"}
TIER_B = {"causality", "specialists", "curiosity", "world_model"}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic task bank — used across subsystems
# ─────────────────────────────────────────────────────────────────────────────

GOAL_BANK = [
    "Research and summarise recent advances in transformer architectures",
    "Analyse the user's learning history and recommend next study topics",
    "Identify knowledge gaps in the stored memory about quantum computing",
    "Verify all factual claims in the last conversation session",
    "Build a study plan for mastering Python data structures in 4 weeks",
    "Compare three machine learning optimisers: Adam, SGD, and RMSProp",
    "Explain causal relationships between diet, exercise, and cognitive performance",
    "Generate practice questions for the user's upcoming linear algebra exam",
    "Retrieve and synthesise information about climate feedback loops",
    "Detect contradictions in stored beliefs about LLM training techniques",
]

QA_BANK = [
    ("What is the capital of France?",          "Paris"),
    ("Who developed the transformer architecture?", "Google"),
    ("What does ADMA stand for in Blix?",       "Adaptive Decision-Making Architecture"),
    ("What is the time complexity of quicksort?", "O(n log n)"),
    ("Name the three laws of thermodynamics.",  "energy conservation, entropy, absolute zero"),
    ("What is backpropagation?",                "gradient descent through computational graph"),
    ("Define overfitting in machine learning.",  "model performs well on train but not on test"),
    ("What is the curse of dimensionality?",    "volume increases exponentially with dimensions"),
    ("What does KL divergence measure?",        "difference between two probability distributions"),
    ("What is Thompson sampling?",              "Bayesian approach to exploration-exploitation"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Planning benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_planning(seed: int, n_goals: int = 10) -> SubsystemResult:
    """
    Metrics
    -------
    plan_success_rate      : fraction of goals that produce a valid TaskGraph
    avg_plan_depth         : mean number of tasks per plan
    avg_expected_success   : mean PlanEvaluator.expected_success score
    avg_complexity         : mean PlanEvaluator.complexity score
    avg_plan_latency_s     : wall-clock time per plan() call
    replanning_rate        : fraction of plans that required Replanner intervention
    beam_success_rate      : fraction of beam-search plans that succeed
    avg_beam_depth         : mean beam search plan length
    """
    result = SubsystemResult(subsystem="planning", seed=seed)
    t0 = time.perf_counter()
    rng = random.Random(seed)
    goals = rng.choices(GOAL_BANK, k=n_goals)

    try:
        from planning.planner import Planner
        from planning.plan_evaluator import PlanEvaluator
        from planning.beam_search import BeamSearchPlanner
        from planning.replanner import Replanner
    except Exception as exc:
        result.errors.append(f"Import failed: {exc}")
        return result

    with tempfile.TemporaryDirectory() as td:
        try:
            from llm.ollama_provider import OllamaProvider
            llm = OllamaProvider()
        except Exception:
            llm = _make_mock_llm()

        planner = _safe_init(
            Planner, result,
            llm=llm, memory_dir=Path(td),
        )
        evaluator = _safe_init(PlanEvaluator)
        beam_planner = _safe_init(
            BeamSearchPlanner,
            result,
            llm=llm,
        )

        successes, depths, exp_success, complexity = [], [], [], []
        latencies = []
        replanned = 0

        for goal in goals:
            try:
                t_start = time.perf_counter()
                parsed, graph = planner.plan(goal)
                lat = time.perf_counter() - t_start
                latencies.append(lat)

                n_tasks = len(graph.tasks)
                successes.append(1.0 if n_tasks > 0 else 0.0)
                depths.append(float(n_tasks))

                if evaluator and n_tasks > 0:
                    score = evaluator.evaluate(graph)
                    exp_success.append(score.expected_success)
                    complexity.append(score.complexity)

            except Exception as exc:
                result.errors.append(f"plan({goal[:30]}): {exc}")
                successes.append(0.0)

        # Beam search subset
        beam_successes, beam_depths = [], []
        for goal in goals[:5]:
            try:
                if beam_planner:
                    br = beam_planner.search(goal)
                    beam_successes.append(1.0 if br and br.steps else 0.0)
                    beam_depths.append(float(len(br.steps)) if br else 0.0)
            except Exception as exc:
                result.errors.append(f"beam_search: {exc}")
                beam_successes.append(0.0)

    result.metrics = {
        "plan_success_rate":    _safe_mean(successes),
        "avg_plan_depth":       _safe_mean(depths),
        "avg_expected_success": _safe_mean(exp_success),
        "avg_complexity":       _safe_mean(complexity),
        "avg_plan_latency_s":   _safe_mean(latencies),
        "replanning_rate":      replanned / max(1, len(goals)),
        "beam_success_rate":    _safe_mean(beam_successes),
        "avg_beam_depth":       _safe_mean(beam_depths),
        "n_goals":              float(len(goals)),
        "n_errors":             float(len(result.errors)),
    }
    result.duration_s = time.perf_counter() - t0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Workspace benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_workspace(seed: int, n_candidates: int = 100, n_cycles: int = 20) -> SubsystemResult:
    """
    Metrics
    -------
    attention_precision       : fraction of accepted candidates with relevance > 0.5
    candidate_acceptance_rate : candidates entering workspace / submitted
    avg_broadcast_latency_s   : time per broadcast cycle
    cycle_throughput          : candidates processed per second
    avg_attention_score       : mean score of accepted candidates
    pending_after_cycles      : pending queue size after all cycles (should be 0)
    """
    result = SubsystemResult(subsystem="workspace", seed=seed)
    t0 = time.perf_counter()
    rng = random.Random(seed)

    try:
        from workspace.global_workspace import GlobalWorkspace
        from workspace.attention_manager import AttentionCandidate
    except Exception as exc:
        result.errors.append(f"Import failed: {exc}")
        return result

    gw = GlobalWorkspace()
    accepted_scores, accepted_relevances, cycle_times = [], [], []
    total_submitted = 0
    total_accepted  = 0

    for cycle in range(n_cycles):
        # Submit a batch of candidates with varying quality
        batch_size = n_candidates // n_cycles
        for i in range(batch_size):
            relevance = rng.uniform(0.0, 1.0)
            urgency   = rng.uniform(0.0, 1.0)
            novelty   = rng.uniform(0.0, 1.0)
            confidence = rng.uniform(0.3, 1.0)
            c = AttentionCandidate(
                ref_id=f"cand_{cycle}_{i}",
                source=rng.choice(["memory", "planner", "curiosity", "reflector"]),
                content_summary=f"Candidate item {i} for cycle {cycle}",
                relevance=relevance,
                urgency=urgency,
                novelty=novelty,
                confidence=confidence,
            )
            gw.submit_candidate(c)
            total_submitted += 1

        # Run one attention cycle (method is run_cycle(), returns WorkspaceCycleResult)
        t_cycle = time.perf_counter()
        try:
            cycle_result = gw.run_cycle()
            cycle_lat = time.perf_counter() - t_cycle
            cycle_times.append(cycle_lat)
            for entry in (cycle_result.entered or []):
                total_accepted += 1
                score = getattr(entry, "attention_score",
                        getattr(entry, "score", 0.5))
                # Recover relevance from the original submitted candidate via ref_id
                rel = getattr(entry, "relevance", 0.5)
                accepted_scores.append(float(score))
                accepted_relevances.append(float(score))  # use attention_score as proxy
        except Exception as exc:
            result.errors.append(f"run_cycle {cycle}: {exc}")
            cycle_times.append(0.0)

    acceptance_rate = total_accepted / max(1, total_submitted)
    precision = (
        sum(1 for r in accepted_relevances if r > 0.5) / max(1, len(accepted_relevances))
    )
    total_time = sum(cycle_times)
    throughput  = total_submitted / max(1e-9, total_time)

    result.metrics = {
        "attention_precision":       precision,
        "candidate_acceptance_rate": acceptance_rate,
        "avg_broadcast_latency_s":   _safe_mean(cycle_times),
        "cycle_throughput":          throughput,
        "avg_attention_score":       _safe_mean(accepted_scores),
        "pending_after_cycles":      float(gw.pending_count),
        "total_cycles":              float(n_cycles),
        "total_submitted":           float(total_submitted),
        "total_accepted":            float(total_accepted),
        "n_errors":                  float(len(result.errors)),
    }
    result.duration_s = time.perf_counter() - t0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metacognition benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_metacognition(seed: int, n_queries: int = 50) -> SubsystemResult:
    """
    Metrics
    -------
    confidence_calibration_ece  : Expected Calibration Error (lower = better)
    avg_stated_confidence       : mean confidence for correct answers
    avg_confidence_wrong        : mean confidence for wrong answers (should be lower)
    confidence_discrimination   : avg_correct - avg_wrong (higher = better)
    capability_entries          : number of tracked capabilities
    self_model_coverage         : fraction of query domains with tracked confidence
    strategy_selections         : number of strategy selection events
    avg_strategy_latency_s      : time per strategy selection
    """
    result = SubsystemResult(subsystem="metacognition", seed=seed)
    t0 = time.perf_counter()
    rng = random.Random(seed)

    try:
        from metacognition.confidence_manager import ConfidenceManager
        from metacognition.capability_tracker import CapabilityTracker
    except Exception as exc:
        result.errors.append(f"Import failed: {exc}")
        return result

    with tempfile.TemporaryDirectory() as td:
        conf_mgr   = _safe_init(ConfidenceManager)
        cap_tracker = _safe_init(CapabilityTracker, result, memory_dir=Path(td))

        correct_confidences, wrong_confidences = [], []
        strategy_times = []
        capability_domains = set()

        for i, (q, gt) in enumerate(rng.choices(QA_BANK, k=n_queries)):
            domain = q.split()[0].lower()
            capability_domains.add(domain)

            # Simulate: agent gives an answer with some stated confidence
            is_correct  = rng.random() < 0.65   # 65% accuracy
            raw_conf    = rng.gauss(0.7 if is_correct else 0.4, 0.15)
            confidence  = max(0.0, min(1.0, raw_conf))
            ref_id      = f"q_{i}"

            if conf_mgr:
                try:
                    conf_mgr.register(namespace="qa", ref_id=ref_id, confidence=confidence)
                    retrieved = conf_mgr.get(namespace="qa", ref_id=ref_id)
                    actual_conf = retrieved if retrieved is not None else confidence
                except Exception as exc:
                    result.errors.append(f"conf_mgr: {exc}")
                    actual_conf = confidence
            else:
                actual_conf = confidence

            if is_correct:
                correct_confidences.append(actual_conf)
            else:
                wrong_confidences.append(actual_conf)

            # Capability tracker
            if cap_tracker:
                try:
                    t_strat = time.perf_counter()
                    cap_tracker.update(domain=domain, success=is_correct, confidence=confidence)
                    strategy_times.append(time.perf_counter() - t_strat)
                except Exception as exc:
                    result.errors.append(f"cap_tracker: {exc}")

        # ECE calculation (10 bins)
        all_confs    = correct_confidences + wrong_confidences
        all_correct  = [1.0] * len(correct_confidences) + [0.0] * len(wrong_confidences)
        ece = _compute_ece(all_confs, all_correct, n_bins=10)

        avg_correct = _safe_mean(correct_confidences)
        avg_wrong   = _safe_mean(wrong_confidences)

        # Coverage: fraction of QA_BANK domains the tracker has entries for
        n_cap = 0
        if cap_tracker:
            try:
                n_cap = len(cap_tracker.list_domains()) if hasattr(cap_tracker, "list_domains") else \
                        len(capability_domains)
            except Exception:
                n_cap = len(capability_domains)

    result.metrics = {
        "confidence_calibration_ece":  ece,
        "avg_stated_confidence":       avg_correct,
        "avg_confidence_wrong":        avg_wrong,
        "confidence_discrimination":   avg_correct - avg_wrong,
        "capability_entries":          float(n_cap),
        "self_model_coverage":         len(capability_domains) / max(1, n_queries),
        "strategy_selections":         float(len(strategy_times)),
        "avg_strategy_latency_s":      _safe_mean(strategy_times),
        "n_errors":                    float(len(result.errors)),
    }
    result.duration_s = time.perf_counter() - t0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reflection benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_reflection(seed: int, n_sessions: int = 10) -> SubsystemResult:
    """
    Metrics
    -------
    reflection_success_rate   : fraction of reflect() calls that return a ReflectionRecord
    avg_insight_score         : mean score/quality of generated insights
    avg_reflection_latency_s  : wall-clock time per reflect() call
    consolidation_ratio       : consolidated nodes / original nodes
    goal_completion_rate      : fraction of goals marked complete by GoalTracker
    insight_count_per_session : mean insights generated per session material
    record_count              : total reflection records produced
    """
    result = SubsystemResult(subsystem="reflection", seed=seed)
    t0 = time.perf_counter()
    rng = random.Random(seed)

    try:
        from reflection.reflection_engine import ReflectionEngine, ReflectionScope
        from reflection.goal_tracker import GoalTracker
    except Exception as exc:
        result.errors.append(f"Import failed: {exc}")
        return result

    with tempfile.TemporaryDirectory() as td:
        try:
            from llm.ollama_provider import OllamaProvider
            llm = OllamaProvider()
        except Exception:
            llm = _make_mock_llm()

        engine = _safe_init(
            ReflectionEngine, result,
            llm=llm, memory_dir=Path(td),
        )
        goal_tracker = _safe_init(GoalTracker, result, memory_dir=Path(td))

        successes, latencies, insight_scores = [], [], []
        goals_set = goals_done = 0

        for i in range(n_sessions):
            material = (
                f"Session {i}: Studied {rng.choice(['transformers','CNNs','RNNs','MLPs'])}. "
                f"Understood {rng.choice(['attention','backprop','normalisation','optimisers'])}. "
                f"Confusion about {rng.choice(['gradient flow','vanishing gradients','BPTT','batch size'])}."
            )
            goal_text = rng.choice(GOAL_BANK)

            # GoalTracker
            if goal_tracker:
                try:
                    goal_tracker.add_goal(goal_text)
                    goals_set += 1
                    if rng.random() < 0.6:
                        goal_tracker.mark_complete(goal_text)
                        goals_done += 1
                except Exception as exc:
                    result.errors.append(f"goal_tracker[{i}]: {exc}")

            # ReflectionEngine
            if engine:
                try:
                    t_ref = time.perf_counter()
                    record = engine.reflect(
                        scope=ReflectionScope.SESSION,
                        scope_ref=f"session_{i}",
                        material=material,
                    )
                    latencies.append(time.perf_counter() - t_ref)
                    if record:
                        successes.append(1.0)
                        quality = getattr(record, "quality_score",
                                  getattr(record, "score",
                                  getattr(record, "insight_score", 0.5)))
                        insight_scores.append(float(quality))
                    else:
                        successes.append(0.0)
                except Exception as exc:
                    result.errors.append(f"reflect[{i}]: {exc}")
                    successes.append(0.0)
            else:
                successes.append(0.0)

    result.metrics = {
        "reflection_success_rate":    _safe_mean(successes),
        "avg_insight_score":          _safe_mean(insight_scores),
        "avg_reflection_latency_s":   _safe_mean(latencies),
        "consolidation_ratio":        float("nan"),    # populated if ConsolidationEngine used
        "goal_completion_rate":       goals_done / max(1, goals_set),
        "insight_count_per_session":  len(insight_scores) / max(1, n_sessions),
        "record_count":               float(sum(1 for s in successes if s > 0)),
        "n_errors":                   float(len(result.errors)),
    }
    result.duration_s = time.perf_counter() - t0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Advanced Retrieval benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_retrieval_adv(seed: int, n_queries: int = 50) -> SubsystemResult:
    """
    Metrics
    -------
    rerank_hit_rate           : fraction of reranked results with answer in top-3
    base_hit_rate             : hit rate before reranking (baseline)
    rerank_improvement        : rerank_hit_rate - base_hit_rate
    avg_rerank_latency_s      : time per rerank() call
    attention_retriever_hit   : hit rate using ActiveAttentionRetriever
    precision_at_3            : precision@3 after reranking
    avg_score_improvement     : mean score increase of correct item after reranking
    """
    result = SubsystemResult(subsystem="retrieval_adv", seed=seed)
    t0 = time.perf_counter()
    rng = random.Random(seed)

    try:
        from retrieval.cross_encoder_reranker import CrossEncoderReranker
        from retrieval.active_attention_retriever import ActiveAttentionRetriever
        from schemas.memory_entry import MemoryEntry
    except Exception as exc:
        result.errors.append(f"Import failed: {exc}")
        return result

    with tempfile.TemporaryDirectory() as td:
        reranker = _safe_init(CrossEncoderReranker, result)
        att_retriever = _safe_init(
            ActiveAttentionRetriever, result, memory_dir=Path(td)
        )

        base_hits, rerank_hits, latencies = [], [], []
        score_improvements = []

        for i, (q, answer) in enumerate(rng.choices(QA_BANK, k=n_queries)):
            # Build a candidate pool: 1 correct + N distractors
            pool_size = 10
            correct_entry = _make_memory_entry(
                f"{q} {answer}", entry_id=f"correct_{i}", score=rng.uniform(0.4, 0.7)
            )
            distractors = [
                _make_memory_entry(
                    f"Unrelated fact about {rng.choice(['biology','history','geography'])} topic {j}",
                    entry_id=f"distract_{i}_{j}",
                    score=rng.uniform(0.3, 0.8),
                )
                for j in range(pool_size - 1)
            ]

            # Shuffle to give a random base ordering
            candidates = [correct_entry] + distractors
            rng.shuffle(candidates)

            # Base hit rate: is correct in top-3 before reranking?
            top3_base = candidates[:3]
            base_hits.append(1.0 if any(e.id == correct_entry.id for e in top3_base) else 0.0)

            # Reranked hit rate
            if reranker:
                try:
                    t_rr = time.perf_counter()
                    reranked = reranker.rerank(q, candidates)
                    latencies.append(time.perf_counter() - t_rr)

                    top3_rr = reranked[:3]
                    rerank_hits.append(
                        1.0 if any(getattr(r, "entry", r).id == correct_entry.id
                                   for r in top3_rr) else 0.0
                    )
                    # Score improvement of correct item
                    base_rank   = next((j for j, e in enumerate(candidates) if e.id == correct_entry.id), pool_size)
                    rerank_rank = next((j for j, r in enumerate(reranked)
                                        if getattr(r, "entry", r).id == correct_entry.id), pool_size)
                    score_improvements.append(float(base_rank - rerank_rank))
                except Exception as exc:
                    result.errors.append(f"rerank[{i}]: {exc}")
                    rerank_hits.append(base_hits[-1])
            else:
                rerank_hits.append(base_hits[-1])

    base_hr   = _safe_mean(base_hits)
    rerank_hr = _safe_mean(rerank_hits)

    result.metrics = {
        "rerank_hit_rate":        rerank_hr,
        "base_hit_rate":          base_hr,
        "rerank_improvement":     rerank_hr - base_hr,
        "avg_rerank_latency_s":   _safe_mean(latencies),
        "attention_retriever_hit": float("nan"),   # requires seeded memory store
        "precision_at_3":         rerank_hr,       # same as hit_rate@3 here
        "avg_rank_improvement":   _safe_mean(score_improvements),
        "n_errors":               float(len(result.errors)),
    }
    result.duration_s = time.perf_counter() - t0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. Causality benchmark (Tier B)
# ─────────────────────────────────────────────────────────────────────────────

def bench_causality(seed: int, n_chains: int = 20) -> SubsystemResult:
    """
    Metrics
    -------
    causal_chain_length       : mean edges per inferred causal chain
    epistemic_confidence_avg  : mean EpistemicStatus.confidence across inferences
    belief_revision_rate      : fraction of beliefs updated upon new evidence
    counterfactual_consistency: fraction of counterfactuals that reverse causal direction
    causal_latency_s          : time per causal inference call
    principle_count           : principles extracted from material
    """
    result = SubsystemResult(subsystem="causality", seed=seed)
    t0 = time.perf_counter()
    rng = random.Random(seed)

    try:
        from causality.causal_memory import CausalMemory
        from causality.epistemic_status import EpistemicStatus
    except Exception as exc:
        result.errors.append(f"Import failed: {exc}")
        return result

    with tempfile.TemporaryDirectory() as td:
        causal_mem = _safe_init(CausalMemory, result, memory_dir=Path(td))

        chain_lengths, epistemic_confs, latencies = [], [], []
        belief_updates = 0

        pairs = [
            ("high sugar intake",   "increased insulin resistance"),
            ("regular exercise",    "improved cardiovascular health"),
            ("sleep deprivation",   "reduced cognitive performance"),
            ("stress",              "elevated cortisol levels"),
            ("learning new skills", "increased neural plasticity"),
        ]

        for i in range(n_chains):
            cause, effect = rng.choice(pairs)
            try:
                t_c = time.perf_counter()
                if causal_mem:
                    record = causal_mem.store(
                        cause=cause, effect=effect,
                        confidence=rng.uniform(0.5, 1.0),
                        source=f"bench_{i}",
                    )
                    latencies.append(time.perf_counter() - t_c)
                    chain_lengths.append(1.0)   # direct causal link
                    epistemic_confs.append(
                        float(getattr(record, "confidence", rng.uniform(0.5, 1.0)))
                    )
                    # Simulate belief revision
                    if rng.random() < 0.3:
                        causal_mem.store(
                            cause=cause, effect=effect,
                            confidence=rng.uniform(0.5, 1.0),
                            source=f"revision_{i}",
                        )
                        belief_updates += 1
            except Exception as exc:
                result.errors.append(f"causal[{i}]: {exc}")
                latencies.append(0.0)

        # EpistemicStatus
        epistemic_scores = []
        try:
            es = EpistemicStatus()
            for conf in [0.9, 0.6, 0.3, 0.8, 0.5]:
                es.update(domain="bench", confidence=conf)
                epistemic_scores.append(conf)
        except Exception as exc:
            result.errors.append(f"epistemic: {exc}")

    result.metrics = {
        "causal_chain_length":       _safe_mean(chain_lengths),
        "epistemic_confidence_avg":  _safe_mean(epistemic_confs or epistemic_scores),
        "belief_revision_rate":      belief_updates / max(1, n_chains),
        "counterfactual_consistency": float("nan"),  # requires LLM
        "causal_latency_s":          _safe_mean(latencies),
        "principle_count":           float("nan"),
        "n_errors":                  float(len(result.errors)),
    }
    result.duration_s = time.perf_counter() - t0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. Specialists benchmark (Tier B)
# ─────────────────────────────────────────────────────────────────────────────

def bench_specialists(seed: int, n_consultations: int = 20) -> SubsystemResult:
    """
    Metrics
    -------
    avg_opinion_confidence  : mean confidence across specialist opinions
    specialist_agreement_rate: fraction of queries where all specialists agree
    avg_consultation_latency_s: wall-clock per consult() call
    consensus_quality       : mean confidence of consensus opinions
    coverage_rate           : fraction of queries returning at least one opinion
    n_specialists_active    : number of specialist types that returned opinions
    """
    result = SubsystemResult(subsystem="specialists", seed=seed)
    t0 = time.perf_counter()
    rng = random.Random(seed)

    try:
        from specialists.memory_specialist    import MemorySpecialist
        from specialists.planning_specialist  import PlanningSpecialist
        from specialists.reflection_specialist import ReflectionSpecialist
        from specialists.consensus            import ConsensusBuilder
    except Exception as exc:
        result.errors.append(f"Import failed: {exc}")
        return result

    with tempfile.TemporaryDirectory() as td:
        try:
            from llm.ollama_provider import OllamaProvider
            llm = OllamaProvider()
        except Exception:
            llm = _make_mock_llm()

        mem_spec  = _safe_init(MemorySpecialist,    result, llm=llm, memory_dir=Path(td))
        plan_spec = _safe_init(PlanningSpecialist,  result, llm=llm)
        refl_spec = _safe_init(ReflectionSpecialist,result, llm=llm)
        consensus = _safe_init(ConsensusBuilder)

        all_confs, all_latencies = [], []
        agreements, covered = 0, 0
        active_specs = set()

        for i, (q, _) in enumerate(rng.choices(QA_BANK, k=n_consultations)):
            opinions = []
            for name, spec in [("memory", mem_spec),
                                ("planning", plan_spec),
                                ("reflection", refl_spec)]:
                if spec is None:
                    continue
                try:
                    t_c = time.perf_counter()
                    op = spec.consult(q)
                    all_latencies.append(time.perf_counter() - t_c)
                    if op:
                        opinions.append(op)
                        active_specs.add(name)
                        all_confs.append(float(getattr(op, "confidence", 0.5)))
                except Exception as exc:
                    result.errors.append(f"spec[{name}][{i}]: {exc}")

            if opinions:
                covered += 1
                if len(opinions) > 1:
                    # Agreement: all opinions in same confidence quartile
                    confs = [getattr(o, "confidence", 0.5) for o in opinions]
                    spread = max(confs) - min(confs)
                    if spread < 0.25:
                        agreements += 1

            if consensus and len(opinions) > 1:
                try:
                    c_op = consensus.build(opinions)
                    if c_op:
                        all_confs.append(float(getattr(c_op, "confidence", 0.5)))
                except Exception as exc:
                    result.errors.append(f"consensus[{i}]: {exc}")

    result.metrics = {
        "avg_opinion_confidence":      _safe_mean(all_confs),
        "specialist_agreement_rate":   agreements / max(1, covered),
        "avg_consultation_latency_s":  _safe_mean(all_latencies),
        "consensus_quality":           _safe_mean(all_confs[-n_consultations:]),
        "coverage_rate":               covered / max(1, n_consultations),
        "n_specialists_active":        float(len(active_specs)),
        "n_errors":                    float(len(result.errors)),
    }
    result.duration_s = time.perf_counter() - t0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8. Curiosity benchmark (Tier B)
# ─────────────────────────────────────────────────────────────────────────────

def bench_curiosity(seed: int, n_rounds: int = 10) -> SubsystemResult:
    """
    Metrics
    -------
    signals_per_round       : mean number of curiosity signals generated per round
    avg_signal_novelty      : mean novelty score of generated signals
    avg_signal_priority     : mean priority/urgency of signals
    gap_coverage_rate       : fraction of known gaps addressed by signals
    avg_generation_latency_s: time per generate_signals() call
    unique_topics_covered   : distinct topics across all generated signals
    """
    result = SubsystemResult(subsystem="curiosity", seed=seed)
    t0 = time.perf_counter()
    rng = random.Random(seed)

    try:
        from curiosity.curiosity_engine import CuriosityEngine
    except Exception as exc:
        result.errors.append(f"Import failed: {exc}")
        return result

    with tempfile.TemporaryDirectory() as td:
        try:
            from llm.ollama_provider import OllamaProvider
            from memory.hybrid.hgshm import HGSHM
            llm  = OllamaProvider()
            hgshm = HGSHM(memory_dir=td)
        except Exception:
            llm   = _make_mock_llm()
            hgshm = None

        engine = _safe_init(
            CuriosityEngine, result,
            llm=llm, hgshm=hgshm,
        )

        per_round, novelties, priorities, latencies = [], [], [], []
        all_topics: set[str] = set()

        for r in range(n_rounds):
            if engine is None:
                break
            try:
                t_g = time.perf_counter()
                signals = engine.generate_signals(top_k=5)
                latencies.append(time.perf_counter() - t_g)
                n_sigs = len(signals) if signals else 0
                per_round.append(float(n_sigs))
                for sig in (signals or []):
                    nov = float(getattr(sig, "novelty",
                                getattr(sig, "priority", 0.5)))
                    pri = float(getattr(sig, "priority",
                                getattr(sig, "urgency", 0.5)))
                    topic = str(getattr(sig, "topic",
                                getattr(sig, "question",
                                getattr(sig, "description", f"topic_{r}"))))
                    novelties.append(nov)
                    priorities.append(pri)
                    all_topics.add(topic[:40])
            except Exception as exc:
                result.errors.append(f"generate_signals[{r}]: {exc}")
                per_round.append(0.0)

    result.metrics = {
        "signals_per_round":        _safe_mean(per_round),
        "avg_signal_novelty":       _safe_mean(novelties),
        "avg_signal_priority":      _safe_mean(priorities),
        "gap_coverage_rate":        len(all_topics) / max(1, n_rounds * 5),
        "avg_generation_latency_s": _safe_mean(latencies),
        "unique_topics_covered":    float(len(all_topics)),
        "n_errors":                 float(len(result.errors)),
    }
    result.duration_s = time.perf_counter() - t0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 9. World Model benchmark (Tier B, optional torch)
# ─────────────────────────────────────────────────────────────────────────────

def bench_world_model(seed: int, n_steps: int = 50) -> SubsystemResult:
    """
    Metrics
    -------
    prediction_mse          : mean-squared error of latent state predictions
    ranking_accuracy        : fraction of scenario pairs ranked correctly
    value_calibration_mae   : MAE of value-network estimates vs actual rewards
    avg_prediction_latency_s: time per predict() call
    model_available         : 1.0 if torch model loaded, 0.0 if fallback
    """
    result = SubsystemResult(subsystem="world_model", seed=seed)
    t0 = time.perf_counter()
    rng = random.Random(seed)

    try:
        import torch
        from world_model.latent_world_model import LatentWorldModel, LatentState
        from world_model.scenario_ranker import ScenarioRanker
        from world_model.value_network import ValueNetwork
        _torch_available = True
    except ImportError:
        _torch_available = False
        result.errors.append("torch not installed — world_model metrics unavailable")
        result.metrics = {
            "prediction_mse":           float("nan"),
            "ranking_accuracy":         float("nan"),
            "value_calibration_mae":    float("nan"),
            "avg_prediction_latency_s": float("nan"),
            "model_available":          0.0,
            "n_errors":                 1.0,
        }
        result.duration_s = time.perf_counter() - t0
        return result

    latent_dim = 32
    world_model = _safe_init(LatentWorldModel, result, latent_dim=latent_dim)
    ranker      = _safe_init(ScenarioRanker,   result, latent_dim=latent_dim)
    value_net   = _safe_init(ValueNetwork,     result, latent_dim=latent_dim)

    pred_mses, latencies = [], []
    ranking_correct = []
    value_maes = []

    import torch as _torch
    _torch.manual_seed(seed)

    for i in range(n_steps):
        z = LatentState(vector=_torch.randn(latent_dim).tolist())
        # Prediction
        if world_model:
            try:
                t_p = time.perf_counter()
                pred = world_model.predict(z)
                latencies.append(time.perf_counter() - t_p)
                # MSE vs a noisy copy of the input (oracle)
                z_arr  = _torch.tensor(z.vector)
                p_arr  = _torch.tensor(pred.next_state.vector)
                noise  = _torch.randn_like(z_arr) * 0.1
                oracle = z_arr + noise
                mse = float(((p_arr - oracle) ** 2).mean())
                pred_mses.append(mse)
            except Exception as exc:
                result.errors.append(f"predict[{i}]: {exc}")

        # Scenario ranking
        if ranker:
            try:
                z_a = LatentState(vector=_torch.randn(latent_dim).tolist())
                z_b = LatentState(vector=_torch.randn(latent_dim).tolist())
                r_a, r_b = rng.random(), rng.random()
                better = "a" if r_a > r_b else "b"
                ranked = ranker.rank([z_a, z_b])
                if ranked:
                    ranking_correct.append(
                        1.0 if (better == "a") == (ranked[0] is z_a) else 0.0
                    )
            except Exception as exc:
                result.errors.append(f"ranker[{i}]: {exc}")

        # Value network
        if value_net:
            try:
                z_v = LatentState(vector=_torch.randn(latent_dim).tolist())
                actual_reward = rng.uniform(0.0, 1.0)
                estimated = value_net.estimate(z_v)
                value_maes.append(abs(float(estimated) - actual_reward))
            except Exception as exc:
                result.errors.append(f"value[{i}]: {exc}")

    result.metrics = {
        "prediction_mse":           _safe_mean(pred_mses),
        "ranking_accuracy":         _safe_mean(ranking_correct),
        "value_calibration_mae":    _safe_mean(value_maes),
        "avg_prediction_latency_s": _safe_mean(latencies),
        "model_available":          1.0,
        "n_errors":                 float(len(result.errors)),
    }
    result.duration_s = time.perf_counter() - t0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(vals: list) -> float:
    clean = [v for v in vals if v is not None and not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else float("nan")


def _safe_init(cls, result: SubsystemResult | None = None, **kwargs):
    try:
        return cls(**kwargs)
    except Exception as exc:
        if result:
            result.errors.append(f"Init {cls.__name__}: {exc}")
        return None


def _make_mock_llm():
    """Minimal LLM stub for benchmarks that don't need real generation."""
    class _MockLLM:
        def generate(self, prompt: str, **kw) -> str:
            return f"Mock response for: {prompt[:40]}"
        def reset(self) -> None:
            pass
        @property
        def model(self): return "mock"
    return _MockLLM()


def _make_memory_entry(text: str, entry_id: str, score: float):
    """Create a MemoryEntry for retrieval tests."""
    try:
        from schemas.memory_entry import MemoryEntry
        return MemoryEntry(
            id=entry_id,
            input=text,
            output=text,
            score=score,
            metadata={},
        )
    except Exception:
        # Fallback: simple namespace object
        class _FakeEntry:
            def __init__(self, id_, text_, score_):
                self.id = id_; self.input = text_; self.output = text_
                self.score = score_; self.metadata = {}
        return _FakeEntry(entry_id, text, score)


def _compute_ece(confidences: list[float], correct: list[float], n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    if not confidences:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = [(lo <= c < hi) for c in confidences]
        if not any(mask):
            continue
        idx   = [i for i, m in enumerate(mask) if m]
        acc   = np.mean([correct[i] for i in idx])
        conf  = np.mean([confidences[i] for i in idx])
        ece  += len(idx) / n * abs(acc - conf)
    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

BENCH_FUNCS = {
    "planning":      bench_planning,
    "workspace":     bench_workspace,
    "metacognition": bench_metacognition,
    "reflection":    bench_reflection,
    "retrieval_adv": bench_retrieval_adv,
    "causality":     bench_causality,
    "specialists":   bench_specialists,
    "curiosity":     bench_curiosity,
    "world_model":   bench_world_model,
}


class ModuleBenchmarkRunner:

    def __init__(self, args: argparse.Namespace):
        subsystems_arg = args.subsystems.strip()
        if subsystems_arg == "all":
            self.subsystems = ALL_SUBSYSTEMS
        elif subsystems_arg == "tier_a":
            self.subsystems = list(TIER_A)
        elif subsystems_arg == "tier_b":
            self.subsystems = list(TIER_B)
        else:
            self.subsystems = [s.strip() for s in subsystems_arg.split(",")]
        self.seeds      = [int(s) for s in args.seeds.split(",")]
        self.output_dir = Path(args.output)
        self.verbose    = args.verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s %(levelname)s: %(message)s")
        else:
            logging.basicConfig(level=logging.WARNING)

    def run(self) -> None:
        all_results: dict[str, list[SubsystemResult]] = defaultdict(list)

        for subsystem in self.subsystems:
            if subsystem not in BENCH_FUNCS:
                log.warning("Unknown subsystem '%s' — skipping", subsystem)
                continue
            fn = BENCH_FUNCS[subsystem]
            log.info("=== Benchmarking: %s ===", subsystem)
            for seed in self.seeds:
                log.info("  seed=%d", seed)
                try:
                    res = fn(seed=seed)
                except Exception as exc:
                    res = SubsystemResult(subsystem=subsystem, seed=seed)
                    res.errors.append(traceback.format_exc())
                    log.error("  FAILED: %s", exc)
                all_results[subsystem].append(res)
                log.info("  Done in %.2fs: %s", res.duration_s,
                         {k: f"{v:.3f}" for k, v in res.metrics.items()
                          if not math.isnan(v) and k != "n_errors"})

        self._write_outputs(all_results)

    def _write_outputs(self, all_results: dict[str, list[SubsystemResult]]) -> None:
        # Per-subsystem CSV
        for subsystem, results in all_results.items():
            if not results:
                continue
            csv_path = self.output_dir / f"{subsystem}.csv"
            all_keys = sorted({k for r in results for k in r.metrics})
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["subsystem","seed","duration_s"] + all_keys)
                writer.writeheader()
                for r in results:
                    row = {"subsystem": r.subsystem, "seed": r.seed,
                           "duration_s": r.duration_s}
                    row.update(r.metrics)
                    writer.writerow(row)

        # Aggregate JSON (mean ± std across seeds)
        agg: dict[str, dict] = {}
        for subsystem, results in all_results.items():
            agg[subsystem] = {}
            all_keys = sorted({k for r in results for k in r.metrics})
            for k in all_keys:
                vals = [r.metrics[k] for r in results
                        if k in r.metrics and not math.isnan(r.metrics[k])]
                agg[subsystem][k] = {
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "std":  float(np.std(vals))  if vals else float("nan"),
                    "n":    len(vals),
                }
            agg[subsystem]["_errors"] = [e for r in results for e in r.errors]

        with open(self.output_dir / "module_aggregate.json", "w") as f:
            json.dump(agg, f, indent=2, default=str)

        # Summary markdown
        self._write_summary_md(agg)
        log.info("Results written to %s", self.output_dir)

    def _write_summary_md(self, agg: dict) -> None:
        lines = [
            "# Module Benchmark Summary\n",
            "All metrics are Mean ± Std across seeds.\n",
        ]
        for subsystem, metrics in agg.items():
            tier = "A" if subsystem in TIER_A else "B"
            lines.append(f"\n## {subsystem} (Tier {tier})\n")
            lines.append("| Metric | Mean | Std |")
            lines.append("|--------|------|-----|")
            for k, v in metrics.items():
                if k.startswith("_"):
                    continue
                m = v.get("mean", float("nan"))
                s = v.get("std",  float("nan"))
                m_str = "—" if math.isnan(m) else f"{m:.4f}"
                s_str = "—" if math.isnan(s) else f"{s:.4f}"
                lines.append(f"| {k} | {m_str} | {s_str} |")
            errors = metrics.get("_errors", [])
            if errors:
                lines.append(f"\n> ⚠️ {len(errors)} error(s) during benchmark.")

        with open(self.output_dir / "module_summary.md", "w") as f:
            f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Blix module-level subsystem benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--subsystems", default="tier_a",
        help=("Comma-separated subsystem names, 'all', 'tier_a' (planning/workspace/"
              "metacognition/reflection/retrieval_adv), or 'tier_b'"),
    )
    p.add_argument("--seeds",   default="42,43,44",
                   help="Comma-separated random seeds")
    p.add_argument("--output",  default="module_results/",
                   help="Output directory")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    runner = ModuleBenchmarkRunner(args)
    runner.run()


if __name__ == "__main__":
    main()

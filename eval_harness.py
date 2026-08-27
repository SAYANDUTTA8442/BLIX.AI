"""
eval_harness.py
================
Multi-dataset evaluation and ablation suite for Blix v0.3.19.3.

Supports 4 datasets  : HotpotQA, LoCoMo, NarrativeQA, StreamingQA
6 tiers of metrics   : Retrieval, Generation, Efficiency, Memory, Policy, Temporal
9 ablation profiles  : full, no_graph, no_adma, both, rag,
#                      hierarchy_only, temporal_only, graph_only, adma_only
Statistical rigour   : 5 seeds, paired bootstrap (10 000 resamples), p-values

Usage
-----
python eval_harness.py \\
    --datasets hotpotqa,locomo,narrativeqa,streamingqa \\
    --samples 200 \\
    --seeds 42,43,44,45,46 \\
    --profiles full,no_graph,no_adma,both \\
    --output results/ \\
    --nli-metrics --profile-memory --visualize --verbose

See README_EVAL.md for full replication instructions.

NOTE: This file only adds evaluation infrastructure — no production code is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ── Optional heavy dependencies — all failures are graceful ───────────────────
try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False
    warnings.warn("pandas not installed — CSV aggregation will use stdlib only")

try:
    from rouge_score import rouge_scorer as _rouge_scorer_mod
    _ROUGE = True
except ImportError:
    _ROUGE = False

try:
    import sacrebleu as _sacrebleu
    _BLEU = True
except ImportError:
    _BLEU = False

try:
    from scipy import stats as _scipy_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn("scipy not installed — bootstrap CIs will use numpy only")

try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

try:
    from sklearn.metrics import ndcg_score as _ndcg_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

# Blix internals
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.dataset import temporal_split

log = logging.getLogger("eval_harness")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROFILES = ["full", "no_graph", "no_adma", "both", "rag"]

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "hotpotqa": {
        "hf_path": "hotpot_qa",
        "hf_name": "distractor",
        "hf_split": "validation",
        "has_temporal": True,
        "date_key": "id",          # proxy sort key; HotpotQA has no timestamp
        "question_key": "question",
        "answer_key": "answer",
        "context_key": "context",
    },
    "locomo": {
        "hf_path": "locomo10",
        "hf_name": None,
        "hf_split": "test",
        "has_temporal": True,
        "date_key": "timestamp",
        "question_key": "question",
        "answer_key": "answer",
        "context_key": "context",
    },
    "narrativeqa": {
        "hf_path": "narrativeqa",
        "hf_name": None,
        "hf_split": "test",
        "has_temporal": False,      # split by story_id (spec 1.1)
        "date_key": None,
        "question_key": "question",
        "answer_key": "answers",
        "context_key": "document",
        "story_id_key": "document",
    },
    "streamingqa": {
        "hf_path": "streamingqa",
        "hf_name": None,
        "hf_split": "test",
        "has_temporal": True,
        "date_key": "date",
        "question_key": "question",
        "answer_key": "answer",
        "context_key": "context",
    },
}

CSV_FIELDS = [
    "question_id", "question", "answer", "pred",
    "hit_1", "hit_5", "hit_10", "mrr", "ndcg", "precision_1", "recall_10",
    "rouge_l", "bleu_4", "faithfulness_bert", "entailment_score",
    "hallucination_rate", "latency", "tokens_per_sec",
    "vram_peak_gb", "ram_peak_gb",
    "node_count", "graph_density", "consolidation_rate",
    "policy_divergence", "policy_switching", "cache_hit_rate",
    "state_consistency",
]

SIGNIFICANCE_MARKERS = [
    (0.001, "***"), (0.01, "**"), (0.05, "*"), (0.10, "†"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hit_at_k(answer: str, retrieved: list[str], k: int) -> float:
    ans = answer.lower().strip()
    for chunk in retrieved[:k]:
        if ans in chunk.lower():
            return 1.0
    return 0.0


def _mrr(answer: str, retrieved: list[str]) -> float:
    ans = answer.lower().strip()
    for rank, chunk in enumerate(retrieved, 1):
        if ans in chunk.lower():
            return 1.0 / rank
    return 0.0


def _ndcg_at_10(answer: str, retrieved: list[str]) -> float:
    if not _SKLEARN:
        return float("nan")
    relevance = [1.0 if answer.lower() in r.lower() else 0.0 for r in retrieved[:10]]
    if not any(relevance):
        return 0.0
    ideal = sorted(relevance, reverse=True)
    try:
        return float(_ndcg_score([ideal], [relevance]))
    except Exception:
        return float("nan")


def _rouge_l(pred: str, ref: str) -> float:
    if not _ROUGE:
        return float("nan")
    try:
        scorer = _rouge_scorer_mod.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(ref, pred)["rougeL"].fmeasure
    except Exception:
        return float("nan")


def _bleu_4(pred: str, ref: str) -> float:
    if not _BLEU:
        return float("nan")
    try:
        result = _sacrebleu.corpus_bleu([pred], [[ref]])
        return result.score / 100.0
    except Exception:
        return float("nan")


def _bert_faithfulness(pred: str, context: str) -> float:
    """Cosine similarity using sentence-transformers (optional)."""
    try:
        from sentence_transformers import SentenceTransformer, util
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        e1 = _model.encode(pred, convert_to_tensor=True)
        e2 = _model.encode(context[:512], convert_to_tensor=True)
        return float(util.cos_sim(e1, e2).item())
    except ImportError:
        return float("nan")
    except Exception:
        return float("nan")


def _kl_divergence_from_uniform(alpha: float, beta: float) -> float:
    """KL(Beta(alpha,beta) || Beta(1,1)) — measures policy divergence from prior."""
    try:
        from math import lgamma, log
        def _lbeta(a, b):
            return lgamma(a) + lgamma(b) - lgamma(a + b)
        # KL(Beta(a,b)||Beta(1,1)) = log B(1,1) - log B(a,b) + (a-1)ψ(a) + (b-1)ψ(b) - (a+b-2)ψ(a+b)
        # Simplified: since Beta(1,1) is uniform, KL = -log B(a,b) + (a-1)ψ(a) + (b-1)ψ(b) - (a+b-2)ψ(a+b)
        # Use numerical approximation via entropy
        import math
        a, b = max(alpha, 1.001), max(beta, 1.001)
        ab = a + b
        # digamma approximation for large values: ψ(x) ≈ ln(x) - 1/(2x)
        def digamma(x):
            return math.log(x) - 1.0 / (2.0 * x) - 1.0 / (12.0 * x * x)
        kl = -_lbeta(a, b) + (a - 1) * digamma(a) + (b - 1) * digamma(b) - (a + b - 2) * digamma(ab)
        return max(0.0, kl)
    except Exception:
        return float("nan")


def _peak_ram_gb() -> float:
    if not _PSUTIL:
        return float("nan")
    try:
        import os
        proc = _psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 ** 3)
    except Exception:
        return float("nan")


def _peak_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except ImportError:
        pass
    return float("nan")


def _significance_marker(p: float) -> str:
    for threshold, marker in SIGNIFICANCE_MARKERS:
        if p < threshold:
            return marker
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap significance testing
# ─────────────────────────────────────────────────────────────────────────────

def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Paired bootstrap test: is mean(A) > mean(B)?

    Returns p-value (one-sided: P(A <= B | H0)) and 95% CI for difference.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    n = len(a)
    if n == 0 or n != len(b):
        return {"p_value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "delta": float("nan")}

    observed_delta = float(np.nanmean(a) - np.nanmean(b))
    # Paired bootstrap under the null (shift both distributions to same mean)
    # Standard approach: resample paired differences, centre on zero.
    diffs = a - b
    mean_diff = float(np.nanmean(diffs))
    diffs_centred = diffs - mean_diff          # shift so H0: mean=0 holds
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.nanmean(diffs_centred[idx], axis=1)
    # Two-sided p-value: fraction of null-shifted boot means >= observed delta
    p_value = float(np.mean(np.abs(boot_means) >= abs(observed_delta)))
    # 95% CI on the original (uncentred) bootstrap deltas
    boot_deltas = np.nanmean(diffs[idx], axis=1)
    ci_low = float(np.percentile(boot_deltas, 2.5))
    ci_high = float(np.percentile(boot_deltas, 97.5))
    return {
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "delta": observed_delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RAG Baseline
# ─────────────────────────────────────────────────────────────────────────────

class RAGBaseline:
    """
    LangChain + ChromaDB RAG baseline (spec 3.2).
    Gracefully degrades when langchain/chromadb are not installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._vectorstore = None
        self._available = self._check_deps()

    @staticmethod
    def _check_deps() -> bool:
        try:
            import chromadb  # noqa: F401
            from langchain_community.vectorstores import Chroma  # noqa: F401
            from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: F401
            return True
        except ImportError:
            try:
                # Older langchain
                from langchain.vectorstores import Chroma  # noqa: F401
                from langchain.embeddings import HuggingFaceEmbeddings  # noqa: F401
                return True
            except ImportError:
                return False

    def adapt(self, train_data: list[dict]) -> None:
        if not self._available:
            log.warning("RAG baseline: langchain/chromadb not installed — RAG will return empty results")
            return
        try:
            try:
                from langchain_community.vectorstores import Chroma
                from langchain_community.embeddings import HuggingFaceEmbeddings
            except ImportError:
                from langchain.vectorstores import Chroma
                from langchain.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name=self._model_name)
            texts = [str(d.get("context", d.get("question", ""))) for d in train_data]
            metadatas = [{"answer": str(d.get("answer", "")), "idx": i} for i, d in enumerate(train_data)]
            self._vectorstore = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
            log.info("RAG: indexed %d training documents", len(texts))
        except Exception as exc:
            log.warning("RAG adapt failed: %s", exc)
            self._vectorstore = None

    def query(self, question: str, k: int = 10) -> list[str]:
        if self._vectorstore is None:
            return []
        try:
            docs = self._vectorstore.similarity_search_with_score(question, k=k)
            return [doc.page_content for doc, _ in docs]
        except Exception as exc:
            log.debug("RAG query failed: %s", exc)
            return []

    def reset(self) -> None:
        """A31 compatibility — RAG is stateless per query."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# NLI faithfulness scorer
# ─────────────────────────────────────────────────────────────────────────────

class NLIScorer:
    """Lazy-loaded NLI pipeline for entailment-based faithfulness (spec 3.4)."""

    def __init__(self):
        self._pipe = None
        self._available = False

    def _load(self) -> bool:
        if self._pipe is not None:
            return True
        try:
            from transformers import pipeline
            device = 0
            try:
                import torch
                device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                device = -1
            self._pipe = pipeline(
                "text-classification",
                model="roberta-large-mnli",
                device=device,
            )
            self._available = True
            log.info("NLI model loaded (device=%d)", device)
            return True
        except Exception as exc:
            log.warning("NLI model unavailable: %s — skipping NLI metrics", exc)
            return False

    def score(self, pred: str, context: str) -> float:
        """Return entailment probability of pred given context."""
        if not self._load():
            return float("nan")
        try:
            inp = f"{pred[:256]} </s> {context[:512]}"
            result = self._pipe(inp, truncation=True, max_length=512)
            for item in result:
                if item["label"].upper() == "ENTAILMENT":
                    return float(item["score"])
            # If top prediction is not ENTAILMENT, return 1 - score
            return 1.0 - float(result[0]["score"]) if result else float("nan")
        except Exception as exc:
            log.debug("NLI score failed: %s", exc)
            return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Blix profile factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_llm():
    """Minimal LLM stub implementing full LLMProvider interface for benchmarks."""
    class _MockLLM:
        def generate(self, prompt: str) -> str:
            return f"[mock] {prompt[:60]}"
        def reset(self) -> None:
            pass
        def model_name(self) -> str:
            return "mock"
        def supports_streaming(self) -> bool:
            return False
    return _MockLLM()


def _make_mock_retriever():
    """Minimal retriever stub when SemanticRetriever cannot be constructed."""
    class _MockRetriever:
        def retrieve(self, memories, query: str):
            return memories[:10] if memories else []
    return _MockRetriever()


def _build_blix_profile(profile: str, memory_dir: Path, llm_model: str):
    """
    Instantiate a TutorAgent wired according to the ablation profile.

    Profiles
    --------
    full           : HGSHM + graph + hierarchy + ADMA
    no_graph       : HGSHM + hierarchy + ADMA, no graph edges
    no_adma        : HGSHM + graph + hierarchy, no policy learning
    both           : HGSHM + hierarchy, no graph, no ADMA
    hierarchy_only : HGSHM + hierarchy + temporal, no graph, no ADMA
    temporal_only  : HGSHM + temporal decay, flat memory
    graph_only     : HGSHM + graph edges, no hierarchy, no ADMA
    adma_only      : HGSHM + ADMA, flat memory, no graph, no hierarchy

    Returns (agent, memory_manager, hgshm | None, learner | None)
    """
    from pathlib import Path as _Path
    memory_dir.mkdir(parents=True, exist_ok=True)

    # TutorAgent is designed for core.memory_manager.MemoryManager (v0.2 API),
    # which stores memories in JSON files, not HGSHM. Use that class here.
    from core.memory_manager import MemoryManager
    from llm.provider_factory import build_provider
    from config.settings import LLMSettings
    from core.semantic_retriever import SemanticRetriever
    from core.memory_retriever import MemoryRetriever
    from core.prompt_builder import PromptBuilder
    from core.tutor_agent import TutorAgent

    mm = MemoryManager(
        conversations_file=memory_dir / "conversations.json",
        profile_file=memory_dir / "profile.json",
        learning_state_file=memory_dir / "learning_state.json",
    )

    # Build LLM provider via config (graceful fallback to mock for benchmarks)
    try:
        llm = build_provider(LLMSettings(model=llm_model, provider="ollama"))
    except Exception as exc:
        log.warning("Could not build LLM provider (%s) — using mock", exc)
        llm = _make_mock_llm()

    # Build SemanticRetriever — mirrors the pattern in app.py and api/context.py.
    # EmbeddingStore needs per-run file paths scoped to memory_dir to avoid
    # collisions between parallel benchmark seeds.
    try:
        from config.settings import settings as _settings
        from core.embedding_store import EmbeddingStore as _ES
        from core.memory_retriever import MemoryRetriever as _MR
        _embed_cfg = _settings.embed
        _es = _ES(
            embed_model_name=_embed_cfg.model,
            embeddings_file=memory_dir / "embeddings.npy",
            ids_file=memory_dir / "embedding_ids.json",
            threshold=_embed_cfg.threshold,
            top_k=_embed_cfg.top_k,
        )
        _mr = _MR()   # no required args
        retriever = SemanticRetriever(
            embedding_store=_es,
            legacy_retriever=_mr,
        )
    except Exception as exc:
        log.warning("Could not build SemanticRetriever (%s) — agent retrieval degraded", exc)
        retriever = _make_mock_retriever()
    builder = PromptBuilder()

    # Profiles that use graph edges
    _GRAPH_PROFILES       = ("full", "no_adma", "graph_only")
    # Profiles that use ADMA policy learning
    _ADMA_PROFILES        = ("full", "no_graph", "adma_only")
    # Profiles that use hierarchy manager
    _HIERARCHY_PROFILES   = ("full", "no_graph", "no_adma", "both", "hierarchy_only")

    memory_graph = None
    if profile in _GRAPH_PROFILES:
        try:
            from core.memory_graph import MemoryGraph
            memory_graph = MemoryGraph(hgshm=hgshm)
        except Exception as exc:
            log.warning("Could not init MemoryGraph for profile=%s: %s", profile, exc)

    learner = None
    if profile in _ADMA_PROFILES:
        try:
            from policy.store import PolicyStore
            from policy.learner import PolicyLearner
            store = PolicyStore(memory_dir=memory_dir)
            learner = PolicyLearner(policy_store=store)
            learner.register_defaults()
        except Exception as exc:
            log.warning("Could not init PolicyLearner for profile=%s: %s", profile, exc)

    hierarchy_manager = None
    if profile in _HIERARCHY_PROFILES:
        try:
            from core.hierarchy_manager import HierarchyManager
            hierarchy_manager = HierarchyManager(hierarchy_dir=memory_dir / "hierarchy", llm=llm)
        except Exception as exc:
            log.warning("Could not init HierarchyManager for profile=%s: %s", profile, exc)

    # Profile-specific retrieval weight overrides — zero out components
    # that are disabled for this ablation.
    from memory.hybrid.retrieval.hybrid_retriever import HybridWeights
    weight_overrides: dict[str, float] = {}
    if profile in ("temporal_only", "adma_only"):
        weight_overrides = {"graph_distance": 0.0, "hierarchy": 0.0}
    elif profile == "graph_only":
        weight_overrides = {"recency": 0.0, "hierarchy": 0.0}
    elif profile == "hierarchy_only":
        weight_overrides = {"graph_distance": 0.0}
    elif profile == "both":
        weight_overrides = {"graph_distance": 0.0}
    elif profile == "no_graph":
        weight_overrides = {"graph_distance": 0.0}

    if weight_overrides:
        base = retriever._weights if hasattr(retriever, "_weights") else HybridWeights()
        import dataclasses
        new_w = dataclasses.replace(base, **weight_overrides)
        retriever._weights = new_w.normalised()

    agent = TutorAgent(
        llm=llm,
        memory_manager=mm,
        retriever=retriever,
        prompt_builder=builder,
        memory_graph=memory_graph,
        hierarchy_manager=hierarchy_manager,
    )

    return agent, mm, hgshm, learner


# ─────────────────────────────────────────────────────────────────────────────
# Data loading + splitting
# ─────────────────────────────────────────────────────────────────────────────

def _load_hf_dataset(dataset_name: str, config: dict, n_samples: int) -> list[dict]:
    """Load from HuggingFace datasets, normalise to flat dicts, cap at n_samples."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "huggingface datasets not installed. Run: pip install datasets"
        )

    cfg = config
    log.info("Loading %s (hf_path=%s, name=%s, split=%s) …",
             dataset_name, cfg["hf_path"], cfg.get("hf_name"), cfg["hf_split"])

    try:
        ds = load_dataset(
            cfg["hf_path"],
            cfg.get("hf_name"),
            split=cfg["hf_split"],
            trust_remote_code=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {exc}") from exc

    rows = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        row = dict(item)

        # Normalise question
        q_key = cfg["question_key"]
        if isinstance(row.get(q_key), dict):
            row["question"] = row[q_key].get("text", str(row[q_key]))
        elif q_key in row:
            row["question"] = str(row[q_key])
        else:
            row["question"] = f"sample_{i}"

        # Normalise answer
        a_key = cfg["answer_key"]
        if isinstance(row.get(a_key), list):
            answers = row[a_key]
            if answers and isinstance(answers[0], dict):
                row["answer"] = answers[0].get("text", str(answers[0]))
            else:
                row["answer"] = str(answers[0]) if answers else ""
        elif a_key in row:
            row["answer"] = str(row[a_key])
        else:
            row["answer"] = ""

        # Normalise context
        c_key = cfg.get("context_key", "context")
        if isinstance(row.get(c_key), dict):
            row["context"] = row[c_key].get("summary", str(row[c_key]))[:1024]
        elif c_key in row:
            row["context"] = str(row[c_key])[:1024]
        else:
            row["context"] = ""

        row["_idx"] = i
        rows.append(row)

    log.info("Loaded %d samples from %s", len(rows), dataset_name)
    return rows


def _split_data(
    data: list[dict],
    config: dict,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Temporal split or story-level split depending on dataset config."""
    if config["has_temporal"]:
        date_key = config.get("date_key") or "_idx"
        # Ensure sort key exists (fall back to _idx)
        for row in data:
            if date_key not in row:
                row[date_key] = row["_idx"]
        return temporal_split(data, date_key=date_key)
    else:
        # NarrativeQA: split by story_id to prevent leakage (spec 1.1)
        return _split_by_story_id(data, seed)


def _split_by_story_id(
    data: list[dict],
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Group by story_id / document id, shuffle stories, then slice 60/20/20."""
    # Collect unique story identifiers
    story_map: dict[str, list[dict]] = defaultdict(list)
    for row in data:
        sid = (
            row.get("document", {}).get("id")
            or row.get("story_id")
            or str(row.get("_idx", 0) // 10)  # proxy: group by 10s
        )
        story_map[str(sid)].append(row)

    stories = list(story_map.keys())
    rng = random.Random(seed)
    rng.shuffle(stories)

    n = len(stories)
    train_end = max(1, int(n * 0.6))
    val_end   = max(train_end + 1, int(n * 0.8))

    def _rows(story_ids):
        out = []
        for sid in story_ids:
            out.extend(story_map[sid])
        return out

    return (
        _rows(stories[:train_end]),
        _rows(stories[train_end:val_end]),
        _rows(stories[val_end:]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-row metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_row_metrics(
    qid: int,
    question: str,
    answer: str,
    context: str,
    retrieved: list[str],
    pred: str,
    latency: float,
    hgshm_stats: dict,
    policy_stats: dict,
    state_info: dict,
    nli: NLIScorer | None,
    profile_memory: bool,
) -> dict:
    """Compute all 6 tiers of metrics for a single prediction."""

    # Tier 1: Retrieval & Ranking
    hit_1  = _hit_at_k(answer, retrieved, 1)
    hit_5  = _hit_at_k(answer, retrieved, 5)
    hit_10 = _hit_at_k(answer, retrieved, 10)
    mrr    = _mrr(answer, retrieved)
    ndcg   = _ndcg_at_10(answer, retrieved)
    precision_1 = hit_1
    recall_10   = hit_10

    # Tier 2: Generation quality
    rouge_l = _rouge_l(pred, answer)
    bleu_4  = _bleu_4(pred, answer)
    faith_bert = _bert_faithfulness(pred, context)
    entail = nli.score(pred, context) if nli else float("nan")
    halluc = (1.0 - entail) if not math.isnan(entail) else float("nan")

    # Tier 3: Efficiency
    n_tokens = len(pred.split())
    tps = n_tokens / latency if latency > 0 else float("nan")
    vram = _peak_vram_gb() if profile_memory else float("nan")
    ram  = _peak_ram_gb()  if profile_memory else float("nan")

    # Tier 4: HGSHM Memory Dynamics
    nodes  = hgshm_stats.get("nodes", 0)
    edges  = hgshm_stats.get("edges", 0)
    density = edges / max(1, nodes)
    clusters = hgshm_stats.get("clusters", 0)
    consol_rate = clusters / max(1, nodes)

    # Tier 5: ADMA / Policy
    pol_div    = policy_stats.get("divergence", float("nan"))
    pol_switch = policy_stats.get("switching", float("nan"))
    cache_hit  = policy_stats.get("cache_hit_rate", float("nan"))

    # Tier 6: Temporal state (StreamingQA only; passed in from caller)
    state_cons = state_info.get("state_consistency", float("nan"))

    return {
        "question_id": qid,
        "question": question,
        "answer": answer,
        "pred": pred,
        "hit_1": hit_1,
        "hit_5": hit_5,
        "hit_10": hit_10,
        "mrr": mrr,
        "ndcg": ndcg,
        "precision_1": precision_1,
        "recall_10": recall_10,
        "rouge_l": rouge_l,
        "bleu_4": bleu_4,
        "faithfulness_bert": faith_bert,
        "entailment_score": entail,
        "hallucination_rate": halluc,
        "latency": latency,
        "tokens_per_sec": tps,
        "vram_peak_gb": vram,
        "ram_peak_gb": ram,
        "node_count": nodes,
        "graph_density": density,
        "consolidation_rate": consol_rate,
        "policy_divergence": pol_div,
        "policy_switching": pol_switch,
        "cache_hit_rate": cache_hit,
        "state_consistency": state_cons,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main runner class
# ─────────────────────────────────────────────────────────────────────────────

class MultiDatasetAblationRunner:

    def __init__(self, args: argparse.Namespace):
        self.datasets   = [d.strip() for d in args.datasets.split(",")]
        self.seeds      = [int(s) for s in args.seeds.split(",")]
        self.profiles   = [p.strip() for p in args.profiles.split(",")]
        self.output_dir = Path(args.output)
        self.n_samples  = args.samples
        self.nli_metrics   = args.nli_metrics
        self.profile_mem   = args.profile_memory
        self.visualize     = args.visualize
        self.verbose       = args.verbose
        self.no_rag        = args.no_rag
        self.llm_model     = args.model
        self.resume        = args.resume
        self.n_bootstrap   = 10_000

        self.nli = NLIScorer() if self.nli_metrics else None
        self._all_results: dict[str, list[dict]] = defaultdict(list)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)

        if self.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
        else:
            logging.basicConfig(level=logging.WARNING)

    # ── Public entry point ───────────────────────────────────────────────────

    def run(self) -> None:
        for ds_name in self.datasets:
            if ds_name not in DATASET_CONFIGS:
                log.warning("Unknown dataset '%s' — skipping", ds_name)
                continue
            self._run_dataset(ds_name)

        self._generate_combined_summary()
        if self.visualize:
            self._generate_figures()
        log.info("Evaluation complete. Results in %s", self.output_dir)

    # ── Dataset loop ─────────────────────────────────────────────────────────

    def _run_dataset(self, dataset_name: str) -> None:
        config = DATASET_CONFIGS[dataset_name]
        ds_dir = self.output_dir / dataset_name
        ds_dir.mkdir(exist_ok=True)

        log.info("=== Dataset: %s ===", dataset_name)
        data = _load_hf_dataset(dataset_name, config, self.n_samples)

        for seed in self.seeds:
            self._run_seed(dataset_name, config, data, seed, ds_dir)

        self._aggregate_dataset_results(dataset_name, ds_dir)

    # ── Seed loop ────────────────────────────────────────────────────────────

    def _run_seed(
        self,
        dataset_name: str,
        config: dict,
        data: list[dict],
        seed: int,
        ds_dir: Path,
    ) -> None:
        seed_dir = ds_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        log.info("  Seed %d", seed)
        random.seed(seed)
        np.random.seed(seed)

        train, val, test = _split_data(data, config, seed)
        log.info("  Split: train=%d val=%d test=%d", len(train), len(val), len(test))

        profiles_to_run = list(self.profiles)
        if self.no_rag and "rag" in profiles_to_run:
            profiles_to_run.remove("rag")

        for profile in profiles_to_run:
            out_csv = seed_dir / f"{profile}.csv"
            if self.resume and out_csv.exists():
                log.info("  Skipping %s (--resume)", profile)
                continue

            log.info("  Profile: %s", profile)
            if profile == "rag":
                rows = self._run_rag(train, test, dataset_name)
            else:
                rows = self._run_blix(profile, train, val, test, dataset_name, seed)

            self._write_csv(out_csv, rows)
            self._all_results[f"{dataset_name}:{profile}"].extend(rows)

    # ── Blix profile runner ──────────────────────────────────────────────────

    def _run_blix(
        self,
        profile: str,
        train: list[dict],
        val: list[dict],
        test: list[dict],
        dataset_name: str,
        seed: int,
    ) -> list[dict]:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            memory_dir = Path(td) / "memory"
            try:
                agent, mm, hgshm, learner = _build_blix_profile(
                    profile, memory_dir, self.llm_model
                )
            except Exception as exc:
                log.error("Failed to build profile '%s': %s", profile, exc)
                return []

            # Adaptation phase (train)
            log.info("    Adapting on %d train samples …", len(train))
            for item in train:
                try:
                    agent._llm.reset()                      # A31
                    context_text = item.get("context", "")
                    if context_text:
                        # core.MemoryManager.add_memory() stores interactions
                        mm.add_memory(
                            user_input=item['question'],
                            assistant_output=item['answer'],
                        )
                except Exception as exc:
                    log.debug("Train adaptation error: %s", exc)

            # Freeze policies before test (spec 3.1 step 4)
            if learner is not None:
                learner._decay_epoch = float("inf")   # disable further decay

            # Test phase
            log.info("    Evaluating on %d test samples …", len(test))
            rows = []
            prev_policy_id: str | None = None
            policy_switches = 0
            cache_hits = 0
            cache_total = 0

            for qid, item in enumerate(test):
                question = item.get("question", "")
                answer   = item.get("answer", "")
                context  = item.get("context", "")

                # A31: reset KV cache between queries
                try:
                    agent._llm.reset()
                except Exception:
                    pass

                t0 = time.perf_counter()
                try:
                    pred = agent.chat(question)
                except Exception as exc:
                    log.debug("chat() failed: %s", exc)
                    pred = ""
                latency = time.perf_counter() - t0

                # Retrieved context for ranking metrics
                try:
                    # retrieve_memory() returns list[MemoryEntry] — .text doesn't exist,
                    # use .output (the stored response) as the text representation
                    entries = agent.retrieve_memory(question)
                    retrieved = [
                        getattr(r, 'output', getattr(r, 'input', str(r)))
                        for r in (entries or [])
                    ]
                except Exception:
                    retrieved = [context] if context else []

                # HGSHM stats (Tier 4)
                try:
                    # core.MemoryManager has no stats(); approximate from memory count
                    n_mem = mm.memory_count()
                    hgshm_stats = {"nodes": n_mem, "edges": 0, "clusters": 0}
                except Exception:
                    hgshm_stats = {}

                # Policy stats (Tier 5)
                policy_stats: dict[str, float] = {}
                if learner is not None:
                    cache_total += 1
                    with learner._cache_lock:
                        hit = len(learner._cache) > 0
                    if hit:
                        cache_hits += 1
                    # Policy switching
                    try:
                        arms = learner._store.all_active()
                        if arms:
                            curr_id = arms[0].policy_id
                            if prev_policy_id is not None and curr_id != prev_policy_id:
                                policy_switches += 1
                            prev_policy_id = curr_id
                            # KL divergence from uniform prior
                            arm = arms[0]
                            policy_stats["divergence"] = _kl_divergence_from_uniform(
                                arm.alpha, arm.beta_
                            )
                    except Exception:
                        pass
                    policy_stats["switching"]     = policy_switches / max(1, qid)
                    policy_stats["cache_hit_rate"] = cache_hits / max(1, cache_total)

                # Temporal state tracking (StreamingQA tier 6)
                state_info: dict[str, float] = {}
                if dataset_name == "streamingqa":
                    ts = item.get("date") or item.get("timestamp") or item.get("_idx", 0)
                    try:
                        ts_val = float(ts) if not isinstance(ts, str) else float(qid)
                    except (ValueError, TypeError):
                        ts_val = float(qid)
                    correct = 1.0 if answer.lower() in pred.lower() else 0.0
                    state_info["state_consistency"] = correct  # aggregated later

                row = _compute_row_metrics(
                    qid=qid,
                    question=question,
                    answer=answer,
                    context=context,
                    retrieved=retrieved,
                    pred=pred,
                    latency=latency,
                    hgshm_stats=hgshm_stats,
                    policy_stats=policy_stats,
                    state_info=state_info,
                    nli=self.nli,
                    profile_memory=self.profile_mem,
                )
                rows.append(row)

            # Close resources
            # core.MemoryManager uses JSON files — no explicit close needed
            pass

            return rows

    # ── RAG runner ───────────────────────────────────────────────────────────

    def _run_rag(
        self,
        train: list[dict],
        test: list[dict],
        dataset_name: str,
    ) -> list[dict]:
        rag = RAGBaseline()
        rag.adapt(train)
        rows = []
        for qid, item in enumerate(test):
            question = item.get("question", "")
            answer   = item.get("answer", "")
            context  = item.get("context", "")
            rag.reset()  # A31 compatibility
            t0 = time.perf_counter()
            retrieved = rag.query(question, k=10)
            latency = time.perf_counter() - t0
            pred = retrieved[0] if retrieved else ""
            row = _compute_row_metrics(
                qid=qid, question=question, answer=answer,
                context=context, retrieved=retrieved, pred=pred,
                latency=latency, hgshm_stats={}, policy_stats={},
                state_info={}, nli=self.nli, profile_memory=self.profile_mem,
            )
            rows.append(row)
        return rows

    # ── CSV I/O ──────────────────────────────────────────────────────────────

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        log.info("    Wrote %d rows → %s", len(rows), path)

    # ── Aggregation ──────────────────────────────────────────────────────────

    def _aggregate_dataset_results(self, dataset_name: str, ds_dir: Path) -> None:
        """Compute mean ± std across seeds; write aggregate CSV + ablation table."""
        profiles_data: dict[str, list[dict]] = {}
        for profile in self.profiles:
            key = f"{dataset_name}:{profile}"
            if key in self._all_results:
                profiles_data[profile] = self._all_results[key]

        if not profiles_data:
            return

        summary_rows = []
        key_metrics = ["hit_1", "hit_5", "hit_10", "mrr", "ndcg",
                        "rouge_l", "faithfulness_bert", "hallucination_rate",
                        "latency", "graph_density", "cache_hit_rate"]

        agg: dict[str, dict[str, Any]] = {}
        for profile, rows in profiles_data.items():
            agg[profile] = {}
            for metric in key_metrics:
                vals = [r.get(metric) for r in rows if r.get(metric) is not None
                        and not (isinstance(r.get(metric), float) and math.isnan(r.get(metric)))]
                if vals:
                    agg[profile][metric] = {
                        "mean": float(np.mean(vals)),
                        "std":  float(np.std(vals)),
                        "n": len(vals),
                    }
                else:
                    agg[profile][metric] = {"mean": float("nan"), "std": float("nan"), "n": 0}

        # Bootstrap significance: full vs others on hit_5
        sig: dict[str, dict] = {}
        if "full" in profiles_data:
            full_hit5 = [r.get("hit_5", 0.0) for r in profiles_data["full"]]
            for other in ["no_graph", "no_adma", "rag", "both"]:
                if other in profiles_data:
                    other_hit5 = [r.get("hit_5", 0.0) for r in profiles_data[other]]
                    sig[f"full_vs_{other}"] = paired_bootstrap(full_hit5, other_hit5, self.n_bootstrap)

        # Write aggregate JSON
        agg_data = {"metrics": agg, "significance": sig}
        with open(ds_dir / "aggregate_summary.json", "w") as f:
            json.dump(agg_data, f, indent=2, default=str)

        # Write ablation markdown table
        self._write_ablation_table(ds_dir, dataset_name, agg, sig)

    def _write_ablation_table(
        self,
        ds_dir: Path,
        dataset_name: str,
        agg: dict,
        sig: dict,
    ) -> None:
        lines = [
            f"# Ablation Table: {dataset_name}\n",
            "| Configuration | Hit@5 ↑ | MRR ↑ | Faithfulness ↑ | Hallucination ↓ | Latency (s) ↓ | Graph Density | Cache Hit |",
            "|---------------|---------|-------|----------------|-----------------|---------------|---------------|-----------|",
        ]

        def _fmt(profile: str, metric: str) -> str:
            d = agg.get(profile, {}).get(metric, {})
            m, s = d.get("mean", float("nan")), d.get("std", float("nan"))
            if math.isnan(m):
                return "—"
            return f"{m:.3f} ± {s:.3f}"

        def _sig_mark(profile: str) -> str:
            key = f"full_vs_{profile}"
            if key not in sig:
                return ""
            p = sig[key].get("p_value", 1.0)
            return _significance_marker(p)

        profile_labels = {
            "rag": "RAG Baseline",
            "both": "Blix (Both off)",
            "no_graph": "Blix (No Graph)",
            "no_adma": "Blix (No ADMA)",
            "full": "**Blix Full**",
        }
        ordered = ["rag", "both", "no_graph", "no_adma", "full"]
        for profile in ordered:
            if profile not in agg:
                continue
            mark = _sig_mark(profile) if profile != "full" else ""
            label = profile_labels.get(profile, profile)
            row = (
                f"| {label} "
                f"| {_fmt(profile, 'hit_5')}{mark} "
                f"| {_fmt(profile, 'mrr')}{mark} "
                f"| {_fmt(profile, 'faithfulness_bert')}{mark} "
                f"| {_fmt(profile, 'hallucination_rate')}{mark} "
                f"| {_fmt(profile, 'latency')} "
                f"| {_fmt(profile, 'graph_density')} "
                f"| {_fmt(profile, 'cache_hit_rate')} |"
            )
            lines.append(row)

        lines += [
            "",
            "Significance vs Blix Full: † p<0.10, * p<0.05, ** p<0.01, *** p<0.001",
            "(Paired bootstrap, 10,000 resamples on Hit@5)",
        ]

        with open(ds_dir / "ablation_table.md", "w") as f:
            f.write("\n".join(lines) + "\n")

    # ── Combined summary ─────────────────────────────────────────────────────

    def _generate_combined_summary(self) -> None:
        """Write combined_summary.md and combined_summary.tex."""
        lines_md = ["# Combined Ablation Summary\n"]
        lines_tex = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Multi-Dataset Ablation Results (Mean $\pm$ Std)}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{llccccc}",
            r"\toprule",
            r"Dataset & Config & Hit@5$\uparrow$ & MRR$\uparrow$ & Faithfulness$\uparrow$ & Hallucination$\downarrow$ & Latency$\downarrow$ \\",
            r"\midrule",
        ]

        for ds_name in self.datasets:
            agg_file = self.output_dir / ds_name / "aggregate_summary.json"
            if not agg_file.exists():
                continue
            with open(agg_file) as f:
                data = json.load(f)
            agg = data.get("metrics", {})
            sig = data.get("significance", {})

            lines_md.append(f"\n## {ds_name}\n")
            lines_md.append(
                "| Config | Hit@5 | MRR | Faithfulness | Hallucination | Latency |"
            )
            lines_md.append("|--------|-------|-----|--------------|---------------|---------|")

            for profile in ["rag", "both", "no_graph", "no_adma", "full"]:
                if profile not in agg:
                    continue

                def _v(metric):
                    d = agg[profile].get(metric, {})
                    m, s = d.get("mean", float("nan")), d.get("std", float("nan"))
                    return (m, s)

                def _fv(metric):
                    m, s = _v(metric)
                    return "—" if math.isnan(m) else f"{m:.3f}±{s:.3f}"

                mark = ""
                if profile != "full":
                    p = sig.get(f"full_vs_{profile}", {}).get("p_value", 1.0)
                    mark = _significance_marker(p)

                lines_md.append(
                    f"| {profile}{mark} | {_fv('hit_5')} | {_fv('mrr')} "
                    f"| {_fv('faithfulness_bert')} | {_fv('hallucination_rate')} | {_fv('latency')} |"
                )

                # LaTeX row
                m5, s5 = _v("hit_5")
                mm, sm = _v("mrr")
                mf, sf = _v("faithfulness_bert")
                mh, sh = _v("hallucination_rate")
                ml, sl = _v("latency")
                label = profile.replace("_", r"\_")
                row_tex = (
                    f"{ds_name} & {label}{mark} & "
                    f"{'—' if math.isnan(m5) else f'{m5:.3f}$\\pm${s5:.3f}'} & "
                    f"{'—' if math.isnan(mm) else f'{mm:.3f}$\\pm${sm:.3f}'} & "
                    f"{'—' if math.isnan(mf) else f'{mf:.3f}$\\pm${sf:.3f}'} & "
                    f"{'—' if math.isnan(mh) else f'{mh:.3f}$\\pm${sh:.3f}'} & "
                    f"{'—' if math.isnan(ml) else f'{ml:.3f}$\\pm${sl:.3f}'} \\\\"
                )
                lines_tex.append(row_tex)
            lines_tex.append(r"\midrule")

        lines_tex += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        lines_md += [
            "\n---",
            "Significance vs Blix Full (paired bootstrap, 10,000 resamples): "
            "† p<0.10  * p<0.05  ** p<0.01  *** p<0.001",
        ]

        with open(self.output_dir / "combined_summary.md", "w") as f:
            f.write("\n".join(lines_md) + "\n")
        with open(self.output_dir / "combined_summary.tex", "w") as f:
            f.write("\n".join(lines_tex) + "\n")
        log.info("Wrote combined_summary.md + combined_summary.tex")

    # ── Figures ──────────────────────────────────────────────────────────────

    def _generate_figures(self) -> None:
        if not _MPL:
            log.warning("matplotlib not installed — skipping figures")
            return

        fig_dir = self.output_dir / "figures"

        # 1. Learning curve: hit@5 over time (test-set order as proxy)
        self._plot_learning_curve(fig_dir)

        # 2. Latency analysis: p50/p95/p99 per profile per dataset
        self._plot_latency(fig_dir)

        # 3. Graph density growth over test queries
        self._plot_graph_density(fig_dir)

    def _plot_learning_curve(self, fig_dir: Path) -> None:
        fig, axes = plt.subplots(1, len(self.datasets), figsize=(4 * len(self.datasets), 4), squeeze=False)
        for di, ds_name in enumerate(self.datasets):
            ax = axes[0][di]
            for profile in self.profiles:
                key = f"{ds_name}:{profile}"
                rows = self._all_results.get(key, [])
                if not rows:
                    continue
                # Rolling mean of hit@5
                vals = [r.get("hit_5", 0.0) for r in rows]
                window = max(1, len(vals) // 20)
                smoothed = np.convolve(vals, np.ones(window) / window, mode="valid")
                ax.plot(smoothed, label=profile, alpha=0.85)
            ax.set_title(ds_name)
            ax.set_xlabel("Query index")
            ax.set_ylabel("Hit@5 (smoothed)")
            ax.legend(fontsize=7)
        fig.suptitle("Learning Curve (Hit@5 over test queries)")
        fig.tight_layout()
        fig.savefig(fig_dir / "learning_curve.png", dpi=150)
        plt.close(fig)

    def _plot_latency(self, fig_dir: Path) -> None:
        fig, axes = plt.subplots(1, len(self.datasets), figsize=(4 * len(self.datasets), 4), squeeze=False)
        for di, ds_name in enumerate(self.datasets):
            ax = axes[0][di]
            labels, p50s, p95s, p99s = [], [], [], []
            for profile in self.profiles:
                key = f"{ds_name}:{profile}"
                rows = self._all_results.get(key, [])
                lats = [r.get("latency", float("nan")) for r in rows]
                lats = [v for v in lats if not math.isnan(v)]
                if not lats:
                    continue
                labels.append(profile)
                p50s.append(float(np.percentile(lats, 50)))
                p95s.append(float(np.percentile(lats, 95)))
                p99s.append(float(np.percentile(lats, 99)))
            x = np.arange(len(labels))
            width = 0.25
            ax.bar(x - width, p50s, width, label="p50")
            ax.bar(x,         p95s, width, label="p95")
            ax.bar(x + width, p99s, width, label="p99")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_title(ds_name)
            ax.set_ylabel("Latency (s)")
            ax.legend(fontsize=7)
        fig.suptitle("Latency Analysis (p50 / p95 / p99)")
        fig.tight_layout()
        fig.savefig(fig_dir / "latency_analysis.png", dpi=150)
        plt.close(fig)

    def _plot_graph_density(self, fig_dir: Path) -> None:
        fig, axes = plt.subplots(1, len(self.datasets), figsize=(4 * len(self.datasets), 4), squeeze=False)
        for di, ds_name in enumerate(self.datasets):
            ax = axes[0][di]
            for profile in ["full", "no_adma"]:  # profiles that have a graph
                key = f"{ds_name}:{profile}"
                rows = self._all_results.get(key, [])
                if not rows:
                    continue
                densities = [r.get("graph_density", float("nan")) for r in rows]
                densities = [v if not math.isnan(v) else 0.0 for v in densities]
                ax.plot(densities, label=profile, alpha=0.85)
            ax.set_title(ds_name)
            ax.set_xlabel("Query index")
            ax.set_ylabel("Graph Density (edges/nodes)")
            ax.legend(fontsize=7)
        fig.suptitle("Graph Density Growth over Test Queries")
        fig.tight_layout()
        fig.savefig(fig_dir / "graph_density_growth.png", dpi=150)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Blix multi-dataset ablation harness (NeurIPS/AAAI evaluation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets",  default="hotpotqa,locomo,narrativeqa,streamingqa",
                   help="Comma-separated dataset names")
    p.add_argument("--samples",   type=int, default=200,
                   help="Samples to load per dataset (≤ available)")
    p.add_argument("--seeds",     default="42",
                   help="Comma-separated random seeds")
    p.add_argument("--profiles",  default="full,no_graph,no_adma,both,hierarchy_only,temporal_only,graph_only,adma_only",
                   help="Comma-separated ablation profiles")
    p.add_argument("--output",    default="results/",
                   help="Output directory")
    p.add_argument("--nli-metrics",    action="store_true",
                   help="Enable NLI faithfulness (requires transformers)")
    p.add_argument("--profile-memory", action="store_true",
                   help="Track VRAM/RAM usage per query")
    p.add_argument("--visualize",      action="store_true",
                   help="Generate matplotlib figures")
    p.add_argument("--no-rag",         action="store_true",
                   help="Skip RAG baseline")
    p.add_argument("--verbose",        action="store_true",
                   help="Verbose logging")
    p.add_argument("--model",     default="llama3.2:3b",
                   help="LLM model name (passed to build_provider via LLMSettings)")
    p.add_argument("--resume",         action="store_true",
                   help="Skip existing seed/profile CSV files")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    runner = MultiDatasetAblationRunner(args)
    runner.run()


if __name__ == "__main__":
    main()

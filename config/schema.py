"""
config/schema.py — Pydantic schemas for v0.3.16 ADMA and HGSHM settings.

Design principles
-----------------
• Every configurable value in policy/ and memory/hybrid/ is represented here.
• Validators enforce invariants at startup (fail-fast).
• Defaults match the hardcoded values they replace, preserving behavior.
• Sub-schemas group related values so callers import only what they need.
• Mathematical constants (z=1.96, prior alpha=beta=1.0) are NOT here — they
  are implementation details, not configuration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Policy / ADMA schemas ────────────────────────────────────────────

class PolicyLearnerSettings(BaseModel):
    """Contextual bandit learning parameters."""

    decay_factor: float = Field(
        default=0.995,
        gt=0.0, le=1.0,
        description=(
            "Multiplicative per-observation decay applied to (α-1) and (β-1). "
            "0.995 ≈ half-life of 139 observations."
        ),
    )
    reward_threshold: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Reward values ≥ this increment alpha; values < this increment beta.",
    )
    snapshot_every: int = Field(
        default=20,
        gt=0,
        description="Save a PolicyVersion checkpoint every N updates per policy.",
    )
    decay_persist_every: int = Field(
        default=50,
        gt=0,
        description=(
            "Flush accumulated epoch decay to SQLite every N observations. "
            "Higher = fewer writes, more potential drift on unclean shutdown."
        ),
    )
    cache_max_size: int = Field(
        default=1000,
        gt=0,
        description="Maximum PolicyRecord entries in the in-process LRU cache.",
    )


class PolicyOptimizerSettings(BaseModel):
    """Lifecycle management parameters for policy aging and mutation."""

    min_observations: int = Field(
        default=10,
        gt=0,
        description="Minimum reward observations before a policy qualifies for retirement.",
    )
    aging_threshold: float = Field(
        default=0.35,
        ge=0.0, le=1.0,
        description=(
            "Policies whose confidence (α/(α+β)) stays below this after "
            "min_observations are candidates for retirement and mutation."
        ),
    )
    convergence_window: int = Field(
        default=5,
        gt=1,
        description="Number of recent version snapshots checked for convergence.",
    )
    convergence_tolerance: float = Field(
        default=0.02,
        ge=0.0, le=1.0,
        description=(
            "If the spread of confidence across the last convergence_window "
            "snapshots is below this, the policy is considered converged."
        ),
    )
    rollback_lookback: int = Field(
        default=5,
        gt=0,
        description="Number of recent snapshots compared to detect performance drops.",
    )
    rollback_drop_threshold: float = Field(
        default=0.10,
        ge=0.0, le=1.0,
        description="Minimum confidence drop triggering automatic rollback.",
    )
    mutation_scale: float = Field(
        default=0.1,
        gt=0.0, le=1.0,
        description="Fractional perturbation applied to numeric config values when spawning mutants.",
    )
    decay_factor: float = Field(
        default=0.995,
        gt=0.0, le=1.0,
        description="Per-cycle decay factor applied to all policies during run_cycle().",
    )


class RewardEngineSettings(BaseModel):
    """Observable reward computation parameters."""

    latency_target_ms: float = Field(
        default=500.0,
        gt=0.0,
        description=(
            "Target latency for the exponential reward decay. "
            "At this latency the reward is ≈0.5; 0 ms → 1.0; 2× target → ~0.25."
        ),
    )
    retrieval_latency_target_ms: float = Field(
        default=200.0,
        gt=0.0,
        description="Separate latency target for memory retrieval operations.",
    )
    token_budget: int = Field(
        default=2000,
        gt=0,
        description="Token budget used by token_efficiency_reward.",
    )


class RewardLogSettings(BaseModel):
    """Reward log retention parameters."""

    max_rows_per_policy: int = Field(
        default=1000,
        gt=0,
        description=(
            "Maximum reward_log rows retained per policy. "
            "Oldest rows are deleted when this limit is exceeded."
        ),
    )
    stats_last_n: int = Field(
        default=1000,
        gt=0,
        description="reward_stats() considers only the most recent N rows.",
    )


class PromptCompilerSettings(BaseModel):
    """Dynamic prompt assembly parameters."""

    token_budget: int = Field(
        default=2000,
        gt=0,
        description="Default token budget for compiled prompts.",
    )
    max_memory_nodes: int = Field(
        default=5,
        gt=0,
        description="Maximum memory nodes to include in a compiled prompt.",
    )
    max_system_instructions_chars: int = Field(
        default=8000,
        gt=0,
        description=(
            "Hard character limit for assembled system_instructions. "
            "Prevents OOM/context-window overrun when many policies fire. "
            "Logged at DEBUG when truncation occurs (A22)."
        ),
    )


class ADMASettings(BaseModel):
    """
    Adaptive Dual Memory Architecture — top-level ADMA configuration.

    Groups all ADMA sub-settings under a single namespace so callers can do:
        from config.settings import settings
        settings.adma.learner.decay_factor
    """

    learner:        PolicyLearnerSettings   = Field(default_factory=PolicyLearnerSettings)
    optimizer:      PolicyOptimizerSettings = Field(default_factory=PolicyOptimizerSettings)
    reward_engine:  RewardEngineSettings    = Field(default_factory=RewardEngineSettings)
    reward_log:     RewardLogSettings       = Field(default_factory=RewardLogSettings)
    prompt_compiler: PromptCompilerSettings = Field(default_factory=PromptCompilerSettings)


# ── HGSHM schemas ───────────────────────────────────────────────────

class EmbeddingSettings(BaseModel):
    """Vector embedding parameters."""

    dim: int = Field(
        default=256,
        gt=0,
        description="Embedding dimensionality for the hash-projection backend.",
    )
    backend: str = Field(
        default="numpy",
        description="Embedding backend: 'numpy' (default, no deps) or 'hash_projection'.",
    )


class HybridWeightsSettings(BaseModel):
    """
    Default retrieval factor weights for HybridRetriever.

    These 11 weights are normalised at runtime. They represent the relative
    importance of each signal when no learned policy overrides them.
    """

    semantic:           float = Field(default=0.25, ge=0.0)
    vector:             float = Field(default=0.20, ge=0.0)
    graph_distance:     float = Field(default=0.10, ge=0.0)
    importance:         float = Field(default=0.15, ge=0.0)
    confidence:         float = Field(default=0.10, ge=0.0)
    recency:            float = Field(default=0.08, ge=0.0)
    hierarchy:          float = Field(default=0.04, ge=0.0)
    context_similarity: float = Field(default=0.03, ge=0.0)
    attention:          float = Field(default=0.02, ge=0.0)
    belief_confidence:  float = Field(default=0.02, ge=0.0)
    planning_relevance: float = Field(default=0.01, ge=0.0)

    @model_validator(mode="after")
    def weights_sum_positive(self) -> "HybridWeightsSettings":
        total = sum([
            self.semantic, self.vector, self.graph_distance, self.importance,
            self.confidence, self.recency, self.hierarchy, self.context_similarity,
            self.attention, self.belief_confidence, self.planning_relevance,
        ])
        if total <= 0:
            raise ValueError("HybridWeightsSettings: all weights are zero — at least one must be positive.")
        return self

    def to_raw_dict(self) -> dict[str, float]:
        """Return raw (un-normalised) weight values as a dict."""
        return {
            "semantic": self.semantic, "vector": self.vector,
            "graph_distance": self.graph_distance, "importance": self.importance,
            "confidence": self.confidence, "recency": self.recency,
            "hierarchy": self.hierarchy, "context_similarity": self.context_similarity,
            "attention": self.attention, "belief_confidence": self.belief_confidence,
            "planning_relevance": self.planning_relevance,
        }

    def to_normalised_dict(self) -> dict[str, float]:
        """Return weights normalised to sum to 1.0 (A23).

        Callers that bypass HybridRetriever.HybridWeights.normalised() must
        use this method to ensure weights are valid input for the retriever.
        """
        raw = self.to_raw_dict()
        total = sum(raw.values())
        if total <= 0:
            raise ValueError(
                "HybridWeightsSettings: cannot normalise — all weights are zero"
            )
        return {k: v / total for k, v in raw.items()}

    def to_dict(self) -> dict[str, float]:
        """Return normalised weights (A23: was raw, now normalised for safety).

        Deprecated alias for :meth:`to_normalised_dict`.
        Use ``to_raw_dict()`` explicitly if you need raw values.
        """
        return self.to_normalised_dict()


class RetrievalSettings(BaseModel):
    """Retrieval pipeline parameters."""

    default_top_k: int = Field(
        default=10,
        gt=0,
        description="Default number of results returned by recall() and retrieve().",
    )
    max_age_hours: float = Field(
        default=24.0,
        gt=0.0,
        description="Maximum age for TemporalRetriever's recency window.",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Minimum final_score threshold for returned results.",
    )
    graph_max_depth: int = Field(
        default=2,
        gt=0,
        description="Graph traversal depth for GraphRetriever.",
    )
    graph_max_nodes: int = Field(
        default=50,
        gt=0,
        description="Maximum nodes explored per graph retrieval.",
    )
    weights: HybridWeightsSettings = Field(default_factory=HybridWeightsSettings)


class ConsolidationSettings(BaseModel):
    """Memory consolidation and deduplication parameters."""

    similarity_threshold: float = Field(
        default=0.92,
        ge=0.0, le=1.0,
        description="Cosine similarity above which two nodes are considered duplicates.",
    )
    jaccard_threshold: float = Field(
        default=0.70,
        ge=0.0, le=1.0,
        description="Jaccard text overlap threshold for duplicate detection.",
    )
    prune_below_importance: float = Field(
        default=0.05,
        ge=0.0, le=1.0,
        description="Nodes with importance below this are eligible for pruning.",
    )
    prune_below_confidence: float = Field(
        default=0.10,
        ge=0.0, le=1.0,
        description="Nodes with confidence below this are eligible for pruning.",
    )
    max_scan: int = Field(
        default=500,
        gt=0,
        description="Maximum nodes scanned per consolidation run.",
    )


class HierarchySettings(BaseModel):
    """Hierarchy compression parameters."""

    max_summary_length: int = Field(
        default=300,
        gt=0,
        description="Maximum character length for extractive summaries.",
    )
    min_cluster_size: int = Field(
        default=3,
        gt=1,
        description="Minimum cluster size before hierarchy compression is triggered.",
    )


class DatabaseFilenameSettings(BaseModel):
    """Configurable database filenames (A06).

    Separating filenames from memory_dir allows blue-green deployments,
    environment-specific names (e.g. policy_test.db), and custom layouts.
    """

    hgshm_db:    str = Field(default="hgshm.db",   description="Main HGSHM graph/node database.")
    vectors_db:  str = Field(default="vectors.db", description="sqlite-vec vector index database.")
    policy_db:   str = Field(default="policy.db",  description="ADMA policy store database.")

    @field_validator("hgshm_db", "vectors_db", "policy_db", mode="before")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        import re
        if not re.match(r'^[\w\-\.]+\.db$', v):
            raise ValueError(
                f"Database filename {v!r} must end with .db and contain only "
                f"alphanumeric characters, hyphens, underscores, and dots."
            )
        return v


class ContextBuilderSettings(BaseModel):
    """
    Tunable limits for the ContextBuilder 11-step pipeline (A30).

    Before this class, every limit inside ContextBuilder.build() was
    hardcoded, making the context assembly pipeline opaque and untunable.
    These fields mirror the hardcoded constants 1-to-1 so existing
    behaviour is preserved at the defaults.
    """

    max_gap_nodes: int = Field(
        default=5,
        ge=0,
        description="Maximum knowledge-gap nodes included in a context.",
    )
    max_neighbourhood_seeds: int = Field(
        default=3,
        ge=0,
        description="Number of primary-memory seeds used to build the graph neighbourhood.",
    )
    max_neighbourhood_edges: int = Field(
        default=50,
        ge=0,
        description="Maximum edges returned in the graph neighbourhood.",
    )
    max_expansion_seeds: int = Field(
        default=3,
        ge=0,
        description="Number of seed nodes used for supporting-context graph expansion.",
    )
    max_causal_seeds: int = Field(
        default=3,
        ge=0,
        description="Number of seed nodes from which causal chains are traced.",
    )
    max_causal_chains: int = Field(
        default=3,
        ge=0,
        description="Maximum number of causal chains extracted per build() call.",
    )
    max_causal_depth: int = Field(
        default=3,
        ge=1,
        description="Maximum depth of each causal chain (DFS steps).",
    )
    graph_score_decay: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description=(
            "Exponential decay applied to graph-expansion scores per BFS depth level. "
            "score = importance * decay ** depth_reached."
        ),
    )


class HGSHMSettings(BaseModel):
    """
    Hybrid Graph-Based Semantic Hierarchical Memory — top-level configuration.
    """

    embedding:       EmbeddingSettings          = Field(default_factory=EmbeddingSettings)
    retrieval:       RetrievalSettings          = Field(default_factory=RetrievalSettings)
    consolidation:   ConsolidationSettings      = Field(default_factory=ConsolidationSettings)
    hierarchy:       HierarchySettings          = Field(default_factory=HierarchySettings)
    database:        DatabaseFilenameSettings   = Field(default_factory=DatabaseFilenameSettings)
    context_builder: ContextBuilderSettings     = Field(default_factory=ContextBuilderSettings,
                                                        description="ContextBuilder pipeline limits (A30).")


# ── Feature flags ────────────────────────────────────────────────────

class FeatureFlags(BaseModel):
    """
    Runtime feature switches for ablation experiments and staged rollouts.

    Setting a flag to False disables the component globally without code changes.
    Use these in conjunction with AblationV3Runner for systematic ablation.
    """

    hybrid_retrieval:  bool = Field(default=True,  description="Enable multi-factor hybrid retrieval.")
    adaptive_policy:   bool = Field(default=True,  description="Enable ADMA policy learning.")
    graph_reasoning:   bool = Field(default=True,  description="Enable graph traversal in retrieval.")
    hierarchy:         bool = Field(default=True,  description="Enable hierarchy compression.")
    semantic_search:   bool = Field(default=True,  description="Enable semantic/vector search.")
    reward_learning:   bool = Field(default=True,  description="Enable reward signal dispatch.")
    prompt_compiler:   bool = Field(default=True,  description="Use dynamic prompt compiler.")
    memory_consolidation: bool = Field(default=True, description="Enable memory consolidation.")


# ── Environment profiles ─────────────────────────────────────────────

Profile = Literal["development", "testing", "benchmark", "production"]

_PROFILE_OVERRIDES: dict[str, dict] = {
    "development": {
        # Faster iteration: smaller caches, lower retention
        "adma": {"learner": {"cache_max_size": 100, "decay_persist_every": 10}},
    },
    "testing": {
        # Deterministic, minimal I/O
        "adma": {
            "learner":   {"decay_persist_every": 5,  "snapshot_every": 5, "cache_max_size": 50},
            "reward_log": {"max_rows_per_policy": 100, "stats_last_n": 100},
        },
    },
    "benchmark": {
        # Maximum fidelity for research runs
        "adma": {
            "learner":   {"cache_max_size": 5000, "decay_persist_every": 100},
            "reward_log": {"max_rows_per_policy": 10000, "stats_last_n": 5000},
        },
        "hgshm": {"retrieval": {"default_top_k": 20}},
    },
    "production": {
        # Conservative defaults with tighter retention
        "adma": {
            "learner":   {"cache_max_size": 2000},
            "reward_log": {"max_rows_per_policy": 2000},
        },
    },
}

"""
policy — Adaptive Dual Memory Architecture (ADMA) policy learning.

Public API
----------
Import the most commonly used names directly from this package:

    from policy import (
        PolicyRecord, PolicyDomain, PolicyType,
        RewardSignal, RewardType,
        PolicyStore,
        RewardEngine,
        PolicyLearner,
        PolicyOptimizer,
        PolicySelector, PolicyCompiler, CompiledPrompt,
        AblationConfig, AblationV3Runner, ABLATION_CONDITIONS,
    )

For less common names import from the specific sub-module:

    from policy.models      import PolicyVersion
    from policy.reward      import SystemRewardEngine, UserRewardEngine
    from policy.compiler    import sanitize_task_text
    from policy.adaptive    import AdaptiveRetriever, AdaptivePlanner
    from policy.ablation_v3 import AblationV3Report, AblationBenchmarkResult

Design note
-----------
All names in ``__all__`` are guaranteed stable across v0.3.x patch
releases.  Names accessible from sub-modules but not listed here are
implementation details and may change without notice.
"""

# ── Core models ──────────────────────────────────────────────────────
from policy.models import (
    PolicyDomain,
    PolicyType,
    RewardType,
    RewardSignal,
    PolicyVersion,
    PolicyRecord,
)

# ── Persistence ──────────────────────────────────────────────────────
from policy.store import PolicyStore

# ── Reward engine ─────────────────────────────────────────────────────
from policy.reward import RewardEngine, SystemRewardEngine, UserRewardEngine

# ── Learning ──────────────────────────────────────────────────────────
from policy.learner import PolicyLearner

# ── Lifecycle ─────────────────────────────────────────────────────────
from policy.optimizer import PolicyOptimizer

# ── Prompt compiler ───────────────────────────────────────────────────
from policy.compiler import (
    PolicySelector,
    PolicyCompiler,
    CompiledPrompt,
    sanitize_task_text,
)

# ── Adaptive retrieval / planning ─────────────────────────────────────
from policy.adaptive import AdaptiveRetriever, AdaptivePlanner

# ── Ablation ──────────────────────────────────────────────────────────
from policy.ablation_v3 import (
    AblationConfig,
    AblationBenchmarkResult,
    AblationConditionResult,
    AblationV3Report,
    AblationV3Runner,
    ABLATION_CONDITIONS,
)

__all__ = [
    # models
    "PolicyDomain",
    "PolicyType",
    "RewardType",
    "RewardSignal",
    "PolicyVersion",
    "PolicyRecord",
    # persistence
    "PolicyStore",
    # reward
    "RewardEngine",
    "SystemRewardEngine",
    "UserRewardEngine",
    # learning
    "PolicyLearner",
    # lifecycle
    "PolicyOptimizer",
    # compiler
    "PolicySelector",
    "PolicyCompiler",
    "CompiledPrompt",
    "sanitize_task_text",
    # adaptive
    "AdaptiveRetriever",
    "AdaptivePlanner",
    # ablation
    "AblationConfig",
    "AblationBenchmarkResult",
    "AblationConditionResult",
    "AblationV3Report",
    "AblationV3Runner",
    "ABLATION_CONDITIONS",
]

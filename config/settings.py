"""
Blix v0.2 configuration settings — Python 3.10 compatible.

All settings are read from environment variables (via .env) first,
then fall back to blix.yaml, then to coded defaults.

Priority: .env  >  blix.yaml  >  code defaults

Usage
-----
    from config.settings import settings
    print(settings.embed_model)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from project root (silently ok if missing)
_ROOT_DIR: Path = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT_DIR / ".env", override=False)

ROOT_DIR: Path = _ROOT_DIR
MEMORY_DIR: Path = ROOT_DIR / "memory"
CONFIG_FILE: Path = ROOT_DIR / "config" / "blix.yaml"


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


class LLMSettings(BaseModel):
    """Settings for the chat LLM provider."""

    provider: str = Field(default="transformers", description="'transformers' or 'ollama'")
    model: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="HuggingFace model id or Ollama tag.",
    )
    ollama_model: str = Field(default="llama3.2", description="Ollama model tag (ollama provider only).")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=512, gt=0)


class EmbedSettings(BaseModel):
    """Settings for the semantic embedding retriever."""

    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model name.",
    )
    threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="Cosine similarity cutoff.")
    top_k: int = Field(default=5, gt=0, description="Max results from semantic search.")
    embeddings_file: Path = MEMORY_DIR / "embeddings.npy"
    embedding_ids_file: Path = MEMORY_DIR / "embedding_ids.json"


class MemorySettings(BaseModel):
    """Controls for the memory subsystem."""

    conversations_file: Path = MEMORY_DIR / "conversations.json"
    profile_file: Path = MEMORY_DIR / "profile.json"
    learning_state_file: Path = MEMORY_DIR / "learning_state.json"

    # Legacy retriever knobs (still used as fallback)
    recent_k: int = Field(default=5, gt=0)
    fuzzy_top_k: int = Field(default=3, gt=0)
    fuzzy_threshold: float = Field(default=60.0, ge=0.0, le=100.0)
    keyword_top_k: int = Field(default=3, gt=0)

    # Auto memory extraction
    auto_extract: bool = Field(default=True, description="Run CoT extractor after each turn.")


class AppSettings(BaseModel):
    """Top-level application settings."""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embed: EmbedSettings = Field(default_factory=EmbedSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    debug: bool = False


# ---------------------------------------------------------------------------
# Loader — env vars override yaml
# ---------------------------------------------------------------------------


def load_settings(path: Optional[Path] = None) -> AppSettings:
    """
    Build ``AppSettings`` by layering sources:

    1.  YAML file (``config/blix.yaml``) for structured overrides.
    2.  Environment variables (read from ``.env`` + shell) for secrets
        and deployment-specific values.
    3.  Pydantic defaults for anything not specified.
    """
    target = path or CONFIG_FILE
    raw: dict = {}
    if target.exists():
        with target.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    # Env var overrides (dotenv already loaded above)
    env_llm: dict = {}
    if os.getenv("BLIX_LLM_PROVIDER"):
        env_llm["provider"] = os.environ["BLIX_LLM_PROVIDER"]
    if os.getenv("BLIX_LLM_MODEL"):
        env_llm["model"] = os.environ["BLIX_LLM_MODEL"]
    if os.getenv("BLIX_OLLAMA_MODEL"):
        env_llm["ollama_model"] = os.environ["BLIX_OLLAMA_MODEL"]
    if os.getenv("BLIX_TEMPERATURE"):
        env_llm["temperature"] = float(os.environ["BLIX_TEMPERATURE"])
    if os.getenv("BLIX_MAX_NEW_TOKENS"):
        env_llm["max_new_tokens"] = int(os.environ["BLIX_MAX_NEW_TOKENS"])
    if env_llm:
        raw.setdefault("llm", {}).update(env_llm)

    env_embed: dict = {}
    if os.getenv("BLIX_EMBED_MODEL"):
        env_embed["model"] = os.environ["BLIX_EMBED_MODEL"]
    if os.getenv("BLIX_SEMANTIC_THRESHOLD"):
        env_embed["threshold"] = float(os.environ["BLIX_SEMANTIC_THRESHOLD"])
    if os.getenv("BLIX_SEMANTIC_TOP_K"):
        env_embed["top_k"] = int(os.environ["BLIX_SEMANTIC_TOP_K"])
    if env_embed:
        raw.setdefault("embed", {}).update(env_embed)

    if os.getenv("BLIX_AUTO_EXTRACT"):
        val = os.environ["BLIX_AUTO_EXTRACT"].lower()
        raw.setdefault("memory", {})["auto_extract"] = val not in ("0", "false", "no")

    return AppSettings.model_validate(raw)


settings: AppSettings = load_settings()


# ---------------------------------------------------------------------------
# v0.3 settings additions
# ---------------------------------------------------------------------------


class HierarchySettings(BaseModel):
    """Settings for the memory hierarchy manager."""

    hierarchy_dir: Path = MEMORY_DIR / "hierarchy"
    session_idle_gap_minutes: int = Field(
        default=30, description="Gap with no messages that starts a new session."
    )
    auto_daily_rollup: bool = True
    auto_weekly_rollup: bool = True


class GraphSettings(BaseModel):
    """Settings for the memory graph."""

    graph_file: Path = MEMORY_DIR / "graph.json"
    enabled: bool = True


class ProjectSettings(BaseModel):
    """Settings for project memory."""

    projects_file: Path = MEMORY_DIR / "projects.json"


class ProfileSettings(BaseModel):
    """Settings for the versioned profile evolver."""

    versioned_profile_file: Path = MEMORY_DIR / "versioned_profile.json"


class BackgroundSettings(BaseModel):
    """Settings for the background processor."""

    enabled: bool = True
    worker_count: int = Field(default=1, gt=0)
    max_queue_size: int = Field(default=100, gt=0)


class ScoringSettings(BaseModel):
    """Configurable weights for the memory scorer."""

    relevance_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    importance_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    frequency_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    recency_half_life_days: float = Field(default=30.0, gt=0.0)


# ---------------------------------------------------------------------------
# v0.3.16 ADMA + HGSHM settings (ISSUE-014)
# ---------------------------------------------------------------------------

import json
import os
from copy import deepcopy
from typing import Any

from config.schema import (
    ADMASettings,
    HGSHMSettings,
    FeatureFlags,
    Profile,
    _PROFILE_OVERRIDES,
)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_adma_settings(
    yaml_overrides: dict | None = None,
    profile: Profile | None = None,
) -> ADMASettings:
    """
    Build ADMASettings by layering sources (lowest to highest priority):
      1. Schema defaults
      2. Profile overrides (BLIX_PROFILE env var or ``profile`` argument)
      3. YAML overrides (from blix.yaml [adma] section or caller-supplied dict)
      4. Individual env var overrides (BLIX_ADMA_*)

    Parameters
    ----------
    yaml_overrides : dict | None
        Dict from the [adma] section of blix.yaml.
    profile : str | None
        One of development / testing / benchmark / production.
        Defaults to the BLIX_PROFILE env var, then None.
    """
    raw: dict[str, Any] = {}

    # ── Profile overrides ────────────────────────────────────────────
    active_profile = profile or os.getenv("BLIX_PROFILE")
    if active_profile and active_profile in _PROFILE_OVERRIDES:
        profile_raw = _PROFILE_OVERRIDES[active_profile].get("adma", {})
        raw = _deep_merge(raw, profile_raw)

    # ── YAML overrides ───────────────────────────────────────────────
    if yaml_overrides:
        raw = _deep_merge(raw, yaml_overrides)

    # ── Per-variable env var overrides ───────────────────────────────
    env_map = {
        "BLIX_ADMA_DECAY_FACTOR":           ("learner",   "decay_factor",         float),
        "BLIX_ADMA_REWARD_THRESHOLD":        ("learner",   "reward_threshold",      float),
        "BLIX_ADMA_SNAPSHOT_EVERY":          ("learner",   "snapshot_every",        int),
        "BLIX_ADMA_DECAY_PERSIST_EVERY":     ("learner",   "decay_persist_every",   int),
        "BLIX_ADMA_CACHE_MAX_SIZE":          ("learner",   "cache_max_size",         int),
        "BLIX_ADMA_MIN_OBSERVATIONS":        ("optimizer", "min_observations",       int),
        "BLIX_ADMA_AGING_THRESHOLD":         ("optimizer", "aging_threshold",        float),
        "BLIX_ADMA_CONVERGENCE_WINDOW":      ("optimizer", "convergence_window",     int),
        "BLIX_ADMA_CONVERGENCE_TOLERANCE":   ("optimizer", "convergence_tolerance",  float),
        "BLIX_ADMA_MUTATION_SCALE":          ("optimizer", "mutation_scale",          float),
        "BLIX_ADMA_ROLLBACK_DROP_THRESHOLD": ("optimizer", "rollback_drop_threshold", float),
        "BLIX_ADMA_LATENCY_TARGET_MS":       ("reward_engine", "latency_target_ms", float),
        "BLIX_ADMA_TOKEN_BUDGET":            ("reward_engine", "token_budget",       int),
        "BLIX_ADMA_MAX_ROWS_PER_POLICY":     ("reward_log",    "max_rows_per_policy", int),
        "BLIX_ADMA_TOKEN_BUDGET_COMPILER":   ("prompt_compiler", "token_budget",     int),
        "BLIX_ADMA_MAX_MEMORY_NODES":        ("prompt_compiler", "max_memory_nodes", int),
    }
    for env_key, (section, field, cast) in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            raw.setdefault(section, {})[field] = cast(val)

    return ADMASettings.model_validate(raw)


def load_hgshm_settings(
    yaml_overrides: dict | None = None,
    profile: Profile | None = None,
) -> HGSHMSettings:
    """Build HGSHMSettings with the same layering as ADMA."""
    raw: dict[str, Any] = {}

    active_profile = profile or os.getenv("BLIX_PROFILE")
    if active_profile and active_profile in _PROFILE_OVERRIDES:
        profile_raw = _PROFILE_OVERRIDES[active_profile].get("hgshm", {})
        raw = _deep_merge(raw, profile_raw)

    if yaml_overrides:
        raw = _deep_merge(raw, yaml_overrides)

    env_map = {
        "BLIX_HGSHM_EMBEDDING_DIM":          ("embedding",     "dim",                    int),
        "BLIX_HGSHM_TOP_K":                  ("retrieval",     "default_top_k",          int),
        "BLIX_HGSHM_MAX_AGE_HOURS":          ("retrieval",     "max_age_hours",           float),
        "BLIX_HGSHM_MIN_SCORE":              ("retrieval",     "min_score",               float),
        "BLIX_HGSHM_GRAPH_MAX_DEPTH":        ("retrieval",     "graph_max_depth",         int),
        "BLIX_HGSHM_GRAPH_MAX_NODES":        ("retrieval",     "graph_max_nodes",         int),
        "BLIX_HGSHM_SIM_THRESHOLD":          ("consolidation", "similarity_threshold",    float),
        "BLIX_HGSHM_JACCARD_THRESHOLD":      ("consolidation", "jaccard_threshold",        float),
        "BLIX_HGSHM_PRUNE_IMPORTANCE":       ("consolidation", "prune_below_importance",  float),
        "BLIX_HGSHM_PRUNE_CONFIDENCE":       ("consolidation", "prune_below_confidence",  float),
        "BLIX_HGSHM_SUMMARY_MAX_LENGTH":     ("hierarchy",     "max_summary_length",      int),
        "BLIX_HGSHM_MIN_CLUSTER_SIZE":       ("hierarchy",     "min_cluster_size",        int),
        "BLIX_DB_HGSHM":                     ("database",      "hgshm_db",                str),
        "BLIX_DB_VECTORS":                   ("database",      "vectors_db",              str),
        "BLIX_DB_POLICY":                    ("database",      "policy_db",               str),
    }
    for env_key, (section, field, cast) in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            raw.setdefault(section, {})[field] = cast(val)

    return HGSHMSettings.model_validate(raw)


def load_feature_flags(yaml_overrides: dict | None = None) -> FeatureFlags:
    """Build FeatureFlags from yaml and env vars."""
    raw: dict[str, Any] = {}
    if yaml_overrides:
        raw.update(yaml_overrides)

    env_map = {
        "BLIX_FEATURE_HYBRID_RETRIEVAL":     "hybrid_retrieval",
        "BLIX_FEATURE_ADAPTIVE_POLICY":      "adaptive_policy",
        "BLIX_FEATURE_GRAPH_REASONING":      "graph_reasoning",
        "BLIX_FEATURE_HIERARCHY":            "hierarchy",
        "BLIX_FEATURE_SEMANTIC_SEARCH":      "semantic_search",
        "BLIX_FEATURE_REWARD_LEARNING":      "reward_learning",
        "BLIX_FEATURE_PROMPT_COMPILER":      "prompt_compiler",
        "BLIX_FEATURE_MEMORY_CONSOLIDATION": "memory_consolidation",
    }
    for env_key, field in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            raw[field] = val.lower() not in ("0", "false", "no", "off")

    return FeatureFlags.model_validate(raw)


def _load_full_yaml() -> dict[str, Any]:
    """Read blix.yaml and return the raw dict (empty if missing)."""
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


# Build the v0.3.16 settings singletons from blix.yaml + env
_yaml = _load_full_yaml()
adma_settings: ADMASettings  = load_adma_settings(_yaml.get("adma"))
hgshm_settings: HGSHMSettings = load_hgshm_settings(_yaml.get("hgshm"))
feature_flags: FeatureFlags   = load_feature_flags(_yaml.get("features"))


# ---------------------------------------------------------------------------
# Configuration snapshot export (for experiment reproducibility)
# ---------------------------------------------------------------------------

def export_config_snapshot(output_dir: Path | str | None = None) -> dict[str, Any]:
    """
    Export the fully resolved configuration to a dict and optionally to disk.

    Writes ``config_snapshot.yaml`` and ``config_snapshot.json`` in
    ``output_dir`` (defaults to ``results/`` under the project root).

    Returns the snapshot dict.

    Usage
    -----
    >>> from config.settings import export_config_snapshot
    >>> snap = export_config_snapshot("results/run_001")
    """
    import platform, sys
    from datetime import datetime, timezone

    snapshot: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile":      os.getenv("BLIX_PROFILE", "default"),
        "python":       sys.version,
        "platform":     platform.platform(),
        "adma":         adma_settings.model_dump(),
        "hgshm":        hgshm_settings.model_dump(),
        "features":     feature_flags.model_dump(),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON
        (out / "config_snapshot.json").write_text(
            json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

        # YAML (only if pyyaml available — it is, we import yaml above)
        with (out / "config_snapshot.yaml").open("w", encoding="utf-8") as fh:
            yaml.dump(snapshot, fh, default_flow_style=False, allow_unicode=True)

    return snapshot

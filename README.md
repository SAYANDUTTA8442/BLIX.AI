<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,40:0f172a,80:1e1b4b,100:4f46e5&height=140&section=header&text=Blix&fontSize=56&fontColor=ffffff&fontAlignY=58&fontAlign=50&desc=Cognitive%20Agent%20Architecture%20%C2%B7%20Memory%20%C2%B7%20Reasoning%20%C2%B7%20Recovery&descSize=14&descAlignY=80&descColor=a5b4fc&animation=fadeIn" width="100%"/>

</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-a5b4fc?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-4f46e5?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![sqlite-vec](https://img.shields.io/badge/sqlite--vec-Vector%20Search-0467df?style=flat-square&logo=sqlite&logoColor=white)](https://github.com/asg017/sqlite-vec)
[![Status](https://img.shields.io/badge/Status-Active%20Development-22c55e?style=flat-square)]()
[![Version](https://img.shields.io/badge/Version-v0.3.19.9-a5b4fc?style=flat-square)]()
[![Tests](https://img.shields.io/badge/Tests-3%2C055%20passing-22c55e?style=flat-square)]()
[![Audit](https://img.shields.io/badge/Audit%20Trail-3%20review%20passes%20%C2%B7%20all%20closed-22c55e?style=flat-square)]()

<br/>

*Built by [Sayan Dutta](https://sayandutta.netlify.app) · AI Researcher · IIT Patna*

</div>

---

> **Blix is not a chatbot wrapper.**
> It is a ground-up cognitive agent architecture exploring what it takes for an AI system to **remember, reason, verify, and adapt** in a principled way — combining hierarchical graph-vector memory, causal reasoning, self-reflective metacognition, and online policy learning, running entirely on local infrastructure.

---

## Table of Contents

- [Why Blix Exists](#-why-blix-exists)
- [Architecture](#-architecture)
  - [HGSHM — Hybrid Graph-Based Semantic Hierarchical Memory](#-hgshm--hybrid-graph-based-semantic-hierarchical-memory)
  - [ADMA — Adaptive Dual Memory Architecture](#-adma--adaptive-dual-memory-architecture)
  - [Causal Reasoning & Truth Maintenance](#-causal-reasoning--truth-maintenance)
  - [Planning & Workspace Coordination](#-planning--workspace-coordination)
  - [Metacognition & Reflection](#-metacognition--reflection)
  - [Agents, Events & Procedural Learning](#-agents-events--procedural-learning)
  - [Curiosity & Autonomous Research](#-curiosity--autonomous-research)
- [StateBench & Paper-Replication Harness](#-statebench--paper-replication-harness)
- [Engineering Practice — Audit-Driven Development](#-engineering-practice--audit-driven-development)
- [Multi-Tenancy & Isolation](#-multi-tenancy--isolation)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Roadmap](#-roadmap)
- [Research Context](#-research-context)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🔍 Why Blix Exists

Most LLM applications today are stateless wrappers: they call an API, format a prompt, and return a response. Between sessions, they remember nothing. Within sessions, they accumulate a context window until it overflows. They have no explicit belief state, no way to detect when a belief is wrong, and no mechanism to *learn from their own behavior* without a training run.

**The problems Blix is built to solve:**

| Problem | What most systems do | What Blix does |
|---|---|---|
| Context loss between sessions | Forget everything | Persistent hierarchical memory (Raw → Session → Daily → Weekly → Project → Principle → World Model) |
| Contradictory knowledge | Accept any new claim silently | `TruthManager` + `ContradictionResolver` detect and repair inconsistencies |
| No model of belief | No explicit belief state | Confidence-scored belief store with a full revision audit trail |
| Static memory | Flat key-value or vector store | Fused graph + vector + hierarchical retrieval (`HybridRetriever`, 11-factor ranking) |
| Fixed heuristics | Hand-tuned config forever | Online **policy learning** (multi-armed bandits) adapts retrieval weights, planning aggressiveness, and prompt style from observed outcomes — no retraining |
| Failure is unrecoverable | Crash or hallucinate | Dedicated failure memory + principled replanning (`agents/failure_memory.py`, `planning/replanner.py`) |
| Cloud dependency | API calls required | Fully local via Ollama or HuggingFace Transformers |
| Silent regressions | Ship and hope | Every finding from three independent audit passes is logged, numbered, and closed with a dedicated regression test — see [Engineering Practice](#-engineering-practice--audit-driven-development) |
| Monolithic design | Hard to extend or ablate | 32 independently testable packages, dependency-injected ablation harness (`policy/ablation_v3.py`) |

---

## 🏗️ Architecture

Blix has grown, across dozens of versioned releases (v0.3 → v0.3.19.9), from a memory-augmented chat tutor into a full cognitive stack of **32 cooperating packages**, **248 source modules**, and roughly **52,300 lines of Python** (plus ~31,000 lines of tests), each independently testable and injectable for ablation studies.

---

### 🧠 HGSHM — Hybrid Graph-Based Semantic Hierarchical Memory

The foundation of Blix's memory (introduced v0.3.15, replacing the earlier flat FAISS index).

```
┌─────────────────────────────────────────────────────────────────┐
│                            HGSHM                                │
├─────────────────────────────────────────────────────────────────┤
│  GraphStore (hgshm.db)  ·  VectorIndex (sqlite-vec, cosine ANN) │
│  HierarchyManager  ·  ConsolidationEngine  ·  ContextBuilder    │
│  HybridRetriever — 11-factor ranked fusion                      │
└─────────────────────────────────────────────────────────────────┘

Raw → Episode → Conversation → Session → Daily → Weekly
   → Monthly → Project → Concept → Principle → Knowledge → WorldModel
```

- **Vector layer** — `sqlite-vec`-backed ANN search (`memory/hybrid/vector/`), with a pure-Python brute-force cosine fallback when the extension isn't installed (the fallback now logs a `WARNING` at module load, closing audit item A13).
- **Graph layer** — entity/relation storage with a **20-relation `EdgeRelation` enum** (`supports`, `contradicts`, `causes`, `depends_on`, `part_of`, `derived_from`, `similar_to`, `explains`, `references`, `precedes`, `follows`, `belongs_to`, `requires`, `related_to`, `enables`, `blocks`, `summarises`, `instance_of`, `evolves_to`, `co_occurs`), full traversal, and an `INVERSE_RELATIONS` map for bidirectional propagation (`memory/hybrid/graph/`, `memory/hybrid/models/memory_edge.py`).
- **Hierarchy layer** — automatic compression from raw memories up through session → daily → weekly → project summaries; `SUMMARISES` edges preserve every source reference so nothing is silently lost.
- **Consolidation** — a background engine merges, dedupes, and rolls up memory nodes on a schedule (`consolidation_engine.py`); the original full-table-scan cost on every run (A05) was replaced with SQL-pushdown filtering (`max_confidence`/`max_importance` bounds propagated through `GraphStore`).
- **Configurable pipeline** — the 11-step context-assembly pipeline (temporal/concept/principle/belief sections) that used to hard-limit itself is now exposed through a `ContextBuilderSettings` sub-schema (A30), and no longer silently floors small `top_k` values to zero (fixed twice — once in `MemoryManager.query()` for the A19-era code path, once again in `context_builder.py` itself during the final v0.3.19.7 audit pass).
- Query pipeline: `Semantic · Vector · Graph Expansion · Temporal → Importance Ranking → Hierarchy → Contradictions → Causal → typed MemoryContext → Reasoning`.

---

### ⚙️ ADMA — Adaptive Dual Memory Architecture

Static configuration is replaced by **online policy learning** — Blix tunes its own retrieval/planning/prompting behavior from observed outcomes, without gradient updates or RLHF. Introduced in v0.3.16 and hardened through every subsequent audit cycle.

```
Experience
    │
    ├───────────────────────────────────────┐
    │ System Experience                     │ User Experience
    ▼                                       ▼
SystemMemory                           UserMemory
 (operational knowledge)          (per-user personalisation)
    │            both backed by HGSHM       │
    └──────────────────┬────────────────────┘
                        ▼
        PolicyLearner — Thompson Sampling over Beta(α, β) arms
                        │
                  PolicyCompiler
              (dynamic prompt assembly)
                        │
              Planner + Global Workspace
                        │
                       LLM
```

- **Dual memory** — `SystemMemory` (workflow traces, benchmark history, failure patterns, operational principles) and `UserMemory` (preferences, corrections, goals, learning progress) as separate domains, unified by `MemoryManager`.
- **Policy memory** — **15 `PolicyType` arms** (8 system-side: retrieval weights, planner config, verification policy, tool selection, reasoning strategy, workspace config, compression policy, memory routing; 7 user-side: answer style, difficulty level, explanation depth, topic preference, feedback style, goal priority, hint policy).
- **Reward engine** — **15 `RewardType` signals** (8 system: benchmark score, latency, verification success, planner success, memory quality, token efficiency, regression stability, failure recovery; 7 user: answer accepted, correction given, task completed, follow-up asked, preference signal, goal advanced, repeated usage) drive arm updates.
- **Learning rule** — reward ≥ 0.5 increments α (fractionally), reward < 0.5 increments β; temporal decay pulls α → 1 + (α−1)×0.995 per observation (~139-observation half-life); automatic rollback if recent mean confidence drops >0.10 vs. the historical mean.
- **Correctly named algorithm** — `policy/learner.py` documents this as a **multi-armed bandit with context-scoped reward attribution**, not a contextual bandit: the arm pool and Thompson-sampling distribution are not themselves conditioned on context features, only which arm receives credit is (framing corrected in the v0.3.19.1 external-review pass, C09).
- **Concurrency-hardened store** — `PolicyStore` originally had unprotected concurrent reads (`get()`, `all_active()`, `get_history()`, `reward_log_count()`, `recent_rewards()`) alongside a locked `save()`/`log_reward()`, which under 20-thread load produced `sqlite3.InterfaceError` roughly 5–10% of the time and silently lost policy updates (~27 lost updates per 2,000 operations) via a non-atomic read-modify-write in `_update_arm()`. Both were reproduced and fixed: every read now holds `self._lock`, and updates go through `update_atomic()` (a `BEGIN IMMEDIATE` transaction).
- **Cache correctness** — the LRU policy cache no longer double-applies decay on eviction/re-fetch (A01, via a per-arm `_epoch_at_last_write` delta); `PolicyLearner.register()` and the in-memory cache are thread-safe (A14, A29); the `_get_hgshm()` shim registry's check-then-create race (two threads both passing a "not registered" check and orphaning a SQLite connection) is now closed under a `threading.Lock` with double-checked locking.
- **Ablation framework v3** — dependency injection replaces the old env-flag mechanism, so every one of the 7 injectable ADMA components can be swapped or stubbed for controlled experiments (`policy/ablation_v3.py`).

---

### ⚖️ Causal Reasoning & Truth Maintenance

Blix maintains an explicit causal and belief model, not just a similarity index.

- **`core/truth_manager.py` + `core/contradiction_resolver.py`** — detect direct contradictions between incoming claims and stored beliefs, resolve by confidence/recency, and propagate updates to dependent beliefs, with a full audit trail.
- **`causality/`** — `CauseGraph`, `CausalMemory`, `CounterfactualEngine`, `CausalReflection` / `MetaCausalReflection`, `BeliefDependencyGraph`, `PrincipleSynthesizer`, `PrincipleGraph`, `EpistemicStatus` — a dedicated package for building and reflecting on cause-effect structure, not just correlational retrieval.
- **`reasoning/`** — `ConfidenceModel`, `ConfidenceReasoner`, and `temporal_query.py` for asking "what did I believe about X at time T?".

---

### 🖥️ Planning & Workspace Coordination

The executive layer that decides what to do and whether to trust the result.

- **`planning/`** — `BeamSearchPlanner`, `Planner`, `Critic`, `SearchCritic`, `PlanEvaluator`, `Replanner` — multi-candidate plan search with scoring and revision, not single-shot generation.
- **`workspace/`** — `GlobalWorkspace`, `BroadcastBus`, `NeuralAttentionManager`, `AttentionManager`, `InnerDialogue`, `WorkspaceMemory`, `snapshot.py` — a global-workspace-theory-inspired coordination layer where subsystems broadcast and compete for attention before a response commits. Each `GlobalWorkspace` instance shares no state with any other (verified explicitly, B11) — separate instances give clean per-user isolation.
- **`verification/`** — an independent `Verifier` checks retrieved evidence and logical consistency before commit.
- **`specialists/`** — planning, memory, verification, and reflection specialists vote via a `consensus.py` module rather than a single component deciding alone.
- **`world_model/`** — `LatentWorldModel`, `ScenarioRanker`, `ValueNetwork` for forward simulation and ranking candidate futures; `simulation/trajectory_graph.py` tracks the resulting branches.

---

### 🪞 Metacognition & Reflection

Blix models *itself*, not just the world.

- **`metacognition/`** — `SelfModel`, `StrategyEvolution`, `StrategySelector` / `StrategyManager`, `ConfidenceManager`, `CapabilityTracker`, `controller.py` — tracks what Blix is good at, evolves its own strategies, and calibrates confidence.
- **`reflection/`** — `ReflectionEngine`, `MetaReflection`, `StateReflection`, `InsightEngine`, `ConsolidationEngine`, `GoalTracker`, `ProjectIntelligence`, `scheduler.py`, and a **Memory Query Language** (`mql.py`, `mql_v2.py`) that lets a user inspect memory directly with commands like `show active goals`, `show contradictions`, or `show strongest skills`.
- **`evaluation/`** — a large internal benchmark harness (`agent_benchmark.py`, `capability_metrics.py`, `cognitive.py`, `coordination_metrics.py`, `metacognition_metrics.py`, `workspace_metrics.py`, `state_metrics.py`, `attention_metrics.py`, `research.py`) plus a CLI runner — this is where **StateBench** lives.

---

### 🤖 Agents, Events & Procedural Learning

- **`agents/`** — `Executor`, `TaskRuntime`, `WorkingMemory`, `ExecutionFeedback`, `FailureMemory`, `ToolReliability`, `ToolSuccessPredictor`, `PlanReflection`, `ReflectionLoop`, `observation.py`, `state.py`, `types.py` — a full agent execution loop with tool-outcome tracking, not just a plan-then-execute pipeline.
- **`events/`** — a synchronous, in-process `EventBus` with a typed `EventType` vocabulary (`TASK_COMPLETED`, `FAILURE`, `BELIEF_UPDATED`, `STATE_CHANGED`, …) and an `event_store.py` for persistence, so subsystems can react to each other's events without directly calling into one another. Deliberately synchronous and broker-free — Blix is single-process, and this is the right level of complexity for that.
- **`procedural/`** — `SkillDiscoveryEngine`, which passively mines reusable skills from completed, successful `TaskGraph` executions (Voyager-inspired), feeding them into `memory/procedural_memory.py`'s `ProceduralMemory` store rather than requiring every caller to remember to report what it learned.

---

### 🔬 Curiosity & Autonomous Research

- **`curiosity/` + `hypothesis/` + `experiments/`** — `CuriosityEngine`, `HypothesisManager`, `ExperimentPlanner`, and `knowledge/knowledge_gap_tracker.py`'s `KnowledgeGapTracker` drive autonomous, self-directed exploration: Blix can notice what it doesn't know and plan an experiment to find out.
- **`knowledge/`** — `DocumentProcessor` (PDF/TXT/MD/DOCX/HTML ingestion via `pdfplumber` + `python-docx`), `MediaProcessor` (OCR via `pytesseract` + `Pillow`), `ResearchAssistant`, `synthesis.py`.
- **`api/`** — a FastAPI surface (`Blix — Cognitive Knowledge Platform`) exposing **21 routers**: chat, memory, knowledge, reflection, graph, documents, stats, goals, reasoning-research, agent, agents, temporal, metacognition, workspace, ml, causality, search, curiosity, world_model, simulation, specialists. The app now carries a request-body size limit (413 on oversize, with a clean 400 rather than an unhandled 500 on a malformed `Content-Length` header), an `X-Request-ID` middleware so every response is traceable, and a corrected CORS configuration (wildcard origins can no longer be combined with credentialed requests).

---

## 📐 StateBench & Paper-Replication Harness

**StateBench** is Blix's internal benchmark suite for evaluating cognitive subsystems independently, run through `evaluation/cli.py` and the `evaluation/` metrics modules described above. Its core metric, the **State Hallucination Rate**, measures how often Blix's stated belief about something diverges from what actually happened — turning "does it remember correctly" into a number instead of a vibe.

| Benchmark area | What it evaluates |
|---|---|
| State tracking | Does Blix correctly maintain and update belief state across a multi-turn sequence? |
| Contradiction detection | Does the Truth Maintenance Engine catch planted contradictions? |
| Retrieval fidelity | Does `HybridRetriever` surface the most contextually relevant items across vector + graph + temporal signals? |
| Replan success rate | When the first plan fails, does replanning recover a correct response? |
| Temporal decay | Does memory correctly down-weight stale information over time? |
| Confidence calibration | Are confidence scores meaningful predictors of response accuracy? |
| Policy convergence | Do bandit arms converge to a stable, high-reward configuration, and does rollback trigger correctly on regression? |

Beyond the internal suite, `eval_harness.py` (with `run_benchmark.sh` and `README_EVAL.md` as a standalone replication guide) runs a full **ablation study against external QA benchmarks** — **HotpotQA, LoCoMo, NarrativeQA, and StreamingQA** — across 4 ablation profiles (`full`, `no_graph`, `no_adma`, `both`) and multiple seeds, producing per-dataset CSVs, a bootstrap-tested `aggregate_summary.json`, paper-ready ablation tables, and LaTeX-formatted combined summaries. Optional NLI-based faithfulness metrics (entailment score, hallucination rate) run against `roberta-large-mnli` when `transformers` + a GPU are available. The harness's own cross-module wiring was itself audited and fixed late in the cycle — it had been importing the wrong `MemoryManager` (the HGSHM-based one instead of `TutorAgent`'s `core.memory_manager.MemoryManager`), calling a `create_provider()` function that had been renamed to `build_provider()`, and using an `EmbeddingStore` constructor signature that had drifted from the eval code.

**Full repository test suite: 3,055 tests, all passing** (verified against the v0.3.19.9 source tree; up from 194 at v0.3.15 and 712 at v0.3.16.9). StateBench results specific to the v0.3.1 milestone are documented separately in a 41-page Minor Project I academic report.

---

## 🩺 Engineering Practice — Audit-Driven Development

Blix is not fixed reactively — it is audited on a schedule, and every fix ships with a dedicated regression test file so a future regression has somewhere obvious to be caught. Three review passes have run against the codebase so far:

**Pass 1 — internal static + dynamic audit (A-series, `Issues.md`, generated 2026-08-02 against v0.3.16.9):** full static analysis across 280 Python modules, dynamic behaviour analysis, algorithm-correctness review against each subsystem's stated guarantees, and comparison against production standards. 30 issues catalogued by severity (HIGH/MEDIUM/LOW), each with root cause, a concrete failure scenario, a recommended fix, and an effort estimate.

| Batch | Issues closed |
|---|---|
| A01–A04 | LRU double-decay, config error swallowing, HGSHM double-close, reward-log counter reset |
| A05, A06, A11 | Consolidation full-table scan, hardcoded DB filenames, ignored `mutation_scale` |
| A07–A10 | Unconditional bandit reward broadcast, `PolicyVersion` field fragility, unsafe `**kwargs`, unbounded shim registry |
| A15 | Test coverage for `AdaptiveRetriever` / `AdaptivePlanner` |
| A19 | Deterministic settings-import timing |
| A31 | KV-cache / state isolation verified across LLM providers |
| A12–A14 | Broad exception handling in the retrieval hot path, unsurfaced vector-store fallback, `PolicyLearner` cache thread-safety |
| A16, A17 | Removed runtime `print()` calls from production modules, added `PolicyVersion` test coverage |
| A18 | `RoutedContext.to_memory_context()` no longer drops supporting memories |
| A20, A21 | Extended injection-pattern sanitisation (incl. memory snippets), explicit-column `PolicyStore` SQL + schema validation |
| A22, A23 | System-instruction length cap in `PolicyCompiler`, normalised `HybridWeightsSettings.to_dict()` |
| A24–A26 | `EmbeddingManager.close()`, populated `__all__` across `memory/`, `policy_versions` no longer silently drops rows on conflict |
| A27, A28 | `UserMemory.store_interaction()` records response content, `export_config_snapshot()` degrades gracefully without `pyyaml` |
| A29, A30 | Atomic `PolicyLearner.register()`, configurable `ContextBuilder` pipeline limits |

**Pass 2 — internal follow-up (B-series):** issues found while closing out Pass 1.

| Batch | Issues closed |
|---|---|
| B01–B03 | Atomic `PolicyStore.update_atomic()`, `pyproject.toml`/`blix.__version__` alignment, removed hardcoded absolute paths from tests |
| B04, B05, B09, B10 | Semantic-similarity `fact_accuracy()`, leakage-free `temporal_split()`, unicode-evasion injection patterns, per-`user_id`-scoped HGSHM shim registry |
| B11, B12 | Verified `GlobalWorkspace` has no shared/singleton state; upgraded silent `except Exception` passes in `memory/future_memory.py` and `api/context.py` to typed, logged warnings |

**Pass 3 — external review (C-series) + final internal sweep:** an outside read of the codebase surfaced production-hardening gaps the internal audits hadn't framed as bugs, plus a last pass caught real concurrency bugs by reproduction rather than code review alone.

| Batch | Issues closed |
|---|---|
| C01 | Reproduced (not just theorised) `PolicyStore` race conditions — unlocked reads causing `sqlite3.InterfaceError` at 5–10% under load, and a non-atomic `_update_arm()` losing ~27 updates per 2,000 ops |
| C03–C06, C08, C10 | CORS wildcard+credentials fix, corrected build backend, `blix.app:main` entry point, removed unsafe builtins from the Python tool sandbox, enforced document-upload size limit, `graph_consistency` no longer defaults to 1.0 on empty predictions |
| C07, C11, C12, C14 | Request body size limit middleware, version-currency check, `X-Request-ID` traceability middleware, single-source version via `importlib.metadata` |
| C09 | Corrected documentation/framing: "contextual bandit" → "multi-armed bandit with context-scoped reward attribution" (no algorithm change) |
| Final sweep | `context_builder.py`'s `k // 2` / `k // 3` floors fixed for small `top_k` (same class as an earlier bug); malformed `Content-Length` header now returns 400 instead of crashing; a missing `await call_next(request)` argument that silently returned coroutine objects instead of responses; four cross-module import/signature mismatches in `eval_harness.py` found before the first full benchmark run |

Every batch above ships with its own test file — `tests/test_v03170_a01_a04.py` through `tests/test_v03199_harness_prereqs.py` — and **all 3,055 tests pass** against the current tree. Zero `TODO`/`FIXME` markers remain in shipped source.

---

## 🔐 Multi-Tenancy & Isolation

Blix is primarily a **single-tenant, local-first** system, but several components support multi-user deployments with varying isolation guarantees (documented in full in `TENANCY.md`):

| Component | Isolation |
|---|---|
| `UserMemory` | ✅ Isolated by constructor-provided `user_id`; queries filter accordingly |
| `PolicyStore` | ✅ `user_id`-indexed column; policies with `user_id=None` are system-wide defaults visible to all users |
| `PolicyLearner` LRU cache | ✅ Keyed by `policy_id`, which inherits per-user isolation from the DB layer |
| `GlobalWorkspace` | ✅ Isolated by instance — no shared state, construct one per user |
| HGSHM shim registry | ⚠️ Partially isolated — `_get_hgshm(memory_dir, user_id=...)` scopes the registry key when a `user_id` is passed, but legacy shims (`BeliefStoreShim`, `CauseGraphShim`, `PrincipleStoreShim`) don't accept one yet |
| HGSHM itself | ⚠️ Not isolated by default — a single instance is a shared database; use separate instances or per-user directories for real isolation |
| `EmbeddingManager` | ✅ Model weights shared read-only across the process (by design); per-instance text→vector cache is cleared on `close()` |

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Language | Python 3.10+ | Core implementation |
| Deep Learning | PyTorch, HuggingFace Transformers | Local model inference and tokenization (`llm/transformers_provider.py`) |
| LLM Runtime | Ollama | Alternative local LLM backend (`llm/ollama_provider.py`) |
| Vector Search | sqlite-vec | Semantic memory ANN search, with pure-Python fallback |
| Embeddings | Sentence-Transformers | Semantic encoding for memory and graph |
| Graph + Hierarchy | Custom (`memory/hybrid/graph`, `memory/hybrid/hierarchy`) | Entity-relation storage, multi-hop traversal, rollup compression |
| Policy Learning | Custom Thompson-sampling multi-armed bandits (`policy/`) | Online, gradient-free adaptation of retrieval/planning/prompting |
| Database | SQLite (WAL mode) | `hgshm.db`, `policy.db`, hierarchy/vector persistence |
| Validation | Pydantic v2 | Schema enforcement across all subsystems |
| Document Parsing | pdfplumber, python-docx | PDF/DOCX ingestion |
| Media Processing | Pillow, pytesseract | OCR for images and scanned documents |
| API | FastAPI + Uvicorn + python-multipart + httpx | REST surface across 21 routers, with body-size-limit, request-ID, and CORS middleware |
| Fuzzy Matching | RapidFuzz | Entity/label deduplication in the memory graph |
| Evaluation | pandas, scipy, matplotlib/seaborn, sacrebleu, rouge-score, datasets | External-benchmark ablation harness (`eval_harness.py`) |
| Testing | Pytest | 3,055 tests across every subsystem |

**Design constraints:**
- ✅ Fully local — no cloud API calls required at runtime
- ✅ Privacy-preserving — all data stays on device
- ✅ Consumer hardware — designed to run on a standard laptop
- ✅ Modular — every subsystem has a clean interface and can be replaced or ablated independently
- ✅ Gradient-free adaptation — ADMA policy learning requires no retraining loop
- ✅ Zero `TODO`/`FIXME` markers in shipped source — outstanding work is tracked as numbered audit items, not code comments

---

## 📁 Project Structure

```
blix_v03/
├── app.py                     # CLI entry point (also reachable via `blix` console script)
├── blix/                      # Installable package shim — __version__, app.py:main bridge
├── pyproject.toml / requirements.txt
├── ARCHITECTURE.md            # Full design rationale
├── TENANCY.md                 # Multi-user isolation guide
├── README_EVAL.md             # Paper-replication guide for eval_harness.py
├── Issues.md                  # A-series audit backlog (severity, root cause, fix)
├── eval_harness.py / run_benchmark.sh / module_benchmark.py
├── CHANGELOG_v0.3.*.md        # Versioned changelogs
│
├── core/                      # TutorAgent orchestration, memory scoring,
│                               #   hierarchy, graph, truth maintenance,
│                               #   contradiction resolution, embeddings
├── memory/
│   ├── hybrid/                # HGSHM: graph · vector · hierarchy ·
│   │                           #   consolidation · context builder
│   ├── system/, user/          # ADMA dual memory domains
│   └── manager.py
├── policy/                    # ADMA: models, store, reward engine,
│                               #   learner (bandits), optimizer,
│                               #   compiler, adaptive retriever/planner
├── planning/                   # Beam search planner, critic, replanner
├── workspace/                  # Global workspace, broadcast bus,
│                               #   attention, inner dialogue
├── causality/                  # Cause graphs, counterfactuals,
│                               #   causal/meta-causal reflection
├── reasoning/                  # Confidence modeling, temporal queries
├── metacognition/               # Self-model, strategy evolution,
│                               #   capability tracking
├── reflection/                  # Reflection/insight engines, MQL,
│                               #   goal tracking, consolidation
├── agents/                     # Executor, task runtime, failure
│                               #   memory, tool reliability
├── events/                     # Typed EventBus + event store
├── procedural/                  # Passive skill discovery from
│                               #   successful task trajectories
├── curiosity/, hypothesis/, experiments/  # Autonomous exploration
├── knowledge/                  # Document/media ingestion, research
│                               #   assistant, synthesis
├── verification/, specialists/ # Independent verifier + consensus
│                               #   voting specialists
├── world_model/, simulation/   # Latent world model, scenario ranking
├── graph/                      # Temporal graph primitives
├── retrieval/                   # Temporal + active-attention retrievers,
│                               #   cross-encoder reranker
├── learning/                    # Continual adapter, failure clusterer
├── tools/                      # Tool registry
├── llm/                        # Provider factory: Transformers / Ollama
├── evaluation/                  # StateBench metrics + CLI runner
├── api/                        # FastAPI app + 21 routers
├── schemas/, config/, utils/    # Pydantic models, settings, helpers
└── tests/                       # 3,055 tests across every subsystem
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.10+
python --version

# Option A: Ollama (local LLM backend)
# Install from https://ollama.com
ollama pull llama3.2        # or mistral, phi3, gemma2

# Option B: HuggingFace Transformers (fully local, no separate server)
# handled directly via `transformers` + `torch`
```

### Installation

```bash
git clone https://github.com/SAYANDUTTA8442/blix.ai.git
cd blix.ai/blix_v03

python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -e .
# Optional extras:
pip install -e ".[transformers]"   # HF Transformers backend
pip install -e ".[ollama]"         # Ollama backend
pip install -e ".[api]"            # FastAPI server — fastapi, uvicorn,
                                    #   python-multipart, httpx
pip install -e ".[documents]"      # PDF/DOCX ingestion
pip install -e ".[media]"          # Image OCR
pip install -e ".[eval]"           # External-benchmark ablation harness
pip install -e ".[dev]"            # pytest, ruff, mypy
```

> **Note:** the API layer (`api/`) requires `fastapi`, `uvicorn`, and `python-multipart` even for local development — install these explicitly (or via the `[api]` extra) before importing anything under `api/` directly.

### Quick Start — HGSHM Memory

```python
from pathlib import Path
from memory.hybrid.hgshm import HGSHM

h = HGSHM(Path("memory/"))
h.believe("Deployment failed due to timeouts", confidence=0.85)
h.observe_cause("peak_traffic", "timeout_errors")
h.add_principle("Monitor timeouts before deployment windows")

ctx = h.recall("deployment failure root cause", top_k=10)
print(ctx.get_text_summary())
```

### Quick Start — CLI Tutor Agent

```bash
python app.py
# or, after `pip install -e .`:
blix
# /memory  /profile  /stats  /graph  /projects  /hierarchy  /eval
```

### Running the API

```bash
uvicorn api.server:app --reload
# Serves 21 routers: chat, memory, knowledge, reflection, graph,
# documents, stats, goals, reasoning-research, agent(s), temporal,
# metacognition, workspace, ml, causality, search, curiosity,
# world_model, simulation, specialists
```

### Running the Test Suite

```bash
pytest tests/ -q
# 3,055 passed
```

### Replicating the Paper Benchmark

```bash
bash run_benchmark.sh
# or:
python eval_harness.py \
    --datasets hotpotqa,locomo,narrativeqa,streamingqa \
    --samples 200 --seeds 42,43,44,45,46 \
    --profiles full,no_graph,no_adma,both \
    --output results/ --nli-metrics --profile-memory --visualize --verbose
```
See `README_EVAL.md` for full hardware/software requirements and expected runtime.

---

## 🗺️ Roadmap

### v0.3 — complete
- [x] Hierarchical memory (Raw → Session → Daily → Weekly → Project → Principle → WorldModel)
- [x] Hybrid Graph-Based Semantic Hierarchical Memory (HGSHM) — sqlite-vec + graph + hierarchy fusion
- [x] Truth Maintenance Engine — contradiction detection and belief repair
- [x] Causal reasoning package — cause graphs, counterfactuals, causal reflection
- [x] Workspace coordination — planner / verifier / evaluator / specialist consensus
- [x] Failure memory and replanning
- [x] Metacognition — self-model, strategy evolution, confidence calibration
- [x] Curiosity engine, hypothesis manager, experiment planner
- [x] Document + media ingestion (PDF, DOCX, HTML, OCR)
- [x] Typed event bus + passive procedural-skill discovery
- [x] REST API layer (FastAPI, 21 routers, hardened middleware)
- [x] **Adaptive Dual Memory Architecture (ADMA)** — multi-armed-bandit policy learning, dynamic prompt compiler, dependency-injected ablation framework
- [x] StateBench benchmark suite — 3,055 passing tests
- [x] External-benchmark ablation harness — HotpotQA / LoCoMo / NarrativeQA / StreamingQA
- [x] Three independent audit passes (A-series internal, B-series follow-up, C-series external review) — all closed with regression tests
- [x] Documented multi-tenancy isolation model (`TENANCY.md`)

### v0.4 — Cognitive Kernel
- [ ] Unify HGSHM + ADMA policy memory under a single kernel API
- [ ] Graph-augmented RAG as the default retrieval path
- [ ] Continual belief update loop driven by live user corrections
- [ ] Web UI (minimal, local) over the existing FastAPI routers
- [ ] `user_id`-aware legacy shims (`BeliefStoreShim`, `CauseGraphShim`, `PrincipleStoreShim`)

### v0.5 / v0.6 — Adaptive Intelligence → Autonomous Researcher
- [ ] Multi-agent coordination primitives
- [ ] Long-horizon goal tracking across the reflection + planning stack
- [ ] Autonomous experiment execution loop (curiosity → hypothesis → experiment → belief update, end to end)
- [ ] Publish StateBench and the ablation harness as a standalone benchmark
- [ ] Academic paper on the Blix / ADMA architecture

### v1.0 — Cognitive Operating System
- [ ] Tool use and external action execution as a first-class subsystem
- [ ] Full multi-agent, multi-user deployment story
- [ ] Row-level security / connection pooling for high-concurrency multi-tenant SQLite deployments

---

## 📄 Research Context

Blix began as an independent project in **June 2024** — a year before the author entered IIT Patna, originally born out of a hackathon question at Aignite 2025 about why an AI companion's conversations weren't really personalized. It is the longest-running and most architecturally ambitious project in this portfolio.

A **323-page architecture document** covers the full design rationale, subsystem specifications, and theoretical grounding for every component. A **41-page Minor Project I academic report** (Blix v0.3.1) documents the original StateBench tests and their results; the subsequent changelogs extend this with causal reasoning, metacognition, curiosity-driven experimentation, the ADMA policy-learning layer, a typed event bus, procedural skill discovery, and — from v0.3.17 onward — three successive rounds of audit-driven hardening culminating in an external-benchmark ablation harness.

The architectural thinking in Blix directly informed the author's research internship at BITS Pilani (April–May 2026), where structured reasoning pipeline design — a core Blix concern — was applied to the ECOT-ERG empathetic dialogue framework, now under review at **EMNLP 2026**.

**Conceptual influences:** ACT-R cognitive architecture · Soar cognitive system · Global Workspace Theory · Truth Maintenance Systems (Doyle, 1979) · Retrieval-Augmented Generation · Chain-of-Thought reasoning · Multi-armed bandits / Thompson sampling · Direct Preference Optimization · Voyager (skill discovery from trajectories)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

**Before contributing:**
1. Open an issue describing the change you want to make
2. Wait for discussion and approval before submitting a PR
3. For major architectural changes, a design doc is expected

**Good first issues:** documentation improvements, new StateBench test cases, alternative embedding model integrations, additional Ollama model configurations, new policy arms for the ADMA bandit layer, `user_id` support for the legacy causality shims.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Sayan Dutta**
AI Researcher · BS-MS CSDA · IIT Patna · ORCID: 0009-0006-4747-8820

[![Portfolio](https://img.shields.io/badge/Portfolio-sayandutta.netlify.app-4f46e5?style=flat-square&logo=safari&logoColor=white)](https://sayandutta.netlify.app)
[![GitHub](https://img.shields.io/badge/GitHub-SAYANDUTTA8442-0f172a?style=flat-square&logo=github&logoColor=white)](https://github.com/SAYANDUTTA8442)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-sayandutta8653128442-0a66c2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/sayandutta8653128442)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Sayan--Dutta--19-00ccbb?style=flat-square&logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Sayan-Dutta-19)
[![Email](https://img.shields.io/badge/Email-sayandutta.developer@gmail.com-ea4335?style=flat-square&logo=gmail&logoColor=white)](mailto:sayandutta.developer@gmail.com)

---

<div align="center">

*Building intelligent systems that reason, remember, and recover.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4f46e5,50:1e1b4b,100:0d1117&height=90&section=footer" width="100%"/>

</div>
# Blix v0.3.16.9 — Issues Backlog

**Generated:** 2026-08-02  
**Codebase:** v0.3.16.9 (712 passing tests)  
**Methodology:** Full static analysis across 280 Python modules, dynamic behaviour analysis, algorithm correctness review, and comparison against production standards.

---

## Severity Legend

| Level | Meaning |
|-------|---------|
| **CRITICAL** | Data loss, silent corruption, or security breach under normal use |
| **HIGH** | Wrong results, resource leaks, or behaviour that breaks guarantees |
| **MEDIUM** | Quality, maintainability, or performance problems with measurable impact |
| **LOW** | Code hygiene, documentation, or minor design inconsistencies |

---

## ISSUE-A01 — Double-Decay on LRU Cache Eviction

**Severity:** HIGH  
**Subsystem:** `policy/learner.py` — `PolicyLearner`

**Description:**  
`_apply_decay()` applies the epoch multiplier in-memory to every arm currently in the LRU cache. If an arm is later evicted (because `cache_max_size` is reached) and then re-fetched from the database, `_get_cached()` applies `_decay_epoch` again to the stale DB value. The arm has now been decayed twice for the same observation window — once in memory, once on re-fetch.

**Concrete scenario:**
```
1. Arms A, B, C in cache (max_size=3). 10 rewards observed.
   _apply_decay applies epoch=0.995^10 to A, B, C in memory.
2. Arm D arrives. A is evicted (LRU). A's DB value = stale (not yet flushed).
3. Caller requests A via _get_cached(). A is loaded from DB.
   _get_cached applies epoch=0.995^10 again to the already-decayed in-memory value.
4. A's alpha is now decayed by 0.995^20 instead of 0.995^10.
```

**Impact:** Policies that fall out of the LRU cache converge more slowly and with lower confidence than the algorithm intends. The Thompson sampling guarantee is violated.

**Root cause:** The epoch is a global multiplier that tracks "how much decay is pending against the DB value." But in-memory values have already been partially decayed — so applying the epoch to them again on re-fetch double-counts.

**Recommended fix:**  
Track a per-policy `_last_epoch_applied: float` alongside each cached arm. When writing to cache, record the current epoch value at which the arm was last decayed. When fetching from DB, apply only the epoch delta since the last DB write: `effective_epoch = current_epoch / last_flushed_epoch`. Alternatively, store `decayed_alpha` and `decayed_beta` separately from the DB-persisted values in the cache entry.

**Effort:** 4 hours. **Breaking change: No.**

---

## ISSUE-A02 — Broad `except Exception` Swallowing Configuration Errors

**Severity:** HIGH  
**Subsystem:** `policy/learner.py`, `policy/optimizer.py`, `policy/reward.py`, `policy/compiler.py`, `policy/store.py`

**Description:**  
All five modules use this pattern to load central settings:

```python
def _default_learner_cfg():
    try:
        from config.settings import adma_settings
        return adma_settings.learner
    except Exception:
        return None
```

A `None` return causes every parameter to fall back to its hardcoded default. This means:
- A malformed `blix.yaml` (YAML syntax error) silently produces default behaviour with no diagnostic
- A Pydantic validation error (e.g., `decay_factor: 2.0`) is silently swallowed and the invalid config is never used
- The operator has no indication that their configuration file was ignored

**Impact:** All of ISSUE-014's centralised configuration benefit is negated if any config error occurs. The system silently uses defaults and the operator thinks their config applied.

**Recommended fix:** Re-raise configuration errors as `RuntimeError` with a descriptive message. Log a structured `ERROR` entry at minimum. Reserve the `except` only for `ImportError` (config module not installed), not general `Exception`.

```python
def _default_learner_cfg():
    try:
        from config.settings import adma_settings
        return adma_settings.learner
    except ImportError:
        return None  # config module not available — use hardcoded defaults
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error(
            "Failed to load ADMA learner config: %s — using hardcoded defaults", exc)
        raise RuntimeError(f"blix.yaml configuration error: {exc}") from exc
```

**Effort:** 1 hour. **Breaking change: No.**

---

## ISSUE-A03 — HGSHM Double-Close Causes Silent Corruption

**Severity:** HIGH  
**Subsystem:** `memory/hybrid/hgshm.py`, `memory/manager.py`

**Description:**  
`HGSHM.close()` calls `self.graph_store.close()` and `self.vector_index.close()`. There is no guard against calling `close()` twice. `GraphStore.close()` delegates to `HGSHMStore.close()` which calls `sqlite3.Connection.close()`. Calling `close()` on an already-closed SQLite connection raises `ProgrammingError: Cannot operate on a closed database.`

The risk is real because:
1. `MemoryManager.close()` calls `self._hgshm.close()`
2. `HGSHM.__exit__()` also calls `self.close()`
3. If a caller uses both `with HGSHM(...) as h:` and passes `h` to `MemoryManager`, both will call `close()` on exit

**Recommended fix:**  
Add an `_closed: bool = False` guard in `HGSHM.close()`:

```python
def close(self) -> None:
    if self._closed:
        return
    self._closed = True
    self.graph_store.close()
    self.vector_index.close()
```

Same pattern in `GraphStore.close()` and `PolicyStore.close()`.

**Effort:** 30 minutes. **Breaking change: No.**

---

## ISSUE-A04 — `_reward_log_counts` Counter Lost on Restart

**Severity:** HIGH  
**Subsystem:** `policy/store.py` — `PolicyStore.log_reward()`

**Description:**  
`_reward_log_counts` is an in-memory dict initialised to `{}` on every `PolicyStore.__init__()`. After a process restart, the counter starts at 0 for every policy regardless of how many rows actually exist in `reward_log`. The first `max_rows_per_policy` inserts after restart will not trigger any pruning, even if the table already has 10× the limit.

**Example:** 
- Policy has 1,000 rows (at limit). Process restarts.
- 500 more rewards are observed: counter = 500, no prune triggered.
- Now 1,500 rows exist — 50% over the intended limit.
- Prune finally triggers at count 1,001 and deletes back to 1,000.

**Impact:** The `reward_log` table can grow to `2 × max_rows_per_policy` between restarts. The audit originally rated this as an infinite growth bug — the fix was partial.

**Recommended fix:**  
Initialise the counter from the DB on first write to a policy:

```python
if pid not in self._reward_log_counts:
    self._reward_log_counts[pid] = self.reward_log_count(pid)
self._reward_log_counts[pid] += 1
```

This adds one `COUNT(*)` query per policy on first post-restart write — negligible cost.

**Effort:** 30 minutes. **Breaking change: No.**

---

## ISSUE-A05 — `ConsolidationEngine` Still Uses O(n) Full Table Scan

**Severity:** HIGH  
**Subsystem:** `memory/hybrid/consolidation/consolidation_engine.py`

**Description:**  
ISSUE-008 fixed `SystemMemory` and `UserMemory` to use `nodes_by_tags()` instead of `all_nodes(limit=N)`. However, `ConsolidationEngine` was not updated — it still calls `self._store.all_nodes(...)` to load all nodes before deduplication. At 100,000 nodes, this loads hundreds of megabytes of `MemoryNode` objects into Python memory for processing.

The same pattern exists in `HierarchyManager`, which also scans all nodes to find candidates for compression.

**Impact:** Memory consolidation and hierarchy compression become the dominant CPU and memory consumers at scale, negating the retrieval improvements from ISSUE-008.

**Recommended fix:**  
Pass tag/type filters into `all_nodes()` queries, or use the new `nodes_by_tags()` API where appropriate. For consolidation, the scan must be batched (process N nodes at a time, not all at once).

**Effort:** 6 hours. **Breaking change: No.**

---

## ISSUE-A06 — Database Filenames Not Configurable

**Severity:** MEDIUM  
**Subsystem:** `memory/hybrid/storage/persistence.py`, `memory/hybrid/vector/vector_store.py`, `policy/store.py`

**Description:**  
Three database filenames are hardcoded as string literals in their respective constructors:

| Module | Hardcoded name |
|--------|---------------|
| `persistence.py` | `"hgshm.db"` |
| `vector_store.py` | `"vectors.db"` |
| `policy/store.py` | `"policy.db"` |

Despite ISSUE-014 adding `ADMASettings` and `HGSHMSettings` to `config/schema.py`, database paths are absent from both schemas. A production deployment that needs to separate databases by environment (e.g., `production/policy.db` vs `test/policy.db`) has no configuration hook — it must monkey-patch the file path or pass different `memory_dir` values.

**Impact:** Cannot differentiate database locations per environment without code changes. Prevents blue-green deployment patterns.

**Recommended fix:**  
Add `db_filename: str = "policy.db"` to the relevant settings classes. Wire the constructors to read from settings as a default.

**Effort:** 2 hours. **Breaking change: No.**

---

## ISSUE-A07 — Broadcast `observe()` Rewards All Arms Unconditionally

**Severity:** MEDIUM  
**Subsystem:** `policy/learner.py` — `PolicyLearner.observe()`

**Description:**  
When `reward.policy_id` is `None` (a broadcast reward), `observe()` calls `_reward_to_policy_types()` to get the affected policy types, then calls `self._store.all_active(policy_type=pt, domain=domain)` and **updates every arm of that type**. 

A `BENCHMARK_SCORE` reward for a planning benchmark updates every `RETRIEVAL_WEIGHTS` arm and every `PLANNER_CONFIG` arm regardless of which arm was actually used. This violates the contextual bandit principle: only the arm that generated the outcome should receive the credit.

**Impact:** Learning signal is diluted across all arms. Arms that were not selected receive credit (or blame) for outcomes they did not produce. Convergence rate slows proportionally to the number of arms.

**Recommended fix:**  
Broadcast rewards should be dispatched only to the arm most recently selected in the matching context, tracked via `_last_selected`. If no recent selection exists for that context, the reward should be dropped rather than broadcast to all arms.

**Effort:** 3 hours. **Breaking change: No** (behaviour improves, existing tests remain valid).

---

## ISSUE-A08 — `PolicyVersion.beta` Field Naming Inconsistency Remains Fragile

**Severity:** MEDIUM  
**Subsystem:** `policy/models.py`, `policy/store.py`

**Description:**  
`PolicyRecord` uses `beta_` (trailing underscore to avoid shadowing `math.beta`). `PolicyVersion` uses `beta` (no underscore). The `PolicyStore` maps between them with `"beta_": d["beta"]` on write and `d["beta"] = d.pop("beta_", 1.0)` on read. This mapping is spread across three methods (`save_version`, `get_history`, `rollback`) and any future developer adding a fourth method that accesses `PolicyVersion` will almost certainly introduce the `beta`/`beta_` mismatch again.

**Impact:** Future data corruption. The fix from ISSUE-002 is correct but brittle.

**Recommended fix:**  
Rename `PolicyVersion.beta` to `PolicyVersion.beta_` for consistency with `PolicyRecord`. Update the SQL schema column from `beta_` to `beta_` (already correct). Remove the mapping code from `save_version`/`get_history`/`rollback`. This eliminates a class of future bugs for a one-character change.

**Effort:** 2 hours. **Breaking change: No** (internal only; add schema migration v2 to rename column if needed).

---

## ISSUE-A09 — `MemoryManager.store_system()` Accepts Unsafe `**kwargs`

**Severity:** MEDIUM  
**Subsystem:** `memory/manager.py`

**Description:**  
```python
def store_system(self, text: str, **kwargs) -> MemoryNode:
    return self._system.store_workflow(text, **kwargs)

def store_user(self, user_id: str, text: str, **kwargs) -> MemoryNode:
    return self.get_user_memory(user_id).store_preference(
        category=kwargs.pop("category", "general"), preference=text, **kwargs)
```

`**kwargs` is passed through to `store_workflow()` and `store_preference()` with no validation. A caller passing `tags=["admin"]` or `importance=1.0` could override the domain-isolation tags that `SystemMemory` and `UserMemory` rely on for correctness. The tag `system_memory` that gates all `SystemMemory` queries could be overridden to an empty list.

**Impact:** Domain isolation can be bypassed via `store_system(..., tags=[])`. Nodes stored without the domain tag will not appear in `stats()`, `recent_failures()`, or any other tagged retrieval.

**Recommended fix:**  
Remove `**kwargs` from both methods. Accept only the parameters the underlying methods actually need. If flexibility is required, add explicit keyword parameters with validation.

**Effort:** 1 hour. **Breaking change: Potential** (callers using `**kwargs` would need updating, but this is the intended breaking change).

---

## ISSUE-A10 — `HGSHM` Shim Registry Has No Size Bound or Eviction

**Severity:** MEDIUM  
**Subsystem:** `memory/hybrid/shims.py`

**Description:**  
`shims.py` maintains `_HGSHM_REGISTRY: dict[str, HGSHM]` — a module-level singleton dict that maps `memory_dir` paths to open `HGSHM` instances. This registry:
1. Has no maximum size
2. Has no eviction policy
3. Has no way to close registered instances when they are no longer needed
4. Keeps SQLite connections open indefinitely for every unique `memory_dir` path ever used in the process

In a multi-tenant server where each user has their own `memory_dir`, this grows to O(users) open SQLite connections.

**Recommended fix:**  
Add a `close_registry()` function for graceful shutdown. Consider replacing with a weakref-keyed dict so instances are released when callers no longer hold a reference. Alternatively, deprecate the singleton pattern in favour of explicit instance management.

**Effort:** 3 hours. **Breaking change: Potentially** (if callers depend on the singleton behaviour).

---

## ISSUE-A11 — `PolicyOptimizer.run_cycle()` Ignores `mutation_scale`

**Severity:** MEDIUM  
**Subsystem:** `policy/optimizer.py`

**Description:**  
`PolicyOptimizer.__init__()` stores `self._mutation_scale = ...` from settings (`adma_settings.optimizer.mutation_scale`, default 0.1). However, `run_cycle()` calls `self.evolve_poor_performers()` with no arguments — `evolve_poor_performers()` has its own `mutation_scale: float = 0.1` default, which ignores the instance-level `self._mutation_scale`.

```python
# run_cycle() — bug
mutants = self.evolve_poor_performers()  # always uses 0.1

# evolve_poor_performers() signature
def evolve_poor_performers(self, mutation_scale: float = 0.1):
    ...
    mutant = self.spawn_mutant(p, mutation_scale)  # uses parameter, not self._mutation_scale
```

This means the `BLIX_ADMA_MUTATION_SCALE` env var and the `mutation_scale` config value set in `blix.yaml` have no effect when using `run_cycle()`.

**Recommended fix:**  
Change `run_cycle()` to pass `self._mutation_scale`:
```python
mutants = self.evolve_poor_performers(mutation_scale=self._mutation_scale)
```

**Effort:** 5 minutes. **Breaking change: No.**

---

## ISSUE-A12 — Broad `except Exception` in Retrieval Hot Path

**Severity:** MEDIUM  
**Subsystem:** `memory/hybrid/retrieval/hybrid_retriever.py`

**Description:**  
Three `except Exception:` clauses in `hybrid_retriever.py` silently catch all exceptions and either return empty results or fall back to a default. These are in the hot retrieval path called on every `HGSHM.recall()`:

```python
try:
    ...vector search...
except Exception:
    return []  # silent failure — caller sees no results
```

A database corruption, schema mismatch, or out-of-memory error will produce an empty result set indistinguishable from "no relevant memories found." This is especially dangerous because the system continues operating normally while the memory retrieval is completely broken.

**Recommended fix:**  
Catch specific exceptions (`sqlite3.Error`, `struct.error` for vector deserialization). Re-raise unexpected exceptions or log them at `ERROR` level with full traceback. Never return an empty list for an unrecoverable error.

**Effort:** 2 hours. **Breaking change: No.**

---

## ISSUE-A13 — `VectorStore` Brute-Force Fallback Not Surfaced

**Severity:** MEDIUM  
**Subsystem:** `memory/hybrid/vector/vector_store.py`

**Description:**  
When `sqlite-vec` is unavailable, `VectorStore` silently falls back to O(n²) brute-force cosine search. The docstring mentions this, but there is no `WARNING` log entry at the point of fallback and no metric emitted. A production operator who hasn't read the source code will not know their retrieval is running in degraded mode.

At 10,000 nodes, brute-force search takes ~100ms vs ~1ms for ANN. At 100,000 nodes it becomes unusable.

**Recommended fix:**  
Log a `WARNING` at module load time when the fallback is active:
```python
if not _SQLITE_VEC_AVAILABLE:
    log.warning(
        "sqlite-vec not available — using O(n) brute-force search. "
        "Install sqlite-vec for production performance: pip install sqlite-vec"
    )
```

**Effort:** 15 minutes. **Breaking change: No.**

---

## ISSUE-A14 — `PolicyLearner._cache` Not Thread-Safe

**Severity:** MEDIUM  
**Subsystem:** `policy/learner.py`

**Description:**  
`PolicyLearner` is documented as single-threaded (only `PolicyStore` has a lock). However, `PolicyLearner._cache` is an `OrderedDict` and `_cache_put()` / `_cache_get()` call `move_to_end()` and `popitem()`. These are not atomic under Python's GIL in all cases — specifically, `dict.popitem()` followed by `dict.__setitem__` in `_cache_put()` is two operations that can interleave with a concurrent `_cache_get()` on the same key.

If `PolicyLearner` is ever used from multiple threads (e.g., concurrent FastAPI request handlers), the LRU cache can corrupt silently.

**Recommended fix:**  
Either document clearly that `PolicyLearner` is single-threaded per instance (and enforce this with an assertion), or add a `threading.Lock` to protect `_cache_put()` and `_cache_get()`. Given `PolicyStore` already has an `RLock`, wrapping the learner cache with a simple `Lock` adds negligible overhead.

**Effort:** 1 hour. **Breaking change: No.**

---

## ISSUE-A15 — `AdaptiveRetriever` and `AdaptivePlanner` Have No Tests

**Severity:** MEDIUM  
**Subsystem:** `policy/adaptive.py`, `tests/`

**Description:**  
`AdaptiveRetriever` and `AdaptivePlanner` — the modules that wire policy learning to actual retrieval and planning — have zero test coverage. They are exported from `policy/__init__.py` as public API but their behavior is entirely untested. Specifically:

- `AdaptiveRetriever.retrieve()` accesses `self._hgshm.hybrid_retriever._weights` (a private attribute) — this is the ISSUE-012 violation that was never fixed with a test to catch regressions
- `AdaptivePlanner.search()` imports `BeamSearchPlanner` inside the method body (deferred import) — this fails silently if the import fails
- Neither class has tests for the reward dispatch path

**Recommended fix:**  
Add `TestAdaptiveRetriever` and `TestAdaptivePlanner` test classes covering: policy weight application, reward dispatch after retrieval, fallback when no policy arms exist, and the `_weights` private attribute access.

**Effort:** 4 hours. **Breaking change: No.**

---

## ISSUE-A16 — `print()` Statements in Production Modules

**Severity:** MEDIUM  
**Subsystem:** `config/settings.py`, `memory/hybrid/hgshm.py`, `policy/ablation_v3.py`

**Description:**  
13 `print()` calls exist in production code. In `config/settings.py`, `print()` is called at module import time. In `policy/ablation_v3.py`, `AblationV3Report.print_report()` writes directly to stdout. In `memory/hybrid/hgshm.py`, debug output uses `print()`.

Production systems must use `logging` exclusively. `print()` cannot be filtered by log level, cannot be redirected to structured log systems (Datadog, CloudWatch), and pollutes test output.

**Recommended fix:**  
Replace all `print()` calls with `log.info()` or `log.debug()`. Convert `AblationV3Report.print_report()` to return a formatted string and let the caller decide whether to print or log it.

**Effort:** 1 hour. **Breaking change: Minor** — `print_report()` API changes from void to `str`.

---

## ISSUE-A17 — `PolicyVersion` Has No Tests

**Severity:** MEDIUM  
**Subsystem:** `policy/models.py`, `tests/`

**Description:**  
`PolicyVersion` is a public class in `policy.__all__` with its own `to_dict()` / `from_dict()` serialization methods, but `TestPolicyRecord` in `test_v0316_adma.py` contains no test for `PolicyVersion` directly. The only coverage is indirect via `PolicyStore.save_version()` tests, which test round-trip storage but not the class's own invariants:

- `from_dict(to_dict(pv)) == pv` (serialization roundtrip)
- `beta` field default value
- `reason` field preserved
- `version_id` is a valid UUID

**Recommended fix:**  
Add `TestPolicyVersion` class with roundtrip serialization, field defaults, and UUID validation tests.

**Effort:** 1 hour. **Breaking change: No.**

---

## ISSUE-A18 — `RoutedContext.to_memory_context()` Loses Supporting Memories

**Severity:** MEDIUM  
**Subsystem:** `memory/manager.py`

**Description:**  
`RoutedContext.to_memory_context()` sets `ctx.primary_memories = self.merged_memories[:10]` and `ctx.supporting_memories = self.merged_memories[10:20]`. This hard-codes a 10/10 split regardless of `top_k` configuration. If `top_k=5` was used, only 5 memories exist and the split is fine. But if `top_k=50` was used, 30 memories are silently discarded when converting to `MemoryContext`.

The converted `MemoryContext` is passed to `PolicyCompiler.compile()` which uses `max_memory_nodes` from settings (default 5). So memories 5–50 are discarded at two different levels, making `top_k` values above ~15 pointless.

**Recommended fix:**  
Make the primary/supporting split respect the actual `merged_memories` length or accept a configurable split point. Store `top_k` on `RoutedContext` and use it in `to_memory_context()`.

**Effort:** 2 hours. **Breaking change: No.**

---

## ISSUE-A19 — Lazy Settings Imports Create Unpredictable Startup Timing

**Severity:** MEDIUM  
**Subsystem:** `policy/learner.py`, `policy/optimizer.py`, `policy/reward.py`, `policy/compiler.py`, `policy/store.py`

**Description:**  
All five ADMA modules load their configuration via a function called at first use:
```python
def _default_learner_cfg():
    try:
        from config.settings import adma_settings
        return adma_settings.learner
    except Exception:
        return None
```

This means `blix.yaml` is parsed on the first `PolicyLearner` instantiation, not at startup. The consequences:

1. Config errors are discovered late (first use, not import time)
2. Settings are read multiple times across constructors (though the singleton ensures consistent values, the `try/except` overhead occurs on every instantiation)
3. Tests that set env vars after import but before instantiation may get unexpected results if settings were already cached elsewhere

**Recommended fix:**  
Import `adma_settings` at module level (top of file, after other imports). The circular import risk is zero — `config.settings` does not import from `policy`. Module-level import also ensures fail-fast at startup.

**Effort:** 30 minutes per module. **Breaking change: No.**

---

## ISSUE-A20 — `sanitize_task_text` Injection Patterns Are Incomplete

**Severity:** MEDIUM  
**Subsystem:** `policy/compiler.py`

**Description:**  
The 8 injection patterns in `_INJECTION_PATTERNS` cover common jailbreak phrases but miss several documented attack classes:

- **Indirect injection via memory:** A memory node could contain `[INST]system override[/INST]` which gets inserted into the compiled prompt through `memory_context_str` without sanitization
- **Token boundary attacks:** `ign​ore` (zero-width space) bypasses the regex
- **Base64/encoding attacks:** `aWdub3JlIGFsbCBpbnN0cnVjdGlvbnM=` (base64 for "ignore all instructions")
- **Markdown injection:** `# System\n You are now...` in the task field
- **Repetition attacks:** Hundreds of newlines to push the system prompt out of context

Furthermore, `memory_context_str` — which contains nodes retrieved from HGSHM — is not sanitized before being inserted into the compiled prompt. A stored memory node (e.g., from a document processor) could contain injection content.

**Recommended fix:**  
Sanitize `memory_context_str` snippets using the same `sanitize_task_text()` function. Document clearly that `sanitize_task_text()` is best-effort. Consider adding a structured separator (`###TASK###`) so the LLM can distinguish task content from system instructions at the token level.

**Effort:** 3 hours. **Breaking change: No.**

---

## ISSUE-A21 — `PolicyStore._ACTIVE_SQL` Not Covered by Schema Migration

**Severity:** LOW  
**Subsystem:** `policy/store.py`

**Description:**  
`PolicyStore._ACTIVE_SQL` is a class-level dict of 8 precomputed SQL strings. When the schema changes in a future migration (e.g., a new column is added to the `policies` table), the `SELECT *` in each of the 8 SQL strings will return the new column — but the `_row_to_policy()` deserializer may not handle it. Conversely, if a column is removed, `SELECT *` will omit it silently.

The migration framework (ISSUE-002) handles DDL but has no mechanism to re-validate `_ACTIVE_SQL` against the actual schema after a migration runs.

**Recommended fix:**  
Replace `SELECT *` with explicit column lists in `_ACTIVE_SQL`. After any migration that changes the `policies` schema, update both the DDL and `_ACTIVE_SQL`. Add a schema version check that validates the column list matches expectations.

**Effort:** 2 hours. **Breaking change: No.**

---

## ISSUE-A22 — `PolicyCompiler` Does Not Validate System Instruction Length

**Severity:** LOW  
**Subsystem:** `policy/compiler.py`

**Description:**  
`PolicyCompiler.compile()` estimates token count as `len(full_text) // 4` and enforces a constraint only when `token_budget < 1000`. At `token_budget=2000` (the default), no truncation occurs even if the assembled prompt is 8,000 tokens. The token estimate is also known to be inaccurate (±30%) for code-heavy content where a "token" is much shorter than 4 characters.

**Recommended fix:**  
Add a `max_system_instructions_chars` limit to `PromptCompilerSettings`. Truncate `system_instructions` before assembling the full prompt. Log a `DEBUG` entry when truncation occurs.

**Effort:** 1 hour. **Breaking change: No.**

---

## ISSUE-A23 — `HybridWeightsSettings.to_dict()` Not Normalised

**Severity:** LOW  
**Subsystem:** `config/schema.py`, `memory/hybrid/retrieval/hybrid_retriever.py`

**Description:**  
`HybridWeightsSettings.to_dict()` returns the raw weight values. `HybridRetriever` expects normalised weights (summing to 1.0). The `PolicySelector.get_retrieval_weights()` fallback also returns unnormalised values. The actual normalisation happens in `HybridWeights.normalised()` inside the retriever — but callers who get weights from settings and use them directly (bypassing the retriever) will get unnormalised values.

**Recommended fix:**  
Add a `normalised()` method to `HybridWeightsSettings` that divides each value by the total. Make `to_dict()` return normalised values, or rename it to `to_raw_dict()` and add `to_normalised_dict()`.

**Effort:** 30 minutes. **Breaking change: No.**

---

## ISSUE-A24 — `EmbeddingManager` Has No `close()` Method

**Severity:** LOW  
**Subsystem:** `memory/hybrid/vector/embedding_manager.py`

**Description:**  
`EmbeddingManager` manages an in-process embedding cache (`cache_size` entries). It has no `close()` method and no context manager protocol. While it holds no file descriptors, it does hold a potentially large numpy array cache. `HGSHM.close()` does not clear the embedding cache, meaning the cache persists in memory even after the HGSHM connection is closed.

**Recommended fix:**  
Add `close()` that clears `self._cache` (the embedding LRU cache). Call it from `HGSHM.close()`.

**Effort:** 30 minutes. **Breaking change: No.**

---

## ISSUE-A25 — Missing `__all__` in `memory/` Package Hierarchy

**Severity:** LOW  
**Subsystem:** `memory/hybrid/`, `memory/system/`, `memory/user/`

**Description:**  
ISSUE-018 added `__all__` to all `policy/` modules. The `memory/` package hierarchy has no `__all__` declarations. `memory/hybrid/__init__.py`, `memory/system/__init__.py`, `memory/user/__init__.py`, and `memory/__init__.py` are all empty or minimally populated. The same problems that ISSUE-018 fixed for `policy/` (invisible API surface, `import *` imports everything) exist for `memory/`.

**Recommended fix:**  
Apply the same `__all__` treatment to `memory/` packages: populate `memory/hybrid/__init__.py` with `HGSHM`, `MemoryNode`, `MemoryType`, etc.; populate `memory/system/__init__.py` with `SystemMemory`; populate `memory/user/__init__.py` with `UserMemory`, `validate_user_id`.

**Effort:** 2 hours. **Breaking change: No.**

---

## ISSUE-A26 — `policy/store.py` `INSERT OR IGNORE` on `policy_versions` Silently Drops Data

**Severity:** LOW  
**Subsystem:** `policy/store.py`

**Description:**  
`save_version()` uses `INSERT OR IGNORE INTO policy_versions`. The unique key is `version_id` (a UUID), so collisions are astronomically unlikely. However, `INSERT OR IGNORE` is semantically wrong here: if a version was not inserted (for any reason — disk full, constraint violation on another column), the caller receives no error and no indication the snapshot was lost. The policy's version history is silently incomplete.

**Recommended fix:**  
Change to `INSERT INTO policy_versions` (no conflict clause). If the `version_id` is truly unique (it is), this never triggers a duplicate error. If something else prevents the insert, it raises, which is the correct behaviour.

**Effort:** 5 minutes. **Breaking change: No.**

---

## ISSUE-A27 — `UserMemory.store_interaction()` Does Not Record Response Content

**Severity:** LOW  
**Subsystem:** `memory/user/user_memory.py`

**Description:**  
`store_interaction()` records the user's query and whether the response was accepted or corrected, but not the actual response content. The text stored is:
```
[ACCEPTED] Query: explain recursion in Python
```

The response that the user accepted (or corrected) is not stored. This means:
- `UserMemory` cannot learn what kinds of responses the user prefers
- Correction tracking shows that a correction was made but not what the correct response looked like
- The learning signal is one-sided (only the query, not the output)

**Recommended fix:**  
Add an optional `response_summary: str = ""` parameter. Store the first 200 characters of the response alongside the query.

**Effort:** 30 minutes. **Breaking change: No** (new optional parameter).

---

## ISSUE-A28 — `config/settings.py` `export_config_snapshot()` Not Tested for YAML Failure

**Severity:** LOW  
**Subsystem:** `config/settings.py`, `tests/test_v03167_settings.py`

**Description:**  
`export_config_snapshot()` writes both `config_snapshot.json` and `config_snapshot.yaml`. The JSON write uses standard `json.dumps()` which is always available. The YAML write uses `yaml.dump()` — but `pyyaml` is not listed in any `requirements.txt` or `pyproject.toml` as a hard dependency. If `pyyaml` is not installed, the snapshot function silently fails mid-write (JSON is written, YAML is not) with no indication of partial failure.

**Recommended fix:**  
Wrap the YAML write in a `try/except ImportError` and log a warning if `pyyaml` is unavailable. Add `pyyaml` to formal dependencies.

**Effort:** 30 minutes. **Breaking change: No.**

---

## ISSUE-A29 — `PolicyLearner.register()` Not Thread-Safe

**Severity:** LOW  
**Subsystem:** `policy/learner.py`

**Description:**  
`PolicyLearner.register()` calls `self._store.all_active()` to check for existing policies with the same name, then conditionally calls `self._store.save()`. Between the read and the write, another thread could have registered a policy with the same name, resulting in duplicate policies. `PolicyStore` methods are individually thread-safe (RLock), but the check-then-act sequence is not atomic.

**Impact:** Low probability in practice (registration typically happens once at startup), but a race could produce duplicate default policies under concurrent initialization.

**Recommended fix:**  
Acquire `self._store._lock` for the entire read-check-write sequence in `register()`, or add a `UNIQUE` constraint on `(name, domain, policy_type, user_id)` to the SQLite schema and handle the `IntegrityError`.

**Effort:** 1 hour. **Breaking change: No.**

---

## ISSUE-A30 — `ContextBuilder` Hard-Limits to 11 Steps Without Configuration

**Severity:** LOW  
**Subsystem:** `memory/hybrid/context/context_builder.py`

**Description:**  
`ContextBuilder.build()` runs an 11-step pipeline with hardcoded step counts, score multipliers, and node limits within each step. None of these are exposed to `HGSHMSettings.retrieval`. This makes the context builder opaque and untunable — the equivalent of `HybridWeights` before ISSUE-014.

**Recommended fix:**  
Add a `ContextBuilderSettings` sub-schema to `HGSHMSettings` with the key limits (max primary memories, max supporting memories, min confidence for inclusion, etc.).

**Effort:** 3 hours. **Breaking change: No.**

---

## Summary Table

| ID | Severity | Subsystem | One-line description |
|----|----------|-----------|---------------------|
| A01 | HIGH | `policy/learner.py` | Double-decay on LRU eviction — correctness violation |
| A02 | HIGH | All policy modules | Silent config swallow — operators don't know config failed |
| A03 | HIGH | `memory/hybrid/hgshm.py` | Double-close raises `ProgrammingError` |
| A04 | HIGH | `policy/store.py` | `_reward_log_counts` lost on restart — table over-grows |
| A05 | HIGH | `consolidation_engine.py` | O(n) full table scan not fixed in consolidation |
| A06 | MEDIUM | 3 storage modules | DB filenames hardcoded, not configurable |
| A07 | MEDIUM | `policy/learner.py` | Broadcast rewards update all arms — violates bandit theory |
| A08 | MEDIUM | `policy/models.py` / `store.py` | `beta` / `beta_` naming — future re-introduction of bug |
| A09 | MEDIUM | `memory/manager.py` | `**kwargs` bypasses domain isolation tags |
| A10 | MEDIUM | `memory/hybrid/shims.py` | HGSHM singleton registry has no size bound |
| A11 | MEDIUM | `policy/optimizer.py` | `mutation_scale` config value ignored in `run_cycle()` |
| A12 | MEDIUM | `hybrid_retriever.py` | Broad `except Exception` swallows retrieval errors |
| A13 | MEDIUM | `vector_store.py` | Brute-force fallback not logged as WARNING |
| A14 | MEDIUM | `policy/learner.py` | `_cache` OrderedDict not thread-safe |
| A15 | MEDIUM | `policy/adaptive.py` | `AdaptiveRetriever` / `AdaptivePlanner` have zero tests |
| A16 | MEDIUM | 3 modules | `print()` in production code — not loggable or filterable |
| A17 | MEDIUM | `policy/models.py` | `PolicyVersion` class has no unit tests |
| A18 | MEDIUM | `memory/manager.py` | `to_memory_context()` hard-codes 10/10 memory split |
| A19 | MEDIUM | 5 policy modules | Lazy settings import — errors discovered late |
| A20 | MEDIUM | `policy/compiler.py` | Injection sanitization misses memory context and encodings |
| A21 | LOW | `policy/store.py` | `SELECT *` in `_ACTIVE_SQL` fragile against schema changes |
| A22 | LOW | `policy/compiler.py` | No maximum length check on compiled system instructions |
| A23 | LOW | `config/schema.py` | `HybridWeightsSettings.to_dict()` returns unnormalised weights |
| A24 | LOW | `embedding_manager.py` | No `close()` — embedding cache not released on shutdown |
| A25 | LOW | `memory/` packages | No `__all__` in `memory/` package hierarchy |
| A26 | LOW | `policy/store.py` | `INSERT OR IGNORE` silently drops failed snapshots |
| A27 | LOW | `memory/user/user_memory.py` | `store_interaction()` doesn't record response content |
| A28 | LOW | `config/settings.py` | `pyyaml` not a formal dependency — YAML snapshot silently fails |
| A29 | LOW | `policy/learner.py` | `register()` check-then-act not atomic |
| A30 | LOW | `context_builder.py` | 11-step pipeline limits hardcoded, not configurable |

---

## Priority Queue for v0.3.17

### Must fix before PCRE

| ID | Reason |
|----|--------|
| **A01** | Correctness violation in core learning algorithm |
| **A02** | Configuration failures silently ignored — operators cannot trust config |
| **A03** | Data corruption on double-close — affects `with` statement usage |
| **A04** | `reward_log` grows 2× the intended limit after every restart |
| **A11** | `mutation_scale` config value has zero effect — ISSUE-014 work wasted |
| **A16** | `print()` in production — blocks structured logging adoption |

### Fix early in v0.3.17

| ID | Reason |
|----|--------|
| **A05** | ConsolidationEngine remains O(n) — negates ISSUE-008 gains |
| **A07** | Broadcast rewards violate bandit guarantee — learning is noisier than intended |
| **A12** | Retrieval errors silently return empty results |
| **A15** | `AdaptiveRetriever` / `AdaptivePlanner` untested — ISSUE-012 regression risk |
| **A19** | Module-level settings import is 5-line change, eliminates entire class of bugs |

### PCRE-blocking

These are not blocking for PCRE start, but must be resolved before PCRE ships:

| ID | Reason |
|----|--------|
| **A14** | Thread-safety required for concurrent PCRE reasoning paths |
| **A08** | `beta`/`beta_` inconsistency will recur as PCRE adds more snapshot paths |
| **A09** | Domain isolation breach risk increases with PCRE's new memory operations |

---

*End of Issues.md — 30 issues catalogued, validated against the live codebase.*

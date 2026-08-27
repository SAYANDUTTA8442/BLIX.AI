# Blix Multi-Tenancy Isolation Guide

**Version:** v0.3.18.3  
**Issue:** B10 — document and improve cross-user isolation guarantees

---

## Current Isolation Model

Blix is primarily designed as a **single-tenant, local-first** system. However, several components support multi-user deployments with varying degrees of isolation.

---

## Component-by-Component Isolation Status

### ✅ UserMemory (`memory/user/user_memory.py`)

**Isolated.** Each `UserMemory` instance is constructed with a `user_id` and stores all nodes tagged with that ID. Queries filter by `user_id`. No cross-user leakage if callers construct separate `UserMemory` instances per user.

### ✅ PolicyStore (`policy/store.py`)

**Isolated.** The `policies` table has a `user_id` column with an index. `all_active()` accepts a `user_id` parameter and filters accordingly. The `_ACTIVE_SQL` dict generates separate SQL variants for user-scoped queries. Policies created without a `user_id` are treated as system-wide defaults visible to all users.

### ✅ PolicyLearner LRU cache (`policy/learner.py`)

**Isolated by policy_id.** The LRU cache keys on `policy_id`. Since policies are already per-user at the DB level (via `user_id` column), the cache inherits that isolation — cache entries for user A's policies are keyed differently from user B's.

### ⚠️ HGSHM Shim Registry (`memory/hybrid/shims.py`)

**Partially isolated (B10 fix applied).** The `_get_hgshm(memory_dir, user_id=None)` function now accepts an optional `user_id`. When provided, the registry key is `"{user_id}:{memory_dir}"`, giving each user a separate HGSHM instance even if they share a `memory_dir`.

**Limitation:** Legacy shim classes (`BeliefStoreShim`, `CauseGraphShim`, `PrincipleStoreShim`) call `_get_hgshm(memory_dir)` without a `user_id`. To get per-user isolation with these shims, construct them with a user-scoped `memory_dir` (e.g. `base_dir / user_id`).

### ⚠️ HGSHM (`memory/hybrid/hgshm.py`)

**Not isolated by itself.** A single `HGSHM` instance is a shared database. User isolation at this level requires either:
1. Separate `HGSHM` instances per user (recommended for research), or
2. Application-level filtering by `user_id` tag on all node queries.

### ✅ GlobalWorkspace (`workspace/global_workspace.py`)

**Isolated by instance.** `GlobalWorkspace` uses injected `WorkspaceMemory` and `BroadcastBus` dependencies. Create separate instances per user, each with their own injected dependencies, for full isolation. There is no global shared state in `GlobalWorkspace` itself.

### ✅ EmbeddingManager (`memory/hybrid/vector/embedding_manager.py`)

**Model is shared (by design); cache is per-instance.** The sentence embedding model is loaded once per process (model weights are read-only). The LRU text→vector cache is per-instance and is cleared on `close()`. No cross-user data leakage through the cache.

---

## Recommended Deployment Patterns

### Single-user (default)

```python
hgshm = HGSHM(memory_dir=Path("~/.blix/memory"))
manager = MemoryManager(hgshm=hgshm, system_memory=SystemMemory(hgshm))
```

### Multi-user with shared memory_dir (B10 pattern)

```python
from memory.hybrid.shims import _get_hgshm

# Each user gets an isolated HGSHM even though memory_dir is shared
hgshm_alice = _get_hgshm(Path("/shared/memory"), user_id="alice")
hgshm_bob   = _get_hgshm(Path("/shared/memory"), user_id="bob")
assert hgshm_alice is not hgshm_bob  # separate instances
```

### Multi-user with per-user directories (strongest isolation)

```python
base = Path("/data/users")
hgshm_alice = HGSHM(base / "alice")
hgshm_bob   = HGSHM(base / "bob")
```

---

## Known Limitations

1. **SQLite WAL mode**: All users sharing the same SQLite file must coordinate writes through the application layer. HGSHM uses WAL mode and an `RLock` for this, but cross-process isolation requires a proper connection pool or separate DB files.

2. **Policy system-defaults**: Policies stored with `user_id=None` are visible to all users via `all_active()`. This is intentional for system-wide policies but could be surprising in a strict multi-tenant deployment.

3. **Embedding cache**: The embedding model is loaded once per process and shared across all users. Model weights are read-only and do not contain user data.

4. **Legacy shims**: `BeliefStoreShim`, `CauseGraphShim`, and `PrincipleStoreShim` do not accept a `user_id` parameter. Use per-user `memory_dir` paths with these shims.

---

## Future Work

- Add `user_id` parameter to `BeliefStoreShim`, `CauseGraphShim`, `PrincipleStoreShim`
- Add row-level security helpers for multi-tenant SQLite deployments
- Consider a proper connection pool for high-concurrency multi-tenant use

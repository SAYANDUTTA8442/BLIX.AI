"""
MemoryManager — unified routing across System, User, and HGSHM domains.

Routes queries to the appropriate memory domain(s), merges results,
removes duplicates, and applies policy-driven ranking.

This is the single entry point for all memory access in v0.3.16.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memory.hybrid.hgshm import HGSHM
from memory.hybrid.models.memory_context import MemoryContext, RetrievedMemory
from memory.hybrid.models.memory_node import MemoryNode, MemoryType
from memory.system.system_memory import SystemMemory
from memory.user.user_memory import UserMemory

log = logging.getLogger(__name__)


@dataclass
class RoutedContext:
    """
    The merged output of MemoryManager.query().

    Combines system + user + general HGSHM results with routing metadata.
    """
    query:              str                   = ""
    system_context:     MemoryContext | None  = None
    user_context:       MemoryContext | None  = None
    general_context:    MemoryContext | None  = None
    merged_memories:    list[RetrievedMemory] = field(default_factory=list)
    routing_latency_ms: float                 = 0.0
    domains_queried:    list[str]             = field(default_factory=list)
    metadata:           dict[str, Any]        = field(default_factory=dict)
    # A18: configurable split between primary and supporting memories.
    # Set by MemoryManager.query() to match top_k so no memories are silently discarded.
    primary_split:      int                   = 10

    @property
    def total_memories(self) -> int:
        return len(self.merged_memories)

    @property
    def top_memory(self) -> MemoryNode | None:
        if self.merged_memories:
            return self.merged_memories[0].node
        return None

    def to_memory_context(self) -> MemoryContext:
        """Convert to a flat MemoryContext for backward compatibility.

        A18: split uses ``primary_split`` (set by MemoryManager.query() from
        ``top_k``) so memories above index 10 are no longer silently discarded
        when top_k > 10.
        """
        ctx = MemoryContext(query=self.query)
        ctx.primary_memories   = self.merged_memories[:self.primary_split]
        ctx.supporting_memories = self.merged_memories[self.primary_split:]
        if self.system_context:
            ctx.principle_nodes = self.system_context.principle_nodes
        if self.user_context:
            ctx.belief_nodes = self.user_context.belief_nodes
            ctx.knowledge_gaps = self.user_context.knowledge_gaps
        ctx.retrieval_latency_ms = self.routing_latency_ms
        return ctx


class MemoryManager:
    """
    Unified memory routing and merging for ADMA.

    Parameters
    ----------
    hgshm : HGSHM
        The shared HGSHM substrate.
    system_memory : SystemMemory
        Operational knowledge store.
    user_memory_cache : dict
        Cache of UserMemory instances by user_id.
    policy_selector : PolicySelector | None
        If set, uses policy-driven routing weights.
    """

    def __init__(
        self,
        hgshm: HGSHM,
        system_memory: SystemMemory,
        policy_selector: Any = None,  # policy.compiler.PolicySelector
    ) -> None:
        self._hgshm   = hgshm
        self._system  = system_memory
        self._users:  dict[str, UserMemory] = {}
        self._selector = policy_selector
        log.debug("MemoryManager initialised")

    def get_user_memory(self, user_id: str) -> UserMemory:
        """Get or create a UserMemory for the given user.

        Validates user_id on every call — the validation is O(1) regex
        and cheap relative to the DB operations that follow.  (ISSUE-010)
        """
        from memory.user.user_memory import validate_user_id
        validate_user_id(user_id)
        if user_id not in self._users:
            self._users[user_id] = UserMemory(self._hgshm, user_id)
        return self._users[user_id]

    # ── Primary query API ─────────────────────────────────────────────

    def query(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 10,
        include_system: bool = True,
        include_user: bool = True,
        include_general: bool = True,
        context: dict[str, Any] | None = None,
    ) -> RoutedContext:
        """
        Route a query across memory domains and return merged results.

        Parameters
        ----------
        query : str
            Natural-language query.
        user_id : str
            Current user (determines which UserMemory to query).
        top_k : int
            Results per domain.
        include_system : bool
            Whether to query SystemMemory.
        include_user : bool
            Whether to query UserMemory.
        include_general : bool
            Whether to query the general HGSHM (cross-domain).
        context : dict
            Additional routing context.
        """
        t0 = time.perf_counter()
        routed = RoutedContext(query=query, primary_split=top_k)

        # ── Domain queries ────────────────────────────────────────────
        if include_system:
            try:
                routed.system_context = self._system.recall(query, top_k=top_k // 2)
                routed.domains_queried.append("system")
            except Exception as exc:
                log.warning("MemoryManager: system query failed: %s", exc)

        if include_user:
            try:
                user_mem = self.get_user_memory(user_id)
                routed.user_context = user_mem.recall(query, top_k=top_k // 2)
                routed.domains_queried.append("user")
            except Exception as exc:
                log.warning("MemoryManager: user query failed: %s", exc)

        if include_general:
            try:
                routed.general_context = self._hgshm.recall(query, top_k=top_k)
                routed.domains_queried.append("general")
            except Exception as exc:
                log.warning("MemoryManager: general query failed: %s", exc)

        # ── Merge and deduplicate ──────────────────────────────────────
        routed.merged_memories = self._merge(
            routed.system_context,
            routed.user_context,
            routed.general_context,
            top_k=top_k * 2,
        )

        routed.routing_latency_ms = (time.perf_counter() - t0) * 1000
        log.debug(
            "MemoryManager: query=%r domains=%s memories=%d latency=%.1fms",
            query[:40], routed.domains_queried,
            routed.total_memories, routed.routing_latency_ms)
        return routed

    def _merge(
        self,
        *contexts: MemoryContext | None,
        top_k: int = 20,
    ) -> list[RetrievedMemory]:
        """Merge RetrievedMemory lists, dedup by node_id, rank by final_score."""
        seen: dict[str, RetrievedMemory] = {}
        for ctx in contexts:
            if ctx is None:
                continue
            for rm in ctx.all_memories:
                nid = rm.node.node_id
                if nid not in seen or rm.final_score > seen[nid].final_score:
                    seen[nid] = rm
        merged = sorted(seen.values(), key=lambda r: r.final_score, reverse=True)
        return merged[:top_k]

    # ── Convenience writes ────────────────────────────────────────────

    def store_system(
        self,
        text: str,
        success: bool = True,
        latency_ms: float = 0.0,
        subsystems: list | None = None,
        metadata: dict | None = None,
    ) -> MemoryNode:
        """Record a workflow event in system memory (A09 — no unsafe **kwargs)."""
        return self._system.store_workflow(
            text,
            success=success,
            latency_ms=latency_ms,
            subsystems=subsystems,
            metadata=metadata,
        )

    def store_user(
        self,
        user_id: str,
        text: str,
        category: str = "general",
        strength: float = 0.8,
        metadata: dict | None = None,
    ) -> MemoryNode:
        """Record a user preference (A09 — no unsafe **kwargs)."""
        return self.get_user_memory(user_id).store_preference(
            category=category,
            preference=text,
            strength=strength,
            metadata=metadata,
        )

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self, user_id: str = "default") -> dict[str, Any]:
        return {
            "hgshm":  self._hgshm.stats(),
            "system": self._system.stats(),
            "user":   self.get_user_memory(user_id).stats(),
        }

    # ── Resource management (ISSUE-011) ──────────────────────────────

    def close(self) -> None:
        """
        Close the underlying HGSHM database connections.

        Must be called when the MemoryManager is no longer needed to
        prevent file descriptor leaks.  In a FastAPI application, call
        this from the lifespan handler's shutdown phase.
        """
        self._hgshm.close()

    def __enter__(self) -> "MemoryManager":
        return self

    def __exit__(self, *_) -> None:
        self.close()

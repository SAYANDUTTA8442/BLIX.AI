"""
memory — Blix Memory Architecture top-level package.

Re-exports the most commonly used public symbols from the three memory
subsystems so callers can write::

    from memory import HGSHM, MemoryManager, SystemMemory, UserMemory

Public API (A25 — explicit __all__):
"""
from memory.hybrid.hgshm import HGSHM
from memory.hybrid.models.memory_node import MemoryNode, MemoryType
from memory.manager import MemoryManager, RoutedContext
from memory.system.system_memory import SystemMemory
from memory.user.user_memory import UserMemory, validate_user_id

__all__ = [
    # Core facades
    "HGSHM",
    "MemoryManager",
    "RoutedContext",
    # System memory
    "SystemMemory",
    # User memory
    "UserMemory",
    "validate_user_id",
    # Node primitives
    "MemoryNode",
    "MemoryType",
]

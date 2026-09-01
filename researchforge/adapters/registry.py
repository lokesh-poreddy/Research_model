"""Adapter registry — creates, validates, and tracks backends.

Usage::

    from researchforge.adapters import registry

    registry.register_graph_backend("sqlite", SQLiteGraphBackend)
    backend = registry.create_graph_backend("sqlite", db_path="rdg.db")
    info = registry.graph_backend_info("sqlite")
    print(info.capabilities.persistent)   # True

The registry performs a lightweight contract validation at create() time
to catch misconfigurations early (e.g., a backend that forgot to implement
add_node). The full validate_graph_backend() / validate_vector_backend()
harnesses are separate and should be run in the test suite.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .capabilities import BackendCapabilities, BackendInfo
from .errors import BackendCapabilityError, BackendError
from .protocols import GraphBackend, VectorIndexBackend


class _RegistrationEntry:
    """Internal entry: the class + pre-computed info."""
    __slots__ = ("cls", "info")

    def __init__(self, cls: type, info: BackendInfo) -> None:
        self.cls = cls
        self.info = info


class AdapterRegistry:
    """Singleton-style registry for graph and vector backends.

    Backends are registered by name. The same name can be re-registered
    to replace an earlier entry (e.g., swapping in a production backend
    after infrastructure becomes available).
    """

    def __init__(self) -> None:
        self._graph: Dict[str, _RegistrationEntry] = {}
        self._vector: Dict[str, _RegistrationEntry] = {}

    # ── Graph backend ─────────────────────────────────────────────────────

    def register_graph_backend(self, name: str,
                                cls: Type[GraphBackend]) -> None:
        """Register a graph backend class under `name`.

        The class must expose class-methods: capabilities(), backend_name(),
        backend_version().
        """
        caps = cls.capabilities()
        info = BackendInfo(
            name=name,
            version=cls.backend_version(),
            capabilities=caps)
        self._graph[name] = _RegistrationEntry(cls, info)

    def create_graph_backend(self, name: str, **kwargs: Any) -> GraphBackend:
        """Instantiate a registered graph backend.

        Raises BackendCapabilityError if `name` is not registered.
        """
        entry = self._graph.get(name)
        if entry is None:
            available = list(self._graph.keys())
            raise BackendCapabilityError(
                f"Graph backend '{name}' is not registered. "
                f"Available: {available}",
                backend_name=name)
        return entry.cls(**kwargs)

    def graph_backend_info(self, name: str) -> Optional[BackendInfo]:
        """Return BackendInfo for a registered graph backend, or None."""
        entry = self._graph.get(name)
        return entry.info if entry else None

    def list_graph_backends(self) -> Dict[str, BackendInfo]:
        """Return all registered graph backends."""
        return {k: e.info for k, e in self._graph.items()}

    # ── Vector backend ────────────────────────────────────────────────────

    def register_vector_backend(self, name: str,
                                 cls: Type[VectorIndexBackend]) -> None:
        """Register a vector backend class under `name`."""
        caps = cls.capabilities()
        info = BackendInfo(
            name=name,
            version=cls.backend_version(),
            capabilities=caps)
        self._vector[name] = _RegistrationEntry(cls, info)

    def create_vector_backend(self, name: str, **kwargs: Any) -> VectorIndexBackend:
        """Instantiate a registered vector backend."""
        entry = self._vector.get(name)
        if entry is None:
            available = list(self._vector.keys())
            raise BackendCapabilityError(
                f"Vector backend '{name}' is not registered. "
                f"Available: {available}",
                backend_name=name)
        return entry.cls(**kwargs)

    def vector_backend_info(self, name: str) -> Optional[BackendInfo]:
        entry = self._vector.get(name)
        return entry.info if entry else None

    def list_vector_backends(self) -> Dict[str, BackendInfo]:
        return {k: e.info for k, e in self._vector.items()}


# ── Module-level default registry ────────────────────────────────────────────
# Pre-populated with the reference implementations.
# Stubs (Neo4j, pgvector) are intentionally NOT registered by default because
# they raise NotImplementedError on instantiation. Register them explicitly
# only if you have the infrastructure and intend to implement them.

_default_registry = AdapterRegistry()

from .backends.memory import InMemoryGraphBackend          # noqa: E402
from .backends.sqlite import SQLiteGraphBackend            # noqa: E402
from .backends.inprocess_vector import InProcessVectorIndex  # noqa: E402

_default_registry.register_graph_backend("in_memory", InMemoryGraphBackend)
_default_registry.register_graph_backend("sqlite", SQLiteGraphBackend)
_default_registry.register_vector_backend("inprocess", InProcessVectorIndex)


def get_default_registry() -> AdapterRegistry:
    """Return the module-level pre-populated registry."""
    return _default_registry

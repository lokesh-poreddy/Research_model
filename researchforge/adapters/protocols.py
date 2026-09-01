"""Protocol contracts for ResearchForge storage backends.

Two Protocol classes define the complete interface contracts:

  GraphBackend   — generic node/edge storage. Has NO knowledge of RDG
                   semantic rules (IDENTIFIES, MOTIVATES, TESTED_BY, etc.).
                   Those constraints live entirely in rdg/graph.py.

  VectorIndexBackend — embedding storage and similarity search. Has NO
                       knowledge of what a MemoryRecord or TrajectoryRecord
                       is; it stores (id, vector, metadata) triples.

Both protocols are runtime-checkable via isinstance(). Any class that
implements the required methods is a valid backend — no inheritance needed.

Transaction contract
--------------------
Backends that support transactions MUST implement the context-manager
protocol on GraphTransaction:

    with backend.transaction() as txn:
        txn.add_node(...)
        txn.add_edge(...)
    # automatic commit on clean exit, rollback on exception

Backends that do NOT support transactions should raise BackendCapabilityError
on .transaction().

Dimension contract (VectorIndexBackend)
----------------------------------------
A backend must declare its dimension at construction time. Any add() call
with a vector of different length raises DimensionMismatchError immediately.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Mapping, Optional, Sequence, Tuple

from .capabilities import BackendCapabilities, HealthStatus
from .errors import BackendCapabilityError


# ── Graph backend ────────────────────────────────────────────────────────────


class GraphTransaction:
    """Context object returned by GraphBackend.transaction().

    The minimal transactional API: add_node, add_edge, commit, rollback.
    Any implementation of GraphBackend that supports transactions returns
    a concrete subclass of this from .transaction().
    """

    def add_node(self, node_id: str, node_type: str, content: str,
                 timestamp: str, attributes_json: str) -> None:
        raise NotImplementedError

    def add_edge(self, from_id: str, to_id: str, relation: str,
                 properties_json: str) -> None:
        raise NotImplementedError

    def commit(self) -> None:
        raise NotImplementedError

    def rollback(self) -> None:
        raise NotImplementedError


class GraphBackend:
    """Protocol for generic graph node/edge storage.

    Implementors must support all methods below. The backend stores data
    as plain strings/dicts — it has no knowledge of RDG semantic rules.
    JSON serialization of attributes/properties is the backend's
    responsibility (incoming = JSON string; outgoing = parsed dict).

    Lifecycle
    ---------
    Backends are open on construction. Call close() when done; after close(),
    all methods except health_check() may raise BackendConnectionError.
    Use as a context manager for automatic close():

        with SQLiteGraphBackend("path.db") as b:
            b.add_node(...)

    Schema
    ------
    Node dict: {id, type, content, timestamp, attributes: dict}
    Edge dict: {from_id, to_id, relation, properties: dict}
    """

    # ── Required capabilities metadata ────────────────────────────────────
    @classmethod
    def capabilities(cls) -> BackendCapabilities:  # noqa: D102
        raise NotImplementedError

    @classmethod
    def backend_name(cls) -> str:  # noqa: D102
        raise NotImplementedError

    @classmethod
    def backend_version(cls) -> str:  # noqa: D102
        return "1.0"

    # ── Node operations ───────────────────────────────────────────────────
    def add_node(self, node_id: str, node_type: str, content: str,
                 timestamp: str, attributes_json: str) -> None:
        """Upsert a node. attributes_json is a JSON-encoded dict."""
        raise NotImplementedError

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return node dict or None if not found.

        Dict keys: id, type, content, timestamp, attributes (parsed dict).
        """
        raise NotImplementedError

    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Return all nodes with the given type."""
        raise NotImplementedError

    def all_nodes(self) -> List[Dict[str, Any]]:
        """Return all stored nodes."""
        raise NotImplementedError

    # ── Edge operations ───────────────────────────────────────────────────
    def add_edge(self, from_id: str, to_id: str, relation: str,
                 properties_json: str) -> None:
        """Append an edge. properties_json is a JSON-encoded dict.

        NOTE: the backend does NOT validate whether (from_type, to_type)
        is a legal combination for this relation. That is the RDG domain
        layer's responsibility.
        """
        raise NotImplementedError

    def get_out_edges(self, node_id: str) -> List[Dict[str, Any]]:
        """Return all edges where from_id == node_id.

        Dict keys: from_id, to_id, relation, properties (parsed dict).
        """
        raise NotImplementedError

    def get_in_edges(self, node_id: str) -> List[Dict[str, Any]]:
        """Return all edges where to_id == node_id."""
        raise NotImplementedError

    def all_edges(self) -> List[Dict[str, Any]]:
        """Return all stored edges."""
        raise NotImplementedError

    # ── Transaction ───────────────────────────────────────────────────────
    @contextmanager
    def transaction(self) -> Generator["GraphTransaction", None, None]:
        """Context manager for atomic multi-operation writes.

        Usage::

            with backend.transaction() as txn:
                txn.add_node(...)
                txn.add_edge(...)
            # commits on clean exit; rolls back on exception

        Backends that declare capabilities().transactional = False
        must raise BackendCapabilityError here.
        """
        raise BackendCapabilityError(
            "This backend does not support transactions",
            backend_name=self.backend_name())

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        """Release all resources. Safe to call multiple times."""

    def health_check(self) -> HealthStatus:
        """Lightweight liveness probe. Must not raise."""
        raise NotImplementedError

    def __enter__(self) -> "GraphBackend":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ── Vector index backend ─────────────────────────────────────────────────────


class VectorIndexBackend:
    """Protocol for embedding storage and similarity search.

    The backend stores (id, vector, metadata) triples. It has no knowledge
    of what the id refers to (MemoryRecord, TrajectoryRecord, etc.).

    Dimension contract
    ------------------
    Declared at construction. Every add() call with a different-length
    vector raises DimensionMismatchError immediately — no silent truncation
    or padding.

    Metadata
    --------
    Optional per-vector metadata dict (strings, numbers, bools only).
    Useful for pre-filtering search results without a second lookup.
    Examples: {"strategy": "increase_capacity", "model_family": "MLP",
               "reliability": 0.87, "tier": "long_term"}
    """

    # ── Required capabilities metadata ────────────────────────────────────
    @classmethod
    def capabilities(cls) -> BackendCapabilities:  # noqa: D102
        raise NotImplementedError

    @classmethod
    def backend_name(cls) -> str:  # noqa: D102
        raise NotImplementedError

    @classmethod
    def backend_version(cls) -> str:  # noqa: D102
        return "1.0"

    # ── Dimension / metric ────────────────────────────────────────────────
    @property
    def dimension(self) -> int:
        """Fixed vector dimension for this index."""
        raise NotImplementedError

    @property
    def metric(self) -> str:
        """Similarity metric: 'cosine' | 'dot' | 'l2'."""
        raise NotImplementedError

    # ── Write ─────────────────────────────────────────────────────────────
    def add(self, vector_id: str, vector: Sequence[float],
            metadata: Optional[Mapping[str, Any]] = None) -> None:
        """Insert or replace a vector.

        Raises DimensionMismatchError if len(vector) != self.dimension.
        """
        raise NotImplementedError

    def remove(self, vector_id: str) -> None:
        """Remove a vector by id. No-op if id is not present."""
        raise NotImplementedError

    # ── Read ──────────────────────────────────────────────────────────────
    def search(self, query_vector: Sequence[float],
               k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Return (id, score) pairs in descending similarity order.

        If k is None, return all stored vectors ranked.
        Raises DimensionMismatchError if len(query_vector) != self.dimension.
        """
        raise NotImplementedError

    def get_metadata(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Return stored metadata dict, or None if id not found."""
        raise NotImplementedError

    def size(self) -> int:
        """Number of vectors currently stored."""
        raise NotImplementedError

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        """Release all resources. Safe to call multiple times."""

    def health_check(self) -> HealthStatus:
        """Lightweight liveness probe. Must not raise."""
        raise NotImplementedError

    def __enter__(self) -> "VectorIndexBackend":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

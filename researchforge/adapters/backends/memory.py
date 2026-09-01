"""In-memory graph backend — zero-dependency reference implementation.

All data lives in Python dicts. No persistence across process restarts.
Transactions are supported via a simple copy-on-write rollback mechanism:
the in-flight transaction writes to a separate buffer; commit() merges it
into the main store; rollback() discards the buffer.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from ..capabilities import BackendCapabilities, HealthStatus
from ..errors import BackendTransactionError
from ..protocols import GraphBackend, GraphTransaction


class _MemoryTransaction(GraphTransaction):
    """Copy-on-write transaction for InMemoryGraphBackend."""

    def __init__(self, owner: "InMemoryGraphBackend") -> None:
        self._owner = owner
        self._node_buffer: Dict[str, Dict[str, Any]] = {}
        self._edge_buffer: List[Dict[str, Any]] = []
        self._committed = False
        self._rolled_back = False

    def add_node(self, node_id: str, node_type: str, content: str,
                 timestamp: str, attributes_json: str) -> None:
        self._node_buffer[node_id] = {
            "id": node_id, "type": node_type, "content": content,
            "timestamp": timestamp, "attributes": json.loads(attributes_json),
        }

    def add_edge(self, from_id: str, to_id: str, relation: str,
                 properties_json: str) -> None:
        self._edge_buffer.append({
            "from_id": from_id, "to_id": to_id, "relation": relation,
            "properties": json.loads(properties_json),
        })

    def commit(self) -> None:
        if self._committed or self._rolled_back:
            raise BackendTransactionError(
                "Transaction already finalized",
                backend_name=InMemoryGraphBackend.backend_name())
        self._owner._nodes.update(self._node_buffer)
        self._owner._edges.extend(self._edge_buffer)
        self._committed = True

    def rollback(self) -> None:
        self._rolled_back = True
        self._node_buffer.clear()
        self._edge_buffer.clear()


class InMemoryGraphBackend(GraphBackend):
    """Pure in-memory graph backend. Zero dependencies, always available.

    This is the fastest backend and the default when no persistence is
    required (e.g., in unit tests, short-lived research loops, or as a
    staging area before writing to a durable backend).
    """

    BACKEND_NAME = "in_memory"
    BACKEND_VERSION = "1.0"

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: List[Dict[str, Any]] = []
        self._closed = False

    # ── Capabilities ──────────────────────────────────────────────────────
    @classmethod
    def capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            persistent=False,
            transactional=True,
            supports_batch_insert=True,
            production_ready=False)

    @classmethod
    def backend_name(cls) -> str:
        return cls.BACKEND_NAME

    @classmethod
    def backend_version(cls) -> str:
        return cls.BACKEND_VERSION

    # ── Nodes ─────────────────────────────────────────────────────────────
    def add_node(self, node_id: str, node_type: str, content: str,
                 timestamp: str, attributes_json: str) -> None:
        self._nodes[node_id] = {
            "id": node_id, "type": node_type, "content": content,
            "timestamp": timestamp, "attributes": json.loads(attributes_json),
        }

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return dict(self._nodes[node_id]) if node_id in self._nodes else None

    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        return [dict(n) for n in self._nodes.values() if n["type"] == node_type]

    def all_nodes(self) -> List[Dict[str, Any]]:
        return [dict(n) for n in self._nodes.values()]

    # ── Edges ─────────────────────────────────────────────────────────────
    def add_edge(self, from_id: str, to_id: str, relation: str,
                 properties_json: str) -> None:
        self._edges.append({
            "from_id": from_id, "to_id": to_id, "relation": relation,
            "properties": json.loads(properties_json),
        })

    def get_out_edges(self, node_id: str) -> List[Dict[str, Any]]:
        return [dict(e) for e in self._edges if e["from_id"] == node_id]

    def get_in_edges(self, node_id: str) -> List[Dict[str, Any]]:
        return [dict(e) for e in self._edges if e["to_id"] == node_id]

    def all_edges(self) -> List[Dict[str, Any]]:
        return [dict(e) for e in self._edges]

    # ── Transactions ──────────────────────────────────────────────────────
    @contextmanager
    def transaction(self) -> Generator[_MemoryTransaction, None, None]:
        txn = _MemoryTransaction(self)
        try:
            yield txn
            if not txn._committed and not txn._rolled_back:
                txn.commit()
        except Exception:
            if not txn._committed:
                txn.rollback()
            raise

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        self._closed = True

    def health_check(self) -> HealthStatus:
        t0 = time.monotonic()
        _ = len(self._nodes)  # trivial probe
        latency = (time.monotonic() - t0) * 1000
        return HealthStatus(
            healthy=not self._closed,
            persistent=False, transactional=True,
            latency_ms=latency,
            message="" if not self._closed else "Backend is closed")

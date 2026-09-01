"""SQLite graph backend — single-file persistent implementation.

Conforms to the GraphBackend protocol. All data is stored in a SQLite
database file; data survives process restarts. Transactions use SQLite's
own BEGIN / COMMIT / ROLLBACK and are verified by the contract test suite.

This is the current substitute for the Neo4j production deployment described
in the design document. The public API is identical to what Neo4j would
expose; the only difference is that queries are SQLite SELECT statements
rather than Cypher MATCH clauses.

Migration path to Neo4j
------------------------
Replace this class with Neo4jGraphBackend. The RDG domain layer and all
callers above it need zero changes because they interact only with the
GraphBackend protocol.

The mapping is straightforward:
  SQLite table rdg_nodes       → Neo4j node (label = type)
  SQLite table rdg_edges       → Neo4j relationship (type = relation)
  attributes_json column       → Neo4j node properties (dict)
  properties_json column       → Neo4j relationship properties (dict)
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from ..capabilities import BackendCapabilities, HealthStatus
from ..errors import BackendConnectionError, BackendTransactionError
from ..protocols import GraphBackend, GraphTransaction

_DDL = """
CREATE TABLE IF NOT EXISTS rdg_nodes (
    id           TEXT PRIMARY KEY,
    type         TEXT NOT NULL,
    content      TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    attributes   TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS rdg_edges (
    rowid        INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id      TEXT NOT NULL,
    to_id        TEXT NOT NULL,
    relation     TEXT NOT NULL,
    properties   TEXT NOT NULL DEFAULT '{}'
);
"""


class _SqliteTransaction(GraphTransaction):
    """Wraps an active sqlite3 connection that has BEGIN outstanding."""

    def __init__(self, con: sqlite3.Connection,
                 backend_name: str) -> None:
        self._con = con
        self._backend_name = backend_name
        self._done = False

    def add_node(self, node_id: str, node_type: str, content: str,
                 timestamp: str, attributes_json: str) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO rdg_nodes VALUES (?,?,?,?,?)",
            (node_id, node_type, content, timestamp, attributes_json))

    def add_edge(self, from_id: str, to_id: str, relation: str,
                 properties_json: str) -> None:
        self._con.execute(
            "INSERT INTO rdg_edges(from_id,to_id,relation,properties) VALUES (?,?,?,?)",
            (from_id, to_id, relation, properties_json))

    def commit(self) -> None:
        if self._done:
            raise BackendTransactionError(
                "Transaction already finalized", backend_name=self._backend_name)
        self._con.commit()
        self._done = True

    def rollback(self) -> None:
        if not self._done:
            self._con.rollback()
        self._done = True


class SQLiteGraphBackend(GraphBackend):
    """SQLite-backed graph backend. Data persists to a single file.

    For in-memory operation (no persistence, testing), pass ':memory:'.
    """

    BACKEND_NAME = "sqlite"
    BACKEND_VERSION = "1.0"

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        try:
            self._con = sqlite3.connect(db_path, check_same_thread=False)
            # Use autocommit-off by default (isolation_level not None).
            # We manage BEGIN/COMMIT explicitly in transaction().
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.executescript(_DDL)
            self._con.commit()
        except sqlite3.Error as exc:
            raise BackendConnectionError(
                f"Cannot open SQLite database at {db_path!r}",
                backend_name=self.BACKEND_NAME, cause=exc) from exc
        self._closed = False

    # ── Capabilities ──────────────────────────────────────────────────────
    @classmethod
    def capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            persistent=True,
            transactional=True,
            supports_batch_insert=True,
            production_ready=False)  # reference impl, not production-grade

    @classmethod
    def backend_name(cls) -> str:
        return cls.BACKEND_NAME

    @classmethod
    def backend_version(cls) -> str:
        return cls.BACKEND_VERSION

    # ── Nodes ─────────────────────────────────────────────────────────────
    def add_node(self, node_id: str, node_type: str, content: str,
                 timestamp: str, attributes_json: str) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO rdg_nodes VALUES (?,?,?,?,?)",
            (node_id, node_type, content, timestamp, attributes_json))
        self._con.commit()

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        row = self._con.execute(
            "SELECT id,type,content,timestamp,attributes FROM rdg_nodes WHERE id=?",
            (node_id,)).fetchone()
        return self._row_to_node(row) if row else None

    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        rows = self._con.execute(
            "SELECT id,type,content,timestamp,attributes FROM rdg_nodes WHERE type=?",
            (node_type,)).fetchall()
        return [self._row_to_node(r) for r in rows]

    def all_nodes(self) -> List[Dict[str, Any]]:
        rows = self._con.execute(
            "SELECT id,type,content,timestamp,attributes FROM rdg_nodes").fetchall()
        return [self._row_to_node(r) for r in rows]

    @staticmethod
    def _row_to_node(row: tuple) -> Dict[str, Any]:
        return {
            "id": row[0], "type": row[1], "content": row[2],
            "timestamp": row[3], "attributes": json.loads(row[4]),
        }

    # ── Edges ─────────────────────────────────────────────────────────────
    def add_edge(self, from_id: str, to_id: str, relation: str,
                 properties_json: str) -> None:
        self._con.execute(
            "INSERT INTO rdg_edges(from_id,to_id,relation,properties) VALUES (?,?,?,?)",
            (from_id, to_id, relation, properties_json))
        self._con.commit()

    def get_out_edges(self, node_id: str) -> List[Dict[str, Any]]:
        rows = self._con.execute(
            "SELECT from_id,to_id,relation,properties FROM rdg_edges WHERE from_id=?",
            (node_id,)).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_in_edges(self, node_id: str) -> List[Dict[str, Any]]:
        rows = self._con.execute(
            "SELECT from_id,to_id,relation,properties FROM rdg_edges WHERE to_id=?",
            (node_id,)).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def all_edges(self) -> List[Dict[str, Any]]:
        rows = self._con.execute(
            "SELECT from_id,to_id,relation,properties FROM rdg_edges").fetchall()
        return [self._row_to_edge(r) for r in rows]

    @staticmethod
    def _row_to_edge(row: tuple) -> Dict[str, Any]:
        return {
            "from_id": row[0], "to_id": row[1],
            "relation": row[2], "properties": json.loads(row[3]),
        }

    # ── Transactions ──────────────────────────────────────────────────────
    @contextmanager
    def transaction(self) -> Generator[_SqliteTransaction, None, None]:
        """Atomic multi-operation write via SQLite BEGIN/COMMIT/ROLLBACK.

        Commit happens automatically on clean exit. Any exception triggers
        rollback, re-raising the original exception unchanged.
        """
        self._con.execute("BEGIN")
        txn = _SqliteTransaction(self._con, self.BACKEND_NAME)
        try:
            yield txn
            if not txn._done:
                txn.commit()
        except Exception:
            if not txn._done:
                txn.rollback()
            raise

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        if not self._closed:
            self._con.close()
            self._closed = True

    def health_check(self) -> HealthStatus:
        try:
            t0 = time.monotonic()
            self._con.execute("SELECT 1").fetchone()
            latency = (time.monotonic() - t0) * 1000
            return HealthStatus(
                healthy=True, persistent=True, transactional=True,
                latency_ms=latency)
        except sqlite3.Error as exc:
            return HealthStatus(
                healthy=False, persistent=True, transactional=True,
                latency_ms=0.0, message=str(exc))

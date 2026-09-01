"""Research Development Graph (RDG): the persistent, typed graph that
represents the state of an autonomous research process -- Problem -> Gap ->
Hypothesis -> Experiment -> Finding -> Claim, plus ModelGenome / Strategy /
MemoryRecord / Insight nodes -- as described in ResearchForge-ECRM Sec. 1 and
Sec. 3 (both source documents).

Storage backends
----------------
The RDG supports two construction paths, both producing identical behaviour:

  OLD (RF-0.x, fully preserved):
      rdg = ResearchDevelopmentGraph(db_path="research.db")
      rdg = ResearchDevelopmentGraph()                      # in-memory

  NEW (RF-1.0+, opt-in):
      from researchforge.adapters import SQLiteGraphBackend, InMemoryGraphBackend
      rdg = ResearchDevelopmentGraph(backend=SQLiteGraphBackend("research.db"))
      rdg = ResearchDevelopmentGraph(backend=InMemoryGraphBackend())

The old db_path path is preserved for full backward compatibility: passing
db_path internally constructs a SQLiteGraphBackend (or InMemoryGraphBackend
when db_path is None). Passing backend= overrides that.

CRITICAL ARCHITECTURAL RULE (AD-003):
  The backend is generic storage. ALL semantic validation (typed-relation
  constraints, schema enforcement) lives HERE in this class, never in the
  backend. Swapping the backend must not require touching any research logic.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .schema import EDGE_TYPE_CONSTRAINTS, validate_edge, validate_node
from ..adapters.protocols import GraphBackend


RDG_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str = "n") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class RDGNode:
    id: str
    type: str
    content: str
    timestamp: str = field(default_factory=_now)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "type": self.type, "content": self.content,
                "timestamp": self.timestamp, "attributes": self.attributes}


@dataclass
class RDGEdge:
    from_id: str
    to_id: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"from_id": self.from_id, "to_id": self.to_id,
                "relation": self.relation, "properties": self.properties}


class EdgeConstraintError(ValueError):
    """Raised when an edge would violate the RDG's typed-relation rules.

    This is a DOMAIN error raised by the RDG layer. It is NOT a BackendError.
    The storage backend has no knowledge of this constraint.
    """


class ResearchDevelopmentGraph:
    """Typed Research Development Graph with pluggable storage backend.

    Maintains an in-process node/edge index (self.nodes, self.edges) for
    fast traversal. Writes are mirrored to the storage backend so that data
    can survive process restarts (for persistent backends) and be read by
    other processes.

    The in-process index is the authoritative source for traversal queries
    (children, parents, evidence_chain). The backend is the persistence
    source. On construction, only the in-process index is populated; the
    backend is written on every add_node/add_edge call.

    If you need to reconstruct the in-process index from an existing backend
    (e.g., after a process restart), call _reload_from_backend().
    """

    def __init__(self, db_path: Optional[str] = None,
                 backend: Optional[GraphBackend] = None) -> None:
        """Construct an RDG.

        Parameters
        ----------
        db_path : str | None
            RF-0.x backward-compatible path. If given, constructs a
            SQLiteGraphBackend at that path. Mutually exclusive with backend=.
        backend : GraphBackend | None
            RF-1.0+ explicit backend. Use this for InMemoryGraphBackend,
            SQLiteGraphBackend with custom options, or any future backend.
            Takes precedence over db_path if both are (accidentally) given.
        """
        # In-process traversal index (always present)
        self.nodes: Dict[str, RDGNode] = {}
        self.edges: List[RDGEdge] = []
        self._out: Dict[str, List[int]] = {}
        self._in: Dict[str, List[int]] = {}

        # Backend selection — NEW path takes precedence
        if backend is not None:
            self._backend: GraphBackend = backend
            self.db_path: Optional[str] = None
        elif db_path is not None:
            from ..adapters.backends.sqlite import SQLiteGraphBackend
            self._backend = SQLiteGraphBackend(db_path)
            self.db_path = db_path
        else:
            from ..adapters.backends.memory import InMemoryGraphBackend
            self._backend = InMemoryGraphBackend()
            self.db_path = None

    # -- construction ---------------------------------------------------------
    def add_node(self, type: str, content: str, attributes: Optional[Dict] = None,
                 node_id: Optional[str] = None) -> RDGNode:
        node = RDGNode(id=node_id or new_id(type[:1].lower()), type=type,
                        content=content, attributes=attributes or {})
        validate_node(node.to_dict())
        self.nodes[node.id] = node
        self._backend.add_node(
            node.id, node.type, node.content, node.timestamp,
            json.dumps(node.attributes))
        return node

    def add_edge(self, from_id: str, to_id: str, relation: str,
                 properties: Optional[Dict] = None,
                 enforce_types: bool = True) -> RDGEdge:
        if from_id not in self.nodes or to_id not in self.nodes:
            raise KeyError("Both endpoints must already exist as RDG nodes")
        # ── DOMAIN CONSTRAINT — lives here, not in the backend ─────────────
        if enforce_types and relation in EDGE_TYPE_CONSTRAINTS:
            pair = (self.nodes[from_id].type, self.nodes[to_id].type)
            if pair not in EDGE_TYPE_CONSTRAINTS[relation]:
                raise EdgeConstraintError(
                    f"Edge '{relation}' does not allow {pair}; "
                    f"allowed: {sorted(EDGE_TYPE_CONSTRAINTS[relation])}")
        edge = RDGEdge(from_id=from_id, to_id=to_id, relation=relation,
                        properties=properties or {})
        validate_edge(edge.to_dict())
        idx = len(self.edges)
        self.edges.append(edge)
        self._out.setdefault(from_id, []).append(idx)
        self._in.setdefault(to_id, []).append(idx)
        self._backend.add_edge(
            edge.from_id, edge.to_id, edge.relation,
            json.dumps(edge.properties))
        return edge

    # -- traversal --------------------------------------------------------------
    def children(self, node_id: str, relation: Optional[str] = None) -> List[RDGNode]:
        out = []
        for idx in self._out.get(node_id, []):
            e = self.edges[idx]
            if relation is None or e.relation == relation:
                out.append(self.nodes[e.to_id])
        return out

    def parents(self, node_id: str, relation: Optional[str] = None) -> List[RDGNode]:
        out = []
        for idx in self._in.get(node_id, []):
            e = self.edges[idx]
            if relation is None or e.relation == relation:
                out.append(self.nodes[e.from_id])
        return out

    def evidence_chain(self, node_id: str) -> List[RDGNode]:
        """Walk backwards from a node (e.g. a Claim) to its Problem root --
        the chain `diagnose_failure` inspects in the design doc's Sec. 4.2."""
        chain = [self.nodes[node_id]]
        current = node_id
        seen = {current}
        while True:
            ps = [p for p in self.parents(current) if p.id not in seen]
            if not ps:
                break
            current = ps[0].id
            seen.add(current)
            chain.append(self.nodes[current])
        return list(reversed(chain))

    def nodes_by_type(self, type: str) -> List[RDGNode]:
        return [n for n in self.nodes.values() if n.type == type]

    # -- reconstruction ---------------------------------------------------------
    def _reload_from_backend(self) -> None:
        """Repopulate the in-process traversal index from the backend.

        Use after constructing an RDG on top of an existing persistent backend
        (e.g., reopening a SQLite file from a previous run). Does NOT clear
        existing in-process state first — call on a freshly constructed RDG.
        """
        for ndict in self._backend.all_nodes():
            node = RDGNode(
                id=ndict["id"], type=ndict["type"],
                content=ndict["content"], timestamp=ndict["timestamp"],
                attributes=ndict["attributes"])
            self.nodes[node.id] = node
        for edict in self._backend.all_edges():
            edge = RDGEdge(
                from_id=edict["from_id"], to_id=edict["to_id"],
                relation=edict["relation"], properties=edict["properties"])
            idx = len(self.edges)
            self.edges.append(edge)
            self._out.setdefault(edge.from_id, []).append(idx)
            self._in.setdefault(edge.to_id, []).append(idx)

    # -- introspection -----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": [n.to_dict() for n in self.nodes.values()],
                "edges": [e.to_dict() for e in self.edges]}

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for n in self.nodes.values():
            counts[n.type] = counts.get(n.type, 0) + 1
        counts["_edges"] = len(self.edges)
        return counts

    # -- lifecycle ---------------------------------------------------------------
    def close(self) -> None:
        self._backend.close()

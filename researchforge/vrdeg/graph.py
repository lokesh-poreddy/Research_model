from __future__ import annotations

import json
import hashlib
from typing import Dict, List, Optional, Tuple
from researchforge.vrdeg.node import GraphNode
from researchforge.vrdeg.edge import GraphEdge, RelationType


class GraphIntegrityError(Exception):
    pass


class VRDEG:
    """In-process Versioned Research Development & Evidence Graph.

    Stores nodes and edges in-memory with deterministic export.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}

    def add_node(self, node: GraphNode) -> None:
        if node.id in self._nodes:
            raise GraphIntegrityError(f"duplicate node id: {node.id}")
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def update_node(self, node: GraphNode) -> None:
        # versioning: allow update only as new version with parent_version_of
        if node.id in self._nodes:
            raise GraphIntegrityError("use add_versioned_node to version an existing node")
        if node.parent_version_of and node.parent_version_of not in self._nodes:
            raise GraphIntegrityError(f"parent version {node.parent_version_of} not found")
        self._nodes[node.id] = node

    def add_versioned_node(self, new_node: GraphNode, previous_id: str) -> None:
        if new_node.id in self._nodes:
            raise GraphIntegrityError("duplicate node id")
        if previous_id not in self._nodes:
            raise GraphIntegrityError("previous node not found")
        self._nodes[new_node.id] = new_node

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.id in self._edges:
            raise GraphIntegrityError(f"duplicate edge id: {edge.id}")
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            raise GraphIntegrityError("edge references unknown node")
        # disallow trivial self-relations for most types
        if edge.source_id == edge.target_id and edge.relation not in (RelationType.PRECEDES.value, RelationType.BRANCH_OF.value):
            raise GraphIntegrityError("illegal self-relation")
        self._edges[edge.id] = edge

    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        return self._edges.get(edge_id)

    def edges_for_node(self, node_id: str) -> List[GraphEdge]:
        return [e for e in self._edges.values() if e.source_id == node_id or e.target_id == node_id]

    def predecessors(self, node_id: str) -> List[GraphNode]:
        preds = [self._nodes[e.source_id] for e in self._edges.values() if e.target_id == node_id]
        return preds

    def successors(self, node_id: str) -> List[GraphNode]:
        succ = [self._nodes[e.target_id] for e in self._edges.values() if e.source_id == node_id]
        return succ

    def trace_lineage(self, node_id: str, depth: int = 10) -> List[GraphNode]:
        seen = []
        frontier = [node_id]
        for _ in range(depth):
            next_frontier = []
            for nid in frontier:
                preds = [e.source_id for e in self._edges.values() if e.target_id == nid]
                for p in preds:
                    if p not in seen:
                        seen.append(p)
                        next_frontier.append(p)
            frontier = next_frontier
        return [self._nodes[s] for s in seen]

    def validate_integrity(self) -> None:
        # referenced nodes exist already enforced on add_edge
        # ensure provenance references exist as nodes if present
        for node in self._nodes.values():
            if node.provenance_id and node.provenance_id not in self._nodes:
                raise GraphIntegrityError(f"provenance {node.provenance_id} missing for node {node.id}")

    def export_canonical(self) -> str:
        # deterministic JSON: sort nodes and edges by id
        nodes = [self._nodes[k].to_dict() for k in sorted(self._nodes.keys())]
        edges = [self._edges[k].to_dict() for k in sorted(self._edges.keys())]
        payload = {"nodes": nodes, "edges": edges}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def fingerprint(self) -> str:
        j = self.export_canonical()
        return hashlib.sha256(j.encode("utf-8")).hexdigest()

    def import_canonical(self, j: str) -> None:
        payload = json.loads(j)
        for n in payload.get("nodes", []):
            self.add_node(GraphNode.from_dict(n))
        for e in payload.get("edges", []):
            self.add_edge(GraphEdge.from_dict(e))

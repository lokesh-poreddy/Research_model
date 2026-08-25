"""
Core Research Development Graph (RDG) implementation.

Backed by an in-memory NetworkX DiGraph; persisted to JSON on disk.
Optional Neo4j sync is handled via neo4j_client.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import networkx as nx

from rdg.consistency import find_broken_chains, semantically_aligned
from rdg.edges import EdgeRelation, RDGEdge
from rdg.nodes import NodeStatus, NodeType, RDGNode


class ResearchDevelopmentGraph:
    """
    The RDG is a directed typed graph representing the full
    research state: Problems, Gaps, Hypotheses, Experiments,
    Findings, Claims, and their typed relationships.
    """

    def __init__(self, graph_id: str = "main"):
        self.graph_id = graph_id
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, RDGNode] = {}
        self._edges: Dict[str, RDGEdge] = {}

    # ── Node CRUD ──────────────────────────────────────────────────────────────

    def add_node(self, node: RDGNode, validate: bool = True) -> RDGNode:
        """Insert a node into the RDG."""
        if validate and node.id in self._nodes:
            raise ValueError(f"Node '{node.id}' already exists.")
        self._nodes[node.id] = node
        self._graph.add_node(node.id, **node.to_dict())
        return node

    def get_node(self, node_id: str) -> Optional[RDGNode]:
        return self._nodes.get(node_id)

    def update_node(self, node_id: str, **kwargs: Any) -> RDGNode:
        node = self._nodes[node_id]
        for k, v in kwargs.items():
            if hasattr(node, k):
                setattr(node, k, v)
            else:
                node.attributes[k] = v
        self._graph.nodes[node_id].update(node.to_dict())
        return node

    def remove_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        if node_id in self._graph:
            self._graph.remove_node(node_id)

    # ── Edge CRUD ─────────────────────────────────────────────────────────────

    def add_edge(self, edge: RDGEdge, validate_semantics: bool = True) -> RDGEdge:
        """Insert a typed edge; optionally enforce semantic constraints."""
        if validate_semantics:
            broken = find_broken_chains(self._nodes, [edge])
            if broken:
                raise ValueError(broken[0][1])
        self._edges[edge.id] = edge
        self._graph.add_edge(
            edge.from_node,
            edge.to_node,
            key=edge.id,
            relation=edge.relation.value,
            **edge.properties,
        )
        return edge

    def connect(
        self,
        from_id: str,
        to_id: str,
        relation: EdgeRelation,
        confidence: float = 1.0,
        validate: bool = True,
    ) -> RDGEdge:
        """Convenience method: create and add an edge."""
        edge = RDGEdge(
            from_node=from_id,
            to_node=to_id,
            relation=relation,
            confidence=confidence,
        )
        return self.add_edge(edge, validate_semantics=validate)

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def hypotheses(self) -> List[RDGNode]:
        return [n for n in self._nodes.values() if n.type == NodeType.HYPOTHESIS]

    @property
    def experiments(self) -> List[RDGNode]:
        return [n for n in self._nodes.values() if n.type == NodeType.EXPERIMENT]

    @property
    def findings(self) -> List[RDGNode]:
        return [n for n in self._nodes.values() if n.type == NodeType.FINDING]

    @property
    def claims(self) -> List[RDGNode]:
        return [n for n in self._nodes.values() if n.type == NodeType.CLAIM]

    def children_of(self, node_id: str) -> List[RDGNode]:
        successors = list(self._graph.successors(node_id))
        return [self._nodes[nid] for nid in successors if nid in self._nodes]

    def parents_of(self, node_id: str) -> List[RDGNode]:
        predecessors = list(self._graph.predecessors(node_id))
        return [self._nodes[nid] for nid in predecessors if nid in self._nodes]

    def get_evidence_chain(self, node_id: str) -> List[RDGEdge]:
        """Return all edges on the path(s) leading TO node_id (ancestors)."""
        edges = []
        for ancestor in nx.ancestors(self._graph, node_id):
            for u, v, data in self._graph.out_edges(ancestor, data=True):
                if v in {node_id} | nx.ancestors(self._graph, node_id):
                    eid = data.get("key")
                    if eid and eid in self._edges:
                        edges.append(self._edges[eid])
        return edges

    def next_hypotheses(self, node: RDGNode) -> List[RDGNode]:
        """Hypotheses reachable from node (for policy lookahead)."""
        result = []
        for nid in nx.descendants(self._graph, node.id):
            n = self._nodes.get(nid)
            if n and n.type == NodeType.HYPOTHESIS:
                result.append(n)
        return result

    def find_broken_chains(self) -> List[Tuple[RDGEdge, str]]:
        return find_broken_chains(self._nodes, list(self._edges.values()))

    def merge_equivalent_nodes(self) -> int:
        """
        Merge nodes that have identical content (deduplication).
        Returns count of merges performed.
        """
        merged = 0
        seen: Dict[str, str] = {}  # content → first node id
        to_remove = []
        for nid, node in list(self._nodes.items()):
            key = f"{node.type.value}:{node.content.strip().lower()}"
            if key in seen:
                # redirect edges from duplicate to canonical
                canonical = seen[key]
                for u, v, data in list(self._graph.in_edges(nid, data=True)):
                    self._graph.add_edge(u, canonical, **data)
                for u, v, data in list(self._graph.out_edges(nid, data=True)):
                    self._graph.add_edge(canonical, v, **data)
                to_remove.append(nid)
                merged += 1
            else:
                seen[key] = nid
        for nid in to_remove:
            self.remove_node(nid)
        return merged

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        total = len(self._nodes)
        by_type: Dict[str, int] = {}
        for n in self._nodes.values():
            by_type[n.type.value] = by_type.get(n.type.value, 0) + 1
        return {
            "total_nodes": total,
            "total_edges": len(self._edges),
            "by_type": by_type,
            "graph_id": self.graph_id,
        }

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
        }

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "ResearchDevelopmentGraph":
        data = json.loads(Path(path).read_text())
        rdg = cls(graph_id=data.get("graph_id", "main"))
        for nd in data.get("nodes", []):
            rdg.add_node(RDGNode.from_dict(nd), validate=False)
        for ed in data.get("edges", []):
            rdg.add_edge(RDGEdge.from_dict(ed), validate_semantics=False)
        return rdg

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[RDGNode]:
        return iter(self._nodes.values())

    def __repr__(self) -> str:
        s = self.stats()
        return f"RDG(nodes={s['total_nodes']}, edges={s['total_edges']})"

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from researchforge.vrdeg.graph import VRDEG
from researchforge.vrdeg.node import GraphNode, NodeType
from researchforge.vrdeg.edge import GraphEdge, RelationType


@dataclass(frozen=True)
class TrajectoryRecord:
    nodes: List[GraphNode]
    edges: List[GraphEdge]


@dataclass(frozen=True)
class LineageRecord:
    spec: Optional[GraphNode]
    nodes: List[GraphNode]
    edges: List[GraphEdge]


@dataclass(frozen=True)
class FailureHistoryRecord:
    failure: Optional[GraphNode]
    related_runs: List[GraphNode]
    edges: List[GraphEdge]


@dataclass(frozen=True)
class StateHistoryRecord:
    state: Optional[GraphNode]
    linked_states: List[GraphNode]
    edges: List[GraphEdge]


@dataclass(frozen=True)
class EvidenceRecord:
    node: Optional[GraphNode]
    evidences: List[GraphNode]
    edges: List[GraphEdge]


def _collect_edges_for_nodes(g: VRDEG, node_ids: List[str]) -> List[GraphEdge]:
    s = set(node_ids)
    edges = [e for e in g._edges.values() if e.source_id in s or e.target_id in s]
    return sorted(edges, key=lambda e: e.id)


def _collect_nodes(g: VRDEG, node_ids: List[str]) -> List[GraphNode]:
    nodes = [g.get_node(nid) for nid in node_ids if g.get_node(nid) is not None]
    return sorted(nodes, key=lambda n: n.id)


def get_research_trajectory(g: VRDEG, start_node_id: str) -> TrajectoryRecord:
    """Return deterministic listing of nodes and edges related to start_node_id.

    This is a read-only deterministic traversal: nodes and edges are returned
    sorted by id. Nothing is inferred beyond explicit graph connections.
    If the start node is missing, `nodes` will be empty and `edges` empty.
    """
    start = g.get_node(start_node_id)
    if start is None:
        return TrajectoryRecord(nodes=[], edges=[])
    # collect predecessors and successors deterministically
    pred_ids = [n.id for n in g.predecessors(start_node_id)]
    succ_ids = [n.id for n in g.successors(start_node_id)]
    node_ids = sorted(set([start_node_id] + pred_ids + succ_ids))
    # put start node first, then deterministic ordering for others
    other_ids = [nid for nid in node_ids if nid != start_node_id]
    nodes = [g.get_node(start_node_id)] + _collect_nodes(g, other_ids)
    edges = _collect_edges_for_nodes(g, node_ids)
    return TrajectoryRecord(nodes=nodes, edges=edges)


def get_experiment_lineage(g: VRDEG, spec_id: str) -> LineageRecord:
    spec = g.get_node(spec_id)
    if spec is None:
        return LineageRecord(spec=None, nodes=[], edges=[])
    # collect directly connected nodes and two-hop successors
    direct = g.successors(spec_id) + g.predecessors(spec_id)
    two_hop = []
    for n in direct:
        two_hop.extend(g.successors(n.id))
        two_hop.extend(g.predecessors(n.id))
    node_ids = sorted({spec_id} | {n.id for n in direct} | {n.id for n in two_hop})
    other_ids = [nid for nid in node_ids if nid != spec_id]
    nodes = [spec] + _collect_nodes(g, other_ids)
    edges = _collect_edges_for_nodes(g, node_ids)
    return LineageRecord(spec=spec, nodes=nodes, edges=edges)


def get_failure_history(g: VRDEG, failure_id: str) -> FailureHistoryRecord:
    f = g.get_node(failure_id)
    if f is None:
        return FailureHistoryRecord(failure=None, related_runs=[], edges=[])
    # runs that target or source to this failure
    related = g.predecessors(failure_id) + g.successors(failure_id)
    node_ids = sorted({failure_id} | {n.id for n in related})
    other_ids = [nid for nid in node_ids if nid != failure_id]
    nodes = [f] + _collect_nodes(g, other_ids)
    edges = _collect_edges_for_nodes(g, node_ids)
    # return only runs among related nodes
    runs = [n for n in nodes if n.node_type == NodeType.EXPERIMENT_RUN.value]
    return FailureHistoryRecord(failure=f, related_runs=runs, edges=edges)


def get_state_history(g: VRDEG, state_id: str) -> StateHistoryRecord:
    s = g.get_node(state_id)
    if s is None:
        return StateHistoryRecord(state=None, linked_states=[], edges=[])
    # linked states via NEXT_STATE or PRECEDES
    linked = []
    for e in g._edges.values():
        if e.relation in (RelationType.NEXT_STATE.value, RelationType.PRECEDES.value):
            if e.source_id == state_id:
                linked.append(g.get_node(e.target_id))
            if e.target_id == state_id:
                linked.append(g.get_node(e.source_id))
    linked_ids = sorted({n.id for n in linked if n is not None})
    nodes = [s] + _collect_nodes(g, linked_ids)
    edges = _collect_edges_for_nodes(g, [state_id] + linked_ids)
    return StateHistoryRecord(state=s, linked_states=nodes[1:], edges=edges)


def get_related_evidence(g: VRDEG, node_id: str) -> EvidenceRecord:
    n = g.get_node(node_id)
    if n is None:
        return EvidenceRecord(node=None, evidences=[], edges=[])
    evid_nodes = []
    for e in g._edges.values():
        if e.source_id == node_id or e.target_id == node_id:
            # check the other end
            other_id = e.target_id if e.source_id == node_id else e.source_id
            other = g.get_node(other_id)
            if other and other.node_type == NodeType.EVIDENCE.value:
                evid_nodes.append(other)
    evid_ids = sorted({n.id for n in evid_nodes})
    evid_nodes_sorted = _collect_nodes(g, evid_ids)
    edges = _collect_edges_for_nodes(g, [node_id] + evid_ids)
    return EvidenceRecord(node=n, evidences=evid_nodes_sorted, edges=edges)

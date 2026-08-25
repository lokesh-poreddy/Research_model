"""
RDG semantic consistency checker.
Validates that edges obey the allowed (source_type, target_type) semantics.
"""
from __future__ import annotations

from typing import List, Tuple

from rdg.edges import EDGE_SEMANTICS, RDGEdge
from rdg.nodes import RDGNode


class ConsistencyError(Exception):
    """Raised when an RDG edge violates semantic constraints."""


def check_edge(edge: RDGEdge, nodes: dict) -> None:
    """Raise ConsistencyError if the edge violates type semantics."""
    src = nodes.get(edge.from_node)
    tgt = nodes.get(edge.to_node)
    if src is None:
        raise ConsistencyError(f"Source node '{edge.from_node}' not found in RDG.")
    if tgt is None:
        raise ConsistencyError(f"Target node '{edge.to_node}' not found in RDG.")

    allowed = EDGE_SEMANTICS.get(edge.relation)
    if allowed is None:
        return  # unknown relation types are allowed (permissive)

    allowed_src, allowed_tgt = allowed
    if src.type != allowed_src:
        raise ConsistencyError(
            f"Edge '{edge.relation.value}' expects source type '{allowed_src.value}', "
            f"got '{src.type.value}' (node {edge.from_node})."
        )
    if tgt.type != allowed_tgt:
        raise ConsistencyError(
            f"Edge '{edge.relation.value}' expects target type '{allowed_tgt.value}', "
            f"got '{tgt.type.value}' (node {edge.to_node})."
        )


def find_broken_chains(nodes: dict, edges: List[RDGEdge]) -> List[Tuple[RDGEdge, str]]:
    """Return a list of (edge, reason) for every violated constraint."""
    broken = []
    for edge in edges:
        try:
            check_edge(edge, nodes)
        except ConsistencyError as exc:
            broken.append((edge, str(exc)))
    return broken


def semantically_aligned(node_a: RDGNode, node_b: RDGNode) -> bool:
    """
    Heuristic check: two nodes are aligned if their content has significant
    word overlap (jaccard similarity > 0.15).
    """
    words_a = set(node_a.content.lower().split())
    words_b = set(node_b.content.lower().split())
    if not words_a or not words_b:
        return False
    jaccard = len(words_a & words_b) / len(words_a | words_b)
    return jaccard > 0.15

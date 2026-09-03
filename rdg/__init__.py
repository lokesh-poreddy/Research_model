LEGACY_STATUS = {
    "canonical": False,
    "replacement": "researchforge.rdg",
    "deprecated_since": "RF-1.0.0-alpha.2.1",
    "removal_target": None,  # preserved as historical/compatibility evidence
    "cross_imports_allowed": False,  # researchforge/ must never import from here
}

from rdg.graph import ResearchDevelopmentGraph
from rdg.nodes import NodeStatus, NodeType, RDGNode
from rdg.edges import EdgeRelation, RDGEdge

__all__ = [
    "LEGACY_STATUS",
    "ResearchDevelopmentGraph",
    "RDGNode",
    "RDGEdge",
    "NodeType",
    "NodeStatus",
    "EdgeRelation",
]

"""
RDG Edge definitions with semantic type enforcement.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from rdg.nodes import NodeType


class EdgeRelation(str, Enum):
    IDENTIFIES = "identifies"          # Problem → Gap
    MOTIVATES = "motivates"            # Gap → Hypothesis
    TESTED_BY = "tested-by"           # Hypothesis → Experiment
    PRODUCES = "produces"              # Experiment → Finding
    SUPPORTS = "supports"              # Finding → Claim
    DERIVES_FROM = "derives-from"     # ModelGenome → ModelGenome
    SAVED_AS = "saved-as"             # Hypothesis → MemoryRecord
    PROPOSES = "proposes"              # Hypothesis → ModelGenome
    EVALUATED_AS = "evaluated-as"     # Experiment → Finding
    INFORMS = "informs"               # Finding → Strategy
    UPDATES = "updates"               # Insight → Hypothesis


# Allowed (source_type, target_type) pairs per relation
EDGE_SEMANTICS: Dict[EdgeRelation, tuple] = {
    EdgeRelation.IDENTIFIES:   (NodeType.PROBLEM, NodeType.GAP),
    EdgeRelation.MOTIVATES:    (NodeType.GAP, NodeType.HYPOTHESIS),
    EdgeRelation.TESTED_BY:    (NodeType.HYPOTHESIS, NodeType.EXPERIMENT),
    EdgeRelation.PRODUCES:     (NodeType.EXPERIMENT, NodeType.FINDING),
    EdgeRelation.SUPPORTS:     (NodeType.FINDING, NodeType.CLAIM),
    EdgeRelation.DERIVES_FROM: (NodeType.MODEL_GENOME, NodeType.MODEL_GENOME),
    EdgeRelation.SAVED_AS:     (NodeType.HYPOTHESIS, NodeType.MEMORY_RECORD),
    EdgeRelation.PROPOSES:     (NodeType.HYPOTHESIS, NodeType.MODEL_GENOME),
    EdgeRelation.EVALUATED_AS: (NodeType.EXPERIMENT, NodeType.FINDING),
    EdgeRelation.INFORMS:      (NodeType.FINDING, NodeType.STRATEGY),
    EdgeRelation.UPDATES:      (NodeType.STRATEGY, NodeType.HYPOTHESIS),
}


@dataclass
class RDGEdge:
    """A typed, directed edge between two RDG nodes."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_node: str = ""
    to_node: str = ""
    relation: EdgeRelation = EdgeRelation.TESTED_BY
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from": self.from_node,
            "to": self.to_node,
            "relation": self.relation.value,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RDGEdge":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            from_node=d["from"],
            to_node=d["to"],
            relation=EdgeRelation(d["relation"]),
            timestamp=datetime.fromisoformat(d.get("timestamp", datetime.now(timezone.utc).isoformat())),
            confidence=d.get("confidence", 1.0),
            properties=d.get("properties", {}),
        )

    def __repr__(self) -> str:
        return f"RDGEdge({self.from_node[:8]} --{self.relation.value}--> {self.to_node[:8]})"

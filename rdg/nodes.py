"""
RDG Node definitions.
Each node type carries typed fields and enforces semantic constraints.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeType(str, Enum):
    PROBLEM = "Problem"
    GAP = "Gap"
    HYPOTHESIS = "Hypothesis"
    EXPERIMENT = "Experiment"
    FINDING = "Finding"
    CLAIM = "Claim"
    MODEL_GENOME = "ModelGenome"
    STRATEGY = "Strategy"
    MEMORY_RECORD = "MemoryRecord"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class RDGNode:
    """Base node in the Research Development Graph."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: NodeType = NodeType.HYPOTHESIS
    content: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: NodeStatus = NodeStatus.PENDING
    attributes: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)   # IDs of related nodes
    times_tried: int = 0
    best_metric: Optional[float] = None
    failure_count: int = 0

    # ── factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def problem(cls, content: str, **kw) -> "RDGNode":
        return cls(type=NodeType.PROBLEM, content=content, **kw)

    @classmethod
    def gap(cls, content: str, **kw) -> "RDGNode":
        return cls(type=NodeType.GAP, content=content, **kw)

    @classmethod
    def hypothesis(cls, content: str, **kw) -> "RDGNode":
        return cls(type=NodeType.HYPOTHESIS, content=content, **kw)

    @classmethod
    def experiment(cls, content: str, code: str = "", **kw) -> "RDGNode":
        attrs = kw.pop("attributes", {})
        attrs["code"] = code
        return cls(type=NodeType.EXPERIMENT, content=content, attributes=attrs, **kw)

    @classmethod
    def finding(cls, content: str, score: float = 0.0, **kw) -> "RDGNode":
        attrs = kw.pop("attributes", {})
        attrs["score"] = score
        return cls(type=NodeType.FINDING, content=content, attributes=attrs, **kw)

    @classmethod
    def claim(cls, content: str, supported_by: Optional[List[str]] = None, **kw) -> "RDGNode":
        attrs = kw.pop("attributes", {})
        attrs["supported_by"] = supported_by or []
        return cls(type=NodeType.CLAIM, content=content, attributes=attrs, **kw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "attributes": self.attributes,
            "links": self.links,
            "times_tried": self.times_tried,
            "best_metric": self.best_metric,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RDGNode":
        node = cls(
            id=d["id"],
            type=NodeType(d["type"]),
            content=d["content"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            status=NodeStatus(d.get("status", "pending")),
            attributes=d.get("attributes", {}),
            links=d.get("links", []),
            times_tried=d.get("times_tried", 0),
            best_metric=d.get("best_metric"),
            failure_count=d.get("failure_count", 0),
        )
        return node

    def __repr__(self) -> str:
        return f"RDGNode({self.type.value}:{self.id[:8]}, status={self.status.value})"

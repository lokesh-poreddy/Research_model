"""researchforge/memory/record.py — MemoryRecord domain object.

RF-1.0.0-alpha.2.1: Class B contextual record with versioned mutable envelope.
Immutable core captures historical truth; mutable envelope tracks dynamic
context (usage, consolidation, retrieval count).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..genome.schema import validate_genome

MEMORY_RECORD_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "MemoryRecord",
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "text_summary", "embedding", "context", "outcome", "strategy"],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string"},
        "text_summary": {"type": "string"},
        "embedding": {"type": "array", "items": {"type": "number"}},
        "context": {"type": "object"},
        "outcome": {"type": "object"},
        "strategy": {"type": "string"},
        "created_at": {"type": "number"},
        "tier": {"type": "string", "enum": ["short_term", "long_term"]},
        "consolidation_passes_survived": {"type": "integer", "minimum": 0},
        "archived": {"type": "boolean"},
        "retrieval_count": {"type": "integer", "minimum": 0},
        "negative_transfer_count": {"type": "integer", "minimum": 0},
        "confidence": {"type": "number"},
        "researchforge_version": {"type": "string"},
    },
}


@dataclass
class MemoryRecord:
    """Class B contextual record with versioned mutable envelope."""
    id: str
    text_summary: str
    embedding: List[float]
    context: Dict[str, Any]
    outcome: Dict[str, Any]
    strategy: str
    created_at: float = field(default_factory=time.time)
    tier: str = "short_term"              # "short_term" (working) | "long_term" (consolidated)
    consolidation_passes_survived: int = 0
    archived: bool = False
    retrieval_count: int = 0
    negative_transfer_count: int = 0
    confidence: float = 1.0
    schema_version: str = "1.0"
    researchforge_version: str = "RF-1.0.0-alpha.2.1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "schema_version": self.schema_version,
            "text_summary": self.text_summary,
            "embedding": list(self.embedding),
            "context": dict(self.context),
            "outcome": dict(self.outcome),
            "strategy": self.strategy,
            "created_at": self.created_at,
            "tier": self.tier,
            "consolidation_passes_survived": self.consolidation_passes_survived,
            "archived": self.archived,
            "retrieval_count": self.retrieval_count,
            "negative_transfer_count": self.negative_transfer_count,
            "confidence": self.confidence,
            "researchforge_version": self.researchforge_version,
        }

    def canonical_dict(self) -> Dict[str, Any]:
        """Immutable core fingerprinting dict — excludes mutable dynamic state and timestamps."""
        return {
            "id": self.id,
            "schema_version": self.schema_version,
            "text_summary": self.text_summary,
            "embedding": [round(x, 6) for x in self.embedding],
            "context": self.context,
            "outcome": self.outcome,
            "strategy": self.strategy,
            "researchforge_version": self.researchforge_version,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        validate_genome(self.to_dict(), MEMORY_RECORD_SCHEMA)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryRecord":
        return cls(
            id=d["id"],
            schema_version=d.get("schema_version", "1.0"),
            text_summary=d["text_summary"],
            embedding=list(d.get("embedding", [])),
            context=dict(d.get("context", {})),
            outcome=dict(d.get("outcome", {})),
            strategy=d["strategy"],
            created_at=float(d.get("created_at", time.time())),
            tier=d.get("tier", "short_term"),
            consolidation_passes_survived=int(d.get("consolidation_passes_survived", 0)),
            archived=bool(d.get("archived", False)),
            retrieval_count=int(d.get("retrieval_count", 0)),
            negative_transfer_count=int(d.get("negative_transfer_count", 0)),
            confidence=float(d.get("confidence", 1.0)),
            researchforge_version=d.get("researchforge_version", "RF-1.0.0-alpha.2.1"),
        )

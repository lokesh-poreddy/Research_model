"""researchforge/research/hypothesis.py — Hypothesis domain object.

RF-1.0.0-alpha.2.1: Class C reserved architecture position.
Schema, fingerprinting, and serialization contract declared in alpha.2.1;
full graph operations implemented in alpha.3 alongside VRDEG.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..genome.schema import validate_genome

HYPOTHESIS_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Hypothesis",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "hypothesis_id", "schema_version", "statement", "predicted_outcome", "status"
    ],
    "properties": {
        "hypothesis_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string"},
        "problem_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "question_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "statement": {"type": "string"},
        "predicted_outcome": {"type": "string"},
        "target_tmg_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "status": {
            "type": "string",
            "enum": ["active", "confirmed", "refuted", "inconclusive"],
        },
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
    },
}


@dataclass
class Hypothesis:
    """Class C: Schema-declared position for scientific hypothesis nodes."""
    hypothesis_id: str
    statement: str
    predicted_outcome: str
    problem_id: Optional[str] = None
    question_id: Optional[str] = None
    target_tmg_id: Optional[str] = None
    status: str = "active"  # "active" | "confirmed" | "refuted" | "inconclusive"
    schema_version: str = "1.0"
    researchforge_version: str = "RF-1.0.0-alpha.2.1"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "question_id": self.question_id,
            "statement": self.statement,
            "predicted_outcome": self.predicted_outcome,
            "target_tmg_id": self.target_tmg_id,
            "status": self.status,
            "researchforge_version": self.researchforge_version,
            "created_at": self.created_at,
        }

    def canonical_dict(self) -> Dict[str, Any]:
        d = self.to_dict()
        d.pop("created_at", None)
        return d

    def fingerprint(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        validate_genome(self.to_dict(), HYPOTHESIS_SCHEMA)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Hypothesis":
        return cls(
            hypothesis_id=d["hypothesis_id"],
            schema_version=d.get("schema_version", "1.0"),
            problem_id=d.get("problem_id"),
            question_id=d.get("question_id"),
            statement=d["statement"],
            predicted_outcome=d["predicted_outcome"],
            target_tmg_id=d.get("target_tmg_id"),
            status=d.get("status", "active"),
            researchforge_version=d.get("researchforge_version", "RF-1.0.0-alpha.2.1"),
            created_at=float(d.get("created_at", time.time())),
        )

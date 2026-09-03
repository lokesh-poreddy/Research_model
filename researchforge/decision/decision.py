"""researchforge/decision/decision.py — ResearchDecision domain object.

RF-1.0.0-alpha.2.1: Canonical representation of a decision made by the research system.
Class A: Identity-bearing immutable research artifact.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..genome.schema import validate_genome

DECISION_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ResearchDecision",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision_id", "schema_version", "decision_type", "context_state_id",
        "rationale", "chosen_option", "candidate_options", "policy_confidence"
    ],
    "properties": {
        "decision_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string"},
        "decision_type": {"type": "string"},
        "context_state_id": {"type": "string"},
        "rationale": {"type": "string"},
        "chosen_option": {"type": "string"},
        "candidate_options": {
            "type": "array",
            "items": {"type": "string"}
        },
        "policy_confidence": {"type": "number"},
        "metadata": {"type": "object"},
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
    },
}


@dataclass
class ResearchDecision:
    """Explicit, fingerprinted record of a decision made by RSG/policy."""
    decision_id: str
    decision_type: str
    context_state_id: str
    rationale: str
    chosen_option: str
    candidate_options: List[str]
    policy_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    researchforge_version: str = "RF-1.0.0-alpha.2.1"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "schema_version": self.schema_version,
            "decision_type": self.decision_type,
            "context_state_id": self.context_state_id,
            "rationale": self.rationale,
            "chosen_option": self.chosen_option,
            "candidate_options": list(self.candidate_options),
            "policy_confidence": self.policy_confidence,
            "metadata": dict(self.metadata),
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
        validate_genome(self.to_dict(), DECISION_SCHEMA)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResearchDecision":
        return cls(
            decision_id=d["decision_id"],
            schema_version=d.get("schema_version", "1.0"),
            decision_type=d["decision_type"],
            context_state_id=d["context_state_id"],
            rationale=d["rationale"],
            chosen_option=d["chosen_option"],
            candidate_options=list(d.get("candidate_options", [])),
            policy_confidence=float(d["policy_confidence"]),
            metadata=dict(d.get("metadata", {})),
            researchforge_version=d.get("researchforge_version", "RF-1.0.0-alpha.2.1"),
            created_at=float(d.get("created_at", time.time())),
        )

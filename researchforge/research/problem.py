"""researchforge/research/problem.py — ResearchProblem domain object.

RF-1.0.0-alpha.2.1: Class C reserved architecture position.
Schema, fingerprinting, and serialization contract declared in alpha.2.1;
full graph operations implemented in alpha.3 alongside VRDEG.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..genome.schema import validate_genome

RESEARCH_PROBLEM_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ResearchProblem",
    "type": "object",
    "additionalProperties": False,
    "required": ["problem_id", "schema_version", "title", "description", "domain"],
    "properties": {
        "problem_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "domain": {"type": "string"},
        "success_criteria": {
            "type": "array",
            "items": {"type": "string"}
        },
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
    },
}


@dataclass
class ResearchProblem:
    """Class C: Schema-declared position for top-level research problem nodes."""
    problem_id: str
    title: str
    description: str
    domain: str
    success_criteria: List[str] = field(default_factory=list)
    schema_version: str = "1.0"
    researchforge_version: str = "RF-1.0.0-alpha.2.1"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "schema_version": self.schema_version,
            "title": self.title,
            "description": self.description,
            "domain": self.domain,
            "success_criteria": list(self.success_criteria),
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
        validate_genome(self.to_dict(), RESEARCH_PROBLEM_SCHEMA)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResearchProblem":
        return cls(
            problem_id=d["problem_id"],
            schema_version=d.get("schema_version", "1.0"),
            title=d["title"],
            description=d["description"],
            domain=d["domain"],
            success_criteria=list(d.get("success_criteria", [])),
            researchforge_version=d.get("researchforge_version", "RF-1.0.0-alpha.2.1"),
            created_at=float(d.get("created_at", time.time())),
        )

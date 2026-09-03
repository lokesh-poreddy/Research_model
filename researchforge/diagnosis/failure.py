"""researchforge/diagnosis/failure.py — Failure domain object.

RF-1.0.0-alpha.2.1: Canonical representation of an experiment or system failure.
Class A: Identity-bearing immutable research artifact.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..genome.schema import validate_genome
from .failure_taxonomy import FailureCategory

FAILURE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Failure",
    "type": "object",
    "additionalProperties": False,
    "required": ["failure_id", "schema_version", "category", "description"],
    "properties": {
        "failure_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string"},
        "category": {"type": "string"},
        "description": {"type": "string"},
        "run_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "tmg_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "error_trace": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
    },
}


@dataclass
class Failure:
    """Canonical immutable record of an experimental or runtime failure."""
    failure_id: str
    category: str
    description: str
    run_id: Optional[str] = None
    tmg_id: Optional[str] = None
    error_trace: Optional[str] = None
    schema_version: str = "1.0"
    researchforge_version: str = "RF-1.0.0-alpha.2.1"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "schema_version": self.schema_version,
            "category": self.category,
            "description": self.description,
            "run_id": self.run_id,
            "tmg_id": self.tmg_id,
            "error_trace": self.error_trace,
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
        validate_genome(self.to_dict(), FAILURE_SCHEMA)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Failure":
        return cls(
            failure_id=d["failure_id"],
            schema_version=d.get("schema_version", "1.0"),
            category=d["category"],
            description=d["description"],
            run_id=d.get("run_id"),
            tmg_id=d.get("tmg_id"),
            error_trace=d.get("error_trace"),
            researchforge_version=d.get("researchforge_version", "RF-1.0.0-alpha.2.1"),
            created_at=float(d.get("created_at", time.time())),
        )

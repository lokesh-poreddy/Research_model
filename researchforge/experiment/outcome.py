"""researchforge/experiment/outcome.py — ExperimentOutcome domain object.

RF-1.0.0-alpha.2.1: Canonical scientific outcome of an experiment.
Class A: Identity-bearing immutable research artifact.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..genome.schema import validate_genome

EXPERIMENT_OUTCOME_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ExperimentOutcome",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "outcome_id", "schema_version", "run_id", "spec_id", "tmg_id",
        "metric", "metric_fn", "svg_verdict", "svg_report_fingerprint",
        "success"
    ],
    "properties": {
        "outcome_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string"},
        "run_id": {"type": "string", "minLength": 1},
        "spec_id": {"type": "string", "minLength": 1},
        "tmg_id": {"type": "string"},
        "rsg_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "metric": {"type": "number"},
        "metric_fn": {"type": "string"},
        "svg_verdict": {"type": "string"},
        "svg_report_fingerprint": {"type": "string"},
        "success": {"type": "boolean"},
        "failure_category": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "failure_fingerprint": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
    },
}


@dataclass
class ExperimentOutcome:
    """The scientific result of an experiment. Class A: immutable."""
    outcome_id: str
    run_id: str
    spec_id: str
    tmg_id: str
    metric: float
    metric_fn: str
    svg_verdict: str
    svg_report_fingerprint: str
    success: bool
    rsg_id: Optional[str] = None
    failure_category: Optional[str] = None
    failure_fingerprint: Optional[str] = None
    schema_version: str = "1.0"
    researchforge_version: str = "RF-1.0.0-alpha.2.1"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "spec_id": self.spec_id,
            "tmg_id": self.tmg_id,
            "rsg_id": self.rsg_id,
            "metric": self.metric,
            "metric_fn": self.metric_fn,
            "svg_verdict": self.svg_verdict,
            "svg_report_fingerprint": self.svg_report_fingerprint,
            "success": self.success,
            "failure_category": self.failure_category,
            "failure_fingerprint": self.failure_fingerprint,
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
        validate_genome(self.to_dict(), EXPERIMENT_OUTCOME_SCHEMA)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentOutcome":
        return cls(
            outcome_id=d["outcome_id"],
            schema_version=d.get("schema_version", "1.0"),
            run_id=d["run_id"],
            spec_id=d["spec_id"],
            tmg_id=d["tmg_id"],
            rsg_id=d.get("rsg_id"),
            metric=float(d["metric"]),
            metric_fn=d["metric_fn"],
            svg_verdict=d["svg_verdict"],
            svg_report_fingerprint=d["svg_report_fingerprint"],
            success=bool(d["success"]),
            failure_category=d.get("failure_category"),
            failure_fingerprint=d.get("failure_fingerprint"),
            researchforge_version=d.get("researchforge_version", "RF-1.0.0-alpha.2.1"),
            created_at=float(d.get("created_at", time.time())),
        )

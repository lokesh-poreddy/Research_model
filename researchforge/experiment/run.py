"""researchforge/experiment/run.py — ExperimentRun domain object.

RF-1.0.0-alpha.2.1: Canonical provenance record of one experiment execution.
Class A: Identity-bearing immutable research artifact.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..genome.schema import validate_genome

EXPERIMENT_RUN_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ExperimentRun",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "run_id", "spec_id", "schema_version", "started_at", "finished_at",
        "wall_time_s", "execution_mode", "runner_version", "exit_status",
        "safety_verdict", "resource_usage"
    ],
    "properties": {
        "run_id": {"type": "string", "minLength": 1},
        "spec_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string"},
        "started_at": {"type": "number"},
        "finished_at": {"type": "number"},
        "wall_time_s": {"type": "number"},
        "execution_mode": {"type": "string"},
        "runner_version": {"type": "string"},
        "adapter_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "exit_status": {"type": "string"},
        "resource_usage": {"type": "object"},
        "safety_verdict": {"type": "string"},
        "svg_verdict": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "outcome_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "stdout_artifact_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "stderr_artifact_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
    },
}


@dataclass
class ExperimentRun:
    """Provenance record of one execution. Not just a connector."""
    run_id: str
    spec_id: str
    started_at: float
    finished_at: float
    wall_time_s: float
    execution_mode: str
    runner_version: str = "RF-1.0.0-alpha.2.1"
    adapter_id: Optional[str] = None
    exit_status: str = "success"  # "success" | "timeout" | "killed" | "error"
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    safety_verdict: str = "pass"   # "pass" | "timeout" | "killed"
    svg_verdict: Optional[str] = None
    outcome_id: Optional[str] = None
    stdout_artifact_id: Optional[str] = None
    stderr_artifact_id: Optional[str] = None
    schema_version: str = "1.0"
    researchforge_version: str = "RF-1.0.0-alpha.2.1"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "spec_id": self.spec_id,
            "schema_version": self.schema_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_time_s": self.wall_time_s,
            "execution_mode": self.execution_mode,
            "runner_version": self.runner_version,
            "adapter_id": self.adapter_id,
            "exit_status": self.exit_status,
            "resource_usage": dict(self.resource_usage),
            "safety_verdict": self.safety_verdict,
            "svg_verdict": self.svg_verdict,
            "outcome_id": self.outcome_id,
            "stdout_artifact_id": self.stdout_artifact_id,
            "stderr_artifact_id": self.stderr_artifact_id,
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
        validate_genome(self.to_dict(), EXPERIMENT_RUN_SCHEMA)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRun":
        return cls(
            run_id=d["run_id"],
            spec_id=d["spec_id"],
            schema_version=d.get("schema_version", "1.0"),
            started_at=float(d["started_at"]),
            finished_at=float(d["finished_at"]),
            wall_time_s=float(d["wall_time_s"]),
            execution_mode=d["execution_mode"],
            runner_version=d.get("runner_version", "RF-1.0.0-alpha.2.1"),
            adapter_id=d.get("adapter_id"),
            exit_status=d.get("exit_status", "success"),
            resource_usage=dict(d.get("resource_usage", {})),
            safety_verdict=d.get("safety_verdict", "pass"),
            svg_verdict=d.get("svg_verdict"),
            outcome_id=d.get("outcome_id"),
            stdout_artifact_id=d.get("stdout_artifact_id"),
            stderr_artifact_id=d.get("stderr_artifact_id"),
            researchforge_version=d.get("researchforge_version", "RF-1.0.0-alpha.2.1"),
            created_at=float(d.get("created_at", time.time())),
        )

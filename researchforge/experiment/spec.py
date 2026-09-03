"""researchforge/experiment/spec.py — ExperimentSpec domain object.

RF-1.0.0-alpha.2.1: Canonical reproducible experiment specification.
Class A: Identity-bearing immutable research artifact.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..genome.schema import validate_genome

EXPERIMENT_SPEC_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ExperimentSpec",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "spec_id", "schema_version", "tmg_id", "tmg_fingerprint",
        "dataset_name", "dataset_fingerprint", "data_pipeline_fingerprint",
        "evaluator", "metric_fn", "seed", "execution_mode"
    ],
    "properties": {
        "spec_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string"},
        "tmg_id": {"type": "string"},
        "tmg_fingerprint": {"type": "string"},
        "rsg_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "validity_config_fingerprint": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "dataset_name": {"type": "string"},
        "dataset_fingerprint": {"type": "string"},
        "data_pipeline_fingerprint": {"type": "string"},
        "evaluator": {"type": "string"},
        "n_splits": {"type": "integer", "minimum": 1},
        "metric_fn": {"type": "string"},
        "seed": {"type": "integer"},
        "execution_mode": {"type": "string"},
        "code_revision": {"type": "string"},
        "environment_fingerprint": {"type": "string"},
        "dependency_lock_fingerprint": {"type": "string"},
        "adapter_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
    },
}


@dataclass
class ExperimentSpec:
    """What was INTENDED to be executed. Enough to reproduce the experiment."""
    spec_id: str
    tmg_id: str
    tmg_fingerprint: str
    dataset_name: str
    dataset_fingerprint: str
    data_pipeline_fingerprint: str
    evaluator: str
    metric_fn: str
    seed: int
    execution_mode: str = "trusted_offline"
    rsg_id: Optional[str] = None
    validity_config_fingerprint: Optional[str] = None
    n_splits: int = 5
    code_revision: str = "unknown"
    environment_fingerprint: str = ""
    dependency_lock_fingerprint: str = ""
    adapter_id: Optional[str] = None
    schema_version: str = "1.0"
    researchforge_version: str = "RF-1.0.0-alpha.2.1"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "schema_version": self.schema_version,
            "tmg_id": self.tmg_id,
            "tmg_fingerprint": self.tmg_fingerprint,
            "rsg_id": self.rsg_id,
            "validity_config_fingerprint": self.validity_config_fingerprint,
            "dataset_name": self.dataset_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "data_pipeline_fingerprint": self.data_pipeline_fingerprint,
            "evaluator": self.evaluator,
            "n_splits": self.n_splits,
            "metric_fn": self.metric_fn,
            "seed": self.seed,
            "execution_mode": self.execution_mode,
            "code_revision": self.code_revision,
            "environment_fingerprint": self.environment_fingerprint,
            "dependency_lock_fingerprint": self.dependency_lock_fingerprint,
            "adapter_id": self.adapter_id,
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
        validate_genome(self.to_dict(), EXPERIMENT_SPEC_SCHEMA)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentSpec":
        return cls(
            spec_id=d["spec_id"],
            schema_version=d.get("schema_version", "1.0"),
            tmg_id=d["tmg_id"],
            tmg_fingerprint=d["tmg_fingerprint"],
            rsg_id=d.get("rsg_id"),
            validity_config_fingerprint=d.get("validity_config_fingerprint"),
            dataset_name=d["dataset_name"],
            dataset_fingerprint=d["dataset_fingerprint"],
            data_pipeline_fingerprint=d["data_pipeline_fingerprint"],
            evaluator=d["evaluator"],
            n_splits=int(d.get("n_splits", 5)),
            metric_fn=d["metric_fn"],
            seed=int(d["seed"]),
            execution_mode=d.get("execution_mode", "trusted_offline"),
            code_revision=d.get("code_revision", "unknown"),
            environment_fingerprint=d.get("environment_fingerprint", ""),
            dependency_lock_fingerprint=d.get("dependency_lock_fingerprint", ""),
            adapter_id=d.get("adapter_id"),
            researchforge_version=d.get("researchforge_version", "RF-1.0.0-alpha.2.1"),
            created_at=float(d.get("created_at", time.time())),
        )

    @classmethod
    def create(
        cls,
        tmg_id: str,
        tmg_fingerprint: str,
        dataset_name: str,
        dataset_fingerprint: str,
        data_pipeline_fingerprint: str,
        evaluator: str,
        metric_fn: str,
        seed: int,
        execution_mode: str = "trusted_offline",
        rsg_id: Optional[str] = None,
        validity_config_fingerprint: Optional[str] = None,
        n_splits: int = 5,
        adapter_id: Optional[str] = None,
    ) -> "ExperimentSpec":
        spec_hash_source = f"{tmg_fingerprint}:{rsg_id}:{validity_config_fingerprint}:{dataset_fingerprint}:{seed}"
        spec_id = f"spec_{hashlib.sha256(spec_hash_source.encode()).hexdigest()[:12]}"
        return cls(
            spec_id=spec_id,
            tmg_id=tmg_id,
            tmg_fingerprint=tmg_fingerprint,
            rsg_id=rsg_id,
            validity_config_fingerprint=validity_config_fingerprint,
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            data_pipeline_fingerprint=data_pipeline_fingerprint,
            evaluator=evaluator,
            n_splits=n_splits,
            metric_fn=metric_fn,
            seed=seed,
            execution_mode=execution_mode,
            adapter_id=adapter_id,
        )

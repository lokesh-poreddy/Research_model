from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List
from .base import DomainObject
from .provenance import Provenance
from .validity import Validity


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"


class OutcomeNature(str, Enum):
    IMPROVEMENT = "IMPROVEMENT"
    DEGRADATION = "DEGRADATION"
    NEUTRAL = "NEUTRAL"
    FAILURE = "FAILURE"
    INCONCLUSIVE = "INCONCLUSIVE"
    INFORMATION_GAIN = "INFORMATION_GAIN"


@dataclass(frozen=True)
class ExperimentSpec(DomainObject):
    research_problem_id: str | None = None
    research_question_id: str | None = None
    hypothesis_id: str | None = None
    decision_id: str | None = None
    target_model_genome_id: str | None = None
    # Legacy aliases for backward compatibility
    tmg_id: str | None = None
    research_system_genome_id: str | None = None
    rsg_id: str | None = None
    dataset_ref: str | None = None
    dataset_id: str | None = None
    preprocessing: Dict[str, Any] | None = None
    intervention_description: str | None = None
    baseline_config: Dict[str, Any] | None = None
    metrics: List[str] | None = None
    seeds: List[int] | None = None
    resource_requirements: Dict[str, Any] | None = None
    expected_outputs: List[str] | None = None
    validity_requirements: Dict[str, Any] | None = None
    provenance: Provenance | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ExperimentSpec":
        prov = obj.get("provenance")
        if isinstance(prov, dict):
            prov = Provenance.from_dict(prov)
        # migrate legacy aliases if present
        if "tmg_id" in obj and not obj.get("target_model_genome_id"):
            obj = dict(obj)
            obj["target_model_genome_id"] = obj.get("tmg_id")
        if "rsg_id" in obj and not obj.get("research_system_genome_id"):
            obj = dict(obj)
            obj["research_system_genome_id"] = obj.get("rsg_id")
        if "dataset_id" in obj and not obj.get("dataset_ref"):
            obj = dict(obj)
            obj["dataset_ref"] = obj.get("dataset_id")
        return cls(**{**obj, "provenance": prov})


@dataclass(frozen=True)
class ExperimentRun(DomainObject):
    experiment_spec_id: str | None = None
    # legacy alias
    spec_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    environment_fingerprint: str | None = None
    code_revision: str | None = None
    dataset_fingerprint: str | None = None
    model_fingerprint: str | None = None
    seed: int | None = None
    resource_usage: Dict[str, Any] | None = None
    produced_artifacts: List[str] | None = None
    failure_info: Dict[str, Any] | None = None
    provenance: Provenance | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ExperimentRun":
        prov = obj.get("provenance")
        if isinstance(prov, dict):
            prov = Provenance.from_dict(prov)
        status = ExecutionStatus(obj.get("status")) if obj.get("status") else ExecutionStatus.PENDING
        # handle legacy spec_id
        if obj.get("spec_id") and not obj.get("experiment_spec_id"):
            obj = dict(obj)
            obj["experiment_spec_id"] = obj.get("spec_id")
        return cls(**{**obj, "provenance": prov, "status": status})


@dataclass(frozen=True)
class Outcome(DomainObject):
    run_id: str
    metrics: Dict[str, float] | None = None
    baseline_comparison: Dict[str, Any] | None = None
    confidence: Dict[str, Any] | None = None
    artifact_refs: List[str] | None = None
    validity: Validity | None = None
    nature: OutcomeNature | None = None
    provenance: Provenance | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Outcome":
        prov = obj.get("provenance")
        if isinstance(prov, dict):
            prov = Provenance.from_dict(prov)
        val = obj.get("validity")
        if isinstance(val, dict):
            val = Validity.from_dict(val)
        nature = OutcomeNature(obj.get("nature")) if obj.get("nature") else None
        return cls(**{**obj, "provenance": prov, "validity": val, "nature": nature})

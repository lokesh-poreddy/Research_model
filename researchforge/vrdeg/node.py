from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any
from researchforge.domain.base import DomainObject


class NodeType(str, Enum):
    RESEARCH_PROBLEM = "ResearchProblem"
    RESEARCH_QUESTION = "ResearchQuestion"
    HYPOTHESIS = "Hypothesis"
    DECISION = "Decision"
    RESEARCH_STATE = "ResearchState"
    TARGET_MODEL_GENOME = "TargetModelGenome"
    RESEARCH_SYSTEM_GENOME = "ResearchSystemGenome"
    EXPERIMENT_SPEC = "ExperimentSpec"
    EXPERIMENT_RUN = "ExperimentRun"
    OUTCOME = "Outcome"
    DIAGNOSIS = "Diagnosis"
    FAILURE = "Failure"
    EVIDENCE = "Evidence"
    VALIDITY = "Validity"
    PROVENANCE = "Provenance"


@dataclass(frozen=True)
class GraphNode(DomainObject):
    node_type: str
    payload_ref: str | None = None
    payload_schema: str | None = None
    semantic_fingerprint: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    provenance_id: str | None = None
    parent_version_of: str | None = None
    metadata: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "GraphNode":
        return cls(**obj)

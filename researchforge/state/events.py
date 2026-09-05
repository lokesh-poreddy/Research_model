from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any
from researchforge.domain.base import DomainObject


class EventType(str, Enum):
    RESEARCH_INITIALIZED = "RESEARCH_INITIALIZED"
    QUESTION_SELECTED = "QUESTION_SELECTED"
    HYPOTHESIS_PROPOSED = "HYPOTHESIS_PROPOSED"
    DECISION_MADE = "DECISION_MADE"
    EXPERIMENT_PLANNED = "EXPERIMENT_PLANNED"
    EXPERIMENT_STARTED = "EXPERIMENT_STARTED"
    EXPERIMENT_COMPLETED = "EXPERIMENT_COMPLETED"
    OUTCOME_RECORDED = "OUTCOME_RECORDED"
    VALIDITY_ASSESSED = "VALIDITY_ASSESSED"
    DIAGNOSIS_RECORDED = "DIAGNOSIS_RECORDED"
    FAILURE_RECORDED = "FAILURE_RECORDED"
    EVIDENCE_ADDED = "EVIDENCE_ADDED"
    BRANCH_CREATED = "BRANCH_CREATED"
    RESEARCH_STATE_UPDATED = "RESEARCH_STATE_UPDATED"


@dataclass(frozen=True)
class Event(DomainObject):
    event_type: str
    payload: Dict[str, Any] | None = None
    timestamp: str | None = None
    provenance_id: str | None = None

    @classmethod
    def create(cls, id: str, schema_version: str, event_type: EventType, payload: Dict[str, Any] | None = None, timestamp: str | None = None, provenance_id: str | None = None) -> "Event":
        return cls(id=id, schema_version=schema_version, event_type=event_type.value, payload=payload, timestamp=timestamp, provenance_id=provenance_id)

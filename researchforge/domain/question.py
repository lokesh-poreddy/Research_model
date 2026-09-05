from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum
from .base import DomainObject


class QuestionStatus(str, Enum):
    OPEN = "OPEN"
    ANSWERED = "ANSWERED"
    ABANDONED = "ABANDONED"


@dataclass(frozen=True)
class ResearchQuestion(DomainObject):
    problem_id: str
    question_text: str
    rationale: str | None = None
    status: QuestionStatus = QuestionStatus.OPEN

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ResearchQuestion":
        if "status" in obj:
            obj = dict(obj)
            obj["status"] = QuestionStatus(obj["status"])
        return cls(**obj)

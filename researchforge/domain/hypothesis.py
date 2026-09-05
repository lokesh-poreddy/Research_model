from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any
from .base import DomainObject


@dataclass(frozen=True)
class Hypothesis(DomainObject):
    research_question_id: str
    statement: str
    prediction: str | None = None
    assumptions: Tuple[str, ...] | None = None
    provenance_id: str | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Hypothesis":
        if "assumptions" in obj and obj["assumptions"] is not None:
            obj = dict(obj)
            obj["assumptions"] = tuple(obj["assumptions"])
        return cls(**obj)

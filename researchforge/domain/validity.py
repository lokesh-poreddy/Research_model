from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class ValidityVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    INCONCLUSIVE = "INCONCLUSIVE"
    PROVISIONAL = "PROVISIONAL"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"


@dataclass(frozen=True)
class Validity:
    verdict: ValidityVerdict
    details: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict.value, "details": self.details}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Validity":
        v = ValidityVerdict(data["verdict"]) if data and "verdict" in data else ValidityVerdict.INCONCLUSIVE
        return cls(verdict=v, details=data.get("details") if data else None)

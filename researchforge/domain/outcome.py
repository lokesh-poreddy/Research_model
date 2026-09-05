from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
from .base import DomainObject
from .validity import Validity


@dataclass(frozen=True)
class Outcome(DomainObject):
    run_id: str
    measured_metrics: Dict[str, float] | None = None
    baseline_metrics: Dict[str, float] | None = None
    improvement: Dict[str, float] | None = None
    statistical_summary: Dict[str, Any] | None = None
    success: bool | None = None
    failure_category: str | None = None
    validity: Validity | None = None
    reproducibility: Dict[str, Any] | None = None
    environment_fingerprint: str | None = None
    artifact_refs: list[str] | None = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        if self.validity is not None:
            base["validity"] = self.validity.to_dict()
        return base

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Outcome":
        if "validity" in obj and obj["validity"] is not None:
            obj = dict(obj)
            obj["validity"] = Validity.from_dict(obj["validity"])
        return cls(**obj)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
from .base import DomainObject


@dataclass(frozen=True)
class TargetModelGenome(DomainObject):
    representation: Dict[str, Any]
    generation: int | None = None
    parent_ids: list[str] | None = None
    operator: str | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "TargetModelGenome":
        return cls(**obj)


@dataclass(frozen=True)
class ResearchSystemGenome(DomainObject):
    config: Dict[str, Any]
    version: str | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ResearchSystemGenome":
        return cls(**obj)

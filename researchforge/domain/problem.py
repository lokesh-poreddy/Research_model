from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
from .base import DomainObject


@dataclass(frozen=True)
class ResearchProblem(DomainObject):
    title: str
    description: str | None = None
    created_at: str | None = None
    tags: List[str] | None = None
    metadata: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ResearchProblem":
        return cls(**obj)

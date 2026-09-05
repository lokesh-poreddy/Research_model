from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
from .base import DomainObject


@dataclass(frozen=True)
class Provenance(DomainObject):
    created_by: str
    created_at: str
    parents: List[str] | None = None
    notes: str | None = None
    metadata: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Provenance":
        return cls(**obj)

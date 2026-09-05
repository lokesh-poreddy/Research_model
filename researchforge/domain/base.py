from __future__ import annotations

import dataclasses
import json
import hashlib
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T", bound="DomainObject")


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _as_primitive(obj: Any) -> Any:
    # Convert nested DomainObjects and enums to primitives for deterministic
    # serialization.
    if isinstance(obj, DomainObject):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj):
        return {k: _as_primitive(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_as_primitive(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _as_primitive(v) for k, v in obj.items()}
    return obj


@dataclass(frozen=True)
class DomainObject:
    """Base class for canonical domain contracts.

    Provides:
    - `schema_version`: contract version
    - deterministic `to_dict` / `from_dict`
    - `fingerprint()` using SHA-256 of canonical JSON
    """

    id: str
    schema_version: str

    def to_dict(self) -> Dict[str, Any]:
        raw = dataclasses.asdict(self)
        return _as_primitive(raw)

    @classmethod
    def from_dict(cls: Type[T], obj: Dict[str, Any]) -> T:
        # Default naive constructor: subclasses may override for nested parsing
        return cls(**obj)  # type: ignore

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def fingerprint(self) -> str:
        # Content-addressed fingerprint using schema_version and canonical JSON
        payload = {"schema_version": self.schema_version, "content": self.to_dict()}
        j = _canonical_json(payload)
        return hashlib.sha256(j.encode("utf-8")).hexdigest()

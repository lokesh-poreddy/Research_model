"""researchforge/artifact/artifact.py — Artifact and Provenance domain objects.

RF-1.0.0-alpha.2.1: Identity-bearing immutable research artifacts and provenance tracking.
Class A: Identity-bearing immutable research artifacts.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..genome.schema import validate_genome

ARTIFACT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Artifact",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "artifact_id", "schema_version", "artifact_type", "uri",
        "checksum_sha256", "size_bytes"
    ],
    "properties": {
        "artifact_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string"},
        "artifact_type": {"type": "string"},
        "uri": {"type": "string"},
        "checksum_sha256": {"type": "string"},
        "size_bytes": {"type": "integer", "minimum": 0},
        "provenance_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "mime_type": {"type": "string"},
        "metadata": {"type": "object"},
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
    },
}

PROVENANCE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Provenance",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "provenance_id", "schema_version", "entity_id", "activity", "agent_id"
    ],
    "properties": {
        "provenance_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string"},
        "entity_id": {"type": "string"},
        "activity": {"type": "string"},
        "agent_id": {"type": "string"},
        "input_artifact_ids": {
            "type": "array",
            "items": {"type": "string"}
        },
        "code_revision": {"type": "string"},
        "environment_fingerprint": {"type": "string"},
        "metadata": {"type": "object"},
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
    },
}


@dataclass
class Artifact:
    """Identity-bearing immutable research artifact (file, weights, dataset, report)."""
    artifact_id: str
    artifact_type: str
    uri: str
    checksum_sha256: str
    size_bytes: int
    provenance_id: Optional[str] = None
    mime_type: str = "application/octet-stream"
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    researchforge_version: str = "RF-1.0.0-alpha.2.1"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "uri": self.uri,
            "checksum_sha256": self.checksum_sha256,
            "size_bytes": self.size_bytes,
            "provenance_id": self.provenance_id,
            "mime_type": self.mime_type,
            "metadata": dict(self.metadata),
            "researchforge_version": self.researchforge_version,
            "created_at": self.created_at,
        }

    def canonical_dict(self) -> Dict[str, Any]:
        d = self.to_dict()
        d.pop("created_at", None)
        return d

    def fingerprint(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        validate_genome(self.to_dict(), ARTIFACT_SCHEMA)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Artifact":
        return cls(
            artifact_id=d["artifact_id"],
            schema_version=d.get("schema_version", "1.0"),
            artifact_type=d["artifact_type"],
            uri=d["uri"],
            checksum_sha256=d["checksum_sha256"],
            size_bytes=int(d["size_bytes"]),
            provenance_id=d.get("provenance_id"),
            mime_type=d.get("mime_type", "application/octet-stream"),
            metadata=dict(d.get("metadata", {})),
            researchforge_version=d.get("researchforge_version", "RF-1.0.0-alpha.2.1"),
            created_at=float(d.get("created_at", time.time())),
        )


@dataclass
class Provenance:
    """Provenance trail capturing activity, inputs, and environment."""
    provenance_id: str
    entity_id: str
    activity: str
    agent_id: str
    input_artifact_ids: List[str] = field(default_factory=list)
    code_revision: str = "unknown"
    environment_fingerprint: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    researchforge_version: str = "RF-1.0.0-alpha.2.1"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provenance_id": self.provenance_id,
            "schema_version": self.schema_version,
            "entity_id": self.entity_id,
            "activity": self.activity,
            "agent_id": self.agent_id,
            "input_artifact_ids": list(self.input_artifact_ids),
            "code_revision": self.code_revision,
            "environment_fingerprint": self.environment_fingerprint,
            "metadata": dict(self.metadata),
            "researchforge_version": self.researchforge_version,
            "created_at": self.created_at,
        }

    def canonical_dict(self) -> Dict[str, Any]:
        d = self.to_dict()
        d.pop("created_at", None)
        return d

    def fingerprint(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        validate_genome(self.to_dict(), PROVENANCE_SCHEMA)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Provenance":
        return cls(
            provenance_id=d["provenance_id"],
            schema_version=d.get("schema_version", "1.0"),
            entity_id=d["entity_id"],
            activity=d["activity"],
            agent_id=d["agent_id"],
            input_artifact_ids=list(d.get("input_artifact_ids", [])),
            code_revision=d.get("code_revision", "unknown"),
            environment_fingerprint=d.get("environment_fingerprint", ""),
            metadata=dict(d.get("metadata", {})),
            researchforge_version=d.get("researchforge_version", "RF-1.0.0-alpha.2.1"),
            created_at=float(d.get("created_at", time.time())),
        )

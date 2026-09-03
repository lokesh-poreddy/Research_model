"""researchforge/artifact/__init__.py — Artifact package.

RF-1.0.0-alpha.2.1: Canonical domain contracts for research artifacts and provenance:
  - Artifact: immutable physical or digital research output
  - Provenance: structured execution history and environment trail
"""
from .artifact import (
    Artifact,
    ARTIFACT_SCHEMA,
    Provenance,
    PROVENANCE_SCHEMA,
)

__all__ = [
    "Artifact",
    "ARTIFACT_SCHEMA",
    "Provenance",
    "PROVENANCE_SCHEMA",
]

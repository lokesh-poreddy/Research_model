"""researchforge/evidence/__init__.py — Evidence package.

RF-1.0.0-alpha.2.1: Canonical domain contracts for scientific evidence:
  - Evidence: adjudicated, citable scientific evidence
  - EvidenceCandidate: unadjudicated search hit
"""
from .evidence import (
    Evidence,
    EVIDENCE_SCHEMA,
    EvidenceCandidate,
    EVIDENCE_CANDIDATE_SCHEMA,
)

__all__ = [
    "Evidence",
    "EVIDENCE_SCHEMA",
    "EvidenceCandidate",
    "EVIDENCE_CANDIDATE_SCHEMA",
]

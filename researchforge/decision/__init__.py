"""researchforge/decision/__init__.py — Decision package.

RF-1.0.0-alpha.2.1: Canonical domain contracts for research decisions:
  - ResearchDecision: explicit, fingerprinted policy decision record
"""
from .decision import ResearchDecision, DECISION_SCHEMA

__all__ = [
    "ResearchDecision",
    "DECISION_SCHEMA",
]

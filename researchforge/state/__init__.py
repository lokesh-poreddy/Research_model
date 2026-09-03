"""researchforge/state/__init__.py — ResearchState package.

RF-1.0.0-alpha.2.1: Class D canonical binding object:
  - ResearchState: complete, fingerprinted state of the research process at generation t
"""
from .research_state import ResearchState, RESEARCH_STATE_SCHEMA

__all__ = [
    "ResearchState",
    "RESEARCH_STATE_SCHEMA",
]

"""researchforge/research/__init__.py — Research domain package.

RF-1.0.0-alpha.2.1: Class C schema-declared domain positions:
  - ResearchProblem: formal research problem statement
  - Hypothesis: testable hypothesis node
"""
from .problem import ResearchProblem, RESEARCH_PROBLEM_SCHEMA
from .hypothesis import Hypothesis, HYPOTHESIS_SCHEMA

__all__ = [
    "ResearchProblem",
    "RESEARCH_PROBLEM_SCHEMA",
    "Hypothesis",
    "HYPOTHESIS_SCHEMA",
]

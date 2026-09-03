"""Failure Taxonomy and diagnosis: classifies an experiment's outcome into a
failure category, feeding the policy learner's failure-avoidance check.

RF-1.0.0-alpha.2.1:
  - Failure: Class A canonical immutable failure artifact
  - FailureCategory, ExperimentResult, diagnose: operational failure taxonomy
"""
from .failure_taxonomy import FailureCategory, ExperimentResult, diagnose
from .failure import Failure, FAILURE_SCHEMA

__all__ = [
    "FailureCategory",
    "ExperimentResult",
    "diagnose",
    "Failure",
    "FAILURE_SCHEMA",
]

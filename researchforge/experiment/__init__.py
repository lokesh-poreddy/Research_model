"""researchforge/experiment/__init__.py — Experiment domain package.

RF-1.0.0-alpha.2.1: Canonical home for experiment domain objects:
  - ExperimentSpec: reproducible specification of an experiment
  - ExperimentRun: execution provenance record
  - ExperimentOutcome: scientific result artifact
  - TrajectoryFingerprint / compute_trajectory_fingerprint: run trajectory hashing
"""
from .spec import ExperimentSpec, EXPERIMENT_SPEC_SCHEMA
from .run import ExperimentRun, EXPERIMENT_RUN_SCHEMA
from .outcome import ExperimentOutcome, EXPERIMENT_OUTCOME_SCHEMA
from .trajectory import TrajectoryFingerprint, compute_trajectory_fingerprint

__all__ = [
    "ExperimentSpec",
    "EXPERIMENT_SPEC_SCHEMA",
    "ExperimentRun",
    "EXPERIMENT_RUN_SCHEMA",
    "ExperimentOutcome",
    "EXPERIMENT_OUTCOME_SCHEMA",
    "TrajectoryFingerprint",
    "compute_trajectory_fingerprint",
]

"""
Failure taxonomy for ResearchForge-ECRM.

Categories:
  Design Failures     — hypothesis misaligned, experiment incorrectly specified
  Implementation Failures — code errors, simulation crashes
  Performance Failures    — convergence issues, catastrophic forgetting
"""
from __future__ import annotations

from enum import Enum
from typing import List


class FailureCategory(str, Enum):
    # Design
    HYPOTHESIS_MISALIGNED = "HypothesisMisaligned"
    EXPERIMENT_MISSPECIFIED = "ExperimentMisspecified"
    UNSUPPORTED_CLAIM = "UnsupportedClaim"

    # Implementation
    CODE_ERROR = "CodeError"
    RUNTIME_CRASH = "RuntimeCrash"
    TIMEOUT = "Timeout"

    # Performance
    OVERFITTING = "Overfitting"
    UNDERFITTING = "Underfitting"
    DIVERGENCE = "Divergence"
    PREMATURE_CONVERGENCE = "PrematureConvergence"
    CATASTROPHIC_FORGETTING = "CatastrophicForgetting"
    LOW_PERFORMANCE = "LowPerformance"

    # Memory
    NEGATIVE_TRANSFER = "NegativeTransfer"
    STALE_MEMORY = "StaleMemory"

    # Unknown
    UNKNOWN = "Unknown"


# Maps failure category → recommended repair action (string key)
REPAIR_ACTIONS: dict = {
    FailureCategory.HYPOTHESIS_MISALIGNED: "regenerate_hypothesis",
    FailureCategory.EXPERIMENT_MISSPECIFIED: "fix_experiment_spec",
    FailureCategory.UNSUPPORTED_CLAIM: "add_evidence_edge",
    FailureCategory.CODE_ERROR: "fix_code_error",
    FailureCategory.RUNTIME_CRASH: "increase_resources",
    FailureCategory.TIMEOUT: "reduce_complexity",
    FailureCategory.OVERFITTING: "add_regularization",
    FailureCategory.UNDERFITTING: "increase_capacity",
    FailureCategory.DIVERGENCE: "reduce_learning_rate",
    FailureCategory.PREMATURE_CONVERGENCE: "increase_exploration",
    FailureCategory.CATASTROPHIC_FORGETTING: "add_replay_buffer",
    FailureCategory.LOW_PERFORMANCE: "switch_strategy",
    FailureCategory.NEGATIVE_TRANSFER: "disable_memory_retrieval",
    FailureCategory.STALE_MEMORY: "consolidate_memory",
    FailureCategory.UNKNOWN: "run_ablation",
}


def all_categories() -> List[str]:
    return [c.value for c in FailureCategory]

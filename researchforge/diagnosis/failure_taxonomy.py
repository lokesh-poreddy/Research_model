"""Failure Taxonomy & diagnosis, merging the categories from both source
documents: Design/Implementation/Performance Failures (design doc Sec. 1) and
Overfitting/Underfitting/Divergence (technical report Sec. 1), operationalised
as a rule-based classifier matching the design doc's Sec. 4.2 `diagnose_failure`
pseudocode (weak-link inspection -> execution-error check -> performance check).
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FailureCategory(str, Enum):
    NONE = "None"
    EXECUTION_ERROR = "ExecutionError"     # Implementation Failure
    OVERFITTING = "Overfitting"            # Performance Failure
    UNDERFITTING = "Underfitting"          # Performance Failure
    LOW_PERFORMANCE = "LowPerformance"     # Performance Failure (below target)
    DIVERGENCE = "Divergence"              # Performance Failure (NaN / invalid metric)


@dataclass
class ExperimentResult:
    metric: float                          # validation score; the reward signal
    train_metric: Optional[float] = None
    success: bool = True
    exception: Optional[str] = None
    target: float = 0.0                    # minimum acceptable metric for this task


def diagnose(result: ExperimentResult, overfit_gap: float = 0.12,
             underfit_floor: float = 0.35) -> FailureCategory:
    if result.exception is not None or not result.success:
        return FailureCategory.EXECUTION_ERROR
    if result.metric != result.metric:  # NaN check (x != x is True only for NaN)
        return FailureCategory.DIVERGENCE
    if result.train_metric is not None:
        if result.train_metric - result.metric > overfit_gap and result.train_metric > 0.9:
            return FailureCategory.OVERFITTING
        if result.train_metric < underfit_floor and result.metric < underfit_floor:
            return FailureCategory.UNDERFITTING
    if result.metric < result.target:
        return FailureCategory.LOW_PERFORMANCE
    return FailureCategory.NONE

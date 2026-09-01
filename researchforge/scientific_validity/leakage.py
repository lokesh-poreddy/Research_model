"""Data Leakage Detector.

Checks whether test-set information has contaminated the training set in a
way that would inflate reported performance. This is one of the most
common sources of invalid results in applied ML research.

Implemented checks (A → G below)
-----------------------------------
A. Duplicate-sample leakage
   Identical feature rows appear in both train and test.

B. Near-duplicate leakage (optional, controlled by near_duplicate_threshold)
   Rows within L2-distance < threshold appear across splits.

C. Target leakage
   A feature has > max_target_correlation correlation with the label.
   Catches cases where a proxy of the label was accidentally included as a
   feature (e.g., a pre-aggregated stat computed on the full dataset).

D. Index leakage
   train_indices and test_indices overlap (the splits are not disjoint).

Currently NOT implemented (planned for later RF versions):
  - Group / subject overlap (requires group_ids argument)
  - Temporal contamination (requires timestamp argument)
  - Preprocessing leakage (requires fitting transformers only on train)

Each failed check produces a BLOCKER CheckResult. The leakage detector does
not block a run by itself -- that is the gate's responsibility.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from .verdicts import (
    CheckResult, CheckSeverity, ValidityVerdict,
)

PROVENANCE = "scientific_validity.leakage.DataLeakageDetector"


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(float) - b.astype(float)
    return float(math.sqrt(float(np.dot(diff, diff))))


class DataLeakageDetector:
    """Detects data leakage between training and test sets.

    Parameters
    ----------
    near_duplicate_threshold : float | None
        If set, check B (near-duplicate leakage) runs. Disabled by default
        because it is O(n_train × n_test) and expensive on large datasets.
    max_target_correlation : float
        Feature-label correlation above this triggers a target-leakage warning.
        Default 0.99 — only catches extreme/obvious cases.
    """

    def __init__(
        self,
        near_duplicate_threshold: Optional[float] = None,
        max_target_correlation: float = 0.99,
    ) -> None:
        self.near_duplicate_threshold = near_duplicate_threshold
        self.max_target_correlation = max_target_correlation

    def check(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        train_indices: Optional[Sequence[int]] = None,
        test_indices: Optional[Sequence[int]] = None,
    ) -> List[CheckResult]:
        """Run all enabled leakage checks. Returns list of CheckResults."""
        results: List[CheckResult] = []

        # A — Duplicate-sample leakage
        results.append(self._check_duplicates(X_train, X_test))

        # B — Near-duplicate (optional)
        if self.near_duplicate_threshold is not None:
            results.append(self._check_near_duplicates(
                X_train, X_test, self.near_duplicate_threshold))

        # C — Target leakage
        results.append(self._check_target_leakage(X_train, y_train))

        # D — Index leakage
        if train_indices is not None and test_indices is not None:
            results.append(self._check_index_leakage(
                train_indices, test_indices))

        return results

    # ── Individual checks ────────────────────────────────────────────────

    def _check_duplicates(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> CheckResult:
        """Check A: exact duplicate rows across train/test boundary."""
        train_set: Set[tuple] = {tuple(row.tolist()) for row in X_train}
        leakers = [i for i, row in enumerate(X_test)
                   if tuple(row.tolist()) in train_set]
        passed = len(leakers) == 0
        return CheckResult(
            check_name="duplicate_sample_leakage",
            passed=passed,
            severity=CheckSeverity.BLOCKER,
            verdict=ValidityVerdict.PASS if passed else ValidityVerdict.FAIL,
            description=(
                "No exact duplicate samples across train/test split."
                if passed else
                f"{len(leakers)} test sample(s) are exact duplicates of "
                f"training samples (indices: {leakers[:5]}{'...' if len(leakers)>5 else ''})."
            ),
            evidence={"n_leaking_test_samples": len(leakers),
                       "example_indices": leakers[:5]},
            provenance=PROVENANCE,
        )

    def _check_near_duplicates(
        self, X_train: np.ndarray, X_test: np.ndarray, threshold: float
    ) -> CheckResult:
        """Check B: near-duplicate rows within L2 < threshold."""
        leakers = []
        for i, test_row in enumerate(X_test):
            for train_row in X_train:
                if _l2(test_row, train_row) < threshold:
                    leakers.append(i)
                    break
        passed = len(leakers) == 0
        return CheckResult(
            check_name="near_duplicate_leakage",
            passed=passed,
            severity=CheckSeverity.WARNING,  # warning only: threshold is heuristic
            verdict=ValidityVerdict.PASS if passed else ValidityVerdict.WARNING,
            description=(
                f"No near-duplicate samples within L2 < {threshold}."
                if passed else
                f"{len(leakers)} test sample(s) have near-duplicate training "
                f"neighbours (L2 < {threshold}, indices: {leakers[:5]})."
            ),
            evidence={"threshold": threshold, "n_near_duplicates": len(leakers)},
            provenance=PROVENANCE,
        )

    def _check_target_leakage(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> CheckResult:
        """Check C: any feature highly correlated with the label."""
        y = y_train.astype(float)
        # For multi-class, use one-vs-rest correlation with the first class
        if y.ndim == 1:
            y_ref = (y == y[0]).astype(float)
        else:
            y_ref = y[:, 0].astype(float)

        max_corr = 0.0
        worst_feature = -1
        for j in range(X_train.shape[1]):
            feat = X_train[:, j].astype(float)
            std_f = float(np.std(feat))
            std_y = float(np.std(y_ref))
            if std_f < 1e-10 or std_y < 1e-10:
                continue
            corr = abs(float(np.corrcoef(feat, y_ref)[0, 1]))
            if corr > max_corr:
                max_corr = corr
                worst_feature = j

        passed = max_corr < self.max_target_correlation
        return CheckResult(
            check_name="target_leakage",
            passed=passed,
            severity=CheckSeverity.BLOCKER,
            verdict=ValidityVerdict.PASS if passed else ValidityVerdict.FAIL,
            description=(
                f"No feature has correlation ≥ {self.max_target_correlation} with target."
                if passed else
                f"Feature {worst_feature} has correlation {max_corr:.4f} ≥ "
                f"{self.max_target_correlation} with target — possible target leakage."
            ),
            evidence={"max_feature_target_correlation": max_corr,
                       "worst_feature_index": worst_feature,
                       "threshold": self.max_target_correlation},
            provenance=PROVENANCE,
        )

    def _check_index_leakage(
        self,
        train_indices: Sequence[int],
        test_indices: Sequence[int],
    ) -> CheckResult:
        """Check D: train and test index sets are disjoint."""
        overlap = set(train_indices) & set(test_indices)
        passed = len(overlap) == 0
        return CheckResult(
            check_name="index_leakage",
            passed=passed,
            severity=CheckSeverity.BLOCKER,
            verdict=ValidityVerdict.PASS if passed else ValidityVerdict.FAIL,
            description=(
                "Train and test index sets are disjoint."
                if passed else
                f"{len(overlap)} index(es) appear in both train and test sets: "
                f"{sorted(overlap)[:5]}."
            ),
            evidence={"n_overlapping_indices": len(overlap),
                       "example_overlap": sorted(overlap)[:5]},
            provenance=PROVENANCE,
        )

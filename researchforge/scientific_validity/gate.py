"""Scientific Validity Gate — orchestrator.

The gate runs all enabled validity checks and produces a ValidityReport.
Results must pass the gate before they are eligible for promotion to the
validated-result pool or for inclusion in the research dossier.

Architecture
------------
The gate is extensible: new check types (DatasetIntegrity, ReproducibilityCheck,
etc.) can be added without changing existing checks. Each check returns a list of
CheckResults that are aggregated into the report.

Current checks (RF-1.0.0-alpha.1)
-----------------------------------
1. Data Leakage Detection       (DataLeakageDetector)
2. Label-Permutation Sanity     (LabelPermutationTest)
3. Baseline Fairness            (BaselineFairnessValidator)
4. Statistical Significance     (StatisticalSignificanceTester)

Planned checks (RF-2.0+)
--------------------------
5. Dataset Integrity (hash check, duplicate-row rate, NaN rate)
6. Split Validation (stratification, temporal ordering)
7. Metric Sanity (predictions in valid range, confusion matrix sanity)
8. Reproducibility Check (same seed → identical result)
9. Seed Consistency (variance across seeds)
10. Ablation Validity (each component contributes)
11. Environment Consistency (dependency hash)
12. Claim-Evidence Consistency (claims derivable from evidence chain)
13. Novelty Verification (result differs from prior art)
14. Human Review (high-stakes decision requiring human sign-off)

Usage
-----
    gate = ScientificValidityGate()
    report = gate.run_leakage_check(
        experiment_id="exp_001",
        X_train=Xtr, X_test=Xte,
        y_train=ytr, y_test=yte)
    print(report.verdict)   # ValidityVerdict.PASS / FAIL / ...

    # Or run the full standard suite:
    report = gate.run_standard_suite(
        experiment_id="exp_001",
        X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
        real_metric=0.87,
        estimator_factory=lambda: RandomForestClassifier(n_estimators=50),
        metric_fn=accuracy_score)
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from .verdicts import ValidityReport, ValidityVerdict, CheckResult
from .leakage import DataLeakageDetector
from .permutation import LabelPermutationTest
from .baseline import BaselineFairnessValidator, ModelConfig
from .significance import StatisticalSignificanceTester

GATE_VERSION = "1.0.0-alpha.1"


class ScientificValidityGate:
    """Orchestrates all validity checks for an experiment result.

    Parameters
    ----------
    enable_near_duplicate_check : bool
        Run the O(n^2) near-duplicate leakage check. Default False.
    permutation_n : int
        Number of label permutations. Default 5.
    permutation_min_gap : float
        Minimum gap between real and permuted performance. Default 0.05.
    significance_alpha : float
        P-value threshold for significance tests. Default 0.05.
    significance_paired : bool
        Use paired (True) or unpaired (False) t-test. Default True.
    max_target_correlation : float
        Feature-label correlation threshold for target leakage. Default 0.99.
    """

    def __init__(
        self,
        enable_near_duplicate_check: bool = False,
        permutation_n: int = 5,
        permutation_min_gap: float = 0.05,
        significance_alpha: float = 0.05,
        significance_paired: bool = True,
        max_target_correlation: float = 0.99,
    ) -> None:
        self.enable_near_duplicate_check = enable_near_duplicate_check
        self.permutation_n = permutation_n
        self.permutation_min_gap = permutation_min_gap
        self.significance_alpha = significance_alpha
        self.significance_paired = significance_paired
        self.max_target_correlation = max_target_correlation

    # ── Individual check entry points ────────────────────────────────────

    def run_leakage_check(
        self,
        experiment_id: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        train_indices: Optional[Sequence[int]] = None,
        test_indices: Optional[Sequence[int]] = None,
    ) -> ValidityReport:
        """Run only the data leakage checks."""
        detector = DataLeakageDetector(
            near_duplicate_threshold=(
                1e-6 if self.enable_near_duplicate_check else None),
            max_target_correlation=self.max_target_correlation)
        checks = detector.check(
            X_train, X_test, y_train, y_test,
            train_indices, test_indices)
        report = ValidityReport(experiment_id=experiment_id, checks=checks)
        return report.finalize()

    def run_permutation_check(
        self,
        experiment_id: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        estimator_factory: Callable[[], Any],
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        real_metric: Optional[float] = None,
    ) -> ValidityReport:
        """Run only the label-permutation sanity test."""
        test = LabelPermutationTest(
            estimator_factory=estimator_factory,
            metric_fn=metric_fn,
            min_gap=self.permutation_min_gap,
            n_permutations=self.permutation_n)
        check = test.check(X_train, X_test, y_train, y_test, real_metric)
        report = ValidityReport(experiment_id=experiment_id, checks=[check])
        return report.finalize()

    def run_significance_check(
        self,
        experiment_id: str,
        new_scores: Sequence[float],
        baseline_scores: Sequence[float],
        metric_name: str = "metric",
    ) -> ValidityReport:
        """Run only the statistical significance test."""
        tester = StatisticalSignificanceTester(
            alpha=self.significance_alpha,
            paired=self.significance_paired)
        check = tester.check(new_scores, baseline_scores, metric_name)
        report = ValidityReport(experiment_id=experiment_id, checks=[check])
        return report.finalize()

    # ── Full standard suite ───────────────────────────────────────────────

    def run_standard_suite(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        estimator_factory: Callable[[], Any],
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        real_metric: Optional[float] = None,
        new_scores: Optional[Sequence[float]] = None,
        baseline_scores: Optional[Sequence[float]] = None,
        train_indices: Optional[Sequence[int]] = None,
        test_indices: Optional[Sequence[int]] = None,
        experiment_id: Optional[str] = None,
        metric_name: str = "metric",
    ) -> ValidityReport:
        """Run all enabled standard checks and aggregate into one report.

        Checks run:
          1. Data leakage (always)
          2. Label permutation (always, requires estimator_factory + metric_fn)
          3. Statistical significance (only if new_scores + baseline_scores given)
        """
        eid = experiment_id or f"exp_{uuid.uuid4().hex[:8]}"
        all_checks: List[CheckResult] = []

        # 1. Leakage
        detector = DataLeakageDetector(
            near_duplicate_threshold=(
                1e-6 if self.enable_near_duplicate_check else None),
            max_target_correlation=self.max_target_correlation)
        all_checks.extend(detector.check(
            X_train, X_test, y_train, y_test,
            train_indices, test_indices))

        # 2. Permutation
        perm_test = LabelPermutationTest(
            estimator_factory=estimator_factory,
            metric_fn=metric_fn,
            min_gap=self.permutation_min_gap,
            n_permutations=self.permutation_n)
        all_checks.append(perm_test.check(
            X_train, X_test, y_train, y_test, real_metric))

        # 3. Significance (optional)
        if new_scores is not None and baseline_scores is not None:
            tester = StatisticalSignificanceTester(
                alpha=self.significance_alpha,
                paired=self.significance_paired)
            all_checks.append(tester.check(
                new_scores, baseline_scores, metric_name))

        report = ValidityReport(experiment_id=eid, checks=all_checks)
        return report.finalize()

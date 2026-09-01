"""Baseline Fairness Validator.

Checks that the comparison between a new model and a baseline is fair: same
dataset, same split, same preprocessing decision (scale/no scale), same
training budget, same evaluation metric, same seed.

This is more important than it sounds. An unfair comparison can "prove" that
a new model is better simply by giving the baseline an unfair configuration.

What is checked
---------------
A. Same split: train/test sizes must match (within tolerance).
B. Same data: feature shapes must match.
C. Same preprocessing: both models must use or not use scaling (if declared).
D. Same training budget: max_iter or n_estimators must be within ratio.
E. Same seed: random states must match (WARNING if they differ, not BLOCKER).
F. Sanity check: baseline metric must be above chance level.

Design note: this validator checks DECLARED configuration equality, not
runtime execution equality. It catches common mistakes, not adversarial
misconfiguration. Runtime equality (same actual data bytes) is the
leakage detector's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .verdicts import CheckResult, CheckSeverity, ValidityVerdict

PROVENANCE = "scientific_validity.baseline.BaselineFairnessValidator"


@dataclass
class ModelConfig:
    """Declared configuration of a model for fairness comparison."""
    name: str
    model_type: str
    uses_scaling: bool = True
    max_iter: Optional[int] = None       # for gradient-based / iterative models
    n_estimators: Optional[int] = None   # for ensembles
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BaselineFairnessValidator:
    """Validates that a new model is compared fairly against a baseline.

    Parameters
    ----------
    budget_ratio_tolerance : float
        Max allowed ratio between the two models' training budgets.
        E.g., 2.0 means new can use up to 2× the baseline budget. Default 2.0.
    split_size_tolerance : float
        Fractional tolerance on train/test sizes. Default 0.01 (1%).
    chance_level : float
        Minimum metric for the baseline to be considered a valid baseline.
        Below this → REQUIRES_HUMAN_REVIEW (the baseline itself may be broken).
    """

    def __init__(
        self,
        budget_ratio_tolerance: float = 2.0,
        split_size_tolerance: float = 0.01,
        chance_level: float = 0.1,
    ) -> None:
        self.budget_ratio_tolerance = budget_ratio_tolerance
        self.split_size_tolerance = split_size_tolerance
        self.chance_level = chance_level

    def check(
        self,
        new_config: ModelConfig,
        baseline_config: ModelConfig,
        X_train_new: np.ndarray,
        X_train_baseline: np.ndarray,
        X_test_new: np.ndarray,
        X_test_baseline: np.ndarray,
        baseline_metric: Optional[float] = None,
    ) -> List[CheckResult]:
        """Run all fairness checks."""
        results: List[CheckResult] = []

        # A — Split size match
        results.append(self._check_split_size(
            X_train_new, X_train_baseline, X_test_new, X_test_baseline))

        # B — Feature shape match
        results.append(self._check_feature_shape(X_train_new, X_train_baseline))

        # C — Preprocessing match
        results.append(self._check_preprocessing(new_config, baseline_config))

        # D — Training budget match
        results.append(self._check_budget(new_config, baseline_config))

        # E — Seed match (warning)
        results.append(self._check_seed(new_config, baseline_config))

        # F — Baseline sanity
        if baseline_metric is not None:
            results.append(self._check_baseline_sanity(baseline_metric))

        return results

    # ── Individual checks ────────────────────────────────────────────────

    def _check_split_size(
        self,
        Xtr_new: np.ndarray, Xtr_base: np.ndarray,
        Xte_new: np.ndarray, Xte_base: np.ndarray,
    ) -> CheckResult:
        tol = self.split_size_tolerance
        n_tr_new, n_tr_base = len(Xtr_new), len(Xtr_base)
        n_te_new, n_te_base = len(Xte_new), len(Xte_base)
        train_diff = abs(n_tr_new - n_tr_base) / max(n_tr_base, 1)
        test_diff = abs(n_te_new - n_te_base) / max(n_te_base, 1)
        passed = train_diff <= tol and test_diff <= tol
        return CheckResult(
            check_name="baseline_split_size_match",
            passed=passed,
            severity=CheckSeverity.BLOCKER,
            verdict=ValidityVerdict.PASS if passed else ValidityVerdict.FAIL,
            description=(
                f"Train/test split sizes match within {tol*100:.0f}%."
                if passed else
                f"Split size mismatch: train ({n_tr_new} vs {n_tr_base}, "
                f"diff={train_diff:.3f}), test ({n_te_new} vs {n_te_base}, "
                f"diff={test_diff:.3f}). Comparison is not on the same data."
            ),
            evidence={"n_train_new": n_tr_new, "n_train_baseline": n_tr_base,
                       "n_test_new": n_te_new, "n_test_baseline": n_te_base,
                       "train_diff_fraction": train_diff,
                       "test_diff_fraction": test_diff},
            provenance=PROVENANCE,
        )

    def _check_feature_shape(
        self, Xtr_new: np.ndarray, Xtr_base: np.ndarray
    ) -> CheckResult:
        passed = Xtr_new.shape[1] == Xtr_base.shape[1]
        return CheckResult(
            check_name="baseline_feature_shape_match",
            passed=passed,
            severity=CheckSeverity.BLOCKER,
            verdict=ValidityVerdict.PASS if passed else ValidityVerdict.FAIL,
            description=(
                f"Feature dimensions match ({Xtr_new.shape[1]})."
                if passed else
                f"Feature dimension mismatch: new={Xtr_new.shape[1]}, "
                f"baseline={Xtr_base.shape[1]}. Models are not on the same task."
            ),
            evidence={"new_features": Xtr_new.shape[1],
                       "baseline_features": Xtr_base.shape[1]},
            provenance=PROVENANCE,
        )

    def _check_preprocessing(
        self, new: ModelConfig, baseline: ModelConfig
    ) -> CheckResult:
        passed = new.uses_scaling == baseline.uses_scaling
        return CheckResult(
            check_name="baseline_preprocessing_match",
            passed=passed,
            severity=CheckSeverity.WARNING,
            verdict=ValidityVerdict.PASS if passed else ValidityVerdict.WARNING,
            description=(
                f"Both models use the same preprocessing (scaling={new.uses_scaling})."
                if passed else
                f"Preprocessing mismatch: new uses_scaling={new.uses_scaling}, "
                f"baseline uses_scaling={baseline.uses_scaling}. This may give "
                f"one model an unfair advantage."
            ),
            evidence={"new_scaling": new.uses_scaling,
                       "baseline_scaling": baseline.uses_scaling},
            provenance=PROVENANCE,
        )

    def _check_budget(
        self, new: ModelConfig, baseline: ModelConfig
    ) -> CheckResult:
        """Check that training budgets are within ratio tolerance."""
        def _budget(cfg: ModelConfig) -> Optional[int]:
            return cfg.max_iter or cfg.n_estimators

        b_new = _budget(new)
        b_base = _budget(baseline)
        if b_new is None or b_base is None:
            return CheckResult(
                check_name="baseline_budget_match",
                passed=True,
                severity=CheckSeverity.INFO,
                verdict=ValidityVerdict.PASS,
                description="Budget not declared for one or both models; skipping.",
                evidence={},
                provenance=PROVENANCE,
            )
        ratio = b_new / max(b_base, 1)
        passed = ratio <= self.budget_ratio_tolerance
        return CheckResult(
            check_name="baseline_budget_match",
            passed=passed,
            severity=CheckSeverity.WARNING,
            verdict=ValidityVerdict.PASS if passed else ValidityVerdict.WARNING,
            description=(
                f"Training budget ratio {ratio:.2f}× within tolerance "
                f"{self.budget_ratio_tolerance:.1f}×."
                if passed else
                f"New model has {ratio:.2f}× the training budget of baseline "
                f"({b_new} vs {b_base}), exceeding tolerance "
                f"{self.budget_ratio_tolerance:.1f}×. Comparison may be unfair."
            ),
            evidence={"new_budget": b_new, "baseline_budget": b_base,
                       "ratio": ratio, "tolerance": self.budget_ratio_tolerance},
            provenance=PROVENANCE,
        )

    def _check_seed(
        self, new: ModelConfig, baseline: ModelConfig
    ) -> CheckResult:
        seeds_match = new.seed == baseline.seed
        return CheckResult(
            check_name="baseline_seed_match",
            passed=seeds_match,
            severity=CheckSeverity.WARNING,
            verdict=ValidityVerdict.PASS if seeds_match else ValidityVerdict.WARNING,
            description=(
                f"Both models use seed {new.seed}."
                if seeds_match else
                f"Seeds differ (new={new.seed}, baseline={baseline.seed}). "
                f"Result differences may be due to randomness, not model quality."
            ),
            evidence={"new_seed": new.seed, "baseline_seed": baseline.seed},
            provenance=PROVENANCE,
        )

    def _check_baseline_sanity(self, baseline_metric: float) -> CheckResult:
        """Baseline metric must be above chance to be a valid reference."""
        passed = baseline_metric >= self.chance_level
        verdict = (ValidityVerdict.PASS if passed
                   else ValidityVerdict.REQUIRES_HUMAN_REVIEW)
        return CheckResult(
            check_name="baseline_above_chance",
            passed=passed,
            severity=(CheckSeverity.BLOCKER if not passed
                      else CheckSeverity.INFO),
            verdict=verdict,
            description=(
                f"Baseline metric {baseline_metric:.4f} is above chance level "
                f"{self.chance_level}."
                if passed else
                f"Baseline metric {baseline_metric:.4f} is at or below chance "
                f"level {self.chance_level}. The baseline may be broken or the "
                f"task definition may be incorrect. Human review required."
            ),
            evidence={"baseline_metric": baseline_metric,
                       "chance_level": self.chance_level},
            provenance=PROVENANCE,
        )

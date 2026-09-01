"""Label-Permutation Sanity Test.

The core question: does the pipeline's reported metric collapse to chance
when the labels are randomly shuffled?

If real_metric ≈ permuted_metric, the pipeline is likely leaking label
information through preprocessing, data ordering, or evaluation procedure.
If permuted_metric is already at chance level, the pipeline is using the
labels correctly.

How it works
------------
1. Train the estimator on (X_train, y_train) — record real_metric.
2. Repeat n_permutations times:
   a. Shuffle y_train randomly (y_perm).
   b. Train the same estimator class (fresh instance) on (X_train, y_perm).
   c. Evaluate on X_test with the ORIGINAL y_test.
   d. Record perm_metric_i.
3. Compute the permuted distribution mean and std.
4. Compute the gap = real_metric - mean(perm_metrics).
5. Report FAIL if gap < min_gap (the pipeline isn't learning from real labels).

Why test against y_test (original)?
-------------------------------------
We permute TRAINING labels only. The test set keeps real labels. This means:
  - A correctly functioning pipeline: perm_metric ≈ chance (can't generalise
    from random labels).
  - A leaky pipeline: perm_metric ≈ real_metric (label info was bypassed).

The test does NOT claim the pipeline is GOOD — only that it is USING the labels.

Parameters
----------
min_gap : float
    Minimum required gap between real performance and mean permuted performance.
    Default 0.05 (5 percentage points). Adjust for chance-level task difficulty.
n_permutations : int
    How many permutations to run. Default 5 (cheap). Increase for publication.
random_state : int
    Seed for the permutation RNG.
"""
from __future__ import annotations

import random
from typing import Any, Callable, List, Optional

import numpy as np

from .verdicts import CheckResult, CheckSeverity, ValidityVerdict

PROVENANCE = "scientific_validity.permutation.LabelPermutationTest"


class LabelPermutationTest:
    """Sanity-check that the pipeline actually uses training labels.

    Parameters
    ----------
    estimator_factory : callable
        No-argument callable that returns a fresh, unfitted estimator with
        .fit(X, y) and .predict(X) methods. Called once per permutation.
    metric_fn : callable
        metric_fn(y_true, y_pred) -> float.
    min_gap : float
        Minimum required real_metric - mean(perm_metrics). Default 0.05.
    n_permutations : int
        Number of label permutations to run. Default 5.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        min_gap: float = 0.05,
        n_permutations: int = 5,
        random_state: int = 0,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.metric_fn = metric_fn
        self.min_gap = min_gap
        self.n_permutations = n_permutations
        self.random_state = random_state

    def check(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        real_metric: Optional[float] = None,
    ) -> CheckResult:
        """Run the permutation test. Returns a single CheckResult.

        Parameters
        ----------
        X_train, X_test, y_train, y_test : arrays
            The same split used for the actual experiment.
        real_metric : float | None
            The metric from the real (non-permuted) training run.
            If None, it is re-computed inside this function.
        """
        rng = random.Random(self.random_state)

        # Real metric
        if real_metric is None:
            est = self.estimator_factory()
            est.fit(X_train, y_train)
            real_metric = self.metric_fn(y_test, est.predict(X_test))

        # Permuted metrics
        perm_metrics: List[float] = []
        for i in range(self.n_permutations):
            seed = self.random_state + i + 1
            np_rng = np.random.default_rng(seed)
            y_perm = np_rng.permutation(y_train)
            try:
                est = self.estimator_factory()
                est.fit(X_train, y_perm)
                perm_score = self.metric_fn(y_test, est.predict(X_test))
            except Exception as exc:
                # A permuted fit can fail (e.g., single-class y_perm for some
                # estimators). That's fine — treat it as a failure to diverge
                # from chance (worst-case: same as real, which is the safe side).
                perm_score = real_metric
            perm_metrics.append(perm_score)

        mean_perm = sum(perm_metrics) / len(perm_metrics)
        std_perm = (
            (sum((x - mean_perm) ** 2 for x in perm_metrics) / len(perm_metrics)) ** 0.5
            if len(perm_metrics) > 1 else 0.0
        )
        gap = real_metric - mean_perm
        passed = gap >= self.min_gap

        return CheckResult(
            check_name="label_permutation_sanity",
            passed=passed,
            severity=CheckSeverity.BLOCKER,
            verdict=ValidityVerdict.PASS if passed else ValidityVerdict.FAIL,
            description=(
                f"Pipeline performance drops {gap:.4f} below permuted baseline "
                f"(real={real_metric:.4f}, perm_mean={mean_perm:.4f}, "
                f"gap ≥ {self.min_gap}). Labels are being used correctly."
                if passed else
                f"Pipeline performance did NOT drop sufficiently below permuted "
                f"baseline (real={real_metric:.4f}, perm_mean={mean_perm:.4f}, "
                f"gap={gap:.4f} < {self.min_gap}). Possible label leakage or "
                f"metric/data issue."
            ),
            evidence={
                "real_metric": real_metric,
                "perm_mean": mean_perm,
                "perm_std": std_perm,
                "perm_metrics": perm_metrics,
                "gap": gap,
                "min_gap": self.min_gap,
                "n_permutations": self.n_permutations,
            },
            provenance=PROVENANCE,
        )

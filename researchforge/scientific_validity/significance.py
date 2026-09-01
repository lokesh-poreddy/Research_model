"""Statistical Significance Tester.

Checks whether the performance difference between two models is statistically
meaningful, or could be explained by random variation.

For RF-1.0.0-alpha.1, we implement paired and unpaired t-tests, which are
appropriate for comparing models across multiple seeds. Later RF versions
will add:
  - Wilcoxon signed-rank test (non-parametric paired)
  - Bootstrap confidence intervals
  - Multiple comparison corrections (Bonferroni, Holm-Bonferroni)
  - Effect-size reporting (Cohen's d, Glass's delta)

What is returned
----------------
For each comparison, returns a CheckResult with:
  - p_value
  - effect_size (Cohen's d)
  - confidence_interval (95%)
  - verdict: PASS (significant improvement), WARNING (marginal), FAIL
    (worse than baseline), INCONCLUSIVE (insufficient data)

Significance thresholds
-----------------------
  alpha = 0.05  for main verdict
  alpha = 0.10  for marginal (WARNING)

A result is INCONCLUSIVE when n < min_samples_for_significance.

IMPORTANT: p-value < 0.05 does NOT prove the new model is better in absolute
terms. It only means the observed difference is unlikely to be due to chance
given the sample size. Effect size and confidence intervals carry more
scientific weight.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .verdicts import CheckResult, CheckSeverity, ValidityVerdict

PROVENANCE = "scientific_validity.significance.StatisticalSignificanceTester"

# Minimum number of samples (seeds) for significance tests to be meaningful
MIN_SAMPLES_FOR_SIGNIFICANCE = 3


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / n
    return m, math.sqrt(var)


def _t_test_paired(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Two-tailed paired t-test. Returns (t_statistic, p_value approx)."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0, 1.0
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    std_d = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / (n - 1))
    if std_d < 1e-12:
        return float("inf") if mean_d != 0 else 0.0, 0.0
    t = mean_d / (std_d / math.sqrt(n))
    # Approximate p-value from t and df=n-1 using normal approximation
    # (adequate for n >= 3; for small n this is slightly anti-conservative)
    p = _approx_p_from_t(abs(t), df=n - 1)
    return t, p


def _t_test_unpaired(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Welch's t-test (unequal variances). Returns (t_statistic, p_value approx)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 1.0
    ma, sa = _mean_std(a)
    mb, sb = _mean_std(b)
    # Unbiased std
    sa_u = math.sqrt(sum((x - ma) ** 2 for x in a) / (na - 1)) if na > 1 else 0.0
    sb_u = math.sqrt(sum((x - mb) ** 2 for x in b) / (nb - 1)) if nb > 1 else 0.0
    denom = math.sqrt(sa_u ** 2 / na + sb_u ** 2 / nb)
    if denom < 1e-12:
        return 0.0, 1.0
    t = (ma - mb) / denom
    # Welch–Satterthwaite df
    num = (sa_u ** 2 / na + sb_u ** 2 / nb) ** 2
    den = ((sa_u ** 2 / na) ** 2 / (na - 1) + (sb_u ** 2 / nb) ** 2 / (nb - 1))
    df = num / den if den > 0 else na + nb - 2
    p = _approx_p_from_t(abs(t), df=max(1, df))
    return t, p


def _approx_p_from_t(t_abs: float, df: float) -> float:
    """Two-tailed p-value approximation from |t| using normal approximation.

    Uses the relationship t → z as df → ∞. Adequate for df >= 3.
    For very small df the approximation is anti-conservative (reports
    smaller p than true). Sufficient for the diagnostic purposes here.
    """
    # Abramowitz & Stegun approximation to Φ (standard normal CDF)
    def _phi(z: float) -> float:
        if z < 0:
            return 1.0 - _phi(-z)
        k = 1.0 / (1.0 + 0.2316419 * z)
        poly = k * (0.319381530
                    + k * (-0.356563782
                           + k * (1.781477937
                                  + k * (-1.821255978 + k * 1.330274429))))
        return 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly

    # Scale t by df correction toward z
    z = t_abs * math.sqrt(df / (df + t_abs ** 2)) if df > 0 else t_abs
    p_one_tail = 1.0 - _phi(z)
    return min(1.0, 2.0 * p_one_tail)


def _cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d effect size: (mean_a - mean_b) / pooled_std."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma = sum(a) / na
    mb = sum(b) / nb
    var_a = sum((x - ma) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mb) ** 2 for x in b) / (nb - 1)
    pooled_std = math.sqrt((var_a + var_b) / 2)
    return (ma - mb) / pooled_std if pooled_std > 1e-12 else 0.0


def _confidence_interval_95(values: Sequence[float]) -> Tuple[float, float]:
    """95% CI for the mean using t-distribution (normal approx for n>=3)."""
    n = len(values)
    if n < 2:
        m = values[0] if values else 0.0
        return m, m
    m = sum(values) / n
    std = math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))
    # t-critical for 95%, df=n-1, two-tailed: use 1.96 approximation
    t_crit = 1.96 if n >= 30 else {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
    }.get(n, 2.0)
    margin = t_crit * std / math.sqrt(n)
    return m - margin, m + margin


class StatisticalSignificanceTester:
    """Tests whether observed performance differences are statistically meaningful.

    Parameters
    ----------
    alpha : float
        Significance level for PASS verdict. Default 0.05.
    alpha_marginal : float
        Significance level for WARNING verdict. Default 0.10.
    paired : bool
        Use paired t-test (True) or Welch's unpaired t-test (False).
        Use paired when both models were evaluated on the same seeds.
    min_samples : int
        Minimum number of samples (seeds) before tests are meaningful.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        alpha_marginal: float = 0.10,
        paired: bool = True,
        min_samples: int = MIN_SAMPLES_FOR_SIGNIFICANCE,
    ) -> None:
        self.alpha = alpha
        self.alpha_marginal = alpha_marginal
        self.paired = paired
        self.min_samples = min_samples

    def check(
        self,
        new_scores: Sequence[float],
        baseline_scores: Sequence[float],
        metric_name: str = "metric",
    ) -> CheckResult:
        """Compare new_scores against baseline_scores.

        Parameters
        ----------
        new_scores : sequence of float
            Per-seed (or per-fold) scores for the new model.
        baseline_scores : sequence of float
            Per-seed scores for the baseline model.
        metric_name : str
            Human-readable name of the metric (e.g., "accuracy").
        """
        n_new, n_base = len(new_scores), len(baseline_scores)

        # Insufficient data
        if n_new < self.min_samples or n_base < self.min_samples:
            mean_new = sum(new_scores) / n_new if n_new else 0.0
            mean_base = sum(baseline_scores) / n_base if n_base else 0.0
            return CheckResult(
                check_name=f"statistical_significance_{metric_name}",
                passed=False,
                severity=CheckSeverity.WARNING,
                verdict=ValidityVerdict.INCONCLUSIVE,
                description=(
                    f"Insufficient data for significance test "
                    f"(n_new={n_new}, n_base={n_base}, "
                    f"min_required={self.min_samples}). "
                    f"Point estimates: new={mean_new:.4f}, "
                    f"baseline={mean_base:.4f}."
                ),
                evidence={"n_new": n_new, "n_baseline": n_base,
                           "min_samples": self.min_samples},
                provenance=PROVENANCE,
            )

        mean_new = sum(new_scores) / n_new
        mean_base = sum(baseline_scores) / n_base

        if self.paired and n_new == n_base:
            t_stat, p_value = _t_test_paired(new_scores, baseline_scores)
        else:
            t_stat, p_value = _t_test_unpaired(new_scores, baseline_scores)

        d = _cohens_d(new_scores, baseline_scores)
        ci_new = _confidence_interval_95(new_scores)
        ci_base = _confidence_interval_95(baseline_scores)

        diff = mean_new - mean_base

        # Determine verdict
        if diff < 0 and p_value < self.alpha:
            # Statistically significantly WORSE
            verdict = ValidityVerdict.FAIL
            passed = False
            desc = (
                f"New model ({mean_new:.4f}) is statistically significantly "
                f"WORSE than baseline ({mean_base:.4f}): "
                f"Δ={diff:+.4f}, p={p_value:.4f}, d={d:.3f}."
            )
        elif p_value < self.alpha and diff > 0:
            verdict = ValidityVerdict.PASS
            passed = True
            desc = (
                f"New model ({mean_new:.4f}) is statistically significantly "
                f"BETTER than baseline ({mean_base:.4f}): "
                f"Δ={diff:+.4f}, p={p_value:.4f}, d={d:.3f}."
            )
        elif p_value < self.alpha_marginal and diff > 0:
            verdict = ValidityVerdict.WARNING
            passed = False
            desc = (
                f"New model ({mean_new:.4f}) vs baseline ({mean_base:.4f}): "
                f"Δ={diff:+.4f} is marginally significant "
                f"(p={p_value:.4f}, α_marginal={self.alpha_marginal}). "
                f"Treat with caution."
            )
        else:
            verdict = ValidityVerdict.INCONCLUSIVE
            passed = False
            desc = (
                f"No statistically significant difference between new "
                f"({mean_new:.4f}) and baseline ({mean_base:.4f}): "
                f"Δ={diff:+.4f}, p={p_value:.4f} (α={self.alpha})."
            )

        return CheckResult(
            check_name=f"statistical_significance_{metric_name}",
            passed=passed,
            severity=CheckSeverity.INFO,  # significance alone doesn't block
            verdict=verdict,
            description=desc,
            evidence={
                "mean_new": mean_new,
                "mean_baseline": mean_base,
                "difference": diff,
                "p_value": p_value,
                "t_statistic": t_stat,
                "cohens_d": d,
                "ci_new_95": ci_new,
                "ci_baseline_95": ci_base,
                "paired": self.paired,
                "n_new": n_new,
                "n_baseline": n_base,
                "alpha": self.alpha,
            },
            provenance=PROVENANCE,
        )

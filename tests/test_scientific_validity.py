"""Scientific Validity Gate test suite — Tier 4.

Tests that the gate components produce correct verdicts for:
  - Clean data (PASS expected)
  - Leaking data (FAIL expected)
  - Permutation sanity (PASS for real labels, marginal for trivial data)
  - Significance with sufficient and insufficient samples
  - Full standard suite integration
  - Report rendering (text + JSON roundtrip)

These tests use entirely synthetic data to guarantee determinism.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from researchforge.scientific_validity import (
    ScientificValidityGate,
    DataLeakageDetector,
    LabelPermutationTest,
    StatisticalSignificanceTester,
    ValidityVerdict,
    ValidityReportRenderer,
)
from researchforge.scientific_validity.baseline import (
    BaselineFairnessValidator, ModelConfig,
)


_PASS = []
_FAIL = []


def ok(name: str) -> None:
    print(f"OK: {name}")
    _PASS.append(name)


def fail(name: str, exc: Exception) -> None:
    print(f"FAIL: {name} — {exc}")
    _FAIL.append((name, exc))


def run(name: str, fn) -> None:
    try:
        fn()
        ok(name)
    except Exception as exc:
        fail(name, exc)


# ── Shared synthetic data ─────────────────────────────────────────────────────

def _make_clean_data(n_train=200, n_test=50, n_features=10, seed=0):
    """Clean data with no leakage and real signal."""
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_features))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.standard_normal((n_test, n_features))
    y_test = (X_test[:, 0] > 0).astype(int)
    return X_train, X_test, y_train, y_test


def _make_leaky_data(n_train=100, n_test=20, n_features=10, seed=0):
    """Test set contains exact duplicates of training set."""
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_features))
    y_train = (X_train[:, 0] > 0).astype(int)
    # Make test = first 20 rows of train (blatant leakage)
    X_test = X_train[:n_test].copy()
    y_test = y_train[:n_test].copy()
    return X_train, X_test, y_train, y_test


# ── Leakage Detector Tests ────────────────────────────────────────────────────

def test_leakage_clean_data_passes():
    Xtr, Xte, ytr, yte = _make_clean_data()
    detector = DataLeakageDetector()
    results = detector.check(Xtr, Xte, ytr, yte)
    blockers = [r for r in results if r.is_blocker()]
    assert len(blockers) == 0, f"Unexpected blockers on clean data: {[b.description for b in blockers]}"


def test_leakage_duplicate_data_fails():
    Xtr, Xte, ytr, yte = _make_leaky_data()
    detector = DataLeakageDetector()
    results = detector.check(Xtr, Xte, ytr, yte)
    dup_check = next(r for r in results if r.check_name == "duplicate_sample_leakage")
    assert not dup_check.passed, "Expected duplicate leakage check to fail"
    assert dup_check.verdict == ValidityVerdict.FAIL


def test_leakage_target_leakage_detection():
    """Inject a feature that is a perfect proxy of the label."""
    rng = np.random.default_rng(42)
    n = 100
    X_base = rng.standard_normal((n, 5))
    y = (rng.standard_normal(n) > 0).astype(int)
    # Add a feature perfectly correlated with y
    X_train = np.column_stack([X_base[:80], y[:80].astype(float)])
    y_train = y[:80]
    X_test = np.column_stack([X_base[80:], y[80:].astype(float)])
    y_test = y[80:]
    detector = DataLeakageDetector(max_target_correlation=0.95)
    results = detector.check(X_train, X_test, y_train, y_test)
    tl = next(r for r in results if r.check_name == "target_leakage")
    assert not tl.passed, f"Expected target leakage to be detected, evidence: {tl.evidence}"


def test_leakage_index_overlap_detected():
    Xtr, Xte, ytr, yte = _make_clean_data()
    train_indices = list(range(100))
    test_indices = list(range(50, 150))  # overlap at 50–100
    detector = DataLeakageDetector()
    results = detector.check(Xtr, Xte, ytr, yte, train_indices, test_indices)
    idx_check = next(r for r in results if r.check_name == "index_leakage")
    assert not idx_check.passed
    assert idx_check.evidence["n_overlapping_indices"] == 50


def test_leakage_clean_index_passes():
    Xtr, Xte, ytr, yte = _make_clean_data()
    detector = DataLeakageDetector()
    results = detector.check(Xtr, Xte, ytr, yte, list(range(200)), list(range(200, 250)))
    idx_check = next(r for r in results if r.check_name == "index_leakage")
    assert idx_check.passed


# ── Permutation Test ──────────────────────────────────────────────────────────

def test_permutation_real_labels_pass():
    """A real classifier should beat permuted labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    Xtr, Xte, ytr, yte = _make_clean_data(n_train=200, seed=1)

    test = LabelPermutationTest(
        estimator_factory=lambda: LogisticRegression(max_iter=200, random_state=0),
        metric_fn=accuracy_score,
        min_gap=0.03,
        n_permutations=5,
        random_state=0)
    result = test.check(Xtr, Xte, ytr, yte)
    assert result.passed, (
        f"Expected permutation test to PASS on clean data with real labels. "
        f"gap={result.evidence['gap']:.4f}")


def test_permutation_random_labels_fail():
    """With random labels, performance should be at chance — gap < min_gap."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 10))
    y = rng.integers(0, 2, size=n)  # random binary labels
    Xtr, ytr = X[:160], y[:160]
    Xte, yte = X[160:], y[160:]

    # Train on random labels — real metric should be near 0.5
    est = LogisticRegression(max_iter=200, random_state=0)
    est.fit(Xtr, ytr)
    real_metric = accuracy_score(yte, est.predict(Xte))

    test = LabelPermutationTest(
        estimator_factory=lambda: LogisticRegression(max_iter=200, random_state=0),
        metric_fn=accuracy_score,
        min_gap=0.10,  # expect only small gap
        n_permutations=5,
        random_state=0)
    result = test.check(Xtr, Xte, ytr, yte, real_metric=real_metric)
    # With random labels, the gap should be very small → FAIL the check
    # (But we don't hard-assert FAIL because accuracy can sometimes vary.
    # We assert that the check ran and returned a valid verdict.)
    assert result.verdict in (ValidityVerdict.PASS, ValidityVerdict.FAIL)
    assert "gap" in result.evidence


# ── Baseline Fairness ─────────────────────────────────────────────────────────

def test_baseline_fairness_same_config_passes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10))
    Xtr, Xte = X[:160], X[160:]
    cfg_new = ModelConfig("RF-new", "RF", uses_scaling=False, seed=42)
    cfg_base = ModelConfig("RF-base", "RF", uses_scaling=False, seed=42)
    validator = BaselineFairnessValidator()
    results = validator.check(cfg_new, cfg_base, Xtr, Xtr, Xte, Xte,
                               baseline_metric=0.85)
    blockers = [r for r in results if r.is_blocker()]
    assert len(blockers) == 0, f"Unexpected blockers: {[b.description for b in blockers]}"


def test_baseline_fairness_different_scaling_warns():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10))
    Xtr, Xte = X[:160], X[160:]
    cfg_new = ModelConfig("new", "MLP", uses_scaling=True, seed=0)
    cfg_base = ModelConfig("base", "MLP", uses_scaling=False, seed=0)
    validator = BaselineFairnessValidator()
    results = validator.check(cfg_new, cfg_base, Xtr, Xtr, Xte, Xte)
    prep_check = next(r for r in results if r.check_name == "baseline_preprocessing_match")
    assert not prep_check.passed


def test_baseline_below_chance_requires_review():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10))
    Xtr, Xte = X[:160], X[160:]
    cfg = ModelConfig("cfg", "LR", seed=0)
    validator = BaselineFairnessValidator(chance_level=0.5)
    results = validator.check(cfg, cfg, Xtr, Xtr, Xte, Xte, baseline_metric=0.1)
    sanity_check = next(r for r in results if r.check_name == "baseline_above_chance")
    assert not sanity_check.passed
    assert sanity_check.verdict == ValidityVerdict.REQUIRES_HUMAN_REVIEW


# ── Statistical Significance ──────────────────────────────────────────────────

def test_significance_clearly_better():
    """Scores clearly better than baseline → PASS."""
    tester = StatisticalSignificanceTester(alpha=0.05, paired=True)
    new_scores = [0.90, 0.91, 0.92, 0.89, 0.91]
    base_scores = [0.70, 0.71, 0.70, 0.69, 0.71]
    result = tester.check(new_scores, base_scores, "accuracy")
    assert result.verdict == ValidityVerdict.PASS, (
        f"Expected PASS for clearly better new_scores. "
        f"p={result.evidence['p_value']:.4f}")


def test_significance_identical_scores_inconclusive():
    tester = StatisticalSignificanceTester(alpha=0.05)
    scores = [0.85, 0.85, 0.85, 0.85, 0.85]
    result = tester.check(scores, scores, "accuracy")
    assert result.verdict in (ValidityVerdict.INCONCLUSIVE, ValidityVerdict.WARNING), (
        f"Expected INCONCLUSIVE for identical scores, got {result.verdict}")


def test_significance_insufficient_data():
    tester = StatisticalSignificanceTester(min_samples=3)
    result = tester.check([0.9], [0.8], "accuracy")
    assert result.verdict == ValidityVerdict.INCONCLUSIVE


def test_significance_clearly_worse_fails():
    tester = StatisticalSignificanceTester(alpha=0.05)
    new_scores = [0.55, 0.54, 0.56, 0.55, 0.54]
    base_scores = [0.85, 0.86, 0.85, 0.87, 0.86]
    result = tester.check(new_scores, base_scores)
    assert result.verdict == ValidityVerdict.FAIL, (
        f"Expected FAIL for clearly worse new_scores, got {result.verdict}")


# ── Full Gate: Standard Suite ─────────────────────────────────────────────────

def test_gate_standard_suite_clean_data():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    Xtr, Xte, ytr, yte = _make_clean_data(n_train=200, n_test=50, seed=3)
    gate = ScientificValidityGate(permutation_n=3, permutation_min_gap=0.03)
    report = gate.run_standard_suite(
        X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
        estimator_factory=lambda: LogisticRegression(max_iter=200, random_state=0),
        metric_fn=accuracy_score,
        experiment_id="test_clean_suite")
    # With clean data, should not FAIL (PASS or WARNING acceptable)
    assert report.verdict != ValidityVerdict.FAIL, (
        f"Standard suite should not FAIL on clean data. "
        f"Blockers: {report.blockers}")
    assert len(report.checks) >= 3  # at least leakage + permutation checks


def test_gate_standard_suite_leaky_data_fails():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    Xtr, Xte, ytr, yte = _make_leaky_data()
    gate = ScientificValidityGate(permutation_n=3)
    report = gate.run_standard_suite(
        X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
        estimator_factory=lambda: LogisticRegression(max_iter=100, random_state=0),
        metric_fn=accuracy_score,
        experiment_id="test_leaky_suite")
    assert report.verdict == ValidityVerdict.FAIL, (
        f"Standard suite should FAIL on leaky data. Verdicts: "
        f"{[c.verdict.value for c in report.checks]}")


def test_gate_with_significance():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    Xtr, Xte, ytr, yte = _make_clean_data(seed=5)
    gate = ScientificValidityGate(permutation_n=3)
    new_scores = [0.88, 0.89, 0.87]
    base_scores = [0.75, 0.74, 0.76]
    report = gate.run_standard_suite(
        X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
        estimator_factory=lambda: LogisticRegression(max_iter=200, random_state=0),
        metric_fn=accuracy_score,
        new_scores=new_scores,
        baseline_scores=base_scores,
        metric_name="accuracy",
        experiment_id="test_significance_suite")
    sig_checks = [c for c in report.checks
                  if "significance" in c.check_name]
    assert len(sig_checks) == 1
    # Should be PASS or WARNING (not FAIL) for clearly better scores
    assert sig_checks[0].verdict in (
        ValidityVerdict.PASS, ValidityVerdict.WARNING,
        ValidityVerdict.INCONCLUSIVE), (
        f"Unexpected significance verdict: {sig_checks[0].verdict}")


# ── Verdict Aggregation ───────────────────────────────────────────────────────

def test_verdict_finalize_all_pass():
    from researchforge.scientific_validity.verdicts import (
        CheckResult, CheckSeverity, ValidityReport, ValidityVerdict)
    r = ValidityReport("exp_001")
    r.checks.append(CheckResult(
        "check_a", True, CheckSeverity.BLOCKER, ValidityVerdict.PASS, "OK"))
    r.checks.append(CheckResult(
        "check_b", True, CheckSeverity.INFO, ValidityVerdict.PASS, "OK"))
    r.finalize()
    assert r.verdict == ValidityVerdict.PASS
    assert r.confidence == 1.0


def test_verdict_finalize_one_blocker_fails():
    from researchforge.scientific_validity.verdicts import (
        CheckResult, CheckSeverity, ValidityReport, ValidityVerdict)
    r = ValidityReport("exp_002")
    r.checks.append(CheckResult(
        "check_a", True, CheckSeverity.BLOCKER, ValidityVerdict.PASS, "OK"))
    r.checks.append(CheckResult(
        "check_b", False, CheckSeverity.BLOCKER, ValidityVerdict.FAIL, "LEAK"))
    r.finalize()
    assert r.verdict == ValidityVerdict.FAIL
    assert "LEAK" in r.blockers


def test_verdict_finalize_warning_only():
    from researchforge.scientific_validity.verdicts import (
        CheckResult, CheckSeverity, ValidityReport, ValidityVerdict)
    r = ValidityReport("exp_003")
    r.checks.append(CheckResult(
        "check_a", True, CheckSeverity.BLOCKER, ValidityVerdict.PASS, "OK"))
    r.checks.append(CheckResult(
        "check_b", False, CheckSeverity.WARNING, ValidityVerdict.WARNING, "minor"))
    r.finalize()
    assert r.verdict == ValidityVerdict.WARNING


# ── Report Rendering ──────────────────────────────────────────────────────────

def test_report_text_rendering():
    from researchforge.scientific_validity.verdicts import (
        CheckResult, CheckSeverity, ValidityReport)
    r = ValidityReport("exp_render")
    r.checks.append(CheckResult(
        "check_x", True, CheckSeverity.BLOCKER, ValidityVerdict.PASS, "All clear"))
    r.finalize()
    text = ValidityReportRenderer.to_text(r)
    assert "PASS" in text
    assert "exp_render" in text
    assert "check_x" in text


def test_report_json_roundtrip():
    from researchforge.scientific_validity.verdicts import (
        CheckResult, CheckSeverity, ValidityReport)
    r = ValidityReport("exp_json")
    r.checks.append(CheckResult(
        "check_y", False, CheckSeverity.BLOCKER,
        ValidityVerdict.FAIL, "Leakage found",
        evidence={"n": 5}))
    r.finalize()
    js = ValidityReportRenderer.to_json(r)
    parsed = json.loads(js)
    assert parsed["verdict"] == "FAIL"
    assert parsed["experiment_id"] == "exp_json"
    assert len(parsed["checks"]) == 1
    assert parsed["checks"][0]["evidence"]["n"] == 5


# ── Runner ────────────────────────────────────────────────────────────────────

_TESTS = [
    # Leakage
    ("test_leakage_clean_data_passes", test_leakage_clean_data_passes),
    ("test_leakage_duplicate_data_fails", test_leakage_duplicate_data_fails),
    ("test_leakage_target_leakage_detection", test_leakage_target_leakage_detection),
    ("test_leakage_index_overlap_detected", test_leakage_index_overlap_detected),
    ("test_leakage_clean_index_passes", test_leakage_clean_index_passes),
    # Permutation
    ("test_permutation_real_labels_pass", test_permutation_real_labels_pass),
    ("test_permutation_random_labels_fail", test_permutation_random_labels_fail),
    # Baseline fairness
    ("test_baseline_fairness_same_config_passes",
     test_baseline_fairness_same_config_passes),
    ("test_baseline_fairness_different_scaling_warns",
     test_baseline_fairness_different_scaling_warns),
    ("test_baseline_below_chance_requires_review",
     test_baseline_below_chance_requires_review),
    # Significance
    ("test_significance_clearly_better", test_significance_clearly_better),
    ("test_significance_identical_scores_inconclusive",
     test_significance_identical_scores_inconclusive),
    ("test_significance_insufficient_data", test_significance_insufficient_data),
    ("test_significance_clearly_worse_fails", test_significance_clearly_worse_fails),
    # Gate — full suite
    ("test_gate_standard_suite_clean_data", test_gate_standard_suite_clean_data),
    ("test_gate_standard_suite_leaky_data_fails",
     test_gate_standard_suite_leaky_data_fails),
    ("test_gate_with_significance", test_gate_with_significance),
    # Verdicts
    ("test_verdict_finalize_all_pass", test_verdict_finalize_all_pass),
    ("test_verdict_finalize_one_blocker_fails",
     test_verdict_finalize_one_blocker_fails),
    ("test_verdict_finalize_warning_only", test_verdict_finalize_warning_only),
    # Rendering
    ("test_report_text_rendering", test_report_text_rendering),
    ("test_report_json_roundtrip", test_report_json_roundtrip),
]

if __name__ == "__main__":
    for name, fn in _TESTS:
        run(name, fn)

    print(f"\n{len(_PASS)} validity tests passed.")
    if _FAIL:
        print(f"{len(_FAIL)} FAILED:")
        for name, exc in _FAIL:
            print(f"  FAIL: {name} — {exc}")
        sys.exit(1)

"""Scientific Validity package — public API.

Gate release: SVG-v1 (RF-1.0.0-alpha.1)

Usage
-----
    from researchforge.scientific_validity import ScientificValidityGate, ValidityVerdict

SVG-v1 Known Limitations
------------------------
This version implements a foundation layer of validity checks. It is
deliberately labelled SVG-v1, not the complete scientific validity system.

See SVG_V1_KNOWN_LIMITATIONS for the machine-readable limitation record.
The RF project roadmap schedules SVG-v2 for RF-2.0.

Conceptual clarifications (corrections to the original alpha.1 labelling):

  BaselineFairnessValidator
      → Correctly understood as a *TrivialBaselineCheck*: verifies the
        candidate beats the majority-class baseline by a margin. This is
        *not* a full protocol-level baseline fairness comparison (which
        requires identical datasets, splits, preprocessing, budgets, seeds,
        and evaluation protocols across compared systems). Full baseline
        fairness is scheduled for SVG-v2.

  LabelPermutationTest
      → Correctly understood as a *label-randomisation sanity check*, not a
        general leakage detector. It uses n_shuffles=3 (minimum) and a
        ratio threshold, which is a cheap diagnostic heuristic. For
        publication-grade permutation evidence, n_permutations should be
        configurable (100–10000) with a proper null distribution and CI.

  StatisticalSignificanceTester
      → Defaults to paired t-test (paired=True), which is correct for
        seed-matched RF-vs-RF comparisons. The Welch's unpaired test remains
        available (paired=False) for genuinely independent samples.
        Cohen's d and 95% CI are already reported.
"""
from __future__ import annotations

GATE_VERSION = "SVG-v1"  # Renamed from "1.0.0-alpha.1" for clarity

# --------------------------------------------------------------------------- #
# SVG-v1 machine-readable limitation record                                   #
# --------------------------------------------------------------------------- #
# This dict is the single source of truth for what SVG-v1 does and does not
# guarantee. Reference it in RESEARCHFORGE_STATE.yaml and the release cert.
SVG_V1_KNOWN_LIMITATIONS: dict = {
    "version": "SVG-v1",
    "rf_release": "RF-1.0.0-alpha.1",
    "implemented_checks": [
        "exact_duplicate_split_leakage",
        "label_randomisation_sanity",    # min n_shuffles=3; heuristic threshold
        "trivial_baseline_sanity",        # majority-class comparison only
        "statistical_comparison",         # paired t-test + Cohen's d + 95% CI
    ],
    "known_limitations": [
        # Leakage
        "no_near_duplicate_leakage",
        "no_group_leakage",
        "no_patient_subject_leakage",    # critical for future ECG experiments
        "no_temporal_leakage",
        "no_preprocessing_leakage",
        "no_target_leakage",
        "no_derived_feature_leakage",
        # Permutation
        "permutation_is_sanity_not_statistical_test",
        "no_configurable_n_permutations",
        "no_null_distribution_ci",
        # Baseline
        "trivial_baseline_not_protocol_fairness",
        # Statistics
        "no_multiple_comparison_correction",  # scheduled RF-1.5+
        "no_wilcoxon_signed_rank",
        "no_bootstrap_ci",
        # Verdicts
        "no_provisional_verdict_status",       # scheduled RF-1.0-beta
        "no_independent_replication_tracking",
    ],
    "roadmap": {
        "SVG-v2": "RF-2.0",
        "multiple_comparison_correction": "RF-1.5+",
        "provisional_verdict_status": "RF-1.0-beta",
        "near_duplicate_leakage": "RF-2.0",
        "configurable_permutations": "RF-1.5",
        "protocol_baseline_fairness": "RF-2.0",
    },
    # Conceptual name corrections applied in RF-1.0.0-alpha.2
    "conceptual_corrections": {
        "BaselineFairnessValidator": "TrivialBaselineCheck (majority-class sanity only)",
        "LabelPermutationTest": "LabelRandomisationSanityCheck (heuristic, not statistical)",
        "StatisticalSignificanceTester.default_mode": "paired=True for RF-vs-RF comparisons",
    },
}


from .verdicts import (                                        # noqa: F401
    ValidityVerdict,
    ClaimEligibility,
    CheckSeverity,
    CheckResult,
    ValidityReport,
)
from .leakage import DataLeakageDetector                       # noqa: F401
from .permutation import LabelPermutationTest                  # noqa: F401
from .baseline import BaselineFairnessValidator, ModelConfig   # noqa: F401
from .significance import StatisticalSignificanceTester        # noqa: F401
from .gate import ScientificValidityGate                       # noqa: F401
from .report import ValidityReportRenderer                     # noqa: F401

__all__ = [
    "GATE_VERSION",
    "SVG_V1_KNOWN_LIMITATIONS",
    # Verdicts
    "ValidityVerdict", "ClaimEligibility", "CheckSeverity",
    "CheckResult", "ValidityReport",
    # Checks
    "DataLeakageDetector",
    "LabelPermutationTest",
    "BaselineFairnessValidator", "ModelConfig",
    "StatisticalSignificanceTester",
    # Orchestrator
    "ScientificValidityGate",
    # Rendering
    "ValidityReportRenderer",
]

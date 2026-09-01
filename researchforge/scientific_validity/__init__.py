"""Scientific Validity package — public API.

Gate version: 1.0.0-alpha.1

from researchforge.scientific_validity import ScientificValidityGate, ValidityVerdict
"""
from __future__ import annotations

GATE_VERSION = "1.0.0-alpha.1"

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

"""Validity verdict and report types for the Scientific Validity Gate.

Every validity check returns a CheckResult. The gate aggregates all
CheckResults into a ValidityReport with a single top-level ValidityVerdict.

Verdict semantics
-----------------
  PASS              All checks passed. Result is eligible for promotion.
  FAIL              At least one BLOCKER check failed. Result is blocked.
  WARNING           Non-blocking issues detected. Human should be aware.
  INCONCLUSIVE      Checks could not be completed (e.g., insufficient data).
  REQUIRES_HUMAN_REVIEW  Automated checks insufficient; human must inspect.

Claim eligibility
-----------------
Every ValidityReport carries a claim_eligibility field:
  EXECUTED          The experiment ran.
  VALIDATED         All automated checks passed (ValidityVerdict = PASS).
  REPLICATED        Result reproduced independently (future; not RF-1.0).
  CLAIM_ELIGIBLE    Validated + replicated (future; not RF-1.0).

This is used by the future documentation engine to distinguish experimental
evidence from validated claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ValidityVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    INCONCLUSIVE = "INCONCLUSIVE"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"


class ClaimEligibility(str, Enum):
    EXECUTED = "EXECUTED"           # experiment ran
    VALIDATED = "VALIDATED"         # automated checks passed
    REPLICATED = "REPLICATED"       # independently reproduced (future)
    CLAIM_ELIGIBLE = "CLAIM_ELIGIBLE"  # validated + replicated (future)


class CheckSeverity(str, Enum):
    BLOCKER = "BLOCKER"    # causes FAIL verdict
    WARNING = "WARNING"    # causes WARNING verdict at most
    INFO = "INFO"          # informational only


@dataclass
class CheckResult:
    """Result of one individual validity check."""
    check_name: str
    passed: bool
    severity: CheckSeverity
    verdict: ValidityVerdict
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    provenance: str = ""  # module + function that produced this result

    def is_blocker(self) -> bool:
        return not self.passed and self.severity == CheckSeverity.BLOCKER


@dataclass
class ValidityReport:
    """Aggregated result of all gate checks for one experiment result.

    The overall verdict is computed from individual check verdicts:
      - Any BLOCKER failure → FAIL
      - Any REQUIRES_HUMAN_REVIEW → REQUIRES_HUMAN_REVIEW (if no FAIL)
      - Any INCONCLUSIVE → INCONCLUSIVE (if no FAIL or HRR)
      - Any WARNING → WARNING (if all checks passed or are warnings)
      - All PASS → PASS
    """
    experiment_id: str
    checks: List[CheckResult] = field(default_factory=list)
    verdict: ValidityVerdict = ValidityVerdict.INCONCLUSIVE
    claim_eligibility: ClaimEligibility = ClaimEligibility.EXECUTED
    confidence: float = 0.0     # [0.0, 1.0] overall confidence in the verdict
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: str = ""

    def finalize(self) -> "ValidityReport":
        """Compute the top-level verdict from all check results.

        Call this after all checks have been appended to self.checks.
        """
        self.blockers = [c.description for c in self.checks if c.is_blocker()]
        self.warnings = [c.description for c in self.checks
                         if not c.passed and c.severity == CheckSeverity.WARNING]

        verdicts = {c.verdict for c in self.checks}

        if not self.checks:
            self.verdict = ValidityVerdict.INCONCLUSIVE
            self.confidence = 0.0
        elif ValidityVerdict.FAIL in verdicts:
            self.verdict = ValidityVerdict.FAIL
            n_pass = sum(1 for c in self.checks if c.passed)
            self.confidence = n_pass / len(self.checks)
        elif ValidityVerdict.REQUIRES_HUMAN_REVIEW in verdicts:
            self.verdict = ValidityVerdict.REQUIRES_HUMAN_REVIEW
            self.confidence = 0.5
        elif ValidityVerdict.INCONCLUSIVE in verdicts:
            self.verdict = ValidityVerdict.INCONCLUSIVE
            self.confidence = 0.5
        elif ValidityVerdict.WARNING in verdicts:
            self.verdict = ValidityVerdict.WARNING
            self.confidence = 0.8
        else:
            self.verdict = ValidityVerdict.PASS
            self.confidence = 1.0

        # Set claim eligibility based on verdict
        self.claim_eligibility = (
            ClaimEligibility.VALIDATED
            if self.verdict == ValidityVerdict.PASS
            else ClaimEligibility.EXECUTED)

        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "verdict": self.verdict.value,
            "claim_eligibility": self.claim_eligibility.value,
            "confidence": self.confidence,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "notes": self.notes,
            "checks": [
                {"name": c.check_name, "passed": c.passed,
                 "severity": c.severity.value, "verdict": c.verdict.value,
                 "description": c.description, "evidence": c.evidence,
                 "provenance": c.provenance}
                for c in self.checks
            ],
        }

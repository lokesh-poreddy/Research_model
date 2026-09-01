"""Human-readable rendering for ValidityReport.

Produces two formats:
  - Plain text (for logs and console output)
  - JSON (for API responses and archival)

The renderer is intentionally separate from the report itself so that the
report dataclass stays serialization-agnostic and the rendering logic can
be updated without changing the validity logic.
"""
from __future__ import annotations

import json as _json
from typing import Optional

from .verdicts import ValidityReport, ValidityVerdict


_VERDICT_ICONS = {
    ValidityVerdict.PASS: "✅",
    ValidityVerdict.FAIL: "❌",
    ValidityVerdict.WARNING: "⚠️ ",
    ValidityVerdict.INCONCLUSIVE: "🔲",
    ValidityVerdict.REQUIRES_HUMAN_REVIEW: "👁 ",
}


class ValidityReportRenderer:
    """Renders ValidityReport to human-readable text or JSON."""

    @staticmethod
    def to_text(report: ValidityReport, verbose: bool = False) -> str:
        """Render as a multi-line text summary.

        Parameters
        ----------
        verbose : bool
            If True, include evidence dicts for each check. Default False.
        """
        icon = _VERDICT_ICONS.get(report.verdict, "?")
        lines = [
            "═" * 70,
            f"  SCIENTIFIC VALIDITY REPORT — {report.experiment_id}",
            "═" * 70,
            f"  Verdict:           {icon} {report.verdict.value}",
            f"  Confidence:        {report.confidence:.0%}",
            f"  Claim Eligibility: {report.claim_eligibility.value}",
            f"  Checks run:        {len(report.checks)}",
            f"  Blockers:          {len(report.blockers)}",
            f"  Warnings:          {len(report.warnings)}",
            "─" * 70,
        ]

        for check in report.checks:
            status = "PASS" if check.passed else check.verdict.value
            icon_c = "✅" if check.passed else _VERDICT_ICONS.get(check.verdict, "?")
            lines.append(f"  {icon_c} [{check.severity.value:7s}] "
                          f"{check.check_name}: {status}")
            if not check.passed:
                lines.append(f"           → {check.description}")
            elif verbose:
                lines.append(f"           → {check.description}")
            if verbose and check.evidence:
                lines.append(f"           evidence: {check.evidence}")

        if report.blockers:
            lines.append("─" * 70)
            lines.append("  BLOCKERS:")
            for b in report.blockers:
                lines.append(f"    • {b}")

        if report.warnings:
            lines.append("─" * 70)
            lines.append("  WARNINGS:")
            for w in report.warnings:
                lines.append(f"    ~ {w}")

        if report.notes:
            lines.append("─" * 70)
            lines.append(f"  Notes: {report.notes}")

        lines.append("═" * 70)
        return "\n".join(lines)

    @staticmethod
    def to_json(report: ValidityReport, indent: int = 2) -> str:
        """Render as a JSON string for API responses and archival."""
        return _json.dumps(report.to_dict(), indent=indent, default=str)

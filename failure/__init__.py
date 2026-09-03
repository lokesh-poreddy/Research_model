LEGACY_STATUS = {
    "canonical": False,
    "replacement": "researchforge.diagnosis",
    "deprecated_since": "RF-1.0.0-alpha.2.1",
    "removal_target": None,  # preserved as historical/compatibility evidence
    "cross_imports_allowed": False,  # researchforge/ must never import from here
}

from failure.taxonomy import FailureCategory, REPAIR_ACTIONS
from failure.diagnosis import diagnose_failure
from failure.repair import apply_repair

__all__ = ["LEGACY_STATUS", "FailureCategory", "REPAIR_ACTIONS", "diagnose_failure", "apply_repair"]

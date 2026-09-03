LEGACY_STATUS = {
    "canonical": False,
    "replacement": "researchforge.config",
    "deprecated_since": "RF-1.0.0-alpha.2.1",
    "removal_target": None,  # preserved as historical/compatibility evidence
    "cross_imports_allowed": False,  # researchforge/ must never import from here
}

from config.settings import settings

__all__ = ["LEGACY_STATUS", "settings"]

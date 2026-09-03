LEGACY_STATUS = {
    "canonical": False,
    "replacement": "researchforge.memory.ecrm",
    "deprecated_since": "RF-1.0.0-alpha.2.1",
    "removal_target": None,  # preserved as historical/compatibility evidence
    "cross_imports_allowed": False,  # researchforge/ must never import from here
}

from ecrm.memory_store import ECRMMemoryStore, MemoryRecord
from ecrm.res_scorer import compute_res, memory_utility
from ecrm.negative_transfer import NTRDetector

__all__ = [
    "LEGACY_STATUS",
    "ECRMMemoryStore",
    "MemoryRecord",
    "compute_res",
    "memory_utility",
    "NTRDetector",
]

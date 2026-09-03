LEGACY_STATUS = {
    "canonical": False,
    "replacement": "researchforge.retrieval",
    "deprecated_since": "RF-1.0.0-alpha.2.1",
    "removal_target": None,  # preserved as historical/compatibility evidence
    "cross_imports_allowed": False,  # researchforge/ must never import from here
}

from tools.openalex import search_papers as openalex_search
from tools.semantic_scholar import search_papers as s2_search
from tools.arxiv_search import search_papers as arxiv_search
from tools.mlflow_tracker import MLflowTracker

__all__ = ["LEGACY_STATUS", "openalex_search", "s2_search", "arxiv_search", "MLflowTracker"]

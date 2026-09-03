LEGACY_STATUS = {
    "canonical": False,
    "replacement": "researchforge.pipeline.controller",
    "deprecated_since": "RF-1.0.0-alpha.2.1",
    "removal_target": None,  # preserved as historical/compatibility evidence
    "cross_imports_allowed": False,  # researchforge/ must never import from here
}

from agents.base_agent import BaseAgent
from agents.controller_agent import ResearchController
from agents.hypothesis_agent import HypothesisAgent
from agents.experiment_agent import ExperimentAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.manuscript_agent import ManuscriptAgent

__all__ = [
    "LEGACY_STATUS",
    "BaseAgent",
    "ResearchController",
    "HypothesisAgent",
    "ExperimentAgent",
    "AnalyzerAgent",
    "ManuscriptAgent",
]

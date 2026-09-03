LEGACY_STATUS = {
    "canonical": False,
    "replacement": "researchforge.genome",
    "deprecated_since": "RF-1.0.0-alpha.2.1",
    "removal_target": None,  # preserved as historical/compatibility evidence
    "cross_imports_allowed": False,  # researchforge/ must never import from here
}

from evolution.genome import ModelGenome
from evolution.mutate import param_mutation, random_mutation
from evolution.crossover import crossover
from evolution.operators import OperatorType, apply_operator, OPERATOR_REGISTRY

__all__ = [
    "LEGACY_STATUS",
    "ModelGenome",
    "param_mutation",
    "random_mutation",
    "crossover",
    "OperatorType",
    "apply_operator",
    "OPERATOR_REGISTRY",
]

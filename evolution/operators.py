"""
Operator registry and compound synthesis operator.
"""
from __future__ import annotations

import random
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from evolution.genome import ModelGenome
from evolution.mutate import (
    augmentation_mutation,
    optimizer_mutation,
    param_mutation,
    random_mutation,
    structure_mutation_add_layer,
    structure_mutation_remove_layer,
)
from evolution.crossover import crossover
from evolution.strategy_mutation import strategy_mutation


class OperatorType(str, Enum):
    PARAM_MUTATION = "param_mutation"
    OPTIMIZER_MUTATION = "optimizer_mutation"
    ADD_LAYER = "add_layer"
    REMOVE_LAYER = "remove_layer"
    AUGMENTATION = "augmentation"
    STRATEGY = "strategy"
    CROSSOVER = "crossover"
    SYNTHESIS = "synthesis"


# Operator registry: name → (function, cost_estimate_relative)
OPERATOR_REGISTRY: Dict[OperatorType, Tuple[Callable, float]] = {
    OperatorType.PARAM_MUTATION:      (param_mutation, 0.1),
    OperatorType.OPTIMIZER_MUTATION:  (optimizer_mutation, 0.1),
    OperatorType.ADD_LAYER:           (structure_mutation_add_layer, 0.2),
    OperatorType.REMOVE_LAYER:        (structure_mutation_remove_layer, 0.2),
    OperatorType.AUGMENTATION:        (augmentation_mutation, 0.05),
    OperatorType.STRATEGY:            (strategy_mutation, 0.05),
}


def apply_operator(
    op_type: OperatorType,
    genome: ModelGenome,
    partner: Optional[ModelGenome] = None,
    delta: float = 0.1,
) -> ModelGenome:
    """Apply a registered operator to a genome."""
    if op_type == OperatorType.CROSSOVER:
        if partner is None:
            raise ValueError("Crossover requires a partner genome.")
        return crossover(genome, partner)
    if op_type == OperatorType.SYNTHESIS:
        return _synthesize(genome, delta)

    fn, _ = OPERATOR_REGISTRY[op_type]
    if op_type == OperatorType.PARAM_MUTATION:
        return fn(genome, delta)
    return fn(genome)


def _synthesize(genome: ModelGenome, delta: float = 0.1) -> ModelGenome:
    """
    Compound synthesis: apply two random operators in sequence.
    Implements the AlgorithmDiscovery trigger when stuck.
    """
    child = random_mutation(genome, delta)
    child = random_mutation(child, delta)
    return child


def operator_cost(op_type: OperatorType) -> float:
    """Return relative compute cost estimate."""
    if op_type in (OperatorType.CROSSOVER, OperatorType.SYNTHESIS):
        return 1.0  # highest
    return OPERATOR_REGISTRY.get(op_type, (None, 0.5))[1]

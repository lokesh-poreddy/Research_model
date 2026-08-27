"""
Operator registry and compound synthesis operator — v2.

v2 changes:
- ``SYNTHESIS`` is now a first-class entry in ``OPERATOR_REGISTRY`` with
  cost 1.0 so ``operator_cost()`` returns the correct value.
- ``_synthesize`` pulls two *distinct* strategy families from a
  ``StrategyPortfolio`` (if provided) instead of calling ``random_mutation``
  twice blindly. Falls back to two random mutations when no portfolio is given.
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
    MUTATION_OPERATORS,
)
from evolution.crossover import crossover
from evolution.strategy_mutation import strategy_mutation


class OperatorType(str, Enum):
    PARAM_MUTATION = "param_mutation"
    OPTIMIZER_MUTATION = "optimizer_mutation"
    ADD_LAYER = "structure_add"
    REMOVE_LAYER = "structure_remove"
    AUGMENTATION = "augmentation_mutation"
    STRATEGY = "strategy"
    CROSSOVER = "crossover"
    SYNTHESIS = "synthesis"


# Operator registry: OperatorType → (function, relative_cost)
OPERATOR_REGISTRY: Dict[OperatorType, Tuple[Callable, float]] = {
    OperatorType.PARAM_MUTATION:     (param_mutation, 0.1),
    OperatorType.OPTIMIZER_MUTATION: (optimizer_mutation, 0.1),
    OperatorType.ADD_LAYER:          (structure_mutation_add_layer, 0.2),
    OperatorType.REMOVE_LAYER:       (structure_mutation_remove_layer, 0.2),
    OperatorType.AUGMENTATION:       (augmentation_mutation, 0.05),
    OperatorType.STRATEGY:           (strategy_mutation, 0.05),
    # v2: SYNTHESIS is a first-class cost-weighted entry
    OperatorType.SYNTHESIS:          (_synthesize_placeholder := (lambda g: g), 1.0),
}


def apply_operator(
    op_type: OperatorType,
    genome: ModelGenome,
    partner: Optional[ModelGenome] = None,
    delta: float = 0.1,
    portfolio: Optional[object] = None,
) -> ModelGenome:
    """Apply a registered operator to a genome.

    Args:
        op_type: Operator to apply.
        genome: Source genome.
        partner: Required for CROSSOVER.
        delta: Mutation magnitude.
        portfolio: Optional StrategyPortfolio for guided SYNTHESIS.
    """
    if op_type == OperatorType.CROSSOVER:
        if partner is None:
            raise ValueError("Crossover requires a partner genome.")
        return crossover(genome, partner)
    if op_type == OperatorType.SYNTHESIS:
        return _synthesize(genome, delta, portfolio=portfolio)

    fn, _ = OPERATOR_REGISTRY[op_type]
    if op_type == OperatorType.PARAM_MUTATION:
        return fn(genome, delta)
    return fn(genome)


def _synthesize(
    genome: ModelGenome,
    delta: float = 0.1,
    portfolio: Optional[object] = None,
) -> ModelGenome:
    """
    Compound synthesis: apply two operators from distinct strategy families.

    When a StrategyPortfolio is provided, the two operators are drawn from
    the two most under-explored families so the discovery step actually
    diversifies rather than repeating the same family twice.
    """
    if portfolio is not None:
        # Collect all strategy_id → family mappings
        family_map: Dict[str, str] = getattr(portfolio, "strategies", {})
        family_evidence = getattr(portfolio, "evidence", {})

        # Sort by trials ascending (least explored first) and pick two distinct families
        sorted_ids = sorted(family_evidence.keys(), key=lambda s: family_evidence[s].trials)
        chosen: List[str] = []
        seen_families: set = set()
        for sid in sorted_ids:
            fam = family_map.get(sid, "unknown")
            if fam not in seen_families:
                chosen.append(sid)
                seen_families.add(fam)
            if len(chosen) == 2:
                break
        # Pad with random choices if fewer than 2 families exist
        while len(chosen) < 2:
            chosen.append(random.choice(list(MUTATION_OPERATORS.keys())))

        child = random_mutation(genome, delta, operator_hint=chosen[0])
        child = random_mutation(child, delta, operator_hint=chosen[1])
        child.strategy_description = f"synthesis:{'+'.join(chosen)}"
        return child

    # Fallback: two random mutations (different operators when possible)
    ops = list(MUTATION_OPERATORS.keys())
    op1 = random.choice(ops)
    op2 = random.choice([o for o in ops if o != op1] or ops)
    child = random_mutation(genome, delta, operator_hint=op1)
    child = random_mutation(child, delta, operator_hint=op2)
    child.strategy_description = f"synthesis:{op1}+{op2}"
    return child


# Fix the placeholder in the registry now that _synthesize is defined
OPERATOR_REGISTRY[OperatorType.SYNTHESIS] = (_synthesize, 1.0)


def operator_cost(op_type: OperatorType) -> float:
    """Return relative compute cost estimate."""
    return OPERATOR_REGISTRY.get(op_type, (None, 0.5))[1]

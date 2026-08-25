"""
Strategy-level mutation operator.
Modifies the natural-language strategy description of a genome,
used by SeaEvo-style strategy-space evolution.
"""
from __future__ import annotations

import random
import uuid
from typing import List

from evolution.genome import ModelGenome

STRATEGY_REFINEMENTS = [
    "increase regularization",
    "use adaptive learning rate schedule",
    "add skip connections",
    "try label smoothing",
    "switch to cosine annealing",
    "apply gradient clipping",
    "use mixed precision training",
    "add warmup epochs",
    "increase data diversity",
    "explore depth-wise separable convolutions",
    "apply knowledge distillation",
    "use stochastic depth",
]


def strategy_mutation(genome: ModelGenome) -> ModelGenome:
    """
    Append a random refinement directive to the strategy description.
    This steers the LLM during subsequent hypothesis generation.
    """
    child = genome.copy()
    child.model_id = str(uuid.uuid4())
    child.parent_id = genome.model_id
    child.generation = genome.generation + 1

    refinement = random.choice(STRATEGY_REFINEMENTS)
    if child.strategy_description:
        child.strategy_description = child.strategy_description + f"; {refinement}"
    else:
        child.strategy_description = refinement
    return child


def strategy_crossover(genome_a: ModelGenome, genome_b: ModelGenome) -> str:
    """
    Combine the strategy descriptions of two parent genomes.
    Useful for generating new LLM prompts from proven strategies.
    """
    parts_a = genome_a.strategy_description.split(";")
    parts_b = genome_b.strategy_description.split(";")
    combined = list(set(parts_a + parts_b))
    return "; ".join(s.strip() for s in combined if s.strip())

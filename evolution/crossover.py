"""
Crossover operators for Model Genomes.
Produces offspring by combining two parent genomes.
"""
from __future__ import annotations

import copy
import random
import uuid
from typing import List, Tuple

from evolution.genome import ModelGenome


def single_point_crossover(parent_a: ModelGenome, parent_b: ModelGenome) -> ModelGenome:
    """
    Combine hyperparameters from A and architecture layers from B (or vice versa)
    with 50/50 probability.  Cost: O(n).
    """
    child = parent_a.copy()
    child.model_id = str(uuid.uuid4())
    child.parent_id = f"{parent_a.model_id}+{parent_b.model_id}"
    child.generation = max(parent_a.generation, parent_b.generation) + 1

    # Hyperparameter crossover: random choice per key
    for key in parent_a.hyperparameters:
        if key in parent_b.hyperparameters and random.random() < 0.5:
            child.hyperparameters[key] = copy.deepcopy(parent_b.hyperparameters[key])

    # Architectural crossover: swap layers at a random point
    layers_a = parent_a.architecture.get("layers", [])
    layers_b = parent_b.architecture.get("layers", [])
    if layers_a and layers_b:
        cut = random.randint(1, min(len(layers_a), len(layers_b)) - 1)
        new_layers = layers_a[:cut] + layers_b[cut:]
        child.architecture["layers"] = [copy.deepcopy(l) for l in new_layers]

    child.strategy_description = (
        f"crossover({parent_a.model_id[:6]}, {parent_b.model_id[:6]})"
    )
    return child


def uniform_crossover(parent_a: ModelGenome, parent_b: ModelGenome) -> ModelGenome:
    """
    Each layer is independently selected from parent A or B with equal probability.
    """
    child = parent_a.copy()
    child.model_id = str(uuid.uuid4())
    child.parent_id = f"{parent_a.model_id}+{parent_b.model_id}"
    child.generation = max(parent_a.generation, parent_b.generation) + 1

    layers_a = parent_a.architecture.get("layers", [])
    layers_b = parent_b.architecture.get("layers", [])
    length = max(len(layers_a), len(layers_b))

    new_layers = []
    for i in range(length):
        if i < len(layers_a) and i < len(layers_b):
            src = layers_a[i] if random.random() < 0.5 else layers_b[i]
        elif i < len(layers_a):
            src = layers_a[i]
        else:
            src = layers_b[i]
        new_layers.append(copy.deepcopy(src))

    child.architecture["layers"] = new_layers
    return child


def crossover(genome1: ModelGenome, genome2: ModelGenome, mode: str = "single") -> ModelGenome:
    """Dispatch to appropriate crossover mode."""
    if mode == "uniform":
        return uniform_crossover(genome1, genome2)
    return single_point_crossover(genome1, genome2)

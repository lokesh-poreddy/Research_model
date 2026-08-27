"""
Mutation operators for Model Genomes.
Each operator returns a NEW genome (immutable pattern).
"""
from __future__ import annotations

import copy
import random
import uuid
from typing import Any, Dict, List, Optional

from evolution.genome import ModelGenome

# Available layer types for structural mutations
LAYER_TYPES = [
    {"type": "Conv2D", "filters": 64, "kernel": 3},
    {"type": "Conv2D", "filters": 128, "kernel": 3},
    {"type": "BatchNorm"},
    {"type": "ReLU"},
    {"type": "Dropout", "rate": 0.3},
    {"type": "Dense", "units": 128},
    {"type": "Dense", "units": 256},
    {"type": "MaxPool"},
    {"type": "GlobalAvgPool"},
    {"type": "GELU"},
    {"type": "LayerNorm"},
]

OPTIMIZERS = ["Adam", "SGD", "AdamW", "RMSprop", "Adagrad"]

AUGMENTATIONS = [
    "flip", "crop", "rotate", "color_jitter",
    "cutout", "mixup", "random_erasing",
]


def param_mutation(genome: ModelGenome, delta: float = 0.1) -> ModelGenome:
    """
    Randomly perturb one numeric hyperparameter by ±delta fraction.
    Cost: O(1) – very cheap.
    """
    child = genome.copy()
    child.model_id = str(uuid.uuid4())
    child.parent_id = genome.model_id
    child.generation = genome.generation + 1

    mutable = {
        k: v for k, v in child.hyperparameters.items()
        if isinstance(v, (int, float))
    }
    if not mutable:
        return child

    param = random.choice(list(mutable.keys()))
    val = mutable[param]
    noise = 1.0 + random.gauss(0, delta)
    new_val = val * noise

    # clamp to sensible bounds
    if param == "learning_rate":
        new_val = max(1e-7, min(1.0, new_val))
    elif param == "batch_size":
        new_val = max(4, min(512, int(round(new_val))))
    elif param == "dropout_rate":
        new_val = max(0.0, min(0.9, new_val))
    elif param == "epochs":
        new_val = max(1, min(500, int(round(new_val))))

    # Integer rounding can otherwise produce a no-op mutation (notably for
    # epochs or batch size).  Evolution operators must always create a
    # distinct candidate when a mutable parameter exists.
    if new_val == val:
        if param in ("batch_size", "epochs"):
            new_val = int(min(512 if param == "batch_size" else 500, val + 1))
        else:
            new_val = float(val) * (1.0 + delta) if val else max(1e-6, delta)

    child.hyperparameters[param] = new_val
    child.strategy_description = (
        genome.strategy_description + f" | param_mut({param}: {val:.4g}→{new_val:.4g})"
    )
    return child


def optimizer_mutation(genome: ModelGenome) -> ModelGenome:
    """Switch to a randomly chosen optimizer."""
    child = genome.copy()
    child.model_id = str(uuid.uuid4())
    child.parent_id = genome.model_id
    child.generation = genome.generation + 1

    current = child.hyperparameters.get("optimizer", "Adam")
    alternatives = [o for o in OPTIMIZERS if o != current]
    child.hyperparameters["optimizer"] = random.choice(alternatives)
    return child


def structure_mutation_add_layer(genome: ModelGenome) -> ModelGenome:
    """
    Insert a new random layer at a random position.
    Cost: O(n) copy.
    """
    child = genome.copy()
    child.model_id = str(uuid.uuid4())
    child.parent_id = genome.model_id
    child.generation = genome.generation + 1

    layers = child.architecture.get("layers", [])
    if len(layers) >= 20:  # cap
        return child

    new_layer = copy.deepcopy(random.choice(LAYER_TYPES))
    # Insert before the last Dense/output layer
    insert_idx = max(0, len(layers) - 1)
    layers.insert(insert_idx, new_layer)
    child.architecture["layers"] = layers
    child.strategy_description = genome.strategy_description + f" | add_layer({new_layer['type']})"
    return child


def structure_mutation_remove_layer(genome: ModelGenome) -> ModelGenome:
    """Remove a non-essential layer."""
    child = genome.copy()
    child.model_id = str(uuid.uuid4())
    child.parent_id = genome.model_id
    child.generation = genome.generation + 1

    layers = child.architecture.get("layers", [])
    removable_idxs = [
        i for i, lyr in enumerate(layers)
        if lyr.get("type") not in ("Dense",) or i < len(layers) - 1
    ]
    if len(layers) <= 2 or not removable_idxs:
        return child

    idx = random.choice(removable_idxs[:-1] if removable_idxs else removable_idxs)
    removed = layers.pop(idx)
    child.architecture["layers"] = layers
    child.strategy_description = genome.strategy_description + f" | rm_layer({removed['type']}@{idx})"
    return child


def augmentation_mutation(genome: ModelGenome) -> ModelGenome:
    """Add or remove a data augmentation."""
    child = genome.copy()
    child.model_id = str(uuid.uuid4())
    child.parent_id = genome.model_id
    child.generation = genome.generation + 1

    current_augs: List[str] = list(child.data_settings.get("augmentations", []))
    if random.random() < 0.5 and len(current_augs) < len(AUGMENTATIONS):
        # add
        new_aug = random.choice([a for a in AUGMENTATIONS if a not in current_augs])
        current_augs.append(new_aug)
    elif current_augs:
        # remove
        current_augs.remove(random.choice(current_augs))
    child.data_settings["augmentations"] = current_augs
    return child


MUTATION_OPERATORS = {
    "param_mutation": param_mutation,
    "optimizer_mutation": optimizer_mutation,
    "structure_add": structure_mutation_add_layer,
    "structure_remove": structure_mutation_remove_layer,
    "augmentation_mutation": augmentation_mutation,
}


def random_mutation(
    genome: ModelGenome, delta: float = 0.1, operator_hint: Optional[str] = None
) -> ModelGenome:
    """Apply a randomly chosen or policy-selected mutation operator."""
    if operator_hint is not None and operator_hint not in MUTATION_OPERATORS:
        raise ValueError(f"Unknown mutation operator: {operator_hint}")
    op = MUTATION_OPERATORS[operator_hint] if operator_hint else random.choice(list(MUTATION_OPERATORS.values()))
    if op == param_mutation:
        child = op(genome, delta)
    else:
        child = op(genome)
    if not child.strategy_description:
        child.strategy_description = operator_hint or next(
            name for name, candidate in MUTATION_OPERATORS.items() if candidate == op
        )
    return child

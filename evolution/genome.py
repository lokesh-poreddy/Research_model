"""
Model Genome: structured JSON description of a candidate model.
Serves as the "DNA" that evolution operators act on.
"""
from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jsonschema

# JSON Schema for validation
GENOME_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Model Genome Schema",
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "parent_id": {"type": ["string", "null"]},
        "generation": {"type": "integer", "minimum": 0},
        "architecture": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "layers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                        },
                        "required": ["type"],
                    },
                },
            },
            "required": ["type", "layers"],
        },
        "hyperparameters": {
            "type": "object",
            "properties": {
                "learning_rate": {"type": "number", "minimum": 1e-8, "maximum": 10.0},
                "batch_size": {"type": "integer", "minimum": 1},
                "optimizer": {"type": "string"},
                "epochs": {"type": "integer", "minimum": 1},
                "weight_decay": {"type": "number"},
                "dropout_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        },
        "data_settings": {
            "type": "object",
            "properties": {
                "augmentations": {"type": "array", "items": {"type": "string"}},
                "normalization": {"type": "boolean"},
                "input_size": {"type": "array", "items": {"type": "integer"}},
            },
        },
        "seed": {"type": "integer"},
        "library_version": {"type": "string"},
        "strategy_description": {"type": "string"},
        "fingerprint": {"type": "string"},
    },
    "required": ["model_id", "architecture", "hyperparameters"],
}


@dataclass
class ModelGenome:
    """Represents a candidate model's full specification."""

    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    generation: int = 0
    architecture: Dict[str, Any] = field(default_factory=lambda: {
        "type": "CNN",
        "layers": [
            {"type": "Conv2D", "filters": 32, "kernel": 3},
            {"type": "BatchNorm"},
            {"type": "ReLU"},
            {"type": "GlobalAvgPool"},
            {"type": "Dense", "units": 10},
        ],
    })
    hyperparameters: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": 0.001,
        "batch_size": 64,
        "optimizer": "Adam",
        "epochs": 10,
        "weight_decay": 1e-4,
        "dropout_rate": 0.0,
    })
    data_settings: Dict[str, Any] = field(default_factory=lambda: {
        "augmentations": ["flip", "crop"],
        "normalization": True,
        "input_size": [32, 32, 3],
    })
    seed: int = 42
    library_version: str = "torch>=2.0.0"
    strategy_description: str = ""

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> bool:
        """Validate against JSON schema; raises jsonschema.ValidationError on failure."""
        jsonschema.validate(instance=self.to_dict(), schema=GENOME_SCHEMA)
        return True

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "architecture": self.architecture,
            "hyperparameters": self.hyperparameters,
            "data_settings": self.data_settings,
            "seed": self.seed,
            "library_version": self.library_version,
            "strategy_description": self.strategy_description,
            "fingerprint": self.fingerprint(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelGenome":
        return cls(
            model_id=d.get("model_id", str(uuid.uuid4())),
            parent_id=d.get("parent_id"),
            generation=d.get("generation", 0),
            architecture=d.get("architecture", {}),
            hyperparameters=d.get("hyperparameters", {}),
            data_settings=d.get("data_settings", {}),
            seed=d.get("seed", 42),
            library_version=d.get("library_version", ""),
            strategy_description=d.get("strategy_description", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ModelGenome":
        return cls.from_dict(json.loads(json_str))

    # ── Utilities ─────────────────────────────────────────────────────────────

    def copy(self) -> "ModelGenome":
        return ModelGenome.from_dict(copy.deepcopy(self.to_dict()))

    def fingerprint(self) -> str:
        """Stable hash of the genome for deduplication."""
        import hashlib
        raw = json.dumps(
            {
                "arch": self.architecture,
                "hp": {k: round(v, 6) if isinstance(v, float) else v
                       for k, v in self.hyperparameters.items()},
            },
            sort_keys=True,
        )
        return hashlib.md5(raw.encode()).hexdigest()

    def __repr__(self) -> str:
        arch_type = self.architecture.get("type", "?")
        n_layers = len(self.architecture.get("layers", []))
        lr = self.hyperparameters.get("learning_rate", "?")
        return f"ModelGenome(arch={arch_type}, layers={n_layers}, lr={lr}, gen={self.generation})"

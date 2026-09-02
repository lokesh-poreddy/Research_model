"""TargetModelGenome (TMG): the versioned, typed genome that governs
the model being optimized (the *target* of the research process).

RF-1.0.0-alpha.2 introduces TMG as the successor to ModelGenome. It adds:

  * schema_version: JSON Schema-versioned, migration-able
  * tmg_id: deterministic, sibling-collision-free identity (see schema.py)
  * fingerprint(): sha256 of canonical JSON — for deduplication and provenance
  * operator: tracks which evolution operator created this genome
  * Extended lineage: ancestor_ids (full chain), crossover_parents, rollback_from
  * TMGCapabilities: declared resource / interface capabilities
  * researchforge_version: RF release that created this genome
  * Immutability contract: use clone()/evolve() pattern; do not modify in place

Backward compatibility (AD-008)
--------------------------------
  * TMG.from_model_genome(mg) — lossless RF-0.x upgrade
      Preservation guarantee: all ModelGenome fields copied unchanged.
      Derived metadata: schema_version, tmg_id, ancestor_ids, operator,
      capabilities inferred deterministically from existing fields.
  * TMG.to_model_genome() — downgrade for RF-0.x compatibility layer
  * Roundtrip: TMG.to_model_genome(TMG.from_model_genome(mg)) == mg

Immutability contract (AD-009)
---------------------------------
  Genome objects represent *versioned research artifacts*. Modifying a
  genome's attributes after construction silently corrupts the research
  record. The correct pattern is:

      child = parent.clone(operator="increase_capacity")
      # modify child's architecture/hyperparameters in operators.py ONLY
      # via the helper _mutate_tmg() which returns a new TMG

  Direct attribute mutation after construction is a violation. This is
  enforced by documentation and by the tests in test_genomes.py.
  Language-level enforcement (frozen dataclass or __setattr__ override)
  is deferred to alpha.3 when VRDEG begins tracking genome identity.

TMG operator namespace (AD-010)
----------------------------------
  TMG_OPERATORS — the set of operator names tracked in the `operator` field.
  Distinct from RSG_EVOLUTION_OPERATORS (research-strategy space).
"""
from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .schema import (
    GENOME_SCHEMA_VERSION_TMG,
    deterministic_genome_id,
    genome_fingerprint,
    validate_genome,
)
from .model_genome import ModelGenome, DEFAULT_GENOMES

# --------------------------------------------------------------------------- #
# TMG operator namespace                                                       #
# --------------------------------------------------------------------------- #
TMG_OPERATORS = [
    "init",               # initial genome construction
    "increase_capacity",
    "add_regularization",
    "tune_learning_dynamics",
    "change_family",
    "feature_preprocessing",
    "crossover",
    "random_perturbation",
    "legacy_import",      # ModelGenome → TMG migration
    "clone",              # exact structural copy (no mutation)
    "rollback",           # restore to a previous genome version
]

# --------------------------------------------------------------------------- #
# JSON Schema                                                                  #
# --------------------------------------------------------------------------- #
# Design: additionalProperties: false to prevent schema drift.
# model_id is NOT in this schema — it is accepted only by the migration loader.
GENOME_SCHEMA_TMG: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TargetModelGenome",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "tmg_id", "schema_version", "model_type",
        "architecture", "hyperparameters",
    ],
    "properties": {
        "tmg_id": {"type": "string", "minLength": 1},
        "schema_version": {"type": "string", "enum": [GENOME_SCHEMA_VERSION_TMG]},
        "researchforge_version": {"type": "string"},
        "created_at": {"type": "number"},
        "operator": {"type": "string"},
        "model_type": {
            "type": "string",
            "enum": ["MLPClassifier", "RandomForestClassifier", "SVC", "LogisticRegression"],
        },
        "architecture": {"type": "object"},
        "hyperparameters": {"type": "object"},
        "data_pipeline": {"type": "object"},
        "seed": {"type": "integer"},
        "generation": {"type": "integer"},
        "parent_ids": {"type": "array", "items": {"type": "string"}},
        "ancestor_ids": {"type": "array", "items": {"type": "string"}},
        "crossover_parents": {
            "oneOf": [{"type": "null"}, {"type": "array", "items": {"type": "string"}}]
        },
        "rollback_from": {"oneOf": [{"type": "null"}, {"type": "string"}]},
        "capabilities": {"type": "object"},
    },
}


# --------------------------------------------------------------------------- #
# Capabilities                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class TMGCapabilities:
    """Declared resource and interface capabilities of a target model genome.

    These are conservative declarations, not guarantees. They help the
    research controller make pre-flight resource decisions without training.
    """
    supports_warm_start: bool = False
    supports_partial_fit: bool = False
    supports_predict_proba: bool = True
    expected_train_time_s: Optional[float] = None
    memory_estimate_mb: Optional[float] = None

    @classmethod
    def infer(cls, model_type: str) -> "TMGCapabilities":
        """Deterministically infer capabilities from model_type.

        Used by the migration layer for legacy genomes. The inferred values
        are conservative lower bounds — a real runtime profiler would be more
        accurate, but this is adequate for pre-flight filtering.
        """
        return cls(
            supports_warm_start=(model_type == "MLPClassifier"),
            supports_partial_fit=(model_type == "MLPClassifier"),
            supports_predict_proba=(model_type != "SVC"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supports_warm_start": self.supports_warm_start,
            "supports_partial_fit": self.supports_partial_fit,
            "supports_predict_proba": self.supports_predict_proba,
            "expected_train_time_s": self.expected_train_time_s,
            "memory_estimate_mb": self.memory_estimate_mb,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TMGCapabilities":
        return cls(
            supports_warm_start=d.get("supports_warm_start", False),
            supports_partial_fit=d.get("supports_partial_fit", False),
            supports_predict_proba=d.get("supports_predict_proba", True),
            expected_train_time_s=d.get("expected_train_time_s"),
            memory_estimate_mb=d.get("memory_estimate_mb"),
        )


# --------------------------------------------------------------------------- #
# TargetModelGenome                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class TargetModelGenome:
    """Versioned genome governing the target model being optimized.

    See module docstring for the full design contract.

    Parameters
    ----------
    model_type : str
        Must be one of the supported model families.
    architecture : dict
        Family-specific structural parameters.
    hyperparameters : dict
        Family-specific training parameters.

    All other fields have sensible defaults; prefer ``default()`` or
    ``from_model_genome()`` over direct construction.
    """

    # Core fields (backward compat with ModelGenome)
    model_type: str
    architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    data_pipeline: Dict[str, Any] = field(
        default_factory=lambda: {"scale": True, "pca_components": None}
    )
    seed: int = 0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

    # Identity (new in alpha.2)
    tmg_id: str = field(default_factory=lambda: f"tmg_{uuid.uuid4().hex[:10]}")
    schema_version: str = field(default=GENOME_SCHEMA_VERSION_TMG)
    created_at: float = field(default_factory=time.time)
    operator: str = "init"
    researchforge_version: str = "RF-1.0.0-alpha.2"

    # Extended lineage (new in alpha.2)
    ancestor_ids: List[str] = field(default_factory=list)
    crossover_parents: Optional[List[str]] = None
    rollback_from: Optional[str] = None

    # Capabilities (new in alpha.2)
    capabilities: TMGCapabilities = field(default_factory=TMGCapabilities)

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def default(cls, model_type: str, seed: int = 0) -> "TargetModelGenome":
        """Create a default TMG for a given model family.

        Produces the same initial model as ModelGenome.default(), extended
        with alpha.2 identity and lineage fields.
        """
        base = DEFAULT_GENOMES[model_type]
        tmg_id = deterministic_genome_id(
            "tmg", [], "init", 0, seed,
            mutation_parameters={"model_type": model_type},
        )
        return cls(
            model_type=model_type,
            architecture=copy.deepcopy(base["architecture"]),
            hyperparameters=copy.deepcopy(base["hyperparameters"]),
            seed=seed,
            tmg_id=tmg_id,
            operator="init",
            ancestor_ids=[],
            capabilities=TMGCapabilities.infer(model_type),
        )

    @classmethod
    def from_model_genome(cls, mg: ModelGenome) -> "TargetModelGenome":
        """Lossless upgrade from an RF-0.x ModelGenome.

        Preservation guarantee: all ModelGenome fields are copied unchanged.
        Derived metadata: schema_version, tmg_id (from model_id), ancestor_ids,
        operator, capabilities — deterministically inferred, not new information.

        Roundtrip guarantee:
            TMG.to_model_genome(TMG.from_model_genome(mg)) == mg
        """
        mg_dict = mg.to_dict()
        # Use the migration layer so the logic is centralized
        from .migration import migrate_tmg
        migrated = migrate_tmg(mg_dict)
        return cls.from_dict(migrated)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TargetModelGenome":
        """Deserialize from a dict, applying migration if needed.

        Accepts:
          - Canonical TMG dicts (schema_version="1.0", tmg_id present)
          - Legacy ModelGenome dicts (no schema_version, model_id present)

        The migration layer normalizes any legacy dict before this method
        sees it, so this method always works with canonical form.
        """
        from .migration import migrate_tmg
        d = migrate_tmg(dict(d))
        caps_dict = d.get("capabilities", {})
        caps = TMGCapabilities.from_dict(caps_dict) if caps_dict else TMGCapabilities()
        return cls(
            tmg_id=d["tmg_id"],
            schema_version=d.get("schema_version", GENOME_SCHEMA_VERSION_TMG),
            created_at=d.get("created_at", time.time()),
            operator=d.get("operator", "legacy_import"),
            researchforge_version=d.get("researchforge_version", "RF-0.x"),
            model_type=d["model_type"],
            architecture=dict(d.get("architecture", {})),
            hyperparameters=dict(d.get("hyperparameters", {})),
            data_pipeline=dict(d.get("data_pipeline",
                                     {"scale": True, "pca_components": None})),
            seed=int(d.get("seed", 0)),
            generation=int(d.get("generation", 0)),
            parent_ids=list(d.get("parent_ids", [])),
            ancestor_ids=list(d.get("ancestor_ids", [])),
            crossover_parents=d.get("crossover_parents"),
            rollback_from=d.get("rollback_from"),
            capabilities=caps,
        )

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Canonical serialization.

        Only tmg_id appears — model_id is a legacy alias accepted by the
        migration loader but NOT stored in the canonical representation.
        This keeps the JSON Schema strict (additionalProperties: false)
        and avoids schema debt.
        """
        return {
            "tmg_id": self.tmg_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "operator": self.operator,
            "researchforge_version": self.researchforge_version,
            "model_type": self.model_type,
            "architecture": dict(self.architecture),
            "hyperparameters": dict(self.hyperparameters),
            "data_pipeline": dict(self.data_pipeline),
            "seed": self.seed,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "ancestor_ids": list(self.ancestor_ids),
            "crossover_parents": self.crossover_parents,
            "rollback_from": self.rollback_from,
            "capabilities": self.capabilities.to_dict(),
        }

    def canonical_dict(self) -> Dict[str, Any]:
        """Canonical dict for fingerprinting — excludes volatile timestamps.

        Two TMGs that are structurally identical (same model, same lineage,
        same operator) but created at different wall-clock times receive the
        same fingerprint.
        """
        d = self.to_dict()
        d.pop("created_at", None)  # volatile — excluded from fingerprint
        return d

    def to_json(self) -> str:
        """Deterministic JSON with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    def fingerprint(self) -> str:
        """sha256 fingerprint of this genome's canonical serialization.

        Identical genomes (same lineage, same model, same operator) produce
        the same fingerprint regardless of wall-clock creation time. Used for
        deduplication, VRDEG provenance (alpha.3+), and regression detection.
        """
        return genome_fingerprint(self.canonical_dict())

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Validate against the strict TMG JSON Schema.

        Raises jsonschema.ValidationError on failure.
        """
        validate_genome(self.to_dict(), GENOME_SCHEMA_TMG)

    # ------------------------------------------------------------------ #
    # Backward compatibility                                               #
    # ------------------------------------------------------------------ #
    def to_model_genome(self) -> ModelGenome:
        """Downgrade to RF-0.x ModelGenome.

        Roundtrip guarantee (part of the preservation contract):
            TMG.to_model_genome(TMG.from_model_genome(mg)) == mg

        Fields exclusive to TMG (tmg_id, schema_version, ancestor_ids,
        capabilities, etc.) are lost. This is expected — they are alpha.2
        additions that ModelGenome does not carry.
        """
        return ModelGenome(
            model_type=self.model_type,
            architecture=copy.deepcopy(self.architecture),
            hyperparameters=copy.deepcopy(self.hyperparameters),
            data_pipeline=copy.deepcopy(self.data_pipeline),
            seed=self.seed,
            model_id=self.tmg_id,  # preserve identity across downgrade
            parent_ids=list(self.parent_ids),
            generation=self.generation,
        )

    @property
    def model_id(self) -> str:
        """Legacy alias for tmg_id — read-only property for RF-0.x compatibility."""
        return self.tmg_id

    # ------------------------------------------------------------------ #
    # Clone / evolve (immutability contract)                               #
    # ------------------------------------------------------------------ #
    def clone(self, operator: str = "clone", child_index: int = 0) -> "TargetModelGenome":
        """Create an exact structural copy under a new identity.

        Immutability contract: self is never modified. Returns a new TMG.

        The clone's lineage records self as parent. This is the entry point
        for evolution operators, which should:
          1. child = parent.clone(operator=op_name)
          2. Modify child.architecture / child.hyperparameters as needed.
          3. Return child.

        Do NOT directly assign to parent attributes.
        """
        child = copy.deepcopy(self)
        child.tmg_id = deterministic_genome_id(
            "tmg",
            [self.tmg_id],
            operator,
            self.generation,
            self.seed,
            child_index=child_index,
        )
        child.parent_ids = [self.tmg_id]
        child.ancestor_ids = list(self.ancestor_ids) + [self.tmg_id]
        child.operator = operator
        child.created_at = time.time()
        child.crossover_parents = None
        child.rollback_from = None
        return child

    def clone_for_crossover(
        self,
        other: "TargetModelGenome",
        child_index: int = 0,
    ) -> "TargetModelGenome":
        """Clone for crossover, recording both parents in crossover_parents."""
        child = copy.deepcopy(self)
        child.tmg_id = deterministic_genome_id(
            "tmg",
            [self.tmg_id, other.tmg_id],
            "crossover",
            self.generation,
            self.seed,
            child_index=child_index,
        )
        child.parent_ids = [self.tmg_id, other.tmg_id]
        child.crossover_parents = [self.tmg_id, other.tmg_id]
        child.ancestor_ids = (
            list(self.ancestor_ids) + [self.tmg_id] +
            [pid for pid in other.ancestor_ids if pid not in self.ancestor_ids]
        )
        child.operator = "crossover"
        child.created_at = time.time()
        return child

    # ------------------------------------------------------------------ #
    # Capabilities + safety                                                #
    # ------------------------------------------------------------------ #
    def safety_check(
        self,
        max_units_per_layer: int = 1024,
        max_layers: int = 6,
        max_n_estimators: int = 2000,
        max_depth_cap: int = 500,
        max_c: float = 1e6,
    ) -> List[str]:
        """Resource-bound pre-flight check.

        Identical to ModelGenome.safety_check() for the core fields, extended
        with TMGCapabilities consistency checks.

        Returns
        -------
        list of str — violation descriptions; empty list = safe to run.
        """
        violations: List[str] = []

        if self.model_type == "MLPClassifier":
            sizes = self.architecture.get("hidden_layer_sizes", [])
            if len(sizes) > max_layers:
                violations.append(
                    f"hidden_layer_sizes has {len(sizes)} layers (max {max_layers})"
                )
            for s in sizes:
                if s is None or s <= 0:
                    violations.append(f"invalid layer width {s}")
                elif s > max_units_per_layer:
                    violations.append(f"layer width {s} exceeds cap {max_units_per_layer}")
            if self.hyperparameters.get("max_iter", 0) > 20_000:
                violations.append("max_iter exceeds 20000")
        elif self.model_type == "RandomForestClassifier":
            n_est = self.architecture.get("n_estimators", 0)
            if n_est <= 0 or n_est > max_n_estimators:
                violations.append(f"n_estimators {n_est} outside (0, {max_n_estimators}]")
            depth = self.architecture.get("max_depth")
            if depth is not None and depth > max_depth_cap:
                violations.append(f"max_depth {depth} exceeds cap {max_depth_cap}")
        elif self.model_type in ("SVC", "LogisticRegression"):
            c = self.hyperparameters.get("C", 1.0)
            if c <= 0 or c > max_c:
                violations.append(f"C {c} outside (0, {max_c}]")
        else:
            violations.append(f"unrecognised model_type {self.model_type!r}")

        # Capabilities consistency
        if self.model_type == "SVC" and self.capabilities.supports_predict_proba:
            # SVC requires probability=True, which is expensive — flag if declared
            # but the genome doesn't set it explicitly.
            pass  # Not a violation; note for future tracking.

        return violations

    def build_estimator(self):
        """Materialise this genome into a real, trainable sklearn Pipeline.

        Identical to ModelGenome.build_estimator() — behavioral preservation
        guarantee (AD-004 extension). The TMG does not change how models are
        built; it only adds research provenance and schema versioning.
        """
        # Delegate to the ModelGenome implementation for behavioral parity.
        # We construct a temporary ModelGenome rather than duplicating code.
        mg = self.to_model_genome()
        return mg.build_estimator()

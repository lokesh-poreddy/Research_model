"""Shared schema utilities for ResearchForge genome types.

Provides:
  - deterministic_genome_id: collision-free, reproducible genome identity
      Includes child_index so multiple children of the same parent produced by
      the same operator in the same generation receive distinct IDs.
  - genome_fingerprint: sha256 of a genome's canonical JSON serialization.
      Useful for deduplication, VRDEG provenance (alpha.3+), caching, and
      detecting accidental in-place mutation of historical research objects.
  - validate_genome: strict jsonschema wrapper with informative error messages.

Version constants
-----------------
  GENOME_SCHEMA_VERSION_TMG = "1.0"   -- TargetModelGenome schema version
  GENOME_SCHEMA_VERSION_RSG = "1.0"   -- ResearchSystemGenome schema version

Canonical serialization contract
---------------------------------
  - Use json.dumps(d, sort_keys=True, separators=(',', ':')) for fingerprinting.
  - Timestamps (created_at) are EXCLUDED from canonical dicts so that two
    genomes created at different times but otherwise identical receive the same
    fingerprint.
  - list fields are serialized in their logical order (parent_ids, ancestor_ids)
    and normalized to lists (not tuples).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

GENOME_SCHEMA_VERSION_TMG = "1.0"
GENOME_SCHEMA_VERSION_RSG = "1.0"


def deterministic_genome_id(
    prefix: str,
    parent_ids: List[str],
    operator: str,
    generation: int,
    seed: int,
    mutation_parameters: Optional[Dict[str, Any]] = None,
    child_index: int = 0,
) -> str:
    """Return a reproducible, collision-free genome ID.

    The ID is derived from the full mutation specification:

        prefix + "_" + sha256(canonical_spec)[:16]

    where canonical_spec is a JSON-serialized dict of all inputs that
    uniquely identify this particular child in the research trajectory.

    Collision safety
    ----------------
    Two children produced from the same parent by the same operator in the
    same generation are distinguished by ``child_index``. For alpha.2, where
    operators produce exactly one child, ``child_index=0`` is always used.
    Portfolio search (RF-1.5+) must increment ``child_index`` per sibling.

    Design principle
    ----------------
    Same research state + same mutation specification + same seed → same ID.
    This makes the research graph reproducible: re-running with the same seed
    and operator sequence regenerates the same genome IDs, enabling VRDEG
    identity tracking (alpha.3+) without storing every object.

    Parameters
    ----------
    prefix : "tmg" or "rsg"
    parent_ids : list of parent genome IDs (sorted for canonical form)
    operator : name of the evolution operator
    generation : generation index when this child was created
    seed : the RNG seed governing this evolutionary step
    mutation_parameters : operator-specific parameters (e.g. delta, capacity)
    child_index : ordinal of this child among siblings from the same parent+op
    """
    spec: Dict[str, Any] = {
        "parent_ids": sorted(parent_ids),
        "operator": operator,
        "generation": generation,
        "seed": seed,
        "mutation_parameters": mutation_parameters or {},
        "child_index": child_index,
    }
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def genome_fingerprint(canonical_dict: Dict[str, Any]) -> str:
    """Return sha256 fingerprint of a genome's canonical serialization.

    The canonical dict must exclude volatile fields (created_at).
    Use ``TargetModelGenome.canonical_dict()`` or
    ``ResearchSystemGenome.canonical_dict()`` to produce the right input.

    Returns a 64-character hex digest.
    """
    canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def validate_genome(d: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validate a genome dict against its JSON Schema.

    Raises
    ------
    jsonschema.ValidationError
        With a human-readable description of the first schema violation.
    ImportError
        If jsonschema is not installed.
    """
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "jsonschema is required for genome validation. "
            "Install it with: pip install jsonschema"
        ) from exc

    try:
        jsonschema.validate(instance=d, schema=schema)
    except jsonschema.ValidationError as exc:
        # Re-raise with a more informative prefix showing which field failed.
        path = " -> ".join(str(p) for p in exc.absolute_path) or "(root)"
        raise jsonschema.ValidationError(
            f"Genome validation failed at [{path}]: {exc.message}"
        ) from exc

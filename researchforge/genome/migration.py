"""Genome schema migration registry.

When a genome dict arrives from disk, an API call, or legacy code, it may
carry an older (or absent) schema_version. The migration functions in this
module transform it to the current canonical schema before the genome class
sees it.

Design principles
-----------------
  * Migration is purely a dict-to-dict transformation.
  * It is *not* the genome class's responsibility to handle schema versions.
  * The canonical output is always the current schema version.
  * Legacy aliases (e.g., model_id) are accepted by the migration layer but
    must NOT appear in the canonical output dict; the strict JSON Schema
    (additionalProperties: false) would reject them.
  * Derived metadata (ancestor_ids, capabilities) is deterministically
    inferred from existing fields; it is not new information.

Preservation guarantee vs. derived metadata
--------------------------------------------
  Preservation guarantee:
      Every RF-0.x ModelGenome field (model_type, architecture,
      hyperparameters, data_pipeline, seed, generation, parent_ids) is
      copied without modification into the TMG canonical dict.

  Derived metadata (new in alpha.2):
      schema_version = "1.0"
      tmg_id         = migrated from model_id (alias accepted only by loader)
      ancestor_ids   = copy of parent_ids (first-generation approximation)
      operator       = "legacy_import"
      capabilities   = inferred from model_type via TMGCapabilities.infer()

Roundtrip contract
------------------
    TMG.to_model_genome(TMG.from_model_genome(mg)) == mg

    That is, the ModelGenome fields are preserved; the additional TMG fields
    are not expected to survive the round-trip (they are TMG-specific).
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional, Tuple

from .schema import GENOME_SCHEMA_VERSION_TMG, GENOME_SCHEMA_VERSION_RSG


# --------------------------------------------------------------------------- #
# TMG migration                                                                #
# --------------------------------------------------------------------------- #

# Registry: (from_version, to_version) → migration_fn
# None means "no version marker" (legacy / RF-0.x ModelGenome dict).
_TMG_MIGRATION_REGISTRY: Dict[
    Tuple[Optional[str], str], Callable[[Dict[str, Any]], Dict[str, Any]]
] = {}


def _register_tmg(from_v: Optional[str], to_v: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _TMG_MIGRATION_REGISTRY[(from_v, to_v)] = fn
        return fn
    return decorator


@_register_tmg(None, "1.0")
def _tmg_legacy_to_v1(d: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a ModelGenome dict (no schema_version) to TMG v1.

    Compatibility loader:
      - Accepts model_id as the legacy alias for tmg_id.
      - Canonical output uses tmg_id only (not model_id).
    """
    out = copy.deepcopy(d)

    # Resolve tmg_id from model_id alias
    if "tmg_id" not in out:
        if "model_id" in out:
            out["tmg_id"] = out.pop("model_id")
        else:
            # Fallback: generate a placeholder ID so the dict is valid
            import uuid
            out["tmg_id"] = f"tmg_{uuid.uuid4().hex[:10]}"
    else:
        # Remove legacy alias if both exist (tmg_id takes precedence)
        out.pop("model_id", None)

    # Add schema version
    out.setdefault("schema_version", GENOME_SCHEMA_VERSION_TMG)

    # Derived metadata
    parent_ids = out.get("parent_ids", [])
    out.setdefault("ancestor_ids", list(parent_ids))
    out.setdefault("operator", "legacy_import")
    out.setdefault("created_at", 0.0)
    out.setdefault("crossover_parents", None)
    out.setdefault("rollback_from", None)
    out.setdefault("researchforge_version", "RF-0.x")

    # Infer capabilities from model_type
    if "capabilities" not in out:
        model_type = out.get("model_type", "")
        out["capabilities"] = _infer_tmg_capabilities(model_type)

    return out


def _infer_tmg_capabilities(model_type: str) -> Dict[str, Any]:
    """Deterministically infer TMGCapabilities from a model_type string."""
    return {
        "supports_warm_start": model_type == "MLPClassifier",
        "supports_partial_fit": model_type == "MLPClassifier",
        "supports_predict_proba": model_type != "SVC",
        "expected_train_time_s": None,
        "memory_estimate_mb": None,
    }


def migrate_tmg(
    d: Dict[str, Any],
    target_version: str = GENOME_SCHEMA_VERSION_TMG,
) -> Dict[str, Any]:
    """Migrate a TMG (or legacy ModelGenome) dict to target_version.

    Applies the appropriate migration chain from the dict's current
    schema_version to target_version.

    Parameters
    ----------
    d : dict — the raw genome dict (may lack schema_version).
    target_version : str — the desired schema version (default: current).

    Returns
    -------
    dict — migrated canonical TMG dict. Never shares memory with ``d``.
    """
    current = d.get("schema_version", None)
    if current == target_version:
        # Remove any residual legacy alias even if already at target
        result = copy.deepcopy(d)
        if "tmg_id" in result:
            result.pop("model_id", None)
        return result

    key = (current, target_version)
    fn = _TMG_MIGRATION_REGISTRY.get(key)
    if fn is None:
        raise ValueError(
            f"No TMG migration path from schema_version={current!r} "
            f"to {target_version!r}. Known paths: {list(_TMG_MIGRATION_REGISTRY.keys())}"
        )
    return fn(d)


# --------------------------------------------------------------------------- #
# RSG migration                                                                #
# --------------------------------------------------------------------------- #

_RSG_MIGRATION_REGISTRY: Dict[
    Tuple[Optional[str], str], Callable[[Dict[str, Any]], Dict[str, Any]]
] = {}


def _register_rsg(from_v: Optional[str], to_v: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _RSG_MIGRATION_REGISTRY[(from_v, to_v)] = fn
        return fn
    return decorator


@_register_rsg(None, "1.0")
def _rsg_legacy_to_v1(d: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a legacy RSG dict (no schema_version) to RSG v1."""
    out = copy.deepcopy(d)
    out.setdefault("schema_version", GENOME_SCHEMA_VERSION_RSG)
    out.setdefault("operator", "legacy_import")
    out.setdefault("created_at", 0.0)
    out.setdefault("parent_rsg_id", None)
    out.setdefault("generation", 0)
    out.setdefault("researchforge_version", "RF-0.x")
    return out


def migrate_rsg(
    d: Dict[str, Any],
    target_version: str = GENOME_SCHEMA_VERSION_RSG,
) -> Dict[str, Any]:
    """Migrate an RSG dict to target_version."""
    current = d.get("schema_version", None)
    if current == target_version:
        return copy.deepcopy(d)

    key = (current, target_version)
    fn = _RSG_MIGRATION_REGISTRY.get(key)
    if fn is None:
        raise ValueError(
            f"No RSG migration path from schema_version={current!r} "
            f"to {target_version!r}. Known paths: {list(_RSG_MIGRATION_REGISTRY.keys())}"
        )
    return fn(d)

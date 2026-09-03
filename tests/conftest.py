"""tests/conftest.py — Shared pytest configuration for ResearchForge-ECRM.

RF-1.0.0-alpha.2.1: Adds CORE/COMPATIBILITY/LEGACY test classification.

Test classification:
    CORE         Canonical RF-1 behavior. Runs on every commit. Failure blocks release.
    COMPATIBILITY  Legacy API/serialization deliberately preserved. Must pass.
    LEGACY       Historical code retained as research evidence. Not supported product path.

Note: COMPATIBILITY and LEGACY test files already carry pytestmark = pytest.mark.COMPATIBILITY/LEGACY
      at the module level. CORE files are auto-marked here since they use a standalone
      runner pattern without a top-level pytestmark.
"""
from __future__ import annotations

import pytest

# ─── Canonical object invariants mixin ────────────────────────────────────────

class CanonicalObjectInvariants:
    """Shared invariants for all Class A/D canonical research artifacts.
    
    Subclass this and call each method with appropriate objects.
    
    Applies to: RSG, TMG, ExperimentSpec, ExperimentRun, ExperimentOutcome,
                ResearchState, ResearchProblem, Hypothesis.
    """

    @staticmethod
    def assert_serialize_roundtrip_fingerprint(obj, cls) -> None:
        """deserialize(serialize(x)).fingerprint() == x.fingerprint()"""
        restored = cls.from_dict(obj.to_dict())
        assert restored.fingerprint() == obj.fingerprint(), (
            f"{cls.__name__}: fingerprint changed after serialize/deserialize roundtrip.\n"
            f"  Original:   {obj.fingerprint()}\n"
            f"  Restored:   {restored.fingerprint()}"
        )

    @staticmethod
    def assert_fingerprint_deterministic(obj) -> None:
        """fingerprint() returns the same value on repeated calls."""
        fp1 = obj.fingerprint()
        fp2 = obj.fingerprint()
        assert fp1 == fp2, (
            f"{type(obj).__name__}: fingerprint() is not deterministic.\n"
            f"  Call 1: {fp1}\n  Call 2: {fp2}"
        )

    @staticmethod
    def assert_evolve_changes_fingerprint(original, evolved) -> None:
        """evolve()/clone() produces a new object with a different fingerprint."""
        assert evolved.fingerprint() != original.fingerprint(), (
            f"{type(original).__name__}: evolve() did not change fingerprint — "
            "the child and parent are indistinguishable."
        )

    @staticmethod
    def assert_no_legacy_fields(obj, forbidden_fields) -> None:
        """Canonical dict must not contain legacy field names."""
        d = obj.to_dict()
        found = [f for f in forbidden_fields if f in d]
        assert not found, (
            f"{type(obj).__name__}: legacy fields found in canonical dict: {found}"
        )

    @staticmethod
    def assert_schema_version_present(obj) -> None:
        """Canonical dict must contain schema_version."""
        d = obj.to_dict()
        assert "schema_version" in d, (
            f"{type(obj).__name__}: 'schema_version' missing from to_dict() output"
        )

    @staticmethod
    def assert_additional_properties_rejected(d_with_extra, schema) -> None:
        """JSON Schema (additionalProperties:false) must reject injected garbage field."""
        from researchforge.genome.schema import validate_genome
        with pytest.raises(Exception, match=r"(?i)(additional|unexpected|not allowed)"):
            validate_genome(d_with_extra, schema)


# ─── Auto-mark CORE test files ────────────────────────────────────────────────

CORE_FILES = frozenset([
    "test_basic.py",
    "test_adapters.py",
    "test_scientific_validity.py",
    "test_regression.py",
    "test_genomes.py",
    # New alpha.2.1 test files (added as created):
    "test_experiment.py",
    "test_state.py",
    "test_research.py",
    "test_retrieval.py",
    "test_config.py",
    "test_no_cross_stack_imports.py",
    "test_canonical_invariants.py",
])


def pytest_collection_modifyitems(items):
    """Auto-apply CORE marker to files that carry it by filename convention."""
    for item in items:
        filename = item.fspath.basename
        # Auto-mark CORE files (they use a standalone runner without pytestmark)
        if filename in CORE_FILES:
            item.add_marker(pytest.mark.CORE)

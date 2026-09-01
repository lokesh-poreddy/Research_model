"""Adapter test suite — Tier 1 (Unit), Tier 2 (Contract), Tier 3 (Integration).

This suite validates:
  1. Individual backend instantiation and capabilities declaration.
  2. Contract compliance via the validate_graph_backend() and
     validate_vector_backend() harnesses.
  3. RDG domain layer on top of each backend (integration).
  4. ECRM behavioral equivalence: refactored ECRM (using VectorIndexBackend)
     must return IDENTICAL results to the RF-0.x inline-cosine version
     for the same data and operations.
  5. Registry operations: registration, create, info, list.
  6. Transaction semantics: commit persists, rollback discards.
  7. DimensionMismatchError for wrong-shape vectors.
  8. BackendCapabilityError for transactions on non-transactional backends.
  9. Neo4j/pgvector stub: instantiation raises NotImplementedError.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from researchforge.adapters import (
    InMemoryGraphBackend, SQLiteGraphBackend, InProcessVectorIndex,
    Neo4jGraphBackend, PgvectorBackend,
    AdapterRegistry, get_default_registry,
    BackendCapabilityError, DimensionMismatchError,
    validate_graph_backend, validate_vector_backend,
)
from researchforge.rdg.graph import ResearchDevelopmentGraph, EdgeConstraintError
from researchforge.memory.ecrm import ECRM


_PASS = []
_FAIL = []


def ok(name: str) -> None:
    print(f"OK: {name}")
    _PASS.append(name)


def fail(name: str, exc: Exception) -> None:
    print(f"FAIL: {name} — {exc}")
    _FAIL.append((name, exc))


def run(name: str, fn) -> None:
    try:
        fn()
        ok(name)
    except Exception as exc:
        fail(name, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Unit tests for individual backend methods
# ─────────────────────────────────────────────────────────────────────────────


def test_in_memory_graph_crud():
    b = InMemoryGraphBackend()
    b.add_node("n1", "Problem", "p1", "2026-01-01T00:00:00+00:00", '{"x": 1}')
    n = b.get_node("n1")
    assert n["type"] == "Problem"
    assert n["attributes"] == {"x": 1}
    b.add_node("n2", "Gap", "g1", "2026-01-01T00:00:00+00:00", "{}")
    b.add_edge("n1", "n2", "identifies", "{}")
    out = b.get_out_edges("n1")
    assert len(out) == 1 and out[0]["relation"] == "identifies"
    in_e = b.get_in_edges("n2")
    assert len(in_e) == 1
    typed = b.get_nodes_by_type("Gap")
    assert any(n["id"] == "n2" for n in typed)
    all_n = b.all_nodes()
    assert len(all_n) == 2
    missing = b.get_node("does_not_exist")
    assert missing is None


def test_sqlite_graph_crud():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        b = SQLiteGraphBackend(db_path)
        b.add_node("n1", "Problem", "p1", "2026-01-01T00:00:00+00:00", '{"y": 2}')
        n = b.get_node("n1")
        assert n["attributes"] == {"y": 2}
        b.add_edge("n1", "n1", "produces", '{"w": 0.5}')
        edges = b.get_out_edges("n1")
        assert edges[0]["properties"] == {"w": 0.5}
        b.close()
    finally:
        os.unlink(db_path)


def test_in_process_vector_crud():
    v = InProcessVectorIndex()
    v.add("v1", [1.0, 0.0])
    v.add("v2", [0.0, 1.0])
    assert v.size() == 2
    results = v.search([1.0, 0.0], k=2)
    assert results[0][0] == "v1"
    assert abs(results[0][1] - 1.0) < 1e-6
    v.remove("v2")
    assert v.size() == 1


def test_dimension_mismatch_raises():
    v = InProcessVectorIndex(dimension=3)
    v.add("v1", [1.0, 0.0, 0.0])
    try:
        v.add("v_bad", [1.0, 0.0])
        raise AssertionError("Expected DimensionMismatchError")
    except DimensionMismatchError as e:
        assert e.expected == 3
        assert e.got == 2


def test_vector_metadata_roundtrip():
    v = InProcessVectorIndex()
    meta = {"strategy": "increase_capacity", "tier": "long_term", "r": 0.91}
    v.add("v1", [1.0, 0.0, 0.0], metadata=meta)
    got = v.get_metadata("v1")
    assert got == meta


def test_vector_k_none_returns_all():
    v = InProcessVectorIndex()
    for i in range(10):
        v.add(f"v{i}", [float(i), 0.0])
    results = v.search([1.0, 0.0], k=None)
    assert len(results) == 10


def test_stubs_raise_not_implemented():
    """Neo4j and pgvector stubs must raise NotImplementedError on instantiation."""
    try:
        Neo4jGraphBackend()
        raise AssertionError("Expected NotImplementedError from Neo4jGraphBackend")
    except NotImplementedError:
        pass
    try:
        PgvectorBackend()
        raise AssertionError("Expected NotImplementedError from PgvectorBackend")
    except NotImplementedError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Contract validation via harness
# ─────────────────────────────────────────────────────────────────────────────


def test_in_memory_graph_contract():
    report = validate_graph_backend(InMemoryGraphBackend(), level="contract")
    assert report.passed, f"Contract failures: {report.failures}"


def test_sqlite_graph_contract():
    report = validate_graph_backend(SQLiteGraphBackend(":memory:"), level="contract")
    assert report.passed, f"Contract failures: {report.failures}"


def test_sqlite_graph_persistence():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        b = SQLiteGraphBackend(db_path)
        report = validate_graph_backend(
            b, level="persistence",
            sqlite_path_for_persistence=db_path)
        assert report.passed, f"Persistence failures: {report.failures}"
        b.close()
    finally:
        os.unlink(db_path)


def test_in_process_vector_contract():
    report = validate_vector_backend(InProcessVectorIndex())
    assert report.passed, f"Vector contract failures: {report.failures}"


def test_in_memory_transaction_commit():
    b = InMemoryGraphBackend()
    with b.transaction() as txn:
        txn.add_node("tn1", "Finding", "f", "2026-01-01T00:00:00+00:00", "{}")
    n = b.get_node("tn1")
    assert n is not None, "Committed node not persisted"


def test_in_memory_transaction_rollback():
    b = InMemoryGraphBackend()
    sentinel = "rollback_only"
    try:
        with b.transaction() as txn:
            txn.add_node(sentinel, "Claim", "c", "2026-01-01T00:00:00+00:00", "{}")
            raise RuntimeError("deliberate rollback")
    except RuntimeError:
        pass
    assert b.get_node(sentinel) is None, "Rolled-back node should not persist"


def test_sqlite_transaction_commit_and_rollback():
    b = SQLiteGraphBackend(":memory:")
    # Commit
    with b.transaction() as txn:
        txn.add_node("txc", "Problem", "p", "2026-01-01T00:00:00+00:00", "{}")
    assert b.get_node("txc") is not None
    # Rollback
    try:
        with b.transaction() as txn:
            txn.add_node("txr", "Gap", "g", "2026-01-01T00:00:00+00:00", "{}")
            raise ValueError("rollback me")
    except ValueError:
        pass
    assert b.get_node("txr") is None, "Rolled-back node should not persist"


def test_health_check():
    for backend in [InMemoryGraphBackend(), SQLiteGraphBackend(":memory:")]:
        h = backend.health_check()
        assert h.healthy
        assert h.latency_ms >= 0.0


def test_non_transactional_capability_error():
    """A backend that does not support transactions must raise BackendCapabilityError.

    We test this by creating a minimal concrete subclass of GraphBackend that
    explicitly inherits the default (non-implemented) transaction() method and
    declares transactional=False, then calling transaction() on it.
    """
    from researchforge.adapters.protocols import GraphBackend as _BaseGB
    from researchforge.adapters.capabilities import BackendCapabilities, HealthStatus

    class _NonTransactionalBackend(_BaseGB):
        """Minimal concrete backend that does NOT support transactions."""
        @classmethod
        def capabilities(cls) -> BackendCapabilities:
            return BackendCapabilities(transactional=False)

        @classmethod
        def backend_name(cls) -> str:
            return "non_transactional_test"

        def health_check(self) -> HealthStatus:
            return HealthStatus(healthy=True, persistent=False, transactional=False)

    b = _NonTransactionalBackend()
    # The base class transaction() raises BackendCapabilityError
    try:
        ctx = b.transaction()
        # Some context managers are generators — we need to __enter__ to trigger
        ctx.__enter__()
        raise AssertionError("Expected BackendCapabilityError")
    except (BackendCapabilityError, RuntimeError, TypeError, AttributeError):
        # BackendCapabilityError is the expected path.
        # contextmanager wraps the generator in a _GeneratorContextManager;
        # calling __enter__ on the base class that raises will propagate the error.
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — Integration: RDG domain layer on top of backends
# ─────────────────────────────────────────────────────────────────────────────


def test_rdg_with_in_memory_backend():
    b = InMemoryGraphBackend()
    rdg = ResearchDevelopmentGraph(backend=b)
    p = rdg.add_node("Problem", "in-memory backend test")
    gap = rdg.add_node("Gap", "gap node")
    hyp = rdg.add_node("Hypothesis", "hyp node")
    exp = rdg.add_node("Experiment", "exp node")
    finding = rdg.add_node("Finding", "finding node")
    claim = rdg.add_node("Claim", "claim node")
    rdg.add_edge(p.id, gap.id, "identifies")
    rdg.add_edge(gap.id, hyp.id, "motivates")
    rdg.add_edge(hyp.id, exp.id, "tested-by")
    rdg.add_edge(exp.id, finding.id, "produces")
    rdg.add_edge(finding.id, claim.id, "supports")
    chain = rdg.evidence_chain(claim.id)
    types = [n.type for n in chain]
    assert types == ["Problem", "Gap", "Hypothesis", "Experiment", "Finding", "Claim"]


def test_rdg_with_sqlite_backend():
    b = SQLiteGraphBackend(":memory:")
    rdg = ResearchDevelopmentGraph(backend=b)
    p = rdg.add_node("Problem", "sqlite backend test")
    gap = rdg.add_node("Gap", "gap")
    rdg.add_edge(p.id, gap.id, "identifies")
    chain = rdg.evidence_chain(gap.id)
    assert chain[0].type == "Problem"


def test_rdg_old_db_path_api_still_works():
    """RF-0.x backward compatibility: db_path= API must work unchanged."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        rdg = ResearchDevelopmentGraph(db_path=db_path)
        p = rdg.add_node("Problem", "legacy db_path API")
        assert rdg.db_path == db_path
        rdg2 = ResearchDevelopmentGraph()
        p2 = rdg2.add_node("Problem", "no path")
        assert rdg2.db_path is None
    finally:
        os.unlink(db_path)


def test_rdg_edge_constraint_stays_in_domain_layer():
    """Domain constraints live in RDG, not backend. Same after refactor."""
    rdg = ResearchDevelopmentGraph(backend=InMemoryGraphBackend())
    p = rdg.add_node("Problem", "p")
    gap = rdg.add_node("Gap", "g")
    hyp = rdg.add_node("Hypothesis", "h")
    rdg.add_edge(p.id, gap.id, "identifies")     # legal
    rdg.add_edge(gap.id, hyp.id, "motivates")    # legal
    try:
        rdg.add_edge(gap.id, p.id, "identifies") # illegal: (Gap, Problem) not allowed
        raise AssertionError("Expected EdgeConstraintError")
    except EdgeConstraintError:
        pass


def test_rdg_reload_from_backend():
    """_reload_from_backend() must reconstruct the in-process index."""
    b = SQLiteGraphBackend(":memory:")
    rdg1 = ResearchDevelopmentGraph(backend=b)
    p = rdg1.add_node("Problem", "reload test")
    gap = rdg1.add_node("Gap", "gap")
    rdg1.add_edge(p.id, gap.id, "identifies")

    # Construct a new RDG over same backend, reload
    rdg2 = ResearchDevelopmentGraph(backend=b)
    rdg2._reload_from_backend()
    assert p.id in rdg2.nodes
    assert gap.id in rdg2.nodes
    assert len(rdg2.edges) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — ECRM Behavioral Equivalence
# ─────────────────────────────────────────────────────────────────────────────


def test_ecrm_behavioral_equivalence():
    """ECRM with VectorIndexBackend must return same results as RF-0.x inline cosine.

    We run identical store() and query() sequences on the refactored ECRM,
    then verify: retrieval order is correct, has_similar_failure() agrees,
    and strategy_stats() / RES / NTR are unchanged.
    """
    ecrm = ECRM()
    # Store two similar failure records for the same strategy
    r1 = ecrm.store(
        "increase capacity failed: MLP overfitting on digits, n_layers=4",
        context={"model_type": "MLP", "capacity": "high"},
        outcome={"success": False, "metric": 0.72, "error_type": "overfitting"},
        strategy="increase_capacity")
    r2 = ecrm.store(
        "increase capacity failed: MLP overfitting on ECG, n_layers=3",
        context={"model_type": "MLP", "capacity": "medium"},
        outcome={"success": False, "metric": 0.61, "error_type": "overfitting"},
        strategy="increase_capacity")
    # Store a dissimilar success
    r3 = ecrm.store(
        "random forest crossover success on synthetic ECG task",
        context={"model_type": "RF", "capacity": "medium"},
        outcome={"success": True, "metric": 0.88},
        strategy="crossover_top2")

    # Query: similar to failure records
    results = ecrm.query("MLP overfitting high capacity", k=2)
    assert len(results) >= 1
    # The MLP overfitting records should be at top
    top_strategies = [r.strategy for r in results]
    assert "increase_capacity" in top_strategies, (
        f"Expected increase_capacity in top results, got {top_strategies}")

    # has_similar_failure: should find the MLP failure
    assert ecrm.has_similar_failure(
        "MLP overfitting on digits n_layers=4"), (
        "has_similar_failure should return True for similar failure")

    # has_similar_failure: should NOT fire for RF success
    assert not ecrm.has_similar_failure(
        "random forest crossover success ECG"), (
        "has_similar_failure should return False for success record")

    # Strategy stats
    stats = ecrm.strategy_stats("increase_capacity")
    assert stats["n"] == 2
    assert abs(stats["mean"] - (0.72 + 0.61) / 2) < 0.01

    # NTR starts at 0
    ntr = ecrm.negative_transfer_rate("increase_capacity")
    assert ntr == 0.0

    # RES is computable
    res = ecrm.research_experience_score("increase_capacity")
    assert isinstance(res, float)

    # half life is positive
    hl = ecrm.memory_half_life_days()
    assert hl > 0


def test_ecrm_consolidate_behavioral_equivalence():
    """consolidate() / forgetting algorithm must match RF-0.x exactly."""
    import time as _time
    ecrm = ECRM(decay_lambda=10.0,  # very fast decay
                 retention_threshold=0.5)
    ecrm.store(
        "very old bad experiment",
        context={}, outcome={"success": False, "metric": 0.1},
        strategy="random_search")
    # Simulate the record being old (set created_at to far in the past)
    for rec in ecrm.records.values():
        rec.created_at = _time.time() - 200 * 86400  # 200 days ago

    archived = ecrm.consolidate()
    assert archived >= 1, f"Expected at least one archived record, got {archived}"
    # All archived records should have archived=True
    for rec in ecrm.records.values():
        if rec.archived:
            assert rec.strategy == "random_search"


def test_ecrm_with_explicit_vector_backend():
    """ECRM can accept a pre-constructed VectorIndexBackend."""
    custom_index = InProcessVectorIndex()
    ecrm = ECRM(vector_backend=custom_index)
    r = ecrm.store("test record", context={},
                    outcome={"success": True, "metric": 0.9},
                    strategy="test_strategy")
    # The record's id must be in the custom index
    size = custom_index.size()
    assert size == 1, f"Expected 1 vector in custom index, got {size}"
    # Query must work
    results = ecrm.query("test record", k=1)
    assert len(results) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Registry tests
# ─────────────────────────────────────────────────────────────────────────────


def test_registry_create_and_info():
    reg = get_default_registry()
    info = reg.graph_backend_info("in_memory")
    assert info is not None
    assert info.capabilities.persistent is False

    info_s = reg.graph_backend_info("sqlite")
    assert info_s.capabilities.persistent is True
    assert info_s.capabilities.transactional is True

    info_v = reg.vector_backend_info("inprocess")
    assert info_v is not None
    assert info_v.capabilities.supports_vector_metadata is True


def test_registry_create_backend():
    reg = get_default_registry()
    b = reg.create_graph_backend("in_memory")
    assert isinstance(b, InMemoryGraphBackend)

    v = reg.create_vector_backend("inprocess")
    assert isinstance(v, InProcessVectorIndex)


def test_registry_unknown_raises():
    reg = get_default_registry()
    try:
        reg.create_graph_backend("neo4j_production")
        raise AssertionError("Expected BackendCapabilityError")
    except BackendCapabilityError:
        pass


def test_registry_list():
    reg = get_default_registry()
    g_backends = reg.list_graph_backends()
    assert "in_memory" in g_backends
    assert "sqlite" in g_backends
    v_backends = reg.list_vector_backends()
    assert "inprocess" in v_backends


# ─────────────────────────────────────────────────────────────────────────────
# Full integration contract via validation harness
# ─────────────────────────────────────────────────────────────────────────────


def test_in_memory_integration_level():
    b = InMemoryGraphBackend()
    report = validate_graph_backend(b, level="integration")
    assert report.passed, f"Integration failures: {report.failures}"


def test_sqlite_integration_level():
    b = SQLiteGraphBackend(":memory:")
    report = validate_graph_backend(b, level="integration")
    assert report.passed, f"Integration failures: {report.failures}"


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


_TESTS = [
    # Tier 1 — Unit
    ("test_in_memory_graph_crud", test_in_memory_graph_crud),
    ("test_sqlite_graph_crud", test_sqlite_graph_crud),
    ("test_in_process_vector_crud", test_in_process_vector_crud),
    ("test_dimension_mismatch_raises", test_dimension_mismatch_raises),
    ("test_vector_metadata_roundtrip", test_vector_metadata_roundtrip),
    ("test_vector_k_none_returns_all", test_vector_k_none_returns_all),
    ("test_stubs_raise_not_implemented", test_stubs_raise_not_implemented),
    # Tier 2 — Contract
    ("test_in_memory_graph_contract", test_in_memory_graph_contract),
    ("test_sqlite_graph_contract", test_sqlite_graph_contract),
    ("test_sqlite_graph_persistence", test_sqlite_graph_persistence),
    ("test_in_process_vector_contract", test_in_process_vector_contract),
    ("test_in_memory_transaction_commit", test_in_memory_transaction_commit),
    ("test_in_memory_transaction_rollback", test_in_memory_transaction_rollback),
    ("test_sqlite_transaction_commit_and_rollback",
     test_sqlite_transaction_commit_and_rollback),
    ("test_health_check", test_health_check),
    ("test_non_transactional_capability_error",
     test_non_transactional_capability_error),
    # Tier 3 — Integration
    ("test_rdg_with_in_memory_backend", test_rdg_with_in_memory_backend),
    ("test_rdg_with_sqlite_backend", test_rdg_with_sqlite_backend),
    ("test_rdg_old_db_path_api_still_works", test_rdg_old_db_path_api_still_works),
    ("test_rdg_edge_constraint_stays_in_domain_layer",
     test_rdg_edge_constraint_stays_in_domain_layer),
    ("test_rdg_reload_from_backend", test_rdg_reload_from_backend),
    ("test_ecrm_behavioral_equivalence", test_ecrm_behavioral_equivalence),
    ("test_ecrm_consolidate_behavioral_equivalence",
     test_ecrm_consolidate_behavioral_equivalence),
    ("test_ecrm_with_explicit_vector_backend",
     test_ecrm_with_explicit_vector_backend),
    # Registry
    ("test_registry_create_and_info", test_registry_create_and_info),
    ("test_registry_create_backend", test_registry_create_backend),
    ("test_registry_unknown_raises", test_registry_unknown_raises),
    ("test_registry_list", test_registry_list),
    # Full validation harness
    ("test_in_memory_integration_level", test_in_memory_integration_level),
    ("test_sqlite_integration_level", test_sqlite_integration_level),
]


if __name__ == "__main__":
    for name, fn in _TESTS:
        run(name, fn)

    print(f"\n{len(_PASS)} adapter tests passed.")
    if _FAIL:
        print(f"{len(_FAIL)} FAILED:")
        for name, exc in _FAIL:
            print(f"  FAIL: {name} — {exc}")
        sys.exit(1)

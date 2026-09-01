"""Backend contract validation harness.

Three levels of validation, each building on the previous:

  Level 1 — Contract
    Does the backend obey the GraphBackend / VectorIndexBackend API?
    Checks: CRUD, error types, transaction lifecycle.
    Pure in-process. No RDG semantics involved.

  Level 2 — Persistence
    Does the backend retain data across close/reopen?
    Only applicable if capabilities().persistent is True.
    For SQLiteGraphBackend: closes the connection, opens a NEW connection
    to the same file, verifies data is still there.

  Level 3 — Integration
    Does the RDG domain layer behave correctly on top of this backend?
    Builds a full RDG (Problem→Gap→Hypothesis→Experiment→Finding→Claim chain),
    exercises evidence_chain(), verifies typed-edge constraint enforcement.

Any validation failure raises ValidationError with a description.
ValidationError is NOT a BackendError — it is a programming error (the
backend doesn't satisfy the contract), not a runtime operational error.
"""
from __future__ import annotations

import math
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .errors import BackendCapabilityError, DimensionMismatchError
from .protocols import GraphBackend, VectorIndexBackend


class ValidationError(Exception):
    """Raised when a backend fails a contract validation check."""


@dataclass
class ValidationReport:
    """Result of running validate_graph_backend() or validate_vector_backend()."""
    backend_name: str
    level: str  # "contract" | "persistence" | "integration"
    passed: bool
    checks_run: int = 0
    checks_passed: int = 0
    failures: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"ValidationReport({self.backend_name}, {self.level}, {status}, "
                f"{self.checks_passed}/{self.checks_run} checks)")


# ── Helpers ─────────────────────────────────────────────────────────────────


def _check(report: ValidationReport, name: str, fn) -> None:
    """Run one check function; record pass/fail in report without raising."""
    report.checks_run += 1
    try:
        fn()
        report.checks_passed += 1
    except Exception as exc:
        report.failures.append(f"{name}: {exc}")


# ── Graph backend validation ─────────────────────────────────────────────────


def validate_graph_backend(
    backend: GraphBackend,
    level: str = "contract",
    sqlite_path_for_persistence: Optional[str] = None,
) -> ValidationReport:
    """Run the contract (and optionally persistence and integration) checks.

    Parameters
    ----------
    backend : GraphBackend
        The backend instance to test. It should be freshly constructed and empty.
    level : str
        "contract"    — CRUD + transaction lifecycle only.
        "persistence" — also verifies data survives close/reopen (SQLite only).
        "integration" — also builds a full RDG chain on top of the backend.
    sqlite_path_for_persistence : str | None
        Required for the persistence level when the backend is SQLite-based.
        Must point to the same file that `backend` was constructed with.
    """
    t0 = time.monotonic()
    report = ValidationReport(
        backend_name=type(backend).backend_name(),
        level=level, passed=False)

    # ── Level 1 — Contract ────────────────────────────────────────────────

    # 1a. Add a node and retrieve it
    def _node_roundtrip():
        backend.add_node("n1", "Problem", "test content",
                         "2026-01-01T00:00:00+00:00", '{"k": 1}')
        n = backend.get_node("n1")
        assert n is not None, "get_node returned None"
        assert n["id"] == "n1"
        assert n["type"] == "Problem"
        assert n["content"] == "test content"
        assert n["attributes"] == {"k": 1}

    _check(report, "node_roundtrip", _node_roundtrip)

    # 1b. get_nodes_by_type
    def _nodes_by_type():
        backend.add_node("n2", "Gap", "gap content", "2026-01-01T00:00:00+00:00", "{}")
        gaps = backend.get_nodes_by_type("Gap")
        ids = {n["id"] for n in gaps}
        assert "n2" in ids, f"Expected n2 in gap nodes, got {ids}"

    _check(report, "nodes_by_type", _nodes_by_type)

    # 1c. Missing node returns None (not an exception)
    def _missing_node():
        result = backend.get_node("does_not_exist")
        assert result is None, f"Expected None for missing node, got {result!r}"

    _check(report, "missing_node_returns_none", _missing_node)

    # 1d. Add and retrieve an edge
    def _edge_roundtrip():
        backend.add_node("n3", "Hypothesis", "h", "2026-01-01T00:00:00+00:00", "{}")
        backend.add_edge("n1", "n3", "motivates", '{"weight": 0.9}')
        out = backend.get_out_edges("n1")
        relations = {e["relation"] for e in out}
        assert "motivates" in relations, f"Expected 'motivates' in out-edges, got {relations}"
        in_edges = backend.get_in_edges("n3")
        assert any(e["from_id"] == "n1" for e in in_edges)
        assert any(e["properties"] == {"weight": 0.9} for e in in_edges)

    _check(report, "edge_roundtrip", _edge_roundtrip)

    # 1e. all_nodes / all_edges
    def _all_nodes_edges():
        all_n = backend.all_nodes()
        assert len(all_n) >= 3, f"Expected ≥3 nodes, got {len(all_n)}"
        all_e = backend.all_edges()
        assert len(all_e) >= 1, f"Expected ≥1 edge, got {len(all_e)}"

    _check(report, "all_nodes_and_edges", _all_nodes_edges)

    # 1f. Transaction — successful commit
    def _transaction_commit():
        caps = type(backend).capabilities()
        if not caps.transactional:
            return  # skip for non-transactional backends
        with backend.transaction() as txn:
            txn.add_node("txn_n1", "Finding", "finding", "2026-01-01T00:00:00+00:00", "{}")
            txn.add_edge("n1", "txn_n1", "produces", "{}")
        n = backend.get_node("txn_n1")
        assert n is not None, "Committed node not found after transaction"

    _check(report, "transaction_commit", _transaction_commit)

    # 1g. Transaction — rollback on exception
    def _transaction_rollback():
        caps = type(backend).capabilities()
        if not caps.transactional:
            return
        sentinel = "rollback_node"
        try:
            with backend.transaction() as txn:
                txn.add_node(sentinel, "Claim", "should not persist",
                             "2026-01-01T00:00:00+00:00", "{}")
                raise RuntimeError("deliberate rollback trigger")
        except RuntimeError:
            pass  # expected
        n = backend.get_node(sentinel)
        assert n is None, f"Node '{sentinel}' should have been rolled back but persisted"

    _check(report, "transaction_rollback", _transaction_rollback)

    # 1h. health_check returns a HealthStatus
    def _health_check():
        h = backend.health_check()
        assert h is not None
        assert isinstance(h.healthy, bool)
        assert isinstance(h.latency_ms, float)

    _check(report, "health_check", _health_check)

    # 1i. Non-transactional backend raises BackendCapabilityError on .transaction()
    def _non_transactional_raises():
        caps = type(backend).capabilities()
        if caps.transactional:
            return  # only meaningful for non-transactional backends
        try:
            with backend.transaction():
                pass
            assert False, "Expected BackendCapabilityError"
        except BackendCapabilityError:
            pass

    _check(report, "non_transactional_backend_raises", _non_transactional_raises)

    # ── Level 2 — Persistence ─────────────────────────────────────────────
    if level in ("persistence", "integration"):
        caps = type(backend).capabilities()
        if caps.persistent and sqlite_path_for_persistence:
            def _persistence():
                from .backends.sqlite import SQLiteGraphBackend
                backend2 = SQLiteGraphBackend(sqlite_path_for_persistence)
                n = backend2.get_node("n1")
                backend2.close()
                assert n is not None, "Data not found after reopen (persistence failure)"
                assert n["content"] == "test content"

            _check(report, "persistence_across_reopen", _persistence)

    # ── Level 3 — Integration (RDG layer on top of backend) ──────────────
    if level == "integration":
        def _rdg_integration():
            from ..rdg.graph import ResearchDevelopmentGraph, EdgeConstraintError
            rdg = ResearchDevelopmentGraph(backend=backend)
            p = rdg.add_node("Problem", "integration test problem")
            gap = rdg.add_node("Gap", "integration test gap")
            hyp = rdg.add_node("Hypothesis", "integration test hypothesis")
            exp = rdg.add_node("Experiment", "integration test experiment")
            finding = rdg.add_node("Finding", "integration test finding")
            claim = rdg.add_node("Claim", "integration test claim")
            rdg.add_edge(p.id, gap.id, "identifies")
            rdg.add_edge(gap.id, hyp.id, "motivates")
            rdg.add_edge(hyp.id, exp.id, "tested-by")
            rdg.add_edge(exp.id, finding.id, "produces")
            rdg.add_edge(finding.id, claim.id, "supports")
            chain = rdg.evidence_chain(claim.id)
            types = [n.type for n in chain]
            assert types == ["Problem", "Gap", "Hypothesis", "Experiment",
                             "Finding", "Claim"], f"Unexpected chain: {types}"
            # Typed-edge constraint is enforced by RDG, not backend
            try:
                rdg.add_edge(gap.id, p.id, "identifies")
                assert False, "Expected EdgeConstraintError"
            except EdgeConstraintError:
                pass

        _check(report, "rdg_integration", _rdg_integration)

    report.elapsed_ms = (time.monotonic() - t0) * 1000
    report.passed = (len(report.failures) == 0)
    return report


# ── Vector backend validation ────────────────────────────────────────────────


def validate_vector_backend(backend: VectorIndexBackend) -> ValidationReport:
    """Run contract checks for a VectorIndexBackend.

    Uses 3-dimensional test vectors with known cosine similarities so
    correctness can be checked deterministically.
    """
    t0 = time.monotonic()
    report = ValidationReport(
        backend_name=type(backend).backend_name(),
        level="contract", passed=False)

    dim3 = [1.0, 0.0, 0.0]  # unit vector along x
    dim3b = [0.0, 1.0, 0.0]  # orthogonal → cosine = 0
    dim3c = [1.0, 1.0, 0.0]  # 45° → cosine ≈ 0.707

    # 2a. Add and search
    def _add_search():
        backend.add("v1", dim3)
        backend.add("v2", dim3b)
        backend.add("v3", dim3c)
        results = backend.search(dim3, k=3)
        ids = [r[0] for r in results]
        assert ids[0] == "v1", f"Expected v1 as top result, got {ids}"
        assert ids[1] == "v3", f"Expected v3 as second, got {ids}"
        # scores in descending order
        scores = [r[1] for r in results]
        assert scores[0] >= scores[1] >= scores[2], f"Scores not descending: {scores}"
        # v1 vs v1 should be ~1.0
        assert abs(scores[0] - 1.0) < 1e-6, f"Self-similarity not 1.0: {scores[0]}"
        # v1 vs v2 should be ~0.0
        assert abs(scores[2]) < 1e-6, f"Orthogonal similarity not 0.0: {scores[2]}"

    _check(report, "add_and_search", _add_search)

    # 2b. Remove
    def _remove():
        backend.remove("v2")
        results = backend.search(dim3, k=10)
        ids = [r[0] for r in results]
        assert "v2" not in ids, f"Removed vector v2 still appears in search: {ids}"

    _check(report, "remove", _remove)

    # 2c. Size
    def _size():
        s = backend.size()
        assert s == 2, f"Expected size 2 after remove, got {s}"

    _check(report, "size", _size)

    # 2d. Dimension mismatch raises DimensionMismatchError
    def _dimension_mismatch():
        wrong_dim = [1.0, 0.0]  # 2D, not 3D
        try:
            backend.add("bad", wrong_dim)
            assert False, "Expected DimensionMismatchError"
        except DimensionMismatchError:
            pass

    _check(report, "dimension_mismatch_raises", _dimension_mismatch)

    # 2e. Metadata roundtrip
    def _metadata():
        meta = {"strategy": "increase_capacity", "reliability": 0.87}
        backend.add("v_meta", dim3, metadata=meta)
        got = backend.get_metadata("v_meta")
        assert got == meta, f"Metadata mismatch: expected {meta}, got {got}"

    _check(report, "metadata_roundtrip", _metadata)

    # 2f. k=None returns all results
    def _all_results():
        results = backend.search(dim3, k=None)
        assert len(results) >= 2, f"Expected ≥2 results for k=None, got {len(results)}"

    _check(report, "search_k_none", _all_results)

    # 2g. health_check
    def _health():
        h = backend.health_check()
        assert isinstance(h.healthy, bool)
        assert isinstance(h.latency_ms, float)

    _check(report, "health_check", _health)

    report.elapsed_ms = (time.monotonic() - t0) * 1000
    report.passed = (len(report.failures) == 0)
    return report

# ResearchForge-ECRM Changelog

All meaningful changes to the ResearchForge-ECRM system are documented here.
Failed experiments, negative results, and wrong decisions are **research evidence**
and must be preserved — they are not removed from this file.

Format follows [Keep a Changelog](https://keepachangelog.com).

---

## [RF-1.0.0-alpha.1] — 2026-09 — Infrastructure Release

**Character of release**: Infrastructure only. Zero changes to research algorithms.
Behavioral equivalence with RF-0.x is verified by the regression harness (8 tests
including re-execution of all 17 original tests, and exact numeric parity checks
for ECRM scoring algorithms).

### Added

**Adapter Layer** (`researchforge/adapters/`)
- `errors.py` — `BackendError` hierarchy: `BackendConnectionError`,
  `BackendNotFoundError`, `BackendSerializationError`, `BackendTransactionError`,
  `BackendCapabilityError`, `DimensionMismatchError`.
- `capabilities.py` — `BackendCapabilities`, `BackendInfo`, `HealthStatus`.
- `protocols.py` — `GraphBackend` and `VectorIndexBackend` Protocol base classes
  with complete docstring-level contracts (transaction protocol, dimension contract,
  metadata support). `CRITICAL RULE (AD-003)`: GraphBackend has NO knowledge of
  RDG semantic constraints. Domain validation stays in `rdg/graph.py`.
- `backends/memory.py` — `InMemoryGraphBackend`: zero-dep, transactional
  (copy-on-write rollback), health-check.
- `backends/sqlite.py` — `SQLiteGraphBackend`: WAL mode, persistent, full
  BEGIN/COMMIT/ROLLBACK transaction support. Current substitute for Neo4j.
- `backends/neo4j.py` — `Neo4jGraphBackend`: **documented production stub**.
  Raises `NotImplementedError` on instantiation. Includes the complete 9-step
  migration path to a live Neo4j instance.
- `backends/inprocess_vector.py` — `InProcessVectorIndex`: brute-force exact
  cosine similarity, dimension-contracted, metadata-aware. Replaces the ad-hoc
  inline cosine loop previously in `ecrm.py`.
- `backends/pgvector.py` — `PgvectorBackend`: **documented production stub**.
  Raises `NotImplementedError`. Includes complete PostgreSQL+pgvector migration path.
- `registry.py` — `AdapterRegistry` with `register_graph_backend()`,
  `create_graph_backend()`, `register_vector_backend()`, `create_vector_backend()`,
  `*_info()`, `list_*()`. Pre-populated default registry with `in_memory`, `sqlite`
  (graph), `inprocess` (vector). Stubs intentionally not registered.
- `validation.py` — Three-level validation harness: Level 1 Contract
  (CRUD + transaction lifecycle + error types), Level 2 Persistence
  (data survives close/reopen), Level 3 Integration (full RDG chain on
  top of backend + typed-edge constraint enforcement). `validate_graph_backend()`
  and `validate_vector_backend()` functions.

**Scientific Validity Gate** (`researchforge/scientific_validity/`)
- `verdicts.py` — `ValidityVerdict` (PASS/FAIL/WARNING/INCONCLUSIVE/
  REQUIRES_HUMAN_REVIEW), `ClaimEligibility`, `CheckSeverity`, `CheckResult`,
  `ValidityReport.finalize()` with correct blocker/warning aggregation.
- `leakage.py` — `DataLeakageDetector`: 4 checks: (A) exact duplicate samples,
  (B) near-duplicate (O(n²), optional), (C) target leakage (feature-label
  correlation), (D) index set overlap.
- `permutation.py` — `LabelPermutationTest`: permutes only training labels;
  tests against real test labels; reports `gap = real_metric - mean(perm_metrics)`.
- `baseline.py` — `BaselineFairnessValidator`: 6 checks: split size match,
  feature shape match, preprocessing match, training budget ratio, seed match,
  baseline above chance.
- `significance.py` — `StatisticalSignificanceTester`: paired t-test or
  Welch's unpaired; p-value approximation; Cohen's d effect size; 95%
  confidence intervals; correct PASS/FAIL/WARNING/INCONCLUSIVE verdict mapping.
- `gate.py` — `ScientificValidityGate`: orchestrates all checks, individual
  entry points (`run_leakage_check`, `run_permutation_check`,
  `run_significance_check`) and `run_standard_suite()`.
- `report.py` — `ValidityReportRenderer`: text (with icons, severity, blockers)
  and JSON renderers.

**Test suites**
- `tests/test_adapters.py` — 30 tests: Tier 1 Unit, Tier 2 Contract
  (validation harness), Tier 3 Integration (RDG + ECRM behavioral equivalence).
- `tests/test_scientific_validity.py` — 22 tests: all 4 check types, full gate
  integration, verdict aggregation, report rendering.
- `tests/test_regression.py` — 8 tests: backward API parity, typed-edge
  enforcement, evidence chain correctness, ECRM algorithm parity (exact numeric),
  ModelGenome pipeline parity, controller smoke (all 4 conditions), 17 original
  tests re-run via subprocess.

**Project state**
- `RESEARCHFORGE_STATE.yaml` — Complete project memory checkpoint: release
  manifest, implementation inventory (implemented/planned), benchmark history
  (RF-0.x digits and ECG results), research record (questions, hypotheses,
  findings, negative results), architectural decisions (AD-001 to AD-004),
  verification matrix, reproducibility, release history.

### Changed

**RDG (`researchforge/rdg/graph.py`)** — backward-compatible refactoring
- Added `backend=` constructor parameter (opt-in, RF-1.0+).
- Original `db_path=` API fully preserved and unchanged.
- `add_node()` and `add_edge()` now route persistence through
  `self._backend` instead of calling `_con.execute()` directly.
- `EdgeConstraintError` and all typed-relation rules remain in this class —
  not in the backend (AD-003).
- Added `_reload_from_backend()` for reconstructing the in-process traversal
  index from an existing persistent backend after process restart.
- Added `close()` that closes the backend.

**ECRM (`researchforge/memory/ecrm.py`)** — behavioral-equivalence refactoring
- `vector_backend=` constructor parameter added (opt-in, RF-1.0+).
- Original `db_path=` API fully preserved.
- Vector similarity search delegated to `VectorIndexBackend.search()` and
  `VectorIndexBackend.add()` instead of the inline cosine loop.
- `store()` now calls `self._vector_backend.add(rec.id, embedding, metadata=...)`.
- `query()` calls `self._vector_backend.search(qvec, k=None)` then applies
  archived-record filtering in Python.
- **ALL SCORING ALGORITHMS UNCHANGED** (AD-004): `strategy_stats()`,
  `negative_transfer_rate()`, `research_experience_score()`, `consolidate()`,
  `working_memory()`, `long_term_memory()`, `reallocate()`,
  `memory_half_life_days()`, `has_similar_failure()`.

### Preserved (from RF-0.x)

These are permanent research artifacts. They must not be altered:

- The trajectory-memory negative result (NR-001): trajectory-conditioned
  memory did NOT improve over flat ECRM at 25-generation budget on either
  task. FRR 0.63 (digits), 0.41 (ECG), same for both conditions.
- The RF-0.x benchmark results in `RESEARCHFORGE_STATE.yaml`.
- All 17 original tests (`tests/test_basic.py`).

### Known Limitations (unchanged from RF-0.x)

- SafeRunner does NOT provide filesystem/network isolation or cgroup limits.
- ECRM uses hashed bag-of-words embedding, not a semantic encoder (AD-002).
- Neo4j and pgvector backends are documented stubs (`NotImplementedError`).

### Acceptance Criteria — VERIFIED

| Criterion | Status |
|---|---|
| 17 RF-0.x tests pass | ✅ 17/17 |
| 30 adapter tests pass (Tier 1+2+3) | ✅ 30/30 |
| 22 validity gate tests pass (Tier 4) | ✅ 22/22 |
| 8 regression tests pass (Tier 5) | ✅ 8/8 |
| ECRM behavioral equivalence (exact numeric) | ✅ verified |
| RDG backward compat (db_path=) | ✅ verified |
| Neo4j stub: NotImplementedError | ✅ verified |
| pgvector stub: NotImplementedError | ✅ verified |
| Stubs not in default registry | ✅ verified |
| RESEARCHFORGE_STATE.yaml created | ✅ |
| NR-001 negative result preserved | ✅ in STATE.yaml |
| CHANGELOG.md | ✅ this file |

---

## [RF-0.x] — 2026-09 — Verified Baseline

**Character of release**: Reference implementation. The permanent experimental
control against which all future RF versions are compared.

### Implemented

- `ResearchDevelopmentGraph` (RDG): typed in-memory graph with SQLite mirror.
  10 node types, 10 edge relations, `EDGE_TYPE_CONSTRAINTS`, `evidence_chain()`.
- `ModelGenome` + 7 evolution operators (increase_capacity, decrease_capacity,
  change_family, tune_hyperparameters, change_pipeline, crossover_top2,
  mutate_random). Real sklearn Pipeline via `build_estimator()`.
- `ECRM`: `store()`, `query()`, `has_similar_failure()`, `strategy_stats()`,
  `negative_transfer_rate()`, `research_experience_score()`, `consolidate()`,
  `working_memory()`, `long_term_memory()`, `reallocate()`,
  `memory_half_life_days()`.
- `TrajectoryMemory`: context-conditioned on (strategy, model_type,
  capacity_bucket). RDG provenance links.
- `UCB PolicyLearner`.
- Rule-based failure diagnosis (5 categories).
- `SafeRunner` with process-level kill.
- `ResearchController` + pipeline (4 conditions).
- RDE-Bench, digits_task, synthetic_ecg_task.
- FastAPI REST layer (`/run`, `/benchmark`, `/memory`, `/rdg`, `/health`).
- GitHub retrieval (real GitHub API).
- MATLAB export (one-way, visualization only).
- SQLite persistence.

### Key Finding (NR-001)

Trajectory-memory (conditioned on strategy + model_type + capacity_bucket)
did NOT improve over flat ECRM at 25-generation budget:
- digits: full=0.870, trajectory=0.8707, no_memory=0.8705. FRR 0.63 for both.
- ECG: full=0.722, trajectory=0.708, no_memory=0.707. FRR 0.41 for both.

**Root cause hypothesis**: context fragmentation. Too many sparse evidence
bins at 25-generation budget. Motivates adaptive backoff (RF-1.0.0-alpha.4).

### Tests

17 tests. All pass.

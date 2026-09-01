# ResearchForge-ECRM — reference implementation

A working, tested implementation of the ResearchForge-ECRM architecture: a
typed Research Development Graph (RDG), an executable Model Genome, an
Evidence- and Outcome-Conditioned Research Memory (ECRM), a policy learner,
failure diagnosis, a real safety sandbox, real GitHub retrieval, a REST API,
database schemas, and a MATLAB export/visualization layer.

**Companion document:** `ResearchForge-ECRM_Documentation.docx` (delivered
alongside this code) is the full write-up — architecture diagrams generated
from this actual source, the research-gap rationale, every result below with
its methodology, and an honest discussion section that also addresses a set
of much more elaborate "RF-1.0" proposals that circulated during this
project's development and were found, on inspection, to rest on fabricated
benchmark numbers. Read that document for the complete narrative; this file
covers running and extending the code.

**What this is, honestly.** The source design docs scope a 2-semester,
~4-person project against a GPU cluster, Neo4j, Milvus/pgvector, and licensed
LLM/literature APIs. This is a single-machine reference implementation: every
architectural piece is implemented and actually runs, but sized for one
machine with no external services. Nothing here prints fabricated numbers —
every accuracy figure came from an actual `fit()`/`predict()` call, and every
claim about the safety sandbox or the MATLAB scripts was verified by actually
executing them, not asserted from familiarity with the APIs.

## Quickstart

```bash
pip install -r requirements.txt

python run_demo.py                             # ~30s: full RDE-Bench ablation report
python -m pytest tests/ -q                      # or: python tests/test_basic.py  (17 tests)

python export_to_matlab.py                      # produces matlab_data/*.mat
cd matlab && octave-cli --eval "visualize_ecg_task"        # or run inside MATLAB
cd matlab && octave-cli --eval "plot_rde_bench_results"

uvicorn researchforge.api.server:app --reload --app-dir .  # REST API on :8000
```

## Architecture map

| Design doc concept | Where it lives | What's real vs. simplified |
|---|---|---|
| Research Development Graph (RDG) | `rdg/graph.py`, `rdg/schema.py` | Real typed graph + traversal (`evidence_chain`, `children`/`parents`) with enforced edge-type constraints. Neo4j → in-memory graph + SQLite mirror; same public API either way. |
| Model Genome | `genome/model_genome.py` | Real JSON-Schema-validated genome. `build_estimator()` turns it into an actual trainable scikit-learn `Pipeline`. `safety_check()` is a separate resource-bound pre-flight check (distinct from schema validation). |
| Evolution operators | `genome/operators.py` | 7 real operators (mutation, capacity/regularization changes, crossover, family-switch, feature-pipeline mutation) acting on genuine hyperparameters. |
| ECRM (flat memory) | `memory/ecrm.py`, `memory/embeddings.py` | Real write/query/forget loop, RES scoring, negative-transfer tracking, plus short-term/long-term **hierarchical tiering** with a `reallocate()` operation (see below). |
| Trajectory memory (contextual) | `memory/trajectory.py` | A second, independent memory design conditioning retrieval on the genome's actual capacity regime, not just its model family. Benchmarked against the flat ECRM as a 4th RDE-Bench condition — see Results. |
| Failure taxonomy | `diagnosis/failure_taxonomy.py` | Real rule-based diagnosis (execution error / overfitting / underfitting / low-performance / divergence) from actual train vs. validation metrics. |
| Policy learner | `policy/policy_learner.py` | UCB bandit + incremental value update, matching the design doc's Sec. 4.1/4.4 formulas. Supports both a binary failure-check (flat memory) and a continuous score multiplier (trajectory memory). |
| Algorithm-discovery pipeline | `pipeline/discovery.py` | `HeuristicSynthesizer` (real, deterministic) implements retrieve→recombine→synthesize→test. `LLMSynthesizer` is a documented, unimplemented extension point. |
| Research loop / controller | `pipeline/controller.py` | The actual select→synthesize→run→diagnose→remember→learn loop, plus the 4-condition RDE-Bench ablation (`full` / `trajectory_memory` / `no_memory` / `random`). |
| Experiment sandbox | `safety/sandbox.py` | Runs each experiment in a forked subprocess and calls `terminate()`/`kill()` on timeout — verified to actually kill a `time.sleep(5)` call under a 0.5s budget in under 2s wall-clock. |
| Retrieval | `retrieval/literature.py` | `GitHubRepositoryRetriever` makes a real `api.github.com/search/repositories` call and parses real results, verified live. Degrades to an empty list on any network failure. |
| MATLAB interop | `interop/matlab_export.py`, `matlab/*.m` | Real `.mat` exports via `scipy.io.savemat`, round-trip-verified. Both `.m` scripts were actually executed in GNU Octave (installed for this purpose) — one had a genuine Octave-compatibility bug, found and fixed by running it, not by inspection. |
| RDE-Bench | `benchmarks/tasks.py`, `benchmarks/rde_bench.py` | Two real, no-network tasks and all 6 metrics from the design doc (RE, SE, FRR, MU, NTR, memory half-life), computed from actual experiment logs. |
| REST API | `api/server.py` | `/experiments`, `/experiments/{id}/run` (executes inside the safety sandbox), `/models/mutate`, `/memory/query`, `/insights`, `/safety/status`. Smoke-tested end to end with FastAPI's `TestClient`. |
| DB schemas | `db/schema.sql`, `db/init_db.py` | `schema.sql` is the Postgres+pgvector reference schema, including an `experiment_runs` safety/audit table. `init_db.py` is what the code actually runs against (SQLite). |

## Real results (5 seeds × 25 generations, `python run_demo.py`)

Four conditions are compared: `full` (flat ECRM), `trajectory_memory`
(contextual ECRM, Section 7.2 of the docx), `no_memory`, and `random`.

**Headline finding 1 — flat memory works, modestly:** `full` consistently
achieves the lowest Failure Repetition Rate on both tasks, and a
small-to-modest improvement in best-found accuracy on the noisier task.

**Headline finding 2 — richer context alone didn't help here, and the real
reason was measured:** `trajectory_memory` tracks close to `no_memory` on
FRR, not close to `full`. Splitting evidence by (strategy, model family,
capacity bucket) fragments a 25-generation run across far more distinct
contexts than the flat design's coarser grouping — on `digits`, 6 of 13
distinct contexts never accumulate 2+ samples within the budget, so the
system spends much of the run on a neutral fallback. This is a genuine
bias-variance finding: more specific context conditioning needs more data
than a modest generation budget supplies, not an automatic win. Full detail,
including the diagnostic numbers, is in the docx's Section 7.2.

Re-run with different seeds/scale: `python run_demo.py --seeds 10 --generations 40`.

## What's a faithful implementation vs. a deliberate simplification

See the docx's Section 4 and Section 9.3 for the full accounting. Briefly:
Neo4j → SQLite/in-memory graph; Postgres+pgvector → SQLite + in-process
cosine similarity; an LLM/sentence encoder → hashed bag-of-words; a live LLM
in the discovery loop → `HeuristicSynthesizer` (with `LLMSynthesizer` as a
documented, unwired extension point); Docker sandboxing → real process
timeout/kill without filesystem/network isolation; CIFAR-10/PhysioNet →
two smaller, harder, no-network tasks tuned to have genuine headroom.

## Repo layout

```
researchforge_ecrm/
├── README.md, requirements.txt
├── run_demo.py                  # end-to-end RDE-Bench ablation demo
├── export_to_matlab.py          # produces matlab_data/*.mat
├── demo_results.json            # output of the run captured for the docx
├── researchforge/
│   ├── rdg/         graph.py, schema.py
│   ├── genome/      model_genome.py, operators.py
│   ├── memory/      ecrm.py, embeddings.py, trajectory.py
│   ├── policy/      policy_learner.py
│   ├── diagnosis/   failure_taxonomy.py
│   ├── evaluators/  sklearn_evaluator.py
│   ├── safety/      sandbox.py
│   ├── retrieval/   literature.py
│   ├── interop/     matlab_export.py
│   ├── pipeline/    discovery.py, controller.py
│   ├── benchmarks/  tasks.py, rde_bench.py
│   ├── api/         server.py
│   └── db/          schema.sql, init_db.py
├── matlab/          visualize_ecg_task.m, plot_rde_bench_results.m
├── matlab_data/     digits_task.mat, synthetic_ecg_task.mat, rde_bench_results.mat
└── tests/           test_basic.py  (17 tests, all passing)
```

## Extending toward the full design

1. **Postgres+pgvector**: apply `db/schema.sql`; change `ECRM`'s similarity
   search to `<=>`-operator SQL instead of computing cosine similarity in Python.
2. **Neo4j**: replace `ResearchDevelopmentGraph`'s `_persist_node`/`_persist_edge`
   with Cypher calls; keep the public `add_node`/`add_edge`/`evidence_chain`
   interface so nothing above the graph layer changes.
3. **Wire up `LLMSynthesizer`**: implement `synthesize()` with a real API
   call as sketched in its docstring; validate against `GENOME_SCHEMA` before use.
4. **Try a hybrid memory**: fall back to the flat ECRM's coarser grouping
   when a trajectory context is under-populated, rather than a fixed neutral
   default — the most direct next experiment suggested by Section 7.2's result.
5. **Real datasets**: point `benchmarks/tasks.py` at PhysioNet/MIT-BIH data
   or your own ECG-DigitizeNet pipeline output.
6. **If pursuing any further "RF-1.x" ideas** (evidence adjudication, a mode
   router, causal failure diagnosis): build and benchmark each one
   individually against this repo's real baseline, the way trajectory memory
   was — one verified increment at a time.

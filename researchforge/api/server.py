"""REST API surface matching ResearchForge-ECRM's "APIs and Endpoints" section
(exec summary Sec. 5 / technical report Sec. 10), extended with the two
pieces the design doc's Sec. 7 sandbox implies but the original endpoint list
didn't spell out: an endpoint that actually *runs* an experiment (as opposed
to just registering one), and an endpoint to inspect the safety budget.

    POST /experiments              register a hypothesis + base model
    GET  /experiments/{id}         fetch an experiment's RDG node
    POST /experiments/{id}/run     execute it inside the safety sandbox,
                                    diagnose the outcome, extend the RDG,
                                    write an ECRM memory record
    POST /models/mutate            apply an evolution operator
    GET  /memory/query             semantic search over ECRM
    POST /insights                 record a new insight, linked to RDG nodes
    GET  /safety/status            current SafeRunner budget/kill-switch state
    GET  /health

Run with:   uvicorn researchforge.api.server:app --reload --app-dir .

This wraps a single in-memory RDG/ECRM/genome-store per process -- enough to
exercise the wire protocol end-to-end. A persistent deployment would swap the
module-level stores below for the same SQLite-backed ones the
ResearchController uses (rdg.graph.ResearchDevelopmentGraph(db_path=...),
memory.ecrm.ECRM(db_path=...)) without changing any route signature.
"""
from __future__ import annotations
import random
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..genome.model_genome import ModelGenome
from ..genome.operators import apply_strategy, STRATEGIES
from ..memory.ecrm import ECRM
from ..rdg.graph import ResearchDevelopmentGraph
from ..evaluators.sklearn_evaluator import evaluate_genome
from ..diagnosis.failure_taxonomy import diagnose, FailureCategory
from ..safety.sandbox import SafeRunner, ResourceBudget, SafetyStatus
from ..benchmarks.tasks import digits_task

MODEL_TYPES = ("MLPClassifier", "RandomForestClassifier", "SVC", "LogisticRegression")

app = FastAPI(
    title="ResearchForge-ECRM API",
    version="0.1.0",
    description="Reference implementation of the RDG/ECRM/Model-Genome API "
                "described in the ResearchForge-ECRM design document.",
)

_rdg = ResearchDevelopmentGraph()
_ecrm = ECRM()
_genomes: Dict[str, ModelGenome] = {}
_rng = random.Random(0)
_task = digits_task(seed=0)  # fixed reference task for API-driven /run calls
_safe_runner = SafeRunner(ResourceBudget(
    max_experiments=1000, max_wall_time_s=1800.0, per_experiment_timeout_s=15.0))


class ExperimentRequest(BaseModel):
    hypothesis_text: str = Field(..., min_length=1, max_length=2000)
    model_type: str = "LogisticRegression"


class MutateRequest(BaseModel):
    model_id: str
    strategy: str


class InsightRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    related_node_ids: Optional[List[str]] = Field(default=None, max_length=50)


@app.post("/experiments")
def create_experiment(req: ExperimentRequest) -> Dict[str, Any]:
    if req.model_type not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"model_type must be one of {MODEL_TYPES}")
    hyp = _rdg.add_node("Hypothesis", req.hypothesis_text)
    genome = ModelGenome.default(req.model_type, seed=_rng.randint(0, 10_000))
    _genomes[genome.model_id] = genome
    exp = _rdg.add_node("Experiment", f"pending evaluation of {genome.model_id}",
                         attributes={"genome_id": genome.model_id, "status": "pending"})
    _rdg.add_edge(hyp.id, exp.id, "tested-by")
    return {"hypothesis_id": hyp.id, "experiment_id": exp.id, "model_id": genome.model_id}


@app.get("/experiments/{exp_id}")
def get_experiment(exp_id: str) -> Dict[str, Any]:
    node = _rdg.nodes.get(exp_id)
    if node is None or node.type != "Experiment":
        raise HTTPException(status_code=404, detail="experiment not found")
    return node.to_dict()


@app.post("/experiments/{exp_id}/run")
def run_experiment(exp_id: str) -> Dict[str, Any]:
    """Actually execute the experiment: build + train + evaluate the genome
    attached to this Experiment node inside the safety sandbox (real process
    isolation, a hard timeout, and a tracked resource budget -- see
    safety.sandbox.SafeRunner), diagnose the outcome, extend the RDG with a
    Finding (and a Claim on success), and write an ECRM memory record. This
    is the same select->run->diagnose->remember sequence
    pipeline.controller.ResearchController runs internally, exposed one step
    at a time over HTTP against a fixed reference task (digits)."""
    node = _rdg.nodes.get(exp_id)
    if node is None or node.type != "Experiment":
        raise HTTPException(status_code=404, detail="experiment not found")
    genome = _genomes.get(node.attributes.get("genome_id"))
    if genome is None:
        raise HTTPException(status_code=404, detail="genome for this experiment not found")

    violations = genome.safety_check()
    if violations:
        node.attributes["status"] = "rejected"
        raise HTTPException(status_code=422, detail={"safety_violations": violations})

    outcome = _safe_runner.run(evaluate_genome, genome, _task.X_train, _task.y_train,
                                _task.X_val, _task.y_val, _task.metric_fn,
                                target=_task.target_metric)
    if outcome.status != SafetyStatus.OK:
        node.attributes["status"] = "failed"
        node.attributes["safety_status"] = outcome.status.value
        code = 503 if outcome.status == SafetyStatus.BUDGET_EXHAUSTED else 500
        raise HTTPException(status_code=code,
                             detail={"safety_status": outcome.status.value, "error": outcome.error})

    exp_result = outcome.value
    failure = diagnose(exp_result)
    node.attributes["status"] = "completed"

    finding = _rdg.add_node("Finding", f"metric={exp_result.metric:.4f} failure={failure.value}",
                             attributes={"metric": exp_result.metric, "failure": failure.value})
    _rdg.add_edge(exp_id, finding.id, "produces")
    claim_id = None
    if failure == FailureCategory.NONE:
        claim = _rdg.add_node("Claim", f"Genome {genome.model_id} reached {exp_result.metric:.4f}")
        _rdg.add_edge(finding.id, claim.id, "supports")
        claim_id = claim.id

    _ecrm.store(text_summary=f"api_run {genome.model_type} {_task.name}",
                context={"task": _task.name, "genome": genome.to_dict()},
                outcome={"metric": exp_result.metric, "success": failure == FailureCategory.NONE,
                         "failure": failure.value},
                strategy="api_run")

    return {"experiment_id": exp_id, "finding_id": finding.id, "claim_id": claim_id,
            "metric": exp_result.metric, "failure": failure.value,
            "duration_s": round(outcome.duration_s, 4)}


@app.post("/models/mutate")
def mutate_model(req: MutateRequest) -> Dict[str, Any]:
    if req.model_id not in _genomes:
        raise HTTPException(status_code=404, detail="model not found")
    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"strategy must be one of {STRATEGIES}")
    parent = _genomes[req.model_id]
    child = apply_strategy(req.strategy, parent, _rng, population=list(_genomes.values()))
    if violations := child.safety_check():
        raise HTTPException(status_code=422, detail={"safety_violations": violations})
    _genomes[child.model_id] = child
    return child.to_dict()


@app.get("/memory/query")
def query_memory(text: str = Query(..., min_length=1, max_length=500),
                  k: int = Query(default=5, ge=1, le=20)) -> Dict[str, Any]:
    recs = _ecrm.query(text, k=k)
    return {"results": [{"id": r.id, "text_summary": r.text_summary, "outcome": r.outcome,
                          "strategy": r.strategy} for r in recs]}


@app.post("/insights")
def create_insight(req: InsightRequest) -> Dict[str, Any]:
    node = _rdg.add_node("Insight", req.text)
    for rid in (req.related_node_ids or []):
        if rid in _rdg.nodes:
            try:
                _rdg.add_edge(rid, node.id, "informs", enforce_types=False)
            except Exception:
                pass  # best-effort linking; the insight itself is still recorded
    return node.to_dict()


@app.get("/safety/status")
def safety_status() -> Dict[str, Any]:
    return _safe_runner.status_report()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

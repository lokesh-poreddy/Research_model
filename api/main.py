"""
FastAPI REST API for ResearchForge-ECRM.
Provides endpoints to control the research loop remotely.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.controller_agent import ResearchController
from ecrm.memory_store import ECRMMemoryStore
from evolution.genome import ModelGenome
from evolution.operators import OperatorType, apply_operator
from rdg.graph import ResearchDevelopmentGraph
from rdg.nodes import RDGNode
from config.settings import settings

logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
_rdg: ResearchDevelopmentGraph = ResearchDevelopmentGraph()
_memory: ECRMMemoryStore = ECRMMemoryStore()
_controller: Optional[ResearchController] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    logger.info("ResearchForge-ECRM API starting.")
    yield
    logger.info("ResearchForge-ECRM API shutting down.")


app = FastAPI(
    title="ResearchForge-ECRM API",
    description="Evidence- and Outcome-Conditioned Research Development Graph API",
    version=settings.version,
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class StartResearchRequest(BaseModel):
    problem: str
    n_iterations: int = 10
    mock: bool = True


class MutateModelRequest(BaseModel):
    genome_json: Dict[str, Any]
    operator: str = "param_mutation"
    delta: float = 0.1


class MemoryQueryRequest(BaseModel):
    query: str
    top_k: int = 5


class MemoryStoreRequest(BaseModel):
    text: str
    outcome: Dict[str, Any]
    link_node: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check() -> Dict[str, str]:
    return {"status": "ok", "version": settings.version}


@app.get("/rdg/stats", tags=["RDG"])
def rdg_stats() -> Dict[str, Any]:
    """Return current RDG statistics."""
    return _rdg.stats()


@app.get("/rdg/nodes", tags=["RDG"])
def list_rdg_nodes(node_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """List RDG nodes, optionally filtered by type."""
    nodes = list(_rdg)
    if node_type:
        nodes = [n for n in nodes if n.type.value.lower() == node_type.lower()]
    return [n.to_dict() for n in nodes]


@app.get("/rdg/nodes/{node_id}", tags=["RDG"])
def get_rdg_node(node_id: str) -> Dict[str, Any]:
    node = _rdg.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node.to_dict()


@app.post("/research/start", tags=["Research"])
def start_research(req: StartResearchRequest) -> Dict[str, Any]:
    """Start a new research run (synchronous, blocks until done)."""
    global _controller

    # Reset graph and memory
    global _rdg, _memory
    _rdg = ResearchDevelopmentGraph()
    _memory = ECRMMemoryStore()

    # Add seed nodes
    problem_node = RDGNode.problem(content=req.problem)
    _rdg.add_node(problem_node)
    gap_node = RDGNode.gap(content=f"Gap: How to solve: {req.problem}")
    _rdg.add_node(gap_node)
    from rdg.edges import EdgeRelation
    _rdg.connect(problem_node.id, gap_node.id, EdgeRelation.IDENTIFIES, validate=False)

    _controller = ResearchController(
        rdg=_rdg,
        memory=_memory,
        problem_description=req.problem,
        use_mock_experiments=req.mock,
    )
    result = _controller.run(n_iterations=req.n_iterations)
    return result


@app.post("/models/mutate", tags=["Evolution"])
def mutate_model(req: MutateModelRequest) -> Dict[str, Any]:
    """Apply an evolution operator to a model genome."""
    try:
        genome = ModelGenome.from_dict(req.genome_json)
        op = OperatorType(req.operator)
        mutated = apply_operator(op, genome, delta=req.delta)
        return mutated.to_dict()
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/memory/store", tags=["Memory"])
def store_memory(req: MemoryStoreRequest) -> Dict[str, str]:
    """Store a new memory record in ECRM."""
    record = _memory.store(
        text=req.text,
        outcome=req.outcome,
        link_node=req.link_node,
    )
    return {"record_id": record.record_id}


@app.post("/memory/query", tags=["Memory"])
def query_memory(req: MemoryQueryRequest) -> List[Dict[str, Any]]:
    """Semantic search in ECRM."""
    results = _memory.retrieve(req.query, top_k=req.top_k)
    return [
        {
            "record_id": rec.record_id,
            "text": rec.text,
            "similarity": float(sim),
            "outcome": rec.outcome,
            "failure_flags": rec.failure_flags,
        }
        for rec, sim in results
    ]


@app.get("/memory/stats", tags=["Memory"])
def memory_stats() -> Dict[str, Any]:
    return _memory.stats()


@app.post("/memory/consolidate", tags=["Memory"])
def consolidate_memory() -> Dict[str, int]:
    removed = _memory.consolidate()
    return {"records_removed": removed}

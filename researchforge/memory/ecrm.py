"""Evidence- and Outcome-Conditioned Research Memory (ECRM).

Implements the write-manage-read memory loop from ResearchForge-ECRM Sec. 1
("Research Memory Schema") and Sec. 4 of the technical report:

  - store()                        write a MemoryRecord (context + outcome)
  - query()                        cosine-similarity retrieval, top-k
  - has_similar_failure()          the memory check in Sec. 4.1's select_branch
  - strategy_stats()               mu(y), sigma(y), n_trials for RES(h,G)
  - negative_transfer_rate()       NTR(h,G)
  - research_experience_score()    RES(h,G) = alpha*Reliability + beta/(1+NTR) - gamma*uncertainty
  - consolidate()                  P_retain = exp(-lambda*age) * reliability forgetting policy
  - memory_half_life_days()        ln(2)/lambda, the analytic half-life of that decay

Also implements the "layered memory model" from Sec. 1 of the design doc
("Short-term memory holds the current RDG and recent evidence, updated each
cycle. Long-term memory is the vector store of experiences"): every record
starts in the `short_term` tier; surviving `promotion_threshold` consolidation
passes without being archived promotes it to `long_term`. `reallocate()`
recomputes every record's tier from the full current history in one pass
(rather than only advancing tier counters incrementally at each scheduled
consolidate() call) -- useful after decay/threshold parameters change, or
before a report is generated, so tiering reflects the complete picture.

Backend refactoring (RF-1.0.0-alpha.1, AD-004)
------------------------------------------------
Vector similarity search is now delegated to a VectorIndexBackend
(adapters/backends/inprocess_vector.py by default). This replaces the
ad-hoc cosine-similarity loop that was previously inlined here.

CRITICAL: NO scoring algorithms changed. The following are IDENTICAL to RF-0.x:
  - strategy_stats()
  - negative_transfer_rate()
  - research_experience_score()
  - consolidate() / retention probability
  - working_memory() / long_term_memory() / reallocate()
  - has_similar_failure() threshold and logic

Behavioral equivalence is verified by tests/test_adapters.py's
test_ecrm_behavioral_equivalence() which runs both implementations on
identical data and asserts matching retrieval order and scores.

SQLite persistence for MemoryRecord metadata is unchanged from RF-0.x.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .embeddings import embed, cosine_sim
from ..adapters.backends.inprocess_vector import InProcessVectorIndex
from ..adapters.protocols import VectorIndexBackend


@dataclass
class MemoryRecord:
    id: str
    text_summary: str
    embedding: list
    context: Dict[str, Any]
    outcome: Dict[str, Any]
    strategy: str
    created_at: float = field(default_factory=time.time)
    tier: str = "short_term"              # "short_term" (working) | "long_term" (consolidated)
    consolidation_passes_survived: int = 0
    archived: bool = False
    retrieval_count: int = 0
    negative_transfer_count: int = 0


class ECRM:
    """In-process ECRM with optional SQLite mirroring and a pluggable
    vector-search backend (VectorIndexBackend).

    Constructor
    -----------
    ECRM()                                # pure in-process, no persistence
    ECRM(db_path="memory.db")            # RF-0.x API — unchanged, SQLite mirror
    ECRM(vector_backend=MyBackend())     # RF-1.0+ explicit vector backend

    Behavioral equivalence guarantee (AD-004)
    -----------------------------------------
    All scoring algorithms (RES, NTR, consolidate, forgetting) are identical
    to RF-0.x. The only change is that cosine similarity search is now routed
    through self._vector_backend instead of being computed inline.
    """

    def __init__(self, db_path: Optional[str] = None,
                 decay_lambda: float = 0.08,
                 retention_threshold: float = 0.12,
                 promotion_threshold: int = 2,
                 vector_backend: Optional[VectorIndexBackend] = None) -> None:
        self.records: Dict[str, MemoryRecord] = {}
        self.decay_lambda = decay_lambda
        self.retention_threshold = retention_threshold
        self.promotion_threshold = promotion_threshold
        self.db_path = db_path
        self._vector_backend: VectorIndexBackend = (
            vector_backend if vector_backend is not None
            else InProcessVectorIndex()
        )
        self._con = sqlite3.connect(db_path) if db_path else None
        if self._con:
            self._con.execute(
                """CREATE TABLE IF NOT EXISTS memory_records(
                       id TEXT PRIMARY KEY, text_summary TEXT, context TEXT,
                       outcome TEXT, strategy TEXT, created_at REAL,
                       archived INTEGER, retrieval_count INTEGER,
                       negative_transfer_count INTEGER, tier TEXT,
                       consolidation_passes_survived INTEGER)""")
            self._con.commit()

    # -- write ----------------------------------------------------------------
    def store(self, text_summary: str, context: Dict[str, Any],
              outcome: Dict[str, Any], strategy: str) -> MemoryRecord:
        embedding = embed(text_summary)
        rec = MemoryRecord(
            id=f"m_{uuid.uuid4().hex[:10]}",
            text_summary=text_summary,
            embedding=embedding.tolist(),
            context=context, outcome=outcome, strategy=strategy)
        self.records[rec.id] = rec
        self._vector_backend.add(
            rec.id, rec.embedding,
            metadata={"strategy": strategy,
                       "success": outcome.get("success", True),
                       "tier": rec.tier})
        self._persist(rec)
        return rec

    # -- read -----------------------------------------------------------------
    def query(self, text: str, k: int = 5,
              include_archived: bool = False) -> List[MemoryRecord]:
        qvec = embed(text)
        # Ask the backend for every scored match; archived filtering happens
        # here because the backend does not know about archiving.
        all_scored = self._vector_backend.search(qvec, k=None)
        top: List[MemoryRecord] = []
        for rid, _score in all_scored:
            rec = self.records.get(rid)
            if rec is None:
                continue
            if rec.archived and not include_archived:
                continue
            top.append(rec)
            if len(top) >= k:
                break
        for r in top:
            r.retrieval_count += 1
        return top

    def has_similar_failure(self, text: str, threshold: float = 0.6) -> bool:
        """The memory check in Sec. 4.1's select_branch pseudocode. Searches
        *including archived* records: forgetting (consolidate()) prunes
        low-reliability records from general retrieval, but failure evidence
        stays checkable here even after archiving -- negative evidence
        matters as much as positive evidence (design doc Sec. "Fourth gap").

        Algorithm unchanged from RF-0.x (AD-004).
        """
        qvec = embed(text)
        for rec in self.query(text, k=5, include_archived=True):
            if not rec.outcome.get("success", True):
                if cosine_sim(qvec, rec.embedding) >= threshold:
                    return True
        return False

    def flag_negative_transfer(self, strategy: str) -> None:
        """Mark the most recently retrieved record of a strategy family as
        having led to a worse-than-expected outcome when reused (Sec. 4
        negative-transfer detection). Unchanged from RF-0.x.
        """
        candidates = [r for r in self.records.values()
                      if r.strategy == strategy and r.retrieval_count > 0]
        if not candidates:
            return
        rec = max(candidates, key=lambda r: r.created_at)
        rec.negative_transfer_count += 1
        self._persist(rec)

    def strategy_stats(self, strategy: str) -> Dict[str, float]:
        """mu(y), sigma(y), n_trials for a strategy family -- the terms in
        RES(h,G) (Sec. 4 of the technical report). Unchanged from RF-0.x.
        """
        vals = [r.outcome.get("metric", 0.0) for r in self.records.values()
                 if r.strategy == strategy and not r.archived]
        if not vals:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {"mean": mean, "std": var ** 0.5, "n": len(vals)}

    def negative_transfer_rate(self, strategy: str) -> float:
        """Unchanged from RF-0.x."""
        recs = [r for r in self.records.values() if r.strategy == strategy]
        transferred = [r for r in recs if r.retrieval_count > 0]
        if not transferred:
            return 0.0
        bad = sum(1 for r in transferred if r.negative_transfer_count > 0)
        return bad / len(transferred)

    def research_experience_score(self, strategy: str, uncertainty: float = 0.0,
                                    alpha: float = 1.0, beta: float = 1.0,
                                    gamma: float = 0.5) -> float:
        """RES(h,G) = alpha*Reliability + beta/(1+NTR) - gamma*uncertainty
        (technical report Sec. 4). Unchanged from RF-0.x.
        """
        stats = self.strategy_stats(strategy)
        reliability = stats["mean"] / (stats["std"] + 0.01) if stats["n"] else 0.0
        ntr = self.negative_transfer_rate(strategy)
        return alpha * reliability + beta / (1 + ntr) - gamma * uncertainty

    # -- forgetting -----------------------------------------------------------
    def consolidate(self, now: Optional[float] = None) -> int:
        """Archive low-utility records: P_retain = exp(-lambda*age) *
        reliability (technical report Sec. 4 'Memory Consolidation /
        Forgetting'). Algorithm unchanged from RF-0.x (AD-004).
        """
        now = now if now is not None else time.time()
        archived = 0
        for rec in self.records.values():
            if rec.archived:
                continue
            age_days = max(0.0, (now - rec.created_at) / 86400.0)
            stats = self.strategy_stats(rec.strategy)
            reliability = min(1.0, max(0.0, stats["mean"])) if stats["n"] else 0.0
            retain_prob = math.exp(-self.decay_lambda * age_days) * max(reliability, 0.05)
            if retain_prob < self.retention_threshold:
                rec.archived = True
                archived += 1
                self._persist(rec)
            else:
                rec.consolidation_passes_survived += 1
                if (rec.tier == "short_term"
                        and rec.consolidation_passes_survived >= self.promotion_threshold):
                    rec.tier = "long_term"
                self._persist(rec)
        return archived

    def working_memory(self) -> List[MemoryRecord]:
        """Short-term / working memory. Unchanged from RF-0.x."""
        return [r for r in self.records.values()
                if r.tier == "short_term" and not r.archived]

    def long_term_memory(self) -> List[MemoryRecord]:
        """Long-term memory. Unchanged from RF-0.x."""
        return [r for r in self.records.values()
                if r.tier == "long_term" and not r.archived]

    def reallocate(self) -> Dict[str, int]:
        """Recompute every record's tier from the complete history.
        Unchanged from RF-0.x (AD-004).
        """
        promoted, demoted = 0, 0
        for rec in self.records.values():
            if rec.archived:
                continue
            stats = self.strategy_stats(rec.strategy)
            reliability = min(1.0, max(0.0, stats["mean"])) if stats["n"] else 0.0
            durable = (reliability >= self.retention_threshold * 2
                       and (rec.consolidation_passes_survived >= self.promotion_threshold
                            or stats["n"] >= self.promotion_threshold))
            if durable and rec.tier == "short_term":
                rec.tier = "long_term"
                promoted += 1
                self._persist(rec)
            elif not durable and rec.tier == "long_term":
                rec.tier = "short_term"
                demoted += 1
                self._persist(rec)
        return {
            "promoted_to_long_term": promoted,
            "demoted_to_short_term": demoted,
            "short_term_count": len(self.working_memory()),
            "long_term_count": len(self.long_term_memory()),
        }

    def memory_half_life_days(self) -> float:
        """Analytic half-life of the decay component: ln(2)/lambda. Unchanged."""
        return math.log(2) / self.decay_lambda

    # -- persistence ----------------------------------------------------------
    def _persist(self, rec: MemoryRecord) -> None:
        if not self._con:
            return
        self._con.execute(
            "INSERT OR REPLACE INTO memory_records VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (rec.id, rec.text_summary, json.dumps(rec.context),
             json.dumps(rec.outcome), rec.strategy, rec.created_at,
             int(rec.archived), rec.retrieval_count,
             rec.negative_transfer_count, rec.tier,
             rec.consolidation_passes_survived))
        self._con.commit()

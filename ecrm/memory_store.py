"""
ECRM Memory Store.

Stores past research episodes as MemoryRecords with:
  - Dense vector embeddings (FAISS index or in-memory fallback)
  - Outcome metadata, RES score, failure flags
  - Consolidation / pruning on every write
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ecrm.consolidation import prune_records
from ecrm.embedder import cosine_similarity, embed_text
from ecrm.negative_transfer import NTRDetector
from ecrm.res_scorer import compute_res

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    """A single entry in the ECRM."""

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""                          # Natural-language summary
    embedding: Optional[np.ndarray] = None  # Dense vector
    outcome: Dict[str, Any] = field(default_factory=dict)  # score, success, …
    link_node: str = ""                     # RDG node ID
    failure_flags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reliability: float = 0.0               # Computed by RES scorer
    domain: str = ""
    task_id: str = ""
    hypothesis_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "text": self.text,
            "embedding": self.embedding.tolist() if self.embedding is not None else [],
            "outcome": self.outcome,
            "link_node": self.link_node,
            "failure_flags": self.failure_flags,
            "timestamp": self.timestamp.isoformat(),
            "reliability": self.reliability,
            "domain": self.domain,
            "task_id": self.task_id,
            "hypothesis_id": self.hypothesis_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryRecord":
        emb_raw = d.get("embedding", [])
        embedding = np.array(emb_raw, dtype=np.float32) if emb_raw else None
        return cls(
            record_id=d.get("record_id", str(uuid.uuid4())),
            text=d.get("text", ""),
            embedding=embedding,
            outcome=d.get("outcome", {}),
            link_node=d.get("link_node", ""),
            failure_flags=d.get("failure_flags", []),
            timestamp=datetime.fromisoformat(d.get("timestamp", datetime.now(timezone.utc).isoformat())),
            reliability=d.get("reliability", 0.0),
            domain=d.get("domain", ""),
            task_id=d.get("task_id", ""),
            hypothesis_id=d.get("hypothesis_id", ""),
        )


class ECRMMemoryStore:
    """
    Long-lived memory for ResearchForge-ECRM.

    Write → Embed → Store (FAISS or in-memory list)
    Retrieve → Query vector → Return top-K records
    Consolidate → Prune low-retention records periodically
    """

    def __init__(
        self,
        dim: int = 384,
        retain_lambda: float = 0.01,
        retain_threshold: float = 0.1,
        max_records: int = 10_000,
        ntr_threshold: float = 0.3,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self._dim = dim
        self._retain_lambda = retain_lambda
        self._retain_threshold = retain_threshold
        self._max_records = max_records
        self._embedding_model = embedding_model
        self._records: List[MemoryRecord] = []
        self._ntr_detector = NTRDetector(threshold=ntr_threshold)
        self._use_faiss = False
        self._faiss_index = None
        self._faiss_id_map: List[str] = []

        self._init_faiss()

    # ── FAISS init ────────────────────────────────────────────────────────────

    def _init_faiss(self) -> None:
        try:
            import faiss  # type: ignore
            self._faiss_index = faiss.IndexFlatIP(self._dim)  # inner product = cosine if normalised
            self._use_faiss = True
            logger.info("FAISS index initialised (dim=%d).", self._dim)
        except ImportError:
            logger.warning("FAISS not available; using linear scan for retrieval.")

    def _add_to_faiss(self, vec: np.ndarray, record_id: str) -> None:
        if self._use_faiss and self._faiss_index is not None:
            v = vec.reshape(1, -1).astype(np.float32)
            self._faiss_index.add(v)
            self._faiss_id_map.append(record_id)

    # ── Core API ──────────────────────────────────────────────────────────────

    def store(
        self,
        text: str,
        outcome: Dict[str, Any],
        link_node: str = "",
        failure_flags: Optional[List[str]] = None,
        domain: str = "",
        task_id: str = "",
        hypothesis_id: str = "",
        force: bool = False,
    ) -> MemoryRecord:
        """Embed and store a reusable research lesson.

        ECRM is intentionally selective.  Routine neutral runs add noise and
        increase harmful transfer, so they are not retained unless ``force``
        is requested.  Failures and meaningful improvements are retained.
        """
        score = float(outcome.get("score", 0.0))
        success = bool(outcome.get("success", False))
        baseline = float(outcome.get("baseline", 0.0))
        # A failed attempt is itself useful negative evidence, even when the
        # evaluator has not supplied a more specific diagnosis yet.
        has_failure = (not success) or bool(failure_flags) or bool(outcome.get("error"))
        if not force and not has_failure and not (success and score > baseline):
            return MemoryRecord(text=text, outcome=outcome, link_node=link_node)
        embedding = embed_text(text, model_name=self._embedding_model, dim=self._dim)
        record = MemoryRecord(
            text=text,
            embedding=embedding,
            outcome=outcome,
            link_node=link_node,
            failure_flags=failure_flags or [],
            reliability=float(score),
            domain=domain,
            task_id=task_id,
            hypothesis_id=hypothesis_id,
        )
        self._records.append(record)
        self._add_to_faiss(embedding, record.record_id)

        # Periodic consolidation
        if len(self._records) % 100 == 0:
            self.consolidate()
        return record

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_failures: bool = False,
    ) -> List[Tuple[MemoryRecord, float]]:
        """
        Find top-K records most similar to query.
        Returns list of (record, similarity_score).
        """
        if not self._records:
            return []

        q_vec = embed_text(query, model_name=self._embedding_model, dim=self._dim)

        if self._use_faiss and self._faiss_index is not None and len(self._faiss_id_map) > 0:
            return self._faiss_retrieve(q_vec, top_k, filter_failures)
        return self._linear_retrieve(q_vec, top_k, filter_failures)

    def _faiss_retrieve(
        self, q_vec: np.ndarray, top_k: int, filter_failures: bool
    ) -> List[Tuple[MemoryRecord, float]]:
        import faiss  # type: ignore

        v = q_vec.reshape(1, -1).astype(np.float32)
        k = min(top_k * 3, len(self._faiss_id_map))
        scores, indices = self._faiss_index.search(v, k)
        record_map = {r.record_id: r for r in self._records}
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._faiss_id_map):
                continue
            rid = self._faiss_id_map[idx]
            rec = record_map.get(rid)
            if rec is None:
                continue
            if filter_failures and rec.failure_flags:
                continue
            results.append((rec, float(score)))
            if len(results) >= top_k:
                break
        return results

    def _linear_retrieve(
        self, q_vec: np.ndarray, top_k: int, filter_failures: bool
    ) -> List[Tuple[MemoryRecord, float]]:
        scored = []
        for rec in self._records:
            if filter_failures and rec.failure_flags:
                continue
            if rec.embedding is None:
                continue
            sim = cosine_similarity(q_vec, rec.embedding)
            scored.append((rec, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def has_similar_failure(self, query: str, threshold: float = 0.8) -> bool:
        """Return True if a very similar past experiment failed."""
        results = self.retrieve(query, top_k=3)
        for rec, sim in results:
            if sim >= threshold and rec.failure_flags:
                return True
        return False

    # ── NTR interface ─────────────────────────────────────────────────────────

    def record_ntr(
        self,
        strategy_id: str,
        used_memory: bool,
        baseline: float,
        achieved: float,
    ) -> None:
        self._ntr_detector.record(strategy_id, used_memory, baseline, achieved)

    def get_ntr(self, strategy_id: Optional[str] = None) -> float:
        if strategy_id:
            return self._ntr_detector.ntr_for_strategy(strategy_id)
        return self._ntr_detector.global_ntr()

    # ── Consolidation ─────────────────────────────────────────────────────────

    def consolidate(self) -> int:
        """Prune stale / low-retention records. Returns number removed."""
        before = len(self._records)
        self._records = prune_records(
            self._records, self._retain_threshold, self._retain_lambda, self._max_records
        )
        removed = before - len(self._records)
        if removed and self._use_faiss:
            # Rebuild FAISS index after pruning
            self._rebuild_faiss()
        return removed

    def _rebuild_faiss(self) -> None:
        try:
            import faiss  # type: ignore
            self._faiss_index = faiss.IndexFlatIP(self._dim)
            self._faiss_id_map = []
            for rec in self._records:
                if rec.embedding is not None:
                    self._add_to_faiss(rec.embedding, rec.record_id)
        except ImportError:
            pass

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = [r.to_dict() for r in self._records]
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> None:
        raw = json.loads(Path(path).read_text())
        self._records = [MemoryRecord.from_dict(d) for d in raw]
        if self._use_faiss:
            self._rebuild_faiss()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        success = sum(1 for r in self._records if r.outcome.get("success"))
        return {
            "total_records": len(self._records),
            "successful": success,
            "failed": len(self._records) - success,
            "global_ntr": self.get_ntr(),
        }

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"ECRMMemoryStore(records={len(self._records)}, faiss={self._use_faiss})"

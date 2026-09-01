"""In-process vector index backend — brute-force cosine similarity.

Conforms to the VectorIndexBackend protocol. All data lives in Python dicts
in process memory. Similarity search is exact (not approximate): O(n) scan
over all stored vectors. This is appropriate at the scale ResearchForge
operates at during a single research run (hundreds to low thousands of
memory records, not millions).

This formalises and replaces the ad-hoc cosine similarity loop that was
previously embedded inline in memory/ecrm.py. The scientific scoring
algorithms in ECRM (RES, NTR, consolidate, forgetting) are unchanged.

Metric
------
Only 'cosine' is implemented. The dimension is fixed at construction time
from the first vector added, or explicitly supplied. Any subsequent add()
call with a different vector length raises DimensionMismatchError.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..capabilities import BackendCapabilities, HealthStatus
from ..errors import DimensionMismatchError
from ..protocols import VectorIndexBackend


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InProcessVectorIndex(VectorIndexBackend):
    """Exact brute-force cosine similarity over in-process Python dicts.

    Zero dependencies. Deterministic. Suitable for unit tests, research
    loops up to ~10,000 vectors. Not suitable for production at scale.
    """

    BACKEND_NAME = "inprocess"
    BACKEND_VERSION = "1.0"

    def __init__(self, dimension: Optional[int] = None) -> None:
        self._dim: Optional[int] = dimension
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Optional[Dict[str, Any]]] = {}
        self._closed = False

    # ── Capabilities ──────────────────────────────────────────────────────
    @classmethod
    def capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            persistent=False,
            transactional=False,
            supports_vector_metadata=True,
            production_ready=False)

    @classmethod
    def backend_name(cls) -> str:
        return cls.BACKEND_NAME

    @classmethod
    def backend_version(cls) -> str:
        return cls.BACKEND_VERSION

    # ── Dimension / metric ────────────────────────────────────────────────
    @property
    def dimension(self) -> int:
        return self._dim or 0

    @property
    def metric(self) -> str:
        return "cosine"

    # ── Write ─────────────────────────────────────────────────────────────
    def add(self, vector_id: str, vector: Sequence[float],
            metadata: Optional[Mapping[str, Any]] = None) -> None:
        vec = list(vector)
        if self._dim is None:
            self._dim = len(vec)
        elif len(vec) != self._dim:
            raise DimensionMismatchError(self._dim, len(vec),
                                         backend_name=self.BACKEND_NAME)
        self._vectors[vector_id] = vec
        self._metadata[vector_id] = dict(metadata) if metadata else None

    def remove(self, vector_id: str) -> None:
        self._vectors.pop(vector_id, None)
        self._metadata.pop(vector_id, None)

    # ── Read ──────────────────────────────────────────────────────────────
    def search(self, query_vector: Sequence[float],
               k: Optional[int] = None) -> List[Tuple[str, float]]:
        qvec = list(query_vector)
        if self._dim is not None and len(qvec) != self._dim:
            raise DimensionMismatchError(self._dim, len(qvec),
                                         backend_name=self.BACKEND_NAME)
        scored = [(vid, _cosine(qvec, vec))
                  for vid, vec in self._vectors.items()]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored if k is None else scored[:k]

    def get_metadata(self, vector_id: str) -> Optional[Dict[str, Any]]:
        return dict(self._metadata[vector_id]) if vector_id in self._metadata else None

    def size(self) -> int:
        return len(self._vectors)

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        self._closed = True

    def health_check(self) -> HealthStatus:
        t0 = time.monotonic()
        _ = len(self._vectors)
        latency = (time.monotonic() - t0) * 1000
        return HealthStatus(
            healthy=not self._closed, persistent=False, transactional=False,
            latency_ms=latency,
            message="" if not self._closed else "Backend is closed")

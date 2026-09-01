"""Backend capability and health metadata.

BackendCapabilities declares what a backend supports. BackendInfo is the
per-instance registration record in AdapterRegistry. HealthStatus is
returned by backend.health_check() and used for runtime selection (e.g.,
automatic fallback when Neo4j is unavailable).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set


# ── Capability flags ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BackendCapabilities:
    """Immutable set of capability flags for a backend class.

    Used by AdapterRegistry and by the system's runtime backend-selection
    logic (e.g., "does this backend support transactions?").

    All flags are False by default so that new backends opt-in explicitly.
    """

    # Storage durability
    persistent: bool = False          # data survives process restart
    transactional: bool = False       # begin/commit/rollback supported

    # Query features
    supports_traversal: bool = False  # efficient graph traversal (e.g. Cypher)
    supports_batch_insert: bool = False
    supports_native_similarity: bool = False  # ANN index vs brute-force

    # Vector-specific
    supports_vector_metadata: bool = False    # metadata attached to each vector

    # Operational
    production_ready: bool = False    # suitable for production deployment

    def __str__(self) -> str:
        flags = [k for k, v in self.__dict__.items() if v]
        return f"BackendCapabilities({', '.join(flags) or 'none'})"


# ── Registration metadata ───────────────────────────────────────────────────


@dataclass
class BackendInfo:
    """Per-instance registration record in AdapterRegistry.

    Every registered backend has exactly one BackendInfo. The registry
    exposes this through AdapterRegistry.info(name).
    """
    name: str
    version: str
    capabilities: BackendCapabilities
    # Additional per-instance fields
    vector_dimension: int = 0          # 0 means N/A or unknown
    vector_metric: str = "cosine"      # "cosine" | "dot" | "l2"
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # noqa: D105
        return (f"BackendInfo(name={self.name!r}, version={self.version!r}, "
                f"capabilities={self.capabilities})")


# ── Health status ───────────────────────────────────────────────────────────


@dataclass
class HealthStatus:
    """Returned by backend.health_check().

    healthy=True means the backend is operable right now. latency_ms is
    the round-trip time in milliseconds for a lightweight probe operation
    (e.g., SELECT 1 for SQLite, PING for Neo4j). message carries a short
    human-readable explanation when healthy=False.
    """
    healthy: bool
    persistent: bool
    transactional: bool
    latency_ms: float = 0.0
    message: str = ""

    def __repr__(self) -> str:  # noqa: D105
        status = "HEALTHY" if self.healthy else "UNHEALTHY"
        return f"HealthStatus({status}, latency={self.latency_ms:.1f}ms)"

"""pgvector backend stub — documented production adapter.

STATUS: PLANNED — NOT IMPLEMENTED.

See module-level docstring for the complete migration path. Every method
raises NotImplementedError. Does not simulate pgvector behaviour.

Migration path from InProcessVectorIndex
-----------------------------------------
1.  Install: pip install psycopg2-binary pgvector
2.  Create extension in Postgres: CREATE EXTENSION vector;
3.  Create table:
      CREATE TABLE memory_vectors (
          id      TEXT PRIMARY KEY,
          vec     vector(<dim>),
          meta    JSONB
      );
      CREATE INDEX ON memory_vectors USING ivfflat (vec vector_cosine_ops);
4.  Implement add() as:
      INSERT INTO memory_vectors(id, vec, meta)
      VALUES (%s, %s::vector, %s)
      ON CONFLICT (id) DO UPDATE SET vec=EXCLUDED.vec, meta=EXCLUDED.meta;
5.  Implement search() as:
      SELECT id, 1 - (vec <=> %s::vector) AS score
      FROM memory_vectors
      ORDER BY vec <=> %s::vector
      LIMIT %s;
6.  Run the contract validation suite:
      from researchforge.adapters.validation import validate_vector_backend
      validate_vector_backend(PgvectorBackend(...))
    All checks must pass before registration.

Implementation deferred until after Adaptive Trajectory-ECRM (RF-1.0.0-alpha.4)
to avoid building against a vector schema that may change.
"""
from __future__ import annotations

from ..capabilities import BackendCapabilities, HealthStatus
from ..protocols import VectorIndexBackend


class PgvectorBackend(VectorIndexBackend):
    """Documented production adapter stub for Postgres+pgvector.

    STATUS: PLANNED — every method raises NotImplementedError.
    DO NOT mock results. DO NOT simulate behaviour.
    See module docstring for the full migration path.
    """

    BACKEND_NAME = "pgvector"
    BACKEND_VERSION = "0.0.0-planned"
    STATUS = "PLANNED"

    _NOT_IMPLEMENTED_MSG = (
        "PgvectorBackend is a documented production adapter stub. "
        "Requires a running Postgres instance with the pgvector extension. "
        "Implementation deferred until RF-1.0.0-alpha.4. "
        "See researchforge/adapters/backends/pgvector.py for the migration path."
    )

    def __init__(self, dsn: str = "", dimension: int = 256) -> None:  # noqa: D107
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    @classmethod
    def capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            persistent=True, transactional=True,
            supports_vector_metadata=True, supports_native_similarity=True,
            production_ready=True)

    @classmethod
    def backend_name(cls) -> str:
        return cls.BACKEND_NAME

    @classmethod
    def backend_version(cls) -> str:
        return cls.BACKEND_VERSION

    @property
    def dimension(self) -> int:
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    @property
    def metric(self) -> str:
        return "cosine"

    def health_check(self) -> HealthStatus:
        return HealthStatus(
            healthy=False, persistent=True, transactional=True,
            latency_ms=0.0, message=self._NOT_IMPLEMENTED_MSG)

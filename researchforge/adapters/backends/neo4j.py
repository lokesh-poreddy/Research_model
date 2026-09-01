"""Neo4j graph backend — documented production adapter stub.

STATUS: PLANNED — NOT IMPLEMENTED.

This module documents the complete interface contract and migration path
for a Neo4j-backed GraphBackend. It is a transparent stub: importing it
does not crash; instantiating it raises NotImplementedError with a clear
explanation. It does NOT simulate Neo4j behaviour or return fake results.

Why this stub exists now
------------------------
Having the stub in the codebase at this stage serves three purposes:

1.  It makes the migration path explicit in code, not just in documentation.
2.  It forces the AdapterRegistry to handle "backend declared but not
    available" gracefully, which is a real operational requirement.
3.  It gives a concrete checklist for when infrastructure is available.

Migration path from SQLiteGraphBackend
---------------------------------------
1.  Install: pip install neo4j
2.  Start Neo4j (AuraDB, Docker, or standalone).
3.  Set environment variables:
      NEO4J_URI=bolt://localhost:7687
      NEO4J_USER=neo4j
      NEO4J_PASSWORD=<password>
4.  Replace Neo4jGraphBackend's __init__ with:
      from neo4j import GraphDatabase
      self._driver = GraphDatabase.driver(uri, auth=(user, password))
5.  Implement add_node() as:
      MERGE (n:<node_type> {id: $id})
      SET n.content = $content, n.timestamp = $timestamp,
          n += $attributes
6.  Implement add_edge() as:
      MATCH (a {id: $from_id}), (b {id: $to_id})
      MERGE (a)-[r:<relation>]->(b)
      SET r += $properties
7.  Implement get_node() as:
      MATCH (n {id: $id}) RETURN n
8.  Implement transaction() using driver.session() as the context.
9.  Run the contract validation suite:
      from researchforge.adapters.validation import validate_graph_backend
      validate_graph_backend(Neo4jGraphBackend(...))
    All checks must pass before the backend is registered.

Implementation is deferred until after VRDEG is finalized (RF-1.0.0-alpha.3)
to avoid building a migration against a schema that may change.
"""
from __future__ import annotations

from ..capabilities import BackendCapabilities, HealthStatus
from ..errors import BackendCapabilityError
from ..protocols import GraphBackend


class Neo4jGraphBackend(GraphBackend):
    """Documented production adapter stub for Neo4j.

    STATUS: PLANNED — every method raises NotImplementedError.
    DO NOT use in production. DO NOT mock results.
    See module docstring for the full migration path.
    """

    BACKEND_NAME = "neo4j"
    BACKEND_VERSION = "0.0.0-planned"
    STATUS = "PLANNED"

    _NOT_IMPLEMENTED_MSG = (
        "Neo4jGraphBackend is a documented production adapter stub. "
        "Implementation requires a running Neo4j instance and is deferred "
        "until after VRDEG schema is finalized (RF-1.0.0-alpha.3). "
        "See researchforge/adapters/backends/neo4j.py for the migration path."
    )

    def __init__(self, uri: str = "", user: str = "", password: str = "") -> None:  # noqa: D107
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    @classmethod
    def capabilities(cls) -> BackendCapabilities:
        # Declared capabilities reflect what Neo4j WOULD support when implemented.
        return BackendCapabilities(
            persistent=True,
            transactional=True,
            supports_traversal=True,
            supports_batch_insert=True,
            production_ready=True)

    @classmethod
    def backend_name(cls) -> str:
        return cls.BACKEND_NAME

    @classmethod
    def backend_version(cls) -> str:
        return cls.BACKEND_VERSION

    def health_check(self) -> HealthStatus:
        return HealthStatus(
            healthy=False, persistent=True, transactional=True,
            latency_ms=0.0, message=self._NOT_IMPLEMENTED_MSG)

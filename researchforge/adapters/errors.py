"""Backend error hierarchy for the ResearchForge adapter layer.

All backend-originated exceptions inherit from BackendError. Callers that
do not care about the specific cause can catch BackendError; callers that
need to distinguish connection failures from missing data can catch the
appropriate subclass.

IMPORTANT: These errors are storage-layer errors only. Research-domain
constraint violations (e.g., an edge that violates RDG typed-relation rules)
are raised by the RDG domain layer (rdg/graph.py), NOT by the backend.
"""
from __future__ import annotations

from typing import Optional


class BackendError(Exception):
    """Base class for all storage-backend errors."""

    def __init__(self, message: str, backend_name: str = "unknown",
                 cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.backend_name = backend_name
        self.cause = cause

    def __str__(self) -> str:  # noqa: D105
        base = super().__str__()
        if self.cause:
            return f"{base} (caused by {type(self.cause).__name__}: {self.cause})"
        return base


class BackendConnectionError(BackendError):
    """Raised when the backend cannot establish or maintain a connection.

    Examples: Neo4j server unreachable, SQLite file locked by another process,
    pgvector extension not installed.
    """


class BackendNotFoundError(BackendError):
    """Raised when a requested node or vector id does not exist in the backend.

    The caller should treat this as a recoverable condition: the id simply
    has not been written yet (or was removed).
    """

    def __init__(self, entity_id: str, backend_name: str = "unknown") -> None:
        super().__init__(f"Entity '{entity_id}' not found", backend_name=backend_name)
        self.entity_id = entity_id


class BackendSerializationError(BackendError):
    """Raised when a value cannot be serialized to or deserialized from the
    backend's storage format (e.g., attributes dict contains a non-JSON-
    serializable type, or stored JSON is corrupt).
    """


class BackendTransactionError(BackendError):
    """Raised for transaction lifecycle violations: commit without begin,
    nested transactions on a backend that does not support them, rollback
    after a committed transaction, or close while a transaction is active.
    """


class BackendCapabilityError(BackendError):
    """Raised when the caller requests a capability that this backend does
    not support (e.g., calling begin_transaction() on a backend whose
    BackendCapabilities.transactional is False, or requesting a dimension
    that the vector index was not constructed for).
    """


class DimensionMismatchError(BackendCapabilityError):
    """Raised when a vector's length does not match the backend's declared
    dimension. Subclass of BackendCapabilityError because it represents an
    API misuse that the backend cannot silently recover from.
    """

    def __init__(self, expected: int, got: int,
                 backend_name: str = "unknown") -> None:
        super().__init__(
            f"Vector dimension mismatch: expected {expected}, got {got}",
            backend_name=backend_name)
        self.expected = expected
        self.got = got

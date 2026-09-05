from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
from .base import DomainObject


@dataclass(frozen=True)
class ExecutionPolicy(DomainObject):
    """Execution / Resource policy controlling how an ExperimentSpec is executed.

    Note: This policy is declarative. The runtime enforces a subset supported
    by the current SafeRunner (per-experiment timeout and a simple resource
    budget). This contract intentionally does NOT claim filesystem, network,
    cgroup, or GPU isolation beyond what the SafeRunner provides.
    """
    use_sandbox: bool = True
    per_experiment_timeout_s: float | None = None
    max_experiments: int | None = None
    allowed_working_dirs: List[str] | None = None
    network_allowlist: List[str] | None = None
    additional_constraints: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ExecutionPolicy":
        return cls(**obj)

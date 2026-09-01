"""Experiment Sandbox (design doc Sec. 7): resource quotas, per-experiment
timeouts, and a kill switch around arbitrary experiment-executing callables.

The design doc specifies Docker containerization + a job scheduler + safety
controls (Resource Quotas, Graceful Fallback, Kill Switch). A single Python
process in this sandbox can't give you filesystem/network isolation the way a
container does, but it CAN give you a *real* enforced timeout and a *real*
kill: this module runs each experiment in its own OS process (via
`multiprocessing`, fork start method) and calls `Process.terminate()`/`kill()`
if it overruns its budget. That is a meaningfully different guarantee from
wrapping a thread with `future.result(timeout=...)`: a thread that ignores a
timeout keeps burning CPU in the background because Python cannot forcibly
stop a thread; a process that overruns actually gets killed by the OS.

What this does NOT give you (be honest about the gap, don't oversell it):
CPU/memory *limits* on the child while it runs (would need `resource.setrlimit`
in the child, or real cgroups/Docker), filesystem isolation, or network
isolation. Sec. 7's Docker+MLflow+scheduler design is still the right answer
for a real multi-tenant deployment; this is the offline, single-machine
approximation of just the "don't let one bad experiment run forever or take
the process down with it" guarantee.
"""
from __future__ import annotations
import multiprocessing as mp
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional


class SafetyStatus(str, Enum):
    OK = "ok"
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    BUDGET_EXHAUSTED = "budget_exhausted"
    KILLED = "killed"


@dataclass
class ResourceBudget:
    max_experiments: int = 10_000
    max_wall_time_s: float = 3600.0
    per_experiment_timeout_s: float = 20.0


@dataclass
class SafetyOutcome:
    status: SafetyStatus
    value: Any = None
    error: Optional[str] = None
    duration_s: float = 0.0


def _worker_entrypoint(fn: Callable, args: tuple, kwargs: dict, conn) -> None:
    try:
        result = fn(*args, **kwargs)
        conn.send(("ok", result))
    except Exception as exc:  # pragma: no cover - defensive, exercised by tests below
        conn.send(("exception", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


class SafeRunner:
    """Runs a callable in a forked subprocess with a hard wall-clock timeout,
    and tracks a cumulative resource budget (experiment count + total time)
    across the whole run, matching Sec. 7's "Resource Quotas". `kill()` is the
    manual kill switch; `should_stop()` reports whether the automatic budget
    has already been exhausted.
    """

    def __init__(self, budget: Optional[ResourceBudget] = None):
        self.budget = budget or ResourceBudget()
        self.experiments_run = 0
        self.total_time_s = 0.0
        self._killed = False

    def kill(self) -> None:
        self._killed = True

    def should_stop(self) -> bool:
        return (self._killed
                or self.experiments_run >= self.budget.max_experiments
                or self.total_time_s >= self.budget.max_wall_time_s)

    def run(self, fn: Callable[..., Any], *args, **kwargs) -> SafetyOutcome:
        if self._killed:
            return SafetyOutcome(status=SafetyStatus.KILLED)
        if self.experiments_run >= self.budget.max_experiments or \
                self.total_time_s >= self.budget.max_wall_time_s:
            return SafetyOutcome(status=SafetyStatus.BUDGET_EXHAUSTED)

        ctx = mp.get_context("fork")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_worker_entrypoint, args=(fn, args, kwargs, child_conn))

        t0 = time.time()
        proc.start()
        child_conn.close()  # parent doesn't write; close its copy of the write end
        proc.join(self.budget.per_experiment_timeout_s)
        dt = time.time() - t0
        self.experiments_run += 1
        self.total_time_s += dt

        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)
            if proc.is_alive():
                proc.kill()
                proc.join()
            parent_conn.close()
            return SafetyOutcome(status=SafetyStatus.TIMEOUT, duration_s=dt,
                                  error=f"exceeded {self.budget.per_experiment_timeout_s}s; "
                                        f"process terminated")

        if parent_conn.poll():
            kind, payload = parent_conn.recv()
            parent_conn.close()
            if kind == "ok":
                return SafetyOutcome(status=SafetyStatus.OK, value=payload, duration_s=dt)
            return SafetyOutcome(status=SafetyStatus.EXCEPTION, error=payload, duration_s=dt)

        parent_conn.close()
        return SafetyOutcome(status=SafetyStatus.EXCEPTION, duration_s=dt,
                              error=f"worker exited (code {proc.exitcode}) without a result")

    def status_report(self) -> dict:
        return {
            "experiments_run": self.experiments_run,
            "experiments_remaining": max(0, self.budget.max_experiments - self.experiments_run),
            "total_time_s": round(self.total_time_s, 3),
            "time_remaining_s": round(max(0.0, self.budget.max_wall_time_s - self.total_time_s), 3),
            "killed": self._killed,
            "should_stop": self.should_stop(),
        }

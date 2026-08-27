"""
ExperimentAgent: translates a Hypothesis + ModelGenome into
runnable experiment code, then executes and captures results — v2.

v2 changes:
- BudgetGate: checks BudgetAllocator before any sandbox execution.
- Uses sys.executable (not bare "python") so the active venv is used.
- Timeout comes from settings.sandbox_timeout_seconds (not hardcoded 300).
- Execution ledger tracks cumulative sandbox seconds per run.
"""
from __future__ import annotations

import logging
import random
import sys
import time
from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from evolution.genome import ModelGenome
from rdg.nodes import RDGNode

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert ML engineer. Given a hypothesis and model genome,
write a complete PyTorch training script that:
1. Defines the model from the genome architecture.
2. Trains it for the specified epochs.
3. Evaluates on validation set.
4. Prints: RESULT: <val_accuracy>
Return ONLY valid Python code in a ```python ... ``` block."""


class ExperimentAgent(BaseAgent):
    """
    Generates and executes experiment code for a given Hypothesis node.
    In production, runs in a sandboxed subprocess; in dev mode, uses mock execution.

    The ``_sandbox_run`` path is guarded by a BudgetAllocator reference.
    The controller injects the allocator after construction so the
    experiment agent does not import it at module load time.
    """

    def __init__(self, sandbox_mode: bool = True) -> None:
        super().__init__(name="ExperimentAgent")
        self.sandbox_mode = sandbox_mode
        self._budget_allocator: Optional[Any] = None   # set by controller
        # Cumulative sandbox execution seconds this session
        self._execution_ledger: Dict[str, float] = {}
        self._total_sandbox_seconds: float = 0.0

    # ── Budget gate ───────────────────────────────────────────────────────────

    def set_budget_allocator(self, allocator: Any) -> None:
        """Inject the shared BudgetAllocator from the controller."""
        self._budget_allocator = allocator

    def _budget_ok(self, strategy_id: str = "unknown") -> bool:
        """Return True if there is remaining compute budget."""
        if self._budget_allocator is None:
            return True
        if self._budget_allocator.is_over_budget():
            logger.warning(
                "[ExperimentAgent] Sandbox blocked for strategy '%s': over budget.", strategy_id
            )
            return False
        return True

    # ── Code generation ───────────────────────────────────────────────────────

    def generate_code(
        self,
        hypothesis: RDGNode,
        genome: ModelGenome,
        task_description: str = "image classification",
    ) -> str:
        """Use LLM to synthesize experiment code."""
        user_prompt = (
            f"Hypothesis: {hypothesis.content}\n\n"
            f"Model Genome:\n{genome.to_json()}\n\n"
            f"Task: {task_description}\n\n"
            "Write the training script."
        )
        code = self.llm_call(SYSTEM_PROMPT, user_prompt)
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        return code

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(
        self,
        hypothesis: RDGNode,
        genome: ModelGenome,
        task_description: str = "image classification",
        use_mock: bool = True,
        task: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute the experiment and return a result dict with:
          - success: bool
          - score: float
          - train_loss: float
          - val_loss: float
          - error: str (if failed)
          - runtime_seconds: float
        """
        start = time.time()
        code = self.generate_code(hypothesis, genome, task_description)
        strategy_id = genome.strategy_description or genome.fingerprint()

        if use_mock:
            result = self._mock_run(genome)
        elif task is not None:
            # Trusted task adapters own their data and evaluator.
            result = self._task_run(task, genome)
        else:
            # Real sandbox path — guarded by budget allocator.
            if not self._budget_ok(strategy_id):
                result = {
                    "success": False,
                    "score": 0.0,
                    "train_loss": 0.0,
                    "val_loss": 0.0,
                    "error": "Blocked: compute budget exhausted.",
                }
            else:
                result = self._sandbox_run(code, genome)

        elapsed = time.time() - start
        result["runtime_seconds"] = elapsed
        result["code_snippet"] = code[:500]

        # Record sandbox seconds in ledger + allocator
        if not use_mock and task is None and self._budget_allocator is not None:
            self._total_sandbox_seconds += elapsed
            self._execution_ledger[strategy_id] = (
                self._execution_ledger.get(strategy_id, 0.0) + elapsed
            )
            self._budget_allocator.record(
                strategy_id,
                family=genome.strategy_description or "unknown",
                seconds=elapsed,
            )

        logger.info(
            "[ExperimentAgent] Experiment done: success=%s, score=%.4f",
            result["success"],
            result.get("score", 0.0),
        )
        return result

    # ── Execution paths ───────────────────────────────────────────────────────

    @staticmethod
    def _task_run(task: Any, genome: ModelGenome) -> Dict[str, Any]:
        try:
            result = dict(task.evaluate(genome.to_dict()))
            score = float(result.get("score", 0.0))
            result.setdefault("success", score >= float(getattr(task, "target_score", 0.0)))
            result.setdefault("train_loss", 1.0 - float(result.get("train_score", score)))
            result.setdefault("val_loss", 1.0 - score)
            result.setdefault("error", "")
            return result
        except Exception as exc:
            return {"success": False, "score": 0.0, "train_loss": 0.0,
                    "val_loss": 0.0, "error": str(exc)}

    def _mock_run(self, genome: ModelGenome) -> Dict[str, Any]:
        """Simulated result for offline testing."""
        base = 0.5 + genome.generation * 0.02 + random.gauss(0, 0.05)
        score = max(0.0, min(1.0, base))
        success = score > 0.55
        return {
            "success": success,
            "score": score,
            "train_loss": random.uniform(0.1, 0.5),
            "val_loss": random.uniform(0.1, 0.6),
            "error": "" if success else "Score below threshold",
        }

    def _sandbox_run(self, code: str, genome: ModelGenome) -> Dict[str, Any]:
        """Run code in isolated subprocess (production).

        Uses ``sys.executable`` to respect the active virtual environment
        and ``settings.sandbox_timeout_seconds`` instead of a hardcoded value.
        """
        import subprocess
        import tempfile
        import os

        from config.settings import settings

        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write(code)
            tmpfile = f.name

        try:
            proc = subprocess.run(
                [sys.executable, tmpfile],    # v2: sys.executable, not bare "python"
                capture_output=True,
                text=True,
                timeout=settings.sandbox_timeout_seconds,  # v2: from settings
            )
            output = proc.stdout + proc.stderr
            score = self._parse_score(output)
            success = proc.returncode == 0 and score > 0
            return {
                "success": success,
                "score": score,
                "train_loss": 0.0,
                "val_loss": 0.0,
                "error": proc.stderr[:500] if proc.returncode != 0 else "",
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "score": 0.0, "error": "Timeout",
                    "train_loss": 0.0, "val_loss": 0.0}
        except Exception as exc:
            return {"success": False, "score": 0.0, "error": str(exc),
                    "train_loss": 0.0, "val_loss": 0.0}
        finally:
            os.unlink(tmpfile)

    @staticmethod
    def _parse_score(output: str) -> float:
        for line in output.split("\n"):
            if "RESULT:" in line:
                try:
                    return float(line.split("RESULT:")[1].strip())
                except ValueError:
                    pass
        return 0.0

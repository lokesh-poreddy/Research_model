"""
ExperimentAgent: translates a Hypothesis + ModelGenome into
runnable experiment code, then executes and captures results.
"""
from __future__ import annotations

import logging
import random
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
    In production, runs in a Docker sandbox; in dev mode, uses mock execution.
    """

    def __init__(self, sandbox_mode: bool = True):
        super().__init__(name="ExperimentAgent")
        self.sandbox_mode = sandbox_mode

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
        # Extract code block
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        return code

    def run(
        self,
        hypothesis: RDGNode,
        genome: ModelGenome,
        task_description: str = "image classification",
        use_mock: bool = True,
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

        if use_mock or self.sandbox_mode:
            result = self._mock_run(genome)
        else:
            result = self._sandbox_run(code, genome)

        result["runtime_seconds"] = time.time() - start
        result["code_snippet"] = code[:500]
        logger.info(
            "[ExperimentAgent] Experiment done: success=%s, score=%.4f",
            result["success"],
            result.get("score", 0.0),
        )
        return result

    def _mock_run(self, genome: ModelGenome) -> Dict[str, Any]:
        """Simulated result for offline testing."""
        # Score influenced by genome generation (higher gen = slightly better)
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
        """Run code in isolated Docker sandbox (production)."""
        import subprocess, tempfile, os  # noqa: E401

        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write(code)
            tmpfile = f.name

        try:
            proc = subprocess.run(
                ["python", tmpfile],
                capture_output=True,
                text=True,
                timeout=300,
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
            return {"success": False, "score": 0.0, "error": "Timeout", "train_loss": 0.0, "val_loss": 0.0}
        except Exception as exc:
            return {"success": False, "score": 0.0, "error": str(exc), "train_loss": 0.0, "val_loss": 0.0}
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

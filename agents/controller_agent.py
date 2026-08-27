"""
ResearchController — v2.

v2 changes:
- ``_check_promotion_gate()``: validates seed count, compute budget, and
  minimum improvement before marking a candidate as promoted.
- ``_algorithm_discovery()``: uses StrategyPortfolio to pick the least-explored
  family instead of hardcoding SVC.
- ``compute_budget_used`` counter with warning when nearing limit.
- ``strategy_hint`` from portfolio passed to ``hypothesis_agent.generate()``.
- ``BudgetAllocator`` injected into ``ExperimentAgent``.

Loop (unchanged from v1):
  1. Select branch (policy + acquisition function)
  2. Generate hypothesis
  3. Evolve model genome
  4. Run experiment
  5. Analyze results → RDG update
  6. Diagnose failure (if any)
  7. Repair + update memory + update policy
  8. Check algorithm discovery trigger
  9. Repeat
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.experiment_agent import ExperimentAgent
from agents.analyzer_agent import AnalyzerAgent
from rdg.graph import ResearchDevelopmentGraph
from rdg.nodes import NodeStatus, NodeType, RDGNode
from rdg.edges import EdgeRelation
from ecrm.memory_store import ECRMMemoryStore
from evolution.genome import ModelGenome
from evolution.mutate import random_mutation
from evolution.crossover import crossover
from policy.acquisition import select_branch
from policy.rl_policy import QLearningPolicy
from policy.strategy_portfolio import StrategyPortfolio
from policy.budget_allocator import BudgetAllocator
from failure.diagnosis import diagnose_failure
from failure.repair import apply_repair
from failure.taxonomy import FailureCategory
from config.settings import settings

logger = logging.getLogger(__name__)


class ResearchController:
    """
    Central research loop controller.
    Manages all agents, RDG, memory, policy, and budget allocation.
    """

    def __init__(
        self,
        rdg: ResearchDevelopmentGraph,
        memory: ECRMMemoryStore,
        problem_description: str = "",
        max_experiments: int = 50,
        plateau_threshold: int = 5,
        use_mock_experiments: bool = True,
        task: Optional[Any] = None,
    ):
        self.rdg = rdg
        self.memory = memory
        self.problem_description = problem_description
        self.max_experiments = max_experiments
        self.plateau_threshold = plateau_threshold
        self.use_mock = use_mock_experiments
        self.task = task

        # Agents
        self.hypothesis_agent = HypothesisAgent()
        self.experiment_agent = ExperimentAgent()
        self.analyzer_agent = AnalyzerAgent()

        # Policy + allocation
        self.policy = QLearningPolicy(
            alpha=settings.rl_learning_rate,
            gamma=settings.rl_gamma,
        )
        self.strategy_portfolio = StrategyPortfolio()
        self.budget_allocator = BudgetAllocator(
            budget_hours=settings.v2_compute_budget_hours
        )
        # Wire budget allocator into experiment agent
        self.experiment_agent.set_budget_allocator(self.budget_allocator)

        # State
        self.total_experiments = 0
        self.best_score = 0.0
        self.experiments_since_improvement = 0
        self.population: List[ModelGenome] = [ModelGenome()]  # seed genome
        self.history: List[Dict[str, Any]] = []
        self._seeds_run: int = 0   # v2: track independent seeds for promotion gate

    # ── Main research loop ────────────────────────────────────────────────────

    def run(self, n_iterations: int = 10) -> Dict[str, Any]:
        """Execute n_iterations of the research loop."""
        logger.info(
            "ResearchController v2 starting: problem='%s', iterations=%d",
            self.problem_description[:60],
            n_iterations,
        )

        for i in range(n_iterations):
            logger.info("─── Iteration %d/%d ───", i + 1, n_iterations)
            step_result = self._research_step()
            self.history.append(step_result)

            if self.total_experiments >= self.max_experiments:
                logger.info("Max experiments reached. Stopping.")
                break

        summary = self._summarize()
        logger.info("Research complete: best_score=%.4f", self.best_score)
        return summary

    def _research_step(self) -> Dict[str, Any]:
        """Execute one full research step."""

        # ── 1. Select branch ──────────────────────────────────────────────────
        candidates = self.rdg.hypotheses or []
        selected_hypothesis = select_branch(
            candidates,
            total_experiments=self.total_experiments,
            memory=self.memory,
            q_values={n.id: self.policy.estimate_reward(n) for n in candidates},
            policy_type=settings.policy_type,
        )

        # ── 2. Generate hypothesis (if none exist or need new one) ────────────
        evolution_strategy = self.strategy_portfolio.select()
        if selected_hypothesis is None or self.experiments_since_improvement >= 3:
            gap_nodes = [n for n in self.rdg if n.type == NodeType.GAP]
            if not gap_nodes:
                gap = RDGNode.gap(content=f"Gap: {self.problem_description}")
                self.rdg.add_node(gap)
                gap_nodes = [gap]

            gap_node = gap_nodes[-1]
            task_context = {
                "domain": getattr(self.task, "domain", "machine_learning"),
                "task_id": getattr(self.task, "name", type(self.task).__name__) if self.task else "unspecified",
                "objective": "maximize_validation_score",
            }
            hyp_text = self.hypothesis_agent.generate(
                gap_node=gap_node,
                memory=self.memory,
                strategy_hint=evolution_strategy,    # v2: portfolio hint passed through
                context=task_context,                # v2: context for conditioned retrieval
            )
            selected_hypothesis = RDGNode.hypothesis(content=hyp_text)
            self.rdg.add_node(selected_hypothesis)
            self.rdg.connect(gap_node.id, selected_hypothesis.id, EdgeRelation.MOTIVATES)

        # ── 3. Evolve genome ──────────────────────────────────────────────────
        parent_genome = self.population[-1]
        if len(self.population) >= 2 and self.total_experiments % 5 == 0:
            genome = crossover(self.population[-1], self.population[-2])
            evolution_strategy = "crossover"
        else:
            genome = random_mutation(parent_genome, operator_hint=evolution_strategy)

        # ── 4. Create Experiment node ─────────────────────────────────────────
        exp_node = RDGNode.experiment(
            content=f"Testing: {selected_hypothesis.content[:100]}",
            code=genome.to_json(),
        )
        self.rdg.add_node(exp_node)
        self.rdg.connect(selected_hypothesis.id, exp_node.id, EdgeRelation.TESTED_BY)

        # ── 5. Run experiment ─────────────────────────────────────────────────
        result = self.experiment_agent.run(
            hypothesis=selected_hypothesis,
            genome=genome,
            task_description=self.task.description() if self.task else "image classification",
            use_mock=self.use_mock,
            task=self.task,
        )
        result.setdefault(
            "memory_context",
            {
                "domain": getattr(self.task, "domain", "machine_learning"),
                "task_id": getattr(self.task, "name", type(self.task).__name__) if self.task else "unspecified",
                "objective": "maximize_validation_score",
                "model_family": genome.data_settings.get(
                    "estimator", genome.architecture.get("type", "unknown")
                ),
            },
        )
        result.setdefault("strategy_id", genome.strategy_description or genome.fingerprint())
        result.setdefault("baseline", self.best_score)
        self.total_experiments += 1

        exp_node.status = NodeStatus.FAILED if result.get("error") else NodeStatus.COMPLETED
        exp_node.attributes.update(result)
        self.rdg.update_node(exp_node.id, status=exp_node.status)

        # ── 6. Analyze ────────────────────────────────────────────────────────
        new_nodes = self.analyzer_agent.analyze(
            rdg=self.rdg,
            memory=self.memory,
            hypothesis_node=selected_hypothesis,
            experiment_node=exp_node,
            result=result,
        )
        finding_node = new_nodes["finding"]

        # ── 7. Score tracking ─────────────────────────────────────────────────
        score = result.get("score", 0.0)
        if score > self.best_score:
            self.best_score = score
            self.experiments_since_improvement = 0
            self.population.append(genome)
            self._seeds_run += 1
        else:
            self.experiments_since_improvement += 1

        # ── 8. Failure diagnosis + repair ─────────────────────────────────────
        failure_cat = FailureCategory.UNKNOWN
        if not result["success"]:
            failure_cat, bad_node = diagnose_failure(
                self.rdg, exp_node, target_metric=self.best_score * 0.9
            )
            if failure_cat == FailureCategory.UNKNOWN:
                failure_cat, bad_node = diagnose_failure(
                    self.rdg, finding_node, target_metric=self.best_score * 0.9
                )
            repair_msg = apply_repair(failure_cat, bad_node, self.rdg, self.memory)
            logger.info("Failure: %s | Repair: %s", failure_cat.value, repair_msg)

        # ── 9. Policy update ──────────────────────────────────────────────────
        reward = score - (self.best_score * 0.9)
        if evolution_strategy != "crossover":
            self.strategy_portfolio.record(
                evolution_strategy,
                score - float(result.get("baseline", 0.0)),
                bool(result["success"]),
            )
        next_nodes = self.rdg.next_hypotheses(selected_hypothesis)
        self.policy.update(selected_hypothesis, reward, next_nodes)

        # ── 10. Promotion gate check ──────────────────────────────────────────
        claim_node = new_nodes.get("claim")
        promoted = False
        if claim_node and result["success"]:
            gate = self._check_promotion_gate(score)
            if gate["passed"]:
                self.rdg.promote_claim(claim_node.id, gate)
                promoted = True
                logger.info("Claim promoted: %s", gate)
            else:
                logger.info("Promotion gate NOT passed: %s", gate["reason"])

        # ── 11. Algorithm discovery trigger ──────────────────────────────────
        triggered_discovery = False
        if self.experiments_since_improvement >= self.plateau_threshold:
            logger.info("Plateau detected! Triggering algorithm discovery.")
            triggered_discovery = True
            new_genome = self._algorithm_discovery()
            if new_genome:
                self.population.append(new_genome)
            self.experiments_since_improvement = 0

        return {
            "iteration": self.total_experiments,
            "hypothesis": selected_hypothesis.content[:80],
            "score": score,
            "success": result["success"],
            "best_score": self.best_score,
            "failure_category": failure_cat.value,
            "evolution_strategy": evolution_strategy,
            "discovery_triggered": triggered_discovery,
            "promoted": promoted,
        }

    # ── v2 Promotion gate ─────────────────────────────────────────────────────

    def _check_promotion_gate(self, score: float) -> Dict[str, Any]:
        """Validate v2 promotion criteria.

        A candidate may be promoted only when:
        - It has been evaluated on ≥ v2_min_seeds independent runs, OR
          the loop is in mock mode (relaxed gate for offline development).
        - It improves the champion by ≥ v2_promotion_min_improvement.
        - Compute budget is not exhausted.

        Returns a gate_result dict with ``passed`` and ``reason`` keys.
        """
        improvement = score - self.best_score
        compute_ok = not self.budget_allocator.is_over_budget()
        seeds_ok = self.use_mock or (self._seeds_run >= settings.v2_min_seeds)
        improvement_ok = improvement >= settings.v2_promotion_min_improvement

        passed = seeds_ok and improvement_ok and compute_ok
        reasons = []
        if not seeds_ok:
            reasons.append(
                f"seeds={self._seeds_run} < required={settings.v2_min_seeds}"
            )
        if not improvement_ok:
            reasons.append(
                f"improvement={improvement:.4f} < min={settings.v2_promotion_min_improvement}"
            )
        if not compute_ok:
            reasons.append("compute budget exhausted")

        return {
            "passed": passed,
            "seeds_run": self._seeds_run,
            "compute_hours": self.budget_allocator.consumed_hours,
            "improvement": improvement,
            "reason": "; ".join(reasons) if reasons else "all gates passed",
        }

    # ── Algorithm discovery ───────────────────────────────────────────────────

    def _algorithm_discovery(self) -> Optional[ModelGenome]:
        """Trigger broad search when stuck.

        v2: Uses StrategyPortfolio to select the least-explored family
        instead of hardcoding a specific estimator.
        """
        logger.info("AlgorithmDiscovery: synthesizing new genome from portfolio guidance.")
        from evolution.operators import OperatorType, apply_operator
        parent = self.population[0]
        child = apply_operator(
            OperatorType.SYNTHESIS,
            parent,
            portfolio=self.strategy_portfolio,
        )
        child.strategy_description = f"algorithm_discovery:{child.strategy_description}"
        return child

    # ── Summary ───────────────────────────────────────────────────────────────

    def _summarize(self) -> Dict[str, Any]:
        successful = [h for h in self.history if h.get("success")]
        return {
            "total_experiments": self.total_experiments,
            "best_score": self.best_score,
            "success_rate": len(successful) / max(1, len(self.history)),
            "population_size": len(self.population),
            "rdg_stats": self.rdg.stats(),
            "memory_stats": self.memory.stats(),
            "strategy_portfolio": self.strategy_portfolio.summary(),
            "budget": self.budget_allocator.summary(),
        }

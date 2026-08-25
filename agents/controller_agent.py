"""
ResearchController (Research Director): the central orchestrator that runs
the full ResearchForge-ECRM research loop.

Loop:
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
from failure.diagnosis import diagnose_failure
from failure.repair import apply_repair
from failure.taxonomy import FailureCategory

logger = logging.getLogger(__name__)


class ResearchController:
    """
    Central research loop controller.
    Manages all agents, RDG, memory, and policy.
    """

    def __init__(
        self,
        rdg: ResearchDevelopmentGraph,
        memory: ECRMMemoryStore,
        problem_description: str = "",
        max_experiments: int = 50,
        plateau_threshold: int = 5,
        use_mock_experiments: bool = True,
    ):
        self.rdg = rdg
        self.memory = memory
        self.problem_description = problem_description
        self.max_experiments = max_experiments
        self.plateau_threshold = plateau_threshold
        self.use_mock = use_mock_experiments

        # Agents
        self.hypothesis_agent = HypothesisAgent()
        self.experiment_agent = ExperimentAgent()
        self.analyzer_agent = AnalyzerAgent()

        # Policy
        self.policy = QLearningPolicy()

        # State
        self.total_experiments = 0
        self.best_score = 0.0
        self.experiments_since_improvement = 0
        self.population: List[ModelGenome] = [ModelGenome()]  # seed genome
        self.history: List[Dict[str, Any]] = []

    # ── Main research loop ────────────────────────────────────────────────────

    def run(self, n_iterations: int = 10) -> Dict[str, Any]:
        """Execute n_iterations of the research loop."""
        logger.info(
            "ResearchController starting: problem='%s', iterations=%d",
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
        )

        # ── 2. Generate hypothesis (if none exist or need new one) ────────────
        if selected_hypothesis is None or self.experiments_since_improvement >= 3:
            gap_nodes = [n for n in self.rdg if n.type == NodeType.GAP]
            if not gap_nodes:
                # Create a default gap
                gap = RDGNode.gap(content=f"Gap: {self.problem_description}")
                self.rdg.add_node(gap)
                gap_nodes = [gap]

            gap_node = gap_nodes[-1]
            hyp_text = self.hypothesis_agent.generate(
                gap_node=gap_node,
                memory=self.memory,
            )
            selected_hypothesis = RDGNode.hypothesis(content=hyp_text)
            self.rdg.add_node(selected_hypothesis)
            self.rdg.connect(gap_node.id, selected_hypothesis.id, EdgeRelation.MOTIVATES)

        # ── 3. Evolve genome ──────────────────────────────────────────────────
        parent_genome = self.population[-1]
        if len(self.population) >= 2 and self.total_experiments % 5 == 0:
            genome = crossover(self.population[-1], self.population[-2])
        else:
            genome = random_mutation(parent_genome)

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
            use_mock=self.use_mock,
        )
        self.total_experiments += 1

        # Update experiment node status
        exp_node.status = NodeStatus.COMPLETED if result["success"] else NodeStatus.FAILED
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
            self.population.append(genome)  # keep successful genome
        else:
            self.experiments_since_improvement += 1

        # ── 8. Failure diagnosis + repair ─────────────────────────────────────
        failure_cat = FailureCategory.UNKNOWN
        if not result["success"]:
            failure_cat, bad_node = diagnose_failure(
                self.rdg, finding_node, target_metric=self.best_score * 0.9
            )
            repair_msg = apply_repair(failure_cat, bad_node, self.rdg, self.memory)
            logger.info("Failure: %s | Repair: %s", failure_cat.value, repair_msg)

        # ── 9. Policy update ──────────────────────────────────────────────────
        reward = score - (self.best_score * 0.9)  # relative improvement
        next_nodes = self.rdg.next_hypotheses(selected_hypothesis)
        self.policy.update(selected_hypothesis, reward, next_nodes)

        # ── 10. Algorithm discovery trigger ───────────────────────────────────
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
            "discovery_triggered": triggered_discovery,
        }

    def _algorithm_discovery(self) -> Optional[ModelGenome]:
        """Trigger broad search when stuck."""
        logger.info("AlgorithmDiscovery: synthesizing new genome from LLM prompt.")
        # Apply synthesis operator (double-mutation)
        from evolution.operators import OperatorType, apply_operator
        parent = self.population[0]  # start from base genome
        child = apply_operator(OperatorType.SYNTHESIS, parent)
        return child

    def _summarize(self) -> Dict[str, Any]:
        successful = [h for h in self.history if h.get("success")]
        return {
            "total_experiments": self.total_experiments,
            "best_score": self.best_score,
            "success_rate": len(successful) / max(1, len(self.history)),
            "population_size": len(self.population),
            "rdg_stats": self.rdg.stats(),
            "memory_stats": self.memory.stats(),
        }

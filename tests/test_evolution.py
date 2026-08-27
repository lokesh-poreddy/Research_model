"""
Unit tests for Evolution operators.
"""
import pytest
from evolution.genome import ModelGenome
from policy.strategy_portfolio import StrategyPortfolio
from evolution.mutate import (
    param_mutation,
    optimizer_mutation,
    structure_mutation_add_layer,
    structure_mutation_remove_layer,
    augmentation_mutation,
    random_mutation,
)
from evolution.crossover import crossover, single_point_crossover, uniform_crossover
from evolution.operators import OperatorType, apply_operator


class TestModelGenome:
    def test_default_genome_valid(self):
        g = ModelGenome()
        assert g.validate()

    def test_fingerprint_stable(self):
        g = ModelGenome()
        assert g.fingerprint() == g.fingerprint()

    def test_copy_independence(self):
        g = ModelGenome()
        g2 = g.copy()
        g2.hyperparameters["learning_rate"] = 999.0
        assert g.hyperparameters["learning_rate"] != 999.0

    def test_json_roundtrip(self):
        g = ModelGenome()
        g2 = ModelGenome.from_json(g.to_json())
        assert g.model_id == g2.model_id
        assert g.fingerprint() == g2.fingerprint()


class TestMutationOperators:
    def setup_method(self):
        self.genome = ModelGenome()

    def test_param_mutation_changes_hyperparams(self):
        mutated = param_mutation(self.genome, delta=0.5)
        # At least one param changed
        changed = any(
            self.genome.hyperparameters[k] != mutated.hyperparameters[k]
            for k in self.genome.hyperparameters
            if isinstance(self.genome.hyperparameters[k], (int, float))
        )
        assert changed

    def test_policy_selected_mutation_is_recorded_in_genome(self):
        child = random_mutation(ModelGenome(), operator_hint="optimizer_mutation")
        assert child.strategy_description == "optimizer_mutation"


    def test_param_mutation_increments_generation(self):
        mutated = param_mutation(self.genome)
        assert mutated.generation == self.genome.generation + 1

    def test_param_mutation_sets_parent_id(self):
        mutated = param_mutation(self.genome)
        assert mutated.parent_id == self.genome.model_id

    def test_optimizer_mutation_changes_optimizer(self):
        mutated = optimizer_mutation(self.genome)
        assert mutated.hyperparameters["optimizer"] != self.genome.hyperparameters["optimizer"]

    def test_add_layer_increases_count(self):
        initial_count = len(self.genome.architecture["layers"])
        mutated = structure_mutation_add_layer(self.genome)
        assert len(mutated.architecture["layers"]) == initial_count + 1

    def test_remove_layer_decreases_count(self):
        initial_count = len(self.genome.architecture["layers"])
        mutated = structure_mutation_remove_layer(self.genome)
        # Count should be <= initial (may not remove if at minimum)
        assert len(mutated.architecture["layers"]) <= initial_count

    def test_augmentation_mutation(self):
        mutated = augmentation_mutation(self.genome)
        # augmentations list may grow or shrink
        assert isinstance(mutated.data_settings.get("augmentations"), list)

    def test_random_mutation_returns_child(self):
        child = random_mutation(self.genome)
        assert child.model_id != self.genome.model_id


class TestStrategyPortfolio:
    def test_portfolio_switches_away_from_saturated_strategy(self):
        portfolio = StrategyPortfolio()
        for _ in range(8):
            portfolio.record("param_mutation", improvement=-0.1, success=False)
        assert portfolio.select() != "param_mutation"


class TestCrossover:
    def test_single_point_crossover(self):
        parent_a = ModelGenome()
        parent_b = ModelGenome()
        parent_b.hyperparameters["learning_rate"] = 0.01
        parent_b.hyperparameters["optimizer"] = "SGD"
        child = single_point_crossover(parent_a, parent_b)
        assert child.model_id not in (parent_a.model_id, parent_b.model_id)
        assert child.generation > 0

    def test_uniform_crossover(self):
        parent_a = ModelGenome()
        parent_b = ModelGenome()
        child = uniform_crossover(parent_a, parent_b)
        assert child.model_id not in (parent_a.model_id, parent_b.model_id)

    def test_crossover_dispatch_single(self):
        a = ModelGenome()
        b = ModelGenome()
        child = crossover(a, b, mode="single")
        assert child is not None

    def test_crossover_dispatch_uniform(self):
        a = ModelGenome()
        b = ModelGenome()
        child = crossover(a, b, mode="uniform")
        assert child is not None


class TestOperatorRegistry:
    def test_apply_param_mutation(self):
        g = ModelGenome()
        result = apply_operator(OperatorType.PARAM_MUTATION, g)
        assert result.model_id != g.model_id

    def test_apply_crossover_requires_partner(self):
        g = ModelGenome()
        with pytest.raises(ValueError):
            apply_operator(OperatorType.CROSSOVER, g)

    def test_apply_crossover_with_partner(self):
        a = ModelGenome()
        b = ModelGenome()
        child = apply_operator(OperatorType.CROSSOVER, a, partner=b)
        assert child is not None

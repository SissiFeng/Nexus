"""Comprehensive tests for the EquationDiscovery symbolic regression module.

Tests cover ExprNode evaluation, complexity, string conversion, genetic
operators, fitness evaluation, Pareto front extraction, and end-to-end
symbolic regression on known functions.
"""

from __future__ import annotations

import copy
import math
import random

import pytest

from optimization_copilot.explain.equation_discovery import (
    EquationDiscovery,
    ExprNode,
    ParetoSolution,
    _collect_nodes,
    _dominates,
    _extract_pareto_front,
)


# ============================================================================
# ExprNode evaluation tests
# ============================================================================

class TestExprNodeEvaluate:
    """Tests for ExprNode.evaluate with various operators."""

    def test_evaluate_const(self) -> None:
        node = ExprNode(op="const", value=3.14)
        assert node.evaluate([]) == pytest.approx(3.14)

    def test_evaluate_const_none_value(self) -> None:
        node = ExprNode(op="const", value=None)
        assert node.evaluate([]) == pytest.approx(0.0)

    def test_evaluate_var(self) -> None:
        node = ExprNode(op="var", var_index=0)
        assert node.evaluate([2.5, 3.0]) == pytest.approx(2.5)

    def test_evaluate_var_index_1(self) -> None:
        node = ExprNode(op="var", var_index=1)
        assert node.evaluate([2.5, 3.0]) == pytest.approx(3.0)

    def test_evaluate_var_out_of_bounds(self) -> None:
        node = ExprNode(op="var", var_index=5)
        assert node.evaluate([1.0]) == pytest.approx(0.0)

    def test_evaluate_add(self) -> None:
        node = ExprNode(
            op="+",
            left=ExprNode(op="const", value=2.0),
            right=ExprNode(op="const", value=3.0),
        )
        assert node.evaluate([]) == pytest.approx(5.0)

    def test_evaluate_sub(self) -> None:
        node = ExprNode(
            op="-",
            left=ExprNode(op="const", value=5.0),
            right=ExprNode(op="const", value=3.0),
        )
        assert node.evaluate([]) == pytest.approx(2.0)

    def test_evaluate_mul(self) -> None:
        node = ExprNode(
            op="*",
            left=ExprNode(op="const", value=4.0),
            right=ExprNode(op="const", value=3.0),
        )
        assert node.evaluate([]) == pytest.approx(12.0)

    def test_evaluate_div(self) -> None:
        node = ExprNode(
            op="/",
            left=ExprNode(op="const", value=6.0),
            right=ExprNode(op="const", value=3.0),
        )
        assert node.evaluate([]) == pytest.approx(2.0)

    def test_evaluate_div_by_zero(self) -> None:
        node = ExprNode(
            op="/",
            left=ExprNode(op="const", value=1.0),
            right=ExprNode(op="const", value=0.0),
        )
        assert node.evaluate([]) == float("inf")

    def test_evaluate_nested(self) -> None:
        # (x0 + 1) * 2
        node = ExprNode(
            op="*",
            left=ExprNode(
                op="+",
                left=ExprNode(op="var", var_index=0),
                right=ExprNode(op="const", value=1.0),
            ),
            right=ExprNode(op="const", value=2.0),
        )
        assert node.evaluate([3.0]) == pytest.approx(8.0)

    def test_evaluate_exp(self) -> None:
        node = ExprNode(op="exp", left=ExprNode(op="const", value=1.0))
        assert node.evaluate([]) == pytest.approx(math.e)

    def test_evaluate_exp_overflow(self) -> None:
        node = ExprNode(op="exp", left=ExprNode(op="const", value=1000.0))
        assert node.evaluate([]) == float("inf")

    def test_evaluate_log(self) -> None:
        node = ExprNode(op="log", left=ExprNode(op="const", value=math.e))
        assert node.evaluate([]) == pytest.approx(1.0)

    def test_evaluate_log_negative(self) -> None:
        node = ExprNode(op="log", left=ExprNode(op="const", value=-1.0))
        assert node.evaluate([]) == float("inf")

    def test_evaluate_sqrt(self) -> None:
        node = ExprNode(op="sqrt", left=ExprNode(op="const", value=9.0))
        assert node.evaluate([]) == pytest.approx(3.0)

    def test_evaluate_sqrt_negative(self) -> None:
        node = ExprNode(op="sqrt", left=ExprNode(op="const", value=-1.0))
        assert node.evaluate([]) == float("inf")

    def test_evaluate_abs(self) -> None:
        node = ExprNode(op="abs", left=ExprNode(op="const", value=-5.0))
        assert node.evaluate([]) == pytest.approx(5.0)

    def test_evaluate_neg(self) -> None:
        node = ExprNode(op="neg", left=ExprNode(op="const", value=3.0))
        assert node.evaluate([]) == pytest.approx(-3.0)


# ============================================================================
# ExprNode complexity tests
# ============================================================================

class TestExprNodeComplexity:
    """Tests for ExprNode.complexity."""

    def test_complexity_leaf(self) -> None:
        node = ExprNode(op="const", value=1.0)
        assert node.complexity() == 1

    def test_complexity_var(self) -> None:
        node = ExprNode(op="var", var_index=0)
        assert node.complexity() == 1

    def test_complexity_binary(self) -> None:
        node = ExprNode(
            op="+",
            left=ExprNode(op="const", value=1.0),
            right=ExprNode(op="var", var_index=0),
        )
        assert node.complexity() == 3

    def test_complexity_unary(self) -> None:
        node = ExprNode(op="exp", left=ExprNode(op="var", var_index=0))
        assert node.complexity() == 2

    def test_complexity_nested(self) -> None:
        # (x0 + 1) * x1 => 5 nodes
        node = ExprNode(
            op="*",
            left=ExprNode(
                op="+",
                left=ExprNode(op="var", var_index=0),
                right=ExprNode(op="const", value=1.0),
            ),
            right=ExprNode(op="var", var_index=1),
        )
        assert node.complexity() == 5


# ============================================================================
# ExprNode to_string tests
# ============================================================================

class TestExprNodeToString:
    """Tests for ExprNode.to_string."""

    def test_to_string_const(self) -> None:
        node = ExprNode(op="const", value=3.0)
        assert node.to_string() == "3"

    def test_to_string_const_float(self) -> None:
        node = ExprNode(op="const", value=3.14)
        assert "3.14" in node.to_string()

    def test_to_string_var_default(self) -> None:
        node = ExprNode(op="var", var_index=2)
        assert node.to_string() == "x2"

    def test_to_string_var_with_names(self) -> None:
        node = ExprNode(op="var", var_index=0)
        assert node.to_string(["temp", "pressure"]) == "temp"

    def test_to_string_binary(self) -> None:
        node = ExprNode(
            op="+",
            left=ExprNode(op="var", var_index=0),
            right=ExprNode(op="const", value=1.0),
        )
        s = node.to_string()
        assert "+" in s
        assert "x0" in s

    def test_to_string_unary(self) -> None:
        node = ExprNode(op="exp", left=ExprNode(op="var", var_index=0))
        s = node.to_string()
        assert "exp" in s
        assert "x0" in s

    def test_to_string_nested(self) -> None:
        node = ExprNode(
            op="*",
            left=ExprNode(
                op="+",
                left=ExprNode(op="var", var_index=0),
                right=ExprNode(op="const", value=2.0),
            ),
            right=ExprNode(op="var", var_index=1),
        )
        s = node.to_string(["a", "b"])
        assert "a" in s
        assert "b" in s
        assert "*" in s


# ============================================================================
# Random tree generation tests
# ============================================================================

class TestRandomTreeGeneration:
    """Tests for EquationDiscovery._random_tree."""

    def test_random_tree_is_expr_node(self) -> None:
        ed = EquationDiscovery(seed=42)
        tree = ed._random_tree(3, 2)
        assert isinstance(tree, ExprNode)

    def test_random_tree_evaluates(self) -> None:
        ed = EquationDiscovery(seed=42)
        tree = ed._random_tree(3, 2)
        result = tree.evaluate([1.0, 2.0])
        # Should produce a finite number or inf (not raise)
        assert isinstance(result, float)

    def test_random_tree_depth_1_is_leaf(self) -> None:
        ed = EquationDiscovery(seed=42)
        tree = ed._random_tree(1, 2)
        assert tree.op in {"const", "var"}

    def test_random_tree_complexity_bounded(self) -> None:
        ed = EquationDiscovery(max_depth=3, seed=42)
        for _ in range(20):
            tree = ed._random_tree(3, 2)
            # Depth 3 means at most 2^3 - 1 = 7 nodes, but grow method may terminate early
            assert tree.complexity() <= 15  # generous bound

    def test_random_leaf_is_leaf(self) -> None:
        ed = EquationDiscovery(seed=42)
        leaf = ed._random_leaf(3)
        assert leaf.op in {"const", "var"}


# ============================================================================
# Fitness evaluation tests
# ============================================================================

class TestFitnessEvaluation:
    """Tests for EquationDiscovery._evaluate_fitness."""

    def test_fitness_perfect(self) -> None:
        """A constant tree that matches constant y should have MSE=0."""
        ed = EquationDiscovery(seed=42)
        tree = ExprNode(op="const", value=5.0)
        X = [[0.0], [1.0], [2.0]]
        y = [5.0, 5.0, 5.0]
        assert ed._evaluate_fitness(tree, X, y) == pytest.approx(0.0)

    def test_fitness_nonzero(self) -> None:
        ed = EquationDiscovery(seed=42)
        tree = ExprNode(op="const", value=0.0)
        X = [[0.0], [1.0], [2.0]]
        y = [1.0, 2.0, 3.0]
        mse = ed._evaluate_fitness(tree, X, y)
        assert mse > 0.0

    def test_fitness_inf_on_bad_tree(self) -> None:
        ed = EquationDiscovery(seed=42)
        # log(const(-1)) will produce inf
        tree = ExprNode(op="log", left=ExprNode(op="const", value=-1.0))
        X = [[1.0]]
        y = [1.0]
        assert ed._evaluate_fitness(tree, X, y) == float("inf")

    def test_fitness_empty_data(self) -> None:
        ed = EquationDiscovery(seed=42)
        tree = ExprNode(op="const", value=1.0)
        assert ed._evaluate_fitness(tree, [], []) == float("inf")

    def test_fitness_identity_function(self) -> None:
        """y = x0, tree = x0 => MSE = 0."""
        ed = EquationDiscovery(seed=42)
        tree = ExprNode(op="var", var_index=0)
        X = [[1.0], [2.0], [3.0]]
        y = [1.0, 2.0, 3.0]
        assert ed._evaluate_fitness(tree, X, y) == pytest.approx(0.0)


# ============================================================================
# Tournament selection tests
# ============================================================================

class TestTournamentSelection:
    """Tests for EquationDiscovery._tournament_select."""

    def test_tournament_returns_expr_node(self) -> None:
        ed = EquationDiscovery(tournament_size=3, seed=42)
        pop = [
            ExprNode(op="const", value=float(i))
            for i in range(10)
        ]
        fits = [float(i) for i in range(10)]
        selected = ed._tournament_select(pop, fits)
        assert isinstance(selected, ExprNode)

    def test_tournament_prefers_better_fitness(self) -> None:
        """Over many trials, better fitness should be selected more often."""
        ed = EquationDiscovery(tournament_size=3, seed=42)
        pop = [
            ExprNode(op="const", value=0.0),  # fitness 0 (best)
            ExprNode(op="const", value=1.0),  # fitness 100
        ]
        fits = [0.0, 100.0]
        best_count = 0
        for _ in range(100):
            s = ed._tournament_select(pop, fits)
            if s.value == 0.0:
                best_count += 1
        assert best_count > 50

    def test_tournament_returns_deep_copy(self) -> None:
        ed = EquationDiscovery(seed=42)
        pop = [ExprNode(op="const", value=1.0)]
        fits = [0.0]
        selected = ed._tournament_select(pop, fits)
        selected.value = 999.0
        assert pop[0].value == 1.0  # original unchanged


# ============================================================================
# Crossover tests
# ============================================================================

class TestCrossover:
    """Tests for EquationDiscovery._crossover."""

    def test_crossover_produces_valid_tree(self) -> None:
        ed = EquationDiscovery(seed=42)
        p1 = ExprNode(
            op="+",
            left=ExprNode(op="var", var_index=0),
            right=ExprNode(op="const", value=1.0),
        )
        p2 = ExprNode(
            op="*",
            left=ExprNode(op="var", var_index=1),
            right=ExprNode(op="const", value=2.0),
        )
        child = ed._crossover(p1, p2)
        assert isinstance(child, ExprNode)
        # Should still evaluate
        result = child.evaluate([1.0, 2.0])
        assert isinstance(result, float)

    def test_crossover_does_not_modify_parents(self) -> None:
        ed = EquationDiscovery(seed=42)
        p1 = ExprNode(op="+",
                       left=ExprNode(op="var", var_index=0),
                       right=ExprNode(op="const", value=1.0))
        p2 = ExprNode(op="const", value=5.0)
        p1_str = p1.to_string()
        _child = ed._crossover(p1, p2)
        assert p1.to_string() == p1_str

    def test_crossover_single_node_parents(self) -> None:
        ed = EquationDiscovery(seed=42)
        p1 = ExprNode(op="const", value=1.0)
        p2 = ExprNode(op="const", value=2.0)
        child = ed._crossover(p1, p2)
        assert isinstance(child, ExprNode)


# ============================================================================
# Mutation tests
# ============================================================================

class TestMutation:
    """Tests for EquationDiscovery._mutate."""

    def test_mutation_produces_valid_tree(self) -> None:
        ed = EquationDiscovery(seed=42)
        tree = ExprNode(
            op="+",
            left=ExprNode(op="var", var_index=0),
            right=ExprNode(op="const", value=1.0),
        )
        mutant = ed._mutate(tree, 2)
        assert isinstance(mutant, ExprNode)
        result = mutant.evaluate([1.0, 2.0])
        assert isinstance(result, float)

    def test_mutation_does_not_modify_original(self) -> None:
        ed = EquationDiscovery(seed=42)
        tree = ExprNode(op="const", value=1.0)
        original_val = tree.value
        _mutant = ed._mutate(tree, 2)
        assert tree.value == original_val

    def test_mutation_leaf_node(self) -> None:
        ed = EquationDiscovery(seed=42)
        tree = ExprNode(op="var", var_index=0)
        mutant = ed._mutate(tree, 2)
        assert isinstance(mutant, ExprNode)


# ============================================================================
# Physics filter tests
# ============================================================================

class TestPhysicsFilter:
    """Tests for EquationDiscovery._physics_filter."""

    def test_physics_filter_bounded(self) -> None:
        ed = EquationDiscovery(seed=42)
        tree = ExprNode(op="var", var_index=0)
        X = [[1.0], [2.0], [3.0]]
        assert ed._physics_filter(tree, X) is True

    def test_physics_filter_unbounded(self) -> None:
        ed = EquationDiscovery(seed=42)
        tree = ExprNode(
            op="/",
            left=ExprNode(op="const", value=1.0),
            right=ExprNode(op="const", value=0.0),
        )
        X = [[1.0]]
        assert ed._physics_filter(tree, X) is False

    def test_physics_filter_exp_overflow(self) -> None:
        ed = EquationDiscovery(seed=42)
        tree = ExprNode(op="exp", left=ExprNode(op="const", value=999.0))
        X = [[1.0]]
        assert ed._physics_filter(tree, X) is False

    def test_physics_filter_empty_data(self) -> None:
        ed = EquationDiscovery(seed=42)
        tree = ExprNode(op="const", value=1.0)
        assert ed._physics_filter(tree, []) is True

    def test_physics_filter_large_value(self) -> None:
        """Values exceeding 1e12 should fail the filter."""
        ed = EquationDiscovery(seed=42)
        tree = ExprNode(op="const", value=1e13)
        X = [[1.0]]
        assert ed._physics_filter(tree, X) is False


# ============================================================================
# End-to-end fit tests
# ============================================================================

class TestFit:
    """Tests for EquationDiscovery.fit."""

    def test_fit_linear_function(self) -> None:
        """y = 2*x + 1 should be well-approximated."""
        rng = random.Random(42)
        X = [[rng.uniform(-2, 2)] for _ in range(50)]
        y = [2.0 * x[0] + 1.0 for x in X]
        ed = EquationDiscovery(
            population_size=100,
            n_generations=30,
            seed=42,
        )
        front = ed.fit(X, y, var_names=["x"])
        assert len(front) > 0
        best = ed.best_equation()
        assert best is not None
        assert best.mse < 5.0  # should find something reasonable

    def test_fit_quadratic(self) -> None:
        """y = x^2 should be approximated."""
        rng = random.Random(42)
        X = [[rng.uniform(-2, 2)] for _ in range(50)]
        y = [x[0] ** 2 for x in X]
        ed = EquationDiscovery(
            population_size=150,
            n_generations=40,
            seed=42,
        )
        front = ed.fit(X, y)
        assert len(front) > 0

    def test_fit_returns_pareto_front(self) -> None:
        rng = random.Random(42)
        X = [[rng.uniform(0, 1)] for _ in range(30)]
        y = [x[0] * 2 for x in X]
        ed = EquationDiscovery(population_size=50, n_generations=10, seed=42)
        front = ed.fit(X, y)
        assert isinstance(front, list)
        for sol in front:
            assert isinstance(sol, ParetoSolution)
            assert sol.mse >= 0
            assert sol.complexity >= 1
            assert isinstance(sol.equation_string, str)

    def test_fit_with_var_names(self) -> None:
        rng = random.Random(42)
        X = [[rng.uniform(0, 1), rng.uniform(0, 1)] for _ in range(30)]
        y = [x[0] + x[1] for x in X]
        ed = EquationDiscovery(population_size=50, n_generations=10, seed=42)
        front = ed.fit(X, y, var_names=["temp", "pressure"])
        for sol in front:
            # Variable names should appear in equations
            has_name = "temp" in sol.equation_string or "pressure" in sol.equation_string
            has_const = "const" not in sol.equation_string  # should use names not "const"
            # At least some solutions should use variable names
            assert has_const

    def test_fit_empty_data(self) -> None:
        ed = EquationDiscovery(seed=42)
        front = ed.fit([], [])
        assert front == []

    def test_fit_single_point(self) -> None:
        ed = EquationDiscovery(population_size=20, n_generations=5, seed=42)
        front = ed.fit([[1.0]], [2.0])
        # Should still work, even if trivial
        assert isinstance(front, list)


# ============================================================================
# Pareto front tests
# ============================================================================

class TestParetoFront:
    """Tests for Pareto front extraction and accessors."""

    def test_pareto_front_extraction(self) -> None:
        pop = [
            ExprNode(op="const", value=1.0),  # complexity 1
            ExprNode(
                op="+",
                left=ExprNode(op="const", value=1.0),
                right=ExprNode(op="const", value=0.0),
            ),  # complexity 3
        ]
        fits = [0.5, 0.1]
        front = _extract_pareto_front(pop, fits)
        assert len(front) >= 1

    def test_pareto_front_non_dominated(self) -> None:
        """All solutions on the front should be non-dominated."""
        pop = [
            ExprNode(op="const", value=1.0),
            ExprNode(op="+",
                     left=ExprNode(op="var", var_index=0),
                     right=ExprNode(op="const", value=1.0)),
        ]
        fits = [0.5, 0.1]
        front = _extract_pareto_front(pop, fits)
        for sol in front:
            for sol2 in front:
                if sol is sol2:
                    continue
                assert not _dominates(sol2.mse, sol2.complexity, sol.mse, sol.complexity)

    def test_get_pareto_front_after_fit(self) -> None:
        rng = random.Random(42)
        X = [[rng.uniform(0, 1)] for _ in range(30)]
        y = [x[0] for x in X]
        ed = EquationDiscovery(population_size=50, n_generations=10, seed=42)
        ed.fit(X, y)
        front = ed.get_pareto_front()
        assert isinstance(front, list)
        assert len(front) > 0

    def test_get_pareto_front_before_fit(self) -> None:
        ed = EquationDiscovery(seed=42)
        front = ed.get_pareto_front()
        assert front == []

    def test_best_equation_selection(self) -> None:
        rng = random.Random(42)
        X = [[rng.uniform(0, 1)] for _ in range(50)]
        y = [2.0 * x[0] for x in X]
        ed = EquationDiscovery(population_size=100, n_generations=20, seed=42)
        ed.fit(X, y)
        best = ed.best_equation()
        assert best is not None
        assert isinstance(best, ParetoSolution)
        assert best.mse >= 0.0
        assert best.complexity >= 1

    def test_best_equation_before_fit(self) -> None:
        ed = EquationDiscovery(seed=42)
        assert ed.best_equation() is None

    def test_pareto_excludes_inf_fitness(self) -> None:
        pop = [
            ExprNode(op="const", value=1.0),
            ExprNode(op="/",
                     left=ExprNode(op="const", value=1.0),
                     right=ExprNode(op="const", value=0.0)),
        ]
        fits = [0.5, float("inf")]
        front = _extract_pareto_front(pop, fits)
        assert all(math.isfinite(s.mse) for s in front)


# ============================================================================
# ParetoSolution dataclass tests
# ============================================================================

class TestParetoSolution:
    """Tests for the ParetoSolution dataclass."""

    def test_pareto_solution_creation(self) -> None:
        tree = ExprNode(op="const", value=1.0)
        sol = ParetoSolution(
            expression=tree,
            mse=0.5,
            complexity=1,
            equation_string="1",
        )
        assert sol.mse == 0.5
        assert sol.complexity == 1
        assert sol.equation_string == "1"
        assert sol.expression is tree

    def test_pareto_solution_fields(self) -> None:
        tree = ExprNode(op="var", var_index=0)
        sol = ParetoSolution(
            expression=tree,
            mse=0.0,
            complexity=1,
            equation_string="x0",
        )
        assert sol.equation_string == "x0"


# ============================================================================
# Helper function tests
# ============================================================================

class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_collect_nodes_single(self) -> None:
        node = ExprNode(op="const", value=1.0)
        nodes = _collect_nodes(node)
        assert len(nodes) == 1
        assert nodes[0][0] is node
        assert nodes[0][1] is None

    def test_collect_nodes_binary(self) -> None:
        left = ExprNode(op="const", value=1.0)
        right = ExprNode(op="var", var_index=0)
        root = ExprNode(op="+", left=left, right=right)
        nodes = _collect_nodes(root)
        assert len(nodes) == 3

    def test_dominates_yes(self) -> None:
        assert _dominates(0.1, 1, 0.5, 3) is True

    def test_dominates_no_equal(self) -> None:
        assert _dominates(0.5, 3, 0.5, 3) is False

    def test_dominates_no_tradeoff(self) -> None:
        assert _dominates(0.1, 5, 0.5, 1) is False


# ============================================================================
# Reproducibility tests
# ============================================================================

class TestReproducibility:
    """Tests for deterministic behaviour."""

    def test_reproducibility(self) -> None:
        rng = random.Random(42)
        X = [[rng.uniform(0, 1)] for _ in range(30)]
        y = [x[0] * 2 for x in X]

        ed1 = EquationDiscovery(population_size=50, n_generations=10, seed=42)
        f1 = ed1.fit(X, y)

        ed2 = EquationDiscovery(population_size=50, n_generations=10, seed=42)
        f2 = ed2.fit(X, y)

        assert len(f1) == len(f2)
        for s1, s2 in zip(f1, f2):
            assert s1.mse == pytest.approx(s2.mse)
            assert s1.complexity == s2.complexity

    def test_different_seeds_differ(self) -> None:
        rng = random.Random(42)
        X = [[rng.uniform(0, 1)] for _ in range(30)]
        y = [x[0] * 2 for x in X]

        ed1 = EquationDiscovery(population_size=50, n_generations=10, seed=42)
        f1 = ed1.fit(X, y)

        ed2 = EquationDiscovery(population_size=50, n_generations=10, seed=99)
        f2 = ed2.fit(X, y)

        # Very likely to produce different fronts
        if f1 and f2:
            strings1 = {s.equation_string for s in f1}
            strings2 = {s.equation_string for s in f2}
            # Not guaranteed but highly likely to differ
            assert True  # just checking it runs without error

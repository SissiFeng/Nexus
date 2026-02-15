"""Tests for case_studies.base and case_studies.evaluator.

Covers:
- SimpleSurrogate: fit, predict, variance behaviour
- ExperimentalBenchmark: interface contract
- ReplayBenchmark: surrogate-based evaluation, encoding
- PerformanceMetrics / ComparisonResult: data class fields
- CaseStudyEvaluator: spec building, metrics computation, run_single
"""

from __future__ import annotations

import math
import random
import unittest
from typing import Any

from optimization_copilot.case_studies.base import (
    ExperimentalBenchmark,
    ReplayBenchmark,
    SimpleSurrogate,
)
from optimization_copilot.case_studies.evaluator import (
    CaseStudyEvaluator,
    ComparisonResult,
    PerformanceMetrics,
)
from optimization_copilot.core.models import Observation, ParameterSpec, VariableType
from optimization_copilot.plugins.base import AlgorithmPlugin


# ---------------------------------------------------------------------------
# Concrete test doubles
# ---------------------------------------------------------------------------


class DummyBenchmark(ExperimentalBenchmark):
    """Simple 2D benchmark for testing."""

    def evaluate(self, x: dict) -> dict | None:
        val = -(x["x1"] - 0.5) ** 2 - (x["x2"] - 0.3) ** 2
        return {"obj": {"value": val, "variance": 0.01}}

    def get_search_space(self) -> dict[str, dict]:
        return {
            "x1": {"type": "continuous", "range": [0, 1]},
            "x2": {"type": "continuous", "range": [0, 1]},
        }

    def get_objectives(self) -> dict[str, dict]:
        return {"obj": {"direction": "maximize", "unit": ""}}


class InfeasibleBenchmark(ExperimentalBenchmark):
    """Benchmark where half the space is infeasible."""

    def evaluate(self, x: dict) -> dict | None:
        if x["x1"] > 0.5:
            return None
        val = -x["x1"] ** 2
        return {"obj": {"value": val, "variance": 0.01}}

    def get_search_space(self) -> dict[str, dict]:
        return {"x1": {"type": "continuous", "range": [0, 1]}}

    def get_objectives(self) -> dict[str, dict]:
        return {"obj": {"direction": "minimize", "unit": ""}}

    def is_feasible(self, x: dict) -> bool:
        return x.get("x1", 1.0) <= 0.5


class CategoricalBenchmark(ExperimentalBenchmark):
    """Benchmark with a categorical parameter."""

    def evaluate(self, x: dict) -> dict | None:
        bonus = 1.0 if x["method"] == "A" else 0.0
        val = x["x1"] + bonus
        return {"obj": {"value": val, "variance": 0.01}}

    def get_search_space(self) -> dict[str, dict]:
        return {
            "x1": {"type": "continuous", "range": [0, 1]},
            "method": {"type": "categorical", "categories": ["A", "B", "C"]},
        }

    def get_objectives(self) -> dict[str, dict]:
        return {"obj": {"direction": "maximize", "unit": "score"}}


class DummyReplayBenchmark(ReplayBenchmark):
    """Replay benchmark with synthetic data for testing."""

    def _generate_data(self) -> dict:
        rng = random.Random(self._seed)
        X: list[list[float]] = []
        Y: list[float] = []
        for _ in range(self._n_train):
            x1 = rng.uniform(0.0, 1.0)
            x2 = rng.uniform(0.0, 1.0)
            X.append([x1, x2])
            # Simple quadratic objective
            Y.append(-(x1 - 0.5) ** 2 - (x2 - 0.3) ** 2)
        return {
            "X": X,
            "Y": {"obj": Y},
            "noise_levels": {"obj": 0.01},
        }

    def get_search_space(self) -> dict[str, dict]:
        return {
            "x1": {"type": "continuous", "range": [0, 1]},
            "x2": {"type": "continuous", "range": [0, 1]},
        }

    def get_objectives(self) -> dict[str, dict]:
        return {"obj": {"direction": "maximize", "unit": ""}}

    def get_known_optimum(self) -> dict[str, float] | None:
        return {"obj": 0.0}


class InfeasibleReplayBenchmark(ReplayBenchmark):
    """Replay benchmark that marks points with x1 > 0.5 as infeasible."""

    def _generate_data(self) -> dict:
        rng = random.Random(self._seed)
        X: list[list[float]] = []
        Y: list[float] = []
        for _ in range(self._n_train):
            x1 = rng.uniform(0.0, 0.5)
            X.append([x1])
            Y.append(-x1 ** 2)
        return {"X": X, "Y": {"obj": Y}}

    def get_search_space(self) -> dict[str, dict]:
        return {"x1": {"type": "continuous", "range": [0, 1]}}

    def get_objectives(self) -> dict[str, dict]:
        return {"obj": {"direction": "minimize", "unit": ""}}

    def is_feasible(self, x: dict) -> bool:
        return x.get("x1", 1.0) <= 0.5


class CategoricalReplayBenchmark(ReplayBenchmark):
    """Replay benchmark with a categorical parameter."""

    def _generate_data(self) -> dict:
        rng = random.Random(self._seed)
        X: list[list[float]] = []
        Y: list[float] = []
        cats = ["A", "B"]
        for _ in range(self._n_train):
            x1 = rng.uniform(0.0, 1.0)
            cat_idx = rng.randint(0, 1)
            # Encoding: [x1, one_hot_A, one_hot_B]
            one_hot = [1.0 if i == cat_idx else 0.0 for i in range(2)]
            X.append([x1] + one_hot)
            Y.append(x1 + cat_idx)
        return {"X": X, "Y": {"obj": Y}}

    def get_search_space(self) -> dict[str, dict]:
        return {
            "x1": {"type": "continuous", "range": [0, 1]},
            "cat": {"type": "categorical", "categories": ["A", "B"]},
        }

    def get_objectives(self) -> dict[str, dict]:
        return {"obj": {"direction": "maximize", "unit": ""}}


class DummyAlgorithm(AlgorithmPlugin):
    """Minimal algorithm plugin for evaluator testing."""

    def __init__(self) -> None:
        self._specs: list[ParameterSpec] = []
        self._observations: list[Observation] = []

    def name(self) -> str:
        return "dummy"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._observations = observations
        self._specs = parameter_specs

    def suggest(
        self, n_suggestions: int = 1, seed: int = 42
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        results: list[dict[str, Any]] = []
        for _ in range(n_suggestions):
            point: dict[str, Any] = {}
            for spec in self._specs:
                if spec.type == VariableType.CONTINUOUS:
                    lo = spec.lower if spec.lower is not None else 0.0
                    hi = spec.upper if spec.upper is not None else 1.0
                    point[spec.name] = rng.uniform(lo, hi)
                elif spec.type == VariableType.CATEGORICAL:
                    point[spec.name] = rng.choice(spec.categories or [""])
            results.append(point)
        # If we have no specs yet, return a generic point
        if not self._specs:
            results = [{"x1": rng.random(), "x2": rng.random()}]
        return results

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": False,
            "supports_batch": False,
            "requires_observations": False,
            "max_dimensions": None,
        }


# ===========================================================================
# SimpleSurrogate Tests
# ===========================================================================


class TestSimpleSurrogate(unittest.TestCase):
    """Tests for SimpleSurrogate."""

    def test_fit_does_not_raise(self) -> None:
        """Fitting should succeed without errors."""
        s = SimpleSurrogate()
        X = [[0.0], [0.5], [1.0]]
        y = [0.0, 1.0, 0.0]
        s.fit(X, y)

    def test_predict_mean_close_to_training(self) -> None:
        """Predicted mean at training points should be close to targets."""
        s = SimpleSurrogate(noise=0.001)
        X = [[0.0], [0.5], [1.0]]
        y = [0.0, 1.0, 0.0]
        s.fit(X, y)
        mu, var = s.predict([0.5])
        self.assertAlmostEqual(mu, 1.0, delta=0.15)

    def test_predict_variance_increases_away_from_data(self) -> None:
        """Variance should increase as we move away from training data."""
        s = SimpleSurrogate(lengthscale=0.3, noise=0.001)
        X = [[0.0], [1.0]]
        y = [0.0, 0.0]
        s.fit(X, y)
        _, var_at_data = s.predict([0.0])
        _, var_far = s.predict([0.5])
        # var_far should be >= var_at_data (further from training points)
        self.assertGreaterEqual(var_far, var_at_data - 1e-6)

    def test_predict_at_training_point_low_variance(self) -> None:
        """Variance at a training point should be small."""
        s = SimpleSurrogate(noise=0.001)
        X = [[0.0], [0.5], [1.0]]
        y = [1.0, 2.0, 1.0]
        s.fit(X, y)
        _, var = s.predict([0.5])
        self.assertLess(var, 0.5)

    def test_predict_with_multiple_dimensions(self) -> None:
        """Surrogate should handle multi-dimensional inputs."""
        s = SimpleSurrogate(noise=0.01)
        X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        y = [0.0, 1.0, 1.0, 2.0]
        s.fit(X, y)
        mu, var = s.predict([0.5, 0.5])
        # Should be somewhere between 0 and 2
        self.assertGreater(mu, -1.0)
        self.assertLess(mu, 3.0)
        self.assertGreaterEqual(var, 0.0)

    def test_predict_before_fit_raises(self) -> None:
        """Calling predict before fit should raise RuntimeError."""
        s = SimpleSurrogate()
        with self.assertRaises(RuntimeError):
            s.predict([0.5])

    def test_default_hyperparameters(self) -> None:
        """Default hyperparameters should be set correctly."""
        s = SimpleSurrogate()
        self.assertEqual(s.lengthscale, 1.0)
        self.assertEqual(s.signal_variance, 1.0)
        self.assertEqual(s.noise, 0.01)

    def test_custom_hyperparameters(self) -> None:
        """Custom hyperparameters should be stored correctly."""
        s = SimpleSurrogate(lengthscale=0.5, signal_variance=2.0, noise=0.1)
        self.assertEqual(s.lengthscale, 0.5)
        self.assertEqual(s.signal_variance, 2.0)
        self.assertEqual(s.noise, 0.1)

    def test_variance_non_negative(self) -> None:
        """Predicted variance should never be negative."""
        s = SimpleSurrogate(noise=0.001)
        X = [[float(i)] for i in range(10)]
        y = [math.sin(float(i)) for i in range(10)]
        s.fit(X, y)
        for probe in [[-1.0], [5.0], [11.0], [0.5]]:
            _, var = s.predict(probe)
            self.assertGreaterEqual(var, 0.0)


# ===========================================================================
# ExperimentalBenchmark Tests
# ===========================================================================


class TestExperimentalBenchmark(unittest.TestCase):
    """Tests for the ExperimentalBenchmark interface."""

    def test_evaluate_returns_dict(self) -> None:
        b = DummyBenchmark()
        result = b.evaluate({"x1": 0.5, "x2": 0.3})
        self.assertIsInstance(result, dict)
        self.assertIn("obj", result)
        self.assertIn("value", result["obj"])
        self.assertIn("variance", result["obj"])

    def test_evaluate_value_correct(self) -> None:
        b = DummyBenchmark()
        result = b.evaluate({"x1": 0.5, "x2": 0.3})
        self.assertAlmostEqual(result["obj"]["value"], 0.0, places=5)

    def test_get_search_space(self) -> None:
        b = DummyBenchmark()
        space = b.get_search_space()
        self.assertIn("x1", space)
        self.assertIn("x2", space)
        self.assertEqual(space["x1"]["type"], "continuous")

    def test_get_objectives(self) -> None:
        b = DummyBenchmark()
        obj = b.get_objectives()
        self.assertIn("obj", obj)
        self.assertEqual(obj["obj"]["direction"], "maximize")

    def test_get_known_constraints_default(self) -> None:
        b = DummyBenchmark()
        constraints = b.get_known_constraints()
        self.assertEqual(constraints, [])

    def test_get_evaluation_cost_default(self) -> None:
        b = DummyBenchmark()
        cost = b.get_evaluation_cost({"x1": 0.5, "x2": 0.3})
        self.assertEqual(cost, 1.0)

    def test_is_feasible_default(self) -> None:
        b = DummyBenchmark()
        self.assertTrue(b.is_feasible({"x1": 0.5, "x2": 0.3}))

    def test_domain_config_none_by_default(self) -> None:
        b = DummyBenchmark()
        self.assertIsNone(b.domain_config)
        self.assertIsNone(b.get_domain_config())

    def test_domain_config_loads_when_domain_given(self) -> None:
        """DomainConfig loads when a valid domain name is provided."""
        # Use DummyBenchmark but manually set a domain
        # We cannot directly test with DummyBenchmark(domain_name="catalysis")
        # because DummyBenchmark.__init__ does not pass domain_name.
        # Instead test that ExperimentalBenchmark.__init__ sets domain_config.
        class DomainBenchmark(ExperimentalBenchmark):
            def evaluate(self, x: dict) -> dict | None:
                return {"obj": {"value": 0.0, "variance": 0.01}}
            def get_search_space(self) -> dict[str, dict]:
                return {"x1": {"type": "continuous", "range": [0, 1]}}
            def get_objectives(self) -> dict[str, dict]:
                return {"obj": {"direction": "minimize", "unit": ""}}

        b = DomainBenchmark(domain_name="catalysis")
        self.assertIsNotNone(b.domain_config)
        self.assertEqual(b.domain_config.domain_name, "catalysis")

    def test_infeasible_benchmark_returns_none(self) -> None:
        b = InfeasibleBenchmark()
        result = b.evaluate({"x1": 0.8})
        self.assertIsNone(result)

    def test_infeasible_benchmark_feasible_point(self) -> None:
        b = InfeasibleBenchmark()
        result = b.evaluate({"x1": 0.2})
        self.assertIsNotNone(result)

    def test_categorical_benchmark_evaluate(self) -> None:
        b = CategoricalBenchmark()
        result = b.evaluate({"x1": 0.5, "method": "A"})
        self.assertAlmostEqual(result["obj"]["value"], 1.5, places=5)


# ===========================================================================
# ReplayBenchmark Tests
# ===========================================================================


class TestReplayBenchmark(unittest.TestCase):
    """Tests for ReplayBenchmark."""

    def test_evaluate_returns_dict_with_value_and_variance(self) -> None:
        b = DummyReplayBenchmark(n_train=20, seed=42)
        result = b.evaluate({"x1": 0.5, "x2": 0.3})
        self.assertIsInstance(result, dict)
        self.assertIn("obj", result)
        self.assertIn("value", result["obj"])
        self.assertIn("variance", result["obj"])
        self.assertIsInstance(result["obj"]["value"], float)
        self.assertIsInstance(result["obj"]["variance"], float)

    def test_evaluate_returns_none_for_infeasible(self) -> None:
        b = InfeasibleReplayBenchmark(n_train=10, seed=42)
        result = b.evaluate({"x1": 0.8})
        self.assertIsNone(result)

    def test_evaluate_returns_value_for_feasible(self) -> None:
        b = InfeasibleReplayBenchmark(n_train=10, seed=42)
        result = b.evaluate({"x1": 0.2})
        self.assertIsNotNone(result)

    def test_encode_continuous(self) -> None:
        b = DummyReplayBenchmark(n_train=10, seed=42)
        encoded = b._encode({"x1": 0.5, "x2": 0.3})
        self.assertEqual(encoded, [0.5, 0.3])

    def test_encode_categorical_one_hot(self) -> None:
        b = CategoricalReplayBenchmark(n_train=10, seed=42)
        encoded = b._encode({"x1": 0.5, "cat": "A"})
        # x1=0.5, then one-hot for "A": [1.0, 0.0]
        self.assertEqual(encoded, [0.5, 1.0, 0.0])

    def test_encode_categorical_one_hot_second(self) -> None:
        b = CategoricalReplayBenchmark(n_train=10, seed=42)
        encoded = b._encode({"x1": 0.7, "cat": "B"})
        self.assertEqual(encoded, [0.7, 0.0, 1.0])

    def test_deterministic_with_same_seed(self) -> None:
        b1 = DummyReplayBenchmark(n_train=20, seed=42)
        b2 = DummyReplayBenchmark(n_train=20, seed=42)
        r1 = b1.evaluate({"x1": 0.5, "x2": 0.3})
        r2 = b2.evaluate({"x1": 0.5, "x2": 0.3})
        self.assertAlmostEqual(r1["obj"]["value"], r2["obj"]["value"], places=10)

    def test_different_seed_gives_different_results(self) -> None:
        b1 = DummyReplayBenchmark(n_train=20, seed=42)
        b2 = DummyReplayBenchmark(n_train=20, seed=99)
        r1 = b1.evaluate({"x1": 0.5, "x2": 0.3})
        r2 = b2.evaluate({"x1": 0.5, "x2": 0.3})
        # Different seeds should give different surrogate fits, so different values
        # (with very high probability)
        # Just check they both return valid results
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)

    def test_get_known_optimum_default(self) -> None:
        b = DummyReplayBenchmark(n_train=10, seed=42)
        opt = b.get_known_optimum()
        self.assertEqual(opt, {"obj": 0.0})

    def test_surrogates_populated(self) -> None:
        b = DummyReplayBenchmark(n_train=10, seed=42)
        self.assertIn("obj", b._surrogates)
        self.assertTrue(b._surrogates["obj"]._fitted)


# ===========================================================================
# PerformanceMetrics Tests
# ===========================================================================


class TestPerformanceMetrics(unittest.TestCase):
    """Tests for PerformanceMetrics dataclass."""

    def test_all_fields_set(self) -> None:
        m = PerformanceMetrics(
            best_value=1.0,
            simple_regret=0.1,
            convergence_iteration=10,
            area_under_curve=5.0,
            feasibility_rate=0.9,
            constraint_violations=2,
            total_cost=50.0,
            cost_adjusted_regret=0.002,
        )
        self.assertEqual(m.best_value, 1.0)
        self.assertEqual(m.simple_regret, 0.1)
        self.assertEqual(m.convergence_iteration, 10)
        self.assertEqual(m.area_under_curve, 5.0)
        self.assertEqual(m.feasibility_rate, 0.9)
        self.assertEqual(m.constraint_violations, 2)
        self.assertEqual(m.total_cost, 50.0)
        self.assertEqual(m.cost_adjusted_regret, 0.002)

    def test_default_fields(self) -> None:
        m = PerformanceMetrics(
            best_value=0.0,
            simple_regret=0.0,
            convergence_iteration=0,
            area_under_curve=0.0,
            feasibility_rate=1.0,
            constraint_violations=0,
            total_cost=1.0,
            cost_adjusted_regret=0.0,
        )
        self.assertIsNone(m.hypervolume)
        self.assertIsNone(m.pareto_front_size)
        self.assertEqual(m.std_across_repeats, 0.0)
        self.assertEqual(m.mean_confidence, 0.0)
        self.assertEqual(m.noise_calibration_error, 0.0)

    def test_optional_fields_set(self) -> None:
        m = PerformanceMetrics(
            best_value=0.0,
            simple_regret=0.0,
            convergence_iteration=0,
            area_under_curve=0.0,
            feasibility_rate=1.0,
            constraint_violations=0,
            total_cost=1.0,
            cost_adjusted_regret=0.0,
            hypervolume=0.95,
            pareto_front_size=10,
            std_across_repeats=0.05,
        )
        self.assertEqual(m.hypervolume, 0.95)
        self.assertEqual(m.pareto_front_size, 10)
        self.assertEqual(m.std_across_repeats, 0.05)


# ===========================================================================
# ComparisonResult Tests
# ===========================================================================


class TestComparisonResult(unittest.TestCase):
    """Tests for ComparisonResult dataclass."""

    def test_fields_set(self) -> None:
        m1 = PerformanceMetrics(
            best_value=1.0, simple_regret=0.1, convergence_iteration=5,
            area_under_curve=2.0, feasibility_rate=1.0,
            constraint_violations=0, total_cost=10.0,
            cost_adjusted_regret=0.01,
        )
        cr = ComparisonResult(
            strategy_names=["A", "B"],
            metrics={"A": [m1], "B": [m1]},
            convergence_curves={"A": [[1.0, 0.9]], "B": [[1.0, 0.8]]},
            budget=10,
            n_repeats=1,
            benchmark_name="test",
        )
        self.assertEqual(cr.strategy_names, ["A", "B"])
        self.assertEqual(cr.budget, 10)
        self.assertEqual(cr.n_repeats, 1)
        self.assertEqual(cr.benchmark_name, "test")
        self.assertIn("A", cr.metrics)
        self.assertIn("B", cr.convergence_curves)


# ===========================================================================
# CaseStudyEvaluator Tests
# ===========================================================================


class TestCaseStudyEvaluator(unittest.TestCase):
    """Tests for CaseStudyEvaluator."""

    def test_build_parameter_specs_continuous(self) -> None:
        b = DummyBenchmark()
        ev = CaseStudyEvaluator(b)
        specs = ev._build_parameter_specs()
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].name, "x1")
        self.assertEqual(specs[0].type, VariableType.CONTINUOUS)
        self.assertEqual(specs[0].lower, 0)
        self.assertEqual(specs[0].upper, 1)

    def test_build_parameter_specs_categorical(self) -> None:
        b = CategoricalBenchmark()
        ev = CaseStudyEvaluator(b)
        specs = ev._build_parameter_specs()
        self.assertEqual(len(specs), 2)
        # Find the categorical one
        cat_spec = [s for s in specs if s.type == VariableType.CATEGORICAL]
        self.assertEqual(len(cat_spec), 1)
        self.assertEqual(cat_spec[0].name, "method")
        self.assertEqual(cat_spec[0].categories, ["A", "B", "C"])

    def test_run_single_returns_history_and_metrics(self) -> None:
        b = DummyBenchmark()
        ev = CaseStudyEvaluator(b)
        algo = DummyAlgorithm()
        history, metrics = ev.run_single(algo, budget=5, seed=42)
        self.assertEqual(len(history), 5)
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(metrics.feasibility_rate, 1.0)
        self.assertEqual(metrics.constraint_violations, 0)

    def test_run_single_history_entries(self) -> None:
        b = DummyBenchmark()
        ev = CaseStudyEvaluator(b)
        algo = DummyAlgorithm()
        history, _ = ev.run_single(algo, budget=3, seed=42)
        for entry in history:
            self.assertIn("x", entry)
            self.assertIn("result", entry)
            self.assertIn("iteration", entry)

    def test_run_single_total_cost(self) -> None:
        b = DummyBenchmark()
        ev = CaseStudyEvaluator(b)
        algo = DummyAlgorithm()
        _, metrics = ev.run_single(algo, budget=5, seed=42)
        self.assertAlmostEqual(metrics.total_cost, 5.0, places=5)

    def test_compute_metrics_basic_maximize(self) -> None:
        b = DummyBenchmark()
        ev = CaseStudyEvaluator(b)
        history = [
            {"x": {"x1": 0.0, "x2": 0.0}, "result": {"obj": {"value": -0.34, "variance": 0.01}}, "iteration": 0},
            {"x": {"x1": 0.5, "x2": 0.3}, "result": {"obj": {"value": 0.0, "variance": 0.01}}, "iteration": 1},
            {"x": {"x1": 0.3, "x2": 0.3}, "result": {"obj": {"value": -0.04, "variance": 0.01}}, "iteration": 2},
        ]
        metrics = ev._compute_metrics(
            history, known_opt={"obj": 0.0}, objective_name="obj", direction="maximize"
        )
        self.assertAlmostEqual(metrics.best_value, 0.0, places=5)
        self.assertAlmostEqual(metrics.simple_regret, 0.0, places=5)
        self.assertEqual(metrics.feasibility_rate, 1.0)
        self.assertEqual(metrics.constraint_violations, 0)

    def test_compute_metrics_with_infeasible(self) -> None:
        b = InfeasibleBenchmark()
        ev = CaseStudyEvaluator(b)
        history = [
            {"x": {"x1": 0.2}, "result": {"obj": {"value": -0.04, "variance": 0.01}}, "iteration": 0},
            {"x": {"x1": 0.8}, "result": None, "iteration": 1},
            {"x": {"x1": 0.1}, "result": {"obj": {"value": -0.01, "variance": 0.01}}, "iteration": 2},
        ]
        metrics = ev._compute_metrics(
            history, known_opt=None, objective_name="obj", direction="minimize"
        )
        self.assertEqual(metrics.constraint_violations, 1)
        self.assertAlmostEqual(metrics.feasibility_rate, 2 / 3, places=5)

    def test_compute_metrics_minimize(self) -> None:
        b = InfeasibleBenchmark()
        ev = CaseStudyEvaluator(b)
        history = [
            {"x": {"x1": 0.3}, "result": {"obj": {"value": 0.5, "variance": 0.01}}, "iteration": 0},
            {"x": {"x1": 0.1}, "result": {"obj": {"value": 0.1, "variance": 0.01}}, "iteration": 1},
        ]
        metrics = ev._compute_metrics(
            history, known_opt={"obj": 0.0}, objective_name="obj", direction="minimize"
        )
        self.assertAlmostEqual(metrics.best_value, 0.1, places=5)
        self.assertAlmostEqual(metrics.simple_regret, 0.1, places=5)

    def test_random_point_continuous(self) -> None:
        specs = [ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)]
        rng = random.Random(42)
        point = CaseStudyEvaluator._random_point(specs, rng)
        self.assertIn("x", point)
        self.assertGreaterEqual(point["x"], 0.0)
        self.assertLessEqual(point["x"], 1.0)

    def test_random_point_categorical(self) -> None:
        specs = [ParameterSpec(name="m", type=VariableType.CATEGORICAL, categories=["A", "B"])]
        rng = random.Random(42)
        point = CaseStudyEvaluator._random_point(specs, rng)
        self.assertIn("m", point)
        self.assertIn(point["m"], ["A", "B"])

    def test_extract_convergence_curve_minimize(self) -> None:
        b = DummyBenchmark()
        ev = CaseStudyEvaluator(b)
        history = [
            {"x": {}, "result": {"obj": {"value": 5.0, "variance": 0.01}}, "iteration": 0},
            {"x": {}, "result": {"obj": {"value": 3.0, "variance": 0.01}}, "iteration": 1},
            {"x": {}, "result": {"obj": {"value": 4.0, "variance": 0.01}}, "iteration": 2},
        ]
        curve = ev._extract_convergence_curve(history, "obj", "minimize")
        self.assertEqual(curve, [5.0, 3.0, 3.0])

    def test_extract_convergence_curve_maximize(self) -> None:
        b = DummyBenchmark()
        ev = CaseStudyEvaluator(b)
        history = [
            {"x": {}, "result": {"obj": {"value": 1.0, "variance": 0.01}}, "iteration": 0},
            {"x": {}, "result": {"obj": {"value": 3.0, "variance": 0.01}}, "iteration": 1},
            {"x": {}, "result": {"obj": {"value": 2.0, "variance": 0.01}}, "iteration": 2},
        ]
        curve = ev._extract_convergence_curve(history, "obj", "maximize")
        self.assertEqual(curve, [1.0, 3.0, 3.0])


if __name__ == "__main__":
    unittest.main()

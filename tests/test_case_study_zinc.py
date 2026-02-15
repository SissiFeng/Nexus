"""Comprehensive tests for the zinc electrodeposition case study.

Covers:
- ZincDataLoader: data generation, shapes, reproducibility, true function,
  noise model, sum constraint, known optimum, search space.
- ZincBenchmark: creation, search space, objectives, evaluate, feasibility,
  known optimum, evaluation cost, reproducibility, domain config, constraints.
- Zinc annotations: anomaly labels, known mechanisms, annotate_observation.
"""

from __future__ import annotations

import math
import unittest

from optimization_copilot.case_studies.zinc.data_loader import (
    ZincDataLoader,
    _PARAM_NAMES,
    _OPTIMAL_POINT,
)
from optimization_copilot.case_studies.zinc.benchmark import ZincBenchmark
from optimization_copilot.case_studies.zinc.annotations import (
    ZINC_ANOMALY_LABELS,
    ZINC_KNOWN_MECHANISMS,
    get_anomaly_labels,
    get_known_mechanisms,
    annotate_observation,
)


# ===========================================================================
# TestZincDataLoader
# ===========================================================================


class TestZincDataLoader(unittest.TestCase):
    """Tests for ZincDataLoader data generation."""

    def setUp(self) -> None:
        self.loader = ZincDataLoader(n_points=50, seed=42)

    # -- shape tests -------------------------------------------------------

    def test_data_has_correct_keys(self) -> None:
        data = self.loader.get_data()
        self.assertIn("X", data)
        self.assertIn("Y", data)
        self.assertIn("noise_levels", data)

    def test_x_shape(self) -> None:
        data = self.loader.get_data()
        X = data["X"]
        self.assertEqual(len(X), 50)
        for row in X:
            self.assertEqual(len(row), 7)

    def test_y_shape(self) -> None:
        data = self.loader.get_data()
        Y = data["Y"]
        self.assertIn("coulombic_efficiency", Y)
        self.assertEqual(len(Y["coulombic_efficiency"]), 50)

    def test_noise_levels_present(self) -> None:
        data = self.loader.get_data()
        self.assertIn("coulombic_efficiency", data["noise_levels"])
        self.assertGreater(data["noise_levels"]["coulombic_efficiency"], 0.0)

    # -- n_points parameter ------------------------------------------------

    def test_n_points_parameter(self) -> None:
        loader = ZincDataLoader(n_points=20, seed=99)
        data = loader.get_data()
        self.assertEqual(len(data["X"]), 20)
        self.assertEqual(len(data["Y"]["coulombic_efficiency"]), 20)

    def test_n_points_large(self) -> None:
        loader = ZincDataLoader(n_points=200, seed=7)
        data = loader.get_data()
        self.assertEqual(len(data["X"]), 200)

    # -- reproducibility ---------------------------------------------------

    def test_deterministic_same_seed(self) -> None:
        loader_a = ZincDataLoader(n_points=30, seed=123)
        loader_b = ZincDataLoader(n_points=30, seed=123)
        data_a = loader_a.get_data()
        data_b = loader_b.get_data()
        self.assertEqual(data_a["X"], data_b["X"])
        self.assertEqual(
            data_a["Y"]["coulombic_efficiency"],
            data_b["Y"]["coulombic_efficiency"],
        )

    def test_different_seeds_produce_different_data(self) -> None:
        loader_a = ZincDataLoader(n_points=30, seed=1)
        loader_b = ZincDataLoader(n_points=30, seed=2)
        data_a = loader_a.get_data()
        data_b = loader_b.get_data()
        # It is extremely unlikely that two different seeds produce identical X
        self.assertNotEqual(data_a["X"], data_b["X"])

    # -- true function tests -----------------------------------------------

    def test_true_function_at_optimum(self) -> None:
        val = self.loader._true_function(_OPTIMAL_POINT)
        # Should be close to 98.5 (within a few percent due to interactions)
        self.assertGreater(val, 95.0)
        self.assertLessEqual(val, 100.0)

    def test_true_function_returns_sensible_values(self) -> None:
        """All generated points should yield CE in [70, 100]."""
        data = self.loader.get_data()
        for row in data["X"]:
            val = self.loader._true_function(row)
            self.assertGreaterEqual(val, 70.0)
            self.assertLessEqual(val, 100.0)

    def test_true_function_at_zero(self) -> None:
        val = self.loader._true_function([0.0] * 7)
        self.assertGreaterEqual(val, 70.0)
        self.assertLessEqual(val, 100.0)

    def test_true_function_at_corner(self) -> None:
        # One additive maxed out, rest zero
        val = self.loader._true_function([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertGreaterEqual(val, 70.0)
        self.assertLessEqual(val, 100.0)

    # -- noise model tests -------------------------------------------------

    def test_noise_model_returns_positive(self) -> None:
        for row in self.loader.get_data()["X"]:
            noise = self.loader._noise_model(row)
            self.assertGreater(noise, 0.0)

    def test_noise_lower_near_optimum(self) -> None:
        noise_near = self.loader._noise_model(_OPTIMAL_POINT)
        far_point = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        noise_far = self.loader._noise_model(far_point)
        self.assertLess(noise_near, noise_far)

    def test_noise_at_optimum_approximately_half_percent(self) -> None:
        noise = self.loader._noise_model(_OPTIMAL_POINT)
        self.assertAlmostEqual(noise, 0.5, places=1)

    # -- sum constraint tests ----------------------------------------------

    def test_sum_constraint_on_generated_data(self) -> None:
        data = self.loader.get_data()
        for row in data["X"]:
            self.assertLessEqual(sum(row), 1.0 + 1e-9)

    def test_all_values_in_unit_interval(self) -> None:
        data = self.loader.get_data()
        for row in data["X"]:
            for val in row:
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

    # -- known optimum tests -----------------------------------------------

    def test_known_optimum_returned(self) -> None:
        opt = self.loader.get_known_optimum()
        self.assertIn("coulombic_efficiency", opt)
        self.assertAlmostEqual(opt["coulombic_efficiency"], 98.5)

    # -- search space tests ------------------------------------------------

    def test_search_space_has_7_parameters(self) -> None:
        space = self.loader.get_search_space()
        self.assertEqual(len(space), 7)

    def test_search_space_all_continuous(self) -> None:
        space = self.loader.get_search_space()
        for name, spec in space.items():
            self.assertEqual(spec["type"], "continuous")
            self.assertEqual(spec["range"], [0.0, 1.0])

    def test_search_space_param_names(self) -> None:
        space = self.loader.get_search_space()
        for i in range(1, 8):
            self.assertIn(f"additive_{i}", space)


# ===========================================================================
# TestZincBenchmark
# ===========================================================================


class TestZincBenchmark(unittest.TestCase):
    """Tests for ZincBenchmark (ReplayBenchmark subclass)."""

    @classmethod
    def setUpClass(cls) -> None:
        # Create once to avoid repeated GP fitting (expensive)
        cls.bench = ZincBenchmark(n_train=50, seed=42)

    # -- creation tests ----------------------------------------------------

    def test_creation_default_params(self) -> None:
        # Uses more training points, separate instance
        bench = ZincBenchmark(n_train=30, seed=99)
        self.assertIsNotNone(bench)

    def test_creation_custom_params(self) -> None:
        bench = ZincBenchmark(n_train=40, seed=7)
        self.assertIsNotNone(bench)

    # -- search space tests ------------------------------------------------

    def test_get_search_space_returns_7_params(self) -> None:
        space = self.bench.get_search_space()
        self.assertEqual(len(space), 7)

    def test_get_search_space_all_continuous(self) -> None:
        space = self.bench.get_search_space()
        for name, spec in space.items():
            self.assertEqual(spec["type"], "continuous")

    def test_search_space_ranges(self) -> None:
        space = self.bench.get_search_space()
        for name, spec in space.items():
            lo, hi = spec["range"]
            self.assertEqual(lo, 0.0)
            self.assertEqual(hi, 1.0)

    # -- objectives tests --------------------------------------------------

    def test_get_objectives_returns_coulombic_efficiency(self) -> None:
        objs = self.bench.get_objectives()
        self.assertIn("coulombic_efficiency", objs)

    def test_objective_direction_maximize(self) -> None:
        objs = self.bench.get_objectives()
        self.assertEqual(
            objs["coulombic_efficiency"]["direction"], "maximize"
        )

    def test_objective_unit_percent(self) -> None:
        objs = self.bench.get_objectives()
        self.assertEqual(objs["coulombic_efficiency"]["unit"], "%")

    # -- evaluate tests ----------------------------------------------------

    def test_evaluate_returns_dict(self) -> None:
        x = {f"additive_{i}": 0.1 for i in range(1, 8)}
        # sum = 0.7, feasible
        result = self.bench.evaluate(x)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_evaluate_has_correct_keys(self) -> None:
        x = {f"additive_{i}": 0.1 for i in range(1, 8)}
        result = self.bench.evaluate(x)
        self.assertIn("coulombic_efficiency", result)

    def test_evaluate_returns_value_and_variance(self) -> None:
        x = {f"additive_{i}": 0.1 for i in range(1, 8)}
        result = self.bench.evaluate(x)
        ce = result["coulombic_efficiency"]
        self.assertIn("value", ce)
        self.assertIn("variance", ce)

    def test_evaluate_value_is_float(self) -> None:
        x = {f"additive_{i}": 0.05 for i in range(1, 8)}
        result = self.bench.evaluate(x)
        self.assertIsInstance(result["coulombic_efficiency"]["value"], float)

    def test_evaluate_variance_is_positive(self) -> None:
        x = {f"additive_{i}": 0.05 for i in range(1, 8)}
        result = self.bench.evaluate(x)
        self.assertGreater(result["coulombic_efficiency"]["variance"], 0.0)

    # -- feasibility tests -------------------------------------------------

    def test_is_feasible_valid_point(self) -> None:
        x = {f"additive_{i}": 0.1 for i in range(1, 8)}
        # sum = 0.7, feasible
        self.assertTrue(self.bench.is_feasible(x))

    def test_is_feasible_boundary(self) -> None:
        # Exactly sum = 1.0 should be feasible
        x = {"additive_1": 0.5, "additive_2": 0.5}
        for i in range(3, 8):
            x[f"additive_{i}"] = 0.0
        self.assertTrue(self.bench.is_feasible(x))

    def test_is_feasible_returns_false_when_sum_exceeds_one(self) -> None:
        x = {f"additive_{i}": 0.2 for i in range(1, 8)}
        # sum = 1.4, infeasible
        self.assertFalse(self.bench.is_feasible(x))

    def test_infeasible_point_returns_none(self) -> None:
        x = {f"additive_{i}": 0.2 for i in range(1, 8)}
        result = self.bench.evaluate(x)
        self.assertIsNone(result)

    # -- known optimum tests -----------------------------------------------

    def test_get_known_optimum_value(self) -> None:
        opt = self.bench.get_known_optimum()
        self.assertIn("coulombic_efficiency", opt)
        self.assertAlmostEqual(opt["coulombic_efficiency"], 98.5)

    # -- evaluation cost tests ---------------------------------------------

    def test_get_evaluation_cost_positive(self) -> None:
        x = {f"additive_{i}": 0.05 for i in range(1, 8)}
        cost = self.bench.get_evaluation_cost(x)
        self.assertGreater(cost, 0.0)

    def test_evaluation_cost_at_zero(self) -> None:
        x = {f"additive_{i}": 0.0 for i in range(1, 8)}
        cost = self.bench.get_evaluation_cost(x)
        self.assertAlmostEqual(cost, 1.0)

    def test_evaluation_cost_scales_with_loading(self) -> None:
        x_low = {f"additive_{i}": 0.01 for i in range(1, 8)}
        x_high = {f"additive_{i}": 0.1 for i in range(1, 8)}
        self.assertLess(
            self.bench.get_evaluation_cost(x_low),
            self.bench.get_evaluation_cost(x_high),
        )

    # -- reproducibility tests ---------------------------------------------

    def test_reproducibility_same_seed(self) -> None:
        bench_a = ZincBenchmark(n_train=30, seed=77)
        bench_b = ZincBenchmark(n_train=30, seed=77)
        x = {f"additive_{i}": 0.05 for i in range(1, 8)}
        # After construction both use the same internal RNG state
        result_a = bench_a.evaluate(x)
        result_b = bench_b.evaluate(x)
        self.assertAlmostEqual(
            result_a["coulombic_efficiency"]["value"],
            result_b["coulombic_efficiency"]["value"],
        )

    # -- domain config tests -----------------------------------------------

    def test_domain_config_is_electrochemistry(self) -> None:
        cfg = self.bench.get_domain_config()
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.domain_name, "electrochemistry")

    # -- constraints tests -------------------------------------------------

    def test_get_known_constraints_returns_list(self) -> None:
        constraints = self.bench.get_known_constraints()
        self.assertIsInstance(constraints, list)

    def test_known_constraints_contains_sum(self) -> None:
        constraints = self.bench.get_known_constraints()
        sum_constraints = [
            c for c in constraints if c.get("type") == "sum_constraint"
        ]
        self.assertGreaterEqual(len(sum_constraints), 1)

    # -- multiple evaluations test -----------------------------------------

    def test_multiple_evaluations_different_points(self) -> None:
        results = []
        for i in range(5):
            x = {f"additive_{j}": 0.02 * (i + 1) for j in range(1, 8)}
            if self.bench.is_feasible(x):
                result = self.bench.evaluate(x)
                if result is not None:
                    results.append(result["coulombic_efficiency"]["value"])
        # We should have at least some feasible results
        self.assertGreater(len(results), 0)

    # -- encoded dimension test --------------------------------------------

    def test_encoded_dimension_matches_7(self) -> None:
        x = {f"additive_{i}": 0.1 for i in range(1, 8)}
        encoded = self.bench._encode(x)
        self.assertEqual(len(encoded), 7)


# ===========================================================================
# TestZincAnnotations
# ===========================================================================


class TestZincAnnotations(unittest.TestCase):
    """Tests for zinc annotations module."""

    # -- anomaly labels tests ----------------------------------------------

    def test_get_anomaly_labels_returns_list(self) -> None:
        labels = get_anomaly_labels()
        self.assertIsInstance(labels, list)
        self.assertGreater(len(labels), 0)

    def test_anomaly_labels_have_required_fields(self) -> None:
        for label in get_anomaly_labels():
            self.assertIn("type", label)
            self.assertIn("description", label)

    def test_anomaly_labels_is_copy(self) -> None:
        labels_a = get_anomaly_labels()
        labels_b = get_anomaly_labels()
        labels_a.append({"type": "test"})
        self.assertNotEqual(len(labels_a), len(labels_b))

    # -- known mechanisms tests --------------------------------------------

    def test_get_known_mechanisms_returns_dict(self) -> None:
        mechs = get_known_mechanisms()
        self.assertIsInstance(mechs, dict)

    def test_known_mechanisms_has_expected_keys(self) -> None:
        mechs = get_known_mechanisms()
        for key in ("leveling", "brightening", "throwing_power"):
            self.assertIn(key, mechs)

    def test_mechanism_entries_have_required_fields(self) -> None:
        mechs = get_known_mechanisms()
        for name, info in mechs.items():
            self.assertIn("primary_additive", info)
            self.assertIn("mechanism", info)

    def test_known_mechanisms_is_copy(self) -> None:
        mechs_a = get_known_mechanisms()
        mechs_b = get_known_mechanisms()
        mechs_a["test"] = {}
        self.assertNotIn("test", mechs_b)

    # -- annotate_observation tests ----------------------------------------

    def test_annotate_normal_point(self) -> None:
        x = {f"additive_{i}": 0.1 for i in range(1, 8)}
        result = {"coulombic_efficiency": {"value": 95.0, "variance": 0.5}}
        ann = annotate_observation(x, result)
        self.assertIn("anomaly_flags", ann)
        self.assertIn("mechanism_flags", ann)
        self.assertIn("failed", ann)
        self.assertFalse(ann["failed"])

    def test_annotate_normal_point_no_anomalies(self) -> None:
        x = {f"additive_{i}": 0.1 for i in range(1, 8)}
        result = {"coulombic_efficiency": {"value": 95.0, "variance": 0.5}}
        ann = annotate_observation(x, result)
        # Normal point should have no anomaly flags
        self.assertEqual(len(ann["anomaly_flags"]), 0)

    def test_annotate_high_additive_1_passivation(self) -> None:
        x = {"additive_1": 0.9}
        for i in range(2, 8):
            x[f"additive_{i}"] = 0.0
        result = {"coulombic_efficiency": {"value": 80.0, "variance": 2.0}}
        ann = annotate_observation(x, result)
        anomaly_types = [f["type"] for f in ann["anomaly_flags"]]
        self.assertIn("passivation", anomaly_types)

    def test_annotate_high_total_loading_dendrites(self) -> None:
        # sum > 0.95 triggers dendrite_formation
        x = {
            "additive_1": 0.2,
            "additive_2": 0.2,
            "additive_3": 0.15,
            "additive_4": 0.15,
            "additive_5": 0.1,
            "additive_6": 0.1,
            "additive_7": 0.06,
        }
        # sum = 0.96 > 0.95
        result = {"coulombic_efficiency": {"value": 82.0, "variance": 3.0}}
        ann = annotate_observation(x, result)
        anomaly_types = [f["type"] for f in ann["anomaly_flags"]]
        self.assertIn("dendrite_formation", anomaly_types)

    def test_annotate_failed_experiment(self) -> None:
        x = {f"additive_{i}": 0.1 for i in range(1, 8)}
        ann = annotate_observation(x, None)
        self.assertTrue(ann["failed"])

    def test_annotate_returns_flags_dict(self) -> None:
        x = {f"additive_{i}": 0.05 for i in range(1, 8)}
        result = {"coulombic_efficiency": {"value": 93.0, "variance": 1.0}}
        ann = annotate_observation(x, result)
        self.assertIsInstance(ann, dict)
        self.assertIsInstance(ann["anomaly_flags"], list)
        self.assertIsInstance(ann["mechanism_flags"], list)

    def test_annotate_mechanism_flags_present(self) -> None:
        # With all additives at 0.05, all mechanisms should be active
        x = {f"additive_{i}": 0.05 for i in range(1, 8)}
        result = {"coulombic_efficiency": {"value": 93.0, "variance": 1.0}}
        ann = annotate_observation(x, result)
        # At least leveling, brightening, throwing_power should be active
        self.assertGreater(len(ann["mechanism_flags"]), 0)
        self.assertIn("leveling", ann["mechanism_flags"])

    def test_annotate_no_mechanism_when_additives_zero(self) -> None:
        x = {f"additive_{i}": 0.0 for i in range(1, 8)}
        result = {"coulombic_efficiency": {"value": 90.0, "variance": 1.0}}
        ann = annotate_observation(x, result)
        self.assertEqual(len(ann["mechanism_flags"]), 0)


if __name__ == "__main__":
    unittest.main()

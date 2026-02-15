"""Tests for perovskite thin film composition case study.

Covers:
- PerovskiteDataLoader data generation and validation
- PerovskiteBenchmark multi-objective offline replay
- Integration between data loader and benchmark
"""

from __future__ import annotations

import unittest


# ---------------------------------------------------------------------------
# TestPerovskiteDataLoader
# ---------------------------------------------------------------------------


class TestPerovskiteDataLoader(unittest.TestCase):
    """Tests for PerovskiteDataLoader synthetic data generation."""

    @classmethod
    def setUpClass(cls) -> None:
        from optimization_copilot.case_studies.perovskite.data_loader import (
            PerovskiteDataLoader,
        )

        cls.loader = PerovskiteDataLoader(n_points=120, seed=42)
        cls.data = cls.loader.get_data()

    # -- shape tests -------------------------------------------------------

    def test_data_has_required_keys(self) -> None:
        for key in ("X", "Y", "noise_levels"):
            self.assertIn(key, self.data)

    def test_x_shape_correct(self) -> None:
        X = self.data["X"]
        self.assertEqual(len(X), 120)
        for row in X:
            self.assertEqual(len(row), 8)

    def test_y_has_both_objectives(self) -> None:
        Y = self.data["Y"]
        self.assertIn("PCE", Y)
        self.assertIn("stability", Y)

    def test_y_lengths_match_x(self) -> None:
        n = len(self.data["X"])
        self.assertEqual(len(self.data["Y"]["PCE"]), n)
        self.assertEqual(len(self.data["Y"]["stability"]), n)

    def test_noise_levels_has_both_objectives(self) -> None:
        nl = self.data["noise_levels"]
        self.assertIn("PCE", nl)
        self.assertIn("stability", nl)

    # -- n_points parameter ------------------------------------------------

    def test_n_points_parameter(self) -> None:
        from optimization_copilot.case_studies.perovskite.data_loader import (
            PerovskiteDataLoader,
        )

        loader = PerovskiteDataLoader(n_points=30, seed=99)
        data = loader.get_data()
        self.assertEqual(len(data["X"]), 30)
        self.assertEqual(len(data["Y"]["PCE"]), 30)
        self.assertEqual(len(data["Y"]["stability"]), 30)

    # -- reproducibility ---------------------------------------------------

    def test_deterministic_same_seed(self) -> None:
        from optimization_copilot.case_studies.perovskite.data_loader import (
            PerovskiteDataLoader,
        )

        loader_a = PerovskiteDataLoader(n_points=50, seed=123)
        loader_b = PerovskiteDataLoader(n_points=50, seed=123)
        self.assertEqual(loader_a.get_data()["X"], loader_b.get_data()["X"])
        self.assertEqual(
            loader_a.get_data()["Y"]["PCE"],
            loader_b.get_data()["Y"]["PCE"],
        )
        self.assertEqual(
            loader_a.get_data()["Y"]["stability"],
            loader_b.get_data()["Y"]["stability"],
        )

    def test_different_seeds_different_data(self) -> None:
        from optimization_copilot.case_studies.perovskite.data_loader import (
            PerovskiteDataLoader,
        )

        loader_a = PerovskiteDataLoader(n_points=50, seed=1)
        loader_b = PerovskiteDataLoader(n_points=50, seed=2)
        self.assertNotEqual(
            loader_a.get_data()["X"], loader_b.get_data()["X"]
        )

    # -- simplex constraints -----------------------------------------------

    def test_cation_fractions_sum_to_one(self) -> None:
        """FA + MA + Cs should sum to ~1.0 for every data point."""
        for row in self.data["X"]:
            cation_sum = row[0] + row[1] + row[2]
            self.assertAlmostEqual(cation_sum, 1.0, places=6,
                                   msg=f"Cation sum {cation_sum} != 1.0")

    def test_halide_fractions_sum_to_one(self) -> None:
        """I + Br should sum to ~1.0 for every data point."""
        for row in self.data["X"]:
            halide_sum = row[3] + row[4]
            self.assertAlmostEqual(halide_sum, 1.0, places=6,
                                   msg=f"Halide sum {halide_sum} != 1.0")

    def test_cation_fractions_non_negative(self) -> None:
        for row in self.data["X"]:
            for i in range(3):
                self.assertGreaterEqual(row[i], 0.0)

    def test_halide_fractions_non_negative(self) -> None:
        for row in self.data["X"]:
            for i in (3, 4):
                self.assertGreaterEqual(row[i], 0.0)

    # -- physical range checks ---------------------------------------------

    def test_pce_values_in_physical_range(self) -> None:
        for val in self.data["Y"]["PCE"]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 33.0)

    def test_stability_values_non_negative(self) -> None:
        for val in self.data["Y"]["stability"]:
            self.assertGreaterEqual(val, 0.0)

    def test_stability_values_capped(self) -> None:
        for val in self.data["Y"]["stability"]:
            self.assertLessEqual(val, 1000.0)

    # -- known optimum -----------------------------------------------------

    def test_known_optimum_returned(self) -> None:
        opt = self.loader.get_known_optimum()
        self.assertIn("PCE", opt)
        self.assertIn("stability", opt)
        self.assertEqual(opt["PCE"], 23.0)
        self.assertEqual(opt["stability"], 800.0)

    # -- search space ------------------------------------------------------

    def test_search_space_has_8_parameters(self) -> None:
        space = self.loader.get_search_space()
        self.assertEqual(len(space), 8)

    def test_search_space_parameter_names(self) -> None:
        space = self.loader.get_search_space()
        expected = {"FA", "MA", "Cs", "I", "Br",
                    "annealing_temp", "spin_speed", "precursor_conc"}
        self.assertEqual(set(space.keys()), expected)

    def test_search_space_all_continuous(self) -> None:
        space = self.loader.get_search_space()
        for name, spec in space.items():
            self.assertEqual(spec["type"], "continuous",
                             msg=f"{name} should be continuous")

    # -- simplex sampling --------------------------------------------------

    def test_simplex_sampling_sums_to_one(self) -> None:
        from optimization_copilot.case_studies.perovskite.data_loader import (
            PerovskiteDataLoader,
        )

        loader = PerovskiteDataLoader(n_points=10, seed=77)
        for n in (2, 3, 5):
            fracs = loader._sample_simplex(n)
            self.assertEqual(len(fracs), n)
            self.assertAlmostEqual(sum(fracs), 1.0, places=10)
            for f in fracs:
                self.assertGreaterEqual(f, 0.0)
                self.assertLessEqual(f, 1.0)


# ---------------------------------------------------------------------------
# TestPerovskiteBenchmark
# ---------------------------------------------------------------------------


class TestPerovskiteBenchmark(unittest.TestCase):
    """Tests for PerovskiteBenchmark multi-objective replay."""

    @classmethod
    def setUpClass(cls) -> None:
        from optimization_copilot.case_studies.perovskite.composition_benchmark import (
            PerovskiteBenchmark,
        )

        cls.bench = PerovskiteBenchmark(n_train=40, seed=42)

    # -- creation ----------------------------------------------------------

    def test_creation_with_defaults(self) -> None:
        from optimization_copilot.case_studies.perovskite.composition_benchmark import (
            PerovskiteBenchmark,
        )

        bench = PerovskiteBenchmark()
        self.assertIsNotNone(bench)

    # -- search space ------------------------------------------------------

    def test_get_search_space_returns_8_params(self) -> None:
        space = self.bench.get_search_space()
        self.assertEqual(len(space), 8)

    def test_search_space_all_continuous(self) -> None:
        space = self.bench.get_search_space()
        for name, spec in space.items():
            self.assertEqual(spec["type"], "continuous",
                             msg=f"{name} should be continuous")

    # -- objectives --------------------------------------------------------

    def test_get_objectives_returns_pce_and_stability(self) -> None:
        objs = self.bench.get_objectives()
        self.assertIn("PCE", objs)
        self.assertIn("stability", objs)
        self.assertEqual(len(objs), 2)

    def test_objectives_maximize(self) -> None:
        objs = self.bench.get_objectives()
        self.assertEqual(objs["PCE"]["direction"], "maximize")
        self.assertEqual(objs["stability"]["direction"], "maximize")

    def test_objectives_units(self) -> None:
        objs = self.bench.get_objectives()
        self.assertEqual(objs["PCE"]["unit"], "%")
        self.assertEqual(objs["stability"]["unit"], "hours")

    # -- evaluate ----------------------------------------------------------

    def test_evaluate_returns_dict_with_both_objectives(self) -> None:
        x = {
            "FA": 0.80, "MA": 0.05, "Cs": 0.15,
            "I": 0.85, "Br": 0.15,
            "annealing_temp": 130.0,
            "spin_speed": 4000.0,
            "precursor_conc": 1.2,
        }
        result = self.bench.evaluate(x)
        self.assertIsNotNone(result)
        self.assertIn("PCE", result)
        self.assertIn("stability", result)

    def test_evaluate_returns_value_and_variance(self) -> None:
        x = {
            "FA": 0.80, "MA": 0.05, "Cs": 0.15,
            "I": 0.85, "Br": 0.15,
            "annealing_temp": 130.0,
            "spin_speed": 4000.0,
            "precursor_conc": 1.2,
        }
        result = self.bench.evaluate(x)
        self.assertIsNotNone(result)
        for obj_name in ("PCE", "stability"):
            self.assertIn("value", result[obj_name])
            self.assertIn("variance", result[obj_name])
            self.assertIsInstance(result[obj_name]["value"], float)
            self.assertIsInstance(result[obj_name]["variance"], float)

    # -- feasibility -------------------------------------------------------

    def test_is_feasible_valid_composition(self) -> None:
        x = {
            "FA": 0.80, "MA": 0.05, "Cs": 0.15,
            "I": 0.85, "Br": 0.15,
            "annealing_temp": 130.0,
            "spin_speed": 4000.0,
            "precursor_conc": 1.2,
        }
        self.assertTrue(self.bench.is_feasible(x))

    def test_is_feasible_false_cation_sum_wrong(self) -> None:
        x = {
            "FA": 0.50, "MA": 0.50, "Cs": 0.50,  # sum = 1.50
            "I": 0.85, "Br": 0.15,
            "annealing_temp": 130.0,
            "spin_speed": 4000.0,
            "precursor_conc": 1.2,
        }
        self.assertFalse(self.bench.is_feasible(x))

    def test_is_feasible_false_halide_sum_wrong(self) -> None:
        x = {
            "FA": 0.80, "MA": 0.05, "Cs": 0.15,
            "I": 0.50, "Br": 0.10,  # sum = 0.60
            "annealing_temp": 130.0,
            "spin_speed": 4000.0,
            "precursor_conc": 1.2,
        }
        self.assertFalse(self.bench.is_feasible(x))

    def test_is_feasible_false_fa_too_high(self) -> None:
        """FA > 0.85 triggers delta-phase formation."""
        x = {
            "FA": 0.90, "MA": 0.02, "Cs": 0.08,  # FA > 0.85
            "I": 0.85, "Br": 0.15,
            "annealing_temp": 130.0,
            "spin_speed": 4000.0,
            "precursor_conc": 1.2,
        }
        self.assertFalse(self.bench.is_feasible(x))

    def test_is_feasible_false_poor_phase_stability(self) -> None:
        """Cs < 0.05 and Br < 0.1 triggers poor phase stability."""
        x = {
            "FA": 0.92, "MA": 0.05, "Cs": 0.03,  # Cs < 0.05
            "I": 0.95, "Br": 0.05,                # Br < 0.1
            "annealing_temp": 130.0,
            "spin_speed": 4000.0,
            "precursor_conc": 1.2,
        }
        self.assertFalse(self.bench.is_feasible(x))

    def test_infeasible_returns_none(self) -> None:
        x = {
            "FA": 0.50, "MA": 0.50, "Cs": 0.50,  # invalid cation sum
            "I": 0.85, "Br": 0.15,
            "annealing_temp": 130.0,
            "spin_speed": 4000.0,
            "precursor_conc": 1.2,
        }
        result = self.bench.evaluate(x)
        self.assertIsNone(result)

    # -- known optimum -----------------------------------------------------

    def test_get_known_optimum_has_both_objectives(self) -> None:
        opt = self.bench.get_known_optimum()
        self.assertIsNotNone(opt)
        self.assertIn("PCE", opt)
        self.assertIn("stability", opt)
        self.assertEqual(opt["PCE"], 23.0)
        self.assertEqual(opt["stability"], 800.0)

    # -- evaluation cost ---------------------------------------------------

    def test_get_evaluation_cost_positive(self) -> None:
        x = {
            "FA": 0.80, "MA": 0.05, "Cs": 0.15,
            "I": 0.85, "Br": 0.15,
            "annealing_temp": 130.0,
            "spin_speed": 4000.0,
            "precursor_conc": 1.2,
        }
        cost = self.bench.get_evaluation_cost(x)
        self.assertGreater(cost, 0.0)

    def test_get_evaluation_cost_increases_with_temp(self) -> None:
        x_low = {
            "FA": 0.80, "MA": 0.05, "Cs": 0.15,
            "I": 0.85, "Br": 0.15,
            "annealing_temp": 100.0,
            "spin_speed": 4000.0,
            "precursor_conc": 1.2,
        }
        x_high = {
            "FA": 0.80, "MA": 0.05, "Cs": 0.15,
            "I": 0.85, "Br": 0.15,
            "annealing_temp": 190.0,
            "spin_speed": 4000.0,
            "precursor_conc": 1.2,
        }
        self.assertGreater(
            self.bench.get_evaluation_cost(x_high),
            self.bench.get_evaluation_cost(x_low),
        )

    # -- reproducibility ---------------------------------------------------

    def test_reproducibility_same_seed(self) -> None:
        from optimization_copilot.case_studies.perovskite.composition_benchmark import (
            PerovskiteBenchmark,
        )

        bench_a = PerovskiteBenchmark(n_train=30, seed=77)
        bench_b = PerovskiteBenchmark(n_train=30, seed=77)
        x = {
            "FA": 0.70, "MA": 0.10, "Cs": 0.20,
            "I": 0.80, "Br": 0.20,
            "annealing_temp": 140.0,
            "spin_speed": 3500.0,
            "precursor_conc": 1.0,
        }
        result_a = bench_a.evaluate(x)
        result_b = bench_b.evaluate(x)
        self.assertIsNotNone(result_a)
        self.assertIsNotNone(result_b)
        self.assertAlmostEqual(
            result_a["PCE"]["value"],
            result_b["PCE"]["value"],
            places=6,
        )
        self.assertAlmostEqual(
            result_a["stability"]["value"],
            result_b["stability"]["value"],
            places=6,
        )

    # -- domain config -----------------------------------------------------

    def test_domain_config_is_perovskite(self) -> None:
        dc = self.bench.get_domain_config()
        self.assertIsNotNone(dc)
        self.assertEqual(dc.domain_name, "perovskite")

    # -- multi-objective nature --------------------------------------------

    def test_two_surrogates_fitted(self) -> None:
        self.assertEqual(len(self.bench._surrogates), 2)
        self.assertIn("PCE", self.bench._surrogates)
        self.assertIn("stability", self.bench._surrogates)

    def test_surrogates_are_fitted(self) -> None:
        for name, surrogate in self.bench._surrogates.items():
            self.assertTrue(
                surrogate._fitted,
                msg=f"Surrogate for {name} not fitted",
            )


# ---------------------------------------------------------------------------
# TestPerovskiteIntegration
# ---------------------------------------------------------------------------


class TestPerovskiteIntegration(unittest.TestCase):
    """Integration tests between data loader and benchmark."""

    def test_loader_and_benchmark_consistency(self) -> None:
        """Data loader and benchmark should produce consistent data shapes."""
        from optimization_copilot.case_studies.perovskite.data_loader import (
            PerovskiteDataLoader,
        )
        from optimization_copilot.case_studies.perovskite.composition_benchmark import (
            PerovskiteBenchmark,
        )

        loader = PerovskiteDataLoader(n_points=30, seed=42)
        bench = PerovskiteBenchmark(n_train=30, seed=42)

        loader_data = loader.get_data()
        # Benchmark should have the same number of training points
        self.assertEqual(len(bench._X_train), 30)
        # Benchmark training X should match loader X
        self.assertEqual(bench._X_train, loader_data["X"])

    def test_search_space_parameter_names_match(self) -> None:
        """Parameter names should match between loader and benchmark."""
        from optimization_copilot.case_studies.perovskite.data_loader import (
            PerovskiteDataLoader,
        )
        from optimization_copilot.case_studies.perovskite.composition_benchmark import (
            PerovskiteBenchmark,
        )

        loader = PerovskiteDataLoader(n_points=10, seed=42)
        bench = PerovskiteBenchmark(n_train=10, seed=42)

        loader_space = loader.get_search_space()
        bench_space = bench.get_search_space()
        self.assertEqual(set(loader_space.keys()), set(bench_space.keys()))

    def test_known_optima_match(self) -> None:
        """Known optima should be the same from loader and benchmark."""
        from optimization_copilot.case_studies.perovskite.data_loader import (
            PerovskiteDataLoader,
        )
        from optimization_copilot.case_studies.perovskite.composition_benchmark import (
            PerovskiteBenchmark,
        )

        loader = PerovskiteDataLoader(n_points=10, seed=42)
        bench = PerovskiteBenchmark(n_train=10, seed=42)

        self.assertEqual(loader.get_known_optimum(), bench.get_known_optimum())

    def test_benchmark_evaluate_multiple_points(self) -> None:
        """Benchmark should handle multiple evaluations without error."""
        from optimization_copilot.case_studies.perovskite.composition_benchmark import (
            PerovskiteBenchmark,
        )

        bench = PerovskiteBenchmark(n_train=30, seed=42)
        points = [
            {"FA": 0.80, "MA": 0.05, "Cs": 0.15,
             "I": 0.85, "Br": 0.15,
             "annealing_temp": 130.0, "spin_speed": 4000.0,
             "precursor_conc": 1.2},
            {"FA": 0.70, "MA": 0.10, "Cs": 0.20,
             "I": 0.80, "Br": 0.20,
             "annealing_temp": 150.0, "spin_speed": 3000.0,
             "precursor_conc": 1.0},
            {"FA": 0.60, "MA": 0.15, "Cs": 0.25,
             "I": 0.75, "Br": 0.25,
             "annealing_temp": 120.0, "spin_speed": 5000.0,
             "precursor_conc": 1.5},
        ]
        for x in points:
            result = bench.evaluate(x)
            self.assertIsNotNone(result)
            self.assertIn("PCE", result)
            self.assertIn("stability", result)

    def test_data_dimensions_match_search_space(self) -> None:
        """Number of columns in X should match search space dimensionality."""
        from optimization_copilot.case_studies.perovskite.data_loader import (
            PerovskiteDataLoader,
        )

        loader = PerovskiteDataLoader(n_points=10, seed=42)
        data = loader.get_data()
        space = loader.get_search_space()
        n_params = len(space)
        for row in data["X"]:
            self.assertEqual(len(row), n_params)


if __name__ == "__main__":
    unittest.main()

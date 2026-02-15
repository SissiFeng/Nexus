"""Tests for the Suzuki-Miyaura catalysis case study.

Covers:
- CatalysisDataLoader: data generation, encoding, true function, search space
- SuzukiBenchmark: evaluation, feasibility, objectives, cost model
- Integration: consistency between data loader and benchmark
"""

from __future__ import annotations

import math
import unittest
from typing import Any

from optimization_copilot.case_studies.catalysis.data_loader import (
    CatalysisDataLoader,
    SOLVENTS,
)
from optimization_copilot.case_studies.catalysis.suzuki_benchmark import (
    SuzukiBenchmark,
)
from optimization_copilot.domain_knowledge.catalysis import (
    CATALYSTS,
    LIGANDS,
    BASES,
    KNOWN_INCOMPATIBILITIES,
)


# ===========================================================================
# CatalysisDataLoader Tests
# ===========================================================================


class TestCatalysisDataLoader(unittest.TestCase):
    """Tests for CatalysisDataLoader."""

    def test_data_generation_shapes(self) -> None:
        """Generated data should have correct shapes."""
        loader = CatalysisDataLoader(n_points=50, seed=42)
        data = loader.get_data()
        self.assertIn("X", data)
        self.assertIn("Y", data)
        self.assertIn("noise_levels", data)
        self.assertEqual(len(data["X"]), 50)
        self.assertEqual(len(data["Y"]["yield"]), 50)

    def test_n_points_parameter(self) -> None:
        """Different n_points should generate different amounts of data."""
        loader30 = CatalysisDataLoader(n_points=30, seed=42)
        loader80 = CatalysisDataLoader(n_points=80, seed=42)
        self.assertEqual(len(loader30.get_data()["X"]), 30)
        self.assertEqual(len(loader80.get_data()["X"]), 80)

    def test_deterministic_same_seed(self) -> None:
        """Same seed should produce identical data."""
        loader1 = CatalysisDataLoader(n_points=20, seed=42)
        loader2 = CatalysisDataLoader(n_points=20, seed=42)
        data1 = loader1.get_data()
        data2 = loader2.get_data()
        for i in range(20):
            self.assertEqual(data1["X"][i], data2["X"][i])
            self.assertEqual(data1["Y"]["yield"][i], data2["Y"]["yield"][i])

    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds should produce different data."""
        loader1 = CatalysisDataLoader(n_points=50, seed=42)
        loader2 = CatalysisDataLoader(n_points=50, seed=99)
        data1 = loader1.get_data()
        data2 = loader2.get_data()
        # At least some X values should differ
        any_differ = any(
            data1["X"][i] != data2["X"][i] for i in range(50)
        )
        self.assertTrue(any_differ)

    def test_true_function_compatible_pair_positive_yield(self) -> None:
        """Compatible catalyst-ligand pairs should give positive yield."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        # Pd(PPh3)4 + XPhos is compatible and high-performing
        y = loader._true_function(
            "Pd(PPh3)4", "XPhos", "Cs2CO3", "THF", 80.0, 12.0
        )
        self.assertGreater(y, 0.0)

    def test_true_function_incompatible_pair_zero(self) -> None:
        """Incompatible pairs should yield 0 %."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        # Pd(OAc)2 + BINAP is incompatible
        y = loader._true_function(
            "Pd(OAc)2", "BINAP", "K2CO3", "THF", 80.0, 12.0
        )
        self.assertAlmostEqual(y, 0.0)

    def test_true_function_incompatible_pair_pdcl2_pcy3(self) -> None:
        """PdCl2 + PCy3 is also incompatible and should yield 0 %."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        y = loader._true_function(
            "PdCl2", "PCy3", "K2CO3", "DMF", 80.0, 12.0
        )
        self.assertAlmostEqual(y, 0.0)

    def test_temperature_optimum_effect(self) -> None:
        """Yield should be highest near 80 deg-C and lower at extremes."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        y_80 = loader._true_function(
            "Pd(PPh3)4", "XPhos", "Cs2CO3", "THF", 80.0, 12.0
        )
        y_40 = loader._true_function(
            "Pd(PPh3)4", "XPhos", "Cs2CO3", "THF", 40.0, 12.0
        )
        y_120 = loader._true_function(
            "Pd(PPh3)4", "XPhos", "Cs2CO3", "THF", 120.0, 12.0
        )
        self.assertGreater(y_80, y_40)
        self.assertGreater(y_80, y_120)

    def test_time_diminishing_returns(self) -> None:
        """Longer time should give higher yield, with diminishing returns."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        y_1h = loader._true_function(
            "Pd(PPh3)4", "XPhos", "Cs2CO3", "THF", 80.0, 1.0
        )
        y_12h = loader._true_function(
            "Pd(PPh3)4", "XPhos", "Cs2CO3", "THF", 80.0, 12.0
        )
        y_24h = loader._true_function(
            "Pd(PPh3)4", "XPhos", "Cs2CO3", "THF", 80.0, 24.0
        )
        self.assertGreater(y_12h, y_1h)
        # Benefit from 12h to 24h should be smaller than 1h to 12h
        gain_1_12 = y_12h - y_1h
        gain_12_24 = y_24h - y_12h
        self.assertGreater(gain_1_12, gain_12_24)

    def test_search_space_has_six_parameters(self) -> None:
        """Search space should have 6 parameters: 4 categorical + 2 continuous."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        space = loader.get_search_space()
        self.assertEqual(len(space), 6)
        cat_params = [k for k, v in space.items() if v["type"] == "categorical"]
        cont_params = [k for k, v in space.items() if v["type"] == "continuous"]
        self.assertEqual(len(cat_params), 4)
        self.assertEqual(len(cont_params), 2)

    def test_search_space_parameter_names(self) -> None:
        """Search space should contain the correct parameter names."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        space = loader.get_search_space()
        expected = {"catalyst", "ligand", "base", "solvent", "temperature", "time"}
        self.assertEqual(set(space.keys()), expected)

    def test_known_optimum_around_95(self) -> None:
        """Known optimum should be around 95 %."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        opt = loader.get_known_optimum()
        self.assertIn("yield", opt)
        self.assertAlmostEqual(opt["yield"], 95.0, delta=1.0)

    def test_encoding_dimension(self) -> None:
        """Encoded point dimension should be 4+6+5+4+1+1 = 21."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        encoded = loader._encode_point(
            "Pd(OAc)2", "PPh3", "K2CO3", "THF", 80.0, 12.0
        )
        self.assertEqual(len(encoded), 21)

    def test_encoding_one_hot_correct(self) -> None:
        """One-hot encoding should have exactly one 1.0 per categorical group."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        encoded = loader._encode_point(
            "Pd(PPh3)4", "SPhos", "Cs2CO3", "DMF", 80.0, 12.0
        )
        # catalyst: positions 0-3 (4 cats), should be [0,1,0,0]
        self.assertEqual(sum(encoded[0:4]), 1.0)
        self.assertEqual(encoded[1], 1.0)  # Pd(PPh3)4 is index 1

        # ligand: positions 4-9 (6 ligs), should have 1.0 at SPhos (index 2)
        self.assertEqual(sum(encoded[4:10]), 1.0)
        self.assertEqual(encoded[4 + 2], 1.0)  # SPhos is index 2

        # base: positions 10-14 (5 bases), should have 1.0 at Cs2CO3 (index 1)
        self.assertEqual(sum(encoded[10:15]), 1.0)
        self.assertEqual(encoded[10 + 1], 1.0)  # Cs2CO3 is index 1

        # solvent: positions 15-18 (4 solvents), should have 1.0 at DMF (index 1)
        self.assertEqual(sum(encoded[15:19]), 1.0)
        self.assertEqual(encoded[15 + 1], 1.0)  # DMF is index 1

        # continuous: positions 19, 20
        self.assertAlmostEqual(encoded[19], 80.0)
        self.assertAlmostEqual(encoded[20], 12.0)

    def test_yield_values_in_range(self) -> None:
        """All generated yield values should be in [0, 100]."""
        loader = CatalysisDataLoader(n_points=100, seed=42)
        data = loader.get_data()
        for y in data["Y"]["yield"]:
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 100.0)

    def test_noise_level_positive(self) -> None:
        """Noise level should be a positive number."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        data = loader.get_data()
        self.assertGreater(data["noise_levels"]["yield"], 0.0)

    def test_solvent_effect(self) -> None:
        """THF should give higher yield than toluene, all else equal."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        y_thf = loader._true_function(
            "Pd(PPh3)4", "XPhos", "K2CO3", "THF", 80.0, 12.0
        )
        y_tol = loader._true_function(
            "Pd(PPh3)4", "XPhos", "K2CO3", "toluene", 80.0, 12.0
        )
        self.assertGreater(y_thf, y_tol)

    def test_base_effect(self) -> None:
        """Cs2CO3 should give higher yield than DBU, all else equal."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        y_cs = loader._true_function(
            "Pd(PPh3)4", "XPhos", "Cs2CO3", "THF", 80.0, 12.0
        )
        y_dbu = loader._true_function(
            "Pd(PPh3)4", "XPhos", "DBU", "THF", 80.0, 12.0
        )
        self.assertGreater(y_cs, y_dbu)

    def test_best_combination_near_optimum(self) -> None:
        """The best combination should achieve yield near 95 %."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        # Pd2(dba)3 + SPhos + Cs2CO3 + THF + 80C + 12h should be near optimum
        y = loader._true_function(
            "Pd2(dba)3", "SPhos", "Cs2CO3", "THF", 80.0, 12.0
        )
        self.assertGreater(y, 90.0)
        self.assertLessEqual(y, 100.0)


# ===========================================================================
# SuzukiBenchmark Tests
# ===========================================================================


class TestSuzukiBenchmark(unittest.TestCase):
    """Tests for SuzukiBenchmark."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create benchmark once for all tests (fitting is expensive)."""
        cls.bench = SuzukiBenchmark(n_train=80, seed=42)

    def test_creation_with_defaults(self) -> None:
        """Benchmark should be creatable with default args."""
        b = SuzukiBenchmark(n_train=50, seed=42)
        self.assertIsNotNone(b)

    def test_get_search_space_returns_six_params(self) -> None:
        """Search space should have 6 parameters."""
        space = self.bench.get_search_space()
        self.assertEqual(len(space), 6)

    def test_search_space_four_categorical(self) -> None:
        """Search space should have 4 categorical parameters."""
        space = self.bench.get_search_space()
        cat_count = sum(
            1 for v in space.values() if v["type"] == "categorical"
        )
        self.assertEqual(cat_count, 4)

    def test_search_space_two_continuous(self) -> None:
        """Search space should have 2 continuous parameters."""
        space = self.bench.get_search_space()
        cont_count = sum(
            1 for v in space.values() if v["type"] == "continuous"
        )
        self.assertEqual(cont_count, 2)

    def test_get_objectives_returns_yield_maximize(self) -> None:
        """Objectives should be yield (maximize, %)."""
        obj = self.bench.get_objectives()
        self.assertIn("yield", obj)
        self.assertEqual(obj["yield"]["direction"], "maximize")
        self.assertEqual(obj["yield"]["unit"], "%")

    def test_evaluate_returns_dict_with_yield(self) -> None:
        """Evaluate should return a dict with yield key."""
        result = self.bench.evaluate({
            "catalyst": "Pd(PPh3)4",
            "ligand": "XPhos",
            "base": "K2CO3",
            "solvent": "THF",
            "temperature": 80.0,
            "time": 12.0,
        })
        self.assertIsNotNone(result)
        self.assertIn("yield", result)

    def test_evaluate_returns_value_and_variance(self) -> None:
        """Evaluate result should contain value and variance."""
        result = self.bench.evaluate({
            "catalyst": "Pd(PPh3)4",
            "ligand": "XPhos",
            "base": "K2CO3",
            "solvent": "THF",
            "temperature": 80.0,
            "time": 12.0,
        })
        self.assertIn("value", result["yield"])
        self.assertIn("variance", result["yield"])
        self.assertIsInstance(result["yield"]["value"], float)
        self.assertIsInstance(result["yield"]["variance"], float)

    def test_is_feasible_compatible_pair(self) -> None:
        """Compatible pair should be feasible."""
        self.assertTrue(self.bench.is_feasible({
            "catalyst": "Pd(PPh3)4",
            "ligand": "XPhos",
        }))

    def test_is_feasible_false_pdoac2_binap(self) -> None:
        """Pd(OAc)2 + BINAP should be infeasible."""
        self.assertFalse(self.bench.is_feasible({
            "catalyst": "Pd(OAc)2",
            "ligand": "BINAP",
        }))

    def test_is_feasible_false_pdcl2_pcy3(self) -> None:
        """PdCl2 + PCy3 should be infeasible."""
        self.assertFalse(self.bench.is_feasible({
            "catalyst": "PdCl2",
            "ligand": "PCy3",
        }))

    def test_infeasible_returns_none_from_evaluate(self) -> None:
        """Evaluating an infeasible point should return None."""
        result = self.bench.evaluate({
            "catalyst": "Pd(OAc)2",
            "ligand": "BINAP",
            "base": "K2CO3",
            "solvent": "THF",
            "temperature": 80.0,
            "time": 12.0,
        })
        self.assertIsNone(result)

    def test_get_known_optimum_returns_95(self) -> None:
        """Known optimum should be 95 %."""
        opt = self.bench.get_known_optimum()
        self.assertIn("yield", opt)
        self.assertAlmostEqual(opt["yield"], 95.0)

    def test_get_evaluation_cost_positive(self) -> None:
        """Evaluation cost should be positive."""
        cost = self.bench.get_evaluation_cost({
            "catalyst": "Pd(PPh3)4",
            "ligand": "XPhos",
            "base": "K2CO3",
            "solvent": "THF",
            "temperature": 80.0,
            "time": 12.0,
        })
        self.assertGreater(cost, 0.0)

    def test_evaluation_cost_expensive_catalyst(self) -> None:
        """Expensive catalyst should have higher cost."""
        cost_cheap = self.bench.get_evaluation_cost({
            "catalyst": "Pd(OAc)2",
            "ligand": "PPh3",
            "time": 1.0,
        })
        cost_expensive = self.bench.get_evaluation_cost({
            "catalyst": "Pd2(dba)3",
            "ligand": "BINAP",
            "time": 1.0,
        })
        self.assertGreater(cost_expensive, cost_cheap)

    def test_reproducibility_same_seed(self) -> None:
        """Two benchmarks with same seed should give identical evaluations."""
        b1 = SuzukiBenchmark(n_train=50, seed=42)
        b2 = SuzukiBenchmark(n_train=50, seed=42)
        point = {
            "catalyst": "Pd(PPh3)4",
            "ligand": "XPhos",
            "base": "K2CO3",
            "solvent": "THF",
            "temperature": 80.0,
            "time": 12.0,
        }
        r1 = b1.evaluate(point)
        r2 = b2.evaluate(point)
        self.assertAlmostEqual(
            r1["yield"]["value"], r2["yield"]["value"], places=10
        )

    def test_domain_config_is_catalysis(self) -> None:
        """Domain config should be loaded for catalysis."""
        self.assertIsNotNone(self.bench.domain_config)
        self.assertEqual(self.bench.domain_config.domain_name, "catalysis")

    def test_get_known_constraints_returns_incompatibilities(self) -> None:
        """get_known_constraints should return known incompatibilities."""
        constraints = self.bench.get_known_constraints()
        self.assertIsInstance(constraints, list)
        self.assertGreater(len(constraints), 0)
        # Check that the known incompatibilities are present
        cats = {(c["catalyst"], c["ligand"]) for c in constraints}
        self.assertIn(("Pd(OAc)2", "BINAP"), cats)
        self.assertIn(("PdCl2", "PCy3"), cats)

    def test_multiple_evaluations(self) -> None:
        """Multiple evaluations should all succeed for feasible points."""
        points = [
            {
                "catalyst": "Pd(PPh3)4",
                "ligand": "PPh3",
                "base": "K2CO3",
                "solvent": "THF",
                "temperature": 60.0,
                "time": 6.0,
            },
            {
                "catalyst": "Pd2(dba)3",
                "ligand": "SPhos",
                "base": "Cs2CO3",
                "solvent": "DMF",
                "temperature": 80.0,
                "time": 12.0,
            },
            {
                "catalyst": "PdCl2",
                "ligand": "dppf",
                "base": "Et3N",
                "solvent": "dioxane",
                "temperature": 100.0,
                "time": 2.0,
            },
        ]
        for p in points:
            result = self.bench.evaluate(p)
            self.assertIsNotNone(result, f"Evaluation failed for {p}")
            self.assertIn("yield", result)
            self.assertIn("value", result["yield"])

    def test_categorical_encoding_dimension(self) -> None:
        """Encoded point from _encode should have correct dimension."""
        point = {
            "catalyst": "Pd(PPh3)4",
            "ligand": "XPhos",
            "base": "K2CO3",
            "solvent": "THF",
            "temperature": 80.0,
            "time": 12.0,
        }
        encoded = self.bench._encode(point)
        # 4 + 6 + 5 + 4 + 1 + 1 = 21
        self.assertEqual(len(encoded), 21)

    def test_get_domain_config(self) -> None:
        """get_domain_config should return a DomainConfig."""
        dc = self.bench.get_domain_config()
        self.assertIsNotNone(dc)
        self.assertEqual(dc.domain_name, "catalysis")

    def test_surrogates_populated(self) -> None:
        """After init, surrogates should be fitted for yield."""
        self.assertIn("yield", self.bench._surrogates)
        self.assertTrue(self.bench._surrogates["yield"]._fitted)

    def test_evaluate_yield_is_numeric(self) -> None:
        """Evaluated yield value should be a finite number."""
        result = self.bench.evaluate({
            "catalyst": "Pd2(dba)3",
            "ligand": "SPhos",
            "base": "Cs2CO3",
            "solvent": "THF",
            "temperature": 80.0,
            "time": 12.0,
        })
        self.assertIsNotNone(result)
        val = result["yield"]["value"]
        self.assertTrue(math.isfinite(val))


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestCatalysisIntegration(unittest.TestCase):
    """Integration tests for benchmark + data_loader consistency."""

    def test_benchmark_and_loader_search_space_match(self) -> None:
        """Benchmark and data loader search spaces should have same keys."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        bench = SuzukiBenchmark(n_train=50, seed=42)
        loader_space = loader.get_search_space()
        bench_space = bench.get_search_space()
        self.assertEqual(set(loader_space.keys()), set(bench_space.keys()))

    def test_search_space_parameter_order_matches(self) -> None:
        """Key ordering in search spaces must match for encoding consistency."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        bench = SuzukiBenchmark(n_train=50, seed=42)
        loader_keys = list(loader.get_search_space().keys())
        bench_keys = list(bench.get_search_space().keys())
        self.assertEqual(loader_keys, bench_keys)

    def test_search_space_categories_match(self) -> None:
        """Categories for each categorical param should match."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        bench = SuzukiBenchmark(n_train=50, seed=42)
        loader_space = loader.get_search_space()
        bench_space = bench.get_search_space()
        for name in loader_space:
            if loader_space[name]["type"] == "categorical":
                self.assertEqual(
                    loader_space[name]["categories"],
                    bench_space[name]["categories"],
                    f"Category mismatch for {name}",
                )

    def test_encoding_consistency(self) -> None:
        """Data loader encoding should match benchmark _encode output."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        bench = SuzukiBenchmark(n_train=50, seed=42)
        point = {
            "catalyst": "Pd(PPh3)4",
            "ligand": "SPhos",
            "base": "Cs2CO3",
            "solvent": "DMF",
            "temperature": 80.0,
            "time": 12.0,
        }
        loader_encoded = loader._encode_point(
            point["catalyst"],
            point["ligand"],
            point["base"],
            point["solvent"],
            point["temperature"],
            point["time"],
        )
        bench_encoded = bench._encode(point)
        self.assertEqual(len(loader_encoded), len(bench_encoded))
        for i, (a, b) in enumerate(zip(loader_encoded, bench_encoded)):
            self.assertAlmostEqual(
                a, b, places=10,
                msg=f"Encoding mismatch at position {i}",
            )

    def test_training_data_dimension_matches_encoding(self) -> None:
        """Training data X dimension should match encoded point dimension."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        data = loader.get_data()
        # All X rows should have dim 21
        for i, row in enumerate(data["X"]):
            self.assertEqual(
                len(row), 21,
                f"Row {i} has dimension {len(row)}, expected 21",
            )

    def test_known_optimum_matches(self) -> None:
        """Known optimum should be consistent between loader and benchmark."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        bench = SuzukiBenchmark(n_train=50, seed=42)
        self.assertEqual(
            loader.get_known_optimum()["yield"],
            bench.get_known_optimum()["yield"],
        )

    def test_continuous_ranges_match(self) -> None:
        """Continuous parameter ranges should match between loader and benchmark."""
        loader = CatalysisDataLoader(n_points=10, seed=42)
        bench = SuzukiBenchmark(n_train=50, seed=42)
        loader_space = loader.get_search_space()
        bench_space = bench.get_search_space()
        for name in ["temperature", "time"]:
            self.assertEqual(
                loader_space[name]["range"],
                bench_space[name]["range"],
                f"Range mismatch for {name}",
            )

    def test_domain_knowledge_import_consistency(self) -> None:
        """Benchmark should use the same domain knowledge as the data loader."""
        bench = SuzukiBenchmark(n_train=50, seed=42)
        space = bench.get_search_space()
        self.assertEqual(space["catalyst"]["categories"], list(CATALYSTS))
        self.assertEqual(space["ligand"]["categories"], list(LIGANDS))
        self.assertEqual(space["base"]["categories"], list(BASES))
        self.assertEqual(space["solvent"]["categories"], list(SOLVENTS))


if __name__ == "__main__":
    unittest.main()

"""Tests for the hybrid theory-data engine (Layer 5).

Covers theory models, residual GP, hybrid model, and discrepancy
analyzer with synthetic data using deterministic seeds.
"""

from __future__ import annotations

import math
import random
import unittest


class TestTheoryModels(unittest.TestCase):
    """Test individual theory model predictions."""

    def test_arrhenius_known_output(self) -> None:
        """Arrhenius at T=300K with default parameters."""
        from optimization_copilot.hybrid.theory import ArrheniusModel

        model = ArrheniusModel(A=1.0, Ea=50000.0, R=8.314)
        X = [[300.0]]
        result = model.predict(X)
        # Expected: exp(-50000 / (8.314 * 300)) = exp(-20.055...)
        expected = math.exp(-50000.0 / (8.314 * 300.0))
        self.assertAlmostEqual(result[0], expected, places=10)

    def test_arrhenius_zero_temperature(self) -> None:
        """Arrhenius should handle T <= 0 safely."""
        from optimization_copilot.hybrid.theory import ArrheniusModel

        model = ArrheniusModel()
        result = model.predict([[0.0], [-10.0]])
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], 0.0)

    def test_arrhenius_parameter_info(self) -> None:
        from optimization_copilot.hybrid.theory import ArrheniusModel

        model = ArrheniusModel()
        self.assertEqual(model.n_parameters(), 3)
        self.assertEqual(model.parameter_names(), ["A", "Ea", "R"])

    def test_michaelis_menten_half_max(self) -> None:
        """V at S = Km should be Vmax / 2."""
        from optimization_copilot.hybrid.theory import MichaelisMentenModel

        model = MichaelisMentenModel(Vmax=10.0, Km=5.0)
        result = model.predict([[5.0]])  # S = Km
        self.assertAlmostEqual(result[0], 5.0, places=10)

    def test_michaelis_menten_saturation(self) -> None:
        """V at S >> Km should approach Vmax."""
        from optimization_copilot.hybrid.theory import MichaelisMentenModel

        model = MichaelisMentenModel(Vmax=10.0, Km=5.0)
        result = model.predict([[50000.0]])
        self.assertAlmostEqual(result[0], 10.0, delta=0.01)

    def test_michaelis_menten_zero_substrate(self) -> None:
        from optimization_copilot.hybrid.theory import MichaelisMentenModel

        model = MichaelisMentenModel(Vmax=10.0, Km=5.0)
        result = model.predict([[0.0]])
        self.assertAlmostEqual(result[0], 0.0, places=10)

    def test_power_law_known_output(self) -> None:
        """y = 2 * x^2 at x=3 should give 18."""
        from optimization_copilot.hybrid.theory import PowerLawModel

        model = PowerLawModel(a=2.0, b=2.0)
        result = model.predict([[3.0]])
        self.assertAlmostEqual(result[0], 18.0, places=10)

    def test_power_law_identity(self) -> None:
        """y = 1 * x^1 should give y = x."""
        from optimization_copilot.hybrid.theory import PowerLawModel

        model = PowerLawModel(a=1.0, b=1.0)
        X = [[1.0], [2.0], [5.0], [0.0]]
        result = model.predict(X)
        for i, row in enumerate(X):
            self.assertAlmostEqual(result[i], row[0], places=10)

    def test_ode_exponential_decay(self) -> None:
        """dy/dt = -y, y(0) = 1 => y(1) ~ exp(-1)."""
        from optimization_copilot.hybrid.theory import ODEModel

        def decay(t: float, y: list[float]) -> list[float]:
            return [-y[0]]

        model = ODEModel(
            rhs_fn=decay,
            y0=[1.0],
            t_span=(0.0, 1.0),
            output_index=0,
            n_steps=200,
        )
        # X[i][0] is the time endpoint
        result = model.predict([[1.0]])
        expected = math.exp(-1.0)
        self.assertAlmostEqual(result[0], expected, places=4)

    def test_ode_multiple_endpoints(self) -> None:
        """Test ODE model at multiple time endpoints."""
        from optimization_copilot.hybrid.theory import ODEModel

        def decay(t: float, y: list[float]) -> list[float]:
            return [-y[0]]

        model = ODEModel(
            rhs_fn=decay,
            y0=[1.0],
            t_span=(0.0, 1.0),
            n_steps=200,
        )
        X = [[0.5], [1.0], [2.0]]
        result = model.predict(X)
        for i, row in enumerate(X):
            expected = math.exp(-row[0])
            self.assertAlmostEqual(result[i], expected, places=3)

    def test_ode_zero_endpoint(self) -> None:
        """ODE at t_end <= t0 returns initial condition."""
        from optimization_copilot.hybrid.theory import ODEModel

        def decay(t: float, y: list[float]) -> list[float]:
            return [-y[0]]

        model = ODEModel(rhs_fn=decay, y0=[1.0], t_span=(0.0, 1.0))
        result = model.predict([[0.0]])
        self.assertAlmostEqual(result[0], 1.0, places=10)


class TestResidualGP(unittest.TestCase):
    """Test the residual GP fitting and prediction."""

    def test_constant_bias(self) -> None:
        """Theory y=x, true data y=x+0.5 => residual mean ~0.5."""
        from optimization_copilot.hybrid.theory import PowerLawModel
        from optimization_copilot.hybrid.residual import ResidualGP

        theory = PowerLawModel(a=1.0, b=1.0)
        gp = ResidualGP(theory, noise=1e-4)

        random.seed(42)
        X = [[float(i)] for i in range(1, 11)]
        y = [float(i) + 0.5 for i in range(1, 11)]

        gp.fit(X, y)

        summary = gp.residual_summary()
        self.assertAlmostEqual(summary["mean"], 0.5, places=5)
        self.assertTrue(summary["has_systematic_bias"])

    def test_predict_shape(self) -> None:
        """Predict returns correct number of means and stds."""
        from optimization_copilot.hybrid.theory import PowerLawModel
        from optimization_copilot.hybrid.residual import ResidualGP

        theory = PowerLawModel(a=1.0, b=1.0)
        gp = ResidualGP(theory, noise=1e-4)

        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [1.5, 2.5, 3.5, 4.5, 5.5]
        gp.fit(X, y)

        X_new = [[1.5], [2.5], [6.0]]
        means, stds = gp.predict(X_new)

        self.assertEqual(len(means), 3)
        self.assertEqual(len(stds), 3)

    def test_predict_nonzero_uncertainty(self) -> None:
        """Predictions at new points should have positive uncertainty."""
        from optimization_copilot.hybrid.theory import PowerLawModel
        from optimization_copilot.hybrid.residual import ResidualGP

        theory = PowerLawModel(a=1.0, b=1.0)
        gp = ResidualGP(theory, noise=1e-4)

        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [1.5, 2.5, 3.5, 4.5, 5.5]
        gp.fit(X, y)

        # Far from training data => higher uncertainty
        X_far = [[10.0], [20.0]]
        _, stds = gp.predict(X_far)
        for s in stds:
            self.assertGreater(s, 0.0)

    def test_predict_before_fit_raises(self) -> None:
        """Calling predict before fit should raise RuntimeError."""
        from optimization_copilot.hybrid.theory import PowerLawModel
        from optimization_copilot.hybrid.residual import ResidualGP

        theory = PowerLawModel(a=1.0, b=1.0)
        gp = ResidualGP(theory)
        with self.assertRaises(RuntimeError):
            gp.predict([[1.0]])

    def test_residual_summary_no_bias(self) -> None:
        """When theory is perfect, residuals should have zero mean."""
        from optimization_copilot.hybrid.theory import PowerLawModel
        from optimization_copilot.hybrid.residual import ResidualGP

        theory = PowerLawModel(a=1.0, b=1.0)
        gp = ResidualGP(theory, noise=1e-4)

        # Theory is perfect: y = x
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        gp.fit(X, y)

        summary = gp.residual_summary()
        self.assertAlmostEqual(summary["mean"], 0.0, places=10)
        self.assertAlmostEqual(summary["std"], 0.0, places=10)


class TestHybridModel(unittest.TestCase):
    """Test the composite hybrid model."""

    def setUp(self) -> None:
        """Create a hybrid model: theory = y=x, true = y = x + sin(x)."""
        from optimization_copilot.hybrid.theory import PowerLawModel
        from optimization_copilot.hybrid.residual import ResidualGP
        from optimization_copilot.hybrid.composite import HybridModel

        random.seed(42)

        self.theory = PowerLawModel(a=1.0, b=1.0)
        self.gp = ResidualGP(self.theory, noise=1e-4)
        self.hybrid = HybridModel(self.theory, self.gp)

        # Training data: y = x + sin(x)
        self.X_train = [[0.5 * i] for i in range(1, 21)]
        self.y_train = [
            row[0] + math.sin(row[0]) for row in self.X_train
        ]
        self.hybrid.fit(self.X_train, self.y_train)

    def test_hybrid_lower_rmse(self) -> None:
        """Hybrid should have lower RMSE than theory alone."""
        # Test data in training range
        X_test = [[0.25 + 0.5 * i] for i in range(1, 16)]
        y_test = [row[0] + math.sin(row[0]) for row in X_test]

        comparison = self.hybrid.compare_to_theory_only(X_test, y_test)

        self.assertGreater(comparison["theory_rmse"], 0.0)
        self.assertLess(comparison["hybrid_rmse"], comparison["theory_rmse"])
        self.assertGreater(comparison["improvement_pct"], 0.0)

    def test_predict_with_uncertainty_shapes(self) -> None:
        """predict_with_uncertainty returns correct shapes."""
        X_new = [[1.0], [2.0], [3.0]]
        means, stds = self.hybrid.predict_with_uncertainty(X_new)
        self.assertEqual(len(means), 3)
        self.assertEqual(len(stds), 3)

    def test_suggest_next_ei(self) -> None:
        """suggest_next with EI returns ranked candidates."""
        X_cand = [[0.5 * i] for i in range(1, 11)]
        results = self.hybrid.suggest_next(X_cand, acquisition="ei")

        self.assertEqual(len(results), 10)
        # Check sorted descending by acquisition value
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i]["acquisition_value"],
                results[i + 1]["acquisition_value"],
            )
        # Each result should have the required keys
        for r in results:
            self.assertIn("index", r)
            self.assertIn("x", r)
            self.assertIn("mean", r)
            self.assertIn("std", r)
            self.assertIn("acquisition_value", r)

    def test_suggest_next_ucb(self) -> None:
        """suggest_next with UCB returns ranked candidates."""
        X_cand = [[0.5 * i] for i in range(1, 6)]
        results = self.hybrid.suggest_next(X_cand, acquisition="ucb")
        self.assertEqual(len(results), 5)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i]["acquisition_value"],
                results[i + 1]["acquisition_value"],
            )

    def test_suggest_next_invalid_acquisition(self) -> None:
        """suggest_next with unknown acquisition raises ValueError."""
        with self.assertRaises(ValueError):
            self.hybrid.suggest_next([[1.0]], acquisition="invalid")

    def test_theory_adequacy_good_theory(self) -> None:
        """Good theory (small residuals) => high adequacy score."""
        from optimization_copilot.hybrid.theory import PowerLawModel
        from optimization_copilot.hybrid.residual import ResidualGP
        from optimization_copilot.hybrid.composite import HybridModel

        # Theory is nearly perfect: y = x, data = x + tiny noise
        theory = PowerLawModel(a=1.0, b=1.0)
        gp = ResidualGP(theory, noise=1e-4)
        hybrid = HybridModel(theory, gp)

        random.seed(42)
        X = [[float(i)] for i in range(1, 21)]
        y = [float(i) + random.gauss(0, 0.01) for i in range(1, 21)]
        hybrid.fit(X, y)

        score = hybrid.theory_adequacy_score()
        self.assertGreater(score, 0.9)

    def test_theory_adequacy_bad_theory(self) -> None:
        """Bad theory (large residuals) => low adequacy score."""
        from optimization_copilot.hybrid.theory import PowerLawModel
        from optimization_copilot.hybrid.residual import ResidualGP
        from optimization_copilot.hybrid.composite import HybridModel

        # Theory is y = x, but data = x^2 (very different)
        theory = PowerLawModel(a=1.0, b=1.0)
        gp = ResidualGP(theory, noise=1e-4)
        hybrid = HybridModel(theory, gp)

        X = [[float(i)] for i in range(1, 21)]
        y = [float(i) ** 2 for i in range(1, 21)]
        hybrid.fit(X, y)

        score = hybrid.theory_adequacy_score()
        self.assertLess(score, 0.5)


class TestDiscrepancyAnalyzer(unittest.TestCase):
    """Test the discrepancy analyzer."""

    def test_systematic_bias_detection(self) -> None:
        """Detect positive bias when theory under-predicts."""
        from optimization_copilot.hybrid.theory import PowerLawModel
        from optimization_copilot.hybrid.residual import ResidualGP
        from optimization_copilot.hybrid.discrepancy import DiscrepancyAnalyzer

        # Theory: y = x, data: y = x + 2 (constant positive bias)
        theory = PowerLawModel(a=1.0, b=1.0)
        gp = ResidualGP(theory, noise=1e-4)

        X = [[float(i)] for i in range(1, 11)]
        y = [float(i) + 2.0 for i in range(1, 11)]
        gp.fit(X, y)

        analyzer = DiscrepancyAnalyzer()
        result = analyzer.systematic_bias(gp)

        self.assertAlmostEqual(result["mean_residual"], 2.0, places=5)
        self.assertTrue(result["is_biased"])
        self.assertEqual(result["bias_direction"], "positive")

    def test_no_bias_detection(self) -> None:
        """No bias when theory is perfect."""
        from optimization_copilot.hybrid.theory import PowerLawModel
        from optimization_copilot.hybrid.residual import ResidualGP
        from optimization_copilot.hybrid.discrepancy import DiscrepancyAnalyzer

        theory = PowerLawModel(a=1.0, b=1.0)
        gp = ResidualGP(theory, noise=1e-4)

        X = [[float(i)] for i in range(1, 11)]
        y = [float(i) for i in range(1, 11)]
        gp.fit(X, y)

        analyzer = DiscrepancyAnalyzer()
        result = analyzer.systematic_bias(gp)
        self.assertFalse(result["is_biased"])
        self.assertEqual(result["bias_direction"], "none")

    def test_failure_regions(self) -> None:
        """Detect failure regions where theory breaks down."""
        from optimization_copilot.hybrid.theory import PowerLawModel
        from optimization_copilot.hybrid.residual import ResidualGP
        from optimization_copilot.hybrid.composite import HybridModel
        from optimization_copilot.hybrid.discrepancy import DiscrepancyAnalyzer

        # Theory: y = x, data has a bump near x=5
        theory = PowerLawModel(a=1.0, b=1.0)
        gp = ResidualGP(theory, noise=1e-4)
        hybrid = HybridModel(theory, gp)

        random.seed(42)
        X_train = [[float(i)] for i in range(1, 21)]
        y_train = []
        for row in X_train:
            x = row[0]
            # Add large deviation near x=5
            if 4 <= x <= 6:
                y_train.append(x + 5.0)
            else:
                y_train.append(x)
        hybrid.fit(X_train, y_train)

        # Evaluate at training points
        analyzer = DiscrepancyAnalyzer()
        failures = analyzer.failure_regions(hybrid, X_train, threshold=1.5)

        # Should find some failures around x=4,5,6
        self.assertGreater(len(failures), 0)
        # Each failure should have severity > threshold
        for f in failures:
            self.assertGreater(f["severity"], 1.5)

    def test_model_adequacy_good(self) -> None:
        """Adequate model: small residuals pass the test."""
        from optimization_copilot.hybrid.discrepancy import DiscrepancyAnalyzer

        random.seed(42)
        # Small residuals relative to noise_std
        residuals = [random.gauss(0, 0.5) for _ in range(20)]

        analyzer = DiscrepancyAnalyzer()
        result = analyzer.model_adequacy_test(residuals, noise_std=1.0)

        self.assertEqual(result["n"], 20)
        self.assertTrue(result["is_adequate"])
        self.assertLess(result["Q_over_n"], 2.0)

    def test_model_adequacy_bad(self) -> None:
        """Inadequate model: large residuals fail the test."""
        from optimization_copilot.hybrid.discrepancy import DiscrepancyAnalyzer

        # Large residuals relative to noise_std
        residuals = [5.0] * 20

        analyzer = DiscrepancyAnalyzer()
        result = analyzer.model_adequacy_test(residuals, noise_std=1.0)

        self.assertFalse(result["is_adequate"])
        self.assertGreater(result["Q_over_n"], 2.0)

    def test_model_adequacy_empty(self) -> None:
        """Empty residuals should pass the test."""
        from optimization_copilot.hybrid.discrepancy import DiscrepancyAnalyzer

        analyzer = DiscrepancyAnalyzer()
        result = analyzer.model_adequacy_test([])
        self.assertTrue(result["is_adequate"])
        self.assertEqual(result["n"], 0)

    def test_suggest_theory_revision(self) -> None:
        """Suggestions should be generated from failure regions."""
        from optimization_copilot.hybrid.discrepancy import DiscrepancyAnalyzer

        analyzer = DiscrepancyAnalyzer()

        # Synthetic failure regions at high x values
        failure_regions = [
            {"index": 0, "x": [8.0], "residual_mean": 3.0, "residual_std": 0.5, "severity": 6.0},
            {"index": 1, "x": [9.0], "residual_mean": 4.0, "residual_std": 0.5, "severity": 8.0},
            {"index": 2, "x": [10.0], "residual_mean": 5.0, "residual_std": 0.5, "severity": 10.0},
        ]

        suggestions = analyzer.suggest_theory_revision(
            failure_regions, var_names=["temperature"]
        )
        self.assertGreater(len(suggestions), 0)
        # Should mention high values and/or severity
        combined = " ".join(suggestions)
        self.assertTrue(
            "high" in combined.lower() or "severity" in combined.lower()
            or "under-predicting" in combined.lower()
        )

    def test_suggest_theory_revision_no_failures(self) -> None:
        """No failures should give an adequate message."""
        from optimization_copilot.hybrid.discrepancy import DiscrepancyAnalyzer

        analyzer = DiscrepancyAnalyzer()
        suggestions = analyzer.suggest_theory_revision([])
        self.assertEqual(len(suggestions), 1)
        self.assertIn("adequate", suggestions[0].lower())


if __name__ == "__main__":
    unittest.main()

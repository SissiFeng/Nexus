"""Tests for KernelSHAPApproximator in _analysis/shap_values.py."""

from __future__ import annotations

import math

import pytest

from optimization_copilot._analysis.shap_values import (
    KernelSHAPApproximator,
    _comb,
    _evaluate_coalition,
    _shap_kernel_weight,
)
from optimization_copilot.visualization.models import SurrogateModel


# ---------------------------------------------------------------------------
# Mock models
# ---------------------------------------------------------------------------

class LinearModel:
    """f(x) = sum(coeff_i * x_i) + intercept."""

    def __init__(self, coeffs: list[float], intercept: float = 0.0) -> None:
        self.coeffs = coeffs
        self.intercept = intercept

    def predict(self, x: list[float]) -> tuple[float, float]:
        return (sum(c * v for c, v in zip(self.coeffs, x)) + self.intercept, 0.0)


class ConstantModel:
    """Always returns the same value."""

    def __init__(self, value: float = 5.0) -> None:
        self.value = value

    def predict(self, x: list[float]) -> tuple[float, float]:
        return (self.value, 0.0)


class QuadraticModel:
    """f(x) = sum(x_i^2)."""

    def predict(self, x: list[float]) -> tuple[float, float]:
        return (sum(v * v for v in x), 0.0)


# ---------------------------------------------------------------------------
# _comb helper
# ---------------------------------------------------------------------------

class TestComb:
    def test_comb_basic(self):
        assert _comb(5, 2) == 10

    def test_comb_zero(self):
        assert _comb(5, 0) == 1

    def test_comb_n_equals_k(self):
        assert _comb(4, 4) == 1

    def test_comb_large(self):
        assert _comb(20, 10) == 184756


# ---------------------------------------------------------------------------
# SHAP kernel weight
# ---------------------------------------------------------------------------

class TestKernelWeight:
    def test_weight_degenerate_empty(self):
        assert _shap_kernel_weight(0, 5) == 0.0

    def test_weight_degenerate_full(self):
        assert _shap_kernel_weight(5, 5) == 0.0

    def test_weight_single_feature(self):
        # d=5, |S|=1: (5-1) / (C(5,1)*1*4) = 4/20 = 0.2
        assert _shap_kernel_weight(1, 5) == pytest.approx(0.2)

    def test_weight_symmetry(self):
        # pi(|S|) = pi(d - |S|) for the SHAP kernel
        d = 6
        for s in range(1, d):
            assert _shap_kernel_weight(s, d) == pytest.approx(
                _shap_kernel_weight(d - s, d)
            )


# ---------------------------------------------------------------------------
# Efficiency property: sum(phi_i) ≈ f(x) - E[f(X)]
# ---------------------------------------------------------------------------

class TestEfficiencyProperty:
    def test_d2_linear_exact(self):
        """For d=2 linear model, SHAP values should recover coefficients."""
        model = LinearModel([2.0, 3.0])
        shap = KernelSHAPApproximator(model, seed=42)
        x = [1.0, 1.0]
        bg = [[0.0, 0.0]]
        phi = shap.compute(x, bg)
        # f(x)=5, E[f]=0, phi should ≈ [2.0, 3.0]
        assert len(phi) == 2
        assert sum(phi) == pytest.approx(5.0, abs=0.1)

    def test_d3_efficiency(self):
        """sum(phi) ≈ f(x) - E[f(X)] for d=3."""
        model = LinearModel([1.0, 2.0, 3.0], intercept=1.0)
        shap = KernelSHAPApproximator(model, seed=0)
        x = [1.0, 1.0, 1.0]
        bg = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]

        phi = shap.compute(x, bg)
        fx = model.predict(x)[0]  # 1+2+3+1 = 7
        e_fx = sum(model.predict(b)[0] for b in bg) / len(bg)  # (1 + 4) / 2 = 2.5
        assert sum(phi) == pytest.approx(fx - e_fx, abs=0.2)

    def test_d5_exact_efficiency(self):
        """Efficiency for a 5-dimensional problem (exact mode)."""
        model = LinearModel([1.0, -1.0, 2.0, -2.0, 0.5])
        shap = KernelSHAPApproximator(model, seed=123)
        x = [1.0, 1.0, 1.0, 1.0, 1.0]
        bg = [[0.0] * 5]

        phi = shap.compute(x, bg)
        fx = model.predict(x)[0]
        e_fx = model.predict(bg[0])[0]
        assert sum(phi) == pytest.approx(fx - e_fx, abs=0.3)


# ---------------------------------------------------------------------------
# Linear model coefficient recovery
# ---------------------------------------------------------------------------

class TestLinearCoefficients:
    def test_d2_coefficients(self):
        """SHAP values for a linear model should match coefficients."""
        model = LinearModel([3.0, -1.0])
        shap = KernelSHAPApproximator(model, seed=42)
        x = [1.0, 1.0]
        bg = [[0.0, 0.0]]
        phi = shap.compute(x, bg)
        assert phi[0] == pytest.approx(3.0, abs=0.2)
        assert phi[1] == pytest.approx(-1.0, abs=0.2)

    def test_d3_symmetric_features(self):
        """If two features have identical coefficients, their SHAP values should be equal."""
        model = LinearModel([2.0, 2.0, 5.0])
        shap = KernelSHAPApproximator(model, seed=99)
        x = [1.0, 1.0, 1.0]
        bg = [[0.0, 0.0, 0.0]]
        phi = shap.compute(x, bg)
        # phi[0] and phi[1] should be approximately equal
        assert phi[0] == pytest.approx(phi[1], abs=0.3)


# ---------------------------------------------------------------------------
# Sampled mode (d >= 12)
# ---------------------------------------------------------------------------

class TestSampledMode:
    def test_d15_does_not_crash(self):
        """Sampled mode for d=15 should complete without error."""
        model = LinearModel([1.0] * 15)
        shap = KernelSHAPApproximator(model, n_samples=200, seed=42)
        x = [1.0] * 15
        bg = [[0.0] * 15] * 5
        phi = shap.compute(x, bg)
        assert len(phi) == 15

    def test_d15_reasonable_values(self):
        """SHAP values should be finite and not absurdly large."""
        model = LinearModel([float(i) for i in range(15)])
        shap = KernelSHAPApproximator(model, n_samples=500, seed=42)
        x = [1.0] * 15
        bg = [[0.0] * 15] * 10
        phi = shap.compute(x, bg)
        for v in phi:
            assert math.isfinite(v)
            assert abs(v) < 100  # reasonable bound

    def test_d15_efficiency(self):
        """Efficiency property should hold approximately in sampled mode."""
        model = LinearModel([1.0] * 15)
        shap = KernelSHAPApproximator(model, n_samples=800, seed=7)
        x = [1.0] * 15
        bg = [[0.0] * 15] * 5
        phi = shap.compute(x, bg)
        fx = model.predict(x)[0]
        e_fx = sum(model.predict(b)[0] for b in bg) / len(bg)
        # Looser tolerance for sampled mode
        assert sum(phi) == pytest.approx(fx - e_fx, abs=2.0)

    def test_d12_boundary(self):
        """d=12 should use sampled mode (boundary case)."""
        model = LinearModel([1.0] * 12)
        shap = KernelSHAPApproximator(model, n_samples=300, seed=42)
        x = [1.0] * 12
        bg = [[0.0] * 12]
        phi = shap.compute(x, bg)
        assert len(phi) == 12


# ---------------------------------------------------------------------------
# Weighted regression
# ---------------------------------------------------------------------------

class TestWeightedRegression:
    def test_simple_system(self):
        """Known solution for a simple weighted regression."""
        model = ConstantModel()
        shap = KernelSHAPApproximator(model, seed=0)
        # Z = [[1, 0], [0, 1]], y = [2, 3], w = [1, 1]
        # (Z^T W Z)^{-1} Z^T W y = [2, 3]
        phi = shap._weighted_regression(
            Z=[[1.0, 0.0], [0.0, 1.0]],
            y=[2.0, 3.0],
            w=[1.0, 1.0],
            d=2,
        )
        assert phi[0] == pytest.approx(2.0, abs=0.01)
        assert phi[1] == pytest.approx(3.0, abs=0.01)

    def test_weighted_system(self):
        """Weighted regression with unequal weights."""
        model = ConstantModel()
        shap = KernelSHAPApproximator(model, seed=0)
        phi = shap._weighted_regression(
            Z=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            y=[1.0, 2.0, 3.0],
            w=[1.0, 1.0, 10.0],
            d=2,
        )
        # With the large weight on the [1,1] row, the solution should
        # favour phi[0]+phi[1] ≈ 3.
        assert abs(phi[0] + phi[1] - 3.0) < 0.5

    def test_empty_Z(self):
        """Empty coalition matrix should return zeros."""
        model = ConstantModel()
        shap = KernelSHAPApproximator(model, seed=0)
        phi = shap._weighted_regression([], [], [], d=3)
        assert phi == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_background_point(self):
        """SHAP with a single background point."""
        model = LinearModel([2.0, 3.0])
        shap = KernelSHAPApproximator(model, seed=42)
        phi = shap.compute([1.0, 1.0], [[0.0, 0.0]])
        assert len(phi) == 2
        assert sum(phi) == pytest.approx(5.0, abs=0.2)

    def test_all_same_background_points(self):
        """All background points are identical."""
        model = LinearModel([1.0, 2.0])
        shap = KernelSHAPApproximator(model, seed=42)
        bg = [[0.5, 0.5]] * 10
        phi = shap.compute([1.0, 1.0], bg)
        fx = model.predict([1.0, 1.0])[0]
        e_fx = model.predict([0.5, 0.5])[0]
        assert sum(phi) == pytest.approx(fx - e_fx, abs=0.2)

    def test_empty_features(self):
        """d=0 should return empty list."""
        model = ConstantModel()
        shap = KernelSHAPApproximator(model, seed=42)
        phi = shap.compute([], [[]])
        assert phi == []

    def test_single_feature(self):
        """d=1 trivial case."""
        model = LinearModel([5.0])
        shap = KernelSHAPApproximator(model, seed=42)
        phi = shap.compute([2.0], [[0.0]])
        assert len(phi) == 1
        # f(x)=10, E[f]=0 -> phi[0]=10
        assert phi[0] == pytest.approx(10.0, abs=0.1)

    def test_constant_model(self):
        """Constant model should have all-zero SHAP values."""
        model = ConstantModel(value=42.0)
        shap = KernelSHAPApproximator(model, seed=42)
        phi = shap.compute([1.0, 2.0, 3.0], [[0.0, 0.0, 0.0]])
        for v in phi:
            assert abs(v) < 0.1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_result(self):
        """Two runs with the same seed produce identical results."""
        model = LinearModel([1.0, 2.0, 3.0])
        x = [1.0, 1.0, 1.0]
        bg = [[0.0, 0.0, 0.0]]

        shap1 = KernelSHAPApproximator(model, seed=99)
        phi1 = shap1.compute(x, bg)

        shap2 = KernelSHAPApproximator(model, seed=99)
        phi2 = shap2.compute(x, bg)

        for a, b in zip(phi1, phi2):
            assert a == pytest.approx(b)

    def test_different_seed_different_result_sampled(self):
        """Different seeds should generally produce different results in sampled mode."""
        # Use non-uniform coefficients and varied background to break symmetry.
        coeffs = [float(i + 1) for i in range(15)]
        model = LinearModel(coeffs)
        x = [1.0] * 15
        import random as _rng
        _rng.seed(555)
        bg = [[_rng.random() for _ in range(15)] for _ in range(10)]

        shap1 = KernelSHAPApproximator(model, n_samples=200, seed=1)
        phi1 = shap1.compute(x, bg)

        shap2 = KernelSHAPApproximator(model, n_samples=200, seed=2)
        phi2 = shap2.compute(x, bg)

        # Not exactly equal (with high probability)
        diffs = [abs(a - b) for a, b in zip(phi1, phi2)]
        assert max(diffs) > 1e-6


# ---------------------------------------------------------------------------
# Coalition evaluation
# ---------------------------------------------------------------------------

class TestCoalitionEvaluation:
    def test_all_present(self):
        """All features in coalition -> f(x)."""
        model = LinearModel([1.0, 2.0])
        val = _evaluate_coalition(
            model, [3.0, 4.0], [[0.0, 0.0]], [True, True]
        )
        assert val == pytest.approx(11.0)

    def test_none_present(self):
        """No features in coalition -> E[f(bg)]."""
        model = LinearModel([1.0, 2.0])
        val = _evaluate_coalition(
            model, [3.0, 4.0], [[1.0, 1.0], [2.0, 2.0]], [False, False]
        )
        expected = (3.0 + 6.0) / 2  # f([1,1])=3, f([2,2])=6
        assert val == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    def test_linear_model_is_surrogate(self):
        m = LinearModel([1.0])
        assert isinstance(m, SurrogateModel)

    def test_constant_model_is_surrogate(self):
        m = ConstantModel()
        assert isinstance(m, SurrogateModel)

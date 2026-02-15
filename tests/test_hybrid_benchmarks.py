"""Tests for optimization_copilot.hybrid.benchmarks module.

Covers SampleEfficiencyBenchmark, SampleEfficiencyResult,
SampleEfficiencyCurve, and the internal _ZeroTheory helper.
"""

from __future__ import annotations

import math

import pytest

from optimization_copilot.hybrid.benchmarks import (
    SampleEfficiencyBenchmark,
    SampleEfficiencyCurve,
    SampleEfficiencyResult,
    _ZeroTheory,
)
from optimization_copilot.hybrid.theory import TheoryModel


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


class _LinearTheory(TheoryModel):
    """Theory: f(x) = 2*x[0]. Perfect for 1D linear data."""

    def predict(self, X: list[list[float]]) -> list[float]:
        return [2.0 * x[0] for x in X]

    def n_parameters(self) -> int:
        return 1

    def parameter_names(self) -> list[str]:
        return ["slope"]


def _true_function(X: list[list[float]]) -> list[float]:
    """True data-generating function: y = 2*x + 0.5*sin(5*x).

    The linear theory captures the main 2*x trend but not the
    oscillatory 0.5*sin(5*x) component, so the residual GP has
    something meaningful to learn.
    """
    return [2.0 * x[0] + 0.5 * math.sin(5.0 * x[0]) for x in X]


_DOMAIN_1D: list[tuple[float, float]] = [(0.0, 5.0)]
_SMALL_SAMPLE_SIZES: list[int] = [5, 10, 20]
_SMALL_N_TEST: int = 20


@pytest.fixture()
def benchmark() -> SampleEfficiencyBenchmark:
    """A benchmark configured with small sizes for fast tests."""
    return SampleEfficiencyBenchmark(
        sample_sizes=_SMALL_SAMPLE_SIZES,
        n_test=_SMALL_N_TEST,
        rmse_threshold=0.5,
        seed=42,
    )


@pytest.fixture()
def linear_theory() -> _LinearTheory:
    return _LinearTheory()


@pytest.fixture()
def result(
    benchmark: SampleEfficiencyBenchmark,
    linear_theory: _LinearTheory,
) -> SampleEfficiencyResult:
    """Run the benchmark once and cache the result for multiple assertions."""
    return benchmark.compare(
        theory=linear_theory,
        data_generator=_true_function,
        X_domain=_DOMAIN_1D,
        noise_std=0.1,
    )


# ---------------------------------------------------------------------------
# 1. Basic operation
# ---------------------------------------------------------------------------


class TestBasicOperation:
    """compare() returns SampleEfficiencyResult with correct structure."""

    def test_returns_sample_efficiency_result(
        self, result: SampleEfficiencyResult
    ) -> None:
        assert isinstance(result, SampleEfficiencyResult)

    def test_contains_three_curves(self, result: SampleEfficiencyResult) -> None:
        assert len(result.curves) == 3

    def test_curve_methods(self, result: SampleEfficiencyResult) -> None:
        methods = {c.method for c in result.curves}
        assert methods == {"hybrid", "pure_gp", "theory_only"}

    def test_curves_have_correct_sample_sizes(
        self, result: SampleEfficiencyResult
    ) -> None:
        for curve in result.curves:
            assert curve.sample_sizes == _SMALL_SAMPLE_SIZES

    def test_curves_have_matching_rmse_length(
        self, result: SampleEfficiencyResult
    ) -> None:
        for curve in result.curves:
            assert len(curve.rmse_values) == len(_SMALL_SAMPLE_SIZES)

    def test_rmse_values_are_non_negative(
        self, result: SampleEfficiencyResult
    ) -> None:
        for curve in result.curves:
            for rmse in curve.rmse_values:
                assert rmse >= 0.0

    def test_rmse_threshold_stored(self, result: SampleEfficiencyResult) -> None:
        assert result.rmse_threshold == 0.5


# ---------------------------------------------------------------------------
# 2. Theory-only curve is constant
# ---------------------------------------------------------------------------


class TestTheoryOnlyCurve:
    """Theory does not use training data, so its RMSE should not change."""

    def test_theory_rmse_constant_across_sample_sizes(
        self, result: SampleEfficiencyResult
    ) -> None:
        theory_curve = next(c for c in result.curves if c.method == "theory_only")
        # All RMSE values should be identical because theory.predict() is
        # independent of training data.
        first = theory_curve.rmse_values[0]
        for rmse in theory_curve.rmse_values[1:]:
            assert rmse == pytest.approx(first, abs=1e-12), (
                f"Theory RMSE should be constant but got {theory_curve.rmse_values}"
            )

    def test_theory_rmse_is_positive(
        self, result: SampleEfficiencyResult
    ) -> None:
        """The linear theory does not perfectly match the true function
        (which has a sin component), so theory RMSE must be > 0."""
        theory_curve = next(c for c in result.curves if c.method == "theory_only")
        assert theory_curve.rmse_values[0] > 0.0


# ---------------------------------------------------------------------------
# 3. Hybrid should improve with more samples
# ---------------------------------------------------------------------------


class TestHybridImprovement:
    """Hybrid RMSE should generally decrease as n grows."""

    def test_hybrid_rmse_decreases_overall(
        self, result: SampleEfficiencyResult
    ) -> None:
        hybrid_curve = next(c for c in result.curves if c.method == "hybrid")
        # The RMSE at the largest sample size should be less than or equal to
        # the RMSE at the smallest sample size. We allow a small tolerance
        # because GP performance can be non-monotonic with very few samples.
        assert hybrid_curve.rmse_values[-1] <= hybrid_curve.rmse_values[0] + 0.1, (
            "Hybrid RMSE at largest n should be no worse than at smallest n "
            f"(got {hybrid_curve.rmse_values})"
        )

    def test_hybrid_rmse_values_are_finite(
        self, result: SampleEfficiencyResult
    ) -> None:
        hybrid_curve = next(c for c in result.curves if c.method == "hybrid")
        for rmse in hybrid_curve.rmse_values:
            assert math.isfinite(rmse)


# ---------------------------------------------------------------------------
# 4. Efficiency ratio computed
# ---------------------------------------------------------------------------


class TestEfficiencyRatio:
    """efficiency_ratio should be a positive float."""

    def test_efficiency_ratio_is_positive(
        self, result: SampleEfficiencyResult
    ) -> None:
        assert result.efficiency_ratio > 0.0

    def test_efficiency_ratio_is_finite(
        self, result: SampleEfficiencyResult
    ) -> None:
        assert math.isfinite(result.efficiency_ratio)

    def test_efficiency_ratio_is_float(
        self, result: SampleEfficiencyResult
    ) -> None:
        assert isinstance(result.efficiency_ratio, float)


# ---------------------------------------------------------------------------
# 5. hybrid_advantage_at_n populated
# ---------------------------------------------------------------------------


class TestHybridAdvantage:
    """hybrid_advantage_at_n dict has correct keys and reasonable values."""

    def test_keys_match_sample_sizes(
        self, result: SampleEfficiencyResult
    ) -> None:
        assert set(result.hybrid_advantage_at_n.keys()) == set(_SMALL_SAMPLE_SIZES)

    def test_values_are_finite_floats(
        self, result: SampleEfficiencyResult
    ) -> None:
        for n, advantage in result.hybrid_advantage_at_n.items():
            assert isinstance(advantage, float), f"n={n}: expected float, got {type(advantage)}"
            assert math.isfinite(advantage), f"n={n}: advantage is not finite"

    def test_advantage_positive_at_some_n(
        self, result: SampleEfficiencyResult
    ) -> None:
        """With a good theory, the hybrid model should outperform the pure GP
        at some sample size (positive advantage means hybrid is better).
        At very small n, GP fitting can be unstable, so we check any n."""
        any_positive = any(
            v > 0.0 for v in result.hybrid_advantage_at_n.values()
        )
        assert any_positive, (
            "Hybrid should have positive advantage over pure GP at some n, "
            f"got {result.hybrid_advantage_at_n}"
        )


# ---------------------------------------------------------------------------
# 6. minimum_n values
# ---------------------------------------------------------------------------


class TestMinimumN:
    """minimum_n_hybrid and minimum_n_pure are either None or valid sizes."""

    def test_minimum_n_hybrid_valid_or_none(
        self, result: SampleEfficiencyResult
    ) -> None:
        val = result.minimum_n_hybrid
        if val is not None:
            assert val in _SMALL_SAMPLE_SIZES

    def test_minimum_n_pure_valid_or_none(
        self, result: SampleEfficiencyResult
    ) -> None:
        val = result.minimum_n_pure
        if val is not None:
            assert val in _SMALL_SAMPLE_SIZES

    def test_minimum_n_hybrid_leq_minimum_n_pure(
        self, result: SampleEfficiencyResult
    ) -> None:
        """If both are found, hybrid should reach the threshold at the same
        or smaller sample size than pure GP."""
        h = result.minimum_n_hybrid
        p = result.minimum_n_pure
        if h is not None and p is not None:
            assert h <= p, (
                f"Expected minimum_n_hybrid ({h}) <= minimum_n_pure ({p})"
            )


# ---------------------------------------------------------------------------
# 7. Deterministic with same seed
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Two runs with the same seed must produce identical results."""

    def test_same_seed_same_results(self, linear_theory: _LinearTheory) -> None:
        bench_a = SampleEfficiencyBenchmark(
            sample_sizes=_SMALL_SAMPLE_SIZES,
            n_test=_SMALL_N_TEST,
            rmse_threshold=0.5,
            seed=99,
        )
        bench_b = SampleEfficiencyBenchmark(
            sample_sizes=_SMALL_SAMPLE_SIZES,
            n_test=_SMALL_N_TEST,
            rmse_threshold=0.5,
            seed=99,
        )

        result_a = bench_a.compare(
            theory=linear_theory,
            data_generator=_true_function,
            X_domain=_DOMAIN_1D,
            noise_std=0.1,
        )
        result_b = bench_b.compare(
            theory=linear_theory,
            data_generator=_true_function,
            X_domain=_DOMAIN_1D,
            noise_std=0.1,
        )

        # Compare all RMSE curves
        for ca, cb in zip(result_a.curves, result_b.curves):
            assert ca.method == cb.method
            assert ca.sample_sizes == cb.sample_sizes
            for va, vb in zip(ca.rmse_values, cb.rmse_values):
                assert va == pytest.approx(vb, abs=1e-15), (
                    f"method={ca.method}: RMSE mismatch {va} vs {vb}"
                )

        assert result_a.efficiency_ratio == pytest.approx(
            result_b.efficiency_ratio, abs=1e-15
        )
        assert result_a.hybrid_advantage_at_n == result_b.hybrid_advantage_at_n
        assert result_a.minimum_n_hybrid == result_b.minimum_n_hybrid
        assert result_a.minimum_n_pure == result_b.minimum_n_pure

    def test_different_seed_different_results(
        self, linear_theory: _LinearTheory
    ) -> None:
        bench_a = SampleEfficiencyBenchmark(
            sample_sizes=_SMALL_SAMPLE_SIZES,
            n_test=_SMALL_N_TEST,
            seed=42,
        )
        bench_b = SampleEfficiencyBenchmark(
            sample_sizes=_SMALL_SAMPLE_SIZES,
            n_test=_SMALL_N_TEST,
            seed=123,
        )

        result_a = bench_a.compare(
            theory=linear_theory,
            data_generator=_true_function,
            X_domain=_DOMAIN_1D,
            noise_std=0.1,
        )
        result_b = bench_b.compare(
            theory=linear_theory,
            data_generator=_true_function,
            X_domain=_DOMAIN_1D,
            noise_std=0.1,
        )

        # At least one curve should differ (different random training data)
        any_different = False
        for ca, cb in zip(result_a.curves, result_b.curves):
            if ca.method == "theory_only":
                # Theory-only is deterministic and test set differs by seed,
                # so RMSE may differ because the test set is different.
                continue
            for va, vb in zip(ca.rmse_values, cb.rmse_values):
                if abs(va - vb) > 1e-10:
                    any_different = True
                    break
            if any_different:
                break
        assert any_different, "Different seeds should produce different results"


# ---------------------------------------------------------------------------
# 8. _ZeroTheory returns zeros
# ---------------------------------------------------------------------------


class TestZeroTheory:
    """Verify the _ZeroTheory helper model works correctly."""

    def test_predict_returns_zeros(self) -> None:
        theory = _ZeroTheory()
        X = [[1.0], [2.0], [3.0], [-5.0, 10.0]]
        result = theory.predict(X)
        assert result == [0.0, 0.0, 0.0, 0.0]

    def test_predict_empty_input(self) -> None:
        theory = _ZeroTheory()
        result = theory.predict([])
        assert result == []

    def test_n_parameters_is_zero(self) -> None:
        theory = _ZeroTheory()
        assert theory.n_parameters() == 0

    def test_parameter_names_is_empty(self) -> None:
        theory = _ZeroTheory()
        assert theory.parameter_names() == []

    def test_is_theory_model_subclass(self) -> None:
        theory = _ZeroTheory()
        assert isinstance(theory, TheoryModel)


# ---------------------------------------------------------------------------
# 9. Custom parameters
# ---------------------------------------------------------------------------


class TestCustomParameters:
    """Different sample_sizes, n_test, rmse_threshold work correctly."""

    def test_custom_sample_sizes(self, linear_theory: _LinearTheory) -> None:
        custom_sizes = [3, 7, 15]
        bench = SampleEfficiencyBenchmark(
            sample_sizes=custom_sizes,
            n_test=10,
            rmse_threshold=1.0,
            seed=42,
        )
        result = bench.compare(
            theory=linear_theory,
            data_generator=_true_function,
            X_domain=_DOMAIN_1D,
            noise_std=0.1,
        )
        for curve in result.curves:
            assert curve.sample_sizes == custom_sizes

    def test_custom_n_test(self, linear_theory: _LinearTheory) -> None:
        """Using a different n_test should still produce valid results."""
        bench = SampleEfficiencyBenchmark(
            sample_sizes=[5, 10],
            n_test=5,
            rmse_threshold=0.5,
            seed=42,
        )
        result = bench.compare(
            theory=linear_theory,
            data_generator=_true_function,
            X_domain=_DOMAIN_1D,
            noise_std=0.1,
        )
        assert isinstance(result, SampleEfficiencyResult)
        assert len(result.curves) == 3

    def test_high_rmse_threshold_all_pass(
        self, linear_theory: _LinearTheory
    ) -> None:
        """With a very high threshold, minimum_n should be the smallest size."""
        bench = SampleEfficiencyBenchmark(
            sample_sizes=_SMALL_SAMPLE_SIZES,
            n_test=_SMALL_N_TEST,
            rmse_threshold=100.0,
            seed=42,
        )
        result = bench.compare(
            theory=linear_theory,
            data_generator=_true_function,
            X_domain=_DOMAIN_1D,
            noise_std=0.1,
        )
        # With a very generous threshold, both methods should reach it early
        assert result.minimum_n_hybrid == _SMALL_SAMPLE_SIZES[0]
        assert result.minimum_n_pure == _SMALL_SAMPLE_SIZES[0]
        assert result.rmse_threshold == 100.0

    def test_impossible_rmse_threshold_returns_none(
        self, linear_theory: _LinearTheory
    ) -> None:
        """With an impossibly low threshold, minimum_n should be None."""
        bench = SampleEfficiencyBenchmark(
            sample_sizes=_SMALL_SAMPLE_SIZES,
            n_test=_SMALL_N_TEST,
            rmse_threshold=1e-15,
            seed=42,
        )
        result = bench.compare(
            theory=linear_theory,
            data_generator=_true_function,
            X_domain=_DOMAIN_1D,
            noise_std=0.05,
        )
        # No model can achieve essentially zero RMSE with noisy data
        assert result.minimum_n_hybrid is None
        assert result.minimum_n_pure is None

    def test_default_parameters(self) -> None:
        """Default construction uses expected default values."""
        bench = SampleEfficiencyBenchmark()
        assert bench._sample_sizes == [5, 10, 20, 50, 100]
        assert bench._n_test == 50
        assert bench._rmse_threshold == 0.5
        assert bench._seed == 42

    def test_single_sample_size(self, linear_theory: _LinearTheory) -> None:
        """Benchmark works with a single sample size."""
        bench = SampleEfficiencyBenchmark(
            sample_sizes=[10],
            n_test=10,
            rmse_threshold=0.5,
            seed=42,
        )
        result = bench.compare(
            theory=linear_theory,
            data_generator=_true_function,
            X_domain=_DOMAIN_1D,
            noise_std=0.1,
        )
        assert len(result.curves) == 3
        for curve in result.curves:
            assert len(curve.sample_sizes) == 1
            assert len(curve.rmse_values) == 1
        assert list(result.hybrid_advantage_at_n.keys()) == [10]

    def test_zero_noise(self, linear_theory: _LinearTheory) -> None:
        """With zero noise, results should still be valid."""
        bench = SampleEfficiencyBenchmark(
            sample_sizes=[5, 10],
            n_test=10,
            rmse_threshold=0.5,
            seed=42,
        )
        result = bench.compare(
            theory=linear_theory,
            data_generator=_true_function,
            X_domain=_DOMAIN_1D,
            noise_std=0.0,
        )
        assert isinstance(result, SampleEfficiencyResult)
        for curve in result.curves:
            for rmse in curve.rmse_values:
                assert math.isfinite(rmse)


# ---------------------------------------------------------------------------
# Dataclass structure tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    """SampleEfficiencyCurve and SampleEfficiencyResult dataclass behaviour."""

    def test_curve_default_factory(self) -> None:
        curve = SampleEfficiencyCurve(method="test")
        assert curve.method == "test"
        assert curve.sample_sizes == []
        assert curve.rmse_values == []

    def test_curve_with_data(self) -> None:
        curve = SampleEfficiencyCurve(
            method="hybrid",
            sample_sizes=[5, 10],
            rmse_values=[0.3, 0.2],
        )
        assert curve.method == "hybrid"
        assert curve.sample_sizes == [5, 10]
        assert curve.rmse_values == [0.3, 0.2]

    def test_result_fields(self, result: SampleEfficiencyResult) -> None:
        """SampleEfficiencyResult has all expected fields."""
        assert hasattr(result, "curves")
        assert hasattr(result, "efficiency_ratio")
        assert hasattr(result, "hybrid_advantage_at_n")
        assert hasattr(result, "minimum_n_hybrid")
        assert hasattr(result, "minimum_n_pure")
        assert hasattr(result, "rmse_threshold")


# ---------------------------------------------------------------------------
# Static / internal method tests
# ---------------------------------------------------------------------------


class TestInternalMethods:
    """Test static helper methods on SampleEfficiencyBenchmark."""

    def test_compute_rmse_identical(self) -> None:
        rmse = SampleEfficiencyBenchmark._compute_rmse(
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]
        )
        assert rmse == pytest.approx(0.0, abs=1e-15)

    def test_compute_rmse_known_value(self) -> None:
        # y_true = [0, 0], y_pred = [1, 1] -> RMSE = 1.0
        rmse = SampleEfficiencyBenchmark._compute_rmse([0.0, 0.0], [1.0, 1.0])
        assert rmse == pytest.approx(1.0, abs=1e-12)

    def test_compute_rmse_empty(self) -> None:
        rmse = SampleEfficiencyBenchmark._compute_rmse([], [])
        assert rmse == 0.0

    def test_find_minimum_n_found(self) -> None:
        curve = SampleEfficiencyCurve(
            method="test",
            sample_sizes=[5, 10, 20],
            rmse_values=[0.8, 0.4, 0.2],
        )
        result = SampleEfficiencyBenchmark._find_minimum_n(curve, 0.5)
        assert result == 10

    def test_find_minimum_n_first_already_below(self) -> None:
        curve = SampleEfficiencyCurve(
            method="test",
            sample_sizes=[5, 10],
            rmse_values=[0.3, 0.1],
        )
        result = SampleEfficiencyBenchmark._find_minimum_n(curve, 0.5)
        assert result == 5

    def test_find_minimum_n_none_below(self) -> None:
        curve = SampleEfficiencyCurve(
            method="test",
            sample_sizes=[5, 10],
            rmse_values=[0.8, 0.6],
        )
        result = SampleEfficiencyBenchmark._find_minimum_n(curve, 0.5)
        assert result is None

    def test_compute_efficiency_ratio_empty_curves(self) -> None:
        h = SampleEfficiencyCurve(method="hybrid")
        p = SampleEfficiencyCurve(method="pure_gp")
        ratio = SampleEfficiencyBenchmark._compute_efficiency_ratio(h, p)
        assert ratio == 1.0

    def test_compute_efficiency_ratio_pure_never_reaches(self) -> None:
        """If pure GP never beats the target, ratio uses the last sample size."""
        h = SampleEfficiencyCurve(
            method="hybrid",
            sample_sizes=[5, 10, 20],
            rmse_values=[0.5, 0.3, 0.1],
        )
        p = SampleEfficiencyCurve(
            method="pure_gp",
            sample_sizes=[5, 10, 20],
            rmse_values=[0.9, 0.5, 0.2],
        )
        # Target RMSE = hybrid's last = 0.1
        # Pure GP never reaches 0.1, so n_pure defaults to last = 20
        # Ratio = 20 / 20 = 1.0
        ratio = SampleEfficiencyBenchmark._compute_efficiency_ratio(h, p)
        assert ratio == pytest.approx(1.0)

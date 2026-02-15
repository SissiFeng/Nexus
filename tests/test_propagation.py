"""Tests for the uncertainty propagation engine."""

import math

from optimization_copilot.uncertainty.types import (
    MeasurementWithUncertainty,
    PropagationMethod,
    PropagationResult,
    UncertaintyBudget,
)
from optimization_copilot.uncertainty.propagation import UncertaintyPropagator


# ── Helpers ───────────────────────────────────────────────────────────────


def _m(
    value: float,
    variance: float,
    source: str = "kpi",
    confidence: float = 0.9,
    metadata: dict | None = None,
) -> MeasurementWithUncertainty:
    """Shorthand factory for test measurements."""
    return MeasurementWithUncertainty(
        value=value,
        variance=variance,
        source=source,
        confidence=confidence,
        metadata=metadata or {},
    )


# ── Propagator fixture ────────────────────────────────────────────────────

prop = UncertaintyPropagator()


# ══════════════════════════════════════════════════════════════════════════
#  LINEAR PROPAGATION
# ══════════════════════════════════════════════════════════════════════════


class TestLinearPropagationBasic:
    """Linear propagation: obj = sum(w_i * kpi_i)."""

    def test_single_kpi_weight_one(self):
        """Single KPI with weight=1 -> variance passes through unchanged."""
        m = _m(value=5.0, variance=0.25, source="kpi_a")
        result = prop.linear_propagation([m], [1.0])
        assert result.objective_value == 5.0
        assert result.objective_variance == 0.25
        assert result.method == PropagationMethod.LINEAR

    def test_two_kpis_equal_weights(self):
        """Two KPIs with equal weights -> correct weighted variance."""
        m1 = _m(value=3.0, variance=0.1, source="kpi_a")
        m2 = _m(value=7.0, variance=0.2, source="kpi_b")
        result = prop.linear_propagation([m1, m2], [1.0, 1.0])
        assert result.objective_value == 10.0
        # var = 1^2 * 0.1 + 1^2 * 0.2 = 0.3
        assert abs(result.objective_variance - 0.3) < 1e-12

    def test_zero_variance_kpi(self):
        """KPI with zero variance contributes nothing to total variance."""
        m1 = _m(value=2.0, variance=0.0, source="exact")
        m2 = _m(value=3.0, variance=0.5, source="noisy")
        result = prop.linear_propagation([m1, m2], [1.0, 1.0])
        assert result.objective_value == 5.0
        assert abs(result.objective_variance - 0.5) < 1e-12
        assert result.budget.dominant_source == "noisy"

    def test_unequal_weights(self):
        """Larger weight dominates variance contribution."""
        m1 = _m(value=1.0, variance=1.0, source="kpi_a")
        m2 = _m(value=1.0, variance=1.0, source="kpi_b")
        result = prop.linear_propagation([m1, m2], [3.0, 1.0])
        # var = 9 * 1.0 + 1 * 1.0 = 10.0
        assert abs(result.objective_variance - 10.0) < 1e-12
        assert result.budget.dominant_source == "kpi_a"
        # kpi_a contributes 9/10 = 0.9
        assert abs(result.budget.fraction("kpi_a") - 0.9) < 1e-12

    def test_negative_weight(self):
        """Negative weight squares correctly."""
        m = _m(value=4.0, variance=1.0, source="kpi")
        result = prop.linear_propagation([m], [-2.0])
        assert result.objective_value == -8.0
        # var = (-2)^2 * 1.0 = 4.0
        assert abs(result.objective_variance - 4.0) < 1e-12

    def test_weight_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        m = _m(value=1.0, variance=0.1, source="kpi")
        try:
            prop.linear_propagation([m], [1.0, 2.0])
            assert False, "Expected ValueError"
        except ValueError:
            pass


class TestLinearPropagationMetadata:
    """Linear propagation metadata and budget."""

    def test_min_confidence(self):
        """Metadata should report the minimum confidence across KPIs."""
        m1 = _m(value=1.0, variance=0.1, source="a", confidence=0.95)
        m2 = _m(value=2.0, variance=0.1, source="b", confidence=0.60)
        result = prop.linear_propagation([m1, m2], [1.0, 1.0])
        meta = prop._aggregate_metadata(
            [m1, m2], result.budget.contributions
        )
        assert meta["min_confidence"] == 0.60
        assert abs(meta["mean_confidence"] - 0.775) < 1e-12

    def test_unreliable_kpis(self):
        """Unreliable KPIs should be listed in metadata."""
        # is_reliable = confidence >= 0.5 and relative_uncertainty < 0.5
        # value=1.0, variance=1.0 => std=1.0, CV=1.0 => not reliable
        m1 = _m(value=1.0, variance=1.0, source="bad", confidence=0.9)
        m2 = _m(value=10.0, variance=0.01, source="good", confidence=0.9)
        meta = prop._aggregate_metadata(
            [m1, m2], {"bad": 1.0, "good": 0.01}
        )
        assert "bad" in meta["unreliable_kpis"]
        assert "good" not in meta["unreliable_kpis"]

    def test_budget_dominant_source(self):
        """UncertaintyBudget.dominant_source is the largest contributor."""
        m1 = _m(value=1.0, variance=0.1, source="minor")
        m2 = _m(value=1.0, variance=5.0, source="major")
        result = prop.linear_propagation([m1, m2], [1.0, 1.0])
        assert result.budget.dominant_source == "major"

    def test_to_observation_with_noise(self):
        """PropagationResult.to_observation_with_noise() round-trips."""
        m = _m(value=3.0, variance=0.5, source="kpi")
        result = prop.linear_propagation([m], [2.0])
        obs = result.to_observation_with_noise()
        assert obs.objective_value == 6.0
        # var = 4 * 0.5 = 2.0
        assert abs(obs.noise_variance - 2.0) < 1e-12
        assert obs.uncertainty_budget is not None
        assert obs.metadata["propagation_method"] == "linear"

    def test_kpi_details_var_fraction(self):
        """Each KPI detail should have a var_fraction summing to 1."""
        m1 = _m(value=1.0, variance=0.3, source="a")
        m2 = _m(value=2.0, variance=0.7, source="b")
        result = prop.linear_propagation([m1, m2], [1.0, 1.0])
        total_frac = sum(d["var_fraction"] for d in result.kpi_details)
        assert abs(total_frac - 1.0) < 1e-12

    def test_quality_flags_aggregation(self):
        """Quality flags from all KPIs should be merged."""
        m1 = _m(
            value=1.0, variance=0.1, source="a",
            metadata={"quality_flags": ["noisy", "drift"]},
        )
        m2 = _m(
            value=2.0, variance=0.1, source="b",
            metadata={"quality_flags": ["outlier"]},
        )
        meta = prop._aggregate_metadata([m1, m2], {"a": 0.1, "b": 0.1})
        assert set(meta["all_quality_flags"]) == {"noisy", "drift", "outlier"}

    def test_quality_flags_deduplication(self):
        """Duplicate quality flags should appear only once."""
        m1 = _m(
            value=1.0, variance=0.1, source="a",
            metadata={"quality_flags": ["noisy"]},
        )
        m2 = _m(
            value=2.0, variance=0.1, source="b",
            metadata={"quality_flags": ["noisy"]},
        )
        meta = prop._aggregate_metadata([m1, m2], {"a": 0.1, "b": 0.1})
        assert meta["all_quality_flags"] == ["noisy"]


# ══════════════════════════════════════════════════════════════════════════
#  NONLINEAR (DELTA METHOD) PROPAGATION
# ══════════════════════════════════════════════════════════════════════════


class TestNonlinearPropagation:
    """Delta method: sigma^2_obj ~ J^T Sigma J."""

    def test_linear_matches_linear_propagation(self):
        """Linear objective through delta method should match linear_propagation."""
        m1 = _m(value=3.0, variance=0.1, source="a")
        m2 = _m(value=7.0, variance=0.2, source="b")
        weights = [2.0, 0.5]

        linear_result = prop.linear_propagation([m1, m2], weights)
        nonlinear_result = prop.nonlinear_propagation(
            [m1, m2],
            objective_func=lambda a, b: 2.0 * a + 0.5 * b,
        )

        assert abs(
            nonlinear_result.objective_value - linear_result.objective_value
        ) < 1e-6
        assert abs(
            nonlinear_result.objective_variance - linear_result.objective_variance
        ) < 1e-6

    def test_quadratic_function(self):
        """f(x) = x^2 -> J = 2x, var = (2x)^2 * sigma^2_x."""
        m = _m(value=3.0, variance=0.5, source="x")
        result = prop.nonlinear_propagation(
            [m], objective_func=lambda x: x ** 2
        )
        expected_var = (2 * 3.0) ** 2 * 0.5  # 36 * 0.5 = 18.0
        assert abs(result.objective_value - 9.0) < 1e-6
        assert abs(result.objective_variance - expected_var) < 1e-3

    def test_product_function(self):
        """f(a, b) = a * b -> J = [b, a], var = b^2*var_a + a^2*var_b."""
        m_a = _m(value=4.0, variance=0.1, source="a")
        m_b = _m(value=5.0, variance=0.2, source="b")
        result = prop.nonlinear_propagation(
            [m_a, m_b],
            objective_func=lambda a, b: a * b,
        )
        expected_var = 5.0 ** 2 * 0.1 + 4.0 ** 2 * 0.2  # 2.5 + 3.2 = 5.7
        assert abs(result.objective_value - 20.0) < 1e-6
        assert abs(result.objective_variance - expected_var) < 1e-3

    def test_custom_jacobian_matches_numerical(self):
        """Analytic Jacobian should give same result as numerical."""
        m = _m(value=3.0, variance=0.5, source="x")

        result_num = prop.nonlinear_propagation(
            [m], objective_func=lambda x: x ** 2
        )
        result_ana = prop.nonlinear_propagation(
            [m],
            objective_func=lambda x: x ** 2,
            jacobian_func=lambda x: [2 * x],
        )

        assert abs(
            result_ana.objective_variance - result_num.objective_variance
        ) < 1e-6

    def test_zero_variance_nonlinear(self):
        """Zero variance input -> zero propagated variance."""
        m = _m(value=2.0, variance=0.0, source="exact")
        result = prop.nonlinear_propagation(
            [m], objective_func=lambda x: x ** 3
        )
        assert result.objective_value == 8.0
        assert abs(result.objective_variance) < 1e-12

    def test_method_is_delta(self):
        """Result should report DELTA method."""
        m = _m(value=1.0, variance=0.1, source="x")
        result = prop.nonlinear_propagation(
            [m], objective_func=lambda x: x
        )
        assert result.method == PropagationMethod.DELTA

    def test_kpi_details_include_jacobian(self):
        """KPI details should include the Jacobian value."""
        m = _m(value=2.0, variance=0.1, source="x")
        result = prop.nonlinear_propagation(
            [m], objective_func=lambda x: 3.0 * x
        )
        assert len(result.kpi_details) == 1
        assert abs(result.kpi_details[0]["jacobian"] - 3.0) < 1e-4


# ══════════════════════════════════════════════════════════════════════════
#  NUMERICAL JACOBIAN
# ══════════════════════════════════════════════════════════════════════════


class TestNumericalJacobian:
    """Tests for the central-difference Jacobian helper."""

    def test_linear_function(self):
        """J of f(x) = 3x + 1 should be [3]."""
        jac = UncertaintyPropagator._numerical_jacobian(
            lambda x: 3 * x + 1, [5.0]
        )
        assert abs(jac[0] - 3.0) < 1e-6

    def test_multivariate(self):
        """J of f(a, b) = 2a + 3b should be [2, 3]."""
        jac = UncertaintyPropagator._numerical_jacobian(
            lambda a, b: 2 * a + 3 * b, [1.0, 2.0]
        )
        assert abs(jac[0] - 2.0) < 1e-6
        assert abs(jac[1] - 3.0) < 1e-6

    def test_nonlinear_at_point(self):
        """J of f(x) = x^3 at x=2 should be 12."""
        jac = UncertaintyPropagator._numerical_jacobian(
            lambda x: x ** 3, [2.0]
        )
        assert abs(jac[0] - 12.0) < 1e-4

    def test_custom_eps(self):
        """Custom eps should still produce correct Jacobian."""
        jac = UncertaintyPropagator._numerical_jacobian(
            lambda x: x ** 2, [3.0], eps=1e-4
        )
        assert abs(jac[0] - 6.0) < 1e-3


# ══════════════════════════════════════════════════════════════════════════
#  MONTE CARLO PROPAGATION
# ══════════════════════════════════════════════════════════════════════════


class TestMonteCarloPropagation:
    """Sampling-based uncertainty propagation."""

    def test_linear_matches_analytic(self):
        """MC for linear objective should approximate linear propagation."""
        m1 = _m(value=3.0, variance=0.1, source="a")
        m2 = _m(value=7.0, variance=0.2, source="b")

        linear = prop.linear_propagation([m1, m2], [1.0, 1.0])
        mc = prop.monte_carlo_propagation(
            [m1, m2],
            objective_func=lambda a, b: a + b,
            n_samples=50_000,
            seed=42,
        )

        # Mean should be close (within ~1%).
        assert abs(mc.objective_value - linear.objective_value) < 0.05
        # Variance within ~5%.
        assert abs(mc.objective_variance - linear.objective_variance) < 0.05

    def test_quadratic_matches_delta(self):
        """MC for quadratic should approximate delta method."""
        m = _m(value=3.0, variance=0.5, source="x")

        delta = prop.nonlinear_propagation(
            [m], objective_func=lambda x: x ** 2
        )
        mc = prop.monte_carlo_propagation(
            [m],
            objective_func=lambda x: x ** 2,
            n_samples=50_000,
            seed=123,
        )

        # MC objective_value is E[x^2] = mu^2 + sigma^2 = 9 + 0.5 = 9.5
        # Delta gives mu^2 = 9.0 (first-order, no bias correction).
        # We check variance is within tolerance.
        assert abs(mc.objective_variance - delta.objective_variance) / delta.objective_variance < 0.15

    def test_reproducible_with_same_seed(self):
        """Same seed should give identical results."""
        m = _m(value=5.0, variance=1.0, source="x")
        r1 = prop.monte_carlo_propagation(
            [m], lambda x: x ** 2, seed=99
        )
        r2 = prop.monte_carlo_propagation(
            [m], lambda x: x ** 2, seed=99
        )
        assert r1.objective_value == r2.objective_value
        assert r1.objective_variance == r2.objective_variance

    def test_different_seeds_differ(self):
        """Different seeds give similar but not identical results."""
        m = _m(value=5.0, variance=1.0, source="x")
        r1 = prop.monte_carlo_propagation(
            [m], lambda x: x ** 2, seed=1
        )
        r2 = prop.monte_carlo_propagation(
            [m], lambda x: x ** 2, seed=2
        )
        # Should be close but not identical.
        assert r1.objective_value != r2.objective_value
        # Means should still be similar (within 5%).
        assert abs(r1.objective_value - r2.objective_value) / abs(r1.objective_value) < 0.05

    def test_handles_exceptions_in_objective(self):
        """MC should gracefully skip samples where objective raises."""
        m = _m(value=1.0, variance=0.01, source="x")

        call_count = {"n": 0}

        def flaky_func(x):
            call_count["n"] += 1
            if call_count["n"] % 10 == 0:
                raise ValueError("intermittent failure")
            return x * 2

        result = prop.monte_carlo_propagation(
            [m], flaky_func, n_samples=1000, seed=42
        )
        # Should still produce a valid result.
        assert math.isfinite(result.objective_value)
        assert math.isfinite(result.objective_variance)
        assert result.method == PropagationMethod.MONTE_CARLO

    def test_mc_method_enum(self):
        """Method should be MONTE_CARLO."""
        m = _m(value=1.0, variance=0.1, source="x")
        result = prop.monte_carlo_propagation(
            [m], lambda x: x, n_samples=100
        )
        assert result.method == PropagationMethod.MONTE_CARLO

    def test_mc_budget_has_contributions(self):
        """MC budget should have per-source contributions."""
        m1 = _m(value=1.0, variance=0.5, source="a")
        m2 = _m(value=2.0, variance=0.1, source="b")
        result = prop.monte_carlo_propagation(
            [m1, m2],
            lambda a, b: a + b,
            n_samples=5000,
            seed=42,
        )
        assert "a" in result.budget.contributions
        assert "b" in result.budget.contributions
        # 'a' has larger variance, should dominate.
        assert result.budget.contributions["a"] > result.budget.contributions["b"]


# ══════════════════════════════════════════════════════════════════════════
#  EDGE CASES
# ══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_measurements_linear(self):
        """Empty list should return zero-valued result."""
        result = prop.linear_propagation([], [])
        assert result.objective_value == 0.0
        assert result.objective_variance == 0.0
        assert result.kpi_details == []

    def test_empty_measurements_nonlinear(self):
        """Empty list for nonlinear should return zero-valued result."""
        result = prop.nonlinear_propagation(
            [], objective_func=lambda: 0.0
        )
        assert result.objective_value == 0.0
        assert result.objective_variance == 0.0

    def test_empty_measurements_mc(self):
        """Empty list for MC should return zero-valued result."""
        result = prop.monte_carlo_propagation(
            [], objective_func=lambda: 0.0
        )
        assert result.objective_value == 0.0
        assert result.objective_variance == 0.0

    def test_single_measurement_linear(self):
        """Single measurement with weight 1 is a trivial case."""
        m = _m(value=42.0, variance=0.001, source="solo")
        result = prop.linear_propagation([m], [1.0])
        assert result.objective_value == 42.0
        assert abs(result.objective_variance - 0.001) < 1e-12
        assert result.budget.dominant_source == "solo"

    def test_very_large_variance(self):
        """Very large variance should propagate correctly."""
        big_var = 1e12
        m = _m(value=1.0, variance=big_var, source="noisy")
        result = prop.linear_propagation([m], [1.0])
        assert abs(result.objective_variance - big_var) < 1.0

    def test_very_small_variance(self):
        """Near-exact measurement (tiny variance)."""
        tiny_var = 1e-20
        m = _m(value=100.0, variance=tiny_var, source="precise")
        result = prop.linear_propagation([m], [1.0])
        assert abs(result.objective_variance - tiny_var) < 1e-30

    def test_aggregate_metadata_empty(self):
        """Metadata aggregation with no measurements."""
        meta = prop._aggregate_metadata([], {})
        assert meta["min_confidence"] == 1.0
        assert meta["mean_confidence"] == 1.0
        assert meta["unreliable_kpis"] == []
        assert meta["all_quality_flags"] == []
        assert meta["uncertainty_budget"] == {}

    def test_aggregate_metadata_no_quality_flags(self):
        """Measurements without quality_flags in metadata."""
        m = _m(value=1.0, variance=0.1, source="a")
        meta = prop._aggregate_metadata([m], {"a": 0.1})
        assert meta["all_quality_flags"] == []

    def test_duplicate_source_names_linear(self):
        """Two measurements with the same source name accumulate variance."""
        m1 = _m(value=1.0, variance=0.3, source="same")
        m2 = _m(value=2.0, variance=0.7, source="same")
        result = prop.linear_propagation([m1, m2], [1.0, 1.0])
        # Both contribute to "same" source: 0.3 + 0.7 = 1.0
        assert abs(result.budget.contributions["same"] - 1.0) < 1e-12
        assert result.budget.dominant_source == "same"

    def test_zero_weight_contributes_nothing(self):
        """Weight of zero should contribute nothing to variance or value."""
        m1 = _m(value=100.0, variance=999.0, source="ignored")
        m2 = _m(value=5.0, variance=0.1, source="used")
        result = prop.linear_propagation([m1, m2], [0.0, 1.0])
        assert result.objective_value == 5.0
        # var = 0^2 * 999 + 1^2 * 0.1 = 0.1
        assert abs(result.objective_variance - 0.1) < 1e-12

    def test_propagation_result_to_dict(self):
        """PropagationResult.to_dict() should serialize correctly."""
        m = _m(value=3.0, variance=0.5, source="kpi")
        result = prop.linear_propagation([m], [1.0])
        d = result.to_dict()
        assert d["method"] == "linear"
        assert d["objective_value"] == 3.0
        assert abs(d["objective_variance"] - 0.5) < 1e-12
        assert "budget" in d

    def test_many_kpis(self):
        """Propagation with many KPIs should work correctly."""
        n = 50
        measurements = [
            _m(value=float(i), variance=0.01 * i, source=f"kpi_{i}")
            for i in range(1, n + 1)
        ]
        weights = [1.0] * n
        result = prop.linear_propagation(measurements, weights)
        expected_value = sum(range(1, n + 1))
        expected_var = sum(0.01 * i for i in range(1, n + 1))
        assert abs(result.objective_value - expected_value) < 1e-6
        assert abs(result.objective_variance - expected_var) < 1e-6

    def test_nonlinear_with_constant_function(self):
        """Constant objective function -> zero variance regardless of inputs."""
        m = _m(value=5.0, variance=10.0, source="x")
        result = prop.nonlinear_propagation(
            [m], objective_func=lambda x: 42.0
        )
        assert result.objective_value == 42.0
        assert abs(result.objective_variance) < 1e-6

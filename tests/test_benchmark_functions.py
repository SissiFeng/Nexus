"""Comprehensive tests for optimization_copilot.benchmark.functions.

Covers all benchmark functions, the BenchmarkFunction dataclass,
the BENCHMARK_SUITE registry, and the make_spec helper.
"""

from __future__ import annotations

import math
import statistics

import pytest

from optimization_copilot.benchmark.functions import (
    BENCHMARK_SUITE,
    BenchmarkFunction,
    ackley10,
    bohachevsky2,
    branin,
    constrained_branin,
    dixon_price10,
    get_benchmark,
    griewank10,
    hartmann3,
    hartmann6,
    levy,
    list_benchmarks,
    make_spec,
    michalewicz10,
    multifidelity_branin,
    noisy_hartmann6,
    rastrigin10,
    rosenbrock,
    schwefel10,
    sphere5,
    styblinski_tang10,
    zakharov10,
    zdt1,
)


# ── BenchmarkFunction dataclass tests ──────────────────────────────────


class TestBenchmarkFunctionDataclass:
    """Tests for the BenchmarkFunction dataclass."""

    def test_creation_with_all_fields(self):
        """BenchmarkFunction can be created with all required fields."""
        bf = BenchmarkFunction(
            name="test_fn",
            evaluate=lambda p: {"objective": p["x1"]},
            parameter_specs=[{"name": "x1", "type": "continuous", "bounds": [0, 1]}],
            known_optimum={"objective": 0.0},
            optimal_params={"x1": 0.0},
            metadata={"dimensionality": 1},
        )
        assert bf.name == "test_fn"
        assert bf.known_optimum == {"objective": 0.0}
        assert bf.optimal_params == {"x1": 0.0}
        assert bf.metadata == {"dimensionality": 1}
        assert len(bf.parameter_specs) == 1

    def test_call_delegates_to_evaluate(self):
        """__call__ should delegate to the evaluate function."""
        def eval_fn(params):
            return {"objective": params["x1"] * 2}

        bf = BenchmarkFunction(
            name="double",
            evaluate=eval_fn,
            parameter_specs=[{"name": "x1", "type": "continuous", "bounds": [0, 10]}],
            known_optimum={"objective": 0.0},
            optimal_params={"x1": 0.0},
        )
        result = bf({"x1": 5.0})
        assert result == {"objective": 10.0}
        # Direct evaluate call should give same result
        assert bf.evaluate({"x1": 5.0}) == result

    def test_metadata_defaults_to_empty_dict(self):
        """metadata field should default to an empty dict."""
        bf = BenchmarkFunction(
            name="no_meta",
            evaluate=lambda p: {"objective": 0.0},
            parameter_specs=[],
            known_optimum={"objective": 0.0},
            optimal_params=None,
        )
        assert bf.metadata == {}

    def test_optimal_params_can_be_none(self):
        """optimal_params may be None (e.g. Pareto front problems)."""
        bf = BenchmarkFunction(
            name="pareto",
            evaluate=lambda p: {"f1": 0.0, "f2": 0.0},
            parameter_specs=[],
            known_optimum={"f1": 0.0, "f2": 0.0},
            optimal_params=None,
        )
        assert bf.optimal_params is None

    def test_metadata_access(self):
        """Metadata entries are accessible by key."""
        bf = BenchmarkFunction(
            name="meta_test",
            evaluate=lambda p: {"objective": 0.0},
            parameter_specs=[],
            known_optimum={"objective": 0.0},
            optimal_params=None,
            metadata={"dimensionality": 5, "difficulty": "hard"},
        )
        assert bf.metadata["dimensionality"] == 5
        assert bf.metadata["difficulty"] == "hard"


# ── branin tests ──────────────────────────────────────────────────────


class TestBranin:
    """Tests for the Branin-Hoo function."""

    def test_at_known_optimum_pi_12(self):
        """branin at (-pi, 12.275) should be close to the global minimum."""
        result = branin({"x1": -math.pi, "x2": 12.275})
        assert "objective" in result
        assert result["objective"] == pytest.approx(0.397887, abs=1e-3)

    def test_at_known_optimum_pi_2(self):
        """branin at (pi, 2.275) should also be close to the global minimum."""
        result = branin({"x1": math.pi, "x2": 2.275})
        assert result["objective"] == pytest.approx(0.397887, abs=1e-3)

    def test_at_known_optimum_third(self):
        """branin at (9.42478, 2.475) should be close to the global minimum."""
        result = branin({"x1": 9.42478, "x2": 2.475})
        assert result["objective"] == pytest.approx(0.397887, abs=1e-2)

    def test_at_origin(self):
        """branin at (0, 0) should return a known value."""
        result = branin({"x1": 0.0, "x2": 0.0})
        # Manual calculation: a*(x2 - b*x1^2 + c*x1 - r)^2 + s*(1-t)*cos(x1) + s
        # = 1*(0 - 0 + 0 - 6)^2 + 10*(1 - 1/(8*pi))*cos(0) + 10
        # = 36 + 10*(1 - 1/(8*pi))*1 + 10
        expected = 36.0 + 10.0 * (1.0 - 1.0 / (8.0 * math.pi)) + 10.0
        assert result["objective"] == pytest.approx(expected, rel=1e-10)

    def test_returns_dict_with_objective_key(self):
        """branin should return a dict with 'objective' key."""
        result = branin({"x1": 0.0, "x2": 0.0})
        assert isinstance(result, dict)
        assert "objective" in result
        assert len(result) == 1

    def test_within_expected_range(self):
        """branin should return values in a plausible range within the domain."""
        # The minimum is ~0.3979. On the domain [-5,10]x[0,15] the function
        # stays bounded.
        for x1 in [-5.0, 0.0, 5.0, 10.0]:
            for x2 in [0.0, 7.5, 15.0]:
                result = branin({"x1": x1, "x2": x2})
                assert result["objective"] >= 0.0, (
                    f"branin should be non-negative, got {result['objective']} at ({x1}, {x2})"
                )


# ── hartmann3 tests ──────────────────────────────────────────────────


class TestHartmann3:
    """Tests for the Hartmann 3D function."""

    def test_at_known_optimum(self):
        """hartmann3 at the known optimal params should be close to -3.86278."""
        params = {"x1": 0.114614, "x2": 0.555649, "x3": 0.852547}
        result = hartmann3(params)
        assert result["objective"] == pytest.approx(-3.86278, abs=1e-3)

    def test_at_all_zeros(self):
        """hartmann3 at (0, 0, 0) should return a known finite value."""
        result = hartmann3({"x1": 0.0, "x2": 0.0, "x3": 0.0})
        assert isinstance(result["objective"], float)
        assert math.isfinite(result["objective"])

    def test_returns_dict_with_objective_key(self):
        """hartmann3 should return a dict with 'objective' key."""
        result = hartmann3({"x1": 0.5, "x2": 0.5, "x3": 0.5})
        assert isinstance(result, dict)
        assert "objective" in result
        assert len(result) == 1

    def test_is_negative_at_optimum(self):
        """hartmann3's optimal value is negative."""
        params = {"x1": 0.114614, "x2": 0.555649, "x3": 0.852547}
        result = hartmann3(params)
        assert result["objective"] < 0.0

    def test_at_all_ones(self):
        """hartmann3 at (1, 1, 1) should return a finite value."""
        result = hartmann3({"x1": 1.0, "x2": 1.0, "x3": 1.0})
        assert math.isfinite(result["objective"])


# ── hartmann6 tests ──────────────────────────────────────────────────


class TestHartmann6:
    """Tests for the Hartmann 6D function."""

    def test_at_known_optimum(self):
        """hartmann6 at the known optimal params should be close to -3.32237."""
        params = {
            "x1": 0.20169,
            "x2": 0.15001,
            "x3": 0.47687,
            "x4": 0.27533,
            "x5": 0.31165,
            "x6": 0.65730,
        }
        result = hartmann6(params)
        assert result["objective"] == pytest.approx(-3.32237, abs=1e-3)

    def test_at_all_zeros(self):
        """hartmann6 at (0,...,0) should return a known finite value."""
        params = {f"x{i}": 0.0 for i in range(1, 7)}
        result = hartmann6(params)
        assert isinstance(result["objective"], float)
        assert math.isfinite(result["objective"])

    def test_returns_dict_with_objective_key(self):
        """hartmann6 should return a dict with 'objective' key."""
        params = {f"x{i}": 0.5 for i in range(1, 7)}
        result = hartmann6(params)
        assert isinstance(result, dict)
        assert "objective" in result
        assert len(result) == 1

    def test_is_negative_at_optimum(self):
        """hartmann6's optimal value is negative."""
        params = {
            "x1": 0.20169,
            "x2": 0.15001,
            "x3": 0.47687,
            "x4": 0.27533,
            "x5": 0.31165,
            "x6": 0.65730,
        }
        result = hartmann6(params)
        assert result["objective"] < 0.0

    def test_deterministic(self):
        """hartmann6 should return the same value for the same input."""
        params = {f"x{i}": 0.3 for i in range(1, 7)}
        r1 = hartmann6(params)
        r2 = hartmann6(params)
        assert r1["objective"] == r2["objective"]


# ── levy10 tests ─────────────────────────────────────────────────────


class TestLevy10:
    """Tests for the Levy function in 10D."""

    def test_at_known_optimum(self):
        """levy at all xi=1 should return 0 (global minimum)."""
        params = {f"x{i}": 1.0 for i in range(1, 11)}
        result = levy(params)
        assert result["objective"] == pytest.approx(0.0, abs=1e-10)

    def test_at_origin(self):
        """levy at the origin should return a positive value."""
        params = {f"x{i}": 0.0 for i in range(1, 11)}
        result = levy(params)
        assert isinstance(result["objective"], float)
        assert math.isfinite(result["objective"])
        # At origin the Levy function is not zero
        assert result["objective"] > 0.0

    def test_10d_input_output(self):
        """levy requires 10 parameters and returns dict with 'objective'."""
        params = {f"x{i}": 0.5 for i in range(1, 11)}
        result = levy(params)
        assert isinstance(result, dict)
        assert "objective" in result
        assert len(result) == 1

    def test_symmetric_inputs(self):
        """levy is not necessarily symmetric but should handle uniform inputs."""
        params_a = {f"x{i}": 2.0 for i in range(1, 11)}
        params_b = {f"x{i}": -2.0 for i in range(1, 11)}
        result_a = levy(params_a)
        result_b = levy(params_b)
        # Both should be finite
        assert math.isfinite(result_a["objective"])
        assert math.isfinite(result_b["objective"])

    def test_returns_nonnegative_at_optimum(self):
        """levy minimum is 0, so value at optimum should be >= 0."""
        params = {f"x{i}": 1.0 for i in range(1, 11)}
        result = levy(params)
        assert result["objective"] >= 0.0


# ── rosenbrock tests ─────────────────────────────────────────────────


class TestRosenbrock:
    """Tests for the Rosenbrock function."""

    def test_at_known_optimum_2d(self):
        """rosenbrock at all xi=1 should return 0 (global minimum) in 2D."""
        params = {"x1": 1.0, "x2": 1.0}
        result = rosenbrock(params)
        assert result["objective"] == pytest.approx(0.0, abs=1e-12)

    def test_at_known_optimum_5d(self):
        """rosenbrock at all xi=1 should return 0 in 5D."""
        params = {f"x{i}": 1.0 for i in range(1, 6)}
        result = rosenbrock(params)
        assert result["objective"] == pytest.approx(0.0, abs=1e-12)

    def test_at_known_optimum_20d(self):
        """rosenbrock at all xi=1 should return 0 in 20D."""
        params = {f"x{i}": 1.0 for i in range(1, 21)}
        result = rosenbrock(params)
        assert result["objective"] == pytest.approx(0.0, abs=1e-12)

    def test_at_origin_2d(self):
        """rosenbrock at (0, 0) = 100*(0-0)^2 + (1-0)^2 = 1."""
        params = {"x1": 0.0, "x2": 0.0}
        result = rosenbrock(params)
        assert result["objective"] == pytest.approx(1.0, abs=1e-12)

    def test_different_dimensions(self):
        """rosenbrock should work for different dimensionalities."""
        for n_dims in [2, 5, 10, 20]:
            params = {f"x{i}": 0.0 for i in range(1, n_dims + 1)}
            result = rosenbrock(params)
            # At origin in n-d: sum of (n-1) terms of 100*0 + 1 = (n-1)
            assert result["objective"] == pytest.approx(float(n_dims - 1), abs=1e-10)


# ── constrained_branin tests ─────────────────────────────────────────


class TestConstrainedBranin:
    """Tests for the constrained Branin function."""

    def test_returns_objective_and_constraint(self):
        """constrained_branin should return both objective and constraint_violation."""
        result = constrained_branin({"x1": 0.0, "x2": 0.0})
        assert "objective" in result
        assert "constraint_violation" in result

    def test_feasible_point_zero_violation(self):
        """A feasible point (x1 + x2 <= 14) should have violation = 0."""
        # x1=-5, x2=0 => sum=-5 < 14 => feasible
        result = constrained_branin({"x1": -5.0, "x2": 0.0})
        assert result["constraint_violation"] == 0.0

    def test_infeasible_point_positive_violation(self):
        """An infeasible point (x1 + x2 > 14) should have violation > 0."""
        # x1=10, x2=15 => sum=25 > 14 => violation = 25-14 = 11
        result = constrained_branin({"x1": 10.0, "x2": 15.0})
        assert result["constraint_violation"] > 0.0
        assert result["constraint_violation"] == pytest.approx(11.0, abs=1e-10)

    def test_boundary_point(self):
        """At the constraint boundary (x1+x2=14), violation should be 0."""
        result = constrained_branin({"x1": 4.0, "x2": 10.0})
        assert result["constraint_violation"] == pytest.approx(0.0, abs=1e-10)

    def test_same_objective_as_unconstrained(self):
        """The objective should match unconstrained branin regardless of feasibility."""
        params = {"x1": 3.0, "x2": 5.0}
        constrained_result = constrained_branin(params)
        unconstrained_result = branin(params)
        assert constrained_result["objective"] == pytest.approx(
            unconstrained_result["objective"], rel=1e-12
        )

    def test_at_optimum_is_feasible(self):
        """The known optimum at (-pi, 12.275) should be feasible (sum ~ 9.13)."""
        params = {"x1": -math.pi, "x2": 12.275}
        result = constrained_branin(params)
        assert result["constraint_violation"] == 0.0
        assert result["objective"] == pytest.approx(0.397887, abs=1e-3)


# ── noisy_hartmann6 tests ────────────────────────────────────────────


class TestNoisyHartmann6:
    """Tests for the noisy Hartmann 6D function."""

    def test_returns_dict_with_objective_key(self):
        """noisy_hartmann6 should return a dict with 'objective' key."""
        params = {f"x{i}": 0.5 for i in range(1, 7)}
        result = noisy_hartmann6(params)
        assert isinstance(result, dict)
        assert "objective" in result

    def test_different_noise_std_values(self):
        """Different noise_std values should produce different amounts of noise."""
        params = {f"x{i}": 0.5 for i in range(1, 7)}
        true_val = hartmann6(params)["objective"]
        result_low = noisy_hartmann6(params, noise_std=0.001)
        result_high = noisy_hartmann6(params, noise_std=10.0)
        # The low-noise result should be closer to the true value
        diff_low = abs(result_low["objective"] - true_val)
        diff_high = abs(result_high["objective"] - true_val)
        # This is a probabilistic test but with such extreme noise_std
        # ratios it should almost always hold
        assert diff_low < diff_high

    def test_deterministic_for_same_params(self):
        """noisy_hartmann6 uses a seed derived from params, so same params = same result."""
        params = {f"x{i}": 0.3 for i in range(1, 7)}
        r1 = noisy_hartmann6(params, noise_std=0.1)
        r2 = noisy_hartmann6(params, noise_std=0.1)
        assert r1["objective"] == r2["objective"]

    def test_different_params_different_noise(self):
        """Different parameters should generally yield different noise realizations."""
        params_a = {f"x{i}": 0.3 for i in range(1, 7)}
        params_b = {f"x{i}": 0.4 for i in range(1, 7)}
        r_a = noisy_hartmann6(params_a, noise_std=0.5)
        r_b = noisy_hartmann6(params_b, noise_std=0.5)
        # Different params should give different results
        assert r_a["objective"] != r_b["objective"]

    def test_mean_close_to_true_value_over_samples(self):
        """Over many slightly different inputs, the mean should approximate the noiseless value."""
        # We vary the last dimension slightly to get different noise seeds
        base_params = {"x1": 0.2, "x2": 0.15, "x3": 0.48, "x4": 0.28, "x5": 0.31}
        true_results = []
        noisy_results = []
        n_samples = 200
        for i in range(n_samples):
            x6 = 0.65 + i * 0.0001  # tiny variation
            params = {**base_params, "x6": x6}
            true_results.append(hartmann6(params)["objective"])
            noisy_results.append(noisy_hartmann6(params, noise_std=0.1)["objective"])

        mean_true = statistics.mean(true_results)
        mean_noisy = statistics.mean(noisy_results)
        # The means should be close (noise averages out)
        assert mean_noisy == pytest.approx(mean_true, abs=0.05)


# ── multifidelity_branin tests ───────────────────────────────────────


class TestMultifidelityBranin:
    """Tests for the multi-fidelity Branin function."""

    def test_fidelity_2_matches_exact_branin(self):
        """fidelity=2 should match the exact branin value."""
        params = {"x1": 3.0, "x2": 7.0}
        mf_result = multifidelity_branin(params, fidelity=2)
        exact_result = branin(params)
        assert mf_result["objective"] == pytest.approx(
            exact_result["objective"], rel=1e-12
        )

    def test_lower_fidelity_has_bias(self):
        """fidelity=0 and fidelity=1 should differ from the exact branin."""
        params = {"x1": 3.0, "x2": 7.0}
        exact = branin(params)["objective"]
        low = multifidelity_branin(params, fidelity=0)["objective"]
        med = multifidelity_branin(params, fidelity=1)["objective"]
        # Low and medium fidelity should not match exact
        assert low != pytest.approx(exact, abs=0.01)
        assert med != pytest.approx(exact, abs=0.01)

    def test_returns_dict_with_objective_and_fidelity(self):
        """multifidelity_branin should return dict with 'objective' and 'fidelity'."""
        params = {"x1": 0.0, "x2": 0.0}
        for fid in [0, 1, 2]:
            result = multifidelity_branin(params, fidelity=fid)
            assert "objective" in result
            assert "fidelity" in result
            assert result["fidelity"] == float(fid)

    def test_fidelity_levels_are_ordered(self):
        """Higher fidelity should be closer to the exact value at most points."""
        params = {"x1": -math.pi, "x2": 12.275}
        exact = branin(params)["objective"]
        low = multifidelity_branin(params, fidelity=0)["objective"]
        med = multifidelity_branin(params, fidelity=1)["objective"]
        high = multifidelity_branin(params, fidelity=2)["objective"]
        # High fidelity should be exact
        assert high == pytest.approx(exact, rel=1e-12)
        # Medium fidelity should be closer to exact than low fidelity at many points
        # (not guaranteed at all points, but at the optimum it should hold)
        diff_med = abs(med - exact)
        diff_low = abs(low - exact)
        assert diff_med < diff_low

    def test_fidelity_0_bias_calculation(self):
        """Verify the low-fidelity bias formula."""
        params = {"x1": 2.5, "x2": 7.5}
        exact = branin(params)["objective"]
        result = multifidelity_branin(params, fidelity=0)
        # bias = 0.5*(x1-2.5)^2 + 0.5*(x2-7.5)^2 - 20 = 0 + 0 - 20 = -20
        # val = exact + bias + 10 = exact - 10
        expected = exact + (-20.0) + 10.0
        assert result["objective"] == pytest.approx(expected, rel=1e-10)


# ── zdt1 tests ──────────────────────────────────────────────────────


class TestZDT1:
    """Tests for the ZDT1 bi-objective function."""

    def test_returns_f1_and_f2_keys(self):
        """zdt1 should return a dict with 'f1' and 'f2' keys."""
        params = {f"x{i}": 0.5 for i in range(1, 31)}
        result = zdt1(params)
        assert isinstance(result, dict)
        assert "f1" in result
        assert "f2" in result

    def test_on_pareto_front(self):
        """On the Pareto front (x1 varies, all other xi=0), f2 = 1 - sqrt(f1)."""
        for x1_val in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            params = {f"x{i}": 0.0 for i in range(1, 31)}
            params["x1"] = x1_val
            result = zdt1(params)
            assert result["f1"] == pytest.approx(x1_val, abs=1e-12)
            expected_f2 = 1.0 - math.sqrt(x1_val) if x1_val >= 0 else 1.0
            assert result["f2"] == pytest.approx(expected_f2, abs=1e-6)

    def test_30d_input(self):
        """zdt1 requires 30 parameters."""
        params = {f"x{i}": 0.0 for i in range(1, 31)}
        result = zdt1(params)
        assert math.isfinite(result["f1"])
        assert math.isfinite(result["f2"])

    def test_f1_equals_x1(self):
        """f1 should always equal x1."""
        for x1_val in [0.0, 0.3, 0.7, 1.0]:
            params = {f"x{i}": 0.5 for i in range(1, 31)}
            params["x1"] = x1_val
            result = zdt1(params)
            assert result["f1"] == pytest.approx(x1_val, abs=1e-12)

    def test_f2_increases_with_other_vars(self):
        """f2 should increase when the other variables increase (away from Pareto front)."""
        params_low = {f"x{i}": 0.0 for i in range(1, 31)}
        params_low["x1"] = 0.5
        params_high = {f"x{i}": 0.5 for i in range(1, 31)}
        params_high["x1"] = 0.5
        r_low = zdt1(params_low)
        r_high = zdt1(params_high)
        assert r_high["f2"] > r_low["f2"]

    def test_pareto_front_f2_formula(self):
        """Verify f2 = 1 - sqrt(f1) on the Pareto front more precisely."""
        params = {f"x{i}": 0.0 for i in range(1, 31)}
        params["x1"] = 0.36
        result = zdt1(params)
        # g = 1 when all xi=0 for i>1
        # f2 = g*(1 - sqrt(f1/g)) = 1*(1 - sqrt(0.36)) = 1 - 0.6 = 0.4
        assert result["f2"] == pytest.approx(0.4, abs=1e-10)


# ── sphere5 tests ──────────────────────────────────────────────────


class TestSphere5:
    """Tests for the Sphere function in 5D."""

    def test_at_known_optimum(self):
        """sphere5 at the origin should return 0."""
        params = {f"x{i}": 0.0 for i in range(1, 6)}
        result = sphere5(params)
        assert result["objective"] == pytest.approx(0.0, abs=1e-12)

    def test_returns_dict_with_objective_key(self):
        """sphere5 should return a dict with 'objective' key."""
        params = {f"x{i}": 1.0 for i in range(1, 6)}
        result = sphere5(params)
        assert isinstance(result, dict)
        assert "objective" in result

    def test_positive_away_from_origin(self):
        """sphere5 at all-ones should return 5.0."""
        params = {f"x{i}": 1.0 for i in range(1, 6)}
        result = sphere5(params)
        assert result["objective"] == pytest.approx(5.0, abs=1e-12)


# ── ackley10 tests ─────────────────────────────────────────────────


class TestAckley10:
    """Tests for the Ackley function in 10D."""

    def test_at_known_optimum(self):
        """ackley10 at the origin should return 0."""
        params = {f"x{i}": 0.0 for i in range(1, 11)}
        result = ackley10(params)
        assert result["objective"] == pytest.approx(0.0, abs=1e-10)

    def test_returns_dict_with_objective_key(self):
        """ackley10 should return a dict with 'objective' key."""
        params = {f"x{i}": 1.0 for i in range(1, 11)}
        result = ackley10(params)
        assert isinstance(result, dict)
        assert "objective" in result

    def test_positive_away_from_origin(self):
        """ackley10 away from origin should be positive."""
        params = {f"x{i}": 1.0 for i in range(1, 11)}
        result = ackley10(params)
        assert result["objective"] > 0.0


# ── rastrigin10 tests ──────────────────────────────────────────────


class TestRastrigin10:
    """Tests for the Rastrigin function in 10D."""

    def test_at_known_optimum(self):
        """rastrigin10 at the origin should return 0."""
        params = {f"x{i}": 0.0 for i in range(1, 11)}
        result = rastrigin10(params)
        assert result["objective"] == pytest.approx(0.0, abs=1e-10)

    def test_returns_dict_with_objective_key(self):
        """rastrigin10 should return a dict with 'objective' key."""
        params = {f"x{i}": 1.0 for i in range(1, 11)}
        result = rastrigin10(params)
        assert isinstance(result, dict)
        assert "objective" in result

    def test_positive_away_from_origin(self):
        """rastrigin10 away from origin should be positive."""
        params = {f"x{i}": 1.0 for i in range(1, 11)}
        result = rastrigin10(params)
        assert result["objective"] > 0.0


# ── griewank10 tests ───────────────────────────────────────────────


class TestGriewank10:
    """Tests for the Griewank function in 10D."""

    def test_at_known_optimum(self):
        """griewank10 at the origin should return 0."""
        params = {f"x{i}": 0.0 for i in range(1, 11)}
        result = griewank10(params)
        assert result["objective"] == pytest.approx(0.0, abs=1e-10)

    def test_returns_dict_with_objective_key(self):
        """griewank10 should return a dict with 'objective' key."""
        params = {f"x{i}": 1.0 for i in range(1, 11)}
        result = griewank10(params)
        assert isinstance(result, dict)
        assert "objective" in result

    def test_nonnegative(self):
        """griewank10 at large values should be nonnegative."""
        params = {f"x{i}": 100.0 for i in range(1, 11)}
        result = griewank10(params)
        assert result["objective"] >= 0.0


# ── schwefel10 tests ───────────────────────────────────────────────


class TestSchwefel10:
    """Tests for the Schwefel function in 10D."""

    def test_at_known_optimum(self):
        """schwefel10 at xi=420.9687 should be close to 0."""
        params = {f"x{i}": 420.9687 for i in range(1, 11)}
        result = schwefel10(params)
        assert result["objective"] == pytest.approx(0.0, abs=1.0)  # wider tolerance

    def test_returns_dict_with_objective_key(self):
        """schwefel10 should return a dict with 'objective' key."""
        params = {f"x{i}": 0.0 for i in range(1, 11)}
        result = schwefel10(params)
        assert isinstance(result, dict)
        assert "objective" in result

    def test_finite_at_bounds(self):
        """schwefel10 should be finite at domain bounds."""
        params = {f"x{i}": -500.0 for i in range(1, 11)}
        result = schwefel10(params)
        assert math.isfinite(result["objective"])


# ── styblinski_tang10 tests ────────────────────────────────────────


class TestStyblinskiTang10:
    """Tests for the Styblinski-Tang function in 10D."""

    def test_at_known_optimum(self):
        """styblinski_tang10 at xi=-2.903534 should be close to -391.6599."""
        params = {f"x{i}": -2.903534 for i in range(1, 11)}
        result = styblinski_tang10(params)
        assert result["objective"] == pytest.approx(-391.6599, abs=0.01)

    def test_returns_dict_with_objective_key(self):
        """styblinski_tang10 should return a dict with 'objective' key."""
        params = {f"x{i}": 0.0 for i in range(1, 11)}
        result = styblinski_tang10(params)
        assert isinstance(result, dict)
        assert "objective" in result

    def test_at_origin(self):
        """styblinski_tang10 at origin should be 0."""
        params = {f"x{i}": 0.0 for i in range(1, 11)}
        result = styblinski_tang10(params)
        assert result["objective"] == pytest.approx(0.0, abs=1e-10)


# ── dixon_price10 tests ────────────────────────────────────────────


class TestDixonPrice10:
    """Tests for the Dixon-Price function in 10D."""

    def test_returns_dict_with_objective_key(self):
        """dixon_price10 should return a dict with 'objective' key."""
        params = {f"x{i}": 0.0 for i in range(1, 11)}
        result = dixon_price10(params)
        assert isinstance(result, dict)
        assert "objective" in result

    def test_nonnegative(self):
        """dixon_price10 should be nonnegative."""
        params = {f"x{i}": 0.0 for i in range(1, 11)}
        result = dixon_price10(params)
        assert result["objective"] >= 0.0

    def test_at_origin(self):
        """dixon_price10 at origin: (0-1)^2 + sum i*(2*0^2 - 0)^2 = 1."""
        params = {f"x{i}": 0.0 for i in range(1, 11)}
        result = dixon_price10(params)
        # At origin: (0-1)^2 + sum i*(2*0^2 - 0)^2 = 1 + 0 = 1
        assert result["objective"] == pytest.approx(1.0, abs=1e-10)


# ── michalewicz10 tests ────────────────────────────────────────────


class TestMichalewicz10:
    """Tests for the Michalewicz function in 10D."""

    def test_returns_dict_with_objective_key(self):
        """michalewicz10 should return a dict with 'objective' key."""
        params = {f"x{i}": 1.0 for i in range(1, 11)}
        result = michalewicz10(params)
        assert isinstance(result, dict)
        assert "objective" in result

    def test_negative_at_good_points(self):
        """michalewicz10 should be negative at reasonable interior points."""
        params = {f"x{i}": 1.5 for i in range(1, 11)}
        result = michalewicz10(params)
        assert result["objective"] < 0.0

    def test_finite_output(self):
        """michalewicz10 should return a finite value at pi/2."""
        params = {f"x{i}": math.pi / 2.0 for i in range(1, 11)}
        result = michalewicz10(params)
        assert math.isfinite(result["objective"])


# ── zakharov10 tests ───────────────────────────────────────────────


class TestZakharov10:
    """Tests for the Zakharov function in 10D."""

    def test_at_known_optimum(self):
        """zakharov10 at origin should return 0."""
        params = {f"x{i}": 0.0 for i in range(1, 11)}
        result = zakharov10(params)
        assert result["objective"] == pytest.approx(0.0, abs=1e-12)

    def test_returns_dict_with_objective_key(self):
        """zakharov10 should return a dict with 'objective' key."""
        params = {f"x{i}": 1.0 for i in range(1, 11)}
        result = zakharov10(params)
        assert isinstance(result, dict)
        assert "objective" in result

    def test_positive_away_from_origin(self):
        """zakharov10 away from origin should be positive."""
        params = {f"x{i}": 1.0 for i in range(1, 11)}
        result = zakharov10(params)
        assert result["objective"] > 0.0


# ── bohachevsky2 tests ─────────────────────────────────────────────


class TestBohachevsky2:
    """Tests for the Bohachevsky #2 function in 2D."""

    def test_at_known_optimum(self):
        """bohachevsky2 at origin should return 0."""
        result = bohachevsky2({"x1": 0.0, "x2": 0.0})
        assert result["objective"] == pytest.approx(0.0, abs=1e-10)

    def test_returns_dict_with_objective_key(self):
        """bohachevsky2 should return a dict with 'objective' key."""
        result = bohachevsky2({"x1": 1.0, "x2": 1.0})
        assert isinstance(result, dict)
        assert "objective" in result

    def test_positive_away_from_origin(self):
        """bohachevsky2 away from origin should be positive."""
        result = bohachevsky2({"x1": 5.0, "x2": 5.0})
        assert result["objective"] > 0.0


# ── BENCHMARK_SUITE registry tests ──────────────────────────────────


class TestBenchmarkSuiteRegistry:
    """Tests for the BENCHMARK_SUITE registry."""

    def test_contains_20_entries(self):
        """The suite should contain exactly 20 benchmark entries."""
        assert len(BENCHMARK_SUITE) == 20

    def test_all_entries_are_benchmark_function(self):
        """Every entry should be a BenchmarkFunction instance."""
        for name, bf in BENCHMARK_SUITE.items():
            assert isinstance(bf, BenchmarkFunction), f"{name} is not BenchmarkFunction"

    def test_get_benchmark_returns_correct_function(self):
        """get_benchmark should return the correct BenchmarkFunction by name."""
        bf = get_benchmark("branin")
        assert bf.name == "branin"
        assert bf is BENCHMARK_SUITE["branin"]

    def test_get_benchmark_unknown_name_raises_key_error(self):
        """get_benchmark should raise KeyError for an unknown name."""
        with pytest.raises(KeyError, match="Unknown benchmark"):
            get_benchmark("nonexistent_function")

    def test_list_benchmarks_returns_sorted_names(self):
        """list_benchmarks should return all names in sorted order."""
        names = list_benchmarks()
        assert names == sorted(names)
        assert len(names) == 20
        assert "branin" in names
        assert "zdt1" in names

    def test_expected_benchmark_names(self):
        """Verify the expected benchmark names are present."""
        expected = {
            "branin",
            "hartmann3",
            "hartmann6",
            "levy10",
            "rosenbrock5",
            "rosenbrock20",
            "constrained_branin",
            "noisy_hartmann6",
            "multifidelity_branin",
            "zdt1",
            "sphere5",
            "ackley10",
            "rastrigin10",
            "griewank10",
            "schwefel10",
            "styblinski_tang10",
            "dixon_price10",
            "michalewicz10",
            "zakharov10",
            "bohachevsky2",
        }
        assert set(BENCHMARK_SUITE.keys()) == expected


# ── make_spec tests ──────────────────────────────────────────────────


class TestMakeSpec:
    """Tests for the make_spec helper."""

    def test_returns_dict_with_correct_structure(self):
        """make_spec should return a dict with the required keys."""
        bf = get_benchmark("branin")
        spec = make_spec(bf)
        assert isinstance(spec, dict)
        required_keys = {
            "name",
            "parameters",
            "objectives",
            "budget_iterations",
            "known_optimum",
            "optimal_params",
            "metadata",
        }
        assert required_keys.issubset(set(spec.keys()))

    def test_parameters_match_benchmark(self):
        """Parameters in the spec should match the benchmark's parameter_specs."""
        bf = get_benchmark("hartmann6")
        spec = make_spec(bf)
        assert spec["parameters"] == bf.parameter_specs
        assert len(spec["parameters"]) == 6

    def test_objectives_derived_from_known_optimum(self):
        """Objectives should have entries matching known_optimum keys."""
        bf = get_benchmark("zdt1")
        spec = make_spec(bf)
        obj_names = {obj["name"] for obj in spec["objectives"]}
        assert obj_names == {"f1", "f2"}
        for obj in spec["objectives"]:
            assert obj["direction"] == "minimize"

    def test_budget_iterations_default(self):
        """Default budget_iterations should be 50."""
        bf = get_benchmark("branin")
        spec = make_spec(bf)
        assert spec["budget_iterations"] == 50

    def test_budget_iterations_custom(self):
        """Custom budget_iterations should be respected."""
        bf = get_benchmark("branin")
        spec = make_spec(bf, budget_iterations=100)
        assert spec["budget_iterations"] == 100

    def test_single_objective_spec(self):
        """Single-objective benchmarks should produce one objective entry."""
        bf = get_benchmark("branin")
        spec = make_spec(bf)
        assert len(spec["objectives"]) == 1
        assert spec["objectives"][0]["name"] == "objective"
        assert spec["objectives"][0]["direction"] == "minimize"


# ── Cross-benchmark validation tests ────────────────────────────────


class TestCrossBenchmarkValidation:
    """Cross-cutting validation tests across all benchmarks."""

    def test_all_benchmarks_callable(self):
        """Every benchmark in the suite should be callable."""
        for name, bf in BENCHMARK_SUITE.items():
            # Build params from parameter_specs
            params = {}
            for spec in bf.parameter_specs:
                lo, hi = spec["bounds"]
                params[spec["name"]] = (lo + hi) / 2.0  # midpoint
            result = bf(params)
            assert isinstance(result, dict), f"{name} did not return a dict"

    def test_all_benchmarks_return_dicts(self):
        """Every benchmark should return a dict from its evaluate function."""
        for name, bf in BENCHMARK_SUITE.items():
            params = {}
            for spec in bf.parameter_specs:
                lo, hi = spec["bounds"]
                params[spec["name"]] = lo  # lower bound
            result = bf.evaluate(params)
            assert isinstance(result, dict), f"{name}.evaluate did not return a dict"

    def test_all_parameter_specs_have_required_keys(self):
        """Every parameter spec should have 'name', 'type', and 'bounds' keys."""
        for name, bf in BENCHMARK_SUITE.items():
            for i, spec in enumerate(bf.parameter_specs):
                assert "name" in spec, f"{name} param {i} missing 'name'"
                assert "type" in spec, f"{name} param {i} missing 'type'"
                assert "bounds" in spec, f"{name} param {i} missing 'bounds'"
                assert len(spec["bounds"]) == 2, f"{name} param {i} bounds should have 2 elements"
                assert spec["bounds"][0] <= spec["bounds"][1], (
                    f"{name} param {i} lower bound > upper bound"
                )

    def test_all_known_optima_achievable(self):
        """For benchmarks with optimal_params, evaluate should match known_optimum.

        Noisy benchmarks are checked with a wider tolerance since they add
        stochastic noise to the true value.
        """
        noisy_benchmarks = {"noisy_hartmann6"}
        for name, bf in BENCHMARK_SUITE.items():
            if bf.optimal_params is None:
                continue
            result = bf.evaluate(bf.optimal_params)
            tol = 0.5 if name in noisy_benchmarks else 1e-2
            for obj_key, opt_val in bf.known_optimum.items():
                assert result[obj_key] == pytest.approx(opt_val, abs=tol), (
                    f"{name}: {obj_key} expected {opt_val}, got {result[obj_key]}"
                )

    def test_all_benchmarks_have_consistent_dimensionality(self):
        """metadata dimensionality should match the number of parameter specs."""
        for name, bf in BENCHMARK_SUITE.items():
            if "dimensionality" in bf.metadata:
                assert bf.metadata["dimensionality"] == len(bf.parameter_specs), (
                    f"{name}: dimensionality mismatch "
                    f"({bf.metadata['dimensionality']} vs {len(bf.parameter_specs)} specs)"
                )

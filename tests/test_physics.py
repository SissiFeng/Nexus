"""Comprehensive tests for the optimization_copilot.physics package.

Covers kernels, priors, ODE solver, and physics constraint model.
"""

from __future__ import annotations

import math

import pytest

from optimization_copilot.physics.kernels import (
    CompositeKernel,
    LinearKernel,
    PeriodicKernel,
    linear_kernel,
    linear_kernel_matrix,
    periodic_kernel,
    periodic_kernel_matrix,
    symmetry_kernel,
)
from optimization_copilot.physics.priors import (
    ArrheniusPrior,
    MichaelisMentenPrior,
    PowerLawPrior,
)
from optimization_copilot.physics.ode_solver import RK4Solver
from optimization_copilot.physics.constraints import (
    ConservationLaw,
    MonotonicityConstraint,
    PhysicsBound,
    PhysicsConstraintModel,
)


# ============================================================================
# 1. Periodic Kernel Tests
# ============================================================================


class TestPeriodicKernel:
    """Tests for periodic_kernel and PeriodicKernel."""

    def test_self_similarity(self) -> None:
        """Periodic kernel of a point with itself is 1."""
        x = [1.0, 2.0, 3.0]
        assert periodic_kernel(x, x) == pytest.approx(1.0)

    def test_periodicity(self) -> None:
        """k(x, x + p) should equal k(x, x) = 1 for period p."""
        x = [0.5]
        x_plus_p = [0.5 + 1.0]  # period = 1.0
        k_val = periodic_kernel(x, x_plus_p, length_scale=1.0, period=1.0)
        assert k_val == pytest.approx(1.0, abs=1e-10)

    def test_half_period_minimum(self) -> None:
        """Points separated by half a period should give lower kernel value."""
        x = [0.0]
        x_half = [0.5]
        k_val = periodic_kernel(x, x_half, length_scale=1.0, period=1.0)
        assert k_val < 1.0

    def test_symmetry(self) -> None:
        """k(x, y) == k(y, x)."""
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        assert periodic_kernel(x, y) == pytest.approx(periodic_kernel(y, x))

    def test_kernel_matrix_symmetric(self) -> None:
        """Periodic kernel matrix is symmetric."""
        X = [[0.0], [0.3], [0.7], [1.0]]
        K = periodic_kernel_matrix(X, length_scale=1.0, period=1.0)
        n = len(X)
        for i in range(n):
            for j in range(n):
                assert K[i][j] == pytest.approx(K[j][i])

    def test_kernel_matrix_positive_diagonal(self) -> None:
        """Diagonal entries of kernel matrix are positive."""
        X = [[0.0], [1.0], [2.0]]
        K = periodic_kernel_matrix(X)
        for i in range(len(X)):
            assert K[i][i] > 0.0

    def test_callable_class(self) -> None:
        """PeriodicKernel callable produces same result as function."""
        x, y = [1.0, 2.0], [3.0, 4.0]
        kern = PeriodicKernel(length_scale=0.5, period=2.0)
        expected = periodic_kernel(x, y, length_scale=0.5, period=2.0)
        assert kern(x, y) == pytest.approx(expected)


# ============================================================================
# 2. Linear Kernel Tests
# ============================================================================


class TestLinearKernel:
    """Tests for linear_kernel and LinearKernel."""

    def test_known_value(self) -> None:
        """Linear kernel of [1,2] and [3,4] with variance=1, offset=0 is 11."""
        assert linear_kernel([1.0, 2.0], [3.0, 4.0]) == pytest.approx(11.0)

    def test_with_offset(self) -> None:
        """Linear kernel with offset adds offset to dot product."""
        k = linear_kernel([1.0], [2.0], variance=1.0, offset=3.0)
        assert k == pytest.approx(5.0)  # 1*2 + 3

    def test_with_variance(self) -> None:
        """Variance scales the kernel."""
        k = linear_kernel([1.0], [2.0], variance=3.0, offset=0.0)
        assert k == pytest.approx(6.0)  # 3 * (1*2)

    def test_symmetry(self) -> None:
        """k(x, y) == k(y, x)."""
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        assert linear_kernel(x, y) == pytest.approx(linear_kernel(y, x))

    def test_kernel_matrix_symmetric(self) -> None:
        """Linear kernel matrix is symmetric."""
        X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        K = linear_kernel_matrix(X)
        n = len(X)
        for i in range(n):
            for j in range(n):
                assert K[i][j] == pytest.approx(K[j][i])

    def test_kernel_matrix_positive_diagonal(self) -> None:
        """Diagonal entries of kernel matrix are positive."""
        X = [[1.0], [2.0], [3.0]]
        K = linear_kernel_matrix(X)
        for i in range(len(X)):
            assert K[i][i] > 0.0

    def test_callable_class(self) -> None:
        """LinearKernel callable produces same result as function."""
        x, y = [1.0, 2.0], [3.0, 4.0]
        kern = LinearKernel(variance=2.0, offset=1.0)
        expected = linear_kernel(x, y, variance=2.0, offset=1.0)
        assert kern(x, y) == pytest.approx(expected)


# ============================================================================
# 3. Composite Kernel Tests
# ============================================================================


class TestCompositeKernel:
    """Tests for CompositeKernel."""

    def test_sum_of_kernels(self) -> None:
        """Sum composite kernel adds kernel values."""
        k1 = PeriodicKernel(length_scale=1.0, period=1.0)
        k2 = LinearKernel(variance=1.0, offset=0.0)
        composite = CompositeKernel([k1, k2], operation="sum")
        x, y = [1.0], [2.0]
        expected = k1(x, y) + k2(x, y)
        assert composite(x, y) == pytest.approx(expected)

    def test_product_of_kernels(self) -> None:
        """Product composite kernel multiplies kernel values."""
        k1 = PeriodicKernel(length_scale=1.0, period=1.0)
        k2 = LinearKernel(variance=1.0, offset=1.0)
        composite = CompositeKernel([k1, k2], operation="product")
        x, y = [1.0], [2.0]
        expected = k1(x, y) * k2(x, y)
        assert composite(x, y) == pytest.approx(expected)

    def test_invalid_operation_raises(self) -> None:
        """Invalid operation raises ValueError."""
        with pytest.raises(ValueError, match="operation must be"):
            CompositeKernel([], operation="divide")

    def test_single_kernel_sum(self) -> None:
        """Sum with a single kernel returns that kernel's value."""
        k = PeriodicKernel()
        composite = CompositeKernel([k], operation="sum")
        x = [0.5, 1.0]
        assert composite(x, x) == pytest.approx(k(x, x))

    def test_single_kernel_product(self) -> None:
        """Product with a single kernel returns that kernel's value."""
        k = LinearKernel(variance=2.0)
        composite = CompositeKernel([k], operation="product")
        x, y = [1.0], [3.0]
        assert composite(x, y) == pytest.approx(k(x, y))


# ============================================================================
# 4. Symmetry Kernel Tests
# ============================================================================


class TestSymmetryKernel:
    """Tests for symmetry_kernel."""

    def test_identity_symmetry(self) -> None:
        """With identity-only group, symmetry kernel equals base kernel."""
        identity = lambda x: x
        x, y = [1.0, 2.0], [3.0, 4.0]
        base_k = periodic_kernel
        k_sym = symmetry_kernel(x, y, base_k, [identity])
        assert k_sym == pytest.approx(base_k(x, y))

    def test_reflection_symmetry(self) -> None:
        """Reflection symmetry: k_sym(x, y) == k_sym(-x, y) when group includes negation."""
        identity = lambda x: list(x)
        negate = lambda x: [-xi for xi in x]
        base_k = periodic_kernel
        group = [identity, negate]

        x, y = [0.3], [0.7]
        k_sym_x = symmetry_kernel(x, y, base_k, group)
        k_sym_neg_x = symmetry_kernel([-0.3], y, base_k, group)
        assert k_sym_x == pytest.approx(k_sym_neg_x)

    def test_permutation_symmetry(self) -> None:
        """Permutation symmetry: swapping coordinates in the group."""
        identity = lambda x: [x[0], x[1]]
        swap = lambda x: [x[1], x[0]]
        base_k = lambda x1, x2: sum(a * b for a, b in zip(x1, x2))  # dot product
        group = [identity, swap]

        x = [1.0, 2.0]
        y = [3.0, 4.0]
        k_sym = symmetry_kernel(x, y, base_k, group)
        # (1*3 + 2*4 + 2*3 + 1*4) / 2 = (11 + 10) / 2 = 10.5
        assert k_sym == pytest.approx(10.5)

    def test_empty_group_returns_base(self) -> None:
        """Empty symmetry group falls back to base kernel."""
        x, y = [1.0], [2.0]
        k_sym = symmetry_kernel(x, y, periodic_kernel, [])
        assert k_sym == pytest.approx(periodic_kernel(x, y))


# ============================================================================
# 5. Arrhenius Prior Tests
# ============================================================================


class TestArrheniusPrior:
    """Tests for ArrheniusPrior."""

    def test_known_value(self) -> None:
        """Check Arrhenius at T=300K with default parameters."""
        prior = ArrheniusPrior(A=1.0, Ea=50000.0, R=8.314)
        result = prior([[300.0]])
        expected = math.exp(-50000.0 / (8.314 * 300.0))
        assert result[0] == pytest.approx(expected, rel=1e-8)

    def test_monotonicity(self) -> None:
        """Arrhenius rate increases with temperature (positive Ea)."""
        prior = ArrheniusPrior(A=1.0, Ea=50000.0)
        temps = [[T] for T in [300.0, 400.0, 500.0, 600.0]]
        values = prior(temps)
        for i in range(len(values) - 1):
            assert values[i + 1] > values[i]

    def test_zero_temperature(self) -> None:
        """Arrhenius at T=0 returns 0 (avoids division by zero)."""
        prior = ArrheniusPrior()
        result = prior([[0.0]])
        assert result[0] == 0.0

    def test_negative_temperature(self) -> None:
        """Arrhenius at negative T returns 0."""
        prior = ArrheniusPrior()
        result = prior([[-100.0]])
        assert result[0] == 0.0

    def test_gradient_positive(self) -> None:
        """Gradient d/dT is positive for positive T and positive Ea."""
        prior = ArrheniusPrior(A=1.0, Ea=50000.0)
        grad = prior.gradient([[300.0], [500.0]])
        assert grad[0] > 0.0
        assert grad[1] > 0.0

    def test_temp_index(self) -> None:
        """temp_index selects the correct column."""
        prior = ArrheniusPrior(A=1.0, Ea=50000.0)
        # Temperature in column 1
        X = [[0.0, 300.0], [0.0, 500.0]]
        result = prior(X, temp_index=1)
        expected_300 = math.exp(-50000.0 / (8.314 * 300.0))
        assert result[0] == pytest.approx(expected_300, rel=1e-8)


# ============================================================================
# 6. Michaelis-Menten Prior Tests
# ============================================================================


class TestMichaelisMentenPrior:
    """Tests for MichaelisMentenPrior."""

    def test_saturation_behavior(self) -> None:
        """When S >> Km, rate approaches Vmax."""
        prior = MichaelisMentenPrior(Vmax=10.0, Km=1.0)
        result = prior([[10000.0]])
        assert result[0] == pytest.approx(10.0, rel=1e-3)

    def test_half_max_at_km(self) -> None:
        """At S = Km, rate = Vmax / 2."""
        prior = MichaelisMentenPrior(Vmax=10.0, Km=5.0)
        result = prior([[5.0]])
        assert result[0] == pytest.approx(5.0)

    def test_zero_substrate(self) -> None:
        """At S = 0, rate = 0."""
        prior = MichaelisMentenPrior(Vmax=10.0, Km=1.0)
        result = prior([[0.0]])
        assert result[0] == pytest.approx(0.0)

    def test_monotonicity(self) -> None:
        """Rate increases with substrate concentration."""
        prior = MichaelisMentenPrior(Vmax=10.0, Km=1.0)
        substrates = [[S] for S in [0.1, 1.0, 5.0, 50.0, 500.0]]
        values = prior(substrates)
        for i in range(len(values) - 1):
            assert values[i + 1] > values[i]

    def test_substrate_index(self) -> None:
        """substrate_index selects the correct column."""
        prior = MichaelisMentenPrior(Vmax=10.0, Km=1.0)
        X = [[99.0, 1.0]]  # substrate in column 1
        result = prior(X, substrate_index=1)
        assert result[0] == pytest.approx(5.0)

    def test_gradient_positive(self) -> None:
        """Gradient is positive for positive S."""
        prior = MichaelisMentenPrior(Vmax=10.0, Km=1.0)
        grad = prior.gradient([[1.0], [10.0]])
        assert grad[0] > 0.0
        assert grad[1] > 0.0


# ============================================================================
# 7. Power Law Prior Tests
# ============================================================================


class TestPowerLawPrior:
    """Tests for PowerLawPrior."""

    def test_known_value(self) -> None:
        """a * x^b = 2 * 3^2 = 18."""
        prior = PowerLawPrior(a=2.0, b=2.0)
        result = prior([[3.0]])
        assert result[0] == pytest.approx(18.0)

    def test_linear_case(self) -> None:
        """b=1 gives linear: a * x = 3 * 5 = 15."""
        prior = PowerLawPrior(a=3.0, b=1.0)
        result = prior([[5.0]])
        assert result[0] == pytest.approx(15.0)

    def test_square_root(self) -> None:
        """b=0.5 gives square root: 1 * 4^0.5 = 2."""
        prior = PowerLawPrior(a=1.0, b=0.5)
        result = prior([[4.0]])
        assert result[0] == pytest.approx(2.0)

    def test_zero_input(self) -> None:
        """x=0 gives 0."""
        prior = PowerLawPrior(a=5.0, b=2.0)
        result = prior([[0.0]])
        assert result[0] == pytest.approx(0.0)

    def test_var_index(self) -> None:
        """var_index selects the correct column."""
        prior = PowerLawPrior(a=1.0, b=2.0)
        X = [[0.0, 4.0]]  # variable in column 1
        result = prior(X, var_index=1)
        assert result[0] == pytest.approx(16.0)


# ============================================================================
# 8. RK4 Solver Tests
# ============================================================================


class TestRK4Solver:
    """Tests for RK4Solver."""

    def test_exponential_decay(self) -> None:
        """dy/dt = -y, y(0) = 1 => y(t) = exp(-t).

        Check |y(1) - exp(-1)| < 1e-6 with 1000 steps.
        """
        solver = RK4Solver()

        def f(t: float, y: list[float]) -> list[float]:
            return [-y[0]]

        t_vals, y_vals = solver.solve(f, [1.0], (0.0, 1.0), n_steps=1000)

        # Check final value
        y_final = y_vals[-1][0]
        expected = math.exp(-1.0)
        assert abs(y_final - expected) < 1e-6

        # Check t range
        assert t_vals[0] == pytest.approx(0.0)
        assert t_vals[-1] == pytest.approx(1.0)

    def test_simple_harmonic_oscillator(self) -> None:
        """2D system: dx/dt = v, dv/dt = -x.

        x(0) = 1, v(0) = 0 => x(t) = cos(t), v(t) = -sin(t).
        Check at t = pi.
        """
        solver = RK4Solver()

        def f(t: float, y: list[float]) -> list[float]:
            return [y[1], -y[0]]

        t_vals, y_vals = solver.solve(f, [1.0, 0.0], (0.0, math.pi), n_steps=1000)

        x_final = y_vals[-1][0]
        v_final = y_vals[-1][1]

        # cos(pi) = -1, -sin(pi) = 0
        assert abs(x_final - math.cos(math.pi)) < 1e-5
        assert abs(v_final - (-math.sin(math.pi))) < 1e-5

    def test_linear_growth(self) -> None:
        """dy/dt = 1, y(0) = 0 => y(t) = t. Check y(5) = 5."""
        solver = RK4Solver()

        def f(t: float, y: list[float]) -> list[float]:
            return [1.0]

        t_vals, y_vals = solver.solve(f, [0.0], (0.0, 5.0), n_steps=100)
        assert y_vals[-1][0] == pytest.approx(5.0, abs=1e-10)

    def test_output_length(self) -> None:
        """Output has n_steps + 1 entries (including initial condition)."""
        solver = RK4Solver()
        n_steps = 50

        def f(t: float, y: list[float]) -> list[float]:
            return [0.0]

        t_vals, y_vals = solver.solve(f, [1.0], (0.0, 1.0), n_steps=n_steps)
        assert len(t_vals) == n_steps + 1
        assert len(y_vals) == n_steps + 1

    def test_steady_state_convergence(self) -> None:
        """dy/dt = -10*(y - 3), y(0) = 0 => steady state y = 3."""
        solver = RK4Solver()

        def f(t: float, y: list[float]) -> list[float]:
            return [-10.0 * (y[0] - 3.0)]

        y_ss = solver.solve_to_steady_state(f, [0.0], dt=0.01, tol=1e-8)
        assert y_ss[0] == pytest.approx(3.0, abs=1e-4)

    def test_steady_state_multidim(self) -> None:
        """2D system converging to (1, 2):
        dy0/dt = -5*(y0 - 1)
        dy1/dt = -5*(y1 - 2)
        """
        solver = RK4Solver()

        def f(t: float, y: list[float]) -> list[float]:
            return [-5.0 * (y[0] - 1.0), -5.0 * (y[1] - 2.0)]

        y_ss = solver.solve_to_steady_state(f, [10.0, -5.0], dt=0.01, tol=1e-8)
        assert y_ss[0] == pytest.approx(1.0, abs=1e-4)
        assert y_ss[1] == pytest.approx(2.0, abs=1e-4)


# ============================================================================
# 9. Physics Constraints Tests
# ============================================================================


class TestPhysicsConstraints:
    """Tests for PhysicsConstraintModel."""

    def test_bound_check_feasible(self) -> None:
        """Point within bounds is feasible."""
        model = PhysicsConstraintModel()
        model.add_bound("temperature", lower=0.0, upper=1000.0, reason="physical")
        feasible, violations = model.check_feasibility({"temperature": 300.0})
        assert feasible is True
        assert violations == []

    def test_bound_check_violated(self) -> None:
        """Point outside bounds is infeasible."""
        model = PhysicsConstraintModel()
        model.add_bound("temperature", lower=0.0, upper=1000.0, reason="physical limit")
        feasible, violations = model.check_feasibility({"temperature": 1500.0})
        assert feasible is False
        assert len(violations) == 1
        assert "upper" in violations[0]

    def test_conservation_law_feasible(self) -> None:
        """Point satisfying conservation law is feasible."""
        model = PhysicsConstraintModel()
        model.add_conservation_law(
            "mass_balance", ["A", "B", "C"], target_sum=1.0
        )
        feasible, violations = model.check_feasibility(
            {"A": 0.3, "B": 0.3, "C": 0.4}
        )
        assert feasible is True
        assert violations == []

    def test_conservation_law_violated(self) -> None:
        """Point violating conservation law is infeasible."""
        model = PhysicsConstraintModel()
        model.add_conservation_law(
            "mass_balance", ["A", "B", "C"], target_sum=1.0
        )
        feasible, violations = model.check_feasibility(
            {"A": 0.5, "B": 0.5, "C": 0.5}
        )
        assert feasible is False
        assert len(violations) == 1
        assert "mass_balance" in violations[0]

    def test_monotonicity_increasing(self) -> None:
        """Increasing sequence passes monotonicity check."""
        model = PhysicsConstraintModel()
        model.add_monotonicity("x", "increasing")
        points = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        ok, violations = model.check_monotonicity(points, "x")
        assert ok is True
        assert violations == []

    def test_monotonicity_decreasing_violated(self) -> None:
        """Increasing sequence violates decreasing monotonicity."""
        model = PhysicsConstraintModel()
        model.add_monotonicity("x", "decreasing")
        points = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        ok, violations = model.check_monotonicity(points, "x")
        assert ok is False
        assert len(violations) > 0

    def test_monotonicity_no_constraint(self) -> None:
        """No monotonicity constraint on variable returns True."""
        model = PhysicsConstraintModel()
        model.add_monotonicity("y", "increasing")
        points = [{"x": 3.0}, {"x": 1.0}]
        ok, violations = model.check_monotonicity(points, "x")
        assert ok is True

    def test_bound_projection(self) -> None:
        """Projection clamps values to bounds."""
        model = PhysicsConstraintModel()
        model.add_bound("x", lower=0.0, upper=10.0)
        model.add_bound("y", lower=-5.0, upper=5.0)
        projected = model.project_to_feasible({"x": 15.0, "y": -10.0})
        assert projected["x"] == pytest.approx(10.0)
        assert projected["y"] == pytest.approx(-5.0)

    def test_conservation_projection(self) -> None:
        """Projection adjusts values to satisfy conservation law."""
        model = PhysicsConstraintModel()
        model.add_conservation_law("total", ["a", "b"], target_sum=1.0)
        projected = model.project_to_feasible({"a": 0.3, "b": 0.3})
        # Total is 0.6, deficit is 0.4, each gets +0.2
        assert projected["a"] + projected["b"] == pytest.approx(1.0)
        assert projected["a"] == pytest.approx(0.5)
        assert projected["b"] == pytest.approx(0.5)

    def test_serialization_roundtrip(self) -> None:
        """to_dict / from_dict roundtrip preserves model."""
        model = PhysicsConstraintModel()
        model.add_conservation_law("mass", ["A", "B"], target_sum=1.0, tolerance=1e-4)
        model.add_monotonicity("T", "increasing")
        model.add_bound("P", lower=0.0, upper=100.0, reason="pressure limit")

        d = model.to_dict()
        restored = PhysicsConstraintModel.from_dict(d)

        assert len(restored.conservation_laws) == 1
        assert restored.conservation_laws[0].name == "mass"
        assert restored.conservation_laws[0].target_sum == 1.0
        assert len(restored.monotonicity) == 1
        assert restored.monotonicity[0].direction == "increasing"
        assert len(restored.bounds) == 1
        assert restored.bounds[0].lower == 0.0
        assert restored.bounds[0].upper == 100.0

    def test_invalid_direction_raises(self) -> None:
        """Invalid monotonicity direction raises ValueError."""
        model = PhysicsConstraintModel()
        with pytest.raises(ValueError, match="direction must be"):
            model.add_monotonicity("x", "sideways")

    def test_combined_constraints(self) -> None:
        """Check feasibility with bounds and conservation together."""
        model = PhysicsConstraintModel()
        model.add_bound("A", lower=0.0, upper=1.0)
        model.add_bound("B", lower=0.0, upper=1.0)
        model.add_conservation_law("balance", ["A", "B"], target_sum=1.0)

        # Feasible point
        ok, _ = model.check_feasibility({"A": 0.6, "B": 0.4})
        assert ok is True

        # Violates bound
        ok, violations = model.check_feasibility({"A": 1.5, "B": -0.5})
        assert ok is False
        assert any("upper" in v for v in violations)
        assert any("lower" in v for v in violations)

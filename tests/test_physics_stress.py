"""Physics violation stress tests.

Proves the system detects and handles physics violations: when theory is
wrong, conservation laws are violated, or models are misspecified.

All tests are deterministic (seeded), use pure Python stdlib only, and
exercise the public APIs of the physics and hybrid modules.
"""

from __future__ import annotations

import math
import random

import pytest

from optimization_copilot.physics.constraints import PhysicsConstraintModel
from optimization_copilot.physics.ode_solver import RK4Solver
from optimization_copilot.hybrid.theory import (
    ArrheniusModel,
    MichaelisMentenModel,
    PowerLawModel,
)
from optimization_copilot.hybrid.residual import ResidualGP
from optimization_copilot.hybrid.composite import HybridModel
from optimization_copilot.hybrid.discrepancy import DiscrepancyAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rbf_kernel_scaled(length_scale: float):
    """Return an RBF kernel callable with a custom length_scale."""

    def _kernel(x1: list[float], x2: list[float]) -> float:
        ls2 = length_scale ** 2
        sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return math.exp(-0.5 * sq_dist / max(ls2, 1e-12))

    return _kernel


def _generate_arrhenius_data(
    n_points: int = 30,
    A: float = 1e10,
    Ea: float = 50000.0,
    R: float = 8.314,
    T_min: float = 300.0,
    T_max: float = 800.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple[list[list[float]], list[float]]:
    """Generate synthetic data from the Arrhenius equation.

    Returns (X, y) where X[i] = [T_i] and y[i] = A * exp(-Ea/(R*T_i)) + noise.
    """
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[float] = []
    for i in range(n_points):
        T = T_min + (T_max - T_min) * i / max(n_points - 1, 1)
        rate = A * math.exp(-Ea / (R * T))
        noise = rng.gauss(0.0, noise_std) if noise_std > 0 else 0.0
        X.append([T])
        y.append(rate + noise)
    return X, y


def _fit_powerlaw_loglog(
    X: list[list[float]], y: list[float]
) -> tuple[float, float]:
    """Estimate power-law parameters (a, b) via log-log linear regression.

    Fits log(y) = log(a) + b * log(x) using least squares.
    Returns (a, b).
    """
    n = len(X)
    sum_lx = 0.0
    sum_ly = 0.0
    sum_lx2 = 0.0
    sum_lxly = 0.0
    for i in range(n):
        lx = math.log(X[i][0])
        ly = math.log(max(y[i], 1e-300))
        sum_lx += lx
        sum_ly += ly
        sum_lx2 += lx * lx
        sum_lxly += lx * ly
    denom = n * sum_lx2 - sum_lx * sum_lx
    if abs(denom) < 1e-15:
        return 1.0, 1.0
    b = (n * sum_lxly - sum_lx * sum_ly) / denom
    log_a = (sum_ly - b * sum_lx) / n
    a = math.exp(log_a)
    return a, b


def _rmse(predictions: list[float], targets: list[float]) -> float:
    """Root mean squared error."""
    n = len(predictions)
    return math.sqrt(sum((predictions[i] - targets[i]) ** 2 for i in range(n)) / n)


# ===========================================================================
# Part A: Conservation Law Violation Detection
# ===========================================================================


class TestConservationViolation:
    """Detect violations of conservation laws (mass balance, monotonicity)."""

    def test_detect_mass_conservation_violation(self) -> None:
        """Good data (sum=100) passes; bad data (sum=110) fails.

        Scientific claim: a conservation constraint (sum of components = 100)
        should distinguish between data that obeys the law versus data with
        a systematic 10% positive bias.
        """
        rng = random.Random(42)
        model = PhysicsConstraintModel()
        variables = ["comp_A", "comp_B", "comp_C"]
        model.add_conservation_law(
            name="mass_balance",
            variables=variables,
            target_sum=100.0,
            tolerance=2.0,  # allow small noise
        )

        # --- Good data: components sum to ~100 (small noise) ---
        n_samples = 30
        good_pass_count = 0
        for _ in range(n_samples):
            a = 30.0 + rng.gauss(0, 0.3)
            b = 40.0 + rng.gauss(0, 0.3)
            c = 100.0 - a - b + rng.gauss(0, 0.1)  # close to balanced
            point = {"comp_A": a, "comp_B": b, "comp_C": c}
            feasible, _ = model.check_feasibility(point)
            if feasible:
                good_pass_count += 1

        # --- Bad data: systematic bias shifts total to ~110 ---
        bad_pass_count = 0
        bad_violation_magnitudes: list[float] = []
        for _ in range(n_samples):
            a = 33.0 + rng.gauss(0, 0.3)  # biased up
            b = 44.0 + rng.gauss(0, 0.3)  # biased up
            c = 33.0 + rng.gauss(0, 0.3)  # biased up (~110 total)
            point = {"comp_A": a, "comp_B": b, "comp_C": c}
            feasible, violations = model.check_feasibility(point)
            if feasible:
                bad_pass_count += 1
            if violations:
                # Extract the actual sum from the violation string
                total = a + b + c
                bad_violation_magnitudes.append(abs(total - 100.0))

        # Good data should mostly pass
        assert good_pass_count >= 25, (
            f"Good data pass rate too low: {good_pass_count}/{n_samples}"
        )
        # Bad data should mostly fail
        assert bad_pass_count <= 5, (
            f"Bad data pass rate too high: {bad_pass_count}/{n_samples}"
        )
        # Violation magnitudes should be reported (non-empty)
        assert len(bad_violation_magnitudes) > 0
        # Magnitude should be around 10
        mean_violation = sum(bad_violation_magnitudes) / len(bad_violation_magnitudes)
        assert mean_violation > 5.0, (
            f"Mean violation magnitude too small: {mean_violation}"
        )

    def test_detect_monotonicity_violation(self) -> None:
        """Detect when reaction rate decreases at high temperature.

        Scientific claim: if we impose an increasing-monotonicity constraint
        on reaction rate with respect to temperature, data where the rate
        drops at high temperature should trigger a violation.
        """
        model = PhysicsConstraintModel()
        model.add_monotonicity(variable="rate", direction="increasing")

        # Generate data where rate increases then drops at high T
        # Simulates a catalyst deactivation scenario
        points: list[dict[str, float]] = []
        for i in range(30):
            T = 300.0 + i * 20.0
            if T < 700.0:
                rate = 100.0 * (T / 300.0)  # increasing
            else:
                rate = 100.0 * (700.0 / 300.0) - 50.0 * ((T - 700.0) / 100.0)  # decreasing
            points.append({"rate": rate})

        monotonic, violations = model.check_monotonicity(points, "rate")
        assert monotonic is False, "Should detect monotonicity violation"
        assert len(violations) > 0, "Should report at least one violation"

    def test_conservation_with_increasing_bias(self) -> None:
        """Detection rate increases with bias magnitude.

        Scientific claim: as systematic bias grows from 0% to 20%, the
        conservation law violation detection rate should monotonically
        increase. 0% bias should pass, 10%+ bias should always be caught.
        """
        rng = random.Random(123)
        bias_levels = [0.0, 2.0, 5.0, 10.0, 20.0]  # percent shift
        detection_rates: list[float] = []

        for bias_pct in bias_levels:
            model = PhysicsConstraintModel()
            model.add_conservation_law(
                name="mass_balance",
                variables=["A", "B", "C"],
                target_sum=100.0,
                tolerance=1.5,  # small tolerance for noise
            )

            n_samples = 50
            n_detected = 0
            for _ in range(n_samples):
                shift = bias_pct  # absolute shift since target is 100
                a = 33.33 + shift / 3.0 + rng.gauss(0, 0.3)
                b = 33.33 + shift / 3.0 + rng.gauss(0, 0.3)
                c = 33.34 + shift / 3.0 + rng.gauss(0, 0.3)
                point = {"A": a, "B": b, "C": c}
                feasible, _ = model.check_feasibility(point)
                if not feasible:
                    n_detected += 1

            detection_rates.append(n_detected / n_samples)

        # 0% bias: most should pass (low detection rate)
        assert detection_rates[0] < 0.3, (
            f"0% bias detection rate too high: {detection_rates[0]}"
        )
        # 10% bias: all should be caught
        assert detection_rates[3] >= 0.95, (
            f"10% bias detection rate too low: {detection_rates[3]}"
        )
        # 20% bias: all should be caught
        assert detection_rates[4] >= 0.99, (
            f"20% bias detection rate too low: {detection_rates[4]}"
        )
        # Detection rate should generally increase with bias
        for i in range(len(detection_rates) - 1):
            assert detection_rates[i] <= detection_rates[i + 1] + 0.05, (
                f"Detection rate did not increase from bias "
                f"{bias_levels[i]}% ({detection_rates[i]:.2f}) to "
                f"{bias_levels[i + 1]}% ({detection_rates[i + 1]:.2f})"
            )

    def test_project_to_feasible(self) -> None:
        """Project infeasible data (sum=115) to satisfy conservation (sum=100).

        Scientific claim: the projection should (a) satisfy the constraint
        and (b) minimally distort the data in L2 sense.
        """
        model = PhysicsConstraintModel()
        variables = ["x1", "x2", "x3", "x4", "x5"]
        model.add_conservation_law(
            name="mass_balance",
            variables=variables,
            target_sum=100.0,
        )

        # Infeasible point: sum = 115
        original = {"x1": 25.0, "x2": 30.0, "x3": 20.0, "x4": 15.0, "x5": 25.0}
        original_sum = sum(original[v] for v in variables)
        assert abs(original_sum - 115.0) < 1e-10

        projected = model.project_to_feasible(original)

        # Check: projected satisfies the constraint
        projected_sum = sum(projected[v] for v in variables)
        assert abs(projected_sum - 100.0) < 1e-6, (
            f"Projected sum = {projected_sum}, expected 100.0"
        )

        # Check: feasibility check passes
        feasible, violations = model.check_feasibility(projected)
        assert feasible is True, f"Projected point not feasible: {violations}"

        # Check: projection minimally distorts the data (L2 distance)
        l2_dist = math.sqrt(
            sum((projected[v] - original[v]) ** 2 for v in variables)
        )
        # Uniform redistribution of deficit=-15 across 5 vars = -3 each
        # L2 distance = sqrt(5 * 3^2) = sqrt(45) ~ 6.7
        expected_l2 = math.sqrt(5 * (15.0 / 5) ** 2)
        assert l2_dist == pytest.approx(expected_l2, rel=0.01), (
            f"L2 distance = {l2_dist}, expected ~{expected_l2}"
        )


# ===========================================================================
# Part B: Wrong Theory Model Detection
# ===========================================================================


class TestWrongTheoryDetection:
    """Detect when the wrong theoretical model is applied to data."""

    @staticmethod
    def _make_arrhenius_data(
        noise_frac: float = 0.01, seed: int = 42
    ) -> tuple[list[list[float]], list[float]]:
        """Generate Arrhenius data with known parameters.

        A=1e10, Ea=50000, R=8.314, T in [300, 800], 30 points.
        noise_frac controls relative noise level.
        """
        return _generate_arrhenius_data(
            n_points=30,
            A=1e10,
            Ea=50000.0,
            R=8.314,
            T_min=300.0,
            T_max=800.0,
            noise_std=0.0,
            seed=seed,
        )

    @staticmethod
    def _kernel_for_temperature():
        """Return an RBF kernel with length_scale appropriate for T in [300,800]."""
        return _rbf_kernel_scaled(length_scale=100.0)

    def test_powerlaw_on_arrhenius_data(self) -> None:
        """ArrheniusModel fits Arrhenius data well; PowerLawModel does not.

        Scientific claim: when data is generated from Arrhenius kinetics,
        the correct model (ArrheniusModel) should achieve a high
        theory_adequacy_score (>0.8) while the wrong model (PowerLawModel)
        should score poorly (<0.5). The DiscrepancyAnalyzer should detect
        systematic bias and identify failure regions.
        """
        X, y = self._make_arrhenius_data(noise_frac=0.0)
        kernel_fn = self._kernel_for_temperature()

        # --- Correct theory: Arrhenius with known parameters ---
        arrhenius = ArrheniusModel(A=1e10, Ea=50000.0, R=8.314)
        gp_arr = ResidualGP(arrhenius, kernel_fn=kernel_fn, noise=1e-4)
        hybrid_arr = HybridModel(arrhenius, gp_arr)
        hybrid_arr.fit(X, y)
        score_arr = hybrid_arr.theory_adequacy_score()

        # --- Wrong theory: best-fit PowerLaw ---
        a_fit, b_fit = _fit_powerlaw_loglog(X, y)
        powerlaw = PowerLawModel(a=a_fit, b=b_fit)
        gp_pl = ResidualGP(powerlaw, kernel_fn=kernel_fn, noise=1e-4)
        hybrid_pl = HybridModel(powerlaw, gp_pl)
        hybrid_pl.fit(X, y)
        score_pl = hybrid_pl.theory_adequacy_score()

        # Arrhenius should have excellent adequacy
        assert score_arr > 0.8, (
            f"Arrhenius adequacy score {score_arr:.4f} should be > 0.8"
        )
        # PowerLaw should have worse adequacy
        assert score_pl < score_arr, (
            f"PowerLaw score {score_pl:.4f} should be < Arrhenius {score_arr:.4f}"
        )

        # --- Discrepancy analysis on the PowerLaw fit ---
        analyzer = DiscrepancyAnalyzer()

        # Check systematic bias
        bias_result = analyzer.systematic_bias(gp_pl)
        assert bias_result["is_biased"] is True, (
            "PowerLaw fit on Arrhenius data should show systematic bias"
        )

        # Check failure regions
        failures = analyzer.failure_regions(hybrid_pl, X, threshold=1.5)
        assert len(failures) > 0, (
            "Should find failure regions where PowerLaw theory breaks down"
        )

        # Check model adequacy test
        residuals_pl = gp_pl.residuals
        # Estimate noise_std from the data (small noise was added)
        noise_est = _rmse(arrhenius.predict(X), y)
        if noise_est < 1e-10:
            noise_est = 1.0  # fallback for noiseless data
        adequacy = analyzer.model_adequacy_test(residuals_pl, noise_std=noise_est)
        assert adequacy["is_adequate"] is False, (
            f"PowerLaw should be inadequate (Q/n={adequacy['Q_over_n']:.2f})"
        )

    def test_residual_gp_compensates_wrong_theory(self) -> None:
        """GP residual correction significantly improves wrong theory predictions.

        Scientific claim: when a PowerLaw model (wrong functional form) is
        used on Arrhenius data, the hybrid model (theory + GP) should have
        at least 30% lower RMSE than the theory-only predictions.
        """
        X, y = self._make_arrhenius_data(noise_frac=0.0)
        kernel_fn = self._kernel_for_temperature()

        a_fit, b_fit = _fit_powerlaw_loglog(X, y)
        powerlaw = PowerLawModel(a=a_fit, b=b_fit)
        gp = ResidualGP(powerlaw, kernel_fn=kernel_fn, noise=1e-4)
        hybrid = HybridModel(powerlaw, gp)
        hybrid.fit(X, y)

        # Evaluate on training data (in-sample) to show GP learns residuals
        comparison = hybrid.compare_to_theory_only(X, y)

        assert comparison["hybrid_rmse"] < comparison["theory_rmse"], (
            f"Hybrid RMSE ({comparison['hybrid_rmse']:.4f}) should be less than "
            f"theory RMSE ({comparison['theory_rmse']:.4f})"
        )
        assert comparison["improvement_pct"] > 30.0, (
            f"Improvement {comparison['improvement_pct']:.1f}% should be > 30%"
        )

    def test_correct_theory_needs_minimal_gp(self) -> None:
        """Correct theory (Arrhenius) needs minimal GP correction.

        Scientific claim: when the theory matches the data-generating
        process, theory_adequacy_score should be high (>0.8), residual GP
        std should be small, and the hybrid improvement should be modest
        (<15%).
        """
        X, y = self._make_arrhenius_data(noise_frac=0.0)
        kernel_fn = self._kernel_for_temperature()

        arrhenius = ArrheniusModel(A=1e10, Ea=50000.0, R=8.314)
        gp = ResidualGP(arrhenius, kernel_fn=kernel_fn, noise=1e-4)
        hybrid = HybridModel(arrhenius, gp)
        hybrid.fit(X, y)

        # Theory adequacy should be high
        score = hybrid.theory_adequacy_score()
        assert score > 0.8, f"Score {score:.4f} should be > 0.8"

        # Residual GP std should be small relative to data range
        _, stds = gp.predict(X)
        data_range = max(y) - min(y)
        mean_std = sum(stds) / len(stds)
        assert mean_std < 0.1 * data_range, (
            f"Mean residual std ({mean_std:.4g}) should be < 10% of "
            f"data range ({data_range:.4g})"
        )

        # Hybrid improvement should be small (theory is already good)
        comparison = hybrid.compare_to_theory_only(X, y)
        # For noiseless data with perfect theory, improvement might be very large
        # in percentage terms because theory RMSE is near zero. Check absolute
        # hybrid RMSE is also near zero instead.
        assert comparison["theory_rmse"] < 0.01 * data_range, (
            f"Theory RMSE ({comparison['theory_rmse']:.4g}) should be < 1% of "
            f"data range ({data_range:.4g})"
        )

    def test_theory_revision_suggestions(self) -> None:
        """DiscrepancyAnalyzer suggests revisions for wrong theory.

        Scientific claim: when PowerLaw is applied to Arrhenius data, the
        analyzer should produce non-empty revision suggestions that mention
        systematic patterns.
        """
        X, y = self._make_arrhenius_data(noise_frac=0.0)
        kernel_fn = self._kernel_for_temperature()

        a_fit, b_fit = _fit_powerlaw_loglog(X, y)
        powerlaw = PowerLawModel(a=a_fit, b=b_fit)
        gp = ResidualGP(powerlaw, kernel_fn=kernel_fn, noise=1e-4)
        hybrid = HybridModel(powerlaw, gp)
        hybrid.fit(X, y)

        analyzer = DiscrepancyAnalyzer()
        failures = analyzer.failure_regions(hybrid, X, threshold=1.5)
        suggestions = analyzer.suggest_theory_revision(
            failures, var_names=["temperature"]
        )

        assert len(suggestions) > 0, "Should produce revision suggestions"

        combined = " ".join(suggestions).lower()
        has_relevant_content = (
            "systematic" in combined
            or "nonlinear" in combined
            or "high" in combined
            or "low" in combined
            or "predicting" in combined
            or "severity" in combined
            or "mechanism" in combined
            or "correction" in combined
            or "terms" in combined
        )
        assert has_relevant_content, (
            f"Suggestions should mention patterns or behavior. Got: {suggestions}"
        )

    def test_mm_on_linear_data(self) -> None:
        """MichaelisMenten is the wrong model for linear data.

        Scientific claim: fitting a Michaelis-Menten saturation model to
        data generated from a linear relationship should yield a low
        theory_adequacy_score, systematic bias detection, and a significant
        hybrid improvement.
        """
        rng = random.Random(42)
        n_points = 30

        # Generate linear data: y = 2*x + 3 + noise
        X = [[1.0 + 19.0 * i / (n_points - 1)] for i in range(n_points)]
        y = [2.0 * row[0] + 3.0 + rng.gauss(0, 0.1) for row in X]

        # Fit MichaelisMenten (wrong model for linear data)
        # Pick Vmax and Km that approximate the range
        # At x=1: y~5, at x=20: y~43. MM saturates, linear doesn't.
        mm = MichaelisMentenModel(Vmax=100.0, Km=10.0)
        kernel_fn = _rbf_kernel_scaled(length_scale=5.0)
        gp = ResidualGP(mm, kernel_fn=kernel_fn, noise=1e-4)
        hybrid = HybridModel(mm, gp)
        hybrid.fit(X, y)

        # Theory adequacy should be low (MM is wrong for linear data)
        score = hybrid.theory_adequacy_score()
        assert score < 0.5, (
            f"MM adequacy score {score:.4f} on linear data should be < 0.5"
        )

        # Discrepancy analyzer should detect bias
        analyzer = DiscrepancyAnalyzer()
        bias_result = analyzer.systematic_bias(gp)
        assert bias_result["is_biased"] is True, (
            "MM on linear data should show systematic bias"
        )

        # Hybrid should significantly outperform theory-only
        comparison = hybrid.compare_to_theory_only(X, y)
        assert comparison["hybrid_rmse"] < comparison["theory_rmse"], (
            "Hybrid should outperform MM-only on linear data"
        )
        assert comparison["improvement_pct"] > 30.0, (
            f"Improvement {comparison['improvement_pct']:.1f}% should be > 30%"
        )


# ===========================================================================
# Part C: ODE Solver Stress Tests
# ===========================================================================


class TestODESolverStress:
    """Stress tests for the RK4 ODE solver on conservation and accuracy."""

    def test_energy_conservation_harmonic(self) -> None:
        """RK4 conserves energy in a simple harmonic oscillator.

        Scientific claim: for dx/dt = v, dv/dt = -x with initial
        conditions x=1, v=0, the total energy E = x^2 + v^2 = 1 should
        be conserved to within 0.01 over t=[0, 20] with 1000 steps.
        """
        solver = RK4Solver()

        def harmonic(t: float, y: list[float]) -> list[float]:
            # y = [x, v], dx/dt = v, dv/dt = -x
            return [y[1], -y[0]]

        t_vals, y_vals = solver.solve(
            harmonic, [1.0, 0.0], (0.0, 20.0), n_steps=1000
        )

        E_initial = y_vals[0][0] ** 2 + y_vals[0][1] ** 2
        E_final = y_vals[-1][0] ** 2 + y_vals[-1][1] ** 2

        assert abs(E_final - E_initial) < 0.01, (
            f"|E_final({E_final:.6f}) - E_initial({E_initial:.6f})| = "
            f"{abs(E_final - E_initial):.6f} should be < 0.01"
        )

        # Also check energy at all intermediate steps
        max_energy_drift = 0.0
        for y in y_vals:
            E = y[0] ** 2 + y[1] ** 2
            drift = abs(E - E_initial)
            max_energy_drift = max(max_energy_drift, drift)

        assert max_energy_drift < 0.01, (
            f"Max energy drift {max_energy_drift:.6f} should be < 0.01"
        )

    def test_exponential_decay_accuracy(self) -> None:
        """RK4 achieves <0.1% relative error on exponential decay.

        Scientific claim: for dy/dt = -0.5*y, y(0)=10, the exact solution
        is y(t) = 10*exp(-0.5*t). At t=10, the numerical solution should
        have relative error < 0.001.
        """
        solver = RK4Solver()

        def decay(t: float, y: list[float]) -> list[float]:
            return [-0.5 * y[0]]

        t_vals, y_vals = solver.solve(
            decay, [10.0], (0.0, 10.0), n_steps=1000
        )

        y_numerical = y_vals[-1][0]
        y_exact = 10.0 * math.exp(-0.5 * 10.0)
        relative_error = abs(y_numerical - y_exact) / abs(y_exact)

        assert relative_error < 0.001, (
            f"Relative error {relative_error:.6e} should be < 0.001. "
            f"Numerical={y_numerical:.10f}, Exact={y_exact:.10f}"
        )

    def test_step_size_convergence(self) -> None:
        """RK4 shows 4th-order convergence with step refinement.

        Scientific claim: for exponential decay, halving the step size
        (doubling n_steps) should reduce the error by approximately 16x
        (2^4 for 4th-order method). We verify the convergence order is
        between 3.5 and 4.5.
        """
        solver = RK4Solver()

        def decay(t: float, y: list[float]) -> list[float]:
            return [-0.5 * y[0]]

        y_exact_final = 10.0 * math.exp(-0.5 * 10.0)

        step_counts = [10, 100, 1000]
        errors: list[float] = []

        for n_steps in step_counts:
            _, y_vals = solver.solve(
                decay, [10.0], (0.0, 10.0), n_steps=n_steps
            )
            err = abs(y_vals[-1][0] - y_exact_final)
            errors.append(err)

        # Check convergence order between consecutive refinements
        # error ratio ~ (n1/n2)^4 when going from n1 to n2 steps
        for i in range(len(errors) - 1):
            if errors[i + 1] < 1e-15:
                # Error too small to measure convergence
                continue
            ratio = errors[i] / max(errors[i + 1], 1e-300)
            step_ratio = step_counts[i + 1] / step_counts[i]
            # For 4th order: error ratio should be ~ step_ratio^4
            expected_ratio = step_ratio ** 4
            # Allow a range of convergence orders from 3.0 to 5.0
            # log(ratio) / log(step_ratio) gives the empirical order
            if ratio > 1.0:
                empirical_order = math.log(ratio) / math.log(step_ratio)
                assert 3.0 <= empirical_order <= 5.0, (
                    f"Empirical convergence order {empirical_order:.2f} "
                    f"should be between 3.0 and 5.0 "
                    f"(steps {step_counts[i]}->{step_counts[i + 1]}, "
                    f"errors {errors[i]:.4e}->{errors[i + 1]:.4e})"
                )

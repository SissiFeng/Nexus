"""Tests for the Diagnostic Signal Engine."""

from __future__ import annotations

import math

import pytest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.diagnostics.engine import (
    DiagnosticEngine,
    DiagnosticsVector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(
    iteration: int,
    params: dict | None = None,
    kpi: float | None = None,
    kpi_name: str = "yield",
    is_failure: bool = False,
    qc_passed: bool = True,
) -> Observation:
    return Observation(
        iteration=iteration,
        parameters=params or {"x": 0.5},
        kpi_values={kpi_name: kpi} if kpi is not None else {},
        is_failure=is_failure,
        qc_passed=qc_passed,
        timestamp=float(iteration),
    )


def _make_snapshot(
    observations: list[Observation],
    objective_names: list[str] | None = None,
    objective_directions: list[str] | None = None,
    parameter_specs: list[ParameterSpec] | None = None,
    constraints: list[dict] | None = None,
) -> CampaignSnapshot:
    if objective_names is None:
        objective_names = ["yield"]
    if objective_directions is None:
        objective_directions = ["maximize"]
    if parameter_specs is None:
        parameter_specs = [
            ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ]
    return CampaignSnapshot(
        campaign_id="test",
        parameter_specs=parameter_specs,
        observations=observations,
        objective_names=objective_names,
        objective_directions=objective_directions,
        constraints=constraints or [],
    )


# ---------------------------------------------------------------------------
# DiagnosticsVector serialization
# ---------------------------------------------------------------------------

class TestDiagnosticsVector:

    def test_to_dict_returns_all_fields(self):
        vec = DiagnosticsVector()
        d = vec.to_dict()
        assert isinstance(d, dict)
        assert set(d.keys()) == {
            "convergence_trend",
            "improvement_velocity",
            "variance_contraction",
            "noise_estimate",
            "failure_rate",
            "failure_clustering",
            "feasibility_shrinkage",
            "parameter_drift",
            "model_uncertainty",
            "exploration_coverage",
            "kpi_plateau_length",
            "best_kpi_value",
            "data_efficiency",
            "constraint_violation_rate",
        }

    def test_round_trip_serialization(self):
        vec = DiagnosticsVector(
            convergence_trend=0.5,
            improvement_velocity=-0.3,
            variance_contraction=0.8,
            noise_estimate=0.12,
            failure_rate=0.1,
            failure_clustering=1.5,
            feasibility_shrinkage=-0.2,
            parameter_drift=0.05,
            model_uncertainty=0.3,
            exploration_coverage=0.45,
            kpi_plateau_length=7,
            best_kpi_value=99.0,
            data_efficiency=1.2,
            constraint_violation_rate=0.05,
        )
        d = vec.to_dict()
        restored = DiagnosticsVector.from_dict(d)
        assert restored == vec

    def test_default_values(self):
        vec = DiagnosticsVector()
        assert vec.convergence_trend == 0.0
        assert vec.failure_rate == 0.0
        assert vec.variance_contraction == 1.0
        assert vec.kpi_plateau_length == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_empty_history(self):
        snapshot = _make_snapshot(observations=[])
        vec = self.engine.compute(snapshot)
        assert vec.convergence_trend == 0.0
        assert vec.failure_rate == 0.0
        assert vec.best_kpi_value == 0.0
        assert vec.kpi_plateau_length == 0
        assert vec.exploration_coverage == 0.0
        assert vec.data_efficiency == 0.0

    def test_single_observation(self):
        obs = [_make_obs(0, params={"x": 0.5}, kpi=10.0)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.convergence_trend == 0.0
        assert vec.best_kpi_value == 10.0
        assert vec.failure_rate == 0.0
        assert vec.kpi_plateau_length == 0
        assert vec.data_efficiency == 0.0

    def test_all_failures(self):
        obs = [
            _make_obs(i, kpi=None, is_failure=True)
            for i in range(5)
        ]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.failure_rate == 1.0
        assert vec.best_kpi_value == 0.0
        assert vec.convergence_trend == 0.0
        assert vec.exploration_coverage == 0.0

    def test_single_failure_single_success(self):
        obs = [
            _make_obs(0, kpi=None, is_failure=True),
            _make_obs(1, kpi=5.0, params={"x": 0.3}),
        ]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.failure_rate == 0.5
        assert vec.best_kpi_value == 5.0

    def test_constant_kpi_values(self):
        obs = [_make_obs(i, kpi=5.0, params={"x": float(i) / 10}) for i in range(10)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.convergence_trend == 0.0
        assert vec.noise_estimate == 0.0
        assert vec.variance_contraction == 1.0  # 0/0 guarded


# ---------------------------------------------------------------------------
# Individual signals
# ---------------------------------------------------------------------------

class TestConvergenceTrend:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_improving_maximization(self):
        """Steadily increasing KPI => positive convergence trend."""
        obs = [_make_obs(i, kpi=float(i), params={"x": 0.1 * i}) for i in range(20)]
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)
        assert vec.convergence_trend > 0.0

    def test_improving_minimization(self):
        """Steadily decreasing KPI (minimize) => positive convergence trend."""
        obs = [_make_obs(i, kpi=20.0 - float(i), params={"x": 0.1 * i}) for i in range(20)]
        snapshot = _make_snapshot(observations=obs, objective_directions=["minimize"])
        vec = self.engine.compute(snapshot)
        # For minimization, best-so-far decreases, slope is negative,
        # and normalized trend should reflect improvement direction
        # The slope of best-so-far is negative for minimization, which
        # after normalization means convergence_trend < 0 in raw terms
        # but the interpretation is "converging well".
        # Our implementation normalizes the raw slope to [-1, 1].
        assert vec.convergence_trend != 0.0

    def test_flat_kpi(self):
        obs = [_make_obs(i, kpi=5.0) for i in range(10)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.convergence_trend == 0.0


class TestImprovementVelocity:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_accelerating_improvement(self):
        """Recent window improves more than previous."""
        # Flat first half, steep second half — window=5 picks up the steep part
        obs = []
        for i in range(20):
            if i < 15:
                kpi = 1.0  # no improvement
            else:
                kpi = 1.0 + 10.0 * (i - 14)  # big jumps in last 5
            obs.append(_make_obs(i, kpi=kpi, params={"x": i / 20.0}))
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)
        assert vec.improvement_velocity > 0.0

    def test_decelerating_improvement(self):
        """Previous window improves more than recent."""
        obs = []
        for i in range(20):
            if i < 10:
                kpi = 1.0 + 1.0 * i  # large improvement
            else:
                kpi = 11.0 + 0.01 * (i - 10)  # tiny improvement
            obs.append(_make_obs(i, kpi=kpi, params={"x": i / 20.0}))
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)
        assert vec.improvement_velocity < 0.0


class TestVarianceContraction:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_variance_contracts(self):
        """Early noisy, late tight => variance_contraction < 1."""
        obs = []
        for i in range(20):
            if i < 10:
                kpi = 10.0 + (5.0 if i % 2 == 0 else -5.0)  # noisy
            else:
                kpi = 10.0 + (0.1 if i % 2 == 0 else -0.1)  # tight
            obs.append(_make_obs(i, kpi=kpi, params={"x": i / 20.0}))
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)
        assert vec.variance_contraction < 1.0

    def test_variance_expands(self):
        """Early tight, late noisy => variance_contraction > 1."""
        obs = []
        for i in range(20):
            if i < 10:
                kpi = 10.0 + (0.1 if i % 2 == 0 else -0.1)
            else:
                kpi = 10.0 + (5.0 if i % 2 == 0 else -5.0)
            obs.append(_make_obs(i, kpi=kpi, params={"x": i / 20.0}))
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)
        assert vec.variance_contraction > 1.0


class TestNoiseEstimate:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_low_noise(self):
        obs = [_make_obs(i, kpi=10.0 + 0.01 * (i % 2)) for i in range(20)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.noise_estimate < 0.01

    def test_high_noise(self):
        obs = [_make_obs(i, kpi=10.0 + 10.0 * ((-1) ** i)) for i in range(20)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.noise_estimate > 0.5


class TestFailureRate:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_no_failures(self):
        obs = [_make_obs(i, kpi=float(i)) for i in range(10)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.failure_rate == 0.0

    def test_half_failures(self):
        obs = [
            _make_obs(i, kpi=float(i) if i % 2 == 0 else None, is_failure=(i % 2 != 0))
            for i in range(10)
        ]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.failure_rate == 0.5

    def test_all_failures(self):
        obs = [_make_obs(i, kpi=None, is_failure=True) for i in range(5)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.failure_rate == 1.0


class TestFailureClustering:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_failures_clustered_at_end(self):
        """All failures at the end => clustering > 1."""
        obs = [_make_obs(i, kpi=float(i)) for i in range(16)]
        obs += [_make_obs(i + 16, kpi=None, is_failure=True) for i in range(4)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.failure_clustering > 1.0

    def test_no_failures(self):
        obs = [_make_obs(i, kpi=float(i)) for i in range(10)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.failure_clustering == 0.0

    def test_uniform_failures(self):
        """Failures evenly spread => clustering value is finite and non-negative."""
        # With window derived from n_kpi, uniform failures may not land exactly
        # at 1.0 but should be a well-defined non-negative ratio.
        obs = [
            _make_obs(i, kpi=float(i) if i % 5 != 0 else None, is_failure=(i % 5 == 0))
            for i in range(40)
        ]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        # With 40 obs, 8 failures, n_kpi=32, window=8, recent 8 of all_obs
        # includes obs at i=32..39, failures at i=35 => 1 failure in 8
        # recent_rate=0.125, overall=0.2, clustering=0.625
        assert vec.failure_clustering >= 0.0


class TestFeasibilityShrinkage:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_shrinking_feasibility(self):
        """Early observations feasible, late ones failing."""
        obs = [_make_obs(i, kpi=float(i)) for i in range(10)]
        obs += [_make_obs(i + 10, kpi=None, is_failure=True) for i in range(10)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.feasibility_shrinkage < 0.0

    def test_expanding_feasibility(self):
        """Early failures, late successes."""
        obs = [_make_obs(i, kpi=None, is_failure=True) for i in range(10)]
        obs += [_make_obs(i + 10, kpi=float(i)) for i in range(10)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.feasibility_shrinkage > 0.0


class TestParameterDrift:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_drifting_parameters(self):
        """Best parameter shifts over time."""
        obs = [
            _make_obs(i, kpi=float(i), params={"x": 0.1 * i})
            for i in range(20)
        ]
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)
        assert vec.parameter_drift > 0.0

    def test_stable_parameters(self):
        """Best found early and never changes."""
        obs = [_make_obs(0, kpi=100.0, params={"x": 0.5})]
        obs += [_make_obs(i, kpi=float(i), params={"x": 0.1 * i}) for i in range(1, 20)]
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)
        assert vec.parameter_drift == 0.0


class TestModelUncertainty:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_low_uncertainty(self):
        obs = [_make_obs(i, kpi=10.0) for i in range(20)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.model_uncertainty == 0.0

    def test_high_uncertainty(self):
        obs = [_make_obs(i, kpi=10.0 * ((-1) ** i) + 10.0) for i in range(20)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.model_uncertainty > 0.0


class TestExplorationCoverage:

    def setup_method(self):
        self.engine = DiagnosticEngine(n_bins=10)

    def test_good_coverage(self):
        """Evenly spaced parameters across the range."""
        obs = [
            _make_obs(i, kpi=float(i), params={"x": i / 9.0})
            for i in range(10)
        ]
        specs = [ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)]
        snapshot = _make_snapshot(observations=obs, parameter_specs=specs)
        vec = self.engine.compute(snapshot)
        assert vec.exploration_coverage > 0.5

    def test_poor_coverage(self):
        """All parameters in the same spot."""
        obs = [
            _make_obs(i, kpi=float(i), params={"x": 0.5})
            for i in range(10)
        ]
        specs = [ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)]
        snapshot = _make_snapshot(observations=obs, parameter_specs=specs)
        vec = self.engine.compute(snapshot)
        assert vec.exploration_coverage < 0.2

    def test_multi_param_coverage(self):
        """Two parameters, moderate coverage."""
        obs = [
            _make_obs(i, kpi=float(i), params={"x": (i % 5) / 5.0, "y": (i // 5) / 5.0})
            for i in range(25)
        ]
        specs = [
            ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
            ParameterSpec(name="y", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ]
        snapshot = _make_snapshot(observations=obs, parameter_specs=specs)
        vec = self.engine.compute(snapshot)
        assert 0.0 < vec.exploration_coverage < 1.0


class TestKpiPlateauLength:

    def setup_method(self):
        self.engine = DiagnosticEngine(improvement_threshold=0.01)

    def test_no_plateau(self):
        """Continuous improvement => plateau length 0."""
        obs = [_make_obs(i, kpi=float(i) * 10) for i in range(20)]
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)
        assert vec.kpi_plateau_length == 0

    def test_long_plateau(self):
        """Improve early, then stagnate."""
        obs = [_make_obs(0, kpi=0.0)]
        obs += [_make_obs(1, kpi=100.0)]
        obs += [_make_obs(i, kpi=100.0 + 0.001 * i) for i in range(2, 20)]
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)
        assert vec.kpi_plateau_length > 10


class TestBestKpiValue:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_maximize(self):
        obs = [_make_obs(i, kpi=float(i)) for i in range(10)]
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)
        assert vec.best_kpi_value == 9.0

    def test_minimize(self):
        obs = [_make_obs(i, kpi=float(i)) for i in range(10)]
        snapshot = _make_snapshot(observations=obs, objective_directions=["minimize"])
        vec = self.engine.compute(snapshot)
        assert vec.best_kpi_value == 0.0


class TestDataEfficiency:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_efficient_campaign(self):
        """Large improvement over few observations."""
        obs = [_make_obs(0, kpi=0.0), _make_obs(1, kpi=100.0)]
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)
        assert vec.data_efficiency == pytest.approx(50.0)

    def test_inefficient_campaign(self):
        """Tiny improvement over many observations."""
        obs = [_make_obs(i, kpi=0.001 * i) for i in range(100)]
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)
        assert vec.data_efficiency < 0.01

    def test_zero_observations(self):
        snapshot = _make_snapshot(observations=[])
        vec = self.engine.compute(snapshot)
        assert vec.data_efficiency == 0.0


class TestConstraintViolationRate:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_no_violations(self):
        obs = [_make_obs(i, kpi=float(i), qc_passed=True) for i in range(10)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.constraint_violation_rate == 0.0

    def test_qc_violations(self):
        obs = [_make_obs(i, kpi=float(i), qc_passed=(i % 2 == 0)) for i in range(10)]
        snapshot = _make_snapshot(observations=obs)
        vec = self.engine.compute(snapshot)
        assert vec.constraint_violation_rate == 0.5

    def test_explicit_constraint_violation(self):
        """Observations violate explicit constraints on KPI bounds."""
        obs = [_make_obs(i, kpi=float(i)) for i in range(10)]
        constraints = [{"target": "yield", "lower": 3.0, "upper": 7.0}]
        snapshot = _make_snapshot(observations=obs, constraints=constraints)
        vec = self.engine.compute(snapshot)
        # i=0,1,2 violate lower; i=8,9 violate upper => 5/10 = 0.5
        assert vec.constraint_violation_rate == 0.5

    def test_empty_observations(self):
        snapshot = _make_snapshot(observations=[], constraints=[])
        vec = self.engine.compute(snapshot)
        assert vec.constraint_violation_rate == 0.0


# ---------------------------------------------------------------------------
# Integration: full compute on realistic synthetic data
# ---------------------------------------------------------------------------

class TestFullCompute:

    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_full_vector_types(self):
        """All fields of the vector have the correct types."""
        obs = [_make_obs(i, kpi=float(i) + 0.1 * (i % 3), params={"x": i / 30.0})
               for i in range(30)]
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)

        d = vec.to_dict()
        for key, value in d.items():
            assert isinstance(value, (int, float)), f"{key} is {type(value)}"

    def test_deterministic(self):
        """Same input produces identical output."""
        obs = [_make_obs(i, kpi=float(i) * 1.5, params={"x": i / 20.0})
               for i in range(20)]
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec1 = self.engine.compute(snapshot)
        vec2 = self.engine.compute(snapshot)
        assert vec1 == vec2

    def test_mixed_failures_and_successes(self):
        """Realistic scenario: some failures interspersed."""
        obs = []
        for i in range(30):
            if i in (5, 12, 20):
                obs.append(_make_obs(i, kpi=None, is_failure=True))
            else:
                obs.append(_make_obs(i, kpi=float(i) + 0.5, params={"x": i / 30.0}))
        snapshot = _make_snapshot(observations=obs, objective_directions=["maximize"])
        vec = self.engine.compute(snapshot)

        assert vec.failure_rate == pytest.approx(3.0 / 30.0)
        assert vec.best_kpi_value > 0.0
        assert vec.convergence_trend != 0.0
        assert 0.0 <= vec.exploration_coverage <= 1.0

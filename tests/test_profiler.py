"""Tests for the ProblemProfiler heuristic classifier.

Each fingerprint dimension is tested independently with synthetic
CampaignSnapshot data.
"""

from __future__ import annotations

from optimization_copilot.core.models import (
    CampaignSnapshot,
    CostProfile,
    DataScale,
    Dynamics,
    FailureInformativeness,
    FeasibleRegion,
    NoiseRegime,
    ObjectiveForm,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.profiler.profiler import ProblemProfiler


# ── Helpers ───────────────────────────────────────────────


def _make_snapshot(
    *,
    specs: list[ParameterSpec] | None = None,
    n_obs: int = 5,
    n_failures: int = 0,
    kpi_values_fn=None,
    objective_names: list[str] | None = None,
    objective_directions: list[str] | None = None,
    constraints: list[dict] | None = None,
    timestamp_fn=None,
    param_fn=None,
) -> CampaignSnapshot:
    """Flexible snapshot factory for profiler tests.

    Parameters
    ----------
    specs : list of ParameterSpec, optional
        Defaults to two continuous parameters.
    n_obs : int
        Total number of observations.
    n_failures : int
        Number of observations marked as failures (first n_failures).
    kpi_values_fn : callable(i: int) -> dict, optional
        Returns kpi_values dict for observation *i*.  Defaults to {"y": float(i)}.
    objective_names : list[str], optional
        Defaults to ["y"].
    objective_directions : list[str], optional
        Defaults to ["maximize"].
    constraints : list[dict], optional
        Defaults to empty.
    timestamp_fn : callable(i: int) -> float, optional
        Returns the timestamp for observation *i*.  Defaults to float(i).
    param_fn : callable(i: int) -> dict, optional
        Returns parameter dict for observation *i*.
    """
    if specs is None:
        specs = [
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
            ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=-1.0, upper=1.0),
        ]
    if objective_names is None:
        objective_names = ["y"]
    if objective_directions is None:
        objective_directions = ["maximize"]
    if constraints is None:
        constraints = []
    if kpi_values_fn is None:
        kpi_values_fn = lambda i: {"y": float(i)}
    if timestamp_fn is None:
        timestamp_fn = lambda i: float(i)
    if param_fn is None:
        param_fn = lambda i: {"x1": i * 0.1, "x2": i * -0.1}

    obs = []
    for i in range(n_obs):
        is_fail = i < n_failures
        obs.append(
            Observation(
                iteration=i,
                parameters=param_fn(i),
                kpi_values=kpi_values_fn(i),
                qc_passed=not is_fail,
                is_failure=is_fail,
                failure_reason="test_fail" if is_fail else None,
                timestamp=timestamp_fn(i),
            )
        )
    return CampaignSnapshot(
        campaign_id="profiler-test",
        parameter_specs=specs,
        observations=obs,
        objective_names=objective_names,
        objective_directions=objective_directions,
        constraints=constraints,
        current_iteration=n_obs,
    )


# ── Profiler instance shared across tests ────────────────

profiler = ProblemProfiler()


# ── Tests for variable_types ─────────────────────────────


class TestVariableTypes:
    def test_all_continuous(self):
        snap = _make_snapshot(
            specs=[
                ParameterSpec(name="a", type=VariableType.CONTINUOUS, lower=0, upper=1),
                ParameterSpec(name="b", type=VariableType.CONTINUOUS, lower=0, upper=1),
            ]
        )
        fp = profiler.profile(snap)
        assert fp.variable_types == VariableType.CONTINUOUS

    def test_all_discrete(self):
        snap = _make_snapshot(
            specs=[
                ParameterSpec(name="a", type=VariableType.DISCRETE, lower=1, upper=10),
                ParameterSpec(name="b", type=VariableType.DISCRETE, lower=1, upper=5),
            ]
        )
        fp = profiler.profile(snap)
        assert fp.variable_types == VariableType.DISCRETE

    def test_all_categorical(self):
        snap = _make_snapshot(
            specs=[
                ParameterSpec(name="a", type=VariableType.CATEGORICAL, categories=["x", "y"]),
                ParameterSpec(name="b", type=VariableType.CATEGORICAL, categories=["p", "q"]),
            ]
        )
        fp = profiler.profile(snap)
        assert fp.variable_types == VariableType.CATEGORICAL

    def test_mixed_types(self):
        snap = _make_snapshot(
            specs=[
                ParameterSpec(name="a", type=VariableType.CONTINUOUS, lower=0, upper=1),
                ParameterSpec(name="b", type=VariableType.CATEGORICAL, categories=["x", "y"]),
            ]
        )
        fp = profiler.profile(snap)
        assert fp.variable_types == VariableType.MIXED

    def test_empty_specs_defaults_to_continuous(self):
        snap = _make_snapshot(specs=[], param_fn=lambda i: {})
        fp = profiler.profile(snap)
        assert fp.variable_types == VariableType.CONTINUOUS


# ── Tests for objective_form ─────────────────────────────


class TestObjectiveForm:
    def test_single_objective(self):
        snap = _make_snapshot(objective_names=["y"])
        fp = profiler.profile(snap)
        assert fp.objective_form == ObjectiveForm.SINGLE

    def test_multi_objective(self):
        snap = _make_snapshot(
            objective_names=["y1", "y2"],
            objective_directions=["maximize", "minimize"],
            kpi_values_fn=lambda i: {"y1": float(i), "y2": float(i) * 0.5},
        )
        fp = profiler.profile(snap)
        assert fp.objective_form == ObjectiveForm.MULTI_OBJECTIVE

    def test_constrained(self):
        snap = _make_snapshot(
            constraints=[{"name": "c1", "type": "<=", "value": 5.0}],
        )
        fp = profiler.profile(snap)
        assert fp.objective_form == ObjectiveForm.CONSTRAINED

    def test_multi_objective_takes_precedence_over_constrained(self):
        """Multi-objective is checked before constraints."""
        snap = _make_snapshot(
            objective_names=["y1", "y2"],
            objective_directions=["maximize", "minimize"],
            constraints=[{"name": "c1", "type": "<=", "value": 5.0}],
            kpi_values_fn=lambda i: {"y1": float(i), "y2": float(i) * 0.5},
        )
        fp = profiler.profile(snap)
        assert fp.objective_form == ObjectiveForm.MULTI_OBJECTIVE


# ── Tests for noise_regime ───────────────────────────────


class TestNoiseRegime:
    def test_low_noise(self):
        """Constant KPI values produce CV ~ 0 -> LOW."""
        snap = _make_snapshot(
            n_obs=10,
            kpi_values_fn=lambda i: {"y": 10.0},
        )
        fp = profiler.profile(snap)
        assert fp.noise_regime == NoiseRegime.LOW

    def test_medium_noise(self):
        """Values around a mean with moderate spread -> MEDIUM."""
        # mean = 10.0, values spread such that CV is between 0.1 and 0.5
        # stdev ~ 2.0, mean = 10.0, CV = 0.2
        import math

        values = [10.0 + 2.0 * math.sin(i) for i in range(20)]
        snap = _make_snapshot(
            n_obs=20,
            kpi_values_fn=lambda i: {"y": values[i]},
        )
        fp = profiler.profile(snap)
        assert fp.noise_regime == NoiseRegime.MEDIUM

    def test_high_noise(self):
        """Widely varying values -> HIGH."""
        # Values: 1, 100, 1, 100, ... => mean ~ 50.5, stdev ~ 49.5, CV ~ 0.98
        snap = _make_snapshot(
            n_obs=20,
            kpi_values_fn=lambda i: {"y": 1.0 if i % 2 == 0 else 100.0},
        )
        fp = profiler.profile(snap)
        assert fp.noise_regime == NoiseRegime.HIGH

    def test_insufficient_data_defaults_low(self):
        """With fewer than 2 successful observations, default to LOW."""
        snap = _make_snapshot(n_obs=1, n_failures=0)
        fp = profiler.profile(snap)
        assert fp.noise_regime == NoiseRegime.LOW

    def test_all_failures_defaults_low(self):
        """No successful observations -> default to LOW."""
        snap = _make_snapshot(n_obs=3, n_failures=3)
        fp = profiler.profile(snap)
        assert fp.noise_regime == NoiseRegime.LOW


# ── Tests for cost_profile ───────────────────────────────


class TestCostProfile:
    def test_uniform_timestamps(self):
        """Evenly spaced timestamps -> UNIFORM."""
        snap = _make_snapshot(
            n_obs=10,
            timestamp_fn=lambda i: float(i * 5),
        )
        fp = profiler.profile(snap)
        assert fp.cost_profile == CostProfile.UNIFORM

    def test_heterogeneous_timestamps(self):
        """Highly variable gaps -> HETEROGENEOUS."""
        # Gaps: 1, 100, 1, 100, ...
        timestamps = [0.0]
        for i in range(19):
            gap = 1.0 if i % 2 == 0 else 100.0
            timestamps.append(timestamps[-1] + gap)

        snap = _make_snapshot(
            n_obs=20,
            timestamp_fn=lambda i: timestamps[i],
        )
        fp = profiler.profile(snap)
        assert fp.cost_profile == CostProfile.HETEROGENEOUS

    def test_too_few_observations_defaults_uniform(self):
        """Fewer than 3 observations -> UNIFORM."""
        snap = _make_snapshot(n_obs=2)
        fp = profiler.profile(snap)
        assert fp.cost_profile == CostProfile.UNIFORM

    def test_zero_timestamps_defaults_uniform(self):
        """All timestamps zero (not populated) -> UNIFORM."""
        snap = _make_snapshot(
            n_obs=10,
            timestamp_fn=lambda i: 0.0,
        )
        fp = profiler.profile(snap)
        assert fp.cost_profile == CostProfile.UNIFORM


# ── Tests for failure_informativeness ────────────────────


class TestFailureInformativeness:
    def test_strong_diverse_failures(self):
        """Failed observations with diverse parameter values -> STRONG."""
        snap = _make_snapshot(
            n_obs=10,
            n_failures=5,
            param_fn=lambda i: {"x1": float(i) * 0.2, "x2": float(i) * -0.3},
        )
        fp = profiler.profile(snap)
        assert fp.failure_informativeness == FailureInformativeness.STRONG

    def test_weak_identical_failures(self):
        """Failed observations all at the same point -> WEAK."""
        snap = _make_snapshot(
            n_obs=10,
            n_failures=5,
            param_fn=lambda i: {"x1": 0.5, "x2": -0.5} if i < 5 else {"x1": float(i) * 0.1, "x2": float(i) * -0.1},
        )
        fp = profiler.profile(snap)
        assert fp.failure_informativeness == FailureInformativeness.WEAK

    def test_zero_failures_defaults_weak(self):
        """No failures at all -> WEAK."""
        snap = _make_snapshot(n_obs=10, n_failures=0)
        fp = profiler.profile(snap)
        assert fp.failure_informativeness == FailureInformativeness.WEAK

    def test_single_failure_defaults_weak(self):
        """Only one failure -> WEAK (cannot assess diversity)."""
        snap = _make_snapshot(n_obs=10, n_failures=1)
        fp = profiler.profile(snap)
        assert fp.failure_informativeness == FailureInformativeness.WEAK


# ── Tests for data_scale ─────────────────────────────────


class TestDataScale:
    def test_tiny(self):
        snap = _make_snapshot(n_obs=5)
        fp = profiler.profile(snap)
        assert fp.data_scale == DataScale.TINY

    def test_tiny_boundary(self):
        snap = _make_snapshot(n_obs=9)
        fp = profiler.profile(snap)
        assert fp.data_scale == DataScale.TINY

    def test_small(self):
        snap = _make_snapshot(n_obs=10)
        fp = profiler.profile(snap)
        assert fp.data_scale == DataScale.SMALL

    def test_small_upper_boundary(self):
        snap = _make_snapshot(n_obs=49)
        fp = profiler.profile(snap)
        assert fp.data_scale == DataScale.SMALL

    def test_moderate(self):
        snap = _make_snapshot(n_obs=50)
        fp = profiler.profile(snap)
        assert fp.data_scale == DataScale.MODERATE

    def test_large_moderate(self):
        snap = _make_snapshot(n_obs=200)
        fp = profiler.profile(snap)
        assert fp.data_scale == DataScale.MODERATE

    def test_zero_observations(self):
        snap = _make_snapshot(n_obs=0)
        fp = profiler.profile(snap)
        assert fp.data_scale == DataScale.TINY


# ── Tests for dynamics ───────────────────────────────────


class TestDynamics:
    def test_static_random_values(self):
        """Uncorrelated values -> STATIC."""
        # Values crafted so that the lag-1 autocorrelation is near zero.
        # Consecutive moves: up, up, down, up, down, down, up, down, up
        # giving roughly balanced positive/negative cross-products.
        values = [2.0, 3.0, 4.0, 1.0, 3.5, 1.5, 0.5, 2.5, 1.0, 3.0]
        snap = _make_snapshot(
            n_obs=len(values),
            kpi_values_fn=lambda i: {"y": values[i]},
        )
        fp = profiler.profile(snap)
        assert fp.dynamics == Dynamics.STATIC

    def test_time_series_trending(self):
        """Monotonically increasing values have high autocorrelation -> TIME_SERIES."""
        snap = _make_snapshot(
            n_obs=20,
            kpi_values_fn=lambda i: {"y": float(i) * 1.5},
        )
        fp = profiler.profile(snap)
        assert fp.dynamics == Dynamics.TIME_SERIES

    def test_insufficient_data_defaults_static(self):
        """Fewer than 4 successful observations -> STATIC."""
        snap = _make_snapshot(n_obs=3)
        fp = profiler.profile(snap)
        assert fp.dynamics == Dynamics.STATIC

    def test_constant_values_are_static(self):
        """Constant series has zero variance -> STATIC."""
        snap = _make_snapshot(
            n_obs=10,
            kpi_values_fn=lambda i: {"y": 5.0},
        )
        fp = profiler.profile(snap)
        assert fp.dynamics == Dynamics.STATIC


# ── Tests for feasible_region ────────────────────────────


class TestFeasibleRegion:
    def test_wide_no_failures(self):
        snap = _make_snapshot(n_obs=20, n_failures=0)
        fp = profiler.profile(snap)
        assert fp.feasible_region == FeasibleRegion.WIDE

    def test_wide_low_failure_rate(self):
        """failure_rate = 1/20 = 0.05 < 0.1 -> WIDE."""
        snap = _make_snapshot(n_obs=20, n_failures=1)
        fp = profiler.profile(snap)
        assert fp.feasible_region == FeasibleRegion.WIDE

    def test_narrow(self):
        """failure_rate = 4/20 = 0.2 -> NARROW."""
        snap = _make_snapshot(n_obs=20, n_failures=4)
        fp = profiler.profile(snap)
        assert fp.feasible_region == FeasibleRegion.NARROW

    def test_fragmented(self):
        """failure_rate = 8/20 = 0.4 -> FRAGMENTED."""
        snap = _make_snapshot(n_obs=20, n_failures=8)
        fp = profiler.profile(snap)
        assert fp.feasible_region == FeasibleRegion.FRAGMENTED

    def test_boundary_narrow_at_0_1(self):
        """failure_rate = 2/20 = 0.1 -> exactly at boundary -> NARROW."""
        snap = _make_snapshot(n_obs=20, n_failures=2)
        fp = profiler.profile(snap)
        assert fp.feasible_region == FeasibleRegion.NARROW

    def test_boundary_fragmented_at_0_3(self):
        """failure_rate = 6/20 = 0.3 -> exactly at boundary -> FRAGMENTED."""
        snap = _make_snapshot(n_obs=20, n_failures=6)
        fp = profiler.profile(snap)
        assert fp.feasible_region == FeasibleRegion.FRAGMENTED

    def test_empty_observations_wide(self):
        """No observations -> failure_rate = 0.0 -> WIDE."""
        snap = _make_snapshot(n_obs=0)
        fp = profiler.profile(snap)
        assert fp.feasible_region == FeasibleRegion.WIDE


# ── Integration test ─────────────────────────────────────


class TestProfilerIntegration:
    def test_full_profile_returns_fingerprint(self):
        """Smoke test: profiler returns a complete ProblemFingerprint."""
        snap = _make_snapshot(n_obs=30, n_failures=3)
        fp = profiler.profile(snap)

        # Check that every field is populated with a valid enum member.
        assert isinstance(fp.variable_types, VariableType)
        assert isinstance(fp.objective_form, ObjectiveForm)
        assert isinstance(fp.noise_regime, NoiseRegime)
        assert isinstance(fp.cost_profile, CostProfile)
        assert isinstance(fp.failure_informativeness, FailureInformativeness)
        assert isinstance(fp.data_scale, DataScale)
        assert isinstance(fp.dynamics, Dynamics)
        assert isinstance(fp.feasible_region, FeasibleRegion)

    def test_to_dict_round_trip(self):
        """The fingerprint can be serialized to a dict of string values."""
        snap = _make_snapshot(n_obs=15)
        fp = profiler.profile(snap)
        d = fp.to_dict()
        assert len(d) == 8
        assert all(isinstance(v, str) for v in d.values())

    def test_to_tuple_length(self):
        """The fingerprint tuple has exactly 8 elements."""
        snap = _make_snapshot(n_obs=15)
        fp = profiler.profile(snap)
        assert len(fp.to_tuple()) == 8

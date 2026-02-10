"""Tests for the Auto Benchmark Generator (Capability 14).

Covers all public APIs in optimization_copilot.benchmark_generator.generator:
  - Module-level objective functions (_sphere, _rosenbrock, _ackley, _rastrigin)
  - Module-level helpers (_check_failure_zone, _check_constraints, _apply_drift)
  - LandscapeType enum
  - SyntheticObjective dataclass
  - BenchmarkSpec dataclass
  - BenchmarkGenerator class
"""

from __future__ import annotations

import math

import pytest

from optimization_copilot.benchmark_generator.generator import (
    BenchmarkGenerator,
    BenchmarkSpec,
    LandscapeType,
    SyntheticObjective,
    _ackley,
    _apply_drift,
    _check_constraints,
    _check_failure_zone,
    _rastrigin,
    _rosenbrock,
    _sphere,
)
from optimization_copilot.core.models import Phase, RiskPosture, VariableType
from optimization_copilot.validation.scenarios import GoldenScenario, ScenarioExpectation


# ── Objective functions ───────────────────────────────────


class TestSphere:
    def test_origin_returns_zero(self) -> None:
        assert _sphere([0.0, 0.0, 0.0]) == 0.0

    def test_known_values(self) -> None:
        # (1-0)^2 + (2-0)^2 + (3-0)^2 = 1 + 4 + 9 = 14
        assert _sphere([1.0, 2.0, 3.0]) == pytest.approx(14.0)

    def test_shift_moves_optimum(self) -> None:
        # At x=[1,1,1] with shift=1.0, each (xi - 1)^2 = 0
        assert _sphere([1.0, 1.0, 1.0], shift=1.0) == pytest.approx(0.0)

    def test_shift_nonzero_at_origin(self) -> None:
        # At x=[0,0] with shift=2.0: (0-2)^2 + (0-2)^2 = 8
        assert _sphere([0.0, 0.0], shift=2.0) == pytest.approx(8.0)


class TestRosenbrock:
    def test_optimum_at_ones(self) -> None:
        assert _rosenbrock([1.0, 1.0]) == pytest.approx(0.0)
        assert _rosenbrock([1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_known_value_at_origin(self) -> None:
        # For x=[0,0]: 100*(0-0)^2 + (1-0)^2 = 1
        assert _rosenbrock([0.0, 0.0]) == pytest.approx(1.0)

    def test_known_value_two_dim(self) -> None:
        # For x=[2,3]: 100*(3-4)^2 + (1-2)^2 = 100 + 1 = 101
        assert _rosenbrock([2.0, 3.0]) == pytest.approx(101.0)


class TestAckley:
    def test_origin_near_zero(self) -> None:
        # Ackley at origin should be very close to 0 (floating-point)
        result = _ackley([0.0, 0.0])
        assert abs(result) < 1e-10

    def test_positive_away_from_origin(self) -> None:
        result = _ackley([1.0, 1.0])
        assert result > 0.0

    def test_single_dimension(self) -> None:
        result = _ackley([0.0])
        assert abs(result) < 1e-10


class TestRastrigin:
    def test_origin_returns_zero(self) -> None:
        assert _rastrigin([0.0, 0.0]) == pytest.approx(0.0)

    def test_nonzero_away_from_origin(self) -> None:
        result = _rastrigin([0.5, 0.5])
        assert result > 0.0

    def test_known_value(self) -> None:
        # For x=[1]: 10*1 + (1 - 10*cos(2*pi*1)) = 10 + 1 - 10 = 1
        assert _rastrigin([1.0]) == pytest.approx(1.0)


# ── Module-level helpers ──────────────────────────────────


class TestCheckFailureZone:
    def test_inside_zone_returns_true(self) -> None:
        zones = [{"x0": (0.0, 1.0), "x1": (0.0, 1.0)}]
        params = {"x0": 0.5, "x1": 0.5}
        assert _check_failure_zone(params, zones) is True

    def test_outside_zone_returns_false(self) -> None:
        zones = [{"x0": (0.0, 0.3), "x1": (0.0, 0.3)}]
        params = {"x0": 0.5, "x1": 0.5}
        assert _check_failure_zone(params, zones) is False

    def test_partial_match_returns_false(self) -> None:
        # One param inside zone, one outside -> not triggered
        zones = [{"x0": (0.0, 1.0), "x1": (0.0, 0.2)}]
        params = {"x0": 0.5, "x1": 0.5}
        assert _check_failure_zone(params, zones) is False

    def test_missing_param_returns_false(self) -> None:
        zones = [{"x0": (0.0, 1.0), "x99": (0.0, 1.0)}]
        params = {"x0": 0.5}
        assert _check_failure_zone(params, zones) is False

    def test_empty_zones_returns_false(self) -> None:
        assert _check_failure_zone({"x0": 0.5}, []) is False


class TestCheckConstraints:
    def test_sum_bound_satisfied(self) -> None:
        constraints = [
            {"type": "sum_bound", "parameters": ["x0", "x1"], "bound": 2.0}
        ]
        assert _check_constraints({"x0": 0.5, "x1": 0.5}, constraints) is True

    def test_sum_bound_violated(self) -> None:
        constraints = [
            {"type": "sum_bound", "parameters": ["x0", "x1"], "bound": 0.5}
        ]
        assert _check_constraints({"x0": 0.5, "x1": 0.5}, constraints) is False

    def test_boundary_satisfied(self) -> None:
        constraints = [
            {"type": "boundary", "parameters": ["x0", "x1"], "bound": 1.0}
        ]
        assert _check_constraints({"x0": 0.5, "x1": 0.8}, constraints) is True

    def test_boundary_violated(self) -> None:
        constraints = [
            {"type": "boundary", "parameters": ["x0"], "bound": 1.0}
        ]
        assert _check_constraints({"x0": 1.5}, constraints) is False

    def test_empty_constraints(self) -> None:
        assert _check_constraints({"x0": 100.0}, []) is True


class TestApplyDrift:
    def test_zero_drift(self) -> None:
        assert _apply_drift(10.0, 0.0, 5) == pytest.approx(10.0)

    def test_positive_drift(self) -> None:
        # value + drift_rate * iteration = 10 + 0.5 * 4 = 12.0
        assert _apply_drift(10.0, 0.5, 4) == pytest.approx(12.0)

    def test_drift_at_iteration_zero(self) -> None:
        assert _apply_drift(5.0, 1.0, 0) == pytest.approx(5.0)


# ── LandscapeType enum ───────────────────────────────────


class TestLandscapeType:
    def test_values(self) -> None:
        assert LandscapeType.SPHERE.value == "sphere"
        assert LandscapeType.ROSENBROCK.value == "rosenbrock"
        assert LandscapeType.ACKLEY.value == "ackley"
        assert LandscapeType.RASTRIGIN.value == "rastrigin"

    def test_all_members(self) -> None:
        assert len(LandscapeType) == 4

    def test_construction_from_string(self) -> None:
        assert LandscapeType("sphere") is LandscapeType.SPHERE


# ── SyntheticObjective ───────────────────────────────────


class TestSyntheticObjective:
    def test_defaults(self) -> None:
        obj = SyntheticObjective(
            name="test", landscape_type=LandscapeType.SPHERE, n_dimensions=2
        )
        assert obj.noise_sigma == 0.0
        assert obj.failure_rate == 0.0
        assert obj.n_objectives == 1
        assert obj.drift_rate == 0.0
        assert obj.has_categorical is False
        assert obj.seed == 42

    def test_evaluate_simple(self) -> None:
        obj = SyntheticObjective(
            name="test", landscape_type=LandscapeType.SPHERE, n_dimensions=2
        )
        result = obj.evaluate({"x0": 0.0, "x1": 0.0})
        assert result["is_failure"] is False
        assert result["constraint_violated"] is False
        assert result["kpi_values"]["kpi_0"] == pytest.approx(0.0)

    def test_evaluate_with_noise_deterministic(self) -> None:
        obj = SyntheticObjective(
            name="noisy",
            landscape_type=LandscapeType.SPHERE,
            n_dimensions=2,
            noise_sigma=1.0,
            seed=99,
        )
        params = {"x0": 0.0, "x1": 0.0}
        r1 = obj.evaluate(params, iteration=0)
        r2 = obj.evaluate(params, iteration=0)
        # Same params + same iteration + same seed = same result
        assert r1["kpi_values"]["kpi_0"] == pytest.approx(
            r2["kpi_values"]["kpi_0"]
        )

    def test_evaluate_with_noise_varies_by_iteration(self) -> None:
        obj = SyntheticObjective(
            name="noisy",
            landscape_type=LandscapeType.SPHERE,
            n_dimensions=2,
            noise_sigma=1.0,
            seed=99,
        )
        params = {"x0": 0.0, "x1": 0.0}
        r1 = obj.evaluate(params, iteration=0)
        r2 = obj.evaluate(params, iteration=1)
        # Different iterations should generally differ (noise-dependent)
        # We use a large enough sigma that coincidence is extremely unlikely
        assert r1["kpi_values"]["kpi_0"] != r2["kpi_values"]["kpi_0"]

    def test_evaluate_failure_rate(self) -> None:
        obj = SyntheticObjective(
            name="fail",
            landscape_type=LandscapeType.SPHERE,
            n_dimensions=2,
            failure_rate=1.0,  # Always fail
            seed=42,
        )
        result = obj.evaluate({"x0": 0.5, "x1": 0.5})
        assert result["is_failure"] is True
        assert result["failure_reason"] == "random_failure"

    def test_evaluate_failure_zone(self) -> None:
        obj = SyntheticObjective(
            name="zone_fail",
            landscape_type=LandscapeType.SPHERE,
            n_dimensions=2,
            failure_zones=[{"x0": (0.0, 1.0), "x1": (0.0, 1.0)}],
        )
        result = obj.evaluate({"x0": 0.5, "x1": 0.5})
        assert result["is_failure"] is True
        assert result["failure_reason"] == "failure_zone"

    def test_evaluate_constraint_violated(self) -> None:
        obj = SyntheticObjective(
            name="constrained",
            landscape_type=LandscapeType.SPHERE,
            n_dimensions=2,
            constraints=[
                {"type": "sum_bound", "parameters": ["x0", "x1"], "bound": 0.5}
            ],
        )
        result = obj.evaluate({"x0": 0.5, "x1": 0.5})
        assert result["constraint_violated"] is True

    def test_evaluate_constraint_satisfied(self) -> None:
        obj = SyntheticObjective(
            name="constrained",
            landscape_type=LandscapeType.SPHERE,
            n_dimensions=2,
            constraints=[
                {"type": "sum_bound", "parameters": ["x0", "x1"], "bound": 5.0}
            ],
        )
        result = obj.evaluate({"x0": 0.5, "x1": 0.5})
        assert result["constraint_violated"] is False

    def test_evaluate_with_drift(self) -> None:
        obj = SyntheticObjective(
            name="drift",
            landscape_type=LandscapeType.SPHERE,
            n_dimensions=2,
            drift_rate=1.0,
        )
        params = {"x0": 0.0, "x1": 0.0}
        r0 = obj.evaluate(params, iteration=0)
        r10 = obj.evaluate(params, iteration=10)
        # At origin, sphere=0. Drift adds drift_rate * iteration = 0 vs 10
        assert r0["kpi_values"]["kpi_0"] == pytest.approx(0.0)
        assert r10["kpi_values"]["kpi_0"] == pytest.approx(10.0)

    def test_evaluate_multi_objective(self) -> None:
        obj = SyntheticObjective(
            name="mo",
            landscape_type=LandscapeType.SPHERE,
            n_dimensions=2,
            n_objectives=3,
        )
        result = obj.evaluate({"x0": 0.0, "x1": 0.0})
        kpis = result["kpi_values"]
        assert "kpi_0" in kpis
        assert "kpi_1" in kpis
        assert "kpi_2" in kpis
        # kpi_0 is sphere at origin = 0
        assert kpis["kpi_0"] == pytest.approx(0.0)
        # kpi_1 is _sphere(x, shift=1.0) at origin = (0-1)^2 + (0-1)^2 = 2
        assert kpis["kpi_1"] == pytest.approx(2.0)
        # kpi_2 is _sphere(x, shift=2.0) at origin = (0-2)^2 + (0-2)^2 = 8
        assert kpis["kpi_2"] == pytest.approx(8.0)

    def test_evaluate_categorical(self) -> None:
        obj = SyntheticObjective(
            name="cat",
            landscape_type=LandscapeType.SPHERE,
            n_dimensions=2,
            has_categorical=True,
            categorical_effect=5.0,
        )
        base = obj.evaluate({"x0": 0.0, "x1": 0.0, "category": "A"})
        best = obj.evaluate({"x0": 0.0, "x1": 0.0, "category": "best"})
        # "best" category subtracts categorical_effect
        assert best["kpi_values"]["kpi_0"] == pytest.approx(
            base["kpi_values"]["kpi_0"] - 5.0
        )

    def test_to_dict_from_dict_roundtrip(self) -> None:
        obj = SyntheticObjective(
            name="rt",
            landscape_type=LandscapeType.ROSENBROCK,
            n_dimensions=5,
            noise_sigma=0.1,
            failure_rate=0.05,
            n_objectives=2,
            drift_rate=0.01,
            has_categorical=True,
            categorical_effect=1.5,
            seed=123,
        )
        d = obj.to_dict()
        obj2 = SyntheticObjective.from_dict(d)
        assert obj2.name == obj.name
        assert obj2.landscape_type == obj.landscape_type
        assert obj2.n_dimensions == obj.n_dimensions
        assert obj2.noise_sigma == obj.noise_sigma
        assert obj2.failure_rate == obj.failure_rate
        assert obj2.n_objectives == obj.n_objectives
        assert obj2.drift_rate == obj.drift_rate
        assert obj2.has_categorical == obj.has_categorical
        assert obj2.categorical_effect == obj.categorical_effect
        assert obj2.seed == obj.seed


# ── BenchmarkSpec ─────────────────────────────────────────


class TestBenchmarkSpec:
    def test_defaults(self) -> None:
        spec = BenchmarkSpec()
        assert spec.dimensionality_range == (2, 10)
        assert len(spec.landscape_types) == 4
        assert spec.noise_levels == [0.0, 0.05, 0.3]
        assert spec.include_constraints is True
        assert spec.include_failures is True
        assert spec.include_multi_objective is True
        assert spec.include_non_stationary is True
        assert spec.n_observations_per_scenario == 30
        assert spec.seed == 42

    def test_custom_construction(self) -> None:
        spec = BenchmarkSpec(
            dimensionality_range=(3, 7),
            landscape_types=[LandscapeType.SPHERE],
            noise_levels=[0.0],
            include_constraints=False,
            include_failures=False,
            include_multi_objective=False,
            include_non_stationary=False,
            n_observations_per_scenario=10,
            seed=99,
        )
        assert spec.dimensionality_range == (3, 7)
        assert len(spec.landscape_types) == 1
        assert spec.seed == 99

    def test_to_dict_from_dict_roundtrip(self) -> None:
        spec = BenchmarkSpec(
            dimensionality_range=(5, 15),
            landscape_types=[LandscapeType.ACKLEY, LandscapeType.RASTRIGIN],
            noise_levels=[0.0, 0.1],
            variable_types=[VariableType.CONTINUOUS, VariableType.MIXED],
            include_constraints=False,
            seed=77,
        )
        d = spec.to_dict()
        spec2 = BenchmarkSpec.from_dict(d)
        assert spec2.dimensionality_range == spec.dimensionality_range
        assert spec2.landscape_types == spec.landscape_types
        assert spec2.noise_levels == spec.noise_levels
        assert spec2.variable_types == spec.variable_types
        assert spec2.include_constraints == spec.include_constraints
        assert spec2.seed == spec.seed


# ── BenchmarkGenerator ────────────────────────────────────


class TestBenchmarkGenerator:
    def test_constructor_seed(self) -> None:
        gen = BenchmarkGenerator(seed=123)
        assert gen._seed == 123

    def test_generate_scenario_structure(self) -> None:
        gen = BenchmarkGenerator(seed=42)
        obj = SyntheticObjective(
            name="basic",
            landscape_type=LandscapeType.SPHERE,
            n_dimensions=2,
        )
        scenario = gen.generate_scenario(obj, n_observations=10)
        assert isinstance(scenario, GoldenScenario)
        assert scenario.name == "synthetic_basic"
        assert scenario.snapshot is not None
        assert scenario.expectation is not None
        assert len(scenario.snapshot.observations) == 10
        assert scenario.snapshot.campaign_id == "bench_synthetic_basic"
        assert scenario.snapshot.objective_names == ["kpi_0"]
        assert scenario.snapshot.objective_directions == ["minimize"]
        assert scenario.snapshot.current_iteration == 10

    def test_generate_scenario_determinism(self) -> None:
        gen1 = BenchmarkGenerator(seed=42)
        gen2 = BenchmarkGenerator(seed=42)
        obj1 = SyntheticObjective(
            name="det", landscape_type=LandscapeType.SPHERE, n_dimensions=3
        )
        obj2 = SyntheticObjective(
            name="det", landscape_type=LandscapeType.SPHERE, n_dimensions=3
        )
        s1 = gen1.generate_scenario(obj1, n_observations=20)
        s2 = gen2.generate_scenario(obj2, n_observations=20)
        # Observations should be identical
        for o1, o2 in zip(s1.snapshot.observations, s2.snapshot.observations):
            assert o1.parameters == o2.parameters
            assert o1.kpi_values == o2.kpi_values
            assert o1.is_failure == o2.is_failure

    def test_generate_scenario_custom_name(self) -> None:
        gen = BenchmarkGenerator(seed=1)
        obj = SyntheticObjective(
            name="test", landscape_type=LandscapeType.ACKLEY, n_dimensions=2
        )
        scenario = gen.generate_scenario(
            obj, n_observations=5, scenario_name="custom_name"
        )
        assert scenario.name == "custom_name"

    def test_generate_scenario_observations_valid(self) -> None:
        gen = BenchmarkGenerator(seed=42)
        obj = SyntheticObjective(
            name="valid",
            landscape_type=LandscapeType.ROSENBROCK,
            n_dimensions=3,
        )
        scenario = gen.generate_scenario(obj, n_observations=15)
        for obs in scenario.snapshot.observations:
            assert "x0" in obs.parameters
            assert "x1" in obs.parameters
            assert "x2" in obs.parameters
            if not obs.is_failure:
                assert "kpi_0" in obs.kpi_values

    def test_generate_scenario_expectation_cold_start(self) -> None:
        gen = BenchmarkGenerator(seed=42)
        obj = SyntheticObjective(
            name="cold",
            landscape_type=LandscapeType.SPHERE,
            n_dimensions=2,
        )
        # Fewer than 10 observations -> COLD_START
        scenario = gen.generate_scenario(obj, n_observations=5)
        assert scenario.expectation.expected_phase == Phase.COLD_START

    def test_generate_scenario_expectation_learning(self) -> None:
        gen = BenchmarkGenerator(seed=42)
        obj = SyntheticObjective(
            name="learn",
            landscape_type=LandscapeType.SPHERE,
            n_dimensions=2,
        )
        # 30 observations, no failures -> LEARNING
        scenario = gen.generate_scenario(obj, n_observations=30)
        assert scenario.expectation.expected_phase == Phase.LEARNING

    def test_generate_from_spec(self) -> None:
        gen = BenchmarkGenerator(seed=42)
        spec = BenchmarkSpec(
            landscape_types=[LandscapeType.SPHERE],
            noise_levels=[0.0],
            variable_types=[VariableType.CONTINUOUS],
            include_constraints=True,
            include_failures=True,
            include_multi_objective=True,
            include_non_stationary=True,
            n_observations_per_scenario=10,
            seed=42,
        )
        scenarios = gen.generate_from_spec(spec)
        # 1 landscape * 1 noise * 1 var_type = 1 base + 4 optional families
        assert len(scenarios) == 5
        for s in scenarios:
            assert isinstance(s, GoldenScenario)
            assert len(s.snapshot.observations) == 10

    def test_generate_suite_minimal(self) -> None:
        gen = BenchmarkGenerator(seed=42)
        scenarios = gen.generate_suite("minimal")
        assert len(scenarios) > 0
        # At least one per landscape type in the base Cartesian product
        landscape_names = {s.snapshot.campaign_id for s in scenarios}
        for lt in LandscapeType:
            matching = [
                s for s in scenarios
                if lt.value in s.description.lower()
            ]
            assert len(matching) >= 1, f"No scenario for {lt.value}"

    def test_generate_suite_pairwise_larger_than_minimal(self) -> None:
        gen = BenchmarkGenerator(seed=42)
        minimal = gen.generate_suite("minimal")
        pairwise = gen.generate_suite("pairwise")
        assert len(pairwise) > len(minimal)

    def test_generate_suite_full_larger_than_pairwise(self) -> None:
        gen = BenchmarkGenerator(seed=42)
        pairwise = gen.generate_suite("pairwise")
        full = gen.generate_suite("full")
        assert len(full) > len(pairwise)

    def test_generate_suite_invalid_coverage(self) -> None:
        gen = BenchmarkGenerator(seed=42)
        with pytest.raises(ValueError, match="Unknown coverage level"):
            gen.generate_suite("nonexistent")

    def test_generate_suite_minimal_has_valid_snapshots(self) -> None:
        gen = BenchmarkGenerator(seed=42)
        scenarios = gen.generate_suite("minimal")
        for s in scenarios:
            snap = s.snapshot
            assert snap.campaign_id != ""
            assert len(snap.parameter_specs) > 0
            assert len(snap.observations) > 0
            assert len(snap.objective_names) > 0
            assert len(snap.objective_directions) > 0
            assert snap.current_iteration > 0

    def test_generate_suite_minimal_has_valid_expectations(self) -> None:
        gen = BenchmarkGenerator(seed=42)
        scenarios = gen.generate_suite("minimal")
        for s in scenarios:
            exp = s.expectation
            assert isinstance(exp, ScenarioExpectation)
            assert isinstance(exp.expected_phase, Phase)

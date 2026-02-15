"""Comprehensive tests for the benchmark_protocol package."""
from __future__ import annotations

import json
import math
import pytest

from optimization_copilot.benchmark_protocol.schema import (
    ParameterDefinition,
    ObjectiveDefinition,
    BenchmarkSchema,
)
from optimization_copilot.benchmark_protocol.protocol import (
    BenchmarkResult,
    SDLBenchmarkProtocol,
    SphereBenchmark,
)
from optimization_copilot.benchmark_protocol.exporters import (
    AtlasExporter,
    BayBEExporter,
    AxExporter,
)
from optimization_copilot.benchmark_protocol.leaderboard import (
    Leaderboard,
    LeaderboardEntry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema(
    name: str = "test_bench",
    n_params: int = 2,
    n_objectives: int = 1,
    evaluation_budget: int = 50,
    noise_level: float = 0.01,
) -> BenchmarkSchema:
    """Create a simple BenchmarkSchema for testing."""
    params = [
        ParameterDefinition(name=f"x{i}", type="continuous", lower=-5.0, upper=5.0)
        for i in range(n_params)
    ]
    objectives = [
        ObjectiveDefinition(name=f"obj{i}", direction="minimize", target=0.0)
        for i in range(n_objectives)
    ]
    return BenchmarkSchema(
        name=name,
        version="1.0",
        description="A test benchmark",
        domain="testing",
        parameters=params,
        objectives=objectives,
        evaluation_budget=evaluation_budget,
        noise_level=noise_level,
    )


def _make_result(
    benchmark_name: str = "test_bench",
    algorithm_name: str = "algo_a",
    best_value: float = 0.5,
    n_evaluations: int = 10,
) -> BenchmarkResult:
    """Create a simple BenchmarkResult for testing."""
    observations = [
        {"parameters": {"x0": float(i)}, "kpi_values": {"obj0": float(i)}}
        for i in range(n_evaluations)
    ]
    return BenchmarkResult(
        benchmark_name=benchmark_name,
        algorithm_name=algorithm_name,
        observations=observations,
        best_value=best_value,
        best_parameters={"x0": 0.0},
        total_cost=float(n_evaluations),
        wall_time_seconds=1.23,
        n_evaluations=n_evaluations,
        metadata={"seed": 42},
    )


# ===========================================================================
# Schema Tests (~12 tests)
# ===========================================================================


class TestParameterDefinition:
    """Tests for ParameterDefinition."""

    def test_continuous_parameter(self) -> None:
        p = ParameterDefinition(name="x", type="continuous", lower=0.0, upper=1.0)
        assert p.name == "x"
        assert p.type == "continuous"
        assert p.lower == 0.0
        assert p.upper == 1.0
        assert p.categories is None

    def test_categorical_parameter(self) -> None:
        p = ParameterDefinition(
            name="color", type="categorical", categories=["red", "green", "blue"]
        )
        assert p.name == "color"
        assert p.type == "categorical"
        assert p.categories == ["red", "green", "blue"]
        assert p.lower is None
        assert p.upper is None

    def test_discrete_parameter(self) -> None:
        p = ParameterDefinition(name="n", type="discrete", lower=1.0, upper=10.0)
        assert p.type == "discrete"
        assert p.lower == 1.0
        assert p.upper == 10.0

    def test_to_dict_continuous(self) -> None:
        p = ParameterDefinition(name="x", type="continuous", lower=-1.0, upper=1.0)
        d = p.to_dict()
        assert d == {"name": "x", "type": "continuous", "lower": -1.0, "upper": 1.0}

    def test_to_dict_categorical_omits_bounds(self) -> None:
        p = ParameterDefinition(name="c", type="categorical", categories=["a", "b"])
        d = p.to_dict()
        assert "lower" not in d
        assert "upper" not in d
        assert d["categories"] == ["a", "b"]

    def test_from_dict_roundtrip(self) -> None:
        original = ParameterDefinition(name="x", type="continuous", lower=0.0, upper=5.0)
        d = original.to_dict()
        restored = ParameterDefinition.from_dict(d)
        assert restored.name == original.name
        assert restored.type == original.type
        assert restored.lower == original.lower
        assert restored.upper == original.upper


class TestObjectiveDefinition:
    """Tests for ObjectiveDefinition."""

    def test_minimize_objective(self) -> None:
        o = ObjectiveDefinition(name="loss", direction="minimize", target=0.0)
        assert o.name == "loss"
        assert o.direction == "minimize"
        assert o.target == 0.0

    def test_maximize_objective_no_target(self) -> None:
        o = ObjectiveDefinition(name="yield", direction="maximize")
        assert o.direction == "maximize"
        assert o.target is None

    def test_to_dict_with_target(self) -> None:
        o = ObjectiveDefinition(name="loss", direction="minimize", target=0.0)
        d = o.to_dict()
        assert d == {"name": "loss", "direction": "minimize", "target": 0.0}

    def test_to_dict_without_target(self) -> None:
        o = ObjectiveDefinition(name="yield", direction="maximize")
        d = o.to_dict()
        assert "target" not in d

    def test_from_dict_roundtrip(self) -> None:
        original = ObjectiveDefinition(name="obj", direction="minimize", target=1.5)
        restored = ObjectiveDefinition.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.direction == original.direction
        assert restored.target == original.target


class TestBenchmarkSchema:
    """Tests for BenchmarkSchema."""

    def test_creation(self) -> None:
        schema = _make_schema()
        assert schema.name == "test_bench"
        assert schema.version == "1.0"
        assert len(schema.parameters) == 2
        assert len(schema.objectives) == 1
        assert schema.evaluation_budget == 50

    def test_to_json_from_json_roundtrip(self) -> None:
        original = _make_schema()
        json_str = original.to_json()
        restored = BenchmarkSchema.from_json(json_str)
        assert restored.name == original.name
        assert restored.version == original.version
        assert len(restored.parameters) == len(original.parameters)
        assert len(restored.objectives) == len(original.objectives)
        assert restored.evaluation_budget == original.evaluation_budget
        assert restored.noise_level == original.noise_level

    def test_to_dict_from_dict_roundtrip(self) -> None:
        original = _make_schema()
        d = original.to_dict()
        restored = BenchmarkSchema.from_dict(d)
        assert restored.name == original.name
        assert restored.domain == original.domain

    def test_to_json_is_valid_json(self) -> None:
        schema = _make_schema()
        json_str = schema.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["name"] == "test_bench"

    def test_validate_valid_schema(self) -> None:
        schema = _make_schema()
        errors = schema.validate()
        assert errors == []

    def test_validate_empty_name(self) -> None:
        schema = _make_schema(name="")
        errors = schema.validate()
        assert any("name" in e.lower() for e in errors)

    def test_validate_no_parameters(self) -> None:
        schema = _make_schema()
        schema.parameters = []
        errors = schema.validate()
        assert any("parameter" in e.lower() for e in errors)

    def test_validate_no_objectives(self) -> None:
        schema = _make_schema()
        schema.objectives = []
        errors = schema.validate()
        assert any("objective" in e.lower() for e in errors)

    def test_validate_invalid_param_type(self) -> None:
        schema = _make_schema()
        schema.parameters[0].type = "invalid_type"
        errors = schema.validate()
        assert any("invalid type" in e.lower() for e in errors)

    def test_validate_continuous_missing_bounds(self) -> None:
        schema = _make_schema()
        schema.parameters[0].lower = None
        errors = schema.validate()
        assert any("bound" in e.lower() for e in errors)

    def test_validate_lower_geq_upper(self) -> None:
        schema = _make_schema()
        schema.parameters[0].lower = 5.0
        schema.parameters[0].upper = 0.0
        errors = schema.validate()
        assert any("less than" in e.lower() for e in errors)

    def test_validate_categorical_no_categories(self) -> None:
        schema = _make_schema()
        schema.parameters.append(
            ParameterDefinition(name="cat", type="categorical", categories=[])
        )
        errors = schema.validate()
        assert any("category" in e.lower() for e in errors)

    def test_validate_invalid_direction(self) -> None:
        schema = _make_schema()
        schema.objectives[0].direction = "invalid"
        errors = schema.validate()
        assert any("direction" in e.lower() for e in errors)

    def test_validate_negative_budget(self) -> None:
        schema = _make_schema()
        schema.evaluation_budget = -1
        errors = schema.validate()
        assert any("budget" in e.lower() for e in errors)

    def test_validate_negative_noise(self) -> None:
        schema = _make_schema()
        schema.noise_level = -0.1
        errors = schema.validate()
        assert any("noise" in e.lower() for e in errors)

    def test_schema_with_metadata(self) -> None:
        schema = _make_schema()
        schema.metadata = {"author": "test", "citation": "doi:xyz"}
        d = schema.to_dict()
        restored = BenchmarkSchema.from_dict(d)
        assert restored.metadata["author"] == "test"

    def test_schema_with_constraints(self) -> None:
        schema = _make_schema()
        schema.constraints = [{"type": "linear", "expression": "x0 + x1 <= 5"}]
        d = schema.to_dict()
        restored = BenchmarkSchema.from_dict(d)
        assert len(restored.constraints) == 1


# ===========================================================================
# Protocol Tests (~12 tests)
# ===========================================================================


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_creation(self) -> None:
        r = _make_result()
        assert r.benchmark_name == "test_bench"
        assert r.algorithm_name == "algo_a"
        assert r.best_value == 0.5

    def test_to_dict(self) -> None:
        r = _make_result()
        d = r.to_dict()
        assert d["benchmark_name"] == "test_bench"
        assert d["best_value"] == 0.5
        assert len(d["observations"]) == 10

    def test_from_dict_roundtrip(self) -> None:
        original = _make_result()
        restored = BenchmarkResult.from_dict(original.to_dict())
        assert restored.benchmark_name == original.benchmark_name
        assert restored.best_value == original.best_value
        assert restored.n_evaluations == original.n_evaluations


class TestSDLBenchmarkProtocol:
    """Tests for SDLBenchmarkProtocol and SphereBenchmark."""

    def test_sphere_evaluate_at_origin(self) -> None:
        bench = SphereBenchmark(n_dims=3, noise_level=0.0)
        result = bench.evaluate({"x0": 0.0, "x1": 0.0, "x2": 0.0})
        assert abs(result["objective"]) < 1e-10

    def test_sphere_evaluate_nonzero(self) -> None:
        bench = SphereBenchmark(n_dims=2, noise_level=0.0)
        result = bench.evaluate({"x0": 1.0, "x1": 2.0})
        # 1^2 + 2^2 = 5
        assert abs(result["objective"] - 5.0) < 1e-10

    def test_evaluate_adds_noise(self) -> None:
        bench = SphereBenchmark(n_dims=2, noise_level=0.5, seed=42)
        result = bench.evaluate({"x0": 0.0, "x1": 0.0})
        # With noise, result should not be exactly zero
        assert result["objective"] != 0.0

    def test_budget_tracking(self) -> None:
        bench = SphereBenchmark(n_dims=2, evaluation_budget=5, noise_level=0.0)
        assert bench.budget_remaining == 5
        bench.evaluate({"x0": 1.0, "x1": 1.0})
        assert bench.budget_remaining == 4

    def test_budget_exhaustion(self) -> None:
        bench = SphereBenchmark(n_dims=2, evaluation_budget=2, noise_level=0.0)
        bench.evaluate({"x0": 1.0, "x1": 1.0})
        bench.evaluate({"x0": 0.5, "x1": 0.5})
        with pytest.raises(RuntimeError, match="budget"):
            bench.evaluate({"x0": 0.0, "x1": 0.0})

    def test_history_recording(self) -> None:
        bench = SphereBenchmark(n_dims=2, evaluation_budget=10, noise_level=0.0)
        bench.evaluate({"x0": 1.0, "x1": 2.0})
        bench.evaluate({"x0": 0.0, "x1": 0.0})
        history = bench.history
        assert len(history) == 2
        assert history[0]["parameters"] == {"x0": 1.0, "x1": 2.0}
        assert history[1]["parameters"] == {"x0": 0.0, "x1": 0.0}

    def test_best_so_far_tracking(self) -> None:
        bench = SphereBenchmark(n_dims=2, evaluation_budget=10, noise_level=0.0)
        bench.evaluate({"x0": 3.0, "x1": 4.0})  # 25
        bench.evaluate({"x0": 1.0, "x1": 0.0})  # 1
        bench.evaluate({"x0": 2.0, "x1": 2.0})  # 8
        best = bench.best_so_far
        assert best is not None
        params, value = best
        assert abs(value - 1.0) < 1e-10
        assert params == {"x0": 1.0, "x1": 0.0}

    def test_best_so_far_none_before_evaluation(self) -> None:
        bench = SphereBenchmark(n_dims=2)
        assert bench.best_so_far is None

    def test_run_with_algorithm(self) -> None:
        bench = SphereBenchmark(n_dims=2, evaluation_budget=5, noise_level=0.0)

        def simple_algorithm(protocol: SDLBenchmarkProtocol) -> None:
            for i in range(5):
                val = 2.0 - i * 0.4
                protocol.evaluate({"x0": val, "x1": val})

        result = bench.run(simple_algorithm, algorithm_name="simple")
        assert result.benchmark_name == "sphere"
        assert result.algorithm_name == "simple"
        assert result.n_evaluations == 5
        assert result.wall_time_seconds >= 0

    def test_reset(self) -> None:
        bench = SphereBenchmark(n_dims=2, evaluation_budget=5, noise_level=0.0)
        bench.evaluate({"x0": 1.0, "x1": 1.0})
        assert bench.budget_remaining == 4
        bench.reset()
        assert bench.budget_remaining == 5
        assert bench.best_so_far is None
        assert bench.history == []

    def test_validate_missing_parameters(self) -> None:
        bench = SphereBenchmark(n_dims=2, noise_level=0.0)
        with pytest.raises(ValueError, match="Missing"):
            bench.evaluate({"x0": 1.0})  # missing x1

    def test_validate_extra_parameters(self) -> None:
        bench = SphereBenchmark(n_dims=2, noise_level=0.0)
        with pytest.raises(ValueError, match="Unexpected"):
            bench.evaluate({"x0": 1.0, "x1": 1.0, "x2": 1.0})

    def test_validate_out_of_bounds(self) -> None:
        bench = SphereBenchmark(n_dims=2, noise_level=0.0)
        with pytest.raises(ValueError, match="outside"):
            bench.evaluate({"x0": 100.0, "x1": 0.0})

    def test_reproducibility_with_same_seed(self) -> None:
        bench1 = SphereBenchmark(n_dims=2, noise_level=0.5, seed=123)
        bench2 = SphereBenchmark(n_dims=2, noise_level=0.5, seed=123)
        r1 = bench1.evaluate({"x0": 1.0, "x1": 1.0})
        r2 = bench2.evaluate({"x0": 1.0, "x1": 1.0})
        assert r1["objective"] == r2["objective"]

    def test_schema_property(self) -> None:
        bench = SphereBenchmark(n_dims=3)
        assert bench.schema.name == "sphere"
        assert len(bench.schema.parameters) == 3

    def test_not_implemented_base(self) -> None:
        schema = _make_schema()
        protocol = SDLBenchmarkProtocol(schema)
        with pytest.raises(NotImplementedError):
            protocol.evaluate({"x0": 0.0, "x1": 0.0})


# ===========================================================================
# Exporter Tests (~15 tests)
# ===========================================================================


class TestAtlasExporter:
    """Tests for AtlasExporter."""

    def test_export_schema_basic(self) -> None:
        schema = _make_schema()
        exported = AtlasExporter.export_schema(schema)
        assert exported["name"] == "test_bench"
        assert len(exported["parameters"]) == 2
        assert len(exported["objectives"]) == 1

    def test_export_schema_parameter_format(self) -> None:
        schema = _make_schema()
        exported = AtlasExporter.export_schema(schema)
        p = exported["parameters"][0]
        assert p["name"] == "x0"
        assert p["type"] == "continuous"
        assert p["low"] == -5.0
        assert p["high"] == 5.0

    def test_export_schema_objective_format(self) -> None:
        schema = _make_schema()
        exported = AtlasExporter.export_schema(schema)
        o = exported["objectives"][0]
        assert o["name"] == "obj0"
        assert o["goal"] == "minimize"
        assert o["target"] == 0.0

    def test_export_schema_categorical(self) -> None:
        schema = _make_schema()
        schema.parameters.append(
            ParameterDefinition(name="solvent", type="categorical", categories=["water", "ethanol"])
        )
        exported = AtlasExporter.export_schema(schema)
        cat_param = exported["parameters"][-1]
        assert cat_param["type"] == "categorical"
        assert cat_param["options"] == ["water", "ethanol"]

    def test_export_result(self) -> None:
        result = _make_result()
        exported = AtlasExporter.export_result(result)
        assert exported["benchmark"] == "test_bench"
        assert exported["algorithm"] == "algo_a"
        assert len(exported["campaign"]) == 10
        assert exported["best"]["value"] == 0.5

    def test_export_result_campaign_format(self) -> None:
        result = _make_result()
        exported = AtlasExporter.export_result(result)
        entry = exported["campaign"][0]
        # Parameters and kpi_values are flattened into one dict
        assert "x0" in entry
        assert "obj0" in entry


class TestBayBEExporter:
    """Tests for BayBEExporter."""

    def test_export_schema_searchspace(self) -> None:
        schema = _make_schema()
        exported = BayBEExporter.export_schema(schema)
        assert "searchspace" in exported
        params = exported["searchspace"]["parameters"]
        assert len(params) == 2
        assert params[0]["type"] == "NumericalContinuousParameter"
        assert params[0]["bounds"] == [-5.0, 5.0]

    def test_export_schema_objective_config(self) -> None:
        schema = _make_schema()
        exported = BayBEExporter.export_schema(schema)
        obj = exported["objective"]
        assert obj["mode"] == "SINGLE"
        assert len(obj["targets"]) == 1
        assert obj["targets"][0]["mode"] == "MINIMIZE"

    def test_export_schema_multi_objective(self) -> None:
        schema = _make_schema(n_objectives=2)
        exported = BayBEExporter.export_schema(schema)
        assert exported["objective"]["mode"] == "DESIRABILITY"

    def test_export_schema_categorical_param(self) -> None:
        schema = _make_schema()
        schema.parameters.append(
            ParameterDefinition(name="cat", type="categorical", categories=["a", "b", "c"])
        )
        exported = BayBEExporter.export_schema(schema)
        cat = exported["searchspace"]["parameters"][-1]
        assert cat["type"] == "CategoricalParameter"
        assert cat["values"] == ["a", "b", "c"]

    def test_export_schema_discrete_param(self) -> None:
        schema = _make_schema()
        schema.parameters.append(
            ParameterDefinition(name="d", type="discrete", lower=1.0, upper=5.0)
        )
        exported = BayBEExporter.export_schema(schema)
        disc = exported["searchspace"]["parameters"][-1]
        assert disc["type"] == "NumericalDiscreteParameter"

    def test_export_result(self) -> None:
        result = _make_result()
        exported = BayBEExporter.export_result(result)
        assert exported["benchmark"] == "test_bench"
        assert exported["recommender"] == "algo_a"
        assert len(exported["measurements"]) == 10

    def test_export_result_measurement_format(self) -> None:
        result = _make_result()
        exported = BayBEExporter.export_result(result)
        m = exported["measurements"][0]
        assert "parameters" in m
        assert "targets" in m


class TestAxExporter:
    """Tests for AxExporter."""

    def test_export_schema_experiment(self) -> None:
        schema = _make_schema()
        exported = AxExporter.export_schema(schema)
        assert "experiment" in exported
        exp = exported["experiment"]
        assert exp["name"] == "test_bench"
        assert len(exp["parameters"]) == 2

    def test_export_schema_parameter_format(self) -> None:
        schema = _make_schema()
        exported = AxExporter.export_schema(schema)
        p = exported["experiment"]["parameters"][0]
        assert p["type"] == "range"
        assert p["value_type"] == "float"
        assert p["bounds"] == [-5.0, 5.0]

    def test_export_schema_discrete_as_int(self) -> None:
        schema = _make_schema()
        schema.parameters.append(
            ParameterDefinition(name="n", type="discrete", lower=1.0, upper=10.0)
        )
        exported = AxExporter.export_schema(schema)
        disc = exported["experiment"]["parameters"][-1]
        assert disc["value_type"] == "int"

    def test_export_schema_categorical_as_choice(self) -> None:
        schema = _make_schema()
        schema.parameters.append(
            ParameterDefinition(name="c", type="categorical", categories=["a", "b"])
        )
        exported = AxExporter.export_schema(schema)
        cat = exported["experiment"]["parameters"][-1]
        assert cat["type"] == "choice"
        assert cat["values"] == ["a", "b"]

    def test_export_schema_metrics(self) -> None:
        schema = _make_schema()
        exported = AxExporter.export_schema(schema)
        metrics = exported["experiment"]["metrics"]
        assert len(metrics) == 1
        assert metrics[0]["name"] == "obj0"
        assert metrics[0]["lower_is_better"] is True

    def test_export_schema_optimization_config(self) -> None:
        schema = _make_schema()
        exported = AxExporter.export_schema(schema)
        opt = exported["optimization_config"]
        assert opt["objective_name"] == "obj0"
        assert opt["minimize"] is True

    def test_export_result(self) -> None:
        result = _make_result()
        exported = AxExporter.export_result(result)
        assert exported["experiment_name"] == "test_bench"
        assert exported["algorithm"] == "algo_a"
        assert len(exported["trials"]) == 10

    def test_export_result_trial_format(self) -> None:
        result = _make_result()
        exported = AxExporter.export_result(result)
        trial = exported["trials"][0]
        assert trial["trial_index"] == 0
        assert "arm_parameters" in trial
        assert "metric_values" in trial
        assert trial["trial_status"] == "COMPLETED"


class TestExporterConsistency:
    """Cross-exporter consistency tests."""

    def test_all_exporters_preserve_benchmark_name(self) -> None:
        schema = _make_schema()
        atlas = AtlasExporter.export_schema(schema)
        baybe = BayBEExporter.export_schema(schema)
        ax = AxExporter.export_schema(schema)
        assert atlas["name"] == "test_bench"
        assert baybe["metadata"]["name"] == "test_bench"
        assert ax["experiment"]["name"] == "test_bench"

    def test_all_exporters_preserve_param_count(self) -> None:
        schema = _make_schema(n_params=4)
        atlas = AtlasExporter.export_schema(schema)
        baybe = BayBEExporter.export_schema(schema)
        ax = AxExporter.export_schema(schema)
        assert len(atlas["parameters"]) == 4
        assert len(baybe["searchspace"]["parameters"]) == 4
        assert len(ax["experiment"]["parameters"]) == 4

    def test_all_exporters_preserve_result_count(self) -> None:
        result = _make_result(n_evaluations=7)
        atlas = AtlasExporter.export_result(result)
        baybe = BayBEExporter.export_result(result)
        ax = AxExporter.export_result(result)
        assert len(atlas["campaign"]) == 7
        assert len(baybe["measurements"]) == 7
        assert len(ax["trials"]) == 7


# ===========================================================================
# Leaderboard Tests (~11 tests)
# ===========================================================================


class TestLeaderboard:
    """Tests for Leaderboard."""

    def test_creation(self) -> None:
        lb = Leaderboard("sphere", direction="minimize")
        assert lb.benchmark_name == "sphere"
        assert lb.direction == "minimize"

    def test_invalid_direction(self) -> None:
        with pytest.raises(ValueError, match="Invalid direction"):
            Leaderboard("sphere", direction="invalid")

    def test_add_result(self) -> None:
        lb = Leaderboard("test_bench")
        result = _make_result()
        lb.add_result(result)
        rankings = lb.get_rankings()
        assert len(rankings) == 1

    def test_add_result_wrong_benchmark(self) -> None:
        lb = Leaderboard("sphere")
        result = _make_result(benchmark_name="different")
        with pytest.raises(ValueError, match="does not match"):
            lb.add_result(result)

    def test_rankings_minimize(self) -> None:
        lb = Leaderboard("test_bench", direction="minimize")
        lb.add_result(_make_result(algorithm_name="algo_a", best_value=5.0))
        lb.add_result(_make_result(algorithm_name="algo_b", best_value=1.0))
        lb.add_result(_make_result(algorithm_name="algo_c", best_value=3.0))
        rankings = lb.get_rankings()
        assert rankings[0].algorithm_name == "algo_b"
        assert rankings[0].rank == 1
        assert rankings[0].best_value == 1.0
        assert rankings[1].algorithm_name == "algo_c"
        assert rankings[2].algorithm_name == "algo_a"

    def test_rankings_maximize(self) -> None:
        lb = Leaderboard("test_bench", direction="maximize")
        lb.add_result(_make_result(algorithm_name="algo_a", best_value=5.0))
        lb.add_result(_make_result(algorithm_name="algo_b", best_value=1.0))
        lb.add_result(_make_result(algorithm_name="algo_c", best_value=3.0))
        rankings = lb.get_rankings()
        assert rankings[0].algorithm_name == "algo_a"
        assert rankings[0].best_value == 5.0
        assert rankings[1].algorithm_name == "algo_c"
        assert rankings[2].algorithm_name == "algo_b"

    def test_empty_leaderboard_rankings(self) -> None:
        lb = Leaderboard("test_bench")
        assert lb.get_rankings() == []

    def test_empty_leaderboard_summary(self) -> None:
        lb = Leaderboard("test_bench")
        summary = lb.get_summary()
        assert summary["n_entries"] == 0
        assert summary["best_value"] is None

    def test_get_summary_with_results(self) -> None:
        lb = Leaderboard("test_bench", direction="minimize")
        lb.add_result(_make_result(algorithm_name="a", best_value=2.0))
        lb.add_result(_make_result(algorithm_name="b", best_value=4.0))
        summary = lb.get_summary()
        assert summary["n_entries"] == 2
        assert summary["best_value"] == 2.0
        assert summary["worst_value"] == 4.0
        assert summary["mean_value"] == 3.0

    def test_to_dict(self) -> None:
        lb = Leaderboard("test_bench")
        lb.add_result(_make_result(algorithm_name="a", best_value=1.0))
        d = lb.to_dict()
        assert d["benchmark_name"] == "test_bench"
        assert "rankings" in d
        assert "summary" in d
        assert len(d["rankings"]) == 1

    def test_render_text(self) -> None:
        lb = Leaderboard("test_bench", direction="minimize")
        lb.add_result(_make_result(algorithm_name="algo_a", best_value=2.5))
        lb.add_result(_make_result(algorithm_name="algo_b", best_value=1.0))
        text = lb.render_text()
        assert "test_bench" in text
        assert "algo_a" in text
        assert "algo_b" in text
        assert "minimize" in text
        # algo_b should be ranked first
        lines = text.strip().split("\n")
        data_lines = [l for l in lines if l.strip().startswith(("1", "2"))]
        assert "algo_b" in data_lines[0]

    def test_render_text_empty(self) -> None:
        lb = Leaderboard("test_bench")
        text = lb.render_text()
        assert "no entries" in text

    def test_duplicate_algorithm_entries(self) -> None:
        lb = Leaderboard("test_bench", direction="minimize")
        lb.add_result(_make_result(algorithm_name="algo_a", best_value=3.0))
        lb.add_result(_make_result(algorithm_name="algo_a", best_value=1.0))
        rankings = lb.get_rankings()
        assert len(rankings) == 2
        # Both entries should be present, ranked by value
        assert rankings[0].best_value == 1.0
        assert rankings[1].best_value == 3.0

    def test_leaderboard_entry_dataclass(self) -> None:
        entry = LeaderboardEntry(
            rank=1,
            algorithm_name="test",
            best_value=0.5,
            n_evaluations=10,
            total_cost=10.0,
            wall_time_seconds=1.0,
        )
        assert entry.rank == 1
        assert entry.algorithm_name == "test"
        assert entry.best_value == 0.5

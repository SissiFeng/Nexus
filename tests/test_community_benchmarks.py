"""Tests for optimization_copilot.community_benchmarks package."""

from __future__ import annotations

import pytest

from optimization_copilot.community_benchmarks import (
    OLYMPUS_REGISTRY,
    OlympusSurface,
    OlympusLoader,
    AccelerationFactor,
    EnhancementFactor,
    DegreeOfAutonomy,
    SDLPerformanceReport,
    SDLMetricsCalculator,
)
from optimization_copilot.benchmark.functions import BenchmarkFunction


# ---------------------------------------------------------------------------
# OLYMPUS_REGISTRY tests
# ---------------------------------------------------------------------------


class TestOlympusRegistry:
    """Tests for the built-in OLYMPUS_REGISTRY."""

    def test_registry_has_exactly_six_surfaces(self) -> None:
        assert len(OLYMPUS_REGISTRY) == 6

    @pytest.mark.parametrize("name", list(OLYMPUS_REGISTRY.keys()))
    def test_each_surface_evaluates_at_midpoint(self, name: str) -> None:
        surface = OLYMPUS_REGISTRY[name]
        midpoint = {
            pname: (lo + hi) / 2.0
            for pname, (lo, hi) in zip(surface.param_names, surface.param_bounds)
        }
        result = surface.evaluate(midpoint)
        assert isinstance(result, dict)
        assert surface.objective_name in result
        assert isinstance(result[surface.objective_name], float)


# ---------------------------------------------------------------------------
# OlympusSurface tests
# ---------------------------------------------------------------------------


class TestOlympusSurface:
    """Tests for OlympusSurface dataclass methods."""

    def test_to_benchmark_returns_benchmark_function(self) -> None:
        surface = OLYMPUS_REGISTRY["photobleaching"]
        bench = surface.to_benchmark()
        assert isinstance(bench, BenchmarkFunction)
        assert bench.name == "olympus_photobleaching"

    def test_evaluate_returns_dict_with_objective_key(self) -> None:
        surface = OLYMPUS_REGISTRY["hplc"]
        result = surface.evaluate({"flow_rate": 1.0, "gradient": 20.0})
        assert isinstance(result, dict)
        assert "resolution" in result


# ---------------------------------------------------------------------------
# OlympusLoader tests
# ---------------------------------------------------------------------------


class TestOlympusLoader:
    """Tests for OlympusLoader factory methods."""

    def test_from_dict_round_trip(self) -> None:
        original = OLYMPUS_REGISTRY["crossed_barrel"]
        data = {
            "name": original.name,
            "param_names": original.param_names,
            "param_bounds": original.param_bounds,
            "lookup_table": original.lookup_table,
            "objective_name": original.objective_name,
        }
        rebuilt = OlympusLoader.from_dict(data)
        assert rebuilt.name == original.name
        assert rebuilt.param_names == original.param_names
        assert rebuilt.objective_name == original.objective_name
        assert len(rebuilt.lookup_table) == len(original.lookup_table)
        # Verify evaluation matches
        mid = {"angle": 45.0, "speed": 200.0}
        assert rebuilt.evaluate(mid) == original.evaluate(mid)

    def test_from_csv_string_parsing(self) -> None:
        csv_text = (
            "x,y,score\n"
            "1.0,2.0,0.5\n"
            "3.0,4.0,0.9\n"
        )
        surface = OlympusLoader.from_csv_string(
            csv_text,
            param_names=["x", "y"],
            objective_name="score",
            param_bounds=[(0.0, 5.0), (0.0, 5.0)],
        )
        assert surface.name == "score"
        assert len(surface.lookup_table) == 2
        result = surface.evaluate({"x": 1.0, "y": 2.0})
        assert "score" in result
        assert result["score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# AccelerationFactor tests
# ---------------------------------------------------------------------------


class TestAccelerationFactor:
    """Tests for AccelerationFactor metric."""

    def test_compute_typical_values(self) -> None:
        # 20 BO iterations vs 100 random -> acceleration = 5.0
        af = AccelerationFactor.compute(
            bo_iterations=20, random_iterations=100, target=0.1
        )
        assert af == pytest.approx(5.0)

    def test_compute_zero_bo_iterations_returns_one(self) -> None:
        af = AccelerationFactor.compute(
            bo_iterations=0, random_iterations=100, target=0.1
        )
        assert af == 1.0


# ---------------------------------------------------------------------------
# EnhancementFactor tests
# ---------------------------------------------------------------------------


class TestEnhancementFactor:
    """Tests for EnhancementFactor metric."""

    def test_compute_minimize_direction(self) -> None:
        # BO found 2.0, random found 10.0 -> enhancement = 5.0
        ef = EnhancementFactor.compute(
            bo_best=2.0, random_best=10.0, budget=50, direction="minimize"
        )
        assert ef == pytest.approx(5.0)

    def test_compute_maximize_direction(self) -> None:
        # BO found 10.0, random found 2.0 -> enhancement = 5.0
        ef = EnhancementFactor.compute(
            bo_best=10.0, random_best=2.0, budget=50, direction="maximize"
        )
        assert ef == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# DegreeOfAutonomy tests
# ---------------------------------------------------------------------------


class TestDegreeOfAutonomy:
    """Tests for DegreeOfAutonomy metric."""

    def test_compute_typical(self) -> None:
        # 5 interventions out of 50 decisions -> 0.9
        doa = DegreeOfAutonomy.compute(human_interventions=5, total_decisions=50)
        assert doa == pytest.approx(0.9)

    def test_compute_zero_decisions_returns_one(self) -> None:
        doa = DegreeOfAutonomy.compute(human_interventions=0, total_decisions=0)
        assert doa == 1.0


# ---------------------------------------------------------------------------
# SDLMetricsCalculator tests
# ---------------------------------------------------------------------------


class TestSDLMetricsCalculator:
    """Tests for SDLMetricsCalculator.evaluate."""

    def test_evaluate_returns_report_with_all_fields(self) -> None:
        calc = SDLMetricsCalculator()
        report = calc.evaluate(
            bo_iterations=20,
            bo_best=2.0,
            random_iterations=100,
            random_best=10.0,
            budget=50,
            human_interventions=5,
            total_decisions=50,
            target=0.1,
            direction="minimize",
        )
        assert isinstance(report, SDLPerformanceReport)
        assert report.acceleration_factor == pytest.approx(5.0)
        assert report.enhancement_factor == pytest.approx(5.0)
        assert report.degree_of_autonomy == pytest.approx(0.9)
        assert report.total_iterations == 20
        assert report.best_value == pytest.approx(2.0)
        assert report.human_interventions == 5
        assert isinstance(report.metadata, dict)
        assert "random_iterations" in report.metadata

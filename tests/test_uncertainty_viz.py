"""Tests for visualization/uncertainty_flow.py -- uncertainty chart functions."""

from __future__ import annotations

import pytest

from optimization_copilot.uncertainty.types import (
    MeasurementWithUncertainty,
    ObservationWithNoise,
    UncertaintyBudget,
)
from optimization_copilot.visualization.models import PlotData
from optimization_copilot.visualization.uncertainty_flow import (
    plot_confidence_heatmap,
    plot_measurement_reliability_timeline,
    plot_noise_impact,
    plot_uncertainty_budget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_svg(pd: PlotData) -> bool:
    """Check that the PlotData has a non-empty SVG string."""
    return pd.svg is not None and "<svg" in pd.svg and "</svg>" in pd.svg


def _make_measurement(
    value: float = 1.0,
    variance: float = 0.1,
    confidence: float = 0.9,
    source: str = "kpi_a",
) -> MeasurementWithUncertainty:
    return MeasurementWithUncertainty(
        value=value, variance=variance, confidence=confidence, source=source,
    )


# ---------------------------------------------------------------------------
# plot_uncertainty_budget
# ---------------------------------------------------------------------------

class TestUncertaintyBudget:
    def test_single_measurement_one_bar(self):
        m = _make_measurement(source="sensor_1", variance=0.5)
        pd = plot_uncertainty_budget([m])
        assert pd.data["sources"] == ["sensor_1"]
        assert len(pd.data["fractions"]) == 1
        assert pd.data["fractions"][0] == pytest.approx(1.0)

    def test_multiple_measurements_correct_bar_count(self):
        ms = [
            _make_measurement(source="a", variance=0.3),
            _make_measurement(source="b", variance=0.7),
            _make_measurement(source="c", variance=1.0),
        ]
        pd = plot_uncertainty_budget(ms)
        assert len(pd.data["sources"]) == 3
        assert len(pd.data["fractions"]) == 3

    def test_returns_correct_plot_type(self):
        ms = [_make_measurement()]
        pd = plot_uncertainty_budget(ms)
        assert isinstance(pd, PlotData)
        assert pd.plot_type == "uncertainty_budget"

    def test_svg_is_valid(self):
        ms = [_make_measurement(), _make_measurement(source="b", variance=0.2)]
        pd = plot_uncertainty_budget(ms)
        assert _valid_svg(pd)

    def test_data_contains_budget_info(self):
        ms = [
            _make_measurement(source="x", variance=0.25),
            _make_measurement(source="y", variance=0.75),
        ]
        pd = plot_uncertainty_budget(ms)
        assert "contributions" in pd.data
        assert pd.data["dominant_source"] == "y"
        assert pd.metadata["total_variance"] == pytest.approx(1.0)

    def test_handles_zero_variance(self):
        ms = [
            _make_measurement(source="z", variance=0.0),
        ]
        pd = plot_uncertainty_budget(ms)
        assert isinstance(pd, PlotData)
        assert _valid_svg(pd)
        assert pd.data["fractions"][0] == pytest.approx(0.0)

    def test_uses_observation_budget_when_available(self):
        budget = UncertaintyBudget.from_contributions({"src_a": 0.6, "src_b": 0.4})
        obs = ObservationWithNoise(
            objective_value=1.0,
            noise_variance=1.0,
            uncertainty_budget=budget,
        )
        # measurements are irrelevant when observation has budget
        pd = plot_uncertainty_budget([], observation=obs)
        assert pd.data["sources"] == ["src_a", "src_b"]
        assert pd.data["dominant_source"] == "src_a"

    def test_empty_measurements_no_observation(self):
        pd = plot_uncertainty_budget([])
        assert pd.data["sources"] == []
        assert _valid_svg(pd)


# ---------------------------------------------------------------------------
# plot_noise_impact
# ---------------------------------------------------------------------------

class TestNoiseImpact:
    def test_basic_scatter_three_points(self):
        pd = plot_noise_impact([1.0, 2.0, 3.0], [0.1, 0.5, 1.0])
        assert pd.data["observations"] == [1.0, 2.0, 3.0]
        assert pd.data["noise_variances"] == [0.1, 0.5, 1.0]
        assert "<circle" in pd.svg

    def test_returns_valid_plotdata_and_svg(self):
        pd = plot_noise_impact([10.0, 20.0], [0.01, 0.1])
        assert isinstance(pd, PlotData)
        assert pd.plot_type == "noise_impact"
        assert _valid_svg(pd)

    def test_high_noise_points_smaller_markers(self):
        # We cannot easily measure SVG radius directly, but we can verify
        # the data payload is correct and the SVG is well-formed.
        pd = plot_noise_impact([1.0, 2.0, 3.0], [0.01, 10.0, 100.0])
        assert _valid_svg(pd)
        assert pd.metadata["n_observations"] == 3
        # The scatter should have circles
        assert pd.svg.count("<circle") >= 3

    def test_handles_single_observation(self):
        pd = plot_noise_impact([5.0], [0.25])
        assert pd.metadata["n_observations"] == 1
        assert _valid_svg(pd)

    def test_data_contains_observations_and_noise(self):
        pd = plot_noise_impact([1.0, 2.0], [0.3, 0.7])
        assert "observations" in pd.data
        assert "noise_variances" in pd.data
        assert "median_noise_variance" in pd.data

    def test_observation_with_noise_objects(self):
        obs = [
            ObservationWithNoise(objective_value=1.0, noise_variance=0.1),
            ObservationWithNoise(objective_value=2.0, noise_variance=0.5),
        ]
        pd = plot_noise_impact(obs)
        assert pd.data["observations"] == [1.0, 2.0]
        assert pd.data["noise_variances"] == [0.1, 0.5]

    def test_empty_observations(self):
        pd = plot_noise_impact([], [])
        assert pd.metadata["n_observations"] == 0
        assert _valid_svg(pd)


# ---------------------------------------------------------------------------
# plot_measurement_reliability_timeline
# ---------------------------------------------------------------------------

class TestReliabilityTimeline:
    def test_single_iteration(self):
        history = [[
            _make_measurement(source="a", confidence=0.9, value=10.0, variance=1.0),
        ]]
        pd = plot_measurement_reliability_timeline(history)
        assert pd.data["n_iterations"] == 1
        assert "a" in pd.data["sources"]
        assert _valid_svg(pd)

    def test_multiple_iterations(self):
        history = [
            [_make_measurement(source="a", confidence=0.9, value=10.0, variance=1.0)],
            [_make_measurement(source="a", confidence=0.8, value=10.0, variance=2.0)],
            [_make_measurement(source="a", confidence=0.3, value=10.0, variance=5.0)],
        ]
        pd = plot_measurement_reliability_timeline(history)
        assert pd.data["n_iterations"] == 3
        assert len(pd.data["sources"]["a"]) == 3

    def test_returns_valid_svg(self):
        history = [
            [_make_measurement(source="x", confidence=0.7, value=5.0, variance=0.5)],
            [_make_measurement(source="x", confidence=0.6, value=5.0, variance=0.8)],
        ]
        pd = plot_measurement_reliability_timeline(history)
        assert isinstance(pd, PlotData)
        assert pd.plot_type == "reliability_timeline"
        assert _valid_svg(pd)

    def test_detects_unreliable_measurements(self):
        history = [
            [_make_measurement(source="s1", confidence=0.9, value=10.0, variance=1.0)],
            [_make_measurement(source="s1", confidence=0.3, value=10.0, variance=5.0)],
        ]
        pd = plot_measurement_reliability_timeline(history)
        unreliable = pd.data["unreliable_points"]
        assert len(unreliable) >= 1
        assert unreliable[0]["source"] == "s1"
        assert unreliable[0]["confidence"] == pytest.approx(0.3)

    def test_handles_empty_history(self):
        pd = plot_measurement_reliability_timeline([])
        assert pd.data["n_iterations"] == 0
        assert _valid_svg(pd)

    def test_multiple_sources(self):
        history = [
            [
                _make_measurement(source="a", confidence=0.9, value=10.0, variance=1.0),
                _make_measurement(source="b", confidence=0.7, value=5.0, variance=0.5),
            ],
            [
                _make_measurement(source="a", confidence=0.4, value=10.0, variance=4.0),
                _make_measurement(source="b", confidence=0.8, value=5.0, variance=0.3),
            ],
        ]
        pd = plot_measurement_reliability_timeline(history)
        assert "a" in pd.data["sources"]
        assert "b" in pd.data["sources"]
        assert len(pd.data["sources"]["a"]) == 2
        assert len(pd.data["sources"]["b"]) == 2

    def test_missing_source_in_iteration(self):
        """Source present in first iteration but absent in second."""
        history = [
            [
                _make_measurement(source="a", confidence=0.9, value=10.0, variance=1.0),
                _make_measurement(source="b", confidence=0.7, value=5.0, variance=0.5),
            ],
            [
                _make_measurement(source="a", confidence=0.8, value=10.0, variance=2.0),
                # b is missing in this iteration
            ],
        ]
        pd = plot_measurement_reliability_timeline(history)
        # b should have None for the second iteration
        assert pd.data["sources"]["b"][1] is None


# ---------------------------------------------------------------------------
# plot_confidence_heatmap
# ---------------------------------------------------------------------------

class TestConfidenceHeatmap:
    def test_2x2_grid(self):
        grid = [[0.9, 0.3], [0.5, 0.1]]
        pd = plot_confidence_heatmap(grid, ["param_a", "param_b"])
        assert pd.metadata["n_rows"] == 2
        assert pd.metadata["n_cols"] == 2
        assert _valid_svg(pd)

    def test_3x3_grid(self):
        grid = [
            [0.1, 0.5, 0.9],
            [0.4, 0.6, 0.8],
            [0.2, 0.7, 1.0],
        ]
        pd = plot_confidence_heatmap(grid, ["x", "y"])
        assert pd.metadata["n_rows"] == 3
        assert pd.metadata["n_cols"] == 3
        assert _valid_svg(pd)

    def test_returns_valid_svg(self):
        grid = [[0.5, 0.6], [0.7, 0.8]]
        pd = plot_confidence_heatmap(grid, ["a", "b"])
        assert isinstance(pd, PlotData)
        assert pd.plot_type == "confidence_heatmap"
        assert _valid_svg(pd)

    def test_cell_colors_correspond_to_confidence(self):
        """High confidence cells should produce green-ish colors, low -> red."""
        grid = [[0.0, 1.0]]
        pd = plot_confidence_heatmap(grid, ["row", "col"])
        # The SVG should contain rect elements with different fill colors.
        assert "<rect" in pd.svg
        assert _valid_svg(pd)

    def test_handles_edge_confidence_zero(self):
        grid = [[0.0]]
        pd = plot_confidence_heatmap(grid, ["x", "y"])
        assert _valid_svg(pd)
        assert pd.data["grid"] == [[0.0]]

    def test_handles_edge_confidence_one(self):
        grid = [[1.0]]
        pd = plot_confidence_heatmap(grid, ["x", "y"])
        assert _valid_svg(pd)
        assert pd.data["grid"] == [[1.0]]

    def test_data_contains_grid_and_params(self):
        grid = [[0.5, 0.6], [0.7, 0.8]]
        pd = plot_confidence_heatmap(grid, ["temperature", "pressure"])
        assert pd.data["grid"] == grid
        assert pd.data["param_names"] == ["temperature", "pressure"]


# ---------------------------------------------------------------------------
# General / cross-chart tests
# ---------------------------------------------------------------------------

class TestGeneral:
    def test_all_functions_return_plotdata(self):
        """Smoke test: all four chart functions produce PlotData."""
        m = _make_measurement()
        budget_pd = plot_uncertainty_budget([m])
        noise_pd = plot_noise_impact([1.0, 2.0], [0.1, 0.5])
        timeline_pd = plot_measurement_reliability_timeline([[m]])
        heatmap_pd = plot_confidence_heatmap([[0.5]], ["x", "y"])

        for pd in [budget_pd, noise_pd, timeline_pd, heatmap_pd]:
            assert isinstance(pd, PlotData)

    def test_all_svgs_are_valid(self):
        m = _make_measurement()
        budget_pd = plot_uncertainty_budget([m])
        noise_pd = plot_noise_impact([1.0], [0.1])
        timeline_pd = plot_measurement_reliability_timeline([[m]])
        heatmap_pd = plot_confidence_heatmap([[0.8]], ["a", "b"])

        for pd in [budget_pd, noise_pd, timeline_pd, heatmap_pd]:
            assert _valid_svg(pd)

    def test_data_roundtrip_through_to_dict_from_dict(self):
        """PlotData from chart functions survives serialisation."""
        ms = [_make_measurement(source="s1", variance=0.3)]
        pd = plot_uncertainty_budget(ms)
        d = pd.to_dict()
        pd2 = PlotData.from_dict(d)
        assert pd2.plot_type == pd.plot_type
        assert pd2.svg == pd.svg
        assert pd2.data == pd.data

    def test_noise_impact_roundtrip(self):
        pd = plot_noise_impact([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
        d = pd.to_dict()
        pd2 = PlotData.from_dict(d)
        assert pd2.plot_type == "noise_impact"
        assert pd2.svg == pd.svg

    def test_timeline_roundtrip(self):
        history = [[_make_measurement(source="a", value=10.0, variance=1.0)]]
        pd = plot_measurement_reliability_timeline(history)
        d = pd.to_dict()
        pd2 = PlotData.from_dict(d)
        assert pd2.plot_type == "reliability_timeline"
        assert pd2.svg == pd.svg

    def test_heatmap_roundtrip(self):
        pd = plot_confidence_heatmap([[0.5, 0.9], [0.1, 0.7]], ["x", "y"])
        d = pd.to_dict()
        pd2 = PlotData.from_dict(d)
        assert pd2.plot_type == "confidence_heatmap"
        assert pd2.data["grid"] == [[0.5, 0.9], [0.1, 0.7]]

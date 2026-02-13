"""Comprehensive tests for the campaign report visualization module.

Covers all 8 figure functions, shared helpers, CampaignReport orchestrator,
CampaignReportData dataclass, and combined SVG generation.
"""

from __future__ import annotations

import math

import pytest

from optimization_copilot.visualization.campaign_report import (
    CampaignReport,
    CampaignReportData,
    _compute_nice_ticks,
    _scale_linear,
    plot_batch_comparison,
    plot_calibration_curve,
    plot_convergence_curve,
    plot_drift_timeline,
    plot_feature_importance,
    plot_pareto_front,
    plot_recommendation_coverage,
    plot_uncertainty_coverage,
)
from optimization_copilot.visualization.models import PlotData


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def convergence_data():
    return {
        "iterations": list(range(1, 11)),
        "best_values": [10.0, 8.0, 7.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5],
    }


@pytest.fixture
def calibration_data():
    return {
        "predicted": [1.0, 2.0, 3.0, 4.0, 5.0],
        "actual": [1.1, 2.2, 2.9, 4.1, 5.2],
    }


@pytest.fixture
def drift_data():
    return {
        "iterations": list(range(1, 11)),
        "model_errors": [0.5, 0.4, 0.6, 0.3, 0.7, 0.5, 0.8, 0.6, 0.9, 0.7],
    }


@pytest.fixture
def batch_data():
    return {
        "labels": ["A", "B", "C"],
        "means": [3.0, 5.0, 4.0],
        "stds": [0.5, 1.0, 0.8],
    }


@pytest.fixture
def coverage_data():
    return {
        "candidate_xy": [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
        "observed_xy": [(2.0, 3.0), (4.0, 5.0)],
    }


@pytest.fixture
def uncertainty_data():
    return {
        "grid_x": [0.0, 1.0, 2.0],
        "grid_y": [0.0, 1.0, 2.0],
        "uncertainty": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
    }


@pytest.fixture
def importance_data():
    return {
        "names": ["temp", "pressure", "time"],
        "importances": [0.5, 0.3, 0.2],
    }


@pytest.fixture
def pareto_data():
    return {
        "obj1": [1.0, 2.0, 3.0, 1.5, 2.5],
        "obj2": [5.0, 3.0, 1.0, 4.0, 2.0],
        "is_dominated": [False, False, False, True, True],
    }


@pytest.fixture
def full_report_data():
    """CampaignReportData with all fields populated."""
    return CampaignReportData(
        iterations=list(range(1, 11)),
        best_values=[10.0, 8.0, 7.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5],
        predicted=[1.0, 2.0, 3.0, 4.0, 5.0],
        actual=[1.1, 2.2, 2.9, 4.1, 5.2],
        drift_iterations=list(range(1, 11)),
        model_errors=[0.5, 0.4, 0.6, 0.3, 0.7, 0.5, 0.8, 0.6, 0.9, 0.7],
        batch_labels=["A", "B", "C"],
        batch_means=[3.0, 5.0, 4.0],
        batch_stds=[0.5, 1.0, 0.8],
        candidate_xy=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
        observed_xy=[(2.0, 3.0), (4.0, 5.0)],
        uncertainty_grid_x=[0.0, 1.0, 2.0],
        uncertainty_grid_y=[0.0, 1.0, 2.0],
        uncertainty_values=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        feature_names=["temp", "pressure", "time"],
        feature_importances=[0.5, 0.3, 0.2],
        pareto_obj1=[1.0, 2.0, 3.0, 1.5, 2.5],
        pareto_obj2=[5.0, 3.0, 1.0, 4.0, 2.0],
        pareto_dominated=[False, False, False, True, True],
    )


# ===========================================================================
# 1. _scale_linear helper tests
# ===========================================================================

class TestScaleLinear:
    """Tests for the _scale_linear helper function."""

    def test_maps_min_to_pixel_min(self):
        assert _scale_linear(0.0, 0.0, 10.0, 100.0, 500.0) == 100.0

    def test_maps_max_to_pixel_max(self):
        assert _scale_linear(10.0, 0.0, 10.0, 100.0, 500.0) == 500.0

    def test_maps_midpoint_to_pixel_midpoint(self):
        assert _scale_linear(5.0, 0.0, 10.0, 100.0, 500.0) == 300.0

    def test_maps_quarter_point(self):
        result = _scale_linear(2.5, 0.0, 10.0, 0.0, 400.0)
        assert result == pytest.approx(100.0)

    def test_same_min_max_returns_midpoint(self):
        """When data_min == data_max, return the midpoint of pixel range."""
        result = _scale_linear(5.0, 5.0, 5.0, 100.0, 500.0)
        assert result == pytest.approx(300.0)

    def test_negative_data_range(self):
        result = _scale_linear(-5.0, -10.0, 0.0, 0.0, 100.0)
        assert result == pytest.approx(50.0)

    def test_inverted_pixel_range(self):
        """SVG Y-axis often goes from bottom (large) to top (small)."""
        result = _scale_linear(0.0, 0.0, 10.0, 300.0, 0.0)
        assert result == pytest.approx(300.0)

    def test_value_outside_range(self):
        """Values outside data range should extrapolate linearly."""
        result = _scale_linear(20.0, 0.0, 10.0, 0.0, 100.0)
        assert result == pytest.approx(200.0)

    def test_float_precision(self):
        result = _scale_linear(1.0, 0.0, 3.0, 0.0, 300.0)
        assert result == pytest.approx(100.0)


# ===========================================================================
# 2. _compute_nice_ticks helper tests
# ===========================================================================

class TestComputeNiceTicks:
    """Tests for the _compute_nice_ticks helper function."""

    def test_returns_list_of_floats(self):
        ticks = _compute_nice_ticks(0.0, 10.0)
        assert isinstance(ticks, list)
        assert all(isinstance(t, float) for t in ticks)

    def test_includes_boundary_values(self):
        """Ticks should span the data range."""
        ticks = _compute_nice_ticks(0.0, 10.0)
        assert len(ticks) >= 2
        assert ticks[0] <= 0.0
        assert ticks[-1] >= 10.0

    def test_same_min_max_returns_single_tick(self):
        ticks = _compute_nice_ticks(5.0, 5.0)
        assert ticks == [5.0]

    def test_ticks_are_sorted(self):
        ticks = _compute_nice_ticks(0.0, 100.0)
        assert ticks == sorted(ticks)

    def test_small_range(self):
        ticks = _compute_nice_ticks(0.0, 1.0)
        assert len(ticks) >= 2
        assert ticks[0] <= 0.0
        assert ticks[-1] >= 1.0

    def test_large_range(self):
        ticks = _compute_nice_ticks(0.0, 1000.0)
        assert len(ticks) >= 2
        assert ticks[0] <= 0.0
        assert ticks[-1] >= 1000.0

    def test_negative_range(self):
        ticks = _compute_nice_ticks(-10.0, -2.0)
        assert len(ticks) >= 2
        assert ticks[0] <= -10.0
        assert ticks[-1] >= -2.0

    def test_custom_n_ticks(self):
        ticks_3 = _compute_nice_ticks(0.0, 10.0, n_ticks=3)
        ticks_10 = _compute_nice_ticks(0.0, 10.0, n_ticks=10)
        assert isinstance(ticks_3, list)
        assert isinstance(ticks_10, list)

    def test_very_small_range(self):
        """Near-zero range should still produce a valid result."""
        ticks = _compute_nice_ticks(1.0, 1.0 + 1e-16)
        assert isinstance(ticks, list)
        assert len(ticks) >= 1


# ===========================================================================
# 3. Individual figure functions: valid SVG output
# ===========================================================================

class TestConvergenceCurve:
    """Tests for plot_convergence_curve."""

    def test_valid_svg(self, convergence_data):
        result = plot_convergence_curve(**convergence_data)
        assert isinstance(result, PlotData)
        assert result.plot_type == "convergence_curve"
        assert "<svg" in result.svg
        assert "</svg>" in result.svg

    def test_empty_data_returns_no_data_svg(self):
        result = plot_convergence_curve([], [])
        assert "No data" in result.svg
        assert result.plot_type == "convergence_curve"
        assert result.data == {}

    def test_metadata_final_best(self, convergence_data):
        result = plot_convergence_curve(**convergence_data)
        assert "final_best" in result.metadata
        assert result.metadata["final_best"] == 1.5

    def test_data_stored(self, convergence_data):
        result = plot_convergence_curve(**convergence_data)
        assert result.data["iterations"] == convergence_data["iterations"]
        assert result.data["best_values"] == convergence_data["best_values"]

    def test_single_point(self):
        result = plot_convergence_curve([1], [5.0])
        assert "<svg" in result.svg
        assert result.metadata["final_best"] == 5.0


class TestCalibrationCurve:
    """Tests for plot_calibration_curve."""

    def test_valid_svg(self, calibration_data):
        result = plot_calibration_curve(**calibration_data)
        assert isinstance(result, PlotData)
        assert result.plot_type == "calibration_curve"
        assert "<svg" in result.svg
        assert "</svg>" in result.svg

    def test_empty_data_returns_no_data_svg(self):
        result = plot_calibration_curve([], [])
        assert "No data" in result.svg
        assert result.plot_type == "calibration_curve"

    def test_metadata_r_squared_and_rmse(self, calibration_data):
        result = plot_calibration_curve(**calibration_data)
        assert "r_squared" in result.metadata
        assert "rmse" in result.metadata
        assert 0.0 <= result.metadata["r_squared"] <= 1.0
        assert result.metadata["rmse"] >= 0.0

    def test_perfect_calibration(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = plot_calibration_curve(vals, vals)
        assert result.metadata["r_squared"] == pytest.approx(1.0)
        assert result.metadata["rmse"] == pytest.approx(0.0)

    def test_r_squared_reasonable(self, calibration_data):
        """R-squared for near-diagonal data should be close to 1."""
        result = plot_calibration_curve(**calibration_data)
        assert result.metadata["r_squared"] > 0.9


class TestDriftTimeline:
    """Tests for plot_drift_timeline."""

    def test_valid_svg(self, drift_data):
        result = plot_drift_timeline(**drift_data)
        assert isinstance(result, PlotData)
        assert result.plot_type == "drift_timeline"
        assert "<svg" in result.svg
        assert "</svg>" in result.svg

    def test_empty_data_returns_no_data_svg(self):
        result = plot_drift_timeline([], [])
        assert "No data" in result.svg
        assert result.plot_type == "drift_timeline"

    def test_metadata_trend_slope(self, drift_data):
        result = plot_drift_timeline(**drift_data)
        assert "trend_slope" in result.metadata
        assert isinstance(result.metadata["trend_slope"], float)

    def test_positive_drift_slope(self, drift_data):
        """With generally increasing errors, slope should be positive."""
        result = plot_drift_timeline(**drift_data)
        assert result.metadata["trend_slope"] > 0

    def test_flat_errors_zero_slope(self):
        iterations = list(range(1, 6))
        errors = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = plot_drift_timeline(iterations, errors)
        assert result.metadata["trend_slope"] == pytest.approx(0.0)


class TestBatchComparison:
    """Tests for plot_batch_comparison."""

    def test_valid_svg(self, batch_data):
        result = plot_batch_comparison(**batch_data)
        assert isinstance(result, PlotData)
        assert result.plot_type == "batch_comparison"
        assert "<svg" in result.svg
        assert "</svg>" in result.svg

    def test_empty_data_returns_no_data_svg(self):
        result = plot_batch_comparison([], [], [])
        assert "No data" in result.svg
        assert result.plot_type == "batch_comparison"

    def test_metadata_is_dict(self, batch_data):
        result = plot_batch_comparison(**batch_data)
        assert isinstance(result.metadata, dict)

    def test_data_stored(self, batch_data):
        result = plot_batch_comparison(**batch_data)
        assert result.data["labels"] == ["A", "B", "C"]
        assert result.data["means"] == [3.0, 5.0, 4.0]
        assert result.data["stds"] == [0.5, 1.0, 0.8]

    def test_single_batch(self):
        result = plot_batch_comparison(["X"], [3.0], [0.5])
        assert "<svg" in result.svg


class TestRecommendationCoverage:
    """Tests for plot_recommendation_coverage."""

    def test_valid_svg(self, coverage_data):
        result = plot_recommendation_coverage(**coverage_data)
        assert isinstance(result, PlotData)
        assert result.plot_type == "recommendation_coverage"
        assert "<svg" in result.svg
        assert "</svg>" in result.svg

    def test_empty_data_returns_no_data_svg(self):
        result = plot_recommendation_coverage([], [])
        assert "No data" in result.svg
        assert result.plot_type == "recommendation_coverage"

    def test_metadata_n_candidates_and_n_observed(self, coverage_data):
        result = plot_recommendation_coverage(**coverage_data)
        assert result.metadata["n_candidates"] == 3
        assert result.metadata["n_observed"] == 2

    def test_only_candidates(self):
        result = plot_recommendation_coverage([(1.0, 2.0)], [])
        assert "<svg" in result.svg
        assert result.metadata["n_candidates"] == 1
        assert result.metadata["n_observed"] == 0

    def test_only_observed(self):
        result = plot_recommendation_coverage([], [(1.0, 2.0)])
        assert "<svg" in result.svg
        assert result.metadata["n_candidates"] == 0
        assert result.metadata["n_observed"] == 1


class TestUncertaintyCoverage:
    """Tests for plot_uncertainty_coverage."""

    def test_valid_svg(self, uncertainty_data):
        result = plot_uncertainty_coverage(**uncertainty_data)
        assert isinstance(result, PlotData)
        assert result.plot_type == "uncertainty_coverage"
        assert "<svg" in result.svg
        assert "</svg>" in result.svg

    def test_empty_data_returns_no_data_svg(self):
        result = plot_uncertainty_coverage([], [], [])
        assert "No data" in result.svg
        assert result.plot_type == "uncertainty_coverage"

    def test_metadata_u_min_u_max(self, uncertainty_data):
        result = plot_uncertainty_coverage(**uncertainty_data)
        assert "u_min" in result.metadata
        assert "u_max" in result.metadata
        assert result.metadata["u_min"] == pytest.approx(0.1)
        assert result.metadata["u_max"] == pytest.approx(0.9)

    def test_uniform_uncertainty(self):
        grid_x = [0.0, 1.0]
        grid_y = [0.0, 1.0]
        uncertainty = [[0.5, 0.5], [0.5, 0.5]]
        result = plot_uncertainty_coverage(grid_x, grid_y, uncertainty)
        assert result.metadata["u_min"] == result.metadata["u_max"]


class TestFeatureImportance:
    """Tests for plot_feature_importance."""

    def test_valid_svg(self, importance_data):
        result = plot_feature_importance(**importance_data)
        assert isinstance(result, PlotData)
        assert result.plot_type == "feature_importance"
        assert "<svg" in result.svg
        assert "</svg>" in result.svg

    def test_empty_data_returns_no_data_svg(self):
        result = plot_feature_importance([], [])
        assert "No data" in result.svg
        assert result.plot_type == "feature_importance"

    def test_metadata_max_importance(self, importance_data):
        result = plot_feature_importance(**importance_data)
        assert "max_importance" in result.metadata
        assert result.metadata["max_importance"] == pytest.approx(0.5)

    def test_single_feature(self):
        result = plot_feature_importance(["x"], [1.0])
        assert "<svg" in result.svg
        assert result.metadata["max_importance"] == pytest.approx(1.0)

    def test_negative_importances(self):
        result = plot_feature_importance(["a", "b"], [-0.3, 0.7])
        assert "<svg" in result.svg
        assert result.metadata["max_importance"] == pytest.approx(0.7)


class TestParetoFront:
    """Tests for plot_pareto_front."""

    def test_valid_svg(self, pareto_data):
        result = plot_pareto_front(**pareto_data)
        assert isinstance(result, PlotData)
        assert result.plot_type == "pareto_front"
        assert "<svg" in result.svg
        assert "</svg>" in result.svg

    def test_empty_data_returns_no_data_svg(self):
        result = plot_pareto_front([], [], [])
        assert "No data" in result.svg
        assert result.plot_type == "pareto_front"

    def test_metadata_n_pareto_and_n_total(self, pareto_data):
        result = plot_pareto_front(**pareto_data)
        assert result.metadata["n_pareto"] == 3
        assert result.metadata["n_total"] == 5

    def test_all_dominated(self):
        result = plot_pareto_front([1.0, 2.0], [3.0, 4.0], [True, True])
        assert result.metadata["n_pareto"] == 0
        assert result.metadata["n_total"] == 2

    def test_none_dominated(self):
        result = plot_pareto_front([1.0, 2.0], [3.0, 4.0], [False, False])
        assert result.metadata["n_pareto"] == 2
        assert result.metadata["n_total"] == 2


# ===========================================================================
# 4. CampaignReport.generate tests
# ===========================================================================

class TestCampaignReportGenerate:
    """Tests for CampaignReport.generate."""

    def test_full_data_generates_all_8_figures(self, full_report_data):
        report = CampaignReport()
        figures = report.generate(full_report_data)
        assert len(figures) == 8
        expected_keys = {
            "convergence", "calibration", "drift", "batch",
            "coverage", "uncertainty", "importance", "pareto",
        }
        assert set(figures.keys()) == expected_keys

    def test_all_figures_are_plotdata(self, full_report_data):
        report = CampaignReport()
        figures = report.generate(full_report_data)
        for key, fig in figures.items():
            assert isinstance(fig, PlotData), f"{key} is not PlotData"
            assert "<svg" in fig.svg, f"{key} missing SVG opening tag"
            assert "</svg>" in fig.svg, f"{key} missing SVG closing tag"

    def test_empty_data_generates_zero_figures(self):
        report = CampaignReport()
        figures = report.generate(CampaignReportData())
        assert len(figures) == 0

    def test_partial_data_only_present_figures(self):
        """Only convergence and calibration provided."""
        data = CampaignReportData(
            iterations=list(range(1, 6)),
            best_values=[5.0, 4.0, 3.0, 2.0, 1.0],
            predicted=[1.0, 2.0, 3.0],
            actual=[1.1, 2.1, 3.1],
        )
        report = CampaignReport()
        figures = report.generate(data)
        assert set(figures.keys()) == {"convergence", "calibration"}
        assert len(figures) == 2

    def test_single_figure_only_convergence(self):
        data = CampaignReportData(
            iterations=[1, 2, 3],
            best_values=[10.0, 5.0, 1.0],
        )
        report = CampaignReport()
        figures = report.generate(data)
        assert set(figures.keys()) == {"convergence"}

    def test_coverage_with_only_candidates(self):
        """Coverage should appear even if only candidate_xy is provided."""
        data = CampaignReportData(
            candidate_xy=[(1.0, 2.0), (3.0, 4.0)],
        )
        report = CampaignReport()
        figures = report.generate(data)
        assert "coverage" in figures
        assert len(figures) == 1

    def test_coverage_with_only_observed(self):
        """Coverage should appear even if only observed_xy is provided."""
        data = CampaignReportData(
            observed_xy=[(1.0, 2.0)],
        )
        report = CampaignReport()
        figures = report.generate(data)
        assert "coverage" in figures
        assert len(figures) == 1

    def test_figure_plot_types_match_keys(self, full_report_data):
        report = CampaignReport()
        figures = report.generate(full_report_data)
        expected_types = {
            "convergence": "convergence_curve",
            "calibration": "calibration_curve",
            "drift": "drift_timeline",
            "batch": "batch_comparison",
            "coverage": "recommendation_coverage",
            "uncertainty": "uncertainty_coverage",
            "importance": "feature_importance",
            "pareto": "pareto_front",
        }
        for key, expected_type in expected_types.items():
            assert figures[key].plot_type == expected_type, (
                f"Figure '{key}' has plot_type '{figures[key].plot_type}', "
                f"expected '{expected_type}'"
            )


# ===========================================================================
# 5. CampaignReport.generate_combined_svg tests
# ===========================================================================

class TestCampaignReportCombinedSvg:
    """Tests for CampaignReport.generate_combined_svg."""

    def test_returns_valid_svg_string(self, full_report_data):
        report = CampaignReport()
        svg = report.generate_combined_svg(full_report_data)
        assert isinstance(svg, str)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_dimensions_800x1200(self, full_report_data):
        """Combined SVG should be 2*400=800 wide and 4*300=1200 tall."""
        report = CampaignReport()
        svg = report.generate_combined_svg(full_report_data)
        assert 'width="800"' in svg
        assert 'height="1200"' in svg

    def test_empty_data_still_returns_valid_svg(self):
        report = CampaignReport()
        svg = report.generate_combined_svg(CampaignReportData())
        assert isinstance(svg, str)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_partial_data_returns_valid_svg(self):
        data = CampaignReportData(
            iterations=[1, 2, 3],
            best_values=[3.0, 2.0, 1.0],
        )
        report = CampaignReport()
        svg = report.generate_combined_svg(data)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_combined_svg_contains_translate_groups(self, full_report_data):
        """Each figure should be placed via translate in the grid."""
        report = CampaignReport()
        svg = report.generate_combined_svg(full_report_data)
        # First row: (0,0) and (400,0)
        assert "translate(0,0)" in svg
        assert "translate(400,0)" in svg
        # Second row: (0,300) and (400,300)
        assert "translate(0,300)" in svg
        assert "translate(400,300)" in svg


# ===========================================================================
# 6. CampaignReportData with all-empty fields
# ===========================================================================

class TestCampaignReportDataEmpty:
    """Tests for CampaignReportData default (all-empty) behavior."""

    def test_default_construction_all_empty(self):
        data = CampaignReportData()
        assert data.iterations == []
        assert data.best_values == []
        assert data.predicted == []
        assert data.actual == []
        assert data.drift_iterations == []
        assert data.model_errors == []
        assert data.batch_labels == []
        assert data.batch_means == []
        assert data.batch_stds == []
        assert data.candidate_xy == []
        assert data.observed_xy == []
        assert data.uncertainty_grid_x == []
        assert data.uncertainty_grid_y == []
        assert data.uncertainty_values == []
        assert data.feature_names == []
        assert data.feature_importances == []
        assert data.pareto_obj1 == []
        assert data.pareto_obj2 == []
        assert data.pareto_dominated == []

    def test_all_empty_generates_empty_dict(self):
        report = CampaignReport()
        figures = report.generate(CampaignReportData())
        assert figures == {}

    def test_all_empty_combined_svg_still_valid(self):
        report = CampaignReport()
        svg = report.generate_combined_svg(CampaignReportData())
        assert "<svg" in svg
        assert "</svg>" in svg


# ===========================================================================
# 7. Edge cases and additional validation
# ===========================================================================

class TestEdgeCases:
    """Additional edge case tests for robustness."""

    def test_convergence_constant_values(self):
        """All best_values the same -- should not crash."""
        result = plot_convergence_curve([1, 2, 3], [5.0, 5.0, 5.0])
        assert "<svg" in result.svg
        assert result.metadata["final_best"] == 5.0

    def test_calibration_single_point(self):
        result = plot_calibration_curve([2.0], [2.0])
        assert "<svg" in result.svg
        assert "r_squared" in result.metadata

    def test_drift_two_points(self):
        result = plot_drift_timeline([1, 2], [0.5, 1.0])
        assert "<svg" in result.svg
        assert result.metadata["trend_slope"] == pytest.approx(0.5)

    def test_batch_zero_stds(self):
        """Standard deviations all zero -- no error bars but no crash."""
        result = plot_batch_comparison(["A", "B"], [3.0, 5.0], [0.0, 0.0])
        assert "<svg" in result.svg

    def test_pareto_single_point(self):
        result = plot_pareto_front([1.0], [2.0], [False])
        assert "<svg" in result.svg
        assert result.metadata["n_pareto"] == 1
        assert result.metadata["n_total"] == 1

    def test_uncertainty_single_cell(self):
        result = plot_uncertainty_coverage([0.0], [0.0], [[0.5]])
        assert "<svg" in result.svg
        assert result.metadata["u_min"] == pytest.approx(0.5)
        assert result.metadata["u_max"] == pytest.approx(0.5)

    def test_feature_importance_many_features(self):
        names = [f"feat_{i}" for i in range(20)]
        importances = [float(i) / 20 for i in range(20)]
        result = plot_feature_importance(names, importances)
        assert "<svg" in result.svg
        assert result.metadata["max_importance"] == pytest.approx(19.0 / 20)

    def test_convergence_empty_iterations_only(self):
        """Empty iterations but non-empty best_values."""
        result = plot_convergence_curve([], [1.0, 2.0])
        assert "No data" in result.svg

    def test_convergence_empty_values_only(self):
        """Non-empty iterations but empty best_values."""
        result = plot_convergence_curve([1, 2], [])
        assert "No data" in result.svg

    def test_plotdata_has_expected_fields(self, convergence_data):
        """Verify PlotData structure after creation."""
        result = plot_convergence_curve(**convergence_data)
        assert hasattr(result, "plot_type")
        assert hasattr(result, "data")
        assert hasattr(result, "metadata")
        assert hasattr(result, "svg")

    def test_plotdata_to_dict_roundtrip(self, convergence_data):
        """PlotData should serialize and deserialize correctly."""
        result = plot_convergence_curve(**convergence_data)
        d = result.to_dict()
        restored = PlotData.from_dict(d)
        assert restored.plot_type == result.plot_type
        assert restored.metadata == result.metadata
        assert restored.svg == result.svg

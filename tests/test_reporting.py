"""Tests for the SVG reporting / chart generation module."""

from __future__ import annotations

import pytest

from optimization_copilot.case_studies.reporting import (
    plot_ablation_bars,
    plot_box_comparison,
    plot_convergence_curves,
    plot_significance_heatmap,
)
from optimization_copilot.case_studies.statistics import paired_comparison_table


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _sample_curves() -> dict:
    """Two strategies, 3 repeats each, 20 iterations."""
    import random
    rng = random.Random(42)
    curves: dict = {}
    for name, base in [("StrategyA", 10.0), ("StrategyB", 8.0)]:
        repeats = []
        for _ in range(3):
            c = []
            v = base
            for _ in range(20):
                v -= rng.uniform(0.0, 0.5)
                c.append(v)
            repeats.append(c)
        curves[name] = repeats
    return curves


def _sample_final_data() -> dict:
    return {
        "A": [5.0, 5.5, 6.0, 4.5, 7.0, 5.2, 6.1, 4.8, 5.9, 5.3],
        "B": [3.0, 3.5, 4.0, 2.5, 4.5, 3.2, 3.8, 2.9, 3.7, 3.3],
        "C": [8.0, 9.0, 7.5, 8.5, 10.0, 9.5, 8.2, 7.8, 9.1, 8.8],
    }


# ---------------------------------------------------------------------------
# plot_convergence_curves
# ---------------------------------------------------------------------------


class TestConvergenceCurvesReturnsSvg:
    def test_convergence_curves_returns_svg(self):
        svg = plot_convergence_curves(_sample_curves())
        assert svg.startswith("<svg")
        assert "</svg>" in svg


class TestConvergenceCurvesHasTitle:
    def test_convergence_curves_has_title(self):
        svg = plot_convergence_curves(_sample_curves(), title="My Title")
        assert "My Title" in svg


class TestConvergenceCurvesMultipleStrategies:
    def test_convergence_curves_multiple_strategies(self):
        svg = plot_convergence_curves(_sample_curves())
        assert "StrategyA" in svg
        assert "StrategyB" in svg


class TestConvergenceCurvesSingleStrategy:
    def test_convergence_curves_single_strategy(self):
        curves = {"OnlyOne": [[10.0, 9.0, 8.0, 7.0]]}
        svg = plot_convergence_curves(curves)
        assert svg.startswith("<svg")
        assert "OnlyOne" in svg


class TestConvergenceCurvesEmpty:
    def test_convergence_curves_empty(self):
        svg = plot_convergence_curves({})
        assert svg.startswith("<svg")
        assert "(no data)" in svg


class TestConvergenceCurvesHasPolyline:
    def test_has_polyline(self):
        svg = plot_convergence_curves(_sample_curves())
        assert "<polyline" in svg


class TestConvergenceCurvesHasPolygon:
    """Shaded std region should be a polygon."""
    def test_has_polygon(self):
        svg = plot_convergence_curves(_sample_curves())
        assert "<polygon" in svg


# ---------------------------------------------------------------------------
# plot_box_comparison
# ---------------------------------------------------------------------------


class TestBoxComparisonReturnsSvg:
    def test_box_comparison_returns_svg(self):
        svg = plot_box_comparison(_sample_final_data())
        assert svg.startswith("<svg")
        assert "</svg>" in svg


class TestBoxComparisonHasBoxes:
    def test_box_comparison_has_boxes(self):
        svg = plot_box_comparison(_sample_final_data())
        assert "<rect" in svg


class TestBoxComparisonHasMedians:
    def test_box_comparison_has_medians(self):
        svg = plot_box_comparison(_sample_final_data())
        # Median lines are drawn with stroke-width="2"
        assert 'stroke-width="2"' in svg


class TestBoxComparisonMultipleStrategies:
    def test_box_comparison_multiple_strategies(self):
        svg = plot_box_comparison(_sample_final_data())
        for name in ("A", "B", "C"):
            assert name in svg


class TestBoxComparisonEmpty:
    def test_box_comparison_empty(self):
        svg = plot_box_comparison({})
        assert svg.startswith("<svg")


class TestBoxComparisonSingle:
    def test_box_comparison_single(self):
        svg = plot_box_comparison({"Solo": [1.0, 2.0, 3.0, 4.0, 5.0]})
        assert "Solo" in svg
        assert "<rect" in svg


# ---------------------------------------------------------------------------
# plot_significance_heatmap
# ---------------------------------------------------------------------------


class TestSignificanceHeatmapReturnsSvg:
    def test_significance_heatmap_returns_svg(self):
        comp = paired_comparison_table(_sample_final_data())
        svg = plot_significance_heatmap(comp)
        assert svg.startswith("<svg")
        assert "</svg>" in svg


class TestSignificanceHeatmapHasCells:
    def test_significance_heatmap_has_cells(self):
        comp = paired_comparison_table(_sample_final_data())
        svg = plot_significance_heatmap(comp)
        assert "<rect" in svg


class TestSignificanceHeatmapColorCoding:
    """Cells with p < 0.01 should be green (#59a14f)."""
    def test_significance_heatmap_color_coding(self):
        # A vs C have very different medians, expect low p-value → green
        data = {
            "lo": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "hi": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
        }
        comp = paired_comparison_table(data)
        svg = plot_significance_heatmap(comp)
        # At least one green cell (p < 0.01)
        assert "#59a14f" in svg


class TestSignificanceHeatmapLabels:
    def test_significance_heatmap_labels(self):
        comp = paired_comparison_table(_sample_final_data())
        svg = plot_significance_heatmap(comp)
        for name in ("A", "B", "C"):
            assert name in svg


class TestSignificanceHeatmapEmpty:
    def test_empty(self):
        svg = plot_significance_heatmap({})
        assert svg.startswith("<svg")


# ---------------------------------------------------------------------------
# plot_ablation_bars
# ---------------------------------------------------------------------------


class TestAblationBarsReturnsSvg:
    def test_ablation_bars_returns_svg(self):
        data = {"baseline": 5.0, "no_feature_A": 6.0, "no_feature_B": 4.0}
        svg = plot_ablation_bars(data, baseline_name="baseline")
        assert svg.startswith("<svg")
        assert "</svg>" in svg


class TestAblationBarsHasBars:
    def test_ablation_bars_has_bars(self):
        data = {"baseline": 5.0, "variant": 4.0}
        svg = plot_ablation_bars(data, baseline_name="baseline")
        assert "<rect" in svg


class TestAblationBarsBaselineHighlighted:
    def test_ablation_bars_baseline_highlighted(self):
        data = {"baseline": 5.0, "variant": 4.0}
        svg = plot_ablation_bars(data, baseline_name="baseline")
        # Baseline should use the blue colour (#4e79a7)
        assert "#4e79a7" in svg


class TestAblationBarsSingleVariant:
    def test_ablation_bars_single_variant(self):
        data = {"base": 3.0}
        svg = plot_ablation_bars(data, baseline_name="base")
        assert svg.startswith("<svg")
        assert "base" in svg


class TestAblationBarsEmpty:
    def test_ablation_bars_empty(self):
        svg = plot_ablation_bars({}, baseline_name="x")
        assert svg.startswith("<svg")


class TestAblationBarsColorCoding:
    """Better-than-baseline variants should be green, worse should be orange."""
    def test_color_coding(self):
        data = {"baseline": 5.0, "better": 3.0, "worse": 8.0}
        svg = plot_ablation_bars(data, baseline_name="baseline")
        assert "#59a14f" in svg  # green for better
        assert "#f28e2b" in svg  # orange for worse


class TestAblationBarsHasTitle:
    def test_has_title(self):
        data = {"baseline": 5.0, "v1": 4.0}
        svg = plot_ablation_bars(data, baseline_name="baseline", title="My Ablation")
        assert "My Ablation" in svg

"""Tests for visualization/explainability.py -- SHAP chart functions."""

from __future__ import annotations

import pytest

from optimization_copilot.visualization.explainability import (
    auto_select_interaction,
    plot_shap_beeswarm,
    plot_shap_dependence,
    plot_shap_force,
    plot_shap_waterfall,
)
from optimization_copilot.visualization.models import PlotData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_svg(pd: PlotData) -> bool:
    """Check that the PlotData has a non-empty SVG string."""
    return pd.svg is not None and "<svg" in pd.svg and "</svg>" in pd.svg


# ---------------------------------------------------------------------------
# Waterfall
# ---------------------------------------------------------------------------

class TestWaterfall:
    def test_returns_valid_plot_data(self):
        pd = plot_shap_waterfall(
            trial_index=0,
            shap_values=[0.5, -0.3, 0.1],
            feature_names=["a", "b", "c"],
            base_value=1.0,
        )
        assert isinstance(pd, PlotData)
        assert pd.plot_type == "shap_waterfall"
        assert _valid_svg(pd)

    def test_sorted_features(self):
        pd = plot_shap_waterfall(
            trial_index=1,
            shap_values=[0.1, -0.5, 0.3],
            feature_names=["x", "y", "z"],
            base_value=0.0,
        )
        # Sorted by |SHAP| descending: y (0.5), z (0.3), x (0.1)
        assert pd.data["sorted_features"] == ["y", "z", "x"]

    def test_cumulative_values(self):
        pd = plot_shap_waterfall(
            trial_index=0,
            shap_values=[1.0, -0.5],
            feature_names=["a", "b"],
            base_value=2.0,
        )
        cum = pd.data["cumulative"]
        # base=2.0, +1.0 (|1.0|>|-0.5|) -> 3.0, -0.5 -> 2.5
        assert cum[0] == pytest.approx(2.0)
        assert cum[-1] == pytest.approx(2.5)

    def test_positive_and_negative_shap(self):
        pd = plot_shap_waterfall(
            trial_index=0,
            shap_values=[2.0, -1.0, 0.5, -0.3],
            feature_names=["a", "b", "c", "d"],
            base_value=0.0,
        )
        sorted_shap = pd.data["sorted_shap"]
        # Largest |SHAP| first: a=2.0, b=-1.0, c=0.5, d=-0.3
        assert sorted_shap[0] == pytest.approx(2.0)
        assert sorted_shap[1] == pytest.approx(-1.0)

    def test_metadata_includes_base_and_final(self):
        pd = plot_shap_waterfall(
            trial_index=5,
            shap_values=[1.0, 2.0],
            feature_names=["a", "b"],
            base_value=3.0,
        )
        assert pd.metadata["base_value"] == pytest.approx(3.0)
        assert pd.metadata["final_value"] == pytest.approx(6.0)
        assert pd.metadata["trial_index"] == 5

    def test_empty_shap_values(self):
        pd = plot_shap_waterfall(
            trial_index=0, shap_values=[], feature_names=[], base_value=1.0
        )
        assert pd.data["sorted_features"] == []
        assert _valid_svg(pd)


# ---------------------------------------------------------------------------
# Beeswarm
# ---------------------------------------------------------------------------

class TestBeeswarm:
    def test_returns_valid_plot_data(self):
        pd = plot_shap_beeswarm(
            shap_matrix=[[0.1, -0.2], [0.3, -0.4]],
            feature_values=[[1.0, 2.0], [3.0, 4.0]],
            feature_names=["x1", "x2"],
        )
        assert isinstance(pd, PlotData)
        assert pd.plot_type == "shap_beeswarm"
        assert _valid_svg(pd)

    def test_features_sorted_by_importance(self):
        pd = plot_shap_beeswarm(
            shap_matrix=[[0.1, -0.5, 0.3]],
            feature_values=[[1.0, 2.0, 3.0]],
            feature_names=["lo", "hi", "mid"],
        )
        # mean |SHAP|: lo=0.1, hi=0.5, mid=0.3 -> order: hi, mid, lo
        assert pd.data["feature_order"] == ["hi", "mid", "lo"]

    def test_single_feature(self):
        pd = plot_shap_beeswarm(
            shap_matrix=[[0.5], [-0.5]],
            feature_values=[[1.0], [2.0]],
            feature_names=["only"],
        )
        assert pd.data["feature_order"] == ["only"]
        assert _valid_svg(pd)

    def test_single_trial(self):
        pd = plot_shap_beeswarm(
            shap_matrix=[[0.1, 0.2]],
            feature_values=[[1.0, 2.0]],
            feature_names=["a", "b"],
        )
        assert pd.data["n_trials"] == 1
        assert _valid_svg(pd)

    def test_empty_matrix(self):
        pd = plot_shap_beeswarm(
            shap_matrix=[], feature_values=[], feature_names=[]
        )
        assert pd.data["feature_order"] == []
        assert _valid_svg(pd)

    def test_svg_contains_circles(self):
        """Beeswarm should render circles for data points."""
        pd = plot_shap_beeswarm(
            shap_matrix=[[0.1, 0.2], [0.3, 0.4]],
            feature_values=[[1.0, 2.0], [3.0, 4.0]],
            feature_names=["a", "b"],
        )
        assert "<circle" in pd.svg


# ---------------------------------------------------------------------------
# Dependence
# ---------------------------------------------------------------------------

class TestDependence:
    def test_returns_valid_plot_data(self):
        pd = plot_shap_dependence(
            feature_idx=0,
            shap_values=[0.1, 0.3, -0.2],
            feature_values=[1.0, 2.0, 3.0],
        )
        assert isinstance(pd, PlotData)
        assert pd.plot_type == "shap_dependence"
        assert _valid_svg(pd)

    def test_with_interaction(self):
        pd = plot_shap_dependence(
            feature_idx=0,
            shap_values=[0.1, 0.2],
            feature_values=[1.0, 2.0],
            interaction_feature_values=[5.0, 6.0],
            interaction_name="x2",
        )
        assert pd.data["has_interaction"] is True
        assert pd.data["interaction_name"] == "x2"

    def test_without_interaction(self):
        pd = plot_shap_dependence(
            feature_idx=1,
            shap_values=[0.5, -0.5],
            feature_values=[10.0, 20.0],
        )
        assert pd.data["has_interaction"] is False

    def test_data_contains_points(self):
        pd = plot_shap_dependence(
            feature_idx=0,
            shap_values=[0.1, 0.2, 0.3],
            feature_values=[1.0, 2.0, 3.0],
        )
        assert pd.data["n_points"] == 3
        assert len(pd.data["points"]) == 3
        assert "feature_value" in pd.data["points"][0]
        assert "shap_value" in pd.data["points"][0]

    def test_empty_data(self):
        pd = plot_shap_dependence(
            feature_idx=0, shap_values=[], feature_values=[]
        )
        assert pd.data["points"] == []
        assert _valid_svg(pd)

    def test_single_point(self):
        pd = plot_shap_dependence(
            feature_idx=0,
            shap_values=[1.0],
            feature_values=[5.0],
        )
        assert pd.data["n_points"] == 1
        assert _valid_svg(pd)


# ---------------------------------------------------------------------------
# Auto-select interaction
# ---------------------------------------------------------------------------

class TestAutoSelectInteraction:
    def test_selects_most_correlated(self):
        # Feature 1 correlates perfectly with SHAP of feature 0
        shap_matrix = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        fv_matrix = [[0.0, 1.0, 5.0], [0.0, 2.0, 5.0], [0.0, 3.0, 5.0]]
        idx = auto_select_interaction(0, shap_matrix, fv_matrix)
        assert idx == 1  # feature 1 has perfect correlation

    def test_single_feature(self):
        idx = auto_select_interaction(0, [[1.0]], [[2.0]])
        assert idx == 0


# ---------------------------------------------------------------------------
# Force plot
# ---------------------------------------------------------------------------

class TestForce:
    def test_returns_valid_plot_data(self):
        pd = plot_shap_force(
            trial_index=0,
            shap_values=[0.5, -0.3, 0.2],
            feature_names=["a", "b", "c"],
            feature_values=[1.0, 2.0, 3.0],
            base_value=1.0,
        )
        assert isinstance(pd, PlotData)
        assert pd.plot_type == "shap_force"
        assert _valid_svg(pd)

    def test_separates_positive_negative(self):
        pd = plot_shap_force(
            trial_index=0,
            shap_values=[1.0, -2.0, 0.5, -0.1],
            feature_names=["a", "b", "c", "d"],
            feature_values=[1.0, 2.0, 3.0, 4.0],
            base_value=5.0,
        )
        assert pd.metadata["n_positive"] == 2
        assert pd.metadata["n_negative"] == 2

    def test_final_value(self):
        pd = plot_shap_force(
            trial_index=0,
            shap_values=[1.0, -0.5],
            feature_names=["a", "b"],
            feature_values=[10.0, 20.0],
            base_value=3.0,
        )
        assert pd.data["final_value"] == pytest.approx(3.5)
        assert pd.data["base_value"] == pytest.approx(3.0)

    def test_all_positive(self):
        pd = plot_shap_force(
            trial_index=0,
            shap_values=[1.0, 2.0, 0.5],
            feature_names=["a", "b", "c"],
            feature_values=[1.0, 2.0, 3.0],
            base_value=0.0,
        )
        assert pd.metadata["n_positive"] == 3
        assert pd.metadata["n_negative"] == 0
        assert len(pd.data["negative"]) == 0

    def test_all_negative(self):
        pd = plot_shap_force(
            trial_index=0,
            shap_values=[-1.0, -2.0, -0.5],
            feature_names=["a", "b", "c"],
            feature_values=[1.0, 2.0, 3.0],
            base_value=10.0,
        )
        assert pd.metadata["n_negative"] == 3
        assert pd.metadata["n_positive"] == 0
        assert len(pd.data["positive"]) == 0

    def test_empty_shap_values(self):
        pd = plot_shap_force(
            trial_index=0,
            shap_values=[],
            feature_names=[],
            feature_values=[],
            base_value=5.0,
        )
        assert pd.data["positive"] == []
        assert pd.data["negative"] == []
        assert _valid_svg(pd)

    def test_positive_entries_contain_feature_info(self):
        pd = plot_shap_force(
            trial_index=0,
            shap_values=[0.7, -0.3],
            feature_names=["feat_a", "feat_b"],
            feature_values=[10.0, 20.0],
            base_value=1.0,
        )
        pos = pd.data["positive"]
        assert len(pos) == 1
        assert pos[0]["feature"] == "feat_a"
        assert pos[0]["shap"] == pytest.approx(0.7)
        assert pos[0]["value"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Cross-chart tests
# ---------------------------------------------------------------------------

class TestCrossChart:
    def test_all_charts_return_plot_data(self):
        """Smoke test: all four chart functions produce PlotData with SVG."""
        w = plot_shap_waterfall(0, [0.1, -0.2], ["a", "b"], 1.0)
        b = plot_shap_beeswarm([[0.1, 0.2]], [[1.0, 2.0]], ["a", "b"])
        d = plot_shap_dependence(0, [0.1, 0.2], [1.0, 2.0])
        f = plot_shap_force(0, [0.1, -0.2], ["a", "b"], [1.0, 2.0], 0.5)

        for pd in [w, b, d, f]:
            assert isinstance(pd, PlotData)
            assert _valid_svg(pd)

    def test_to_dict_roundtrip(self):
        """PlotData from chart functions survives serialisation."""
        pd = plot_shap_waterfall(0, [0.5], ["x"], 0.0)
        d = pd.to_dict()
        pd2 = PlotData.from_dict(d)
        assert pd2.plot_type == pd.plot_type
        assert pd2.svg == pd.svg

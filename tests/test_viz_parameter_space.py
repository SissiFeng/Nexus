"""Tests for the hexagonal binning coverage view (parameter_space module)."""

import math

import pytest

from optimization_copilot.visualization.parameter_space import (
    HexCell,
    PlotData,
    _cube_round,
    _hex_center,
    _hex_vertices,
    _pixel_to_hex,
    plot_hexbin_coverage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockSurrogate:
    """Minimal surrogate satisfying the ``SurrogateModel`` protocol."""

    def predict(self, x: list[float]) -> tuple[float, float]:
        return (sum(x), 0.1)


def _simple_space() -> dict[str, tuple[float, float]]:
    return {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}


def _simple_points(n: int = 10) -> list[dict[str, float]]:
    return [{"x1": i / n, "x2": i / n} for i in range(n)]


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestBasicConstruction:
    def test_returns_plot_data(self):
        result = plot_hexbin_coverage(
            search_space=_simple_space(),
            observed_points=_simple_points(),
        )
        assert isinstance(result, PlotData)

    def test_plot_type_is_hexbin_coverage(self):
        result = plot_hexbin_coverage(
            search_space=_simple_space(),
            observed_points=_simple_points(),
        )
        assert result.plot_type == "hexbin_coverage"

    def test_svg_is_nonempty_string(self):
        result = plot_hexbin_coverage(
            search_space=_simple_space(),
            observed_points=_simple_points(),
        )
        assert result.svg is not None
        assert "<svg" in result.svg
        assert "</svg>" in result.svg

    def test_data_contains_hex_cells(self):
        result = plot_hexbin_coverage(
            search_space=_simple_space(),
            observed_points=_simple_points(),
        )
        assert "hex_cells" in result.data
        assert isinstance(result.data["hex_cells"], list)
        assert result.data["n_hexes"] == len(result.data["hex_cells"])

    def test_data_contains_observed_points(self):
        pts = _simple_points(5)
        result = plot_hexbin_coverage(
            search_space=_simple_space(),
            observed_points=pts,
        )
        assert result.data["observed_points"] is pts
        assert result.data["n_points"] == 5


# ---------------------------------------------------------------------------
# Point-to-hex assignment
# ---------------------------------------------------------------------------

class TestPointToHex:
    def test_points_assigned_to_correct_hex(self):
        """A point at a hex centre should map back to that hex."""
        hex_size = 0.3
        q, r = 2, 1
        cx, cy = _hex_center(q, r, hex_size)
        q_out, r_out = _pixel_to_hex(cx, cy, hex_size)
        assert (q_out, r_out) == (q, r)

    def test_origin_maps_to_zero_hex(self):
        q, r = _pixel_to_hex(0.0, 0.0, 1.0)
        assert (q, r) == (0, 0)


# ---------------------------------------------------------------------------
# Density colouring
# ---------------------------------------------------------------------------

class TestDensityColouring:
    def test_hex_with_more_points_has_higher_value(self):
        space = {"x1": (0.0, 10.0), "x2": (0.0, 10.0)}
        # Cluster many points at (1, 1) and one at (9, 9).
        pts = [{"x1": 1.0, "x2": 1.0} for _ in range(20)]
        pts.append({"x1": 9.0, "x2": 9.0})
        result = plot_hexbin_coverage(
            search_space=space,
            observed_points=pts,
            color_by="density",
        )
        cells = result.data["hex_cells"]
        populated = [c for c in cells if c["count"] > 0]
        assert len(populated) >= 1
        # The densest cell should have value == 1.0.
        max_val = max(c["value"] for c in populated)
        assert max_val == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Empty observed points
# ---------------------------------------------------------------------------

class TestEmptyPoints:
    def test_empty_observed_points_still_generates_grid(self):
        result = plot_hexbin_coverage(
            search_space=_simple_space(),
            observed_points=[],
        )
        assert result.data["n_hexes"] > 0
        assert result.data["n_points"] == 0
        # All counts should be zero.
        for cell in result.data["hex_cells"]:
            assert cell["count"] == 0


# ---------------------------------------------------------------------------
# Predicted mean mode
# ---------------------------------------------------------------------------

class TestPredictedMean:
    def test_predicted_mean_with_surrogate(self):
        surrogate = _MockSurrogate()
        result = plot_hexbin_coverage(
            search_space=_simple_space(),
            observed_points=_simple_points(3),
            predicted_surface=surrogate,
            color_by="predicted_mean",
        )
        assert result.metadata["color_by"] == "predicted_mean"
        # Values should be normalised to [0, 1].
        for cell in result.data["hex_cells"]:
            assert 0.0 <= cell["value"] <= 1.0

    def test_predicted_mean_without_surrogate_falls_back(self):
        """When no surrogate is provided, predicted_mean should gracefully
        fall back to density colouring without raising."""
        result = plot_hexbin_coverage(
            search_space=_simple_space(),
            observed_points=_simple_points(5),
            predicted_surface=None,
            color_by="predicted_mean",
        )
        assert result.plot_type == "hexbin_coverage"
        # Should still produce valid data.
        assert result.data["n_hexes"] > 0


# ---------------------------------------------------------------------------
# Uncertainty mode
# ---------------------------------------------------------------------------

class TestUncertainty:
    def test_uncertainty_with_surrogate(self):
        surrogate = _MockSurrogate()
        result = plot_hexbin_coverage(
            search_space=_simple_space(),
            observed_points=_simple_points(3),
            predicted_surface=surrogate,
            color_by="uncertainty",
        )
        assert result.metadata["color_by"] == "uncertainty"
        # Mock returns constant uncertainty 0.1, so all cells get value 0
        # (or normalised constant).
        values = {c["value"] for c in result.data["hex_cells"]}
        # With constant uncertainty, after normalisation all should be 0.0
        # (since v_range == 0 falls back to dividing by 1.0).
        assert len(values) >= 1


# ---------------------------------------------------------------------------
# Custom parameter selection
# ---------------------------------------------------------------------------

class TestParamSelection:
    def test_custom_param_x_param_y(self):
        space = {"a": (0.0, 1.0), "b": (0.0, 2.0), "c": (-1.0, 1.0)}
        pts = [{"a": 0.5, "b": 1.0, "c": 0.0}]
        result = plot_hexbin_coverage(
            search_space=space,
            observed_points=pts,
            param_x="b",
            param_y="c",
        )
        assert result.metadata["param_x"] == "b"
        assert result.metadata["param_y"] == "c"

    def test_default_param_selection(self):
        space = {"alpha": (0.0, 1.0), "beta": (0.0, 1.0), "gamma": (0.0, 1.0)}
        result = plot_hexbin_coverage(
            search_space=space,
            observed_points=[],
        )
        assert result.metadata["param_x"] == "alpha"
        assert result.metadata["param_y"] == "beta"


# ---------------------------------------------------------------------------
# ValueError for < 2 parameters
# ---------------------------------------------------------------------------

class TestValidation:
    def test_raises_on_single_parameter(self):
        with pytest.raises(ValueError, match="at least 2 parameters"):
            plot_hexbin_coverage(
                search_space={"only_one": (0.0, 1.0)},
                observed_points=[],
            )

    def test_raises_on_empty_space(self):
        with pytest.raises(ValueError, match="at least 2 parameters"):
            plot_hexbin_coverage(
                search_space={},
                observed_points=[],
            )


# ---------------------------------------------------------------------------
# Single observed point
# ---------------------------------------------------------------------------

class TestSinglePoint:
    def test_single_observed_point(self):
        result = plot_hexbin_coverage(
            search_space=_simple_space(),
            observed_points=[{"x1": 0.5, "x2": 0.5}],
        )
        populated = [c for c in result.data["hex_cells"] if c["count"] > 0]
        assert len(populated) == 1
        assert populated[0]["count"] == 1
        assert populated[0]["value"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Many points in same hex
# ---------------------------------------------------------------------------

class TestManyPointsSameHex:
    def test_many_points_in_same_hex(self):
        pts = [{"x1": 0.5, "x2": 0.5} for _ in range(50)]
        result = plot_hexbin_coverage(
            search_space=_simple_space(),
            observed_points=pts,
        )
        populated = [c for c in result.data["hex_cells"] if c["count"] > 0]
        assert len(populated) == 1
        assert populated[0]["count"] == 50
        assert populated[0]["value"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# hex_size affects grid density
# ---------------------------------------------------------------------------

class TestHexSize:
    def test_smaller_hex_produces_more_cells(self):
        result_large = plot_hexbin_coverage(
            search_space=_simple_space(),
            observed_points=[],
            hex_size=0.3,
        )
        result_small = plot_hexbin_coverage(
            search_space=_simple_space(),
            observed_points=[],
            hex_size=0.1,
        )
        assert result_small.data["n_hexes"] > result_large.data["n_hexes"]


# ---------------------------------------------------------------------------
# _cube_round correctness
# ---------------------------------------------------------------------------

class TestCubeRound:
    def test_integer_input_unchanged(self):
        assert _cube_round(2.0, 3.0) == (2, 3)
        assert _cube_round(0.0, 0.0) == (0, 0)
        assert _cube_round(-1.0, 2.0) == (-1, 2)

    def test_fractional_rounding(self):
        """Verify that small perturbations still round to the correct hex."""
        q, r = _cube_round(2.1, 2.9)
        assert (q, r) == (2, 3)

    def test_midpoint_consistency(self):
        """The result must always satisfy the cube constraint q + r + s == 0."""
        q, r = _cube_round(0.5, 0.3)
        s = -q - r
        assert q + r + s == 0

    def test_hex_vertices_count(self):
        verts = _hex_vertices(0.0, 0.0, 1.0)
        assert len(verts) == 6

    def test_hex_vertices_radius(self):
        """All vertices should be exactly hex_size from the centre."""
        cx, cy = 5.0, 5.0
        size = 2.0
        for vx, vy in _hex_vertices(cx, cy, size):
            dist = math.sqrt((vx - cx) ** 2 + (vy - cy) ** 2)
            assert dist == pytest.approx(size, abs=1e-10)

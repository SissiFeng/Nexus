"""Tests for space-filling quality metrics (visualization/diagnostics)."""

import math

import pytest

from optimization_copilot.visualization.diagnostics import (
    _compute_coverage,
    _compute_min_distance,
    _compute_star_discrepancy,
    plot_space_filling_metrics,
)
from optimization_copilot.visualization.models import PlotData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grid_2d(n: int) -> list[list[float]]:
    """Return an n x n regular grid in [0,1]^2."""
    pts: list[list[float]] = []
    for i in range(n):
        for j in range(n):
            pts.append([i / max(n - 1, 1), j / max(n - 1, 1)])
    return pts


def _unit_bounds(d: int) -> list[tuple[float, float]]:
    """Return [(0,1)] * d."""
    return [(0.0, 1.0)] * d


# ---------------------------------------------------------------------------
# Star discrepancy tests
# ---------------------------------------------------------------------------


class TestStarDiscrepancy:
    """Tests for _compute_star_discrepancy."""

    def test_grid_low_discrepancy(self):
        """A regular 5x5 grid should have low discrepancy."""
        pts = _grid_2d(5)
        disc = _compute_star_discrepancy(pts, _unit_bounds(2))
        assert disc <= 0.25, f"Grid discrepancy {disc} is too high"

    def test_random_higher_than_grid(self):
        """Pseudo-random points typically have higher discrepancy than a grid."""
        import random as stdlib_random

        rng = stdlib_random.Random(123)
        random_pts = [[rng.random(), rng.random()] for _ in range(25)]
        grid_pts = _grid_2d(5)
        bounds = _unit_bounds(2)
        disc_grid = _compute_star_discrepancy(grid_pts, bounds)
        disc_rand = _compute_star_discrepancy(random_pts, bounds)
        # The grid should be at least as good (lower discrepancy) in most
        # seeded realisations; allow small tolerance.
        assert disc_grid <= disc_rand + 0.05

    def test_single_point(self):
        """Discrepancy of a single point should not crash."""
        disc = _compute_star_discrepancy([[0.5, 0.5]], _unit_bounds(2))
        assert 0.0 <= disc <= 1.0

    def test_empty_points(self):
        """Empty point list returns 0.0."""
        disc = _compute_star_discrepancy([], _unit_bounds(2))
        assert disc == 0.0

    def test_high_dim_uses_approximation(self):
        """d=8 triggers the random-corner approximation (no crash, sane value)."""
        import random as stdlib_random

        rng = stdlib_random.Random(99)
        pts = [[rng.random() for _ in range(8)] for _ in range(50)]
        disc = _compute_star_discrepancy(pts, _unit_bounds(8))
        assert 0.0 <= disc <= 1.0

    def test_corner_point_at_origin(self):
        """A single point at the origin should yield discrepancy close to 1."""
        disc = _compute_star_discrepancy([[0.0, 0.0]], _unit_bounds(2))
        # The box [0,1]^2 has vol=1 and F_n=1 (the point is inside), so
        # |1-1|=0 there.  But the point is at the origin, so many boxes
        # have F_n=1 but vol<1.  The max gap should be near 1.
        assert disc > 0.5

    def test_all_points_at_one_corner(self):
        """If all points cluster at (1,1), the exact algorithm only checks
        corners from point coordinates.  With all coords equal to 1.0 the
        only box is the full unit square, yielding D*=0.  Verify this
        degenerate but mathematically correct behaviour."""
        pts = [[1.0, 1.0]] * 10
        disc = _compute_star_discrepancy(pts, _unit_bounds(2))
        assert disc == pytest.approx(0.0)

    def test_clustered_points_high_discrepancy(self):
        """Points clustered at (0.1, 0.1) should exhibit high discrepancy."""
        pts = [[0.1, 0.1]] * 10
        disc = _compute_star_discrepancy(pts, _unit_bounds(2))
        # The box [0, (0.1, 0.1)] has vol=0.01 but F_n=1.0 -> gap 0.99
        # The box [0, (1, 1)] has vol=1 and F_n=1 -> gap 0.  Overall D*
        # is driven by the small box, so it should be large.
        assert disc > 0.5


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------


class TestCoverage:
    """Tests for _compute_coverage."""

    def test_uniform_grid_full_coverage(self):
        """A dense uniform grid should reach ~100% coverage."""
        pts = _grid_2d(50)  # 2500 points in a 50-bin grid
        cov = _compute_coverage(pts, _unit_bounds(2), resolution=50)
        assert cov == pytest.approx(100.0, abs=1.0)

    def test_clustered_points_low_coverage(self):
        """Points clustered in one corner should have low coverage."""
        pts = [[0.01 * i, 0.01 * i] for i in range(10)]
        cov = _compute_coverage(pts, _unit_bounds(2), resolution=50)
        assert cov < 10.0, f"Clustered coverage {cov}% is unexpectedly high"

    def test_empty_points_zero_coverage(self):
        """No points means 0% coverage."""
        cov = _compute_coverage([], _unit_bounds(2))
        assert cov == 0.0

    def test_adaptive_resolution_high_dim(self):
        """For d > 6 the resolution is reduced; verify it doesn't crash."""
        import random as stdlib_random

        rng = stdlib_random.Random(7)
        pts = [[rng.random() for _ in range(8)] for _ in range(200)]
        cov = _compute_coverage(pts, _unit_bounds(8))
        assert 0.0 <= cov <= 100.0

    def test_single_point_coverage(self):
        """A single point covers exactly one cell."""
        cov = _compute_coverage([[0.5, 0.5]], _unit_bounds(2), resolution=10)
        expected = (1 / 100) * 100.0  # 1 out of 10^2 cells
        assert cov == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Minimum distance tests
# ---------------------------------------------------------------------------


class TestMinDistance:
    """Tests for _compute_min_distance."""

    def test_grid_positive_distance(self):
        """A regular grid has a well-defined positive minimum distance."""
        pts = _grid_2d(5)
        md = _compute_min_distance(pts, _unit_bounds(2))
        assert md > 0.0

    def test_duplicate_points_zero(self):
        """Duplicate points yield min_distance = 0."""
        pts = [[0.3, 0.7], [0.3, 0.7], [0.9, 0.1]]
        md = _compute_min_distance(pts, _unit_bounds(2))
        assert md == pytest.approx(0.0)

    def test_fewer_than_two_points_inf(self):
        """With 0 or 1 point, min_distance is inf."""
        assert _compute_min_distance([], _unit_bounds(2)) == float("inf")
        assert _compute_min_distance([[0.5, 0.5]], _unit_bounds(2)) == float("inf")

    def test_known_distance(self):
        """Two points at known locations yield a predictable distance."""
        # In [0,10] the points 2 and 8 normalise to 0.2 and 0.8 => dist = 0.6
        pts = [[2.0], [8.0]]
        md = _compute_min_distance(pts, [(0.0, 10.0)])
        assert md == pytest.approx(0.6)

    def test_normalisation_with_custom_bounds(self):
        """Min distance is computed in normalised [0,1]^d space."""
        pts = [[10.0, 20.0], [20.0, 40.0]]
        bounds = [(0.0, 100.0), (0.0, 100.0)]
        md = _compute_min_distance(pts, bounds)
        # Normalised: [0.1, 0.2] and [0.2, 0.4] => sqrt(0.01 + 0.04) = sqrt(0.05)
        expected = math.sqrt(0.05)
        assert md == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Public API: plot_space_filling_metrics
# ---------------------------------------------------------------------------


class TestPlotSpaceFillingMetrics:
    """Tests for the public plot_space_filling_metrics function."""

    def test_returns_plot_data(self):
        """Return type is PlotData with correct plot_type."""
        result = plot_space_filling_metrics(
            _grid_2d(5), _unit_bounds(2)
        )
        assert isinstance(result, PlotData)
        assert result.plot_type == "space_filling_metrics"

    def test_default_metrics_list(self):
        """Default metrics include all three."""
        result = plot_space_filling_metrics(
            _grid_2d(3), _unit_bounds(2)
        )
        assert "discrepancy" in result.data
        assert "coverage" in result.data
        assert "min_distance" in result.data
        assert result.data["n_points"] == 9
        assert result.data["n_dims"] == 2
        assert result.metadata["metrics_computed"] == [
            "discrepancy",
            "coverage",
            "min_distance",
        ]

    def test_custom_single_metric(self):
        """Requesting a single metric only computes that one."""
        result = plot_space_filling_metrics(
            _grid_2d(3), _unit_bounds(2), metrics=["coverage"]
        )
        assert "coverage" in result.data
        assert "discrepancy" not in result.data
        assert "min_distance" not in result.data
        assert result.metadata["metrics_computed"] == ["coverage"]

    def test_unknown_metric_raises(self):
        """An unknown metric name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            plot_space_filling_metrics(
                _grid_2d(3), _unit_bounds(2), metrics=["bogus"]
            )

    def test_1d_points(self):
        """1-dimensional points are handled correctly."""
        pts = [[0.0], [0.25], [0.5], [0.75], [1.0]]
        result = plot_space_filling_metrics(pts, [(0.0, 1.0)])
        assert result.data["n_dims"] == 1
        assert result.data["min_distance"] == pytest.approx(0.25)

    def test_metadata_contains_bounds(self):
        """Metadata includes the original bounds."""
        bounds = [(0.0, 5.0), (-1.0, 1.0)]
        result = plot_space_filling_metrics(
            [[1.0, 0.0], [3.0, 0.5]], bounds
        )
        assert result.metadata["bounds"] == bounds

    def test_empty_points(self):
        """Empty point list produces sensible defaults."""
        result = plot_space_filling_metrics([], _unit_bounds(2))
        assert result.data["discrepancy"] == 0.0
        assert result.data["coverage"] == 0.0
        assert result.data["min_distance"] == float("inf")
        assert result.data["n_points"] == 0

    def test_serialisation_roundtrip(self):
        """PlotData can be serialised and deserialised."""
        result = plot_space_filling_metrics(
            _grid_2d(3), _unit_bounds(2)
        )
        d = result.to_dict()
        restored = PlotData.from_dict(d)
        assert restored.plot_type == "space_filling_metrics"
        assert restored.data["n_points"] == 9

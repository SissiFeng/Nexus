"""Tests for design-space exploration visualizations (v3 spec section 7)."""

from __future__ import annotations

import math

import pytest

from optimization_copilot.visualization.design_space import (
    plot_forward_inverse_design,
    plot_isom_landscape,
    plot_latent_space_exploration,
)
from optimization_copilot.visualization.models import PlotData, SurrogateModel


# ---------------------------------------------------------------------------
# Test helper: mock surrogate model
# ---------------------------------------------------------------------------


class _MockSurrogate:
    """Satisfies the ``SurrogateModel`` protocol.

    ``predict(x)`` returns ``(sum(x), 0.1)``.
    """

    def predict(self, x: list[float]) -> tuple[float, float]:
        return (sum(x), 0.1)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_3d_data(n: int = 30, seed: int = 42):
    """Generate *n* random 3-D points and objective values."""
    import random

    rng = random.Random(seed)
    X = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(n)]
    Y = [sum(row) + rng.gauss(0, 0.1) for row in X]
    return X, Y


def _make_2d_data(n: int = 20, seed: int = 42):
    """Generate *n* random 2-D points and objective values."""
    import random

    rng = random.Random(seed)
    X = [[rng.uniform(0, 1), rng.uniform(0, 1)] for _ in range(n)]
    Y = [x[0] + x[1] for x in X]
    return X, Y


# ===========================================================================
# PCA tests (7)
# ===========================================================================


class TestPCA:
    """Tests for PCA projection via ``plot_latent_space_exploration``."""

    def test_returns_valid_plotdata_with_svg(self):
        X, Y = _make_3d_data()
        result = plot_latent_space_exploration(X, Y, method="pca")
        assert isinstance(result, PlotData)
        assert result.plot_type == "latent_space_exploration"
        assert result.svg is not None
        assert "<svg" in result.svg
        assert "</svg>" in result.svg

    def test_2d_data_unchanged(self):
        """2-D data projected to 2-D should preserve point count."""
        X, Y = _make_2d_data(n=15)
        result = plot_latent_space_exploration(X, Y, method="pca")
        pts = result.data["points_2d"]
        assert len(pts) == 15
        assert all(len(p) == 2 for p in pts)

    def test_3d_data_projected_to_2d(self):
        X, Y = _make_3d_data(n=25)
        result = plot_latent_space_exploration(X, Y, method="pca")
        pts = result.data["points_2d"]
        assert len(pts) == 25
        assert all(len(p) == 2 for p in pts)

    def test_explained_variance_ratios_in_metadata(self):
        X, Y = _make_3d_data(n=30)
        result = plot_latent_space_exploration(X, Y, method="pca")
        ev = result.metadata.get("explained_variance_ratios")
        assert ev is not None
        assert len(ev) == 2
        # Each ratio should be in [0, 1].
        for r in ev:
            assert 0.0 <= r <= 1.0 + 1e-9
        # Sum should be <= 1.0 (partial variance).
        assert sum(ev) <= 1.0 + 1e-9

    def test_deterministic_with_same_seed(self):
        X, Y = _make_3d_data()
        r1 = plot_latent_space_exploration(X, Y, method="pca", seed=123)
        r2 = plot_latent_space_exploration(X, Y, method="pca", seed=123)
        assert r1.data["points_2d"] == r2.data["points_2d"]

    def test_handles_identical_points(self):
        """Degenerate case: all points identical -> zero variance PCA."""
        X = [[1.0, 2.0, 3.0]] * 10
        Y = [0.5] * 10
        result = plot_latent_space_exploration(X, Y, method="pca")
        assert isinstance(result, PlotData)
        # All projected points should be at the origin (zero variance).
        for p in result.data["points_2d"]:
            assert abs(p[0]) < 1e-6
            assert abs(p[1]) < 1e-6

    def test_color_modes(self):
        X, Y = _make_3d_data(n=10)
        for mode in ("objective", "iteration"):
            result = plot_latent_space_exploration(X, Y, method="pca", color_by=mode)
            assert result.metadata["color_by"] == mode
            assert len(result.data["hex_colors"]) == 10
            # Verify hex colours are valid.
            for c in result.data["hex_colors"]:
                assert c.startswith("#")
                assert len(c) == 7  # #RRGGBB


# ===========================================================================
# t-SNE tests (5)
# ===========================================================================


class TestTSNE:
    """Tests for t-SNE projection via ``plot_latent_space_exploration``."""

    def test_returns_valid_plotdata(self):
        X, Y = _make_3d_data(n=20)
        result = plot_latent_space_exploration(X, Y, method="tsne")
        assert isinstance(result, PlotData)
        assert result.plot_type == "latent_space_exploration"
        assert result.svg is not None

    def test_output_has_n_points_in_2d(self):
        X, Y = _make_3d_data(n=15)
        result = plot_latent_space_exploration(X, Y, method="tsne")
        pts = result.data["points_2d"]
        assert len(pts) == 15
        assert all(len(p) == 2 for p in pts)

    def test_deterministic_with_same_seed(self):
        X, Y = _make_3d_data(n=10)
        r1 = plot_latent_space_exploration(X, Y, method="tsne", seed=99)
        r2 = plot_latent_space_exploration(X, Y, method="tsne", seed=99)
        assert r1.data["points_2d"] == r2.data["points_2d"]

    def test_handles_small_n(self):
        """t-SNE should not crash with fewer than 5 points."""
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        Y = [1.0, 2.0, 3.0]
        result = plot_latent_space_exploration(X, Y, method="tsne")
        assert len(result.data["points_2d"]) == 3

    def test_different_perplexity_doesnt_crash(self):
        """Ensuring very small n (where perplexity is clamped) works."""
        X = [[1.0], [2.0]]
        Y = [0.0, 1.0]
        # Perplexity will be clamped to n-1 = 1 internally.
        result = plot_latent_space_exploration(X, Y, method="tsne")
        assert isinstance(result, PlotData)
        assert len(result.data["points_2d"]) == 2


# ===========================================================================
# iSOM tests (6)
# ===========================================================================


class TestISOM:
    """Tests for ``plot_isom_landscape``."""

    def test_returns_valid_plotdata(self):
        X, Y = _make_2d_data(n=30)
        result = plot_isom_landscape(X, Y, grid_size=(5, 5))
        assert isinstance(result, PlotData)
        assert result.plot_type == "isom_landscape"
        assert result.svg is not None
        assert "<svg" in result.svg

    def test_grid_dimensions_correct(self):
        X, Y = _make_2d_data(n=20)
        result = plot_isom_landscape(X, Y, grid_size=(4, 6))
        data = result.data
        assert data["grid_size"] == [4, 6]
        assert len(data["prototypes"]) == 4 * 6
        assert len(data["node_colors"]) == 4 * 6
        assert len(data["u_matrix"]) == 4 * 6

    def test_u_matrix_computed(self):
        X, Y = _make_2d_data(n=20)
        result = plot_isom_landscape(X, Y, grid_size=(5, 5))
        u = result.data["u_matrix"]
        assert len(u) == 25
        # All U-matrix values should be non-negative.
        assert all(v >= 0.0 for v in u)

    def test_deterministic_with_same_seed(self):
        X, Y = _make_2d_data(n=15)
        r1 = plot_isom_landscape(X, Y, grid_size=(4, 4), seed=77)
        r2 = plot_isom_landscape(X, Y, grid_size=(4, 4), seed=77)
        assert r1.data["prototypes"] == r2.data["prototypes"]
        assert r1.data["u_matrix"] == r2.data["u_matrix"]

    def test_small_dataset(self):
        """Fewer data points than grid nodes should not crash."""
        X = [[0.0, 0.0], [1.0, 1.0]]
        Y = [0.0, 1.0]
        result = plot_isom_landscape(X, Y, grid_size=(5, 5))
        assert isinstance(result, PlotData)
        # Only 2 unique BMU mappings expected.
        mapped_count = sum(
            1 for c in result.data["node_colors"] if c is not None
        )
        assert mapped_count >= 1  # At least one node has mapped data.

    def test_node_colours_based_on_y(self):
        """Nodes mapped to low-Y points should have lower colour values."""
        import random

        rng = random.Random(42)
        X = [[rng.uniform(0, 1), rng.uniform(0, 1)] for _ in range(50)]
        Y = [x[0] + x[1] for x in X]  # Objective correlates with position.
        result = plot_isom_landscape(X, Y, grid_size=(5, 5), seed=42)
        colors = result.data["node_colors"]
        valid = [c for c in colors if c is not None]
        # There should be variation in node colours.
        assert len(valid) > 1
        assert max(valid) > min(valid)


# ===========================================================================
# Forward / Inverse design tests (7)
# ===========================================================================


class TestForwardInverse:
    """Tests for ``plot_forward_inverse_design``."""

    @pytest.fixture()
    def surrogate(self) -> _MockSurrogate:
        return _MockSurrogate()

    def test_forward_returns_plotdata_with_predictions(self, surrogate):
        result = plot_forward_inverse_design(
            parameter_space={"x": (0, 1), "y": (0, 1)},
            objective_space={"f": (0, 2)},
            mapping_model=surrogate,
            grid_resolution=5,
        )
        assert isinstance(result, PlotData)
        assert result.plot_type == "forward_inverse_design"
        assert result.svg is not None
        assert len(result.data["predictions"]) == 5 * 5
        assert len(result.data["uncertainties"]) == 5 * 5
        # All uncertainties should be 0.1 from mock.
        for u in result.data["uncertainties"]:
            assert abs(u - 0.1) < 1e-9

    def test_forward_grid_resolution_controls_density(self, surrogate):
        r5 = plot_forward_inverse_design(
            parameter_space={"x": (0, 1)},
            objective_space={"f": (0, 1)},
            mapping_model=surrogate,
            grid_resolution=5,
        )
        r10 = plot_forward_inverse_design(
            parameter_space={"x": (0, 1)},
            objective_space={"f": (0, 1)},
            mapping_model=surrogate,
            grid_resolution=10,
        )
        assert r5.data["n_grid_points"] == 5
        assert r10.data["n_grid_points"] == 10
        assert r10.data["n_grid_points"] > r5.data["n_grid_points"]

    def test_inverse_with_targets_highlights_feasible(self, surrogate):
        result = plot_forward_inverse_design(
            parameter_space={"x": (0, 1), "y": (0, 1)},
            objective_space={"f": (0, 2)},
            mapping_model=surrogate,
            target_objectives=[1.0],
            grid_resolution=10,
            tolerance=0.15,
        )
        # With sum(x, y) as model and target=1.0, tolerance=0.15,
        # points where |x+y - 1.0| <= 0.15 should be feasible.
        assert len(result.data["feasible_indices"]) > 0
        for idx in result.data["feasible_indices"]:
            pred = result.data["predictions"][idx]
            assert abs(pred - 1.0) <= 0.15 + 1e-9

    def test_inverse_no_feasible_points(self, surrogate):
        """Target far outside model range -> empty feasible set."""
        result = plot_forward_inverse_design(
            parameter_space={"x": (0, 1)},
            objective_space={"f": (0, 1)},
            mapping_model=surrogate,
            target_objectives=[100.0],
            grid_resolution=10,
            tolerance=0.01,
        )
        assert result.data["feasible_indices"] == []
        assert result.data["feasible_points"] == []

    def test_inverse_without_targets_returns_only_forward(self, surrogate):
        result = plot_forward_inverse_design(
            parameter_space={"x": (0, 1), "y": (0, 1)},
            objective_space={"f": (0, 2)},
            mapping_model=surrogate,
            target_objectives=None,
            grid_resolution=5,
        )
        assert result.data["feasible_indices"] == []
        assert len(result.data["predictions"]) == 25

    def test_mock_surrogate_satisfies_protocol(self):
        m = _MockSurrogate()
        assert isinstance(m, SurrogateModel)
        mean, unc = m.predict([1.0, 2.0, 3.0])
        assert mean == pytest.approx(6.0)
        assert unc == pytest.approx(0.1)

    def test_1d_parameter_space(self, surrogate):
        result = plot_forward_inverse_design(
            parameter_space={"x": (0, 10)},
            objective_space={"f": (0, 10)},
            mapping_model=surrogate,
            target_objectives=[5.0],
            grid_resolution=20,
            tolerance=0.5,
        )
        assert isinstance(result, PlotData)
        # 1-D parameter space -> grid_resolution points.
        assert result.data["n_grid_points"] == 20
        # Check feasible points are within tolerance.
        for idx in result.data["feasible_indices"]:
            pred = result.data["predictions"][idx]
            assert abs(pred - 5.0) <= 0.5 + 1e-9

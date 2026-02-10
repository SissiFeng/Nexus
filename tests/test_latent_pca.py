"""Numerical tests for the pure-stdlib PCA implementation.

Tests cover covariance matrix computation, power iteration eigendecomposition,
standardisation, explained variance ratios, and determinism guarantees.
"""

from __future__ import annotations

import math

import pytest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.latent.transform import (
    LatentTransform,
    _build_data_matrix,
    _covariance_matrix,
    _power_iteration_eigendecomp,
)


# ── Helpers ──────────────────────────────────────────────


def _make_snapshot_2d(
    xs: list[float],
    ys: list[float],
    *,
    param_names: tuple[str, str] = ("x", "y"),
) -> CampaignSnapshot:
    """Create a CampaignSnapshot with two continuous params from x,y lists.

    Each (x, y) pair becomes a successful observation with a dummy KPI.
    """
    assert len(xs) == len(ys), "xs and ys must have the same length"
    observations = [
        Observation(
            iteration=i,
            parameters={param_names[0]: x, param_names[1]: y},
            kpi_values={"obj": float(i)},
            qc_passed=True,
            is_failure=False,
        )
        for i, (x, y) in enumerate(zip(xs, ys))
    ]
    specs = [
        ParameterSpec(name=param_names[0], type=VariableType.CONTINUOUS, lower=-100.0, upper=100.0),
        ParameterSpec(name=param_names[1], type=VariableType.CONTINUOUS, lower=-100.0, upper=100.0),
    ]
    return CampaignSnapshot(
        campaign_id="test-pca",
        parameter_specs=specs,
        observations=observations,
        objective_names=["obj"],
        objective_directions=["maximize"],
    )


# ── TestCovarianceMatrix ─────────────────────────────────


class TestCovarianceMatrix:
    """Known 2x2 data -> correct covariance values."""

    def test_identity_covariance_from_standardised_data(self):
        """Centred data [[1,-1],[-1,1]] should give covariance [[2, -2],[-2, 2]]/(n-1)."""
        # Two points, already centred: (1, -1) and (-1, 1)
        data = [[1.0, -1.0], [-1.0, 1.0]]
        cov = _covariance_matrix(data, n_samples=2, n_features=2)
        # With n=2, divisor = max(2-1, 1) = 1
        # cov[0][0] = (1*1 + (-1)*(-1)) / 1 = 2.0
        # cov[0][1] = (1*(-1) + (-1)*1) / 1 = -2.0
        assert cov[0][0] == pytest.approx(2.0)
        assert cov[0][1] == pytest.approx(-2.0)
        assert cov[1][0] == pytest.approx(-2.0)
        assert cov[1][1] == pytest.approx(2.0)

    def test_known_3x2_covariance(self):
        """Three centred observations with 2 features."""
        # data: [[1, 2], [3, 4], [-4, -6]] (centred: mean is [0, 0])
        # Actually let's use truly centred data: mean([1,3,-4]) = 0, mean([2,4,-6]) = 0
        data = [[1.0, 2.0], [3.0, 4.0], [-4.0, -6.0]]
        cov = _covariance_matrix(data, n_samples=3, n_features=2)
        # divisor = 2
        # cov[0][0] = (1 + 9 + 16) / 2 = 13.0
        # cov[0][1] = (2 + 12 + 24) / 2 = 19.0
        # cov[1][1] = (4 + 16 + 36) / 2 = 28.0
        assert cov[0][0] == pytest.approx(13.0)
        assert cov[0][1] == pytest.approx(19.0)
        assert cov[1][0] == pytest.approx(19.0)
        assert cov[1][1] == pytest.approx(28.0)

    def test_symmetric(self):
        """Covariance matrix must be symmetric."""
        data = [[2.0, 3.0, 1.0], [-1.0, 0.5, -0.5], [0.0, -2.0, 3.0]]
        cov = _covariance_matrix(data, n_samples=3, n_features=3)
        for i in range(3):
            for j in range(3):
                assert cov[i][j] == pytest.approx(cov[j][i])

    def test_single_sample_divisor(self):
        """With n=1, divisor = max(0, 1) = 1 (avoid division by zero)."""
        data = [[5.0, 3.0]]
        cov = _covariance_matrix(data, n_samples=1, n_features=2)
        # cov[0][0] = 25/1 = 25, cov[0][1] = 15/1 = 15
        assert cov[0][0] == pytest.approx(25.0)
        assert cov[0][1] == pytest.approx(15.0)


# ── TestPowerIteration ───────────────────────────────────


class TestPowerIteration:
    """Test eigenvalue/eigenvector extraction via power iteration."""

    def test_diagonal_matrix_eigenvalues(self):
        """Diagonal matrix -> eigenvalues recovered exactly (sorted by discovery)."""
        # [[5, 0], [0, 2]] has eigenvalues 5 and 2.
        matrix = [[5.0, 0.0], [0.0, 2.0]]
        eigenvalues, eigenvectors = _power_iteration_eigendecomp(
            matrix, n_components=2, seed=42, max_iterations=200, tol=1e-10,
        )
        # Sort eigenvalues descending to compare
        sorted_evals = sorted(eigenvalues, reverse=True)
        assert sorted_evals[0] == pytest.approx(5.0, abs=1e-6)
        assert sorted_evals[1] == pytest.approx(2.0, abs=1e-6)

    def test_known_2x2_symmetric_matrix(self):
        """Known 2x2 symmetric matrix with known eigenvalues.

        [[2, 1], [1, 2]] has eigenvalues 3 and 1.
        """
        matrix = [[2.0, 1.0], [1.0, 2.0]]
        eigenvalues, eigenvectors = _power_iteration_eigendecomp(
            matrix, n_components=2, seed=42, max_iterations=300, tol=1e-10,
        )
        sorted_evals = sorted(eigenvalues, reverse=True)
        assert sorted_evals[0] == pytest.approx(3.0, abs=1e-4)
        assert sorted_evals[1] == pytest.approx(1.0, abs=1e-4)

    def test_eigenvectors_are_unit_length(self):
        """Each eigenvector must have L2 norm == 1."""
        matrix = [[4.0, 1.0], [1.0, 3.0]]
        _, eigenvectors = _power_iteration_eigendecomp(
            matrix, n_components=2, seed=42, max_iterations=200, tol=1e-10,
        )
        for vec in eigenvectors:
            norm = math.sqrt(sum(v ** 2 for v in vec))
            assert norm == pytest.approx(1.0, abs=1e-6)

    def test_eigenvalues_are_non_negative(self):
        """Eigenvalues from a positive semi-definite matrix must be >= 0."""
        # A covariance matrix is always PSD. Use one here.
        matrix = [[3.0, 1.0], [1.0, 2.0]]
        eigenvalues, _ = _power_iteration_eigendecomp(
            matrix, n_components=2, seed=42, max_iterations=200, tol=1e-10,
        )
        for ev in eigenvalues:
            assert ev >= 0.0

    def test_3x3_diagonal(self):
        """3x3 diagonal matrix eigenvalues."""
        matrix = [[10.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]]
        eigenvalues, _ = _power_iteration_eigendecomp(
            matrix, n_components=3, seed=7, max_iterations=200, tol=1e-10,
        )
        sorted_evals = sorted(eigenvalues, reverse=True)
        assert sorted_evals[0] == pytest.approx(10.0, abs=1e-4)
        assert sorted_evals[1] == pytest.approx(5.0, abs=1e-4)
        assert sorted_evals[2] == pytest.approx(1.0, abs=1e-4)

    def test_single_component_extraction(self):
        """Extract only the top eigenvalue."""
        matrix = [[5.0, 2.0], [2.0, 1.0]]
        eigenvalues, eigenvectors = _power_iteration_eigendecomp(
            matrix, n_components=1, seed=42, max_iterations=200, tol=1e-10,
        )
        assert len(eigenvalues) == 1
        assert len(eigenvectors) == 1
        # Top eigenvalue of [[5,2],[2,1]] = 3 + sqrt(4+1) ~= 5.828
        # (exact: (6 + sqrt(20))/2 = 3 + sqrt(5) ~ 5.236)
        # eigenvalues of [[5,2],[2,1]]: trace=6, det=1. lambda = (6 +/- sqrt(36-4))/2 = (6+/-sqrt(32))/2
        # = (6 +/- 4*sqrt(2))/2 = 3 +/- 2*sqrt(2) ~ 5.828, 0.172
        assert eigenvalues[0] == pytest.approx(3.0 + 2.0 * math.sqrt(2.0), abs=1e-4)


# ── TestStandardization ──────────────────────────────────


class TestStandardization:
    """Test mean and std computation from LatentTransform.fit."""

    def test_mean_and_std_correctly_computed(self):
        """Check that the latent space stores correct mean and std."""
        # Data: x=[1, 3, 5], y=[2, 4, 6]
        # mean_x = 3, mean_y = 4
        # std_x = sqrt(((1-3)^2 + (3-3)^2 + (5-3)^2) / 2) = sqrt(4) = 2
        # std_y = sqrt(((2-4)^2 + (4-4)^2 + (6-4)^2) / 2) = sqrt(4) = 2
        snap = _make_snapshot_2d([1.0, 3.0, 5.0], [2.0, 4.0, 6.0])
        transform = LatentTransform()
        ls = transform.fit(snap, seed=42)

        assert ls.mean[0] == pytest.approx(3.0)
        assert ls.mean[1] == pytest.approx(4.0)
        assert ls.std[0] == pytest.approx(2.0)
        assert ls.std[1] == pytest.approx(2.0)

    def test_zero_variance_columns_get_std_one(self):
        """Columns with zero variance should get std=1.0 to avoid division by zero."""
        # All x values identical, y varies widely (std clearly != 1.0)
        snap = _make_snapshot_2d([5.0, 5.0, 5.0], [0.0, 10.0, 20.0])
        transform = LatentTransform()
        ls = transform.fit(snap, seed=42)

        # x has zero variance -> std should be 1.0 (fallback)
        assert ls.std[0] == pytest.approx(1.0)
        # y has nonzero variance: std = sqrt(((0-10)^2 + (10-10)^2 + (20-10)^2)/2) = 10.0
        assert ls.std[1] == pytest.approx(10.0)
        assert ls.std[1] != 1.0  # Should be actual std, not the fallback

    def test_all_zero_variance_columns(self):
        """When all columns are constant, all stds should be 1.0."""
        snap = _make_snapshot_2d([3.0, 3.0, 3.0], [7.0, 7.0, 7.0])
        transform = LatentTransform()
        ls = transform.fit(snap, seed=42)

        assert ls.std[0] == pytest.approx(1.0)
        assert ls.std[1] == pytest.approx(1.0)


# ── TestExplainedVariance ────────────────────────────────


class TestExplainedVariance:
    """Test explained variance ratio computation."""

    def test_data_along_one_axis_first_component_dominates(self):
        """2D data with all variance along x-axis -> first PC explains ~100%."""
        # y is constant, x varies widely
        snap = _make_snapshot_2d(
            [1.0, 10.0, 20.0, 30.0, 40.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
        )
        transform = LatentTransform(min_variance_explained=0.5)
        ls = transform.fit(snap, seed=42)

        # First component should explain nearly all variance.
        # Since y is constant (std=1.0 fallback), standardised y is all 0.
        # All variance is in x.
        assert ls.explained_variance_ratio[0] > 0.95
        assert ls.total_explained_variance > 0.95

    def test_equal_variance_components_roughly_equal(self):
        """When both dimensions have equal, independent variance, each ~50%."""
        # Create data where x and y are independently, equally distributed
        # Use enough points and equal variance along both axes
        xs = [0.0, 10.0, 0.0, 10.0, 5.0]
        ys = [0.0, 0.0, 10.0, 10.0, 5.0]
        snap = _make_snapshot_2d(xs, ys)
        transform = LatentTransform(min_variance_explained=0.99)
        ls = transform.fit(snap, seed=42)

        # Both components should explain roughly equal variance
        assert len(ls.explained_variance_ratio) == 2
        for ratio in ls.explained_variance_ratio:
            assert ratio == pytest.approx(0.5, abs=0.15)

    def test_variance_ratios_sum_to_total(self):
        """Individual explained ratios must sum to total_explained_variance."""
        snap = _make_snapshot_2d([1.0, 2.0, 3.0, 4.0], [5.0, 3.0, 8.0, 1.0])
        transform = LatentTransform(min_variance_explained=0.99)
        ls = transform.fit(snap, seed=42)

        assert sum(ls.explained_variance_ratio) == pytest.approx(
            ls.total_explained_variance, abs=1e-10,
        )

    def test_ratios_are_between_zero_and_one(self):
        """Each explained variance ratio should be in [0, 1]."""
        snap = _make_snapshot_2d([1.0, 5.0, 9.0, 2.0], [3.0, 7.0, 2.0, 8.0])
        transform = LatentTransform(min_variance_explained=0.99)
        ls = transform.fit(snap, seed=42)

        for ratio in ls.explained_variance_ratio:
            assert 0.0 <= ratio <= 1.0 + 1e-10


# ── TestDeterminism ──────────────────────────────────────


class TestDeterminism:
    """Same seed -> same eigenvalues and eigenvectors."""

    def test_same_seed_same_result(self):
        """Fitting PCA with the same seed produces identical LatentSpace."""
        snap = _make_snapshot_2d([1.0, 4.0, 7.0, 2.0, 9.0], [3.0, 6.0, 1.0, 8.0, 5.0])
        transform = LatentTransform()

        ls1 = transform.fit(snap, seed=123)
        ls2 = transform.fit(snap, seed=123)

        assert ls1.eigenvalues == ls2.eigenvalues
        assert ls1.components == ls2.components
        assert ls1.mean == ls2.mean
        assert ls1.std == ls2.std
        assert ls1.n_components == ls2.n_components
        assert ls1.explained_variance_ratio == ls2.explained_variance_ratio

    def test_different_seeds_can_differ(self):
        """Different seeds may produce different eigenvectors (sign/direction)."""
        snap = _make_snapshot_2d([1.0, 4.0, 7.0, 2.0, 9.0], [3.0, 6.0, 1.0, 8.0, 5.0])
        transform = LatentTransform()

        ls1 = transform.fit(snap, seed=42)
        ls2 = transform.fit(snap, seed=999)

        # Eigenvalues should be the same (they are properties of the matrix),
        # but eigenvectors might differ in sign. At minimum, the eigenvalues
        # should be close regardless of seed.
        for ev1, ev2 in zip(ls1.eigenvalues, ls2.eigenvalues):
            assert ev1 == pytest.approx(ev2, abs=1e-4)

    def test_power_iteration_deterministic(self):
        """Direct call to _power_iteration_eigendecomp is deterministic."""
        matrix = [[3.0, 1.0], [1.0, 2.0]]
        ev1, vec1 = _power_iteration_eigendecomp(matrix, 2, seed=7, max_iterations=200, tol=1e-10)
        ev2, vec2 = _power_iteration_eigendecomp(matrix, 2, seed=7, max_iterations=200, tol=1e-10)

        assert ev1 == ev2
        for v1_row, v2_row in zip(vec1, vec2):
            for a, b in zip(v1_row, v2_row):
                assert a == pytest.approx(b, abs=1e-12)


# ── TestBuildDataMatrix ──────────────────────────────────


class TestBuildDataMatrix:
    """Test the _build_data_matrix helper."""

    def test_basic_extraction(self):
        """Observations are turned into a correctly shaped numeric matrix."""
        obs = [
            Observation(iteration=0, parameters={"a": 1.0, "b": 2.0}, kpi_values={"obj": 0}),
            Observation(iteration=1, parameters={"a": 3.0, "b": 4.0}, kpi_values={"obj": 1}),
        ]
        mat = _build_data_matrix(obs, param_names=["a", "b"])
        assert mat == [[1.0, 2.0], [3.0, 4.0]]

    def test_column_order_follows_param_names(self):
        """Matrix columns should follow the param_names ordering, not dict order."""
        obs = [
            Observation(iteration=0, parameters={"b": 10.0, "a": 20.0}, kpi_values={"obj": 0}),
        ]
        mat = _build_data_matrix(obs, param_names=["a", "b"])
        assert mat == [[20.0, 10.0]]

    def test_missing_param_defaults_to_zero(self):
        """If a parameter is missing from an observation, it defaults to 0.0."""
        obs = [
            Observation(iteration=0, parameters={"a": 5.0}, kpi_values={"obj": 0}),
        ]
        mat = _build_data_matrix(obs, param_names=["a", "b"])
        assert mat == [[5.0, 0.0]]

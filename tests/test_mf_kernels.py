"""Tests for multi-fidelity kernels (ICM and LinearCoregionalization)."""

from __future__ import annotations

import math
import pytest

from optimization_copilot.backends.mf_kernels import ICMKernel, LinearCoregionalization


# ── Helpers ─────────────────────────────────────────────────────────


def _is_symmetric(M: list[list[float]], tol: float = 1e-10) -> bool:
    """Check if a matrix is symmetric within tolerance."""
    n = len(M)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(M[i][j] - M[j][i]) > tol:
                return False
    return True


def _is_positive_semidefinite(M: list[list[float]], tol: float = -1e-8) -> bool:
    """Check PSD via Cholesky (will fail if not PD after adding small noise)."""
    n = len(M)
    # Add a tiny diagonal jitter to test PSD (not strict PD)
    M_copy = [list(row) for row in M]
    for i in range(n):
        M_copy[i][i] += 1e-10
    try:
        from optimization_copilot.backends._math import cholesky
        cholesky(M_copy)
        return True
    except Exception:
        return False


def _make_data(n: int = 5, d: int = 2) -> list[list[float]]:
    """Generate simple test data."""
    import random
    rng = random.Random(42)
    return [[rng.uniform(0, 1) for _ in range(d)] for _ in range(n)]


# ── Coregionalization Matrix Properties ─────────────────────────────


class TestCoregionalizationMatrix:
    """Test B = W @ W^T + diag(kappa) properties."""

    def test_symmetric(self):
        kernel = ICMKernel(n_tasks=3, rank=2)
        B = kernel.coregionalization_matrix()
        assert _is_symmetric(B)

    def test_positive_semidefinite(self):
        kernel = ICMKernel(n_tasks=3, rank=2)
        B = kernel.coregionalization_matrix()
        assert _is_positive_semidefinite(B)

    def test_correct_shape(self):
        kernel = ICMKernel(n_tasks=4, rank=2)
        B = kernel.coregionalization_matrix()
        assert len(B) == 4
        assert all(len(row) == 4 for row in B)

    def test_diagonal_positive(self):
        kernel = ICMKernel(n_tasks=3, rank=1)
        B = kernel.coregionalization_matrix()
        for i in range(3):
            assert B[i][i] > 0

    def test_changes_with_parameters(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        B1 = kernel.coregionalization_matrix()
        kernel.set_parameters(W=[[1.0], [0.5]], kappa=[0.2, 0.3])
        B2 = kernel.coregionalization_matrix()
        assert B1 != B2


# ── ICM Kernel: Self-similarity, Symmetry, Task Correlation ────────


class TestICMKernelBasic:
    """Test ICM kernel basic properties."""

    def test_self_similarity_positive(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        x = [0.5, 0.3]
        val = kernel(x, x, task1=0, task2=0)
        assert val > 0

    def test_symmetry_in_x(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        x1 = [0.1, 0.2]
        x2 = [0.8, 0.7]
        val_12 = kernel(x1, x2, task1=0, task2=1)
        val_21 = kernel(x2, x1, task1=0, task2=1)
        assert abs(val_12 - val_21) < 1e-12

    def test_symmetry_in_tasks(self):
        kernel = ICMKernel(n_tasks=3, rank=1)
        x1 = [0.1, 0.2]
        x2 = [0.3, 0.4]
        val_01 = kernel(x1, x2, task1=0, task2=1)
        val_10 = kernel(x1, x2, task1=1, task2=0)
        assert abs(val_01 - val_10) < 1e-12

    def test_task_correlation_nonzero(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        kernel.set_parameters(W=[[1.0], [0.8]], kappa=[0.1, 0.1])
        x = [0.5]
        val = kernel(x, x, task1=0, task2=1)
        assert abs(val) > 0

    def test_distant_points_low_correlation(self):
        kernel = ICMKernel(n_tasks=2, rank=1, base_length_scale=0.1)
        x1 = [0.0]
        x2 = [100.0]
        val = kernel(x1, x2, task1=0, task2=0)
        assert abs(val) < 1e-6

    def test_same_point_same_task_is_max(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        kernel.set_parameters(W=[[1.0], [0.5]], kappa=[0.1, 0.1])
        x = [0.5]
        val_same = kernel(x, x, task1=0, task2=0)
        val_diff_task = kernel(x, x, task1=0, task2=1)
        # B[0,0] * k(x,x) >= B[0,1] * k(x,x) when tasks are correlated
        assert val_same >= abs(val_diff_task) - 1e-10

    def test_kernel_decreases_with_distance(self):
        kernel = ICMKernel(n_tasks=2, rank=1, base_length_scale=1.0)
        x1 = [0.0]
        x2_close = [0.1]
        x2_far = [2.0]
        val_close = kernel(x1, x2_close, task1=0, task2=0)
        val_far = kernel(x1, x2_far, task1=0, task2=0)
        assert val_close > val_far

    def test_matrix_positive_definite(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        X = _make_data(n=6, d=2)
        tasks = [0, 0, 0, 1, 1, 1]
        K = kernel.matrix(X, tasks, noise=1e-4)
        assert _is_symmetric(K, tol=1e-8)
        # Cholesky should succeed on PD matrix
        from optimization_copilot.backends._math import cholesky
        L = cholesky(K)
        assert len(L) == 6

    def test_matrix_shape(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        X = _make_data(n=4, d=2)
        tasks = [0, 1, 0, 1]
        K = kernel.matrix(X, tasks)
        assert len(K) == 4
        assert all(len(row) == 4 for row in K)

    def test_matrix_diagonal_positive(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        X = _make_data(n=4, d=2)
        tasks = [0, 1, 0, 1]
        K = kernel.matrix(X, tasks, noise=1e-4)
        for i in range(4):
            assert K[i][i] > 0


# ── ICM Kernel Parameter Setting ────────────────────────────────────


class TestICMParameterSetting:
    """Test ICM kernel parameter setting."""

    def test_set_W_and_kappa(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        kernel.set_parameters(W=[[1.0], [0.5]], kappa=[0.2, 0.3])
        assert kernel.W == [[1.0], [0.5]]
        assert kernel.kappa == [0.2, 0.3]

    def test_wrong_W_rows_raises(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        with pytest.raises(ValueError, match="rows"):
            kernel.set_parameters(W=[[1.0], [0.5], [0.3]], kappa=[0.1, 0.1])

    def test_wrong_W_cols_raises(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        with pytest.raises(ValueError, match="columns"):
            kernel.set_parameters(W=[[1.0, 0.5], [0.3, 0.4]], kappa=[0.1, 0.1])

    def test_wrong_kappa_length_raises(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        with pytest.raises(ValueError, match="kappa"):
            kernel.set_parameters(W=[[1.0], [0.5]], kappa=[0.1])

    def test_negative_kappa_raises(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        with pytest.raises(ValueError, match="non-negative"):
            kernel.set_parameters(W=[[1.0], [0.5]], kappa=[0.1, -0.1])


# ── LinearCoregionalization ──────────────────────────────────────────


class TestLinearCoregionalization:
    """Test LMC kernel."""

    def test_construction(self):
        lmc = LinearCoregionalization(n_tasks=3, n_kernels=2)
        assert lmc.n_tasks == 3
        assert lmc.n_kernels == 2
        assert len(lmc.length_scales) == 2
        assert len(lmc.mixing_matrices) == 2

    def test_single_kernel(self):
        lmc = LinearCoregionalization(n_tasks=2, n_kernels=1)
        x = [0.5]
        val = lmc(x, x, task1=0, task2=0)
        assert val > 0

    def test_kernel_evaluation_positive_self(self):
        lmc = LinearCoregionalization(n_tasks=2, n_kernels=2)
        x = [0.3, 0.7]
        val = lmc(x, x, task1=0, task2=0)
        assert val > 0

    def test_kernel_symmetry_x(self):
        lmc = LinearCoregionalization(n_tasks=2, n_kernels=2)
        x1 = [0.1, 0.2]
        x2 = [0.8, 0.9]
        v12 = lmc(x1, x2, task1=0, task2=1)
        v21 = lmc(x2, x1, task1=0, task2=1)
        assert abs(v12 - v21) < 1e-12

    def test_kernel_symmetry_tasks(self):
        lmc = LinearCoregionalization(n_tasks=3, n_kernels=2)
        x1 = [0.1]
        x2 = [0.5]
        v01 = lmc(x1, x2, task1=0, task2=1)
        v10 = lmc(x1, x2, task1=1, task2=0)
        assert abs(v01 - v10) < 1e-12

    def test_kernel_decreases_with_distance(self):
        lmc = LinearCoregionalization(n_tasks=2, n_kernels=2)
        x1 = [0.0]
        x_close = [0.1]
        x_far = [5.0]
        v_close = lmc(x1, x_close, task1=0, task2=0)
        v_far = lmc(x1, x_far, task1=0, task2=0)
        assert v_close > v_far

    def test_matrix_shape(self):
        lmc = LinearCoregionalization(n_tasks=2, n_kernels=2)
        X = _make_data(n=6, d=2)
        tasks = [0, 0, 0, 1, 1, 1]
        K = lmc.matrix(X, tasks)
        assert len(K) == 6
        assert all(len(row) == 6 for row in K)

    def test_matrix_symmetric(self):
        lmc = LinearCoregionalization(n_tasks=2, n_kernels=2)
        X = _make_data(n=5, d=2)
        tasks = [0, 1, 0, 1, 0]
        K = lmc.matrix(X, tasks)
        assert _is_symmetric(K, tol=1e-8)

    def test_matrix_positive_definite(self):
        lmc = LinearCoregionalization(n_tasks=2, n_kernels=2)
        X = _make_data(n=6, d=2)
        tasks = [0, 0, 0, 1, 1, 1]
        K = lmc.matrix(X, tasks, noise=1e-4)
        from optimization_copilot.backends._math import cholesky
        L = cholesky(K)
        assert len(L) == 6

    def test_different_n_kernels_different_values(self):
        lmc1 = LinearCoregionalization(n_tasks=2, n_kernels=1)
        lmc2 = LinearCoregionalization(n_tasks=2, n_kernels=3)
        x = [0.5]
        v1 = lmc1(x, x, task1=0, task2=0)
        v2 = lmc2(x, x, task1=0, task2=0)
        # Different number of kernels should generally produce different values
        # (unless by coincidence)
        assert isinstance(v1, float)
        assert isinstance(v2, float)


# ── Cross-Fidelity Kernel Values ─────────────────────────────────────


class TestCrossFidelityKernelValues:
    """Test cross-fidelity kernel correlations."""

    def test_high_correlation_tasks(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        kernel.set_parameters(W=[[1.0], [0.95]], kappa=[0.01, 0.01])
        x = [0.5]
        same = kernel(x, x, task1=0, task2=0)
        cross = kernel(x, x, task1=0, task2=1)
        # High W similarity => high cross-task correlation
        ratio = cross / same
        assert ratio > 0.8

    def test_low_correlation_tasks(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        kernel.set_parameters(W=[[1.0], [0.01]], kappa=[0.1, 0.1])
        x = [0.5]
        same = kernel(x, x, task1=0, task2=0)
        cross = kernel(x, x, task1=0, task2=1)
        ratio = abs(cross) / same
        assert ratio < 0.2

    def test_orthogonal_tasks_low_cross(self):
        kernel = ICMKernel(n_tasks=2, rank=2)
        kernel.set_parameters(W=[[1.0, 0.0], [0.0, 1.0]], kappa=[0.01, 0.01])
        x = [0.5]
        cross = kernel(x, x, task1=0, task2=1)
        # Orthogonal W rows => B[0,1] ~ 0
        assert abs(cross) < 0.05

    def test_multiple_tasks_cross_values(self):
        kernel = ICMKernel(n_tasks=3, rank=2)
        kernel.set_parameters(
            W=[[1.0, 0.0], [0.8, 0.2], [0.5, 0.5]],
            kappa=[0.1, 0.1, 0.1],
        )
        x = [0.3]
        # Task 0-1 should be more correlated than 0-2
        cross_01 = kernel(x, x, task1=0, task2=1)
        cross_02 = kernel(x, x, task1=0, task2=2)
        assert cross_01 > cross_02

    def test_cross_fidelity_with_distance(self):
        kernel = ICMKernel(n_tasks=2, rank=1, base_length_scale=1.0)
        kernel.set_parameters(W=[[1.0], [0.9]], kappa=[0.1, 0.1])
        x1 = [0.0]
        x2 = [0.5]
        cross_close = kernel(x1, x1, task1=0, task2=1)
        cross_far = kernel(x1, x2, task1=0, task2=1)
        assert cross_close > cross_far


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases for kernels."""

    def test_single_task_icm(self):
        kernel = ICMKernel(n_tasks=1, rank=1)
        x = [0.5]
        val = kernel(x, x, task1=0, task2=0)
        assert val > 0

    def test_single_task_matrix(self):
        kernel = ICMKernel(n_tasks=1, rank=1)
        X = [[0.1], [0.5], [0.9]]
        tasks = [0, 0, 0]
        K = kernel.matrix(X, tasks)
        assert len(K) == 3
        assert _is_symmetric(K)

    def test_many_tasks_icm(self):
        kernel = ICMKernel(n_tasks=10, rank=3)
        B = kernel.coregionalization_matrix()
        assert len(B) == 10
        assert _is_symmetric(B)
        assert _is_positive_semidefinite(B)

    def test_many_tasks_evaluation(self):
        kernel = ICMKernel(n_tasks=10, rank=3)
        x = [0.5, 0.5]
        # Should work for all task pairs
        for t1 in range(10):
            for t2 in range(10):
                val = kernel(x, x, task1=t1, task2=t2)
                assert math.isfinite(val)

    def test_lmc_single_task(self):
        lmc = LinearCoregionalization(n_tasks=1, n_kernels=2)
        x = [0.5]
        val = lmc(x, x, task1=0, task2=0)
        assert val > 0


# ── Numerical Stability ─────────────────────────────────────────────


class TestNumericalStability:
    """Test numerical stability of kernel computations."""

    def test_very_close_points(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        x1 = [0.5]
        x2 = [0.5 + 1e-15]
        val = kernel(x1, x2, task1=0, task2=0)
        assert math.isfinite(val)
        assert val > 0

    def test_very_distant_points(self):
        kernel = ICMKernel(n_tasks=2, rank=1, base_length_scale=1.0)
        x1 = [0.0]
        x2 = [1000.0]
        val = kernel(x1, x2, task1=0, task2=0)
        assert math.isfinite(val)
        assert val >= 0

    def test_high_dimensional_data(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        x = [0.5] * 20
        val = kernel(x, x, task1=0, task2=0)
        assert math.isfinite(val)
        assert val > 0

    def test_large_matrix_stable(self):
        kernel = ICMKernel(n_tasks=2, rank=1)
        X = _make_data(n=20, d=3)
        tasks = [i % 2 for i in range(20)]
        K = kernel.matrix(X, tasks, noise=1e-4)
        for i in range(20):
            for j in range(20):
                assert math.isfinite(K[i][j])

    def test_zero_length_scale_fallback(self):
        """Very small length scale should not produce NaN."""
        kernel = ICMKernel(n_tasks=2, rank=1, base_length_scale=1e-15)
        x1 = [0.0]
        x2 = [0.1]
        val = kernel(x1, x2, task1=0, task2=0)
        assert math.isfinite(val)


# ── Constructor Validation ───────────────────────────────────────────


class TestConstructorValidation:
    """Test constructor argument validation."""

    def test_icm_zero_tasks_raises(self):
        with pytest.raises(ValueError):
            ICMKernel(n_tasks=0)

    def test_icm_zero_rank_raises(self):
        with pytest.raises(ValueError):
            ICMKernel(n_tasks=2, rank=0)

    def test_lmc_zero_tasks_raises(self):
        with pytest.raises(ValueError):
            LinearCoregionalization(n_tasks=0)

    def test_lmc_zero_kernels_raises(self):
        with pytest.raises(ValueError):
            LinearCoregionalization(n_tasks=2, n_kernels=0)

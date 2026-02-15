"""Comprehensive tests for the optimization_copilot.backends._math package.

Covers all five modules: linalg, stats, kernels, acquisition, and sobol.
"""

from __future__ import annotations

import math

import pytest

from optimization_copilot.backends._math import (
    # linalg
    cholesky,
    determinant,
    identity,
    mat_add,
    mat_inv,
    mat_mul,
    mat_scale,
    mat_vec,
    outer_product,
    solve_cholesky,
    solve_lower,
    solve_upper,
    transpose,
    vec_dot,
    # stats
    binary_entropy,
    norm_cdf,
    norm_logpdf,
    norm_pdf,
    norm_ppf,
    # kernels
    distance_matrix,
    kernel_matrix,
    matern52_kernel,
    rbf_kernel,
    # acquisition
    expected_improvement,
    log_expected_improvement_per_cost,
    probability_of_improvement,
    upper_confidence_bound,
    # sobol
    SOBOL_DIRECTION_NUMBERS,
    sobol_sequence,
)


# ============================================================================
# 1. linalg.py tests
# ============================================================================


class TestVecDot:
    """Tests for vec_dot."""

    def test_known_dot_product(self) -> None:
        """Dot product of [1,2,3] and [4,5,6] equals 32."""
        assert vec_dot([1, 2, 3], [4, 5, 6]) == pytest.approx(32.0)

    def test_orthogonal_vectors(self) -> None:
        """Dot product of orthogonal vectors is zero."""
        assert vec_dot([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_self_dot_is_squared_norm(self) -> None:
        """Dot product of a vector with itself equals its squared norm."""
        v = [3.0, 4.0]
        assert vec_dot(v, v) == pytest.approx(25.0)


class TestMatMul:
    """Tests for mat_mul."""

    def test_identity_multiplication(self) -> None:
        """Multiplying by identity returns the same matrix."""
        A = [[1.0, 2.0], [3.0, 4.0]]
        I = identity(2)
        result = mat_mul(A, I)
        for i in range(2):
            for j in range(2):
                assert result[i][j] == pytest.approx(A[i][j])

    def test_known_2x2_product(self) -> None:
        """Known 2x2 matrix product."""
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        result = mat_mul(A, B)
        assert result[0][0] == pytest.approx(19.0)
        assert result[0][1] == pytest.approx(22.0)
        assert result[1][0] == pytest.approx(43.0)
        assert result[1][1] == pytest.approx(50.0)


class TestMatVec:
    """Tests for mat_vec."""

    def test_identity_times_vector(self) -> None:
        """Identity matrix times a vector returns the same vector."""
        v = [3.0, 7.0, 11.0]
        I = identity(3)
        result = mat_vec(I, v)
        for i in range(3):
            assert result[i] == pytest.approx(v[i])

    def test_known_product(self) -> None:
        """Known matrix-vector product."""
        A = [[1.0, 2.0], [3.0, 4.0]]
        v = [5.0, 6.0]
        result = mat_vec(A, v)
        assert result[0] == pytest.approx(17.0)
        assert result[1] == pytest.approx(39.0)


class TestTranspose:
    """Tests for transpose."""

    def test_double_transpose_is_identity(self) -> None:
        """Transpose of transpose gives back the original matrix."""
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = transpose(transpose(A))
        for i in range(len(A)):
            for j in range(len(A[0])):
                assert result[i][j] == pytest.approx(A[i][j])

    def test_shape_change(self) -> None:
        """Transpose swaps rows and columns."""
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        At = transpose(A)
        assert len(At) == 3
        assert len(At[0]) == 2

    def test_empty_matrix(self) -> None:
        """Transpose of empty matrix returns empty list."""
        assert transpose([]) == []


class TestIdentity:
    """Tests for identity."""

    def test_diagonal_pattern(self) -> None:
        """Identity has 1s on diagonal and 0s elsewhere."""
        I = identity(3)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert I[i][j] == pytest.approx(expected)


class TestMatAdd:
    """Tests for mat_add."""

    def test_elementwise_addition(self) -> None:
        """Element-wise addition of two matrices."""
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        result = mat_add(A, B)
        assert result[0][0] == pytest.approx(6.0)
        assert result[0][1] == pytest.approx(8.0)
        assert result[1][0] == pytest.approx(10.0)
        assert result[1][1] == pytest.approx(12.0)


class TestMatScale:
    """Tests for mat_scale."""

    def test_scalar_multiplication(self) -> None:
        """Multiply all elements by a scalar."""
        A = [[1.0, 2.0], [3.0, 4.0]]
        result = mat_scale(A, 3.0)
        assert result[0][0] == pytest.approx(3.0)
        assert result[0][1] == pytest.approx(6.0)
        assert result[1][0] == pytest.approx(9.0)
        assert result[1][1] == pytest.approx(12.0)


class TestOuterProduct:
    """Tests for outer_product."""

    def test_known_outer_product(self) -> None:
        """Outer product of [1,2] and [3,4,5]."""
        result = outer_product([1.0, 2.0], [3.0, 4.0, 5.0])
        assert result[0] == [pytest.approx(3.0), pytest.approx(4.0), pytest.approx(5.0)]
        assert result[1] == [pytest.approx(6.0), pytest.approx(8.0), pytest.approx(10.0)]


class TestCholesky:
    """Tests for cholesky."""

    def test_decomposition_reconstructs_original(self) -> None:
        """Verify L @ L^T approximately equals the original PD matrix."""
        A = [[4.0, 2.0], [2.0, 3.0]]
        L = cholesky(A)
        # Reconstruct A from L @ L^T
        Lt = transpose(L)
        reconstructed = mat_mul(L, Lt)
        for i in range(2):
            for j in range(2):
                assert reconstructed[i][j] == pytest.approx(A[i][j], abs=1e-10)

    def test_lower_triangular(self) -> None:
        """Cholesky factor is lower triangular."""
        A = [[4.0, 2.0], [2.0, 3.0]]
        L = cholesky(A)
        # Upper triangle (excluding diagonal) should be zero
        assert L[0][1] == pytest.approx(0.0)


class TestSolveLower:
    """Tests for solve_lower."""

    def test_known_system(self) -> None:
        """Solve L*x = b for a known lower triangular system."""
        L = [[2.0, 0.0], [1.0, 3.0]]
        b = [4.0, 7.0]
        x = solve_lower(L, b)
        # x[0] = 4/2 = 2, x[1] = (7 - 1*2)/3 = 5/3
        assert x[0] == pytest.approx(2.0)
        assert x[1] == pytest.approx(5.0 / 3.0)


class TestSolveUpper:
    """Tests for solve_upper."""

    def test_known_system(self) -> None:
        """Solve U*x = b for a known upper triangular system."""
        U = [[2.0, 1.0], [0.0, 3.0]]
        b = [5.0, 6.0]
        x = solve_upper(U, b)
        # x[1] = 6/3 = 2, x[0] = (5 - 1*2)/2 = 1.5
        assert x[1] == pytest.approx(2.0)
        assert x[0] == pytest.approx(1.5)


class TestSolveCholesky:
    """Tests for solve_cholesky."""

    def test_solve_pd_system(self) -> None:
        """Solve A*x = b for a positive-definite A using Cholesky."""
        A = [[4.0, 2.0], [2.0, 3.0]]
        b = [1.0, 2.0]
        L = cholesky(A)
        x = solve_cholesky(L, b)
        # Verify A*x = b
        Ax = mat_vec(A, x)
        for i in range(2):
            assert Ax[i] == pytest.approx(b[i], abs=1e-10)


class TestMatInv:
    """Tests for mat_inv."""

    def test_inverse_times_original_is_identity(self) -> None:
        """inv(A) * A should approximate the identity for a small PD matrix."""
        A = [[4.0, 2.0], [2.0, 3.0]]
        A_inv = mat_inv(A)
        product = mat_mul(A_inv, A)
        I = identity(2)
        for i in range(2):
            for j in range(2):
                assert product[i][j] == pytest.approx(I[i][j], abs=1e-10)


class TestDeterminant:
    """Tests for determinant."""

    def test_2x2_determinant(self) -> None:
        """Determinant of [[4,2],[2,3]] = 4*3 - 2*2 = 8."""
        A = [[4.0, 2.0], [2.0, 3.0]]
        assert determinant(A) == pytest.approx(8.0, abs=1e-10)

    def test_identity_determinant(self) -> None:
        """Determinant of identity matrix is 1."""
        I = identity(3)
        assert determinant(I) == pytest.approx(1.0, abs=1e-10)


# ============================================================================
# 2. stats.py tests
# ============================================================================


class TestNormPdf:
    """Tests for norm_pdf."""

    def test_at_zero(self) -> None:
        """norm_pdf(0) is approximately 1/sqrt(2*pi)."""
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert norm_pdf(0.0) == pytest.approx(expected, rel=1e-10)

    def test_symmetry(self) -> None:
        """norm_pdf is symmetric: pdf(x) == pdf(-x)."""
        assert norm_pdf(1.5) == pytest.approx(norm_pdf(-1.5))


class TestNormCdf:
    """Tests for norm_cdf."""

    def test_at_zero(self) -> None:
        """norm_cdf(0) is 0.5."""
        assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-10)

    def test_large_positive(self) -> None:
        """norm_cdf of a large positive value is close to 1."""
        assert norm_cdf(10.0) == pytest.approx(1.0, abs=1e-10)

    def test_large_negative(self) -> None:
        """norm_cdf of a large negative value is close to 0."""
        assert norm_cdf(-10.0) == pytest.approx(0.0, abs=1e-10)


class TestNormPpf:
    """Tests for norm_ppf."""

    def test_median(self) -> None:
        """norm_ppf(0.5) should be approximately 0."""
        assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-6)

    def test_roundtrip_with_cdf(self) -> None:
        """norm_ppf(norm_cdf(x)) should approximately equal x."""
        for x in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]:
            p = norm_cdf(x)
            assert norm_ppf(p) == pytest.approx(x, abs=1e-6)

    def test_raises_at_boundary_zero(self) -> None:
        """norm_ppf(0.0) should raise ValueError."""
        with pytest.raises(ValueError):
            norm_ppf(0.0)

    def test_raises_at_boundary_one(self) -> None:
        """norm_ppf(1.0) should raise ValueError."""
        with pytest.raises(ValueError):
            norm_ppf(1.0)


class TestNormLogPdf:
    """Tests for norm_logpdf."""

    def test_consistent_with_log_of_pdf(self) -> None:
        """norm_logpdf(x) should equal log(norm_pdf(x)) for moderate x."""
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            assert norm_logpdf(x) == pytest.approx(math.log(norm_pdf(x)), abs=1e-10)


class TestBinaryEntropy:
    """Tests for binary_entropy."""

    def test_maximum_at_half(self) -> None:
        """Binary entropy is maximized at p=0.5, giving H=1.0 bit."""
        assert binary_entropy(0.5) == pytest.approx(1.0)

    def test_zero_at_boundaries(self) -> None:
        """Binary entropy is 0 at p=0 and p=1."""
        assert binary_entropy(0.0) == pytest.approx(0.0)
        assert binary_entropy(1.0) == pytest.approx(0.0)

    def test_symmetry(self) -> None:
        """Binary entropy is symmetric: H(p) == H(1-p)."""
        assert binary_entropy(0.3) == pytest.approx(binary_entropy(0.7))


# ============================================================================
# 3. kernels.py tests
# ============================================================================


class TestRbfKernel:
    """Tests for rbf_kernel."""

    def test_self_similarity(self) -> None:
        """RBF kernel of a point with itself is 1."""
        x = [1.0, 2.0, 3.0]
        assert rbf_kernel(x, x) == pytest.approx(1.0)

    def test_different_points_less_than_one(self) -> None:
        """RBF kernel between distinct points is less than 1."""
        x = [0.0, 0.0]
        y = [1.0, 1.0]
        assert rbf_kernel(x, y) < 1.0

    def test_larger_length_scale_increases_similarity(self) -> None:
        """Larger length_scale gives higher kernel value for same distance."""
        x = [0.0]
        y = [1.0]
        k_small = rbf_kernel(x, y, length_scale=0.5)
        k_large = rbf_kernel(x, y, length_scale=2.0)
        assert k_large > k_small


class TestMatern52Kernel:
    """Tests for matern52_kernel."""

    def test_self_similarity(self) -> None:
        """Matern 5/2 kernel of a point with itself is 1."""
        x = [1.0, 2.0]
        assert matern52_kernel(x, x) == pytest.approx(1.0)

    def test_different_points_less_than_one(self) -> None:
        """Matern 5/2 kernel between distinct points is less than 1."""
        x = [0.0]
        y = [1.0]
        assert matern52_kernel(x, y) < 1.0

    def test_positive_value(self) -> None:
        """Matern 5/2 kernel value is always positive."""
        x = [0.0, 0.0]
        y = [10.0, 10.0]
        assert matern52_kernel(x, y) > 0.0


class TestDistanceMatrix:
    """Tests for distance_matrix."""

    def test_diagonal_is_zero(self) -> None:
        """Diagonal entries (self-distances) are zero."""
        X = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        D = distance_matrix(X)
        for i in range(len(X)):
            assert D[i][i] == pytest.approx(0.0)

    def test_symmetry(self) -> None:
        """Distance matrix is symmetric."""
        X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        D = distance_matrix(X)
        for i in range(len(X)):
            for j in range(len(X)):
                assert D[i][j] == pytest.approx(D[j][i])

    def test_known_distance(self) -> None:
        """Known squared Euclidean distance between two points."""
        X = [[0.0, 0.0], [3.0, 4.0]]
        D = distance_matrix(X)
        assert D[0][1] == pytest.approx(25.0)


class TestKernelMatrix:
    """Tests for kernel_matrix."""

    def test_symmetric(self) -> None:
        """Kernel matrix is symmetric."""
        X = [[0.0], [1.0], [2.0]]
        K = kernel_matrix(X, rbf_kernel)
        for i in range(3):
            for j in range(3):
                assert K[i][j] == pytest.approx(K[j][i])

    def test_positive_diagonal(self) -> None:
        """Diagonal entries are positive (kernel value + noise)."""
        X = [[0.0], [1.0]]
        K = kernel_matrix(X, rbf_kernel)
        for i in range(2):
            assert K[i][i] > 0.0

    def test_diagonal_includes_noise(self) -> None:
        """Diagonal entries include the noise term."""
        X = [[0.0]]
        noise = 0.01
        K = kernel_matrix(X, rbf_kernel, noise=noise)
        # rbf_kernel(x, x) = 1.0, so diagonal should be 1.0 + noise
        assert K[0][0] == pytest.approx(1.0 + noise)


# ============================================================================
# 4. acquisition.py tests
# ============================================================================


class TestExpectedImprovement:
    """Tests for expected_improvement."""

    def test_zero_sigma_returns_zero(self) -> None:
        """EI is zero when sigma is negligible."""
        assert expected_improvement(mu=0.5, sigma=0.0, best_y=1.0) == pytest.approx(0.0)

    def test_positive_ei_when_possible_improvement(self) -> None:
        """EI is positive when there is room for improvement with uncertainty."""
        ei = expected_improvement(mu=0.5, sigma=1.0, best_y=1.0)
        assert ei > 0.0

    def test_higher_sigma_increases_ei(self) -> None:
        """Higher sigma (more uncertainty) generally increases EI."""
        ei_low = expected_improvement(mu=2.0, sigma=0.1, best_y=1.0)
        ei_high = expected_improvement(mu=2.0, sigma=2.0, best_y=1.0)
        assert ei_high > ei_low


class TestUpperConfidenceBound:
    """Tests for upper_confidence_bound."""

    def test_formula(self) -> None:
        """UCB follows the formula mu - kappa * sigma."""
        mu, sigma, kappa = 5.0, 2.0, 3.0
        assert upper_confidence_bound(mu, sigma, kappa) == pytest.approx(
            mu - kappa * sigma
        )

    def test_default_kappa(self) -> None:
        """Default kappa is 2.0."""
        result = upper_confidence_bound(mu=5.0, sigma=1.0)
        assert result == pytest.approx(5.0 - 2.0 * 1.0)


class TestProbabilityOfImprovement:
    """Tests for probability_of_improvement."""

    def test_zero_sigma_returns_zero(self) -> None:
        """PI is zero when sigma is negligible."""
        assert probability_of_improvement(mu=0.5, sigma=0.0, best_y=1.0) == pytest.approx(
            0.0
        )

    def test_very_good_mean(self) -> None:
        """PI is high when mu is much lower than best_y (minimization)."""
        pi = probability_of_improvement(mu=-10.0, sigma=1.0, best_y=0.0)
        assert pi > 0.99

    def test_value_between_zero_and_one(self) -> None:
        """PI is always between 0 and 1."""
        pi = probability_of_improvement(mu=0.5, sigma=1.0, best_y=1.0)
        assert 0.0 <= pi <= 1.0


class TestLogExpectedImprovementPerCost:
    """Tests for log_expected_improvement_per_cost."""

    def test_positive_ei_and_cost(self) -> None:
        """Positive EI with positive cost gives a finite log value."""
        result = log_expected_improvement_per_cost([1.0, 2.0], [1.0, 0.5])
        assert result[0] == pytest.approx(math.log(1.0) - math.log(1.0))
        assert result[1] == pytest.approx(math.log(2.0) - math.log(0.5))

    def test_zero_ei_gives_neg_inf(self) -> None:
        """Zero or negative EI returns -inf."""
        result = log_expected_improvement_per_cost([0.0, -1.0], [1.0, 1.0])
        assert result[0] == float("-inf")
        assert result[1] == float("-inf")

    def test_zero_cost_gives_neg_inf(self) -> None:
        """Zero or negative cost returns -inf."""
        result = log_expected_improvement_per_cost([1.0], [0.0])
        assert result[0] == float("-inf")


# ============================================================================
# 5. sobol.py tests
# ============================================================================


class TestSobolSequence:
    """Tests for sobol_sequence."""

    def test_first_point_is_origin(self) -> None:
        """First Sobol point is the origin [0.0, 0.0]."""
        points = sobol_sequence(1, 2)
        assert points[0] == [pytest.approx(0.0), pytest.approx(0.0)]

    def test_all_values_in_unit_interval(self) -> None:
        """All generated values should be in [0, 1)."""
        points = sobol_sequence(16, 3)
        for pt in points:
            for val in pt:
                assert 0.0 <= val < 1.0

    def test_output_shape(self) -> None:
        """Output has the correct number of points and dimensions."""
        n_points, n_dims = 10, 5
        points = sobol_sequence(n_points, n_dims)
        assert len(points) == n_points
        for pt in points:
            assert len(pt) == n_dims

    def test_reasonable_uniformity(self) -> None:
        """Points should spread across the unit interval without obvious gaps.

        We check that the mean of each coordinate is roughly near 0.5
        for a sufficient number of points.
        """
        n_points = 64
        n_dims = 2
        points = sobol_sequence(n_points, n_dims)
        for d in range(n_dims):
            col_mean = sum(pt[d] for pt in points) / n_points
            assert 0.2 < col_mean < 0.8


class TestSobolDirectionNumbers:
    """Tests for SOBOL_DIRECTION_NUMBERS constant."""

    def test_has_20_entries(self) -> None:
        """Direction numbers table has 20 entries (for dims 1-20)."""
        assert len(SOBOL_DIRECTION_NUMBERS) == 20

    def test_each_entry_has_10_values(self) -> None:
        """Each direction number entry has 10 values."""
        for dn in SOBOL_DIRECTION_NUMBERS:
            assert len(dn) == 10

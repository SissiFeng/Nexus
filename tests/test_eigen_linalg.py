"""Tests for eigen_symmetric in backends/_math/linalg.py."""

from __future__ import annotations

import math

import pytest

from optimization_copilot.backends._math.linalg import eigen_symmetric


class TestEigenSymmetric:
    def test_identity_eigenvalues(self):
        """Identity matrix should have all eigenvalues = 1."""
        I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        vals, vecs = eigen_symmetric(I3, k=3, seed=0)
        for v in vals:
            assert v == pytest.approx(1.0, abs=1e-6)

    def test_diagonal_matrix(self):
        """Diagonal matrix: eigenvalues = diagonal entries (sorted desc)."""
        D = [[5, 0, 0], [0, 2, 0], [0, 0, 8]]
        vals, vecs = eigen_symmetric(D, k=3, seed=1)
        assert vals[0] == pytest.approx(8.0, abs=1e-4)
        assert vals[1] == pytest.approx(5.0, abs=1e-4)
        assert vals[2] == pytest.approx(2.0, abs=1e-4)

    def test_2x2_symmetric(self):
        """Known 2x2 symmetric matrix eigenvalues."""
        # [[2, 1], [1, 2]] has eigenvalues 3 and 1.
        M = [[2, 1], [1, 2]]
        vals, vecs = eigen_symmetric(M, k=2, seed=42)
        assert vals[0] == pytest.approx(3.0, abs=1e-4)
        assert vals[1] == pytest.approx(1.0, abs=1e-4)

    def test_eigenvector_orthogonality(self):
        """Eigenvectors of a symmetric matrix should be orthogonal."""
        M = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
        vals, vecs = eigen_symmetric(M, k=3, seed=7)
        for i in range(3):
            for j in range(i + 1, 3):
                dot = sum(vecs[i][k] * vecs[j][k] for k in range(3))
                assert abs(dot) < 0.05, f"Eigenvectors {i} and {j} not orthogonal: dot={dot}"

    def test_eigenvectors_unit_norm(self):
        """Each eigenvector should have unit norm."""
        M = [[3, 1], [1, 3]]
        vals, vecs = eigen_symmetric(M, k=2, seed=0)
        for vec in vecs:
            norm = math.sqrt(sum(x * x for x in vec))
            assert norm == pytest.approx(1.0, abs=1e-6)

    def test_k_less_than_d(self):
        """Request fewer eigenvalues than matrix dimension."""
        M = [[4, 2, 0], [2, 5, 1], [0, 1, 3]]
        vals, vecs = eigen_symmetric(M, k=2, seed=0)
        assert len(vals) == 2
        assert len(vecs) == 2
        # Top eigenvalue should be the largest.
        all_vals, _ = eigen_symmetric(M, k=3, seed=0)
        assert vals[0] == pytest.approx(all_vals[0], abs=1e-4)

    def test_k_none_extracts_all(self):
        """k=None should extract all d eigenvalues."""
        M = [[2, 0], [0, 3]]
        vals, vecs = eigen_symmetric(M, k=None, seed=0)
        assert len(vals) == 2

    def test_deterministic(self):
        """Same seed produces identical results."""
        M = [[5, 2, 1], [2, 4, 0], [1, 0, 3]]
        r1 = eigen_symmetric(M, k=3, seed=99)
        r2 = eigen_symmetric(M, k=3, seed=99)
        assert r1[0] == r2[0]
        assert r1[1] == r2[1]

    def test_eigenvalues_non_negative(self):
        """Eigenvalues should be clamped to >= 0."""
        # Near-singular matrix.
        M = [[1, 1], [1, 1]]  # eigenvalues: 2, 0
        vals, _ = eigen_symmetric(M, k=2, seed=0)
        for v in vals:
            assert v >= 0.0

    def test_does_not_mutate_input(self):
        """Input matrix should not be modified by deflation."""
        M = [[3, 1], [1, 3]]
        original = [list(row) for row in M]
        eigen_symmetric(M, k=2, seed=0)
        assert M == original

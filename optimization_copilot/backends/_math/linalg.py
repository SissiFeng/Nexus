"""Pure-Python linear algebra helpers (no external dependencies).

All functions operate on plain Python lists so the optimization backends
can remain dependency-free.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

def vec_dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors.

    Parameters
    ----------
    a, b : list[float]
        Vectors of equal length.

    Returns
    -------
    float
        The scalar dot product ``sum(a_i * b_i)``.
    """
    return sum(ai * bi for ai, bi in zip(a, b))


# ---------------------------------------------------------------------------
# Matrix operations
# ---------------------------------------------------------------------------

def mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply two matrices (lists of rows).

    Parameters
    ----------
    A : list[list[float]]
        Matrix of shape (n, k).
    B : list[list[float]]
        Matrix of shape (k, m).

    Returns
    -------
    list[list[float]]
        Result matrix of shape (n, m).
    """
    n = len(A)
    m = len(B[0])
    k = len(B)
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C


def mat_vec(A: list[list[float]], v: list[float]) -> list[float]:
    """Matrix-vector product.

    Parameters
    ----------
    A : list[list[float]]
        Matrix of shape (n, m).
    v : list[float]
        Vector of length m.

    Returns
    -------
    list[float]
        Result vector of length n.
    """
    return [vec_dot(row, v) for row in A]


def transpose(A: list[list[float]]) -> list[list[float]]:
    """Transpose a matrix.

    Parameters
    ----------
    A : list[list[float]]
        Matrix of shape (n, m).

    Returns
    -------
    list[list[float]]
        Transposed matrix of shape (m, n).
    """
    if not A:
        return []
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def identity(n: int) -> list[list[float]]:
    """Return n x n identity matrix.

    Parameters
    ----------
    n : int
        Size of the identity matrix.

    Returns
    -------
    list[list[float]]
        The n x n identity matrix.
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Element-wise matrix addition.

    Parameters
    ----------
    A, B : list[list[float]]
        Matrices of equal shape (n, m).

    Returns
    -------
    list[list[float]]
        Result matrix with A[i][j] + B[i][j].
    """
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_scale(A: list[list[float]], scalar: float) -> list[list[float]]:
    """Multiply all elements of a matrix by a scalar.

    Parameters
    ----------
    A : list[list[float]]
        Input matrix.
    scalar : float
        Scale factor.

    Returns
    -------
    list[list[float]]
        Scaled matrix.
    """
    return [[A[i][j] * scalar for j in range(len(A[0]))] for i in range(len(A))]


def outer_product(a: list[float], b: list[float]) -> list[list[float]]:
    """Outer product of two vectors.

    Parameters
    ----------
    a : list[float]
        Vector of length n.
    b : list[float]
        Vector of length m.

    Returns
    -------
    list[list[float]]
        Matrix of shape (n, m) where result[i][j] = a[i] * b[j].
    """
    return [[ai * bj for bj in b] for ai in a]


# ---------------------------------------------------------------------------
# Cholesky decomposition and solvers
# ---------------------------------------------------------------------------

def cholesky(A: list[list[float]]) -> list[list[float]]:
    """Cholesky decomposition A = L L^T.  Returns lower triangular L.

    Parameters
    ----------
    A : list[list[float]]
        Symmetric positive-definite matrix.

    Returns
    -------
    list[list[float]]
        Lower triangular factor L such that A = L @ L^T.
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                L[i][j] = math.sqrt(max(val, 1e-12))
            else:
                L[i][j] = (A[i][j] - s) / max(L[j][j], 1e-12)
    return L


def solve_lower(L: list[list[float]], b: list[float]) -> list[float]:
    """Forward substitution: solve L x = b where L is lower triangular.

    Parameters
    ----------
    L : list[list[float]]
        Lower triangular matrix.
    b : list[float]
        Right-hand side vector.

    Returns
    -------
    list[float]
        Solution vector x.
    """
    n = len(b)
    x = [0.0] * n
    for i in range(n):
        x[i] = (b[i] - sum(L[i][j] * x[j] for j in range(i))) / max(L[i][i], 1e-12)
    return x


def solve_upper(U: list[list[float]], b: list[float]) -> list[float]:
    """Back substitution: solve U x = b where U is upper triangular.

    Parameters
    ----------
    U : list[list[float]]
        Upper triangular matrix.
    b : list[float]
        Right-hand side vector.

    Returns
    -------
    list[float]
        Solution vector x.
    """
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / max(U[i][i], 1e-12)
    return x


def solve_cholesky(L: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b given L where A = L L^T.

    Parameters
    ----------
    L : list[list[float]]
        Lower triangular Cholesky factor.
    b : list[float]
        Right-hand side vector.

    Returns
    -------
    list[float]
        Solution vector x.
    """
    y = solve_lower(L, b)
    # L^T is upper triangular
    n = len(L)
    Lt = [[L[j][i] for j in range(n)] for i in range(n)]
    return solve_upper(Lt, y)


def mat_inv(A: list[list[float]]) -> list[list[float]]:
    """Matrix inverse via Cholesky decomposition.

    Parameters
    ----------
    A : list[list[float]]
        Symmetric positive-definite matrix.

    Returns
    -------
    list[list[float]]
        The inverse A^{-1}.

    Notes
    -----
    Only valid for positive-definite matrices.  For non-PD matrices the
    result is undefined.
    """
    n = len(A)
    L = cholesky(A)
    inv = [[0.0] * n for _ in range(n)]
    for j in range(n):
        e_j = [1.0 if i == j else 0.0 for i in range(n)]
        col = solve_cholesky(L, e_j)
        for i in range(n):
            inv[i][j] = col[i]
    return inv


def determinant(A: list[list[float]]) -> float:
    """Determinant via Cholesky decomposition (for positive-definite matrices).

    Parameters
    ----------
    A : list[list[float]]
        Symmetric positive-definite matrix.

    Returns
    -------
    float
        The determinant det(A).

    Notes
    -----
    det(A) = det(L)^2 = (prod of diagonal of L)^2.
    Only valid for positive-definite matrices.
    """
    L = cholesky(A)
    log_det = 0.0
    for i in range(len(L)):
        log_det += math.log(max(L[i][i], 1e-300))
    return math.exp(2.0 * log_det)

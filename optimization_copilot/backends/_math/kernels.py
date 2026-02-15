"""Pure-Python kernel functions for Gaussian process models.

All functions operate on plain Python lists to avoid external dependencies.
"""

from __future__ import annotations

import math
from typing import Callable


def rbf_kernel(x1: list[float], x2: list[float], length_scale: float = 1.0) -> float:
    """RBF (squared exponential) kernel.

    Parameters
    ----------
    x1, x2 : list[float]
        Input vectors of equal length.
    length_scale : float
        Length-scale parameter (default 1.0).

    Returns
    -------
    float
        ``exp(-0.5 * ||x1 - x2||^2 / length_scale^2)``.
    """
    ls2 = length_scale ** 2
    sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
    return math.exp(-0.5 * sq_dist / max(ls2, 1e-12))


def matern52_kernel(x1: list[float], x2: list[float], length_scale: float = 1.0) -> float:
    """Matern 5/2 kernel.

    Parameters
    ----------
    x1, x2 : list[float]
        Input vectors of equal length.
    length_scale : float
        Length-scale parameter (default 1.0).

    Returns
    -------
    float
        The Matern 5/2 kernel value:
        ``(1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)``
        where ``r = ||x1 - x2||``.
    """
    sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
    r = math.sqrt(sq_dist)
    ls = max(length_scale, 1e-12)
    sqrt5_r_l = math.sqrt(5.0) * r / ls
    return (1.0 + sqrt5_r_l + 5.0 * sq_dist / (3.0 * ls * ls)) * math.exp(-sqrt5_r_l)


def distance_matrix(X: list[list[float]]) -> list[list[float]]:
    """Pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : list[list[float]]
        A list of n vectors, each of length d.

    Returns
    -------
    list[list[float]]
        An n x n matrix where entry (i, j) is ``||X[i] - X[j]||^2``.
    """
    n = len(X)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sq = sum((X[i][k] - X[j][k]) ** 2 for k in range(len(X[i])))
            D[i][j] = sq
            D[j][i] = sq
    return D


def kernel_matrix(
    X: list[list[float]],
    kernel_fn: Callable[[list[float], list[float]], float],
    noise: float = 1e-4,
) -> list[list[float]]:
    """Build a full kernel (Gram) matrix with optional diagonal noise.

    Parameters
    ----------
    X : list[list[float]]
        A list of n input vectors.
    kernel_fn : callable
        A kernel function ``(x1, x2) -> float``.
    noise : float
        Noise variance added to the diagonal (default 1e-4).

    Returns
    -------
    list[list[float]]
        An n x n positive-definite kernel matrix.
    """
    n = len(X)
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            k = kernel_fn(X[i], X[j])
            if i == j:
                k += noise
            K[i][j] = k
            K[j][i] = k
    return K

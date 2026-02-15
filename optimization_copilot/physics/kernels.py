"""Physics-informed kernel functions for Gaussian process models.

Provides periodic, linear, composite, and symmetry-aware kernels.
All functions operate on plain Python lists to avoid external dependencies.
"""

from __future__ import annotations

import math
from typing import Callable


def periodic_kernel(
    x1: list[float],
    x2: list[float],
    length_scale: float = 1.0,
    period: float = 1.0,
) -> float:
    """Periodic (exp-sine-squared) kernel.

    Parameters
    ----------
    x1, x2 : list[float]
        Input vectors of equal length.
    length_scale : float
        Length-scale parameter (default 1.0).
    period : float
        Period of the kernel (default 1.0).

    Returns
    -------
    float
        ``exp(-2 * sum(sin^2(pi * |x1_i - x2_i| / period)) / length_scale^2)``.
    """
    ls2 = max(length_scale, 1e-12) ** 2
    p = max(abs(period), 1e-12)
    sin_sq_sum = sum(
        math.sin(math.pi * abs(a - b) / p) ** 2
        for a, b in zip(x1, x2)
    )
    return math.exp(-2.0 * sin_sq_sum / ls2)


def linear_kernel(
    x1: list[float],
    x2: list[float],
    variance: float = 1.0,
    offset: float = 0.0,
) -> float:
    """Linear kernel.

    Parameters
    ----------
    x1, x2 : list[float]
        Input vectors of equal length.
    variance : float
        Variance scaling factor (default 1.0).
    offset : float
        Offset added to the dot product (default 0.0).

    Returns
    -------
    float
        ``variance * (x1 . x2 + offset)``.
    """
    dot = sum(a * b for a, b in zip(x1, x2))
    return variance * (dot + offset)


class PeriodicKernel:
    """Callable periodic kernel with stored hyperparameters.

    Parameters
    ----------
    length_scale : float
        Length-scale parameter.
    period : float
        Period of the kernel.
    """

    def __init__(self, length_scale: float = 1.0, period: float = 1.0) -> None:
        self.length_scale = length_scale
        self.period = period

    def __call__(self, x1: list[float], x2: list[float]) -> float:
        """Evaluate the periodic kernel."""
        return periodic_kernel(x1, x2, self.length_scale, self.period)

    def __repr__(self) -> str:
        return (
            f"PeriodicKernel(length_scale={self.length_scale}, "
            f"period={self.period})"
        )


class LinearKernel:
    """Callable linear kernel with stored hyperparameters.

    Parameters
    ----------
    variance : float
        Variance scaling factor.
    offset : float
        Offset added to the dot product.
    """

    def __init__(self, variance: float = 1.0, offset: float = 0.0) -> None:
        self.variance = variance
        self.offset = offset

    def __call__(self, x1: list[float], x2: list[float]) -> float:
        """Evaluate the linear kernel."""
        return linear_kernel(x1, x2, self.variance, self.offset)

    def __repr__(self) -> str:
        return (
            f"LinearKernel(variance={self.variance}, "
            f"offset={self.offset})"
        )


class CompositeKernel:
    """Sum or product of kernel functions.

    Parameters
    ----------
    kernels : list[Callable]
        List of callables with signature ``(x1, x2) -> float``.
    operation : str
        ``"sum"`` or ``"product"``.

    Raises
    ------
    ValueError
        If *operation* is not ``"sum"`` or ``"product"``.
    """

    def __init__(
        self,
        kernels: list[Callable[[list[float], list[float]], float]],
        operation: str = "sum",
    ) -> None:
        if operation not in ("sum", "product"):
            raise ValueError(f"operation must be 'sum' or 'product', got {operation!r}")
        self.kernels = list(kernels)
        self.operation = operation

    def __call__(self, x1: list[float], x2: list[float]) -> float:
        """Evaluate the composite kernel."""
        if self.operation == "sum":
            return sum(k(x1, x2) for k in self.kernels)
        else:
            result = 1.0
            for k in self.kernels:
                result *= k(x1, x2)
            return result

    def __repr__(self) -> str:
        return (
            f"CompositeKernel(n_kernels={len(self.kernels)}, "
            f"operation={self.operation!r})"
        )


def symmetry_kernel(
    x1: list[float],
    x2: list[float],
    base_kernel: Callable[[list[float], list[float]], float],
    symmetry_group: list[Callable[[list[float]], list[float]]],
) -> float:
    """Symmetry-averaged kernel.

    Averages the base kernel over all group transformations applied
    to the first argument:

        k_sym(x, x') = (1 / |G|) * sum_{g in G} base_kernel(g(x), x')

    Parameters
    ----------
    x1, x2 : list[float]
        Input vectors.
    base_kernel : callable
        A kernel function ``(x1, x2) -> float``.
    symmetry_group : list[callable]
        List of group element functions, each ``g: list[float] -> list[float]``.

    Returns
    -------
    float
        The symmetry-averaged kernel value.
    """
    if not symmetry_group:
        return base_kernel(x1, x2)
    n = len(symmetry_group)
    total = sum(base_kernel(g(x1), x2) for g in symmetry_group)
    return total / n


def periodic_kernel_matrix(
    X: list[list[float]],
    length_scale: float = 1.0,
    period: float = 1.0,
    noise: float = 1e-4,
) -> list[list[float]]:
    """Build a full periodic kernel (Gram) matrix with optional diagonal noise.

    Parameters
    ----------
    X : list[list[float]]
        A list of n input vectors.
    length_scale : float
        Length-scale parameter (default 1.0).
    period : float
        Period of the kernel (default 1.0).
    noise : float
        Noise variance added to the diagonal (default 1e-4).

    Returns
    -------
    list[list[float]]
        An n x n positive-definite kernel matrix.
    """
    n = len(X)
    K: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            k = periodic_kernel(X[i], X[j], length_scale, period)
            if i == j:
                k += noise
            K[i][j] = k
            K[j][i] = k
    return K


def linear_kernel_matrix(
    X: list[list[float]],
    variance: float = 1.0,
    offset: float = 0.0,
    noise: float = 1e-4,
) -> list[list[float]]:
    """Build a full linear kernel (Gram) matrix with optional diagonal noise.

    Parameters
    ----------
    X : list[list[float]]
        A list of n input vectors.
    variance : float
        Variance scaling factor (default 1.0).
    offset : float
        Offset added to the dot product (default 0.0).
    noise : float
        Noise variance added to the diagonal (default 1e-4).

    Returns
    -------
    list[list[float]]
        An n x n kernel matrix.
    """
    n = len(X)
    K: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            k = linear_kernel(X[i], X[j], variance, offset)
            if i == j:
                k += noise
            K[i][j] = k
            K[j][i] = k
    return K

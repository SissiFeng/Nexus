"""Multi-fidelity kernels for Gaussian process models.

Implements the Intrinsic Coregionalization Model (ICM) and Linear
Coregionalization kernels for modelling correlations across fidelity
levels.  All operations use pure-Python lists -- zero external
dependencies.
"""

from __future__ import annotations

import math
import random
from typing import Any

from optimization_copilot.backends._math import (
    rbf_kernel,
    kernel_matrix,
    mat_mul,
    transpose,
    identity,
    mat_add,
    mat_scale,
    cholesky,
    solve_cholesky,
    mat_inv,
    vec_dot,
)


class ICMKernel:
    """Intrinsic Coregionalization Model (ICM) kernel.

    Combines a base spatial (RBF) kernel with a task-coregionalization
    matrix ``B = W @ W^T + diag(kappa)`` to model correlations across
    multiple fidelity levels (tasks).

    Parameters
    ----------
    n_tasks : int
        Number of fidelity levels / tasks.
    rank : int
        Rank of the coregionalization mixing matrix ``W``.
    base_length_scale : float
        Length-scale for the base RBF kernel.
    """

    def __init__(
        self,
        n_tasks: int,
        rank: int = 1,
        base_length_scale: float = 1.0,
    ) -> None:
        if n_tasks < 1:
            raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")

        self.n_tasks = n_tasks
        self.rank = rank
        self.base_length_scale = base_length_scale

        # Initialise W with small random values (deterministic seed)
        rng = random.Random(42)
        self._W: list[list[float]] = [
            [rng.gauss(0.0, 0.5) for _ in range(rank)]
            for _ in range(n_tasks)
        ]
        # Per-task diagonal noise
        self._kappa: list[float] = [0.1] * n_tasks

    # -- public API ---------------------------------------------------------

    def coregionalization_matrix(self) -> list[list[float]]:
        """Compute the task coregionalization matrix B = W @ W^T + diag(kappa).

        Returns
        -------
        list[list[float]]
            Symmetric positive-semi-definite matrix of shape (n_tasks, n_tasks).
        """
        Wt = transpose(self._W)
        B = mat_mul(self._W, Wt)
        # Add diagonal kappa
        for i in range(self.n_tasks):
            B[i][i] += self._kappa[i]
        return B

    def __call__(
        self,
        x1: list[float],
        x2: list[float],
        task1: int,
        task2: int,
    ) -> float:
        """Evaluate the ICM kernel between two points at given task indices.

        Parameters
        ----------
        x1, x2 : list[float]
            Input feature vectors.
        task1, task2 : int
            Task (fidelity level) indices for each point.

        Returns
        -------
        float
            Kernel value ``B[task1, task2] * k_base(x1, x2)``.
        """
        B = self.coregionalization_matrix()
        k_base = rbf_kernel(x1, x2, length_scale=self.base_length_scale)
        return B[task1][task2] * k_base

    def matrix(
        self,
        X: list[list[float]],
        tasks: list[int],
        noise: float = 1e-6,
    ) -> list[list[float]]:
        """Build the full kernel (Gram) matrix for multi-task data.

        Parameters
        ----------
        X : list[list[float]]
            Input feature vectors, one per observation.
        tasks : list[int]
            Task index for each observation (same length as X).
        noise : float
            Diagonal noise variance added for numerical stability.

        Returns
        -------
        list[list[float]]
            Symmetric positive-definite kernel matrix of shape (n, n).
        """
        n = len(X)
        B = self.coregionalization_matrix()
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                k_base = rbf_kernel(X[i], X[j], length_scale=self.base_length_scale)
                val = B[tasks[i]][tasks[j]] * k_base
                if i == j:
                    val += noise
                K[i][j] = val
                K[j][i] = val
        return K

    def set_parameters(
        self,
        W: list[list[float]],
        kappa: list[float],
    ) -> None:
        """Set the coregionalization parameters directly.

        Parameters
        ----------
        W : list[list[float]]
            Mixing matrix of shape (n_tasks, rank).
        kappa : list[float]
            Per-task diagonal noise, length n_tasks.

        Raises
        ------
        ValueError
            If dimensions are inconsistent.
        """
        if len(W) != self.n_tasks:
            raise ValueError(
                f"W must have {self.n_tasks} rows, got {len(W)}"
            )
        if any(len(row) != self.rank for row in W):
            raise ValueError(f"Each row of W must have {self.rank} columns")
        if len(kappa) != self.n_tasks:
            raise ValueError(
                f"kappa must have length {self.n_tasks}, got {len(kappa)}"
            )
        if any(k < 0 for k in kappa):
            raise ValueError("All kappa values must be non-negative")
        self._W = [list(row) for row in W]
        self._kappa = list(kappa)

    @property
    def W(self) -> list[list[float]]:
        """Current mixing matrix W (n_tasks x rank)."""
        return [list(row) for row in self._W]

    @property
    def kappa(self) -> list[float]:
        """Current per-task diagonal noise vector."""
        return list(self._kappa)


class LinearCoregionalization:
    """Linear Model of Coregionalization (LMC).

    Combines multiple base kernels with per-kernel mixing weights to
    model richer cross-task correlations than a single ICM kernel.

    Parameters
    ----------
    n_tasks : int
        Number of fidelity levels / tasks.
    n_kernels : int
        Number of base kernels to combine.
    """

    def __init__(self, n_tasks: int, n_kernels: int = 2) -> None:
        if n_tasks < 1:
            raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
        if n_kernels < 1:
            raise ValueError(f"n_kernels must be >= 1, got {n_kernels}")

        self.n_tasks = n_tasks
        self.n_kernels = n_kernels

        # Each base kernel has its own length scale and mixing weights
        rng = random.Random(123)
        self._length_scales: list[float] = [
            0.5 + i * 0.5 for i in range(n_kernels)
        ]
        # Mixing weights: n_kernels x n_tasks x n_tasks (symmetric)
        self._mixing: list[list[list[float]]] = []
        for _ in range(n_kernels):
            # Generate a PSD mixing matrix via A @ A^T + small diag
            A = [
                [rng.gauss(0.0, 0.5) for _ in range(n_tasks)]
                for _ in range(n_tasks)
            ]
            At = transpose(A)
            M = mat_mul(A, At)
            for t in range(n_tasks):
                M[t][t] += 0.1
            self._mixing.append(M)

    def __call__(
        self,
        x1: list[float],
        x2: list[float],
        task1: int,
        task2: int,
    ) -> float:
        """Evaluate the LMC kernel between two points.

        Parameters
        ----------
        x1, x2 : list[float]
            Input feature vectors.
        task1, task2 : int
            Task (fidelity level) indices.

        Returns
        -------
        float
            Sum of ``mixing_q[task1, task2] * k_q(x1, x2)`` over base kernels q.
        """
        total = 0.0
        for q in range(self.n_kernels):
            k_base = rbf_kernel(x1, x2, length_scale=self._length_scales[q])
            total += self._mixing[q][task1][task2] * k_base
        return total

    def matrix(
        self,
        X: list[list[float]],
        tasks: list[int],
        noise: float = 1e-6,
    ) -> list[list[float]]:
        """Build the full kernel matrix for multi-task data.

        Parameters
        ----------
        X : list[list[float]]
            Input feature vectors.
        tasks : list[int]
            Task index per observation.
        noise : float
            Diagonal noise for numerical stability.

        Returns
        -------
        list[list[float]]
            Symmetric positive-definite kernel matrix.
        """
        n = len(X)
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                val = self(X[i], X[j], tasks[i], tasks[j])
                if i == j:
                    val += noise
                K[i][j] = val
                K[j][i] = val
        return K

    @property
    def length_scales(self) -> list[float]:
        """Length scales for each base kernel."""
        return list(self._length_scales)

    @property
    def mixing_matrices(self) -> list[list[list[float]]]:
        """Per-kernel mixing weight matrices."""
        return [
            [list(row) for row in M]
            for M in self._mixing
        ]

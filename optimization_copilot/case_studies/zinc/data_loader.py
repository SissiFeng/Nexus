"""Data loader for zinc electrodeposition additive formulation benchmark.

Generates synthetic realistic 7D additive formulation data for zinc
electrodeposition coulombic efficiency optimisation.  All data generation
uses the Python stdlib ``random`` module for reproducibility -- no external
dependencies.

Search Space
------------
7 continuous parameters ``additive_1`` .. ``additive_7``, each in [0, 1].
Sum constraint: ``sum(additives) <= 1.0`` (total additive loading).

Objective
---------
``coulombic_efficiency`` (maximize, unit "%", realistic range ~85-100%).

Data Generation
---------------
* Seeded ``random.Random(seed)`` for full reproducibility.
* True function: synthetic landscape with quadratic interaction terms,
  base efficiency ~90 %, known optimum CE = 98.5 %.
* Heteroscedastic noise: noise increases with distance from optimum.
"""

from __future__ import annotations

import math
import random
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PARAM_NAMES: list[str] = [f"additive_{i}" for i in range(1, 8)]
_N_PARAMS: int = len(_PARAM_NAMES)

# Optimal additive concentrations (normalised [0, 1])
_OPTIMAL_POINT: list[float] = [0.15, 0.25, 0.10, 0.08, 0.05, 0.03, 0.02]

# True function parameters
_BASE_EFFICIENCY: float = 90.0
_OPTIMAL_EFFICIENCY: float = 98.5

# Noise model parameters
_NOISE_NEAR_OPTIMUM: float = 0.5   # std dev near optimum (%)
_NOISE_FAR_FROM_OPTIMUM: float = 2.5  # std dev far from optimum (%)
_NOISE_DISTANCE_SCALE: float = 1.5  # distance at which noise is at maximum


# ---------------------------------------------------------------------------
# Quadratic interaction matrix (upper-triangular, symmetric effects)
# ---------------------------------------------------------------------------

def _build_interaction_matrix() -> list[list[float]]:
    """Build a 7x7 symmetric interaction matrix for additive pairs.

    Positive entries denote synergistic interactions, negative entries
    denote antagonistic (interference) interactions.
    """
    # Upper-triangular raw values (hand-tuned for realistic landscape)
    raw: list[list[float]] = [
        # a1     a2     a3     a4     a5     a6     a7
        [ 0.0,  3.5,  1.2, -0.8,  0.5,  0.3,  0.1],  # a1
        [ 0.0,  0.0,  2.8,  0.6, -1.0,  0.2,  0.4],  # a2
        [ 0.0,  0.0,  0.0,  1.5,  0.7,  0.1, -0.3],  # a3
        [ 0.0,  0.0,  0.0,  0.0,  1.2, -0.5,  0.2],  # a4
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.8,  0.6],  # a5
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.4],  # a6
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # a7
    ]
    # Symmetrise
    n = len(raw)
    for i in range(n):
        for j in range(i + 1, n):
            raw[j][i] = raw[i][j]
    return raw


_INTERACTION_MATRIX: list[list[float]] = _build_interaction_matrix()


# ---------------------------------------------------------------------------
# ZincDataLoader
# ---------------------------------------------------------------------------


class ZincDataLoader:
    """Generates synthetic zinc electrodeposition additive formulation data.

    Parameters
    ----------
    n_points : int
        Number of data points to generate.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_points: int = 100, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._n_points = n_points
        self._data = self._generate()

    # -- public API --------------------------------------------------------

    def get_data(self) -> dict[str, Any]:
        """Return generated data.

        Returns
        -------
        dict
            ``{"X": list[list[float]],
              "Y": {"coulombic_efficiency": list[float]},
              "noise_levels": {"coulombic_efficiency": float}}``
        """
        return self._data

    def get_known_optimum(self) -> dict[str, float]:
        """Return the known optimum objective value.

        Returns
        -------
        dict[str, float]
            ``{"coulombic_efficiency": 98.5}``
        """
        return {"coulombic_efficiency": _OPTIMAL_EFFICIENCY}

    def get_search_space(self) -> dict[str, dict[str, Any]]:
        """Return the search space definition.

        Returns
        -------
        dict[str, dict]
            7 continuous parameters, each in [0, 1].
        """
        return {
            name: {"type": "continuous", "range": [0.0, 1.0]}
            for name in _PARAM_NAMES
        }

    # -- true function & noise model (public for testing) ------------------

    def _true_function(self, x: list[float]) -> float:
        """Synthetic coulombic efficiency landscape.

        Combines:
        * Base efficiency (~90 %).
        * Linear contribution from each additive (towards optimal).
        * Pairwise quadratic interaction terms.
        * Penalty for high total additive loading.

        The function is calibrated so that ``_true_function(_OPTIMAL_POINT)``
        returns approximately ``_OPTIMAL_EFFICIENCY``.
        """
        # Distance from optimum per dimension
        delta = [x[i] - _OPTIMAL_POINT[i] for i in range(_N_PARAMS)]

        # Quadratic distance term (negative contribution)
        quad_dist = sum(d * d for d in delta)

        # Pairwise interaction term
        interaction = 0.0
        for i in range(_N_PARAMS):
            for j in range(i + 1, _N_PARAMS):
                interaction += (
                    _INTERACTION_MATRIX[i][j]
                    * x[i]
                    * x[j]
                )

        # Total additive loading penalty
        total_loading = sum(x)
        loading_penalty = 0.0
        if total_loading > 0.7:
            loading_penalty = 5.0 * (total_loading - 0.7) ** 2

        # Combine
        # Scale quadratic distance so that at optimum, value = _OPTIMAL_EFFICIENCY
        # and at corners, value ~ 85-88 %
        efficiency = (
            _BASE_EFFICIENCY
            + 12.0 * math.exp(-8.0 * quad_dist)
            + 1.5 * interaction
            - loading_penalty
        )

        # Clamp to physically reasonable range
        return max(70.0, min(100.0, efficiency))

    def _noise_model(self, x: list[float]) -> float:
        """Heteroscedastic noise standard deviation.

        Noise increases with distance from the known optimum.

        Returns
        -------
        float
            Positive noise standard deviation (%).
        """
        delta = [x[i] - _OPTIMAL_POINT[i] for i in range(_N_PARAMS)]
        dist = math.sqrt(sum(d * d for d in delta))

        # Interpolate between near-optimum and far-from-optimum noise
        t = min(dist / _NOISE_DISTANCE_SCALE, 1.0)
        noise_std = _NOISE_NEAR_OPTIMUM + t * (
            _NOISE_FAR_FROM_OPTIMUM - _NOISE_NEAR_OPTIMUM
        )
        return noise_std

    # -- internal ----------------------------------------------------------

    def _generate(self) -> dict[str, Any]:
        """Generate synthetic data points.

        Points are sampled uniformly in [0, 1]^7 and rejected if they
        violate the sum constraint ``sum(additives) <= 1.0``.
        """
        X: list[list[float]] = []
        Y_ce: list[float] = []
        noise_sum = 0.0
        attempts = 0
        max_attempts = self._n_points * 20

        while len(X) < self._n_points and attempts < max_attempts:
            attempts += 1
            point = [self._rng.random() for _ in range(_N_PARAMS)]

            # Enforce sum constraint
            if sum(point) > 1.0:
                # Rescale to satisfy constraint (keeps relative proportions)
                s = sum(point)
                scale = self._rng.uniform(0.5, 0.98) / s
                point = [v * scale for v in point]

            true_val = self._true_function(point)
            noise_std = self._noise_model(point)
            noise = self._rng.gauss(0.0, noise_std)
            observed = true_val + noise

            X.append(point)
            Y_ce.append(observed)
            noise_sum += noise_std

        # Average noise level across all generated points
        avg_noise = noise_sum / len(X) if X else 1.0

        return {
            "X": X,
            "Y": {"coulombic_efficiency": Y_ce},
            "noise_levels": {"coulombic_efficiency": avg_noise},
        }

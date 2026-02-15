"""Zinc electrodeposition benchmark for offline replay evaluation.

Wraps :class:`ZincDataLoader` data in a :class:`ReplayBenchmark` so it can
be used by the ``CaseStudyEvaluator`` and optimisation algorithms.

The benchmark evaluates a 7-dimensional additive formulation space for
zinc electrodeposition coulombic efficiency, with:

* 7 continuous parameters (``additive_1`` .. ``additive_7``) in [0, 1].
* Sum constraint: ``sum(additives) <= 1.0``.
* Single objective: ``coulombic_efficiency`` (maximize, unit "%").
* Known optimum: CE = 98.5 %.
* Cost model: ``base 1.0 + 0.1 * sum(additives)``.
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.case_studies.base import ReplayBenchmark
from optimization_copilot.case_studies.zinc.data_loader import (
    ZincDataLoader,
    _PARAM_NAMES,
    _OPTIMAL_EFFICIENCY,
    _OPTIMAL_POINT,
    _NOISE_NEAR_OPTIMUM,
    _NOISE_FAR_FROM_OPTIMUM,
    _NOISE_DISTANCE_SCALE,
)


class ZincBenchmark(ReplayBenchmark):
    """Offline replay benchmark for zinc electrodeposition additive optimisation.

    Parameters
    ----------
    n_train : int
        Number of training data points for the GP surrogate.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_train: int = 100, seed: int = 42) -> None:
        # Store params before super().__init__ which calls _initialize -> _generate_data
        self._loader_n_train = n_train
        self._loader_seed = seed
        super().__init__(
            domain_name="electrochemistry",
            n_train=n_train,
            seed=seed,
        )

    # -- ReplayBenchmark interface -----------------------------------------

    def _generate_data(self) -> dict[str, Any]:
        """Generate training data using :class:`ZincDataLoader`.

        Returns
        -------
        dict
            ``{"X": ..., "Y": ..., "noise_levels": ...}``
        """
        loader = ZincDataLoader(
            n_points=self._loader_n_train,
            seed=self._loader_seed,
        )
        return loader.get_data()

    def get_search_space(self) -> dict[str, dict[str, Any]]:
        """Return 7 continuous parameters, each in [0, 1].

        Returns
        -------
        dict[str, dict]
            Parameter definitions keyed by name.
        """
        return {
            name: {"type": "continuous", "range": [0.0, 1.0]}
            for name in _PARAM_NAMES
        }

    def get_objectives(self) -> dict[str, dict[str, Any]]:
        """Return objective definition.

        Returns
        -------
        dict[str, dict]
            ``{"coulombic_efficiency": {"direction": "maximize", "unit": "%"}}``
        """
        return {
            "coulombic_efficiency": {
                "direction": "maximize",
                "unit": "%",
            }
        }

    def is_feasible(self, x: dict) -> bool:
        """Check the sum constraint: ``sum(additives) <= 1.0``.

        Parameters
        ----------
        x : dict
            Point in the parameter space.

        Returns
        -------
        bool
            ``True`` if the total additive loading does not exceed 1.0.
        """
        total = sum(float(x.get(name, 0.0)) for name in _PARAM_NAMES)
        return total <= 1.0

    def get_known_optimum(self) -> dict[str, float]:
        """Return known best objective value.

        Returns
        -------
        dict[str, float]
            ``{"coulombic_efficiency": 98.5}``
        """
        return {"coulombic_efficiency": _OPTIMAL_EFFICIENCY}

    def get_evaluation_cost(self, x: dict) -> float:
        """Evaluation cost model.

        Cost scales linearly with total additive loading:
        ``base 1.0 + 0.1 * sum(additives)``.

        Parameters
        ----------
        x : dict
            Point in the parameter space.

        Returns
        -------
        float
            Normalised cost (>= 1.0).
        """
        total = sum(float(x.get(name, 0.0)) for name in _PARAM_NAMES)
        return 1.0 + 0.1 * total

    def _noise_at_point(
        self, obj_name: str, x_encoded: list[float]
    ) -> float:
        """Heteroscedastic noise variance from the zinc noise model.

        Noise increases with distance from the known optimum, matching
        the data generation model in :class:`ZincDataLoader`.

        Returns variance (std**2), not standard deviation.
        """
        n = min(len(x_encoded), len(_OPTIMAL_POINT))
        delta = [x_encoded[i] - _OPTIMAL_POINT[i] for i in range(n)]
        dist = math.sqrt(sum(d * d for d in delta))
        t = min(dist / _NOISE_DISTANCE_SCALE, 1.0)
        noise_std = _NOISE_NEAR_OPTIMUM + t * (
            _NOISE_FAR_FROM_OPTIMUM - _NOISE_NEAR_OPTIMUM
        )
        return noise_std ** 2

    def get_known_constraints(self) -> list[dict[str, Any]]:
        """Return known constraints for the zinc electrodeposition domain.

        Returns
        -------
        list[dict]
            List of constraint descriptions.
        """
        base_constraints = super().get_known_constraints()
        zinc_constraints: list[dict[str, Any]] = [
            {
                "type": "sum_constraint",
                "parameters": list(_PARAM_NAMES),
                "operator": "<=",
                "value": 1.0,
                "description": "Total additive loading must not exceed 1.0",
            },
        ]
        return base_constraints + zinc_constraints

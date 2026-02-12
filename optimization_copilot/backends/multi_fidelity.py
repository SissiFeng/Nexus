"""Multi-fidelity Bayesian optimization backend.

Implements a full multi-fidelity GP-based optimizer that uses an
Intrinsic Coregionalization Model (ICM) kernel to jointly model
observations across fidelity levels and cost-aware acquisition
functions to decide which fidelity level to evaluate next.

Pure Python standard library only -- zero external dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.plugins.base import AlgorithmPlugin
from optimization_copilot.core.models import Observation, ParameterSpec, VariableType
from optimization_copilot.fidelity.config import FidelityConfig, FidelityLevel, CostModel
from optimization_copilot.backends.mf_kernels import ICMKernel
from optimization_copilot.backends.mf_acquisition import cost_aware_ei, fidelity_weighted_ei
from optimization_copilot.backends._math import (
    cholesky,
    solve_cholesky,
    solve_lower,
    kernel_matrix,
    mat_mul,
    transpose,
    identity,
    mat_add,
    mat_scale,
    rbf_kernel,
    expected_improvement,
    vec_dot,
)


@dataclass
class MFObservation:
    """A single multi-fidelity observation.

    Parameters
    ----------
    parameters : dict[str, float]
        Parameter values for this observation.
    kpi_value : float
        Observed objective value.
    fidelity_level : str
        Name of the fidelity level at which this was evaluated.
    cost : float
        Cost incurred for this evaluation.
    """

    parameters: dict[str, float]
    kpi_value: float
    fidelity_level: str
    cost: float


def _sample_param(spec: ParameterSpec, rng: random.Random) -> Any:
    """Draw one random value for *spec* using the given RNG."""
    if spec.type == VariableType.CATEGORICAL:
        return rng.choice(spec.categories)
    if spec.type == VariableType.DISCRETE:
        return rng.randint(int(spec.lower), int(spec.upper))
    return rng.uniform(spec.lower, spec.upper)


def _clamp(value: float, spec: ParameterSpec) -> Any:
    """Clamp *value* to the bounds of *spec*."""
    if spec.type == VariableType.CATEGORICAL:
        return value
    if spec.type == VariableType.DISCRETE:
        return max(int(spec.lower), min(int(spec.upper), int(round(value))))
    return max(spec.lower, min(spec.upper, value))


class MultiFidelityBackend(AlgorithmPlugin):
    """Multi-fidelity Bayesian optimization backend.

    Uses an ICM kernel to jointly model observations across fidelity
    levels.  Cost-aware acquisition decides which fidelity level to
    evaluate at next.

    Parameters
    ----------
    fidelity_config : FidelityConfig
        Configuration describing available fidelity levels and budget.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, fidelity_config: FidelityConfig, seed: int = 42) -> None:
        self._config = fidelity_config
        self._seed = seed

        # Map fidelity level names to task indices
        self._fidelity_to_task: dict[str, int] = {
            lv.name: i for i, lv in enumerate(fidelity_config.levels)
        }

        # ICM kernel
        self._kernel = ICMKernel(
            n_tasks=fidelity_config.n_levels,
            rank=max(1, fidelity_config.n_levels - 1),
            base_length_scale=1.0,
        )

        # Cost model
        self._cost_model = CostModel(fidelity_config.cost_budget)

        # Observations storage
        self._mf_observations: list[MFObservation] = []
        self._parameter_specs: list[ParameterSpec] = []

        # GP state (populated after fit)
        self._X_train: list[list[float]] = []
        self._y_train: list[float] = []
        self._tasks_train: list[int] = []
        self._L: list[list[float]] | None = None
        self._alpha: list[float] | None = None
        self._best_y: float = float("inf")

    def name(self) -> str:
        """Return the unique identifier for this algorithm."""
        return "multi_fidelity_bo"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        """Ingest standard observations (treated as target fidelity).

        For full multi-fidelity support, use :meth:`add_mf_observation`
        and call :meth:`_fit_gp` directly.
        """
        self._parameter_specs = list(parameter_specs)

        # Convert standard observations to MF observations at target fidelity
        target = self._config.target_fidelity
        target_cost = self._config.target_level.cost
        for obs in observations:
            if obs.is_failure:
                continue
            kpi_val = list(obs.kpi_values.values())[0] if obs.kpi_values else 0.0
            params = {
                k: float(v) for k, v in obs.parameters.items()
                if not isinstance(v, str)
            }
            mf_obs = MFObservation(
                parameters=params,
                kpi_value=kpi_val,
                fidelity_level=target,
                cost=target_cost,
            )
            # Avoid duplicates
            already_exists = any(
                existing.parameters == mf_obs.parameters
                and existing.fidelity_level == mf_obs.fidelity_level
                for existing in self._mf_observations
            )
            if not already_exists:
                self._mf_observations.append(mf_obs)

        self._fit_gp()

    def add_mf_observation(self, obs: MFObservation) -> None:
        """Add a multi-fidelity observation and update cost tracking.

        Parameters
        ----------
        obs : MFObservation
            The observation to add.
        """
        self._mf_observations.append(obs)
        self._cost_model.spend(obs.cost)
        self._fit_gp()

    def _numeric_specs(self) -> list[ParameterSpec]:
        """Return only numeric parameter specs."""
        return [s for s in self._parameter_specs if s.type != VariableType.CATEGORICAL]

    def _to_vec(self, params: dict[str, float]) -> list[float]:
        """Convert parameter dict to numeric vector."""
        vec: list[float] = []
        for s in self._numeric_specs():
            vec.append(float(params.get(s.name, 0.0)))
        return vec

    def _fit_gp(self) -> None:
        """Fit the multi-fidelity GP using current observations."""
        if len(self._mf_observations) < 2:
            self._L = None
            self._alpha = None
            return

        self._X_train = []
        self._y_train = []
        self._tasks_train = []

        for obs in self._mf_observations:
            vec = self._to_vec(obs.parameters)
            self._X_train.append(vec)
            self._y_train.append(obs.kpi_value)
            task = self._fidelity_to_task.get(obs.fidelity_level, 0)
            self._tasks_train.append(task)

        # Auto-scale length scale based on data ranges
        if self._X_train and self._X_train[0]:
            n_dims = len(self._X_train[0])
            ranges = []
            for d in range(n_dims):
                vals = [x[d] for x in self._X_train]
                r = max(vals) - min(vals)
                ranges.append(r if r > 0 else 1.0)
            avg_range = sum(ranges) / max(len(ranges), 1)
            self._kernel.base_length_scale = max(avg_range * 0.5, 0.1)

        # Build kernel matrix and fit GP
        noise = 1e-4
        K = self._kernel.matrix(self._X_train, self._tasks_train, noise=noise)
        try:
            self._L = cholesky(K)
            self._alpha = solve_cholesky(self._L, self._y_train)
        except Exception:
            # Fallback: add more jitter
            n = len(K)
            for i in range(n):
                K[i][i] += 1e-3
            self._L = cholesky(K)
            self._alpha = solve_cholesky(self._L, self._y_train)

        # Track best observed value at target fidelity
        target_task = self._fidelity_to_task[self._config.target_fidelity]
        target_vals = [
            obs.kpi_value for obs in self._mf_observations
            if self._fidelity_to_task.get(obs.fidelity_level, -1) == target_task
        ]
        if target_vals:
            self._best_y = min(target_vals)
        elif self._y_train:
            self._best_y = min(self._y_train)

    def predict(
        self,
        X: list[list[float]],
        fidelity: str | None = None,
    ) -> tuple[list[float], list[float]]:
        """Predict mean and variance at given points.

        Parameters
        ----------
        X : list[list[float]]
            Input points to predict at.
        fidelity : str | None
            Fidelity level name. If None, uses target fidelity.

        Returns
        -------
        tuple[list[float], list[float]]
            (means, variances) for each point in X.
        """
        if self._L is None or self._alpha is None:
            # No model fitted yet -- return prior
            return ([0.0] * len(X), [1.0] * len(X))

        fid_name = fidelity or self._config.target_fidelity
        task = self._fidelity_to_task.get(fid_name, 0)

        means: list[float] = []
        variances: list[float] = []

        B = self._kernel.coregionalization_matrix()

        for x in X:
            # k_star[i] = B[task, tasks_train[i]] * k_base(x, X_train[i])
            k_star = []
            for i in range(len(self._X_train)):
                k_base = rbf_kernel(
                    x, self._X_train[i],
                    length_scale=self._kernel.base_length_scale,
                )
                k_star.append(B[task][self._tasks_train[i]] * k_base)

            mu = vec_dot(k_star, self._alpha)
            v = solve_lower(self._L, k_star)
            k_ss = B[task][task] * 1.0 + 1e-4  # k_base(x, x) = 1.0
            var = max(k_ss - vec_dot(v, v), 1e-12)

            means.append(mu)
            variances.append(var)

        return (means, variances)

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Suggest parameter configurations (at target fidelity).

        Falls back to random sampling when insufficient data.
        """
        rng = random.Random(seed)

        if not self._parameter_specs:
            return [{}] * n_suggestions

        # Fallback: not enough observations
        if self._L is None or self._alpha is None:
            return [
                {s.name: _sample_param(s, rng) for s in self._parameter_specs}
                for _ in range(n_suggestions)
            ]

        numeric_specs = self._numeric_specs()
        n_dims = len(numeric_specs)
        n_candidates = max(100, 20 * max(n_dims, 1))

        suggestions: list[dict[str, Any]] = []

        for _ in range(n_suggestions):
            best_ei = -1.0
            best_point: dict[str, Any] = {}

            for _ in range(n_candidates):
                cand_point: dict[str, Any] = {}
                cand_vec: list[float] = []

                for s in self._parameter_specs:
                    val = _sample_param(s, rng)
                    cand_point[s.name] = val
                    if s.type != VariableType.CATEGORICAL:
                        cand_vec.append(float(val))

                # Predict at target fidelity
                mu_list, var_list = self.predict([cand_vec])
                mu = mu_list[0]
                var = var_list[0]
                sigma = math.sqrt(max(var, 1e-12))

                ei = expected_improvement(mu, sigma, self._best_y)

                if ei > best_ei:
                    best_ei = ei
                    best_point = cand_point

            suggestions.append(best_point)

        return suggestions

    def suggest_with_fidelity(
        self,
        n_suggestions: int = 1,
    ) -> list[tuple[dict[str, float], str]]:
        """Suggest parameter configurations with recommended fidelity level.

        Uses cost-aware EI to jointly decide the point and fidelity.

        Returns
        -------
        list[tuple[dict[str, float], str]]
            List of (parameters, fidelity_level_name) pairs.
        """
        rng = random.Random(self._seed)

        if not self._parameter_specs:
            return [(dict(), self._config.target_fidelity)] * n_suggestions

        numeric_specs = self._numeric_specs()
        n_dims = len(numeric_specs)
        n_candidates = max(50, 10 * max(n_dims, 1))

        results: list[tuple[dict[str, float], str]] = []

        for _ in range(n_suggestions):
            best_score = -1.0
            best_point: dict[str, float] = {}
            best_fidelity = self._config.target_fidelity

            for _ in range(n_candidates):
                # Generate random candidate
                cand_point: dict[str, float] = {}
                cand_vec: list[float] = []

                for s in self._parameter_specs:
                    val = _sample_param(s, rng)
                    if s.type != VariableType.CATEGORICAL:
                        cand_point[s.name] = float(val)
                        cand_vec.append(float(val))

                # Evaluate cost-aware EI at each fidelity level
                for lv in self._config.levels:
                    if not self._cost_model.can_afford(lv.cost):
                        continue

                    mu_list, var_list = self.predict([cand_vec], fidelity=lv.name)
                    mu = mu_list[0]
                    var = var_list[0]

                    score = fidelity_weighted_ei(
                        mean=mu,
                        variance=var,
                        best_y=self._best_y,
                        fidelity=lv.fidelity,
                        cost=lv.cost,
                    )

                    if score > best_score:
                        best_score = score
                        best_point = dict(cand_point)
                        best_fidelity = lv.name

            # If no fidelity was affordable, default to cheapest
            if best_score <= 0.0:
                cheapest = min(self._config.levels, key=lambda lv: lv.cost)
                best_fidelity = cheapest.name
                if not best_point:
                    best_point = {
                        s.name: float(_sample_param(s, rng))
                        for s in self._parameter_specs
                        if s.type != VariableType.CATEGORICAL
                    }

            results.append((best_point, best_fidelity))

        return results

    def get_cost_report(self) -> dict[str, Any]:
        """Return a summary of cost allocation across fidelity levels.

        Returns
        -------
        dict[str, Any]
            Dictionary with budget, spent, remaining, and per-level breakdown.
        """
        per_level: dict[str, dict[str, Any]] = {}
        for lv in self._config.levels:
            level_obs = [
                obs for obs in self._mf_observations
                if obs.fidelity_level == lv.name
            ]
            level_cost = sum(obs.cost for obs in level_obs)
            per_level[lv.name] = {
                "n_evaluations": len(level_obs),
                "total_cost": level_cost,
                "cost_per_eval": lv.cost,
                "fidelity": lv.fidelity,
            }

        return {
            "budget": self._config.cost_budget,
            "spent": self._cost_model.spent,
            "remaining": self._cost_model.remaining,
            "per_level": per_level,
        }

    def capabilities(self) -> dict[str, Any]:
        """Advertise the capabilities of this backend."""
        return {
            "supports_categorical": False,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": False,
            "max_dimensions": None,
            "supports_multi_fidelity": True,
            "n_fidelity_levels": self._config.n_levels,
            "fidelity_levels": [lv.name for lv in self._config.levels],
        }

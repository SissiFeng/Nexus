"""Robust optimization via Monte Carlo perturbation.

Converts standard acquisition functions to robust versions by averaging
over input perturbations. This accounts for input noise (measurement
uncertainty, manufacturing tolerances) to select solutions that perform
well under real-world conditions.

robust_acq(x) = E_delta[acq(x + delta)]  where delta ~ N(0, Sigma_input)

Inspired by OptunaHub Robust BO (v4.6) and robust optimization literature.

References:
- OptunaHub: Robust Bayesian Optimization sampler
- Nogueira et al. (2016): Unscented Bayesian Optimization for robust solutions
- Ur Rehman et al. (2014): Expected improvement-based robust optimization
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable


class RobustOptimizer:
    """Robust optimization wrapper.

    Converts standard acquisition function to robust version by
    averaging acquisition values over Monte Carlo perturbations:

        robust_acq(x) = E_delta[acq(x + delta)]
        where delta ~ N(0, Sigma_input)

    This selects candidates whose acquisition value is robust to
    input noise, not just optimal at the exact point.

    Capabilities:
    1. Robustify acquisition: MC average over perturbations
    2. Robustify candidates: penalize high-sensitivity regions
    3. Per-parameter sensitivity analysis
    4. Serialization/deserialization for persistence
    """

    def __init__(
        self,
        input_noise: dict[str, float] | None = None,
        n_perturbations: int = 20,
        seed: int | None = None,
    ) -> None:
        """Initialize robust optimizer.

        Args:
            input_noise: Per-parameter noise standard deviations.
                Keys are parameter names, values are noise std in the
                parameter's native units. Only continuous parameters
                with entries here will be perturbed.
            n_perturbations: Number of Monte Carlo samples per candidate.
                Higher = more accurate but slower. 20 is a good default.
            seed: Random seed for reproducibility.
        """
        self._noise: dict[str, float] = dict(input_noise) if input_noise else {}
        self._n_perturb = max(1, n_perturbations)
        self._rng = random.Random(seed)

    @property
    def noise_config(self) -> dict[str, float]:
        """Return copy of per-parameter noise configuration."""
        return dict(self._noise)

    @property
    def n_perturbations(self) -> int:
        """Number of Monte Carlo perturbation samples."""
        return self._n_perturb

    def robustify_acquisition(
        self,
        candidates: list[dict[str, Any]],
        acquisition_fn: Callable[[dict[str, Any]], float],
        parameter_specs: list[dict[str, Any]],
    ) -> list[float]:
        """Compute robust acquisition values via MC perturbation.

        For each candidate x, computes:
            robust_acq(x) = (1/N) * sum_{i=1}^{N} acq(x + delta_i)

        where delta_i ~ N(0, sigma^2) for each noisy parameter,
        clipped to parameter bounds.

        Args:
            candidates: List of candidate parameter dicts.
            acquisition_fn: Function mapping a parameter dict to a
                scalar acquisition value.
            parameter_specs: List of parameter specification dicts,
                each with at least 'name' and optionally 'min'/'max'
                for continuous parameters.

        Returns:
            List of robust acquisition values, one per candidate.
        """
        if not self._noise:
            # No noise specified: fall back to standard acquisition
            return [acquisition_fn(c) for c in candidates]

        robust_values: list[float] = []
        for candidate in candidates:
            total = 0.0
            for _ in range(self._n_perturb):
                perturbed = self._perturb(candidate, parameter_specs)
                total += acquisition_fn(perturbed)
            robust_values.append(total / self._n_perturb)
        return robust_values

    def robustify_candidates(
        self,
        candidates: list[dict[str, Any]],
        acquisition_values: list[float],
        parameter_specs: list[dict[str, Any]],
    ) -> list[float]:
        """Penalize candidates in high-noise sensitivity regions.

        Estimates local sensitivity for each candidate and reduces
        the acquisition value for candidates that are highly sensitive
        to perturbation (e.g., near parameter boundaries or in
        steep regions).

        Penalty formula:
            penalized_acq(x) = acq(x) * (1 - sensitivity_penalty)

        where sensitivity_penalty is based on proximity to boundaries
        and overall sensitivity magnitude.

        Args:
            candidates: List of candidate parameter dicts.
            acquisition_values: Pre-computed acquisition values (one per candidate).
            parameter_specs: Parameter specifications for bounds.

        Returns:
            List of penalized acquisition values.
        """
        if not self._noise:
            return list(acquisition_values)

        specs_by_name = {s["name"]: s for s in parameter_specs}
        penalized: list[float] = []

        for candidate, acq_val in zip(candidates, acquisition_values):
            penalty = 0.0
            n_noisy = 0

            for param_name, noise_std in self._noise.items():
                if param_name not in candidate:
                    continue
                spec = specs_by_name.get(param_name, {})
                p_min = spec.get("min")
                p_max = spec.get("max")

                if p_min is None or p_max is None:
                    continue

                val = float(candidate[param_name])
                param_range = p_max - p_min
                if param_range <= 0:
                    continue

                n_noisy += 1

                # Boundary proximity penalty: higher penalty near edges
                # Normalized distance to nearest boundary
                dist_to_min = (val - p_min) / param_range
                dist_to_max = (p_max - val) / param_range
                boundary_dist = min(dist_to_min, dist_to_max)

                # Noise-to-range ratio
                noise_ratio = noise_std / param_range

                # Penalty increases with noise ratio and boundary proximity
                # Maximum penalty per parameter: ~0.3
                param_penalty = noise_ratio * (1.0 - boundary_dist)
                param_penalty = min(0.3, max(0.0, param_penalty))
                penalty += param_penalty

            if n_noisy > 0:
                # Average penalty across noisy parameters, capped at 0.5
                avg_penalty = min(0.5, penalty / n_noisy)
                penalized.append(acq_val * (1.0 - avg_penalty))
            else:
                penalized.append(acq_val)

        return penalized

    def _perturb(
        self,
        x: dict[str, Any],
        specs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Add Gaussian perturbation to a candidate, clip to bounds.

        Only perturbs continuous parameters that have noise specified.
        Categorical and other parameters are left unchanged.

        Args:
            x: Candidate parameter dict.
            specs: Parameter specifications with bounds.

        Returns:
            New parameter dict with perturbed values.
        """
        perturbed = dict(x)
        specs_by_name = {s["name"]: s for s in specs}

        for param_name, noise_std in self._noise.items():
            if param_name not in perturbed:
                continue
            if noise_std <= 0:
                continue

            spec = specs_by_name.get(param_name, {})

            # Only perturb if the value is numeric
            val = perturbed[param_name]
            if not isinstance(val, (int, float)):
                continue

            # Gaussian perturbation
            delta = self._rng.gauss(0, noise_std)
            new_val = float(val) + delta

            # Clip to parameter bounds if available
            p_min = spec.get("min")
            p_max = spec.get("max")
            if p_min is not None:
                new_val = max(float(p_min), new_val)
            if p_max is not None:
                new_val = min(float(p_max), new_val)

            perturbed[param_name] = new_val

        return perturbed

    def sensitivity_analysis(
        self,
        candidate: dict[str, Any],
        acquisition_fn: Callable[[dict[str, Any]], float],
        parameter_specs: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Per-parameter sensitivity analysis.

        For each parameter with noise, estimates how much perturbation
        of that single parameter changes the acquisition value.

        Sensitivity is measured as the standard deviation of acquisition
        values under perturbation of each parameter individually.

        Args:
            candidate: Candidate parameter dict.
            acquisition_fn: Acquisition function.
            parameter_specs: Parameter specifications with bounds.

        Returns:
            Dictionary mapping parameter name to sensitivity score.
            Higher values indicate more sensitive parameters.
        """
        base_acq = acquisition_fn(candidate)
        sensitivities: dict[str, float] = {}

        for param_name, noise_std in self._noise.items():
            if param_name not in candidate:
                continue
            if noise_std <= 0:
                sensitivities[param_name] = 0.0
                continue

            # Create specs with only this parameter's noise
            single_noise = {param_name: noise_std}
            single_optimizer = RobustOptimizer(
                input_noise=single_noise,
                n_perturbations=self._n_perturb,
                seed=None,
            )
            # Share the RNG state for consistency
            single_optimizer._rng = random.Random()
            single_optimizer._rng.setstate(self._rng.getstate())

            # Collect acquisition values under perturbation
            acq_values: list[float] = []
            for _ in range(self._n_perturb):
                perturbed = single_optimizer._perturb(candidate, parameter_specs)
                acq_values.append(acquisition_fn(perturbed))

            # Sensitivity = std of acquisition values
            if len(acq_values) > 1:
                mean_acq = sum(acq_values) / len(acq_values)
                variance = sum(
                    (v - mean_acq) ** 2 for v in acq_values
                ) / (len(acq_values) - 1)
                sensitivities[param_name] = math.sqrt(max(0.0, variance))
            else:
                sensitivities[param_name] = 0.0

        return sensitivities

    def to_dict(self) -> dict[str, Any]:
        """Serialize robust optimizer configuration.

        Returns:
            Dictionary with noise config and perturbation settings.
        """
        return {
            "input_noise": dict(self._noise),
            "n_perturbations": self._n_perturb,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobustOptimizer:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with 'input_noise' and 'n_perturbations'.

        Returns:
            Restored RobustOptimizer instance.
        """
        return cls(
            input_noise=data.get("input_noise"),
            n_perturbations=data.get("n_perturbations", 20),
            seed=data.get("seed"),
        )

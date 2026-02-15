"""AlgorithmPlugin wrapper that operates in a learned latent space."""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.core.models import (
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.latent.models import LatentSpace
from optimization_copilot.latent.transform import LatentTransform
from optimization_copilot.plugins.base import AlgorithmPlugin


class LatentPlugin(AlgorithmPlugin):
    """Wraps an inner ``AlgorithmPlugin`` so that it operates in latent space.

    The wrapper:

    1. Converts historical observations to latent coordinates before
       calling ``inner.fit()``.
    2. Asks the inner plugin for suggestions in latent space, then maps
       them back to the original parameter space via ``from_latent()``.

    Parameters
    ----------
    inner_plugin:
        The actual optimisation algorithm that will operate in latent space.
    latent_space:
        A pre-fitted :class:`LatentSpace`.
    transform:
        The :class:`LatentTransform` used for projections.
    original_specs:
        Original-space parameter specifications.  When provided these are
        used to clamp and round reconstructed parameters.
    """

    def __init__(
        self,
        inner_plugin: AlgorithmPlugin,
        latent_space: LatentSpace,
        transform: LatentTransform,
        original_specs: list[ParameterSpec] | None = None,
    ) -> None:
        self._inner = inner_plugin
        self._latent_space = latent_space
        self._transform = transform
        self._original_specs = original_specs

        # Build latent parameter specs: z0, z1, ...
        # Bounds are +/-3*sqrt(eigenvalue) (approx 3-sigma in latent space).
        self._latent_specs: list[ParameterSpec] = []
        for i in range(latent_space.n_components):
            ev = latent_space.eigenvalues[i]
            spread = 3.0 * math.sqrt(max(ev, 0.0))
            self._latent_specs.append(
                ParameterSpec(
                    name=f"z{i}",
                    type=VariableType.CONTINUOUS,
                    lower=-spread,
                    upper=spread,
                )
            )

    # -- AlgorithmPlugin interface -------------------------------------------

    def name(self) -> str:
        """Return a composite name indicating latent wrapping."""
        return f"latent_{self._inner.name()}"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        """Transform observations to latent coordinates and fit the inner plugin.

        Failed observations (``is_failure=True``) are retained but their
        latent parameters are set to zero vectors so that the inner plugin
        still receives them.
        """
        latent_observations: list[Observation] = []
        for obs in observations:
            if obs.is_failure:
                # Map failures to the origin in latent space.
                latent_params: dict[str, Any] = {
                    f"z{i}": 0.0
                    for i in range(self._latent_space.n_components)
                }
            else:
                coords = self._transform.to_latent(
                    obs.parameters, self._latent_space
                )
                latent_params = {
                    f"z{i}": coords[i]
                    for i in range(self._latent_space.n_components)
                }

            latent_observations.append(
                Observation(
                    iteration=obs.iteration,
                    parameters=latent_params,
                    kpi_values=obs.kpi_values,
                    qc_passed=obs.qc_passed,
                    is_failure=obs.is_failure,
                    failure_reason=obs.failure_reason,
                    timestamp=obs.timestamp,
                    metadata=obs.metadata,
                )
            )

        self._inner.fit(latent_observations, self._latent_specs)

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Get latent suggestions from the inner plugin and map back to original space."""
        latent_suggestions = self._inner.suggest(n_suggestions, seed)

        original_suggestions: list[dict[str, Any]] = []
        for latent_params in latent_suggestions:
            coords = [
                float(latent_params.get(f"z{i}", 0.0))
                for i in range(self._latent_space.n_components)
            ]
            original_params = self._transform.from_latent(
                coords, self._latent_space, self._original_specs
            )
            original_suggestions.append(original_params)

        return original_suggestions

    def capabilities(self) -> dict[str, Any]:
        """Return inner capabilities augmented with latent wrapping metadata."""
        caps = dict(self._inner.capabilities())
        caps["latent_wrapped"] = True
        caps["supports_categorical"] = False
        return caps

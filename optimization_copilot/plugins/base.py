"""Abstract base classes for optimization algorithm plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from optimization_copilot.core.models import Observation, ParameterSpec


class AlgorithmPlugin(ABC):
    """Base class that every optimization backend must implement.

    Plugins follow a fit-then-suggest lifecycle:
      1. ``fit()`` ingests historical observations and parameter specs.
      2. ``suggest()`` returns candidate parameter configurations.
      3. ``capabilities()`` advertises what the plugin supports so the
         meta-controller can select the right backend for each campaign.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this algorithm."""
        ...

    @abstractmethod
    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        """Ingest historical observations and parameter definitions.

        Parameters
        ----------
        observations:
            All recorded experimental results so far.
        parameter_specs:
            The search-space definition (bounds, types, categories).
        """
        ...

    @abstractmethod
    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Return *n_suggestions* candidate parameter configurations.

        Each element of the returned list is a dict mapping parameter
        names to their suggested values.

        Parameters
        ----------
        n_suggestions:
            How many candidates to produce.
        seed:
            Random seed for reproducibility.
        """
        ...

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Advertise the capabilities and characteristics of this backend.

        The returned dict should include keys such as:
          - ``supports_categorical`` (bool)
          - ``supports_continuous`` (bool)
          - ``supports_discrete`` (bool)
          - ``supports_batch`` (bool)
          - ``requires_observations`` (bool)
          - ``max_dimensions`` (int | None)

        The meta-controller uses these to match backends to campaigns.
        """
        ...

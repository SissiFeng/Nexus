"""Baseline adapters for case study comparisons.

Wraps existing backend ``AlgorithmPlugin`` implementations as named baselines
for benchmarking and case study comparisons.  Provides a registry / factory
pattern so that case studies can request baselines by name without importing
backend internals.

Example usage::

    from optimization_copilot.case_studies.baselines.adapters import (
        BaselineAdapter,
        get_default_baselines,
    )

    # Get a single baseline by name
    gp = BaselineAdapter.get("gp_bo")

    # Get the default comparison set (random, sobol, gp_bo)
    defaults = get_default_baselines()

    # Register a custom baseline
    BaselineAdapter.register("my_algo", lambda: MyAlgorithm())
"""

from __future__ import annotations

from typing import Any, Callable

from optimization_copilot.plugins.base import AlgorithmPlugin
from optimization_copilot.backends.builtin import (
    RandomSampler,
    SobolSampler,
    GaussianProcessBO,
    CMAESSampler,
)
from optimization_copilot.backends.gp_heteroscedastic import HeteroscedasticGP


# ---------------------------------------------------------------------------
# Registry / Factory
# ---------------------------------------------------------------------------


class BaselineAdapter:
    """Registry and factory for baseline optimization strategies.

    Provides named access to existing ``AlgorithmPlugin`` implementations,
    with optional configuration overrides for case study comparisons.

    The default registry includes five strategies:

    =========  =================================================
    Name       Backend class
    =========  =================================================
    random     ``RandomSampler``  -- uniform random sampling
    sobol      ``SobolSampler``   -- Sobol quasi-random sequence
    gp_bo      ``GaussianProcessBO`` -- GP-based BO with EI
    cma_es     ``CMAESSampler``   -- CMA-ES evolutionary strategy
    het_gp     ``HeteroscedasticGP`` -- GP with per-point noise
    =========  =================================================
    """

    # Default registry mapping name -> factory function.
    # Each factory returns a *fresh* AlgorithmPlugin instance.
    REGISTRY: dict[str, Callable[[], AlgorithmPlugin]] = {
        "random": lambda: RandomSampler(),
        "sobol": lambda: SobolSampler(),
        "gp_bo": lambda: GaussianProcessBO(),
        "cma_es": lambda: CMAESSampler(),
        "het_gp": lambda: HeteroscedasticGP(),
    }

    # -- retrieval -----------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> AlgorithmPlugin:
        """Get a baseline strategy by name.

        Parameters
        ----------
        name : str
            Registry key (e.g. ``"random"``, ``"gp_bo"``).

        Returns
        -------
        AlgorithmPlugin
            A freshly constructed plugin instance.

        Raises
        ------
        ValueError
            If *name* is not in the registry.
        """
        if name not in cls.REGISTRY:
            raise ValueError(
                f"Unknown baseline: {name!r}. "
                f"Available: {list(cls.REGISTRY)}"
            )
        return cls.REGISTRY[name]()

    @classmethod
    def get_all(cls) -> dict[str, AlgorithmPlugin]:
        """Get all registered baselines.

        Returns
        -------
        dict[str, AlgorithmPlugin]
            Mapping of name -> fresh plugin instance for every registered
            baseline.
        """
        return {name: factory() for name, factory in cls.REGISTRY.items()}

    @classmethod
    def get_subset(cls, names: list[str]) -> dict[str, AlgorithmPlugin]:
        """Get a subset of baselines by names.

        Parameters
        ----------
        names : list[str]
            Registry keys to include.

        Returns
        -------
        dict[str, AlgorithmPlugin]
            Mapping of name -> plugin for the requested subset.

        Raises
        ------
        ValueError
            If any name is not in the registry.
        """
        return {name: cls.get(name) for name in names}

    # -- introspection -------------------------------------------------------

    @classmethod
    def available(cls) -> list[str]:
        """List available baseline names.

        Returns
        -------
        list[str]
            Sorted list of registered baseline names.
        """
        return list(cls.REGISTRY.keys())

    # -- mutation ------------------------------------------------------------

    @classmethod
    def register(cls, name: str, factory: Callable[[], AlgorithmPlugin]) -> None:
        """Register a custom baseline.

        Parameters
        ----------
        name : str
            Registry key.
        factory : callable
            Zero-argument callable returning an ``AlgorithmPlugin``.
        """
        cls.REGISTRY[name] = factory

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a baseline from the registry.

        Parameters
        ----------
        name : str
            Registry key to remove.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        if name not in cls.REGISTRY:
            raise KeyError(f"Baseline {name!r} not in registry")
        del cls.REGISTRY[name]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_default_baselines() -> dict[str, AlgorithmPlugin]:
    """Return the default set of baselines for case study comparison.

    Returns a fresh set of baselines (random, sobol, gp_bo).
    These are the most commonly compared strategies:
    a non-adaptive baseline, a space-filling design, and a model-based method.

    Returns
    -------
    dict[str, AlgorithmPlugin]
        Three baseline instances keyed by name.
    """
    return BaselineAdapter.get_subset(["random", "sobol", "gp_bo"])


def get_all_baselines() -> dict[str, AlgorithmPlugin]:
    """Return all available baselines.

    Returns
    -------
    dict[str, AlgorithmPlugin]
        All registered baselines as fresh instances.
    """
    return BaselineAdapter.get_all()


def get_baseline_capabilities() -> dict[str, dict[str, Any]]:
    """Return capabilities of all baselines for selection guidance.

    Useful for programmatic baseline selection: e.g. filter baselines that
    support categorical parameters or do not require observations.

    Returns
    -------
    dict[str, dict]
        ``{baseline_name: capabilities_dict}`` for every registered baseline.
    """
    result: dict[str, dict[str, Any]] = {}
    for name in BaselineAdapter.available():
        plugin = BaselineAdapter.get(name)
        result[name] = plugin.capabilities()
    return result


def select_baselines_for_space(
    has_categorical: bool = False,
    requires_no_observations: bool = False,
) -> dict[str, AlgorithmPlugin]:
    """Select baselines compatible with a given search space.

    Parameters
    ----------
    has_categorical : bool
        If True, only return baselines that support categorical parameters.
    requires_no_observations : bool
        If True, only return baselines that do not require observations.

    Returns
    -------
    dict[str, AlgorithmPlugin]
        Compatible baselines as fresh instances.
    """
    capabilities = get_baseline_capabilities()
    selected: list[str] = []
    for name, caps in capabilities.items():
        if has_categorical and not caps.get("supports_categorical", False):
            continue
        if requires_no_observations and caps.get("requires_observations", False):
            continue
        selected.append(name)
    return BaselineAdapter.get_subset(selected)

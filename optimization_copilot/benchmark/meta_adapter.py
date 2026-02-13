"""MetaController adapter for benchmark evaluation.

Wraps the full MetaController pipeline as an ``AlgorithmPlugin`` so it
can compete on the same harness as individual backends.

On each fit/suggest cycle the adapter:

1. Builds a ``CampaignSnapshot`` from the current observations.
2. Runs the ``DiagnosticEngine`` to produce a diagnostics dict.
3. Runs the ``ProblemProfiler`` to produce a ``ProblemFingerprint``.
4. Calls ``MetaController.decide()`` to select a strategy.
5. Delegates to the corresponding backend plugin for suggestion.

When any dependency is missing or raises, the adapter falls back to a
simple heuristic and ultimately to random sampling.
"""

from __future__ import annotations

import random as _random
import uuid
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    ProblemFingerprint,
)
from optimization_copilot.plugins.base import AlgorithmPlugin


# ---------------------------------------------------------------------------
# Name mapping: MetaController strategy names --> plugin name() strings
# ---------------------------------------------------------------------------
# The MetaController uses short names (e.g. "tpe", "random") in its
# PHASE_BACKEND_MAP and FINGERPRINT_BACKEND_OVERRIDES.  Actual plugin
# classes registered via PluginRegistry expose longer names via name().
# This dict bridges the two vocabularies.

_META_TO_PLUGIN: dict[str, str] = {
    # Names that appear in controller.PHASE_BACKEND_MAP /
    # controller.FINGERPRINT_BACKEND_OVERRIDES
    "random": "random_sampler",
    "latin_hypercube": "latin_hypercube_sampler",
    "tpe": "tpe_sampler",
    "random_forest_surrogate": "random_forest_bo",
    "cma_es": "cmaes_sampler",
    # Additional short-hand names that may appear through portfolio scoring
    # or future controller updates
    "sobol": "sobol_sampler",
    "gp_bo": "gaussian_process_bo",
    "rf_bo": "random_forest_bo",
    "de": "differential_evolution",
    "turbo": "turbo_sampler",
    "nsga2": "nsga2_sampler",
    "lhs": "latin_hypercube_sampler",
}

# Reverse mapping for diagnostics/logging: plugin name --> meta name
_PLUGIN_TO_META: dict[str, str] = {v: k for k, v in _META_TO_PLUGIN.items()}


class MetaControllerAdapter(AlgorithmPlugin):
    """Adapts the MetaController decision pipeline to the AlgorithmPlugin interface.

    Parameters
    ----------
    backend_factories
        Mapping of plugin ``name()`` string to ``AlgorithmPlugin`` **class**.
        For example ``{"tpe_sampler": TPESampler, "random_sampler": RandomSampler}``.
        When *None* the adapter will attempt to import and auto-discover
        built-in backends.
    meta_kwargs
        Extra keyword arguments forwarded to ``MetaController.__init__``
        (e.g. custom ``SwitchingThresholds`` or ``available_backends``).
    """

    def __init__(
        self,
        backend_factories: dict[str, type[AlgorithmPlugin]] | None = None,
        meta_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._backend_factories: dict[str, type[AlgorithmPlugin]] = (
            backend_factories if backend_factories is not None else self._auto_discover()
        )
        self._meta_kwargs: dict[str, Any] = meta_kwargs or {}
        self._observations: list[Observation] = []
        self._param_specs: list[ParameterSpec] = []
        self._current_backend: AlgorithmPlugin | None = None
        self._current_strategy: str = "random"
        self._campaign_id: str = uuid.uuid4().hex[:12]

    # ── AlgorithmPlugin interface ──────────────────────────────

    def name(self) -> str:
        return "meta_controller"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._observations = list(observations)
        self._param_specs = list(parameter_specs)

        # Build CampaignSnapshot
        snapshot = self._build_snapshot()

        # Compute diagnostics (as dict[str, float] for MetaController)
        diagnostics = self._compute_diagnostics(snapshot)

        # Compute problem fingerprint
        fingerprint = self._compute_fingerprint(snapshot)

        # Call MetaController to decide strategy
        strategy_name = self._decide_strategy(snapshot, diagnostics, fingerprint)

        # Map meta name to plugin name
        plugin_name = _META_TO_PLUGIN.get(strategy_name, strategy_name)

        # Instantiate and fit the selected backend
        factory = self._backend_factories.get(plugin_name)
        if factory is not None:
            try:
                backend = factory()
                backend.fit(observations, parameter_specs)
                self._current_backend = backend
                self._current_strategy = strategy_name
            except Exception:
                # Backend instantiation or fit failed -- keep previous or None
                self._current_backend = None
        else:
            # No factory for this name -- try the name as-is (plugin name == meta name)
            factory = self._backend_factories.get(strategy_name)
            if factory is not None:
                try:
                    backend = factory()
                    backend.fit(observations, parameter_specs)
                    self._current_backend = backend
                    self._current_strategy = strategy_name
                except Exception:
                    self._current_backend = None
            else:
                self._current_backend = None

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        # Delegate to the backend selected during fit()
        if self._current_backend is not None:
            try:
                return self._current_backend.suggest(n_suggestions, seed)
            except Exception:
                pass  # Fall through to random fallback

        # Fallback: uniform random sampling within bounds
        return self._random_suggestions(n_suggestions, seed)

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": False,
            "max_dimensions": None,
            "adaptive": True,
            "meta_controller": True,
        }

    # ── Internal helpers ──────────────────────────────────────

    def _build_snapshot(self) -> CampaignSnapshot:
        """Build a CampaignSnapshot from the current observation buffer."""
        return CampaignSnapshot(
            campaign_id=self._campaign_id,
            parameter_specs=self._param_specs,
            observations=self._observations,
            objective_names=self._infer_objective_names(),
            objective_directions=["minimize"],
            current_iteration=len(self._observations),
        )

    def _infer_objective_names(self) -> list[str]:
        """Infer objective names from observations' kpi_values keys.

        Falls back to ``["objective"]`` when observations are empty or
        have inconsistent keys.
        """
        if not self._observations:
            return ["objective"]
        # Use the kpi_values keys from the first observation
        first_kpis = self._observations[0].kpi_values
        if first_kpis:
            return list(first_kpis.keys())[:1]  # Primary objective only
        return ["objective"]

    def _compute_diagnostics(self, snapshot: CampaignSnapshot) -> dict[str, float]:
        """Compute diagnostic signals and return as a plain dict.

        The MetaController expects ``diagnostics: dict[str, float]`` with
        the 17 signal keys produced by ``DiagnosticsVector.to_dict()``.
        """
        try:
            from optimization_copilot.diagnostics.engine import DiagnosticEngine

            engine = DiagnosticEngine()
            vector = engine.compute(snapshot)
            return vector.to_dict()
        except Exception:
            # Return safe zero-defaults that let MetaController still decide
            return {
                "convergence_trend": 0.0,
                "improvement_velocity": 0.0,
                "variance_contraction": 1.0,
                "noise_estimate": 0.0,
                "failure_rate": 0.0,
                "failure_clustering": 0.0,
                "feasibility_shrinkage": 0.0,
                "parameter_drift": 0.0,
                "model_uncertainty": 0.0,
                "exploration_coverage": 0.0,
                "kpi_plateau_length": 0,
                "best_kpi_value": 0.0,
                "data_efficiency": 0.0,
                "constraint_violation_rate": 0.0,
                "miscalibration_score": 0.0,
                "overconfidence_rate": 0.0,
                "signal_to_noise_ratio": 0.0,
            }

    def _compute_fingerprint(self, snapshot: CampaignSnapshot) -> ProblemFingerprint:
        """Compute problem fingerprint from the snapshot."""
        try:
            from optimization_copilot.profiler.profiler import ProblemProfiler

            profiler = ProblemProfiler()
            return profiler.profile(snapshot)
        except Exception:
            return ProblemFingerprint()

    def _decide_strategy(
        self,
        snapshot: CampaignSnapshot,
        diagnostics: dict[str, float],
        fingerprint: ProblemFingerprint,
    ) -> str:
        """Call MetaController.decide() and extract the chosen backend name.

        Falls back to a simple observation-count heuristic when the
        MetaController is unavailable.
        """
        try:
            from optimization_copilot.meta_controller.controller import MetaController

            # Build available_backends list from _META_TO_PLUGIN keys
            # so the controller knows what it can select
            available_meta_names = list(_META_TO_PLUGIN.keys())
            # Filter to backends we actually have factories for
            available = [
                meta_name
                for meta_name, plugin_name in _META_TO_PLUGIN.items()
                if plugin_name in self._backend_factories
            ]
            if not available:
                available = ["random"]

            controller = MetaController(
                available_backends=available,
                **self._meta_kwargs,
            )
            decision = controller.decide(
                snapshot=snapshot,
                diagnostics=diagnostics,
                fingerprint=fingerprint,
            )
            return decision.backend_name
        except Exception:
            return self._simple_heuristic(len(self._observations))

    @staticmethod
    def _simple_heuristic(n_obs: int) -> str:
        """Simple strategy selection heuristic as fallback.

        Returns MetaController-style short names.
        """
        if n_obs < 10:
            return "random"
        elif n_obs < 30:
            return "tpe"
        else:
            return "latin_hypercube"

    def _random_suggestions(
        self, n_suggestions: int, seed: int
    ) -> list[dict[str, Any]]:
        """Generate uniform random points within parameter bounds."""
        rng = _random.Random(seed)
        suggestions: list[dict[str, Any]] = []
        for _ in range(n_suggestions):
            point: dict[str, Any] = {}
            for ps in self._param_specs:
                lo = ps.lower if ps.lower is not None else 0.0
                hi = ps.upper if ps.upper is not None else 1.0
                point[ps.name] = rng.uniform(lo, hi)
            suggestions.append(point)
        return suggestions

    @staticmethod
    def _auto_discover() -> dict[str, type[AlgorithmPlugin]]:
        """Attempt to import built-in backends and return a factory dict.

        Returns an empty dict if the import fails so the adapter can
        still function (random fallback).
        """
        factories: dict[str, type[AlgorithmPlugin]] = {}
        try:
            from optimization_copilot.backends.builtin import (  # type: ignore[import-untyped]
                CMAESSampler,
                DifferentialEvolution,
                GaussianProcessBO,
                LatinHypercubeSampler,
                NSGA2Sampler,
                RandomForestBO,
                RandomSampler,
                SobolSampler,
                TPESampler,
                TuRBOSampler,
            )

            _auto: list[type[AlgorithmPlugin]] = [
                RandomSampler,
                LatinHypercubeSampler,
                TPESampler,
                SobolSampler,
                GaussianProcessBO,
                RandomForestBO,
                CMAESSampler,
                DifferentialEvolution,
                NSGA2Sampler,
                TuRBOSampler,
            ]
            for cls in _auto:
                instance = cls()
                factories[instance.name()] = cls
        except Exception:
            pass
        return factories

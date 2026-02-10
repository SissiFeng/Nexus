"""Counterfactual evaluation module for offline what-if analysis.

Given a campaign history, estimate what would have happened if a different
optimization backend had been used.  Two estimation methods are supported:

* **replay** -- re-run the alternative backend on the same observation
  history, matching each suggestion to the nearest actually-observed point.
* **surrogate** -- (placeholder for future surrogate-model-based estimation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.plugins.base import AlgorithmPlugin


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualResult:
    """Outcome of a single counterfactual evaluation."""

    baseline_backend: str
    alternative_backend: str
    baseline_best_kpi: float
    estimated_alternative_kpi: float
    estimated_speedup: float  # fewer iterations to reach baseline best (fraction)
    confidence: float  # 0-1, reliability of the estimate
    method: str  # "replay" or "surrogate"
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalised_distance(
    suggestion: dict[str, Any],
    obs: Observation,
    specs: list[ParameterSpec],
) -> float:
    """Normalised Euclidean distance between *suggestion* and *obs* parameters.

    Continuous / discrete parameters are normalised by their range.
    Categorical parameters contribute 0.0 (match) or 1.0 (mismatch).
    Missing parameters on either side are treated as maximum distance for that
    dimension.
    """
    total = 0.0
    n_dims = 0
    for spec in specs:
        s_val = suggestion.get(spec.name)
        o_val = obs.parameters.get(spec.name)

        if s_val is None or o_val is None:
            total += 1.0
            n_dims += 1
            continue

        if spec.type == VariableType.CATEGORICAL:
            total += 0.0 if s_val == o_val else 1.0
        else:
            lo = spec.lower if spec.lower is not None else 0.0
            hi = spec.upper if spec.upper is not None else 1.0
            rng = hi - lo if hi != lo else 1.0
            diff = (float(s_val) - float(o_val)) / rng
            total += diff * diff

        n_dims += 1

    if n_dims == 0:
        return 0.0
    return math.sqrt(total / n_dims)


def _nearest_observation(
    suggestion: dict[str, Any],
    observations: list[Observation],
    specs: list[ParameterSpec],
) -> Observation:
    """Return the observation whose parameters are closest to *suggestion*.

    Distance is normalised Euclidean over all parameter specs.

    Raises ``ValueError`` if *observations* is empty.
    """
    if not observations:
        raise ValueError("Cannot find nearest observation in an empty list.")

    best_obs = observations[0]
    best_dist = _normalised_distance(suggestion, best_obs, specs)

    for obs in observations[1:]:
        d = _normalised_distance(suggestion, obs, specs)
        if d < best_dist:
            best_dist = d
            best_obs = obs

    return best_obs


def _best_kpi_value(
    observations: list[Observation],
    objective_name: str,
    direction: str,
) -> float:
    """Return the best KPI value seen so far among *observations*.

    Parameters
    ----------
    direction:
        ``"minimize"`` or ``"maximize"``.
    """
    values = [
        obs.kpi_values[objective_name]
        for obs in observations
        if not obs.is_failure and objective_name in obs.kpi_values
    ]
    if not values:
        return float("inf") if direction == "minimize" else float("-inf")
    return min(values) if direction == "minimize" else max(values)


def _cumulative_best_curve(
    observations: list[Observation],
    objective_name: str,
    direction: str,
) -> list[float]:
    """Build a monotone best-so-far curve over the observation sequence."""
    curve: list[float] = []
    best = float("inf") if direction == "minimize" else float("-inf")
    cmp = min if direction == "minimize" else max

    for obs in observations:
        if not obs.is_failure and objective_name in obs.kpi_values:
            best = cmp(best, obs.kpi_values[objective_name])
        curve.append(best)

    return curve


def _iterations_to_reach(
    curve: list[float],
    target: float,
    direction: str,
) -> int | None:
    """Return the first iteration index where *curve* reaches *target*.

    Returns ``None`` if the target is never reached.
    """
    for i, val in enumerate(curve):
        if direction == "minimize" and val <= target:
            return i
        if direction == "maximize" and val >= target:
            return i
    return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class CounterfactualEvaluator:
    """Performs offline counterfactual analysis on campaign histories."""

    # Minimum observations required for a meaningful replay.
    MIN_OBSERVATIONS = 5

    def evaluate_replay(
        self,
        snapshot: CampaignSnapshot,
        actual_backend: str,
        alternative_backend: str,
        alternative_plugin: AlgorithmPlugin,
        seed: int = 42,
    ) -> CounterfactualResult:
        """Replay an alternative backend against recorded campaign history.

        At each step *i* the alternative is fit on ``observations[:i]``, asked
        for a suggestion, then the nearest *actual* observation is used to
        represent the KPI the alternative would have obtained.  This builds
        a simulated convergence curve that is compared to the real one.

        Parameters
        ----------
        snapshot:
            Full campaign history.
        actual_backend:
            Name of the backend that was actually used.
        alternative_backend:
            Name of the alternative backend being evaluated.
        alternative_plugin:
            An ``AlgorithmPlugin`` instance for the alternative.
        seed:
            Random seed forwarded to the plugin's ``suggest()``.

        Returns
        -------
        CounterfactualResult
        """
        observations = snapshot.successful_observations
        specs = snapshot.parameter_specs

        # Use the first objective for single-objective analysis.
        obj_name = snapshot.objective_names[0]
        direction = snapshot.objective_directions[0]

        # -- Insufficient data guard ------------------------------------
        if len(observations) < self.MIN_OBSERVATIONS:
            baseline_best = _best_kpi_value(observations, obj_name, direction)
            return CounterfactualResult(
                baseline_backend=actual_backend,
                alternative_backend=alternative_backend,
                baseline_best_kpi=baseline_best,
                estimated_alternative_kpi=baseline_best,
                estimated_speedup=0.0,
                confidence=max(0.0, len(observations) / self.MIN_OBSERVATIONS * 0.3),
                method="replay",
                details={"reason": "insufficient_data",
                         "n_observations": len(observations)},
            )

        # -- Baseline curve ---------------------------------------------
        baseline_curve = _cumulative_best_curve(observations, obj_name, direction)
        baseline_best = baseline_curve[-1] if baseline_curve else (
            float("inf") if direction == "minimize" else float("-inf")
        )

        # -- Simulate alternative curve ---------------------------------
        alt_kpis: list[Observation] = []

        # We need at least 1 observation for the first fit, so start
        # replaying from index 1 onward.
        for i in range(1, len(observations)):
            history = observations[:i]
            alternative_plugin.fit(history, specs)

            suggestions = alternative_plugin.suggest(n_suggestions=1, seed=seed + i)
            if not suggestions:
                # Plugin returned nothing; carry forward previous observation.
                alt_kpis.append(observations[i])
                continue

            nearest = _nearest_observation(suggestions[0], observations, specs)
            alt_kpis.append(nearest)

        # Prepend the first observation (both backends see the same seed point).
        alt_kpis.insert(0, observations[0])

        alt_curve = _cumulative_best_curve(alt_kpis, obj_name, direction)
        alt_best = alt_curve[-1] if alt_curve else baseline_best

        # -- Speedup estimation -----------------------------------------
        baseline_iters = _iterations_to_reach(baseline_curve, baseline_best, direction)
        alt_iters = _iterations_to_reach(alt_curve, baseline_best, direction)

        if baseline_iters is not None and alt_iters is not None and baseline_iters > 0:
            speedup = (baseline_iters - alt_iters) / baseline_iters
        else:
            speedup = 0.0

        # -- Confidence heuristic ---------------------------------------
        # More observations and a tighter nearest-match contribute to
        # higher confidence.
        n = len(observations)
        data_confidence = min(1.0, n / 30.0)  # saturates at 30 observations

        # Average nearest-match distance as a proxy for interpolation quality.
        avg_dist = 0.0
        dist_count = 0
        for i in range(1, len(observations)):
            history = observations[:i]
            alternative_plugin.fit(history, specs)
            suggestions = alternative_plugin.suggest(n_suggestions=1, seed=seed + i)
            if suggestions:
                d = _normalised_distance(suggestions[0], alt_kpis[i], specs)
                avg_dist += d
                dist_count += 1
        if dist_count > 0:
            avg_dist /= dist_count
        match_confidence = max(0.0, 1.0 - avg_dist)

        confidence = round(0.6 * data_confidence + 0.4 * match_confidence, 3)
        confidence = max(0.0, min(1.0, confidence))

        return CounterfactualResult(
            baseline_backend=actual_backend,
            alternative_backend=alternative_backend,
            baseline_best_kpi=baseline_best,
            estimated_alternative_kpi=alt_best,
            estimated_speedup=round(speedup, 4),
            confidence=confidence,
            method="replay",
            details={
                "n_observations": n,
                "baseline_curve": baseline_curve,
                "alternative_curve": alt_curve,
                "baseline_iters_to_best": baseline_iters,
                "alternative_iters_to_best": alt_iters,
            },
        )

    def evaluate_all_alternatives(
        self,
        snapshot: CampaignSnapshot,
        actual_backend: str,
        alternatives: dict[str, AlgorithmPlugin],
        seed: int = 42,
    ) -> list[CounterfactualResult]:
        """Run :meth:`evaluate_replay` for every alternative backend.

        Parameters
        ----------
        alternatives:
            Mapping of backend name to plugin instance.

        Returns
        -------
        list[CounterfactualResult]
            One result per alternative, in insertion order.
        """
        results: list[CounterfactualResult] = []
        for alt_name, plugin in alternatives.items():
            result = self.evaluate_replay(
                snapshot=snapshot,
                actual_backend=actual_backend,
                alternative_backend=alt_name,
                alternative_plugin=plugin,
                seed=seed,
            )
            results.append(result)
        return results

    @staticmethod
    def summary_sentence(result: CounterfactualResult) -> str:
        """Return a human-readable one-liner summarising the result.

        Examples
        --------
        >>> r = CounterfactualResult(...)
        >>> CounterfactualEvaluator.summary_sentence(r)
        'Historical data suggests GP-BO would have converged 18% faster ...'
        """
        if result.confidence < 0.3:
            return (
                f"Insufficient data to reliably compare {result.alternative_backend} "
                f"against {result.baseline_backend} (confidence {result.confidence:.0%})."
            )

        speedup_pct = result.estimated_speedup * 100

        if abs(speedup_pct) < 1.0:
            comparison = "performed similarly to"
        elif speedup_pct > 0:
            comparison = f"converged {speedup_pct:.0f}% faster than"
        else:
            comparison = f"converged {abs(speedup_pct):.0f}% slower than"

        return (
            f"Historical data suggests {result.alternative_backend} would have "
            f"{comparison} {result.baseline_backend} for this problem "
            f"(confidence {result.confidence:.0%})."
        )

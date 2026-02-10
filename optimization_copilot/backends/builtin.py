"""Built-in optimization backends using only the Python standard library."""

from __future__ import annotations

import random
from typing import Any

from optimization_copilot.core.models import Observation, ParameterSpec, VariableType
from optimization_copilot.plugins.base import AlgorithmPlugin


# ── helpers ───────────────────────────────────────────────────────────

def _sample_param(spec: ParameterSpec, rng: random.Random) -> Any:
    """Draw one random value for *spec* using the given RNG."""
    if spec.type == VariableType.CATEGORICAL:
        return rng.choice(spec.categories)
    if spec.type == VariableType.DISCRETE:
        return rng.randint(int(spec.lower), int(spec.upper))
    # CONTINUOUS (and fallback)
    return rng.uniform(spec.lower, spec.upper)


def _clamp(value: float, spec: ParameterSpec) -> Any:
    """Clamp *value* to the bounds of *spec*."""
    if spec.type == VariableType.CATEGORICAL:
        return value  # no numeric bounds
    if spec.type == VariableType.DISCRETE:
        return max(int(spec.lower), min(int(spec.upper), int(round(value))))
    return max(spec.lower, min(spec.upper, value))


# ── RandomSampler ─────────────────────────────────────────────────────

class RandomSampler(AlgorithmPlugin):
    """Uniform random sampling within parameter bounds.

    Useful as a baseline and during cold-start phases when no prior
    observations are available.
    """

    def __init__(self) -> None:
        self._specs: list[ParameterSpec] = []

    def name(self) -> str:
        return "random_sampler"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = list(parameter_specs)

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        suggestions: list[dict[str, Any]] = []
        for _ in range(n_suggestions):
            point = {spec.name: _sample_param(spec, rng) for spec in self._specs}
            suggestions.append(point)
        return suggestions

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": False,
            "max_dimensions": None,
        }


# ── LatinHypercubeSampler ─────────────────────────────────────────────

class LatinHypercubeSampler(AlgorithmPlugin):
    """Latin Hypercube Sampling (LHS) for space-filling experimental design.

    Divides each dimension into *n* equal strata and places exactly one
    sample in each stratum.  Strata assignments are shuffled independently
    per dimension.  For categorical parameters each category is assigned
    to strata as evenly as possible.
    """

    def __init__(self) -> None:
        self._specs: list[ParameterSpec] = []

    def name(self) -> str:
        return "latin_hypercube_sampler"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = list(parameter_specs)

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        n = n_suggestions

        # For each spec, build a column of n values (one per stratum).
        columns: dict[str, list[Any]] = {}
        for spec in self._specs:
            if spec.type == VariableType.CATEGORICAL:
                # Distribute categories across strata as evenly as possible.
                cats = spec.categories
                col = [cats[i % len(cats)] for i in range(n)]
                rng.shuffle(col)
            elif spec.type == VariableType.DISCRETE:
                lo, hi = int(spec.lower), int(spec.upper)
                # Strata boundaries in continuous space, then round.
                width = (hi - lo + 1) / n
                col: list[Any] = []
                indices = list(range(n))
                rng.shuffle(indices)
                for idx in indices:
                    low_edge = lo + idx * width
                    high_edge = lo + (idx + 1) * width
                    val = rng.uniform(low_edge, high_edge)
                    col.append(max(lo, min(hi, int(round(val)))))
                columns[spec.name] = col
                continue
            else:
                # Continuous: divide [lower, upper] into n equal strata.
                lo, hi = spec.lower, spec.upper
                width = (hi - lo) / n
                col = []
                indices = list(range(n))
                rng.shuffle(indices)
                for idx in indices:
                    low_edge = lo + idx * width
                    high_edge = lo + (idx + 1) * width
                    col.append(rng.uniform(low_edge, high_edge))
            columns[spec.name] = col

        # Assemble per-sample dicts.
        return [
            {spec.name: columns[spec.name][i] for spec in self._specs}
            for i in range(n)
        ]

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": False,
            "max_dimensions": None,
        }


# ── TPESampler ────────────────────────────────────────────────────────

class TPESampler(AlgorithmPlugin):
    """Simplified Tree-structured Parzen Estimator (TPE).

    Splits historical observations into *good* (top percentile) and *bad*
    groups, then samples new candidates from the good region with small
    jitter.  Falls back to uniform random when insufficient observations
    are available.
    """

    def __init__(self, gamma: float = 0.25) -> None:
        self._specs: list[ParameterSpec] = []
        self._observations: list[Observation] = []
        self._gamma = gamma  # fraction considered "good"

    def name(self) -> str:
        return "tpe_sampler"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = list(parameter_specs)
        self._observations = [o for o in observations if not o.is_failure]

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)

        # Not enough observations — fall back to uniform random.
        if len(self._observations) < 4:
            return [
                {s.name: _sample_param(s, rng) for s in self._specs}
                for _ in range(n_suggestions)
            ]

        # Rank observations by their first KPI value (lower is better for
        # a simplified implementation; the meta-controller handles direction).
        sorted_obs = sorted(
            self._observations,
            key=lambda o: list(o.kpi_values.values())[0],
        )
        n_good = max(1, int(len(sorted_obs) * self._gamma))
        good_obs = sorted_obs[:n_good]

        suggestions: list[dict[str, Any]] = []
        for _ in range(n_suggestions):
            # Pick a random observation from the good set and jitter it.
            base = rng.choice(good_obs)
            point: dict[str, Any] = {}
            for spec in self._specs:
                base_val = base.parameters.get(spec.name)
                if spec.type == VariableType.CATEGORICAL:
                    # With small probability, explore a different category.
                    if rng.random() < 0.2:
                        point[spec.name] = rng.choice(spec.categories)
                    else:
                        point[spec.name] = base_val
                elif spec.type == VariableType.DISCRETE:
                    lo, hi = int(spec.lower), int(spec.upper)
                    spread = max(1.0, (hi - lo) * 0.1)
                    jittered = base_val + rng.gauss(0, spread)
                    point[spec.name] = max(lo, min(hi, int(round(jittered))))
                else:
                    # Continuous: Gaussian jitter proportional to range.
                    lo, hi = spec.lower, spec.upper
                    spread = (hi - lo) * 0.1
                    jittered = base_val + rng.gauss(0, spread)
                    point[spec.name] = max(lo, min(hi, jittered))
            suggestions.append(point)
        return suggestions

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": True,
            "max_dimensions": None,
        }

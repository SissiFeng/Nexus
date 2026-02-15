"""Expert prior injection for human-in-the-loop optimization.

Allows domain experts to express beliefs about parameter values
as probabilistic priors that influence suggestion ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from optimization_copilot.core.models import ParameterSpec


class PriorType(Enum):
    """Types of expert priors that can be placed on parameters."""

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    RANKING = "ranking"


@dataclass
class ExpertPrior:
    """A single expert prior on a named parameter.

    Attributes:
        parameter_name: Name of the parameter this prior applies to.
        prior_type: Type of prior distribution or constraint.
        mean: Center value for GAUSSIAN and PREFERENCE priors.
        std: Standard deviation for GAUSSIAN priors.
        lower: Lower bound for UNIFORM, CONSTRAINT, and PREFERENCE priors.
        upper: Upper bound for UNIFORM, CONSTRAINT, and PREFERENCE priors.
        confidence: Expert confidence in this prior, 0.0 to 1.0.
        source: Identifier of who provided this prior.
    """

    parameter_name: str
    prior_type: PriorType
    mean: float | None = None
    std: float | None = None
    lower: float | None = None
    upper: float | None = None
    confidence: float = 0.5
    source: str = ""


class PriorRegistry:
    """Registry for collecting and applying expert priors to suggestions.

    Stores priors by parameter name and provides scoring logic
    to re-rank candidate suggestions based on accumulated expert beliefs.
    """

    def __init__(self) -> None:
        self._priors: list[ExpertPrior] = []

    def add_prior(self, prior: ExpertPrior) -> None:
        """Add an expert prior to the registry."""
        self._priors.append(prior)

    def get_priors(self, parameter_name: str) -> list[ExpertPrior]:
        """Return all priors for a given parameter name."""
        return [p for p in self._priors if p.parameter_name == parameter_name]

    def all_priors(self) -> list[ExpertPrior]:
        """Return all registered priors."""
        return list(self._priors)

    @property
    def n_priors(self) -> int:
        """Number of registered priors."""
        return len(self._priors)

    def _score_single(
        self,
        prior: ExpertPrior,
        value: float,
        spec: ParameterSpec | None,
    ) -> float:
        """Score a single value against a single prior.

        Args:
            prior: The expert prior to evaluate against.
            value: The parameter value to score.
            spec: Optional parameter specification for range information.

        Returns:
            A float score (higher is better, negative for penalties).
        """
        if prior.prior_type == PriorType.GAUSSIAN:
            if prior.mean is None or prior.std is None or prior.std == 0.0:
                return 0.0
            z = (value - prior.mean) / prior.std
            return math.exp(-0.5 * z * z) * prior.confidence

        if prior.prior_type == PriorType.UNIFORM:
            lo = prior.lower
            hi = prior.upper
            if lo is None or hi is None:
                return 0.0
            if lo <= value <= hi:
                return prior.confidence
            return 0.0

        if prior.prior_type == PriorType.CONSTRAINT:
            lo = prior.lower
            hi = prior.upper
            if lo is None or hi is None:
                return 0.0
            if lo <= value <= hi:
                return prior.confidence
            return -prior.confidence

        if prior.prior_type == PriorType.PREFERENCE:
            if prior.mean is None:
                return 0.0
            # Determine range from prior bounds or spec bounds
            rng: float | None = None
            if prior.lower is not None and prior.upper is not None:
                rng = prior.upper - prior.lower
            elif spec is not None and spec.lower is not None and spec.upper is not None:
                rng = spec.upper - spec.lower
            if rng is None or rng == 0.0:
                return 0.0
            return prior.confidence * (1.0 - abs(value - prior.mean) / rng)

        # RANKING: not implemented yet
        return 0.0

    def apply_to_suggestions(
        self,
        suggestions: list[dict[str, float]],
        specs: list[ParameterSpec] | None = None,
    ) -> list[dict[str, float]]:
        """Re-rank suggestions by scoring them against all registered priors.

        Each suggestion is scored as the sum of individual prior scores
        for every matching parameter. Suggestions are returned sorted
        by descending total score.

        Args:
            suggestions: List of candidate parameter dictionaries.
            specs: Optional list of ParameterSpec for range information.

        Returns:
            The same suggestions re-ordered by descending prior score.
        """
        if not self._priors or not suggestions:
            return list(suggestions)

        # Build spec lookup
        spec_map: dict[str, ParameterSpec] = {}
        if specs:
            for s in specs:
                spec_map[s.name] = s

        scored: list[tuple[float, int, dict[str, float]]] = []
        for idx, suggestion in enumerate(suggestions):
            total = 0.0
            for param_name, value in suggestion.items():
                matching_priors = self.get_priors(param_name)
                spec = spec_map.get(param_name)
                for prior in matching_priors:
                    total += self._score_single(prior, value, spec)
            scored.append((total, idx, suggestion))

        # Sort by descending score; use original index as tiebreaker for stability
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [s for _, _, s in scored]

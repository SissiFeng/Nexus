"""Feasibility-first scoring with adaptive alpha blending.

Implements a dual-objective scorer that dynamically balances feasibility
and objective scores based on the current campaign failure rate.  Early in
a campaign (high failure rate) the scorer prioritises feasibility; as the
campaign matures it shifts weight toward objective optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimization_copilot.core.models import CampaignSnapshot
from optimization_copilot.feasibility_first.classifier import FeasibilityClassifier


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoredCandidate:
    """A candidate annotated with feasibility, objective, and combined scores."""

    parameters: dict[str, Any]
    feasibility_score: float
    objective_score: float
    combined_score: float
    alpha: float  # weight applied to feasibility
    phase: str  # "feasibility_first" or "objective_first"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class FeasibilityFirstScorer:
    """Adaptive scorer blending feasibility and objective quality.

    The blending weight *alpha* is computed from the campaign failure rate:

    * **High failure rate** -> alpha close to ``alpha_max`` (feasibility-first)
    * **Low failure rate** -> alpha close to ``alpha_min`` (objective-first)

    ``combined = alpha * feasibility_score + (1 - alpha) * objective_score``

    Parameters
    ----------
    alpha_max : float
        Maximum feasibility weight (used when all observations fail).
    alpha_min : float
        Minimum feasibility weight (used when all observations succeed).
    feasibility_threshold : float
        Reserved for future gating logic (not currently enforced).
    """

    def __init__(
        self,
        alpha_max: float = 0.9,
        alpha_min: float = 0.1,
        feasibility_threshold: float = 0.5,
    ) -> None:
        self._alpha_max = alpha_max
        self._alpha_min = alpha_min
        self._feasibility_threshold = feasibility_threshold

    def compute_alpha(self, snapshot: CampaignSnapshot) -> float:
        """Compute the adaptive blending weight.

        Parameters
        ----------
        snapshot :
            Current campaign state.

        Returns
        -------
        float
            Alpha in ``[alpha_min, alpha_max]``.
        """
        feas_rate = 1.0 - snapshot.failure_rate
        alpha = self._alpha_max * (1.0 - feas_rate) + self._alpha_min * feas_rate
        return alpha

    def score_candidates(
        self,
        candidates: list[dict[str, Any]],
        snapshot: CampaignSnapshot,
        classifier: FeasibilityClassifier,
        objective_values: list[float] | None = None,
        direction: str = "maximize",
    ) -> list[ScoredCandidate]:
        """Score and rank candidates by combined feasibility + objective.

        Parameters
        ----------
        candidates :
            Parameter dictionaries for each candidate.
        snapshot :
            Current campaign state.
        classifier :
            Feasibility classifier to use for predictions.
        objective_values :
            Optional raw objective values, one per candidate.  When
            *None* the objective component is set to 0.
        direction :
            ``"maximize"`` or ``"minimize"`` — controls objective
            normalisation direction.

        Returns
        -------
        list[ScoredCandidate]
            Candidates sorted by ``combined_score`` (descending).
        """
        alpha = self.compute_alpha(snapshot)
        phase = "feasibility_first" if alpha > 0.5 else "objective_first"

        predictions = classifier.predict_batch(candidates, snapshot)

        if objective_values is not None and len(objective_values) == len(candidates):
            norm_objectives = self._normalize_objectives(objective_values, direction)
        else:
            norm_objectives = [0.0] * len(candidates)

        scored: list[ScoredCandidate] = []
        for cand, pred, norm_obj in zip(candidates, predictions, norm_objectives):
            feasibility_score = pred.p_feasible
            objective_score = norm_obj
            combined_score = alpha * feasibility_score + (1.0 - alpha) * objective_score
            scored.append(ScoredCandidate(
                parameters=cand,
                feasibility_score=feasibility_score,
                objective_score=objective_score,
                combined_score=combined_score,
                alpha=alpha,
                phase=phase,
            ))

        scored.sort(key=lambda s: s.combined_score, reverse=True)
        return scored

    # -- Private helpers ----------------------------------------------------

    @staticmethod
    def _normalize_objectives(
        values: list[float],
        direction: str,
    ) -> list[float]:
        """Min-max normalise objective values to [0, 1].

        Parameters
        ----------
        values :
            Raw objective values.
        direction :
            ``"maximize"`` keeps original ordering; ``"minimize"``
            inverts it.

        Returns
        -------
        list[float]
        """
        if len(values) <= 1:
            return [0.5] * len(values)

        lo = min(values)
        hi = max(values)

        if hi == lo:
            return [0.5] * len(values)

        if direction == "maximize":
            return [(v - lo) / (hi - lo) for v in values]
        else:
            return [(hi - v) / (hi - lo) for v in values]

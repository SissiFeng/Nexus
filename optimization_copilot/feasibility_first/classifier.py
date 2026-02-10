"""Feasibility classifier wrapping the failure surface learner.

Translates failure-probability estimates into feasibility predictions
with confidence scores based on neighbor density.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimization_copilot.core.models import CampaignSnapshot
from optimization_copilot.feasibility.surface import FailureSurfaceLearner


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityPrediction:
    """Feasibility estimate for a single candidate point."""

    p_feasible: float  # probability of feasibility [0, 1]
    p_failure: float  # 1 - p_feasible
    n_neighbors: int  # neighbors used for estimation
    avg_distance: float  # average distance to neighbors
    confidence: float  # [0, 1] based on n_neighbors / k


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class FeasibilityClassifier:
    """Classifies candidate points as feasible / infeasible.

    Wraps :class:`FailureSurfaceLearner` to produce
    :class:`FeasibilityPrediction` objects with calibrated confidence.

    Parameters
    ----------
    k : int
        Number of neighbors for KNN estimation (passed to the learner).
    """

    def __init__(self, k: int = 5) -> None:
        self._k = k
        self._learner = FailureSurfaceLearner(k=k)

    def predict(
        self,
        candidate: dict[str, Any],
        snapshot: CampaignSnapshot,
    ) -> FeasibilityPrediction:
        """Predict feasibility for a single candidate.

        Parameters
        ----------
        candidate :
            Parameter values for the candidate point.
        snapshot :
            Campaign history to learn from.

        Returns
        -------
        FeasibilityPrediction
        """
        failure_prob = self._learner.predict_failure(candidate, snapshot)
        p_feasible = 1.0 - failure_prob.p_fail
        confidence = (
            min(1.0, failure_prob.n_neighbors / self._k)
            if self._k > 0
            else 0.0
        )
        return FeasibilityPrediction(
            p_feasible=p_feasible,
            p_failure=failure_prob.p_fail,
            n_neighbors=failure_prob.n_neighbors,
            avg_distance=failure_prob.avg_distance,
            confidence=confidence,
        )

    def predict_batch(
        self,
        candidates: list[dict[str, Any]],
        snapshot: CampaignSnapshot,
    ) -> list[FeasibilityPrediction]:
        """Predict feasibility for a batch of candidates.

        Parameters
        ----------
        candidates :
            List of parameter dictionaries.
        snapshot :
            Campaign history to learn from.

        Returns
        -------
        list[FeasibilityPrediction]
        """
        return [self.predict(c, snapshot) for c in candidates]

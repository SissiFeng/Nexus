"""Data models for preference learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PairwisePreference:
    """A single pairwise preference judgment."""

    winner_idx: int
    loser_idx: int
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PairwisePreference:
        return cls(**data)


@dataclass
class PreferenceModel:
    """Fitted preference model (Bradley-Terry scores)."""

    scores: dict[int, float]
    n_preferences: int
    n_items: int
    converged: bool
    n_iterations: int
    log_likelihood: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreferenceModel:
        data = data.copy()
        # Convert string keys back to int for scores dict
        data["scores"] = {int(k): v for k, v in data["scores"].items()}
        return cls(**data)


@dataclass
class PreferenceRanking:
    """Combined Pareto-dominance + preference ranking."""

    ranked_indices: list[int]
    utility_scores: dict[int, float]
    dominance_ranks: list[int]
    preference_within_rank: dict[int, list[int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreferenceRanking:
        data = data.copy()
        # Convert string keys back to int for utility_scores
        data["utility_scores"] = {int(k): v for k, v in data["utility_scores"].items()}
        # Convert string keys back to int for preference_within_rank
        data["preference_within_rank"] = {
            int(k): v for k, v in data["preference_within_rank"].items()
        }
        return cls(**data)

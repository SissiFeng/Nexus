"""Data models for the candidate pool module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PoolCandidate:
    """A candidate from an external pool."""

    parameters: dict[str, Any]
    score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "external"


@dataclass
class PoolVersion:
    """A scored snapshot of the candidate pool."""

    version: int
    candidates: list[PoolCandidate]
    scoring_seed: int
    scoring_snapshot_hash: str
    timestamp: float = 0.0

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    def top_n(self, n: int) -> list[PoolCandidate]:
        """Return top-n candidates by rank (ascending rank = better)."""
        sorted_candidates = sorted(self.candidates, key=lambda c: c.rank)
        return sorted_candidates[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "n_candidates": self.n_candidates,
            "scoring_seed": self.scoring_seed,
            "scoring_snapshot_hash": self.scoring_snapshot_hash,
            "timestamp": self.timestamp,
            "candidates": [
                {
                    "parameters": c.parameters,
                    "score": c.score,
                    "rank": c.rank,
                    "source": c.source,
                    "metadata": c.metadata,
                }
                for c in self.candidates
            ],
        }

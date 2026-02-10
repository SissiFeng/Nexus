"""Algorithm Portfolio Learning module.

Tracks per-backend performance across different ProblemFingerprints,
enabling informed backend selection based on historical outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import ProblemFingerprint


def _fingerprint_key(fp: ProblemFingerprint) -> str:
    """Serialize a ProblemFingerprint to a stable string key."""
    return str(fp.to_tuple())


@dataclass
class BackendRecord:
    """Performance record for a single backend on a specific fingerprint."""

    fingerprint_key: str
    backend_name: str
    n_uses: int = 0
    win_count: int = 0
    avg_convergence_speed: float = 0.0
    avg_regret: float = 0.0
    failure_rate: float = 0.0
    sample_efficiency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint_key": self.fingerprint_key,
            "backend_name": self.backend_name,
            "n_uses": self.n_uses,
            "win_count": self.win_count,
            "avg_convergence_speed": self.avg_convergence_speed,
            "avg_regret": self.avg_regret,
            "failure_rate": self.failure_rate,
            "sample_efficiency": self.sample_efficiency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackendRecord:
        return cls(**data)


@dataclass
class PortfolioStats:
    """Ranking and scoring of backends for a given problem context."""

    portfolio_rank: list[str] = field(default_factory=list)
    expected_gain: dict[str, float] = field(default_factory=dict)
    risk_penalty: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)


class AlgorithmPortfolio:
    """Tracks per-backend performance and ranks backends for new problems.

    The portfolio accumulates outcome records keyed by
    (fingerprint_key, backend_name) and uses them to rank available
    backends when a new optimization run is being planned.
    """

    def __init__(self) -> None:
        # Keyed by (fingerprint_key, backend_name) -> BackendRecord
        self._records: dict[tuple[str, str], BackendRecord] = {}

    # ── Recording ────────────────────────────────────────

    def record_outcome(
        self,
        fingerprint: ProblemFingerprint,
        backend_name: str,
        outcome: dict[str, Any],
    ) -> None:
        """Update statistics for a backend on a specific fingerprint.

        Parameters
        ----------
        fingerprint:
            The problem fingerprint for this campaign.
        backend_name:
            Name of the backend that was used.
        outcome:
            Dictionary with keys: convergence_speed, regret,
            failure_rate, sample_efficiency, is_winner.
        """
        fp_key = _fingerprint_key(fingerprint)
        rec_key = (fp_key, backend_name)

        if rec_key not in self._records:
            self._records[rec_key] = BackendRecord(
                fingerprint_key=fp_key,
                backend_name=backend_name,
            )

        rec = self._records[rec_key]
        n = rec.n_uses

        # Incremental mean update for each metric.
        rec.avg_convergence_speed = _incremental_mean(
            rec.avg_convergence_speed, n, outcome.get("convergence_speed", 0.0)
        )
        rec.avg_regret = _incremental_mean(
            rec.avg_regret, n, outcome.get("regret", 0.0)
        )
        rec.failure_rate = _incremental_mean(
            rec.failure_rate, n, outcome.get("failure_rate", 0.0)
        )
        rec.sample_efficiency = _incremental_mean(
            rec.sample_efficiency, n, outcome.get("sample_efficiency", 0.0)
        )

        rec.n_uses = n + 1
        if outcome.get("is_winner", False):
            rec.win_count += 1

    # ── Ranking ──────────────────────────────────────────

    def rank_backends(
        self,
        fingerprint: ProblemFingerprint,
        available: list[str],
    ) -> PortfolioStats:
        """Rank available backends for the given fingerprint.

        For a known fingerprint, ranking is based on:
            score = win_rate * confidence - risk_penalty

        For an unknown fingerprint, aggregate across all fingerprints
        with a decay factor (records from other fingerprints are
        weighted at 0.3).

        Parameters
        ----------
        fingerprint:
            The problem fingerprint to rank backends for.
        available:
            List of backend names to consider.

        Returns
        -------
        PortfolioStats with rankings, expected gains, risk penalties,
        and confidence estimates.
        """
        fp_key = _fingerprint_key(fingerprint)
        stats = PortfolioStats()

        scores: dict[str, float] = {}

        for name in available:
            exact_key = (fp_key, name)
            if exact_key in self._records:
                rec = self._records[exact_key]
                confidence = min(1.0, rec.n_uses / 10.0)
                win_rate = rec.win_count / rec.n_uses if rec.n_uses > 0 else 0.0
                risk = rec.failure_rate + rec.avg_regret
                score = win_rate * confidence - risk

                stats.expected_gain[name] = win_rate
                stats.risk_penalty[name] = risk
                stats.confidence[name] = confidence
                scores[name] = score
            else:
                # Fallback: aggregate across all fingerprints with decay.
                agg = self._aggregate_for_backend(name, decay=0.3)
                if agg is not None:
                    confidence = min(1.0, agg["n_uses"] / 10.0)
                    win_rate = agg["win_rate"]
                    risk = agg["failure_rate"] + agg["avg_regret"]
                    score = win_rate * confidence - risk

                    stats.expected_gain[name] = win_rate
                    stats.risk_penalty[name] = risk
                    stats.confidence[name] = confidence * 0.3  # reduced for cross-fp
                    scores[name] = score
                else:
                    # No data at all -- neutral score.
                    stats.expected_gain[name] = 0.0
                    stats.risk_penalty[name] = 0.0
                    stats.confidence[name] = 0.0
                    scores[name] = 0.0

        # Sort by score descending.
        ranked = sorted(scores, key=lambda b: scores[b], reverse=True)
        stats.portfolio_rank = ranked
        return stats

    # ── Aggregation helper ───────────────────────────────

    def _aggregate_for_backend(
        self, backend_name: str, decay: float = 0.3
    ) -> dict[str, float] | None:
        """Aggregate statistics for a backend across all fingerprints.

        Each record is weighted by ``decay`` since it comes from a
        different fingerprint than the one being queried.
        """
        total_uses = 0
        total_wins = 0.0
        total_failure = 0.0
        total_regret = 0.0
        count = 0

        for (_, bname), rec in self._records.items():
            if bname != backend_name:
                continue
            total_uses += rec.n_uses
            total_wins += rec.win_count
            total_failure += rec.failure_rate * rec.n_uses
            total_regret += rec.avg_regret * rec.n_uses
            count += 1

        if count == 0 or total_uses == 0:
            return None

        return {
            "n_uses": total_uses,
            "win_rate": (total_wins / total_uses) * decay,
            "failure_rate": (total_failure / total_uses) * decay,
            "avg_regret": (total_regret / total_uses) * decay,
        }

    # ── Serialization ────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the portfolio to a plain dict."""
        return {
            "records": [rec.to_dict() for rec in self._records.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlgorithmPortfolio:
        """Deserialize a portfolio from a plain dict."""
        portfolio = cls()
        for rec_data in data.get("records", []):
            rec = BackendRecord.from_dict(rec_data)
            key = (rec.fingerprint_key, rec.backend_name)
            portfolio._records[key] = rec
        return portfolio

    # ── Merging ──────────────────────────────────────────

    def merge(self, other: AlgorithmPortfolio) -> None:
        """Merge another portfolio into this one.

        For records that exist in both portfolios, statistics are
        combined using weighted averages based on ``n_uses``.
        """
        for key, other_rec in other._records.items():
            if key not in self._records:
                # Deep-copy via dict round-trip.
                self._records[key] = BackendRecord.from_dict(other_rec.to_dict())
            else:
                self_rec = self._records[key]
                n_self = self_rec.n_uses
                n_other = other_rec.n_uses
                n_total = n_self + n_other

                if n_total == 0:
                    continue

                self_rec.avg_convergence_speed = _weighted_mean(
                    self_rec.avg_convergence_speed,
                    n_self,
                    other_rec.avg_convergence_speed,
                    n_other,
                )
                self_rec.avg_regret = _weighted_mean(
                    self_rec.avg_regret,
                    n_self,
                    other_rec.avg_regret,
                    n_other,
                )
                self_rec.failure_rate = _weighted_mean(
                    self_rec.failure_rate,
                    n_self,
                    other_rec.failure_rate,
                    n_other,
                )
                self_rec.sample_efficiency = _weighted_mean(
                    self_rec.sample_efficiency,
                    n_self,
                    other_rec.sample_efficiency,
                    n_other,
                )
                self_rec.n_uses = n_total
                self_rec.win_count = self_rec.win_count + other_rec.win_count


# ── Utility functions ────────────────────────────────────


def _incremental_mean(old_mean: float, old_count: int, new_value: float) -> float:
    """Compute the new mean after adding one observation."""
    return (old_mean * old_count + new_value) / (old_count + 1)


def _weighted_mean(
    mean_a: float, n_a: int, mean_b: float, n_b: int
) -> float:
    """Compute the weighted mean of two populations."""
    total = n_a + n_b
    if total == 0:
        return 0.0
    return (mean_a * n_a + mean_b * n_b) / total

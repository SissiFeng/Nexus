"""Drift-robustness tracker — ranks backends by resilience under drift conditions.

Tracks per-backend regret with and without drift, computing a resilience
score that indicates how well each backend maintains performance when the
objective landscape shifts during a campaign.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from optimization_copilot.meta_learning.models import (
    CampaignOutcome,
    DriftRobustness,
    MetaLearningConfig,
)
from optimization_copilot.meta_learning.experience_store import ExperienceStore


class DriftRobustnessTracker:
    """Learns which backends best maintain performance under drift.

    Accumulates per-backend regret observations under drift and non-drift
    conditions, then computes a resilience score in [0, 1] where higher
    values indicate stronger drift robustness.
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        config: MetaLearningConfig | None = None,
    ) -> None:
        self._experience_store = experience_store
        self._config = config or MetaLearningConfig()
        # backend -> [(regret_under_drift, drift_score)]
        self._backend_drift_data: dict[str, list[tuple[float, float]]] = defaultdict(
            list
        )
        # backend -> [regret_without_drift]
        self._backend_nodrift_data: dict[str, list[float]] = defaultdict(list)
        self._robustness: dict[str, DriftRobustness] = {}

    # ── Queries ────────────────────────────────────────────

    def rank_by_resilience(self) -> list[DriftRobustness]:
        """Return all tracked backends sorted by drift resilience (best first)."""
        return sorted(
            self._robustness.values(),
            key=lambda r: r.drift_resilience_score,
            reverse=True,
        )

    def get_resilience(self, backend_name: str) -> DriftRobustness | None:
        """Return the drift robustness entry for a specific backend, or None."""
        return self._robustness.get(backend_name)

    # ── Learning ───────────────────────────────────────────

    def update_from_outcome(self, outcome: CampaignOutcome) -> None:
        """Integrate a completed campaign's backend performances.

        Separates drift vs. non-drift observations for each backend,
        then recomputes the robustness score for every affected backend.
        """
        affected_backends: set[str] = set()

        for bp in outcome.backend_performances:
            affected_backends.add(bp.backend_name)
            if bp.drift_encountered:
                self._backend_drift_data[bp.backend_name].append(
                    (bp.regret, bp.drift_score)
                )
            else:
                self._backend_nodrift_data[bp.backend_name].append(bp.regret)

        # Recompute robustness only for backends touched by this outcome.
        for backend_name in affected_backends:
            self._recompute_robustness(backend_name)

    # ── Internal ───────────────────────────────────────────

    def _recompute_robustness(self, backend_name: str) -> None:
        """Recompute drift resilience for a single backend."""
        drift_data = self._backend_drift_data.get(backend_name, [])
        nodrift_data = self._backend_nodrift_data.get(backend_name, [])

        if not drift_data:
            # No drift observations yet — cannot compute resilience.
            return

        avg_drift_regret = sum(r for r, _ in drift_data) / len(drift_data)

        if nodrift_data:
            avg_nodrift_regret = sum(nodrift_data) / len(nodrift_data)
        else:
            avg_nodrift_regret = 0.0

        kpi_loss = max(0.0, avg_drift_regret - avg_nodrift_regret)
        resilience = max(0.0, 1.0 - kpi_loss)

        self._robustness[backend_name] = DriftRobustness(
            backend_name=backend_name,
            drift_resilience_score=resilience,
            n_drift_campaigns=len(drift_data),
            avg_kpi_loss_under_drift=kpi_loss,
        )

    # ── Serialization ──────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state to a plain dict."""
        return {
            "backend_drift_data": {
                name: [[r, d] for r, d in pairs]
                for name, pairs in self._backend_drift_data.items()
            },
            "backend_nodrift_data": {
                name: list(regrets)
                for name, regrets in self._backend_nodrift_data.items()
            },
            "robustness": {
                name: rob.to_dict() for name, rob in self._robustness.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        experience_store: ExperienceStore,
    ) -> DriftRobustnessTracker:
        """Reconstruct a tracker from a serialized dict."""
        tracker = cls(experience_store=experience_store)

        # Restore drift data.
        for name, pairs in data.get("backend_drift_data", {}).items():
            tracker._backend_drift_data[name] = [
                (float(r), float(d)) for r, d in pairs
            ]

        # Restore non-drift data.
        for name, regrets in data.get("backend_nodrift_data", {}).items():
            tracker._backend_nodrift_data[name] = [float(r) for r in regrets]

        # Restore robustness entries.
        for name, rob_data in data.get("robustness", {}).items():
            tracker._robustness[name] = DriftRobustness.from_dict(rob_data)

        return tracker

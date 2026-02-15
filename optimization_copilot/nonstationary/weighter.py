"""Time-aware observation weighting for non-stationary optimization campaigns.

Supports exponential decay, sliding window, and linear decay strategies
to down-weight stale observations when the underlying process drifts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import CampaignSnapshot, Observation


# ── Data Structures ───────────────────────────────────────


@dataclass
class TimeWeights:
    """Result of a time-weighting computation."""

    weights: dict[int, float]  # observation iteration -> weight
    effective_window: int  # number of observations with weight > 0.01
    decay_rate: float
    strategy: str


# ── Time Weighter ─────────────────────────────────────────


class TimeWeighter:
    """Compute recency-based weights for campaign observations.

    Parameters
    ----------
    strategy : str
        One of ``"exponential"``, ``"sliding_window"``, ``"linear_decay"``.
    decay_rate : float
        Controls how quickly older observations are down-weighted.
        Interpretation depends on the strategy.
    window_size : int | None
        Used only for the ``"sliding_window"`` strategy.  Observations
        within the last *window_size* (by iteration order) receive
        weight 1.0; older observations receive 0.0.
    """

    _VALID_STRATEGIES = {"exponential", "sliding_window", "linear_decay"}

    def __init__(
        self,
        strategy: str = "exponential",
        decay_rate: float = 0.1,
        window_size: int | None = None,
    ) -> None:
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Choose from {sorted(self._VALID_STRATEGIES)}."
            )
        self.strategy = strategy
        self.decay_rate = decay_rate
        self.window_size = window_size

    # ── Public API ────────────────────────────────────────

    def compute_weights(self, snapshot: CampaignSnapshot) -> TimeWeights:
        """Compute per-observation weights based on the selected strategy.

        Returns a :class:`TimeWeights` with ``weights`` mapping each
        observation's iteration to its normalised weight (max = 1.0).
        """
        observations = snapshot.observations
        if not observations:
            return TimeWeights(
                weights={},
                effective_window=0,
                decay_rate=self.decay_rate,
                strategy=self.strategy,
            )

        # Extract timestamps; fall back to iteration index when all are zero.
        timestamps = [obs.timestamp for obs in observations]
        all_zero = all(t == 0.0 for t in timestamps)
        if all_zero:
            timestamps = [float(obs.iteration) for obs in observations]

        iterations = [obs.iteration for obs in observations]

        # Compute raw weights using the selected strategy.
        if self.strategy == "exponential":
            t_max = max(timestamps)
            raw = self._exponential_decay(timestamps, t_max)
        elif self.strategy == "sliding_window":
            ws = self.window_size if self.window_size is not None else len(iterations)
            raw = self._sliding_window(iterations, ws)
        elif self.strategy == "linear_decay":
            t_max = max(timestamps)
            raw = self._linear_decay(timestamps, t_max)
        else:
            # Defensive fallback — should not be reachable.
            raw = [1.0] * len(observations)

        # Normalise so that the maximum weight equals 1.0.
        max_w = max(raw) if raw else 1.0
        if max_w > 0.0:
            normalised = [w / max_w for w in raw]
        else:
            normalised = [1.0] * len(raw)

        weights = {obs.iteration: w for obs, w in zip(observations, normalised)}
        effective_window = sum(1 for w in normalised if w > 0.01)

        return TimeWeights(
            weights=weights,
            effective_window=effective_window,
            decay_rate=self.decay_rate,
            strategy=self.strategy,
        )

    def weight_observations(self, snapshot: CampaignSnapshot) -> list[Observation]:
        """Return shallow copies of observations with ``metadata["time_weight"]`` set.

        The original observations are never mutated.
        """
        tw = self.compute_weights(snapshot)
        weighted: list[Observation] = []
        for obs in snapshot.observations:
            new_meta = dict(obs.metadata)
            new_meta["time_weight"] = tw.weights.get(obs.iteration, 0.0)
            weighted.append(
                Observation(
                    iteration=obs.iteration,
                    parameters=obs.parameters,
                    kpi_values=obs.kpi_values,
                    qc_passed=obs.qc_passed,
                    is_failure=obs.is_failure,
                    failure_reason=obs.failure_reason,
                    timestamp=obs.timestamp,
                    metadata=new_meta,
                )
            )
        return weighted

    # ── Private strategy implementations ──────────────────

    def _exponential_decay(
        self, timestamps: list[float], t_max: float
    ) -> list[float]:
        """w_i = exp(-decay_rate * (t_max - t_i))"""
        return [math.exp(-self.decay_rate * (t_max - t)) for t in timestamps]

    def _sliding_window(
        self, iterations: list[int], window_size: int
    ) -> list[float]:
        """1.0 for the last *window_size* observations (by iteration order), 0.0 otherwise."""
        n = len(iterations)
        if window_size >= n:
            return [1.0] * n

        # Determine the iteration-order cutoff: keep the last window_size entries.
        sorted_indices = sorted(range(n), key=lambda i: iterations[i])
        keep_set = set(sorted_indices[n - window_size:])
        return [1.0 if i in keep_set else 0.0 for i in range(n)]

    def _linear_decay(
        self, timestamps: list[float], t_max: float
    ) -> list[float]:
        """w_i = max(0.0, 1.0 - decay_rate * (t_max - t_i))"""
        return [max(0.0, 1.0 - self.decay_rate * (t_max - t)) for t in timestamps]

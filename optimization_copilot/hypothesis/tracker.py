"""Hypothesis lifecycle management and experiment suggestion.

Provides the :class:`HypothesisTracker` which manages hypothesis
registration, status transitions, evidence collection, discriminating
experiment design, and serialization.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

from optimization_copilot.hypothesis.models import (
    Evidence,
    Hypothesis,
    HypothesisStatus,
    Prediction,
)
from optimization_copilot.hypothesis.testing import HypothesisTester


class HypothesisTracker:
    """Manage hypothesis lifecycle and suggest discriminating experiments.

    Maintains a registry of hypotheses, an audit trail of status changes,
    and integrates with :class:`HypothesisTester` for evidence evaluation.
    """

    def __init__(self) -> None:
        self.hypotheses: dict[str, Hypothesis] = {}
        self.history: list[dict[str, Any]] = []
        self._tester = HypothesisTester()

    # -- Registration ----------------------------------------------------------

    def add(self, hypothesis: Hypothesis) -> None:
        """Register a new hypothesis."""
        self.hypotheses[hypothesis.id] = hypothesis
        self._record("added", hypothesis.id)

    def get(self, hypothesis_id: str) -> Hypothesis | None:
        """Return the hypothesis with *hypothesis_id*, or ``None``."""
        return self.hypotheses.get(hypothesis_id)

    # -- Status management -----------------------------------------------------

    def update_status(
        self, hypothesis_id: str, new_status: HypothesisStatus
    ) -> None:
        """Transition a hypothesis to *new_status*."""
        h = self.hypotheses.get(hypothesis_id)
        if h is None:
            raise KeyError(f"Hypothesis {hypothesis_id} not found")
        old_status = h.status
        h.status = new_status
        self._record(
            "status_change",
            hypothesis_id,
            old_status=old_status.value,
            new_status=new_status.value,
        )

    # -- Evidence collection ---------------------------------------------------

    def update_with_observation(
        self,
        observation: dict[str, float],
        var_names: list[str] | None = None,
    ) -> None:
        """Check each active hypothesis's predictions against new data.

        For hypotheses with equations, evaluates the equation at the
        observation inputs and compares to the observed target.

        Parameters
        ----------
        observation : dict[str, float]
            Must contain ``"y"`` (observed target) and feature values
            keyed by name (e.g. ``"x0"``, ``"x1"``).
        var_names : list[str] | None
            Ordered variable names to build the input vector.
        """
        observed_y = observation.get("y", 0.0)

        for h in self.get_active_hypotheses():
            if h.equation is None:
                continue

            # Build the input row from observation
            if var_names:
                x_row = [observation.get(v, 0.0) for v in var_names]
            else:
                # Auto-detect x0, x1, ... keys
                x_keys = sorted(
                    [k for k in observation if k.startswith("x")],
                    key=lambda k: int(k[1:]) if k[1:].isdigit() else 0,
                )
                x_row = [observation[k] for k in x_keys]

            self._tester.sequential_update(h, x_row, observed_y, var_names)

    # -- Experiment suggestion -------------------------------------------------

    def suggest_discriminating_experiment(
        self,
        h1_id: str,
        h2_id: str,
        parameter_ranges: dict[str, tuple[float, float]],
        n_candidates: int = 20,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Find parameter settings where *h1* and *h2* predict most differently.

        Samples *n_candidates* random points from *parameter_ranges*,
        evaluates both hypotheses, and returns the point with maximum
        ``|pred_h1 - pred_h2|``.

        Parameters
        ----------
        h1_id, h2_id : str
            Hypothesis identifiers.
        parameter_ranges : dict[str, tuple[float, float]]
            ``{param_name: (low, high)}``.
        n_candidates : int
            Number of random candidate points.
        seed : int
            Random seed.

        Returns
        -------
        dict
            ``{"point": {param: value}, "h1_pred": float, "h2_pred": float,
            "divergence": float}``.
        """
        h1 = self.hypotheses.get(h1_id)
        h2 = self.hypotheses.get(h2_id)
        if h1 is None or h2 is None:
            raise KeyError("One or both hypotheses not found")

        rng = random.Random(seed)
        param_names = sorted(parameter_ranges.keys())

        best_point: dict[str, float] = {}
        best_div = -1.0
        best_p1 = 0.0
        best_p2 = 0.0

        for _ in range(n_candidates):
            point: dict[str, float] = {}
            for name in param_names:
                lo, hi = parameter_ranges[name]
                point[name] = rng.uniform(lo, hi)

            x_row = [point[name] for name in param_names]

            p1 = self._tester._evaluate_equation(
                h1.equation or "0", x_row, param_names
            )
            p2 = self._tester._evaluate_equation(
                h2.equation or "0", x_row, param_names
            )

            if not (math.isfinite(p1) and math.isfinite(p2)):
                continue

            div = abs(p1 - p2)
            if div > best_div:
                best_div = div
                best_point = dict(point)
                best_p1 = p1
                best_p2 = p2

        return {
            "point": best_point,
            "h1_pred": best_p1,
            "h2_pred": best_p2,
            "divergence": best_div,
        }

    # -- Reporting -------------------------------------------------------------

    def get_status_report(self) -> dict[str, Any]:
        """Summary of hypothesis counts by status and top hypotheses."""
        counts: dict[str, int] = {}
        for status in HypothesisStatus:
            counts[status.value] = 0
        for h in self.hypotheses.values():
            counts[h.status.value] = counts.get(h.status.value, 0) + 1

        # Rank by evidence ratio (descending)
        ranked = sorted(
            self.hypotheses.values(),
            key=lambda h: h.evidence_ratio(),
            reverse=True,
        )
        top = [
            {
                "id": h.id,
                "description": h.description,
                "status": h.status.value,
                "evidence_ratio": h.evidence_ratio(),
                "support": h.support_count,
                "refute": h.refute_count,
            }
            for h in ranked[:5]
        ]

        return {
            "total": len(self.hypotheses),
            "counts_by_status": counts,
            "top_hypotheses": top,
        }

    def get_active_hypotheses(self) -> list[Hypothesis]:
        """Return hypotheses that are PROPOSED or TESTING."""
        return [
            h
            for h in self.hypotheses.values()
            if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING)
        ]

    # -- Audit trail -----------------------------------------------------------

    def _record(
        self, action: str, hypothesis_id: str, **kwargs: Any
    ) -> None:
        entry: dict[str, Any] = {
            "action": action,
            "hypothesis_id": hypothesis_id,
            "timestamp": time.time(),
        }
        entry.update(kwargs)
        self.history.append(entry)

    # -- Serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tracker to a plain dict."""
        return {
            "hypotheses": {
                hid: h.to_dict() for hid, h in self.hypotheses.items()
            },
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HypothesisTracker:
        """Deserialize a tracker from a plain dict."""
        tracker = cls()
        for hid, hdata in d.get("hypotheses", {}).items():
            tracker.hypotheses[hid] = Hypothesis.from_dict(hdata)
        tracker.history = d.get("history", [])
        return tracker

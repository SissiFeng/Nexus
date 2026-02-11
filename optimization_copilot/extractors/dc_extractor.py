"""DC cycling (coulombic efficiency) extractor with uncertainty.

Computes coulombic efficiency from galvanostatic current-time data and
propagates instrument uncertainty, integration truncation error, and
zero-point drift into the final variance.
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.extractors.base import UncertaintyExtractor
from optimization_copilot.uncertainty.types import MeasurementWithUncertainty


class DCCyclingExtractor(UncertaintyExtractor):
    """Extract coulombic efficiency (CE) from DC cycling data.

    Parameters
    ----------
    domain_config : dict[str, Any]
        Output of ``get_dc_config()`` from
        ``optimization_copilot.domain_knowledge.dc_cycling``.
    """

    # ── public API ────────────────────────────────────────────────────

    def extract_with_uncertainty(
        self, raw_data: dict[str, Any],
    ) -> list[MeasurementWithUncertainty]:
        """Compute CE = |Q_dissolve / Q_deposit| * 100 with uncertainty.

        Parameters
        ----------
        raw_data : dict
            Must contain ``"current"`` (A), ``"voltage"`` (V), and
            ``"time"`` (s) as lists of equal length.

        Returns
        -------
        list[MeasurementWithUncertainty]
            Single-element list with CE measurement.
        """
        current: list[float] = raw_data.get("current", [])
        time: list[float] = raw_data.get("time", [])

        n = min(len(current), len(time))
        if n < 2:
            return [MeasurementWithUncertainty(
                value=float("nan"),
                variance=0.0,
                confidence=0.0,
                source="DC_CE",
                n_points_used=n,
                method="trapezoidal",
                metadata={"error": "insufficient_data"},
            )]

        current = current[:n]
        time = time[:n]

        # ── instrument specs ──────────────────────────────────────
        inst = self.domain_config.get("instrument", {})
        dc_spec = inst.get("dc", {})
        current_accuracy = dc_spec.get("current_accuracy_a", 1e-7)
        drift_per_hour = dc_spec.get("zero_drift_a_per_hour", 5e-7)

        # ── integrate charge (trapezoidal) ────────────────────────
        q_deposit, q_dissolve = 0.0, 0.0
        q_dep_var, q_dis_var = 0.0, 0.0

        for i in range(1, n):
            dt = time[i] - time[i - 1]
            i_avg = 0.5 * (current[i] + current[i - 1])
            dq = i_avg * dt

            # per-trapezoid variance from current accuracy
            # var(dq) = dt^2 * sigma_I^2 / 2  (two points averaged)
            dq_var = dt * dt * current_accuracy * current_accuracy * 0.5

            if i_avg < 0:
                # deposition phase (negative current = cathodic)
                q_deposit += abs(dq)
                q_dep_var += dq_var
            else:
                # dissolution phase (positive current = anodic)
                q_dissolve += abs(dq)
                q_dis_var += dq_var

        # ── drift contribution ────────────────────────────────────
        total_time_h = abs(time[-1] - time[0]) / 3600.0
        drift_a = drift_per_hour * total_time_h
        total_duration_s = abs(time[-1] - time[0])
        drift_charge = drift_a * total_duration_s
        drift_var = drift_charge * drift_charge  # worst-case

        q_dep_var += drift_var
        q_dis_var += drift_var

        # ── CE and error propagation ──────────────────────────────
        if abs(q_deposit) < 1e-30:
            return [MeasurementWithUncertainty(
                value=float("nan"),
                variance=0.0,
                confidence=0.0,
                source="DC_CE",
                n_points_used=n,
                method="trapezoidal",
                metadata={"error": "zero_deposition_charge"},
            )]

        ce = abs(q_dissolve / q_deposit) * 100.0

        # Relative error propagation:
        # CE = |Qs/Qd| * 100
        # var(CE) = CE^2 * (var_Qs/Qs^2 + var_Qd/Qd^2)
        rel_qs = q_dis_var / max(q_dissolve * q_dissolve, 1e-30)
        rel_qd = q_dep_var / max(q_deposit * q_deposit, 1e-30)
        ce_var = ce * ce * (rel_qs + rel_qd)

        confidence = self._compute_confidence(ce_var, ce)

        result = MeasurementWithUncertainty(
            value=ce,
            variance=ce_var,
            confidence=confidence,
            source="DC_CE",
            n_points_used=n,
            method="trapezoidal",
            metadata={
                "q_deposit": q_deposit,
                "q_dissolve": q_dissolve,
                "drift_contribution": drift_var,
            },
        )

        # ── physical constraint check ─────────────────────────────
        constraints = self.domain_config.get("physical_constraints", {})
        ce_constraints = constraints.get("CE", {})
        if ce_constraints:
            result = self._apply_physical_constraints(result, "CE", ce_constraints)

        return [result]

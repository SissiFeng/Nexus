"""Layer 1: Raw signal-level anomaly checks.

Provides domain-specific sanity checks on raw experimental data before
KPI extraction.  Each check returns a ``SignalAnomaly`` when an issue
is found or ``None`` when the signal is clean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Data types ─────────────────────────────────────────────────────────


@dataclass
class SignalAnomaly:
    """Description of a signal-level anomaly detected in raw data."""

    check_name: str
    severity: str  # "warning", "error"
    message: str
    affected_indices: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


# ── SignalChecker ──────────────────────────────────────────────────────


class SignalChecker:
    """Run domain-specific raw-signal anomaly checks.

    All checks are static — no fitting or GP involved.  They operate on
    the raw measurement vectors (voltages, impedances, absorbance, etc.)
    and flag obvious instrument or measurement artefacts.
    """

    # ── Individual checks ──────────────────────────────────────────

    @staticmethod
    def check_eis_consistency(
        z_real: list[float],
        z_imag: list[float],
    ) -> SignalAnomaly | None:
        """Simplified Kramers-Kronig consistency check for EIS data.

        Verifies that impedance magnitude is monotonically decreasing with
        frequency (as a simplified KK consistency proxy).  Data is assumed
        to be ordered from high to low frequency, so impedance magnitude
        should generally increase.  We check for *decreasing* consecutive
        pairs (which would violate monotonicity of the expected trend).

        Returns an anomaly if > 20 % of consecutive pairs violate the
        monotonicity expectation.
        """
        n = min(len(z_real), len(z_imag))
        if n < 2:
            return None

        magnitudes = [
            math.sqrt(z_real[i] ** 2 + z_imag[i] ** 2) for i in range(n)
        ]

        violations: list[int] = []
        for i in range(n - 1):
            # Expect magnitude[i] <= magnitude[i+1] (increasing with index
            # = decreasing frequency).  A violation is when it decreases.
            if magnitudes[i + 1] < magnitudes[i]:
                violations.append(i + 1)

        fraction = len(violations) / (n - 1)
        if fraction > 0.20:
            return SignalAnomaly(
                check_name="eis_consistency",
                severity="warning",
                message=(
                    f"KK consistency violation: {len(violations)}/{n - 1} "
                    f"({fraction:.0%}) consecutive pairs break monotonicity"
                ),
                affected_indices=violations,
                metadata={"violation_fraction": fraction},
            )
        return None

    @staticmethod
    def check_voltage_spike(
        voltages: list[float],
        threshold_sigma: float = 4.0,
    ) -> SignalAnomaly | None:
        """Detect voltage spikes > *threshold_sigma* from a running mean.

        Uses a rolling window of 5 to compute a local mean and standard
        deviation.  The point being tested is excluded from the window
        statistics to prevent the spike from inflating the std.

        When the local std is zero (neighbours are identical), any nonzero
        deviation is treated as a spike.
        """
        n = len(voltages)
        if n < 3:
            return None

        window = 5
        spikes: list[int] = []

        for i in range(n):
            lo = max(0, i - window // 2)
            hi = min(n, i + window // 2 + 1)
            # Exclude the point itself from the window
            neighbours = [voltages[j] for j in range(lo, hi) if j != i]
            if not neighbours:
                continue
            mean = sum(neighbours) / len(neighbours)
            var = sum((v - mean) ** 2 for v in neighbours) / len(neighbours)
            std = math.sqrt(var) if var > 0 else 0.0
            deviation = abs(voltages[i] - mean)

            if std > 0:
                if deviation > threshold_sigma * std:
                    spikes.append(i)
            else:
                # Zero std: neighbours are identical.  Any nonzero deviation
                # is infinitely many sigma away, so flag it.
                if deviation > 0:
                    spikes.append(i)

        if spikes:
            return SignalAnomaly(
                check_name="voltage_spike",
                severity="error",
                message=(
                    f"Voltage spike(s) detected at {len(spikes)} point(s) "
                    f"exceeding {threshold_sigma}\u03c3 from running mean"
                ),
                affected_indices=spikes,
                metadata={"threshold_sigma": threshold_sigma},
            )
        return None

    @staticmethod
    def check_negative_absorbance(
        absorbance: list[float],
    ) -> SignalAnomaly | None:
        """UV-Vis: flag if > 5 % of absorbance values are negative."""
        n = len(absorbance)
        if n == 0:
            return None

        neg_indices = [i for i, a in enumerate(absorbance) if a < 0]
        fraction = len(neg_indices) / n

        if fraction > 0.05:
            return SignalAnomaly(
                check_name="negative_absorbance",
                severity="warning",
                message=(
                    f"{len(neg_indices)}/{n} ({fraction:.0%}) absorbance "
                    f"values are negative"
                ),
                affected_indices=neg_indices,
                metadata={"negative_fraction": fraction},
            )
        return None

    @staticmethod
    def check_xrd_peak_saturation(
        intensities: list[float],
        max_counts: float = 65535.0,
    ) -> SignalAnomaly | None:
        """XRD: flag if any intensity is within 1 % of *max_counts*.

        Detector saturation produces flat-topped peaks that distort
        quantitative analysis.
        """
        n = len(intensities)
        if n == 0:
            return None

        threshold = max_counts * 0.99
        saturated = [i for i, v in enumerate(intensities) if v >= threshold]

        if saturated:
            return SignalAnomaly(
                check_name="xrd_peak_saturation",
                severity="error",
                message=(
                    f"{len(saturated)} intensity value(s) within 1% of "
                    f"detector maximum ({max_counts})"
                ),
                affected_indices=saturated,
                metadata={"max_counts": max_counts, "threshold": threshold},
            )
        return None

    # ── Aggregate runner ───────────────────────────────────────────

    def check_all(self, raw_data: dict[str, Any]) -> list[SignalAnomaly]:
        """Run all applicable checks based on keys in *raw_data*.

        Expected keys (all optional):
        - ``z_real``, ``z_imag``: EIS impedance data
        - ``voltages``: voltage trace
        - ``absorbance``: UV-Vis absorbance spectrum
        - ``xrd_intensities``: XRD intensity counts

        Returns a list of all detected anomalies (may be empty).
        """
        anomalies: list[SignalAnomaly] = []

        # EIS consistency
        if "z_real" in raw_data and "z_imag" in raw_data:
            result = self.check_eis_consistency(
                raw_data["z_real"], raw_data["z_imag"]
            )
            if result is not None:
                anomalies.append(result)

        # Voltage spike
        if "voltages" in raw_data:
            result = self.check_voltage_spike(raw_data["voltages"])
            if result is not None:
                anomalies.append(result)

        # Negative absorbance
        if "absorbance" in raw_data:
            result = self.check_negative_absorbance(raw_data["absorbance"])
            if result is not None:
                anomalies.append(result)

        # XRD saturation
        if "xrd_intensities" in raw_data:
            result = self.check_xrd_peak_saturation(
                raw_data["xrd_intensities"]
            )
            if result is not None:
                anomalies.append(result)

        return anomalies

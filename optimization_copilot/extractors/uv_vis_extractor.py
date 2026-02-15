"""UV-Vis spectrophotometry extractor with uncertainty.

Extracts absorbance at a target wavelength via linear interpolation and
propagates instrument noise, Beer-Lambert non-linearity effects, and
baseline correction uncertainty.
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.extractors.base import UncertaintyExtractor
from optimization_copilot.uncertainty.types import MeasurementWithUncertainty


class UVVisExtractor(UncertaintyExtractor):
    """Extract absorbance at a target wavelength from UV-Vis data.

    Parameters
    ----------
    domain_config : dict[str, Any]
        Output of ``get_uvvis_config()`` from
        ``optimization_copilot.domain_knowledge.uv_vis``.
    """

    # ── public API ────────────────────────────────────────────────────

    def extract_with_uncertainty(
        self, raw_data: dict[str, Any],
    ) -> list[MeasurementWithUncertainty]:
        """Extract absorbance at target wavelength with uncertainty.

        Parameters
        ----------
        raw_data : dict
            Must contain ``"wavelengths"`` (nm), ``"absorbance"`` (AU),
            and ``"target_wavelength"`` (nm).  Optional ``"baseline"``
            (AU) for baseline correction.

        Returns
        -------
        list[MeasurementWithUncertainty]
            Single-element list with absorbance measurement.
        """
        wavelengths: list[float] = raw_data.get("wavelengths", [])
        absorbance: list[float] = raw_data.get("absorbance", [])
        target_wl: float = raw_data.get("target_wavelength", 0.0)
        baseline: list[float] | None = raw_data.get("baseline")

        n = min(len(wavelengths), len(absorbance))
        if n < 1:
            return [MeasurementWithUncertainty(
                value=float("nan"),
                variance=0.0,
                confidence=0.0,
                source="UVVIS_absorbance",
                n_points_used=0,
                method="interpolation",
                metadata={"error": "insufficient_data"},
            )]

        wavelengths = wavelengths[:n]
        absorbance = absorbance[:n]

        # ── baseline correction ───────────────────────────────────
        if baseline is not None:
            bl = baseline[:n]
            absorbance = [a - b for a, b in zip(absorbance, bl)]

        # ── instrument config ─────────────────────────────────────
        inst = self.domain_config.get("instrument", {})
        noise_spec = inst.get("absorbance_noise", {})
        linear_max = inst.get("linear_range_max", 2.5)

        # ── interpolate to target wavelength ──────────────────────
        abs_val, interp_var = self._interpolate(wavelengths, absorbance, target_wl)

        # ── instrument noise ──────────────────────────────────────
        noise_sigma = self._get_noise_sigma(abs(abs_val), noise_spec)
        instrument_var = noise_sigma * noise_sigma

        # If baseline was applied, double the instrument variance
        # (two independent measurements).
        if baseline is not None:
            instrument_var *= 2.0

        total_var = interp_var + instrument_var

        confidence = self._compute_confidence(total_var, abs_val)

        # ── Beer-Lambert linearity check ──────────────────────────
        if abs(abs_val) > linear_max:
            confidence *= 0.5

        # ── physical constraints ──────────────────────────────────
        result = MeasurementWithUncertainty(
            value=abs_val,
            variance=total_var,
            confidence=max(0.0, min(1.0, confidence)),
            source="UVVIS_absorbance",
            n_points_used=n,
            method="interpolation",
            metadata={
                "target_wavelength": target_wl,
                "instrument_variance": instrument_var,
                "interp_variance": interp_var,
                "baseline_corrected": baseline is not None,
            },
        )

        constraints = self.domain_config.get("physical_constraints", {})
        abs_constraints = constraints.get("absorbance", {})
        if abs_constraints:
            result = self._apply_physical_constraints(
                result, "absorbance", abs_constraints,
            )

        return [result]

    # ── private helpers ───────────────────────────────────────────────

    @staticmethod
    def _interpolate(
        wavelengths: list[float],
        absorbance: list[float],
        target: float,
    ) -> tuple[float, float]:
        """Linear interpolation with variance estimate.

        Returns ``(value, variance)`` where variance is zero when the
        target exactly matches a data point.
        """
        # Exact match check
        for i, wl in enumerate(wavelengths):
            if abs(wl - target) < 1e-9:
                return absorbance[i], 0.0

        # Find bracketing indices (assume monotonic wavelengths)
        left_idx: int | None = None
        right_idx: int | None = None

        for i in range(len(wavelengths) - 1):
            wl_a, wl_b = wavelengths[i], wavelengths[i + 1]
            if (wl_a <= target <= wl_b) or (wl_b <= target <= wl_a):
                left_idx = i
                right_idx = i + 1
                break

        if left_idx is None or right_idx is None:
            # Target outside range: use nearest endpoint
            dists = [abs(wl - target) for wl in wavelengths]
            nearest = dists.index(min(dists))
            # Large extrapolation variance
            extrap_dist = min(dists)
            avg_spacing = abs(wavelengths[-1] - wavelengths[0]) / max(len(wavelengths) - 1, 1)
            # Variance proportional to extrapolation distance
            extrap_var = (absorbance[nearest] * extrap_dist / max(avg_spacing, 1e-9)) ** 2 * 0.01
            return absorbance[nearest], extrap_var

        wl_a = wavelengths[left_idx]
        wl_b = wavelengths[right_idx]
        a_a = absorbance[left_idx]
        a_b = absorbance[right_idx]

        dw = wl_b - wl_a
        if abs(dw) < 1e-12:
            return a_a, 0.0

        frac = (target - wl_a) / dw
        val = a_a + frac * (a_b - a_a)

        # Interpolation variance: proportional to spacing and curvature
        # Simple model: assume linear is exact, so interp_var ~ 0
        # but we add a small term for grid spacing uncertainty
        spacing_factor = abs(dw)
        interp_var = (abs(a_b - a_a) * 0.01) ** 2 * (frac * (1.0 - frac))

        return val, interp_var

    @staticmethod
    def _get_noise_sigma(
        abs_val: float,
        noise_spec: dict[str, float],
    ) -> float:
        """Return noise sigma for the given absorbance tier."""
        if abs_val < 0.5:
            return noise_spec.get("low_abs", 0.002)
        if abs_val < 2.0:
            return noise_spec.get("mid_abs", 0.005)
        return noise_spec.get("high_abs", 0.02)

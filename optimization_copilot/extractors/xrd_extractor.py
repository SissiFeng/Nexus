"""XRD crystallite-size extractor with uncertainty.

Applies the Scherrer equation to estimate crystallite size from a
diffraction peak, propagating uncertainties in the shape factor K,
FWHM estimation, and instrument broadening correction.
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.extractors.base import UncertaintyExtractor
from optimization_copilot.uncertainty.types import MeasurementWithUncertainty


class XRDExtractor(UncertaintyExtractor):
    """Extract crystallite size from XRD peak data.

    Parameters
    ----------
    domain_config : dict[str, Any]
        Output of ``get_xrd_config()`` from
        ``optimization_copilot.domain_knowledge.xrd``.
    """

    # ── public API ────────────────────────────────────────────────────

    def extract_with_uncertainty(
        self, raw_data: dict[str, Any],
    ) -> list[MeasurementWithUncertainty]:
        """Compute crystallite size via Scherrer equation.

        D = K * lambda / (beta_sample * cos(theta))

        Parameters
        ----------
        raw_data : dict
            Must contain ``"two_theta"`` (deg), ``"intensity"``
            (counts), and ``"peak_position"`` (deg 2-theta).

        Returns
        -------
        list[MeasurementWithUncertainty]
            Single-element list with crystallite size in **nm**.
        """
        two_theta: list[float] = raw_data.get("two_theta", [])
        intensity: list[float] = raw_data.get("intensity", [])
        peak_pos: float = raw_data.get("peak_position", 0.0)

        n = min(len(two_theta), len(intensity))
        if n < 3:
            return [self._nan_result(n, "insufficient_data")]

        two_theta = two_theta[:n]
        intensity = intensity[:n]

        # ── instrument specs ──────────────────────────────────────
        inst = self.domain_config.get("instrument", {})
        k_lo, k_hi = inst.get("scherrer_k_range", (0.89, 0.94))
        beta_inst_deg = inst.get("instrument_broadening_deg", 0.05)
        wavelength_A = inst.get("wavelength_angstrom", 1.5406)

        k_mean = 0.5 * (k_lo + k_hi)
        # Uniform distribution variance: (b-a)^2 / 12
        k_var = (k_hi - k_lo) ** 2 / 12.0

        # ── find peak and estimate FWHM ──────────────────────────
        beta_obs_deg, beta_var_deg2 = self._estimate_fwhm(
            two_theta, intensity, peak_pos,
        )

        if math.isnan(beta_obs_deg):
            return [self._nan_result(n, "fwhm_estimation_failed")]

        # ── instrument broadening correction ─────────────────────
        beta_obs2 = beta_obs_deg * beta_obs_deg
        beta_inst2 = beta_inst_deg * beta_inst_deg

        if beta_obs2 <= beta_inst2:
            # Observed broadening is within instrument broadening
            return [MeasurementWithUncertainty(
                value=float("nan"),
                variance=0.0,
                confidence=0.0,
                source="XRD_crystallite_size",
                n_points_used=n,
                method="scherrer",
                metadata={"error": "broadening_below_instrument"},
            )]

        beta_sample_deg = math.sqrt(beta_obs2 - beta_inst2)

        # Propagate variance through the sqrt subtraction:
        # beta_s = sqrt(beta_o^2 - beta_i^2)
        # d(beta_s)/d(beta_o) = beta_o / beta_s
        # var(beta_s) ≈ (beta_o / beta_s)^2 * var(beta_o)
        deriv = beta_obs_deg / beta_sample_deg
        beta_sample_var = deriv * deriv * beta_var_deg2

        # Convert to radians
        beta_sample_rad = math.radians(beta_sample_deg)
        beta_sample_var_rad = beta_sample_var * (math.pi / 180.0) ** 2

        # ── Scherrer equation ─────────────────────────────────────
        theta_rad = math.radians(peak_pos / 2.0)
        cos_theta = math.cos(theta_rad)

        if abs(cos_theta) < 1e-12 or abs(beta_sample_rad) < 1e-15:
            return [self._nan_result(n, "degenerate_geometry")]

        # D in angstroms
        D_A = k_mean * wavelength_A / (beta_sample_rad * cos_theta)

        # Error propagation:
        # D = K * lam / (beta * cos(theta))
        # var(D) = D^2 * (var_K / K^2 + var_beta / beta^2)
        rel_k = k_var / max(k_mean * k_mean, 1e-30)
        rel_beta = beta_sample_var_rad / max(
            beta_sample_rad * beta_sample_rad, 1e-30,
        )
        D_var_A = D_A * D_A * (rel_k + rel_beta)

        # Convert to nm
        D_nm = D_A / 10.0
        D_var_nm = D_var_A / 100.0  # (1/10)^2

        confidence = self._compute_confidence(D_var_nm, D_nm)

        result = MeasurementWithUncertainty(
            value=D_nm,
            variance=D_var_nm,
            confidence=confidence,
            source="XRD_crystallite_size",
            n_points_used=n,
            method="scherrer",
            metadata={
                "peak_position_deg": peak_pos,
                "fwhm_obs_deg": beta_obs_deg,
                "fwhm_sample_deg": beta_sample_deg,
                "k_mean": k_mean,
                "wavelength_A": wavelength_A,
            },
        )

        # ── physical constraints ──────────────────────────────────
        constraints = self.domain_config.get("physical_constraints", {})
        cs_constraints = constraints.get("crystallite_size", {})
        if cs_constraints:
            result = self._apply_physical_constraints(
                result, "crystallite_size", cs_constraints,
            )

        return [result]

    # ── private helpers ───────────────────────────────────────────────

    @staticmethod
    def _estimate_fwhm(
        two_theta: list[float],
        intensity: list[float],
        peak_pos: float,
    ) -> tuple[float, float]:
        """Estimate FWHM via parabolic interpolation around the peak.

        Returns ``(fwhm_deg, fwhm_variance_deg2)``.
        Returns ``(nan, nan)`` on failure.
        """
        # Find the index closest to peak_pos
        dists = [abs(tt - peak_pos) for tt in two_theta]
        peak_idx = dists.index(min(dists))

        if peak_idx == 0 or peak_idx == len(two_theta) - 1:
            # Need neighbours for parabolic fit
            # Fall back to simple half-max search
            return XRDExtractor._fwhm_half_max(two_theta, intensity, peak_idx)

        # Parabolic interpolation around peak (3-point fit)
        x0 = two_theta[peak_idx - 1]
        x1 = two_theta[peak_idx]
        x2 = two_theta[peak_idx + 1]
        y0 = intensity[peak_idx - 1]
        y1 = intensity[peak_idx]
        y2 = intensity[peak_idx + 1]

        # Fit y = a*(x-x1)^2 + b*(x-x1) + c
        dx0 = x0 - x1
        dx2 = x2 - x1

        denom = dx0 * dx2 * (dx0 - dx2)
        if abs(denom) < 1e-30:
            return XRDExtractor._fwhm_half_max(two_theta, intensity, peak_idx)

        a = (dx2 * (y0 - y1) - dx0 * (y2 - y1)) / denom
        # c = y1 (peak value from parabola)

        if a >= 0:
            # Parabola opens upward: not a valid peak shape
            return XRDExtractor._fwhm_half_max(two_theta, intensity, peak_idx)

        # For a downward parabola y = a*(x-x1)^2 + y1:
        # half max level = y1/2 (relative to zero baseline)
        # Actually, FWHM means width at half the peak height.
        # half_height = y1 / 2
        # a*(x-x1)^2 + y1 = y1/2  =>  (x-x1)^2 = -y1/(2a)
        if y1 <= 0:
            return float("nan"), float("nan")

        arg = -y1 / (2.0 * a)
        if arg < 0:
            return float("nan"), float("nan")

        half_width = math.sqrt(arg)
        fwhm = 2.0 * half_width

        # Variance estimate: based on how well the parabola represents
        # the data.  Use the residual of the 3-point fit as a proxy.
        y0_pred = a * dx0 * dx0 + y1
        y2_pred = a * dx2 * dx2 + y1
        residual = ((y0 - y0_pred) ** 2 + (y2 - y2_pred) ** 2) / 2.0
        # Scale to FWHM variance (heuristic: residual relative to peak)
        fwhm_var = (fwhm * 0.05) ** 2  # baseline 5% uncertainty
        if y1 > 0:
            fwhm_var += (fwhm * math.sqrt(residual) / y1) ** 2

        return fwhm, fwhm_var

    @staticmethod
    def _fwhm_half_max(
        two_theta: list[float],
        intensity: list[float],
        peak_idx: int,
    ) -> tuple[float, float]:
        """Simple FWHM estimation by scanning for half-maximum crossings."""
        peak_val = intensity[peak_idx]
        if peak_val <= 0:
            return float("nan"), float("nan")

        half_val = peak_val / 2.0

        # Scan left
        left_pos = two_theta[peak_idx]
        for i in range(peak_idx, 0, -1):
            if intensity[i - 1] <= half_val:
                # Linear interpolation
                frac = (half_val - intensity[i - 1]) / max(
                    intensity[i] - intensity[i - 1], 1e-30,
                )
                left_pos = two_theta[i - 1] + frac * (
                    two_theta[i] - two_theta[i - 1]
                )
                break
        else:
            left_pos = two_theta[0]

        # Scan right
        right_pos = two_theta[peak_idx]
        for i in range(peak_idx, len(two_theta) - 1):
            if intensity[i + 1] <= half_val:
                frac = (half_val - intensity[i + 1]) / max(
                    intensity[i] - intensity[i + 1], 1e-30,
                )
                right_pos = two_theta[i + 1] + frac * (
                    two_theta[i] - two_theta[i + 1]
                )
                break
        else:
            right_pos = two_theta[-1]

        fwhm = abs(right_pos - left_pos)
        # Larger uncertainty for this cruder method
        fwhm_var = (fwhm * 0.10) ** 2
        return fwhm, fwhm_var

    def _nan_result(
        self, n: int, error: str,
    ) -> MeasurementWithUncertainty:
        return MeasurementWithUncertainty(
            value=float("nan"),
            variance=0.0,
            confidence=0.0,
            source="XRD_crystallite_size",
            n_points_used=n,
            method="scherrer",
            metadata={"error": error},
        )

"""EIS (Electrochemical Impedance Spectroscopy) extractor with uncertainty.

Extracts two KPIs:

1. **|Z| at target frequency** -- interpolation-based, straightforward.
2. **R_ct (charge-transfer resistance)** -- via Levenberg-Marquardt fitting
   of equivalent-circuit models with AIC-weighted ensemble.

All linear-algebra operations use the stdlib-only helpers from
``optimization_copilot.backends._math.linalg``.
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.backends._math.linalg import (
    identity,
    mat_add,
    mat_inv,
    mat_mul,
    mat_scale,
    mat_vec,
    transpose,
    vec_dot,
)
from optimization_copilot.extractors.base import UncertaintyExtractor
from optimization_copilot.uncertainty.types import MeasurementWithUncertainty


# ── Circuit impedance models ──────────────────────────────────────────


def _z_rc(omega: float, r: float, c: float) -> complex:
    """Impedance of a parallel R||C element."""
    if abs(c) < 1e-30 or abs(r) < 1e-30:
        return complex(r, 0.0)
    zc = 1.0 / (1j * omega * c)
    return (r * zc) / (r + zc)


def _z_randles(omega: float, params: list[float]) -> complex:
    """Z = R_s + R_ct || C_dl."""
    r_s, r_ct, c_dl = params[0], params[1], params[2]
    return r_s + _z_rc(omega, r_ct, c_dl)


def _z_randles_warburg(omega: float, params: list[float]) -> complex:
    """Z = R_s + R_ct || C_dl + W (Warburg)."""
    r_s, r_ct, c_dl, w_s, w_n = (
        params[0], params[1], params[2], params[3], params[4],
    )
    z_w = w_s / ((1j * omega) ** w_n) if omega > 0 else complex(w_s, 0)
    return r_s + _z_rc(omega, r_ct, c_dl) + z_w


def _z_2rc(omega: float, params: list[float]) -> complex:
    """Z = R_s + (R1 || C1) + (R2 || C2)."""
    r_s, r1, c1, r2, c2 = (
        params[0], params[1], params[2], params[3], params[4],
    )
    return r_s + _z_rc(omega, r1, c1) + _z_rc(omega, r2, c2)


_CIRCUIT_MODELS: dict[str, Any] = {
    "randles": {"func": _z_randles, "n_params": 3},
    "randles_warburg": {"func": _z_randles_warburg, "n_params": 5},
    "2rc": {"func": _z_2rc, "n_params": 5},
}


# ── Levenberg-Marquardt fitter ────────────────────────────────────────


def _residuals(
    z_model_func: Any,
    params: list[float],
    omegas: list[float],
    z_data_re: list[float],
    z_data_im: list[float],
) -> list[float]:
    """Residual vector: [re_0, im_0, re_1, im_1, ...]."""
    res: list[float] = []
    for i, omega in enumerate(omegas):
        z_calc = z_model_func(omega, params)
        res.append(z_data_re[i] - z_calc.real)
        res.append(z_data_im[i] - z_calc.imag)
    return res


def _jacobian(
    z_model_func: Any,
    params: list[float],
    omegas: list[float],
    eps: float = 1e-8,
) -> list[list[float]]:
    """Numerical Jacobian via central finite differences.

    Returns a (2*n_freq, n_params) matrix.
    """
    n_freq = len(omegas)
    n_params = len(params)
    J: list[list[float]] = [[0.0] * n_params for _ in range(2 * n_freq)]

    for j in range(n_params):
        p_plus = list(params)
        p_minus = list(params)
        h = max(eps * abs(params[j]), eps)
        p_plus[j] += h
        p_minus[j] -= h

        for i, omega in enumerate(omegas):
            z_plus = z_model_func(omega, p_plus)
            z_minus = z_model_func(omega, p_minus)
            # Jacobian of *negative* residual model wrt params
            # residual = data - model, d(residual)/d(param) = -d(model)/d(param)
            J[2 * i][j] = -(z_plus.real - z_minus.real) / (2.0 * h)
            J[2 * i + 1][j] = -(z_plus.imag - z_minus.imag) / (2.0 * h)

    return J


def _lm_fit(
    z_model_func: Any,
    init_params: list[float],
    omegas: list[float],
    z_data_re: list[float],
    z_data_im: list[float],
    bounds: list[tuple[float, float]] | None = None,
    max_iter: int = 200,
    lam: float = 1e-3,
    lam_up: float = 10.0,
    lam_down: float = 0.1,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """Levenberg-Marquardt least-squares fit.

    Returns a dict with ``params``, ``covariance``, ``rss``,
    ``converged``, ``n_iter``.
    """
    params = list(init_params)
    n_params = len(params)
    n_data = 2 * len(omegas)

    res = _residuals(z_model_func, params, omegas, z_data_re, z_data_im)
    rss = sum(r * r for r in res)

    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        J = _jacobian(z_model_func, params, omegas)
        Jt = transpose(J)

        # JtJ = J^T J
        JtJ = mat_mul(Jt, J)

        # JtR = J^T r
        JtR = mat_vec(Jt, res)

        # Damped normal equations: (JtJ + lam*I) delta = -JtR
        damped = mat_add(JtJ, mat_scale(identity(n_params), lam))

        try:
            neg_JtR = [-v for v in JtR]
            delta = mat_vec(mat_inv(damped), neg_JtR)
        except Exception:
            lam *= lam_up
            continue

        # Trial update
        trial = [params[j] + delta[j] for j in range(n_params)]

        # Enforce bounds
        if bounds is not None:
            for j in range(n_params):
                lo, hi = bounds[j]
                trial[j] = max(lo, min(hi, trial[j]))

        trial_res = _residuals(
            z_model_func, trial, omegas, z_data_re, z_data_im,
        )
        trial_rss = sum(r * r for r in trial_res)

        if trial_rss < rss:
            params = trial
            res = trial_res
            improvement = rss - trial_rss
            rss = trial_rss
            lam *= lam_down
            if improvement < tol * rss + tol:
                converged = True
                break
        else:
            lam *= lam_up

    # ── Parameter covariance: (J^T J)^{-1} * s^2 ─────────────────
    dof = max(n_data - n_params, 1)
    s2 = rss / dof

    J_final = _jacobian(z_model_func, params, omegas)
    Jt_final = transpose(J_final)
    JtJ_final = mat_mul(Jt_final, J_final)

    try:
        cov = mat_scale(mat_inv(JtJ_final), s2)
    except Exception:
        cov = [[0.0] * n_params for _ in range(n_params)]

    return {
        "params": params,
        "covariance": cov,
        "rss": rss,
        "converged": converged,
        "n_iter": iteration,
        "s2": s2,
    }


# ── EIS Extractor ─────────────────────────────────────────────────────


class EISExtractor(UncertaintyExtractor):
    """Extract |Z| and R_ct from EIS data with uncertainty.

    Parameters
    ----------
    domain_config : dict[str, Any]
        Output of ``get_eis_config()`` from
        ``optimization_copilot.domain_knowledge.eis``.
    """

    # ── public API ────────────────────────────────────────────────────

    def extract_with_uncertainty(
        self, raw_data: dict[str, Any],
    ) -> list[MeasurementWithUncertainty]:
        """Extract |Z| at target frequency and/or R_ct.

        Parameters
        ----------
        raw_data : dict
            Must contain ``"frequency"`` (Hz), ``"z_real"`` (ohm),
            ``"z_imag"`` (ohm).  Optional ``"target_frequency"``
            for |Z| extraction.

        Returns
        -------
        list[MeasurementWithUncertainty]
            Up to two entries: |Z| and R_ct.
        """
        freq: list[float] = raw_data.get("frequency", [])
        z_real: list[float] = raw_data.get("z_real", [])
        z_imag: list[float] = raw_data.get("z_imag", [])
        target_freq: float | None = raw_data.get("target_frequency")

        n = min(len(freq), len(z_real), len(z_imag))
        if n < 1:
            return [self._nan_result("EIS_Z_magnitude", 0, "insufficient_data")]

        freq = freq[:n]
        z_real = z_real[:n]
        z_imag = z_imag[:n]

        results: list[MeasurementWithUncertainty] = []

        # ── KPI 1: |Z| at target frequency ───────────────────────
        if target_freq is not None:
            z_mag_result = self._extract_z_magnitude(
                freq, z_real, z_imag, target_freq, n,
            )
            results.append(z_mag_result)

        # ── KPI 2: R_ct via circuit fitting ───────────────────────
        if n >= 3:  # Need at least 3 points for fitting
            rct_result = self._extract_rct(freq, z_real, z_imag, n)
            results.append(rct_result)

        return results

    # ── |Z| extraction ────────────────────────────────────────────────

    def _extract_z_magnitude(
        self,
        freq: list[float],
        z_real: list[float],
        z_imag: list[float],
        target_freq: float,
        n: int,
    ) -> MeasurementWithUncertainty:
        """Extract |Z| at target frequency via log-linear interpolation."""
        inst = self.domain_config.get("instrument", {})
        imp_spec = inst.get("impedance", {})
        z_rel_acc = imp_spec.get("z_relative_accuracy", 0.001)

        # Compute |Z| at each frequency
        z_mag = [math.sqrt(zr ** 2 + zi ** 2) for zr, zi in zip(z_real, z_imag)]

        # Log-frequency interpolation
        log_freq = [math.log(max(f, 1e-30)) for f in freq]
        log_target = math.log(max(target_freq, 1e-30))

        z_val, interp_var = self._log_interpolate(log_freq, z_mag, log_target)

        # Instrument noise
        instrument_var = (z_val * z_rel_acc) ** 2

        # Low-frequency noise amplification
        amp_spec = imp_spec.get("low_freq_noise_amplification", {})
        if target_freq < 1.0:
            amp_factor = amp_spec.get("below_1hz", 3.0)
        elif target_freq < 10.0:
            amp_factor = amp_spec.get("below_10hz", 1.5)
        else:
            amp_factor = 1.0
        instrument_var *= amp_factor * amp_factor

        total_var = interp_var + instrument_var
        confidence = self._compute_confidence(total_var, z_val)

        result = MeasurementWithUncertainty(
            value=z_val,
            variance=total_var,
            confidence=confidence,
            source="EIS_Z_magnitude",
            n_points_used=n,
            method="interpolation",
            metadata={
                "target_frequency": target_freq,
                "instrument_variance": instrument_var,
                "interp_variance": interp_var,
                "noise_amplification_factor": amp_factor,
            },
        )

        # Physical constraints
        constraints = self.domain_config.get("physical_constraints", {})
        z_constraints = constraints.get("z_magnitude", {})
        if z_constraints:
            result = self._apply_physical_constraints(
                result, "z_magnitude", z_constraints,
            )

        return result

    @staticmethod
    def _log_interpolate(
        log_freq: list[float],
        values: list[float],
        log_target: float,
    ) -> tuple[float, float]:
        """Interpolate in log-frequency space.

        Returns ``(value, variance)``.
        """
        # Exact match
        for i, lf in enumerate(log_freq):
            if abs(lf - log_target) < 1e-9:
                return values[i], 0.0

        # Find bracketing indices
        # Frequencies may be in descending order, so handle both
        for i in range(len(log_freq) - 1):
            lf_a, lf_b = log_freq[i], log_freq[i + 1]
            if (lf_a <= log_target <= lf_b) or (lf_b <= log_target <= lf_a):
                span = lf_b - lf_a
                if abs(span) < 1e-30:
                    return values[i], 0.0
                frac = (log_target - lf_a) / span
                val = values[i] + frac * (values[i + 1] - values[i])
                # Interpolation variance
                diff = abs(values[i + 1] - values[i])
                interp_var = (diff * 0.01) ** 2 * abs(frac * (1 - frac))
                return val, interp_var

        # Extrapolation: use nearest
        dists = [abs(lf - log_target) for lf in log_freq]
        nearest = dists.index(min(dists))
        extrap_factor = min(dists) / max(
            abs(log_freq[-1] - log_freq[0]) if len(log_freq) > 1 else 1.0,
            1e-9,
        )
        extrap_var = (values[nearest] * extrap_factor * 0.1) ** 2
        return values[nearest], extrap_var

    # ── R_ct extraction via circuit fitting ───────────────────────────

    def _extract_rct(
        self,
        freq: list[float],
        z_real: list[float],
        z_imag: list[float],
        n: int,
    ) -> MeasurementWithUncertainty:
        """Extract R_ct using multi-circuit AIC-weighted ensemble."""
        circuits = self.domain_config.get("circuits", [])
        model_sel = self.domain_config.get("model_selection", {})
        max_iter = model_sel.get("convergence_max_iter", 200)

        omegas = [2.0 * math.pi * f for f in freq]

        fit_results: list[dict[str, Any]] = []

        for circuit in circuits:
            name = circuit["name"]
            if name not in _CIRCUIT_MODELS:
                continue

            model_info = _CIRCUIT_MODELS[name]
            z_func = model_info["func"]
            n_params = model_info["n_params"]
            rct_index = circuit.get("rct_index", 1)

            # Initial parameter guess: midpoint of bounds
            bounds_dict = circuit.get("init_bounds", {})
            param_names = circuit.get("params", [])
            init_params: list[float] = []
            bounds_list: list[tuple[float, float]] = []
            for pname in param_names:
                lo, hi = bounds_dict.get(pname, (0.1, 1000))
                # Use geometric mean for log-scale parameters
                init_params.append(math.sqrt(lo * hi))
                bounds_list.append((lo, hi))

            try:
                result = _lm_fit(
                    z_func,
                    init_params,
                    omegas,
                    z_real,
                    z_imag,
                    bounds=bounds_list,
                    max_iter=max_iter,
                )
            except Exception:
                continue

            if not result["converged"] and result["rss"] > 1e10:
                continue

            # AIC = n * ln(RSS/n) + 2k
            n_data = 2 * n  # real + imag
            rss = max(result["rss"], 1e-30)
            aic = n_data * math.log(rss / n_data) + 2 * n_params

            rct_val = result["params"][rct_index]
            rct_var = result["covariance"][rct_index][rct_index]
            rct_var = max(rct_var, 0.0)

            fit_results.append({
                "name": name,
                "rct": rct_val,
                "rct_var": rct_var,
                "aic": aic,
                "rss": result["rss"],
                "converged": result["converged"],
                "params": result["params"],
                "n_iter": result["n_iter"],
            })

        if not fit_results:
            return MeasurementWithUncertainty(
                value=float("nan"),
                variance=0.0,
                confidence=0.0,
                source="EIS_Rct_ensemble",
                n_points_used=n,
                method="lm_fit",
                metadata={"error": "all_fits_failed"},
            )

        # ── AIC-weighted ensemble ─────────────────────────────────
        if len(fit_results) == 1:
            fr = fit_results[0]
            rct_val = fr["rct"]
            rct_var = fr["rct_var"]
            confidence = self._compute_confidence(rct_var, rct_val)

            fit_residual = fr["rss"]
            result_m = MeasurementWithUncertainty(
                value=rct_val,
                variance=rct_var,
                confidence=confidence,
                source="EIS_Rct_ensemble",
                fit_residual=fit_residual,
                n_points_used=n,
                method="lm_fit",
                metadata={
                    "circuit": fr["name"],
                    "converged": fr["converged"],
                    "n_circuits_fitted": 1,
                    "aic": fr["aic"],
                },
            )
        else:
            # Multiple circuits: AIC-weighted average
            aic_values = [fr["aic"] for fr in fit_results]
            aic_min = min(aic_values)
            delta_aic = [a - aic_min for a in aic_values]

            # Weights: w_i = exp(-0.5 * ΔAIC) / Σ exp(-0.5 * ΔAIC)
            raw_weights = [math.exp(-0.5 * da) for da in delta_aic]
            total_weight = sum(raw_weights)
            if total_weight < 1e-30:
                total_weight = 1.0
            weights = [w / total_weight for w in raw_weights]

            # Weighted mean
            mu_rct = sum(w * fr["rct"] for w, fr in zip(weights, fit_results))

            # Total variance = intra-model + inter-model
            intra_var = sum(
                w * fr["rct_var"] for w, fr in zip(weights, fit_results)
            )
            inter_var = sum(
                w * (fr["rct"] - mu_rct) ** 2
                for w, fr in zip(weights, fit_results)
            )
            total_var = intra_var + inter_var

            confidence = self._compute_confidence(total_var, mu_rct)

            # Best-fit residual (lowest AIC)
            best_idx = aic_values.index(aic_min)
            fit_residual = fit_results[best_idx]["rss"]

            circuit_details = [
                {
                    "name": fr["name"],
                    "rct": fr["rct"],
                    "weight": w,
                    "aic": fr["aic"],
                    "converged": fr["converged"],
                }
                for fr, w in zip(fit_results, weights)
            ]

            result_m = MeasurementWithUncertainty(
                value=mu_rct,
                variance=total_var,
                confidence=confidence,
                source="EIS_Rct_ensemble",
                fit_residual=fit_residual,
                n_points_used=n,
                method="lm_fit",
                metadata={
                    "n_circuits_fitted": len(fit_results),
                    "circuit_details": circuit_details,
                    "intra_model_variance": intra_var,
                    "inter_model_variance": inter_var,
                },
            )

        # Physical constraints
        constraints = self.domain_config.get("physical_constraints", {})
        rct_constraints = constraints.get("Rct", {})
        if rct_constraints:
            result_m = self._apply_physical_constraints(
                result_m, "Rct", rct_constraints,
            )

        return result_m

    # ── helpers ────────────────────────────────────────────────────────

    def _nan_result(
        self, source: str, n: int, error: str,
    ) -> MeasurementWithUncertainty:
        return MeasurementWithUncertainty(
            value=float("nan"),
            variance=0.0,
            confidence=0.0,
            source=source,
            n_points_used=n,
            method="direct",
            metadata={"error": error},
        )

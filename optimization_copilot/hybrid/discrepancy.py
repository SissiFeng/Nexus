"""Discrepancy analysis for hybrid theory-data models.

Identifies where and how the theory model fails, detects systematic
biases in residuals, and generates heuristic suggestions for theory
revision.
"""

from __future__ import annotations

import math

from optimization_copilot.hybrid.residual import ResidualGP


class DiscrepancyAnalyzer:
    """Analyze where a theory model breaks down.

    Provides diagnostic tools for understanding the discrepancy
    between theory predictions and observed data, using the
    fitted residual GP as evidence.
    """

    def systematic_bias(self, residual_gp: ResidualGP) -> dict:
        """Detect if residuals show systematic patterns.

        Uses a simple t-test approximation: check whether the
        mean residual is significantly different from zero.

        Parameters
        ----------
        residual_gp : ResidualGP
            A fitted residual GP.

        Returns
        -------
        dict
            Keys:
            - ``mean_residual`` (float): mean of training residuals.
            - ``is_biased`` (bool): True if significant bias detected.
            - ``bias_direction`` (str): ``"positive"``, ``"negative"``,
              or ``"none"``.
        """
        summary = residual_gp.residual_summary()
        mean_r = summary["mean"]
        is_biased = summary["has_systematic_bias"]

        if not is_biased:
            direction = "none"
        elif mean_r > 0:
            direction = "positive"
        else:
            direction = "negative"

        return {
            "mean_residual": mean_r,
            "is_biased": is_biased,
            "bias_direction": direction,
        }

    def failure_regions(
        self,
        hybrid_model: "HybridModel",
        X: list[list[float]],
        threshold: float = 2.0,
    ) -> list[dict]:
        """Find input regions where the theory significantly fails.

        An input ``x`` is a failure point if
        ``|residual_mean(x)| > threshold * residual_std(x)``.

        Parameters
        ----------
        hybrid_model : HybridModel
            A fitted hybrid model.
        X : list[list[float]]
            Input points to evaluate.
        threshold : float
            Multiplier for the significance threshold (default 2.0).

        Returns
        -------
        list[dict]
            Sorted by severity (descending).  Each dict has keys:
            ``index``, ``x``, ``residual_mean``, ``residual_std``,
            ``severity``.
        """
        # Avoid circular import
        from optimization_copilot.hybrid.composite import HybridModel as _HM

        res_mean, res_std = hybrid_model._gp.predict(X)

        failures: list[dict] = []
        for i in range(len(X)):
            mu = res_mean[i]
            sigma = res_std[i]
            if sigma < 1e-15:
                severity = abs(mu) / 1e-15
            else:
                severity = abs(mu) / sigma
            if severity > threshold:
                failures.append({
                    "index": i,
                    "x": X[i],
                    "residual_mean": mu,
                    "residual_std": sigma,
                    "severity": severity,
                })

        failures.sort(key=lambda d: d["severity"], reverse=True)
        return failures

    def model_adequacy_test(
        self,
        residuals: list[float],
        noise_std: float = 1.0,
    ) -> dict:
        """Simple chi-squared-like adequacy test.

        Computes ``Q = sum(r_i^2) / noise_std^2``.  Under an
        adequate model, ``Q ~ n`` (chi-squared with n dof).
        The model is considered adequate if ``Q / n < 2``.

        Parameters
        ----------
        residuals : list[float]
            Residual values.
        noise_std : float
            Expected noise standard deviation (default 1.0).

        Returns
        -------
        dict
            Keys: ``Q``, ``n``, ``Q_over_n``,
            ``is_adequate`` (True if ``Q / n < 2``).
        """
        n = len(residuals)
        if n == 0:
            return {"Q": 0.0, "n": 0, "Q_over_n": 0.0, "is_adequate": True}

        noise_var = noise_std ** 2
        if noise_var < 1e-15:
            noise_var = 1e-15
        Q = sum(r * r for r in residuals) / noise_var
        Q_over_n = Q / max(n, 1)

        return {
            "Q": Q,
            "n": n,
            "Q_over_n": Q_over_n,
            "is_adequate": Q_over_n < 2.0,
        }

    def suggest_theory_revision(
        self,
        failure_regions: list[dict],
        var_names: list[str] | None = None,
    ) -> list[str]:
        """Heuristic suggestions based on failure pattern analysis.

        Analyzes the failure regions to identify which input
        dimensions are associated with the largest discrepancies
        and generates human-readable revision suggestions.

        Parameters
        ----------
        failure_regions : list[dict]
            Output from :meth:`failure_regions`.
        var_names : list[str] or None
            Names for each input dimension.  If None, uses
            generic names (``"x_0"``, ``"x_1"``, ...).

        Returns
        -------
        list[str]
            Heuristic suggestions for theory revision.
        """
        if not failure_regions:
            return ["No significant failure regions detected. Theory appears adequate."]

        # Determine dimensionality from the first failure region
        dim = len(failure_regions[0]["x"])
        if var_names is None:
            var_names = [f"x_{i}" for i in range(dim)]

        suggestions: list[str] = []

        # Analyze which dimensions are extreme in failure regions
        for d in range(dim):
            vals = [fr["x"][d] for fr in failure_regions]
            if not vals:
                continue
            mean_val = sum(vals) / len(vals)
            min_val = min(vals)
            max_val = max(vals)

            # Check if failures cluster at high or low values
            all_vals_high = all(v > mean_val for v in vals) if len(vals) > 1 else False
            all_vals_low = all(v < mean_val for v in vals) if len(vals) > 1 else False

            name = var_names[d] if d < len(var_names) else f"x_{d}"

            if len(vals) >= 2:
                spread = max_val - min_val
                if spread < 1e-10:
                    # Failures at a single point
                    suggestions.append(
                        f"Theory may have a singularity or discontinuity "
                        f"near {name} = {mean_val:.4g}."
                    )
                elif all_vals_high:
                    suggestions.append(
                        f"Theory may need higher-order terms in {name} "
                        f"(failures cluster at high values)."
                    )
                elif all_vals_low:
                    suggestions.append(
                        f"Theory may need correction terms for low {name} "
                        f"(failures cluster at low values)."
                    )

        # Check overall bias direction
        mean_residual = sum(fr["residual_mean"] for fr in failure_regions) / len(
            failure_regions
        )
        if abs(mean_residual) > 1e-6:
            direction = "over-predicting" if mean_residual < 0 else "under-predicting"
            suggestions.append(
                f"Theory is systematically {direction} in failure regions "
                f"(mean residual = {mean_residual:.4g}). "
                f"Consider adding a correction offset or scaling factor."
            )

        # Severity analysis
        max_severity = max(fr["severity"] for fr in failure_regions)
        if max_severity > 5.0:
            suggestions.append(
                f"Maximum failure severity is {max_severity:.2f} sigma. "
                f"Theory may be fundamentally missing a mechanism in "
                f"this regime."
            )

        if not suggestions:
            suggestions.append(
                "Failure regions detected but no clear pattern identified. "
                "Consider collecting more data in the affected regions."
            )

        return suggestions

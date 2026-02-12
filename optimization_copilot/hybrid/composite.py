"""Composite hybrid model: theory(x) + GP_residual(x).

Combines a deterministic theory model with a data-driven residual
Gaussian Process to get the best of both worlds: physics-informed
predictions with data-driven uncertainty quantification.
"""

from __future__ import annotations

import math

from optimization_copilot.hybrid.theory import TheoryModel
from optimization_copilot.hybrid.residual import ResidualGP


class HybridModel:
    """Hybrid model: ``theory(x) + GP_residual(x)``.

    The theory model provides a physics-based mean prediction,
    while the residual GP learns systematic deviations and provides
    calibrated uncertainty estimates.

    Parameters
    ----------
    theory_model : TheoryModel
        The deterministic theory component.
    residual_gp : ResidualGP
        The residual GP (wrapping the same theory model).
    """

    def __init__(
        self,
        theory_model: TheoryModel,
        residual_gp: ResidualGP,
    ) -> None:
        self._theory = theory_model
        self._gp = residual_gp
        self._X_train: list[list[float]] = []
        self._y_train: list[float] = []

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        """Fit the residual GP on training data.

        Parameters
        ----------
        X : list[list[float]]
            Training inputs.
        y : list[float]
            Observed outputs.
        """
        self._X_train = [list(row) for row in X]
        self._y_train = list(y)
        self._gp.fit(X, y)

    def predict_with_uncertainty(
        self, X_new: list[list[float]]
    ) -> tuple[list[float], list[float]]:
        """Predict mean and standard deviation at new points.

        ``mean = theory(X) + gp_residual_mean(X)``
        ``std = gp_residual_std(X)``  (theory is deterministic)

        Parameters
        ----------
        X_new : list[list[float]]
            Query inputs (m rows).

        Returns
        -------
        tuple[list[float], list[float]]
            ``(mean, std)`` each of length m.
        """
        y_theory = self._theory.predict(X_new)
        res_mean, res_std = self._gp.predict(X_new)
        means = [y_theory[i] + res_mean[i] for i in range(len(X_new))]
        return means, res_std

    def suggest_next(
        self,
        X_candidates: list[list[float]],
        acquisition: str = "ei",
        best_y: float | None = None,
    ) -> list[dict]:
        """Rank candidates by acquisition function value.

        Parameters
        ----------
        X_candidates : list[list[float]]
            Candidate input points.
        acquisition : str
            Acquisition function: ``"ei"`` (Expected Improvement) or
            ``"ucb"`` (Upper Confidence Bound).  Default ``"ei"``.
        best_y : float or None
            Best observed objective value (for EI).  If None, uses
            the minimum of training y values.

        Returns
        -------
        list[dict]
            Sorted list of dicts with keys: ``index``, ``x``,
            ``mean``, ``std``, ``acquisition_value``.
        """
        from optimization_copilot.backends._math.stats import norm_cdf, norm_pdf

        means, stds = self.predict_with_uncertainty(X_candidates)

        if best_y is None and self._y_train:
            best_y = min(self._y_train)
        elif best_y is None:
            best_y = 0.0

        results: list[dict] = []
        for i in range(len(X_candidates)):
            mu = means[i]
            sigma = stds[i]

            if acquisition == "ei":
                # Expected Improvement (minimization)
                if sigma < 1e-12:
                    acq_val = 0.0
                else:
                    z = (best_y - mu) / sigma
                    acq_val = (best_y - mu) * norm_cdf(z) + sigma * norm_pdf(z)
            elif acquisition == "ucb":
                # Upper Confidence Bound (for minimization: lower is better)
                # Return negative UCB so that higher acq_val = better candidate
                kappa = 2.0
                acq_val = -(mu - kappa * sigma)
            else:
                raise ValueError(
                    f"Unknown acquisition function: {acquisition!r}. "
                    "Supported: 'ei', 'ucb'"
                )

            results.append({
                "index": i,
                "x": X_candidates[i],
                "mean": mu,
                "std": sigma,
                "acquisition_value": acq_val,
            })

        # Sort by acquisition value descending (higher = better)
        results.sort(key=lambda d: d["acquisition_value"], reverse=True)
        return results

    def compare_to_theory_only(
        self,
        X_test: list[list[float]],
        y_test: list[float],
    ) -> dict:
        """RMSE comparison: theory-only versus hybrid model.

        Parameters
        ----------
        X_test : list[list[float]]
            Test inputs.
        y_test : list[float]
            True test outputs.

        Returns
        -------
        dict
            Keys: ``theory_rmse``, ``hybrid_rmse``,
            ``improvement_pct`` (positive means hybrid is better).
        """
        n = len(y_test)
        y_theory = self._theory.predict(X_test)
        y_hybrid, _ = self.predict_with_uncertainty(X_test)

        # Theory RMSE
        theory_se = sum((y_test[i] - y_theory[i]) ** 2 for i in range(n))
        theory_rmse = math.sqrt(theory_se / max(n, 1))

        # Hybrid RMSE
        hybrid_se = sum((y_test[i] - y_hybrid[i]) ** 2 for i in range(n))
        hybrid_rmse = math.sqrt(hybrid_se / max(n, 1))

        # Improvement percentage
        if theory_rmse > 1e-15:
            improvement_pct = (theory_rmse - hybrid_rmse) / theory_rmse * 100.0
        else:
            improvement_pct = 0.0

        return {
            "theory_rmse": theory_rmse,
            "hybrid_rmse": hybrid_rmse,
            "improvement_pct": improvement_pct,
        }

    def theory_adequacy_score(self) -> float:
        """How adequate is the theory model?

        Computes ``1.0 - std(residuals) / std(y_train)``.
        A high score (close to 1.0) means the theory captures most
        of the variance and the GP contribution is small.
        A low score means the theory is inadequate and the GP is
        doing most of the work.

        Returns
        -------
        float
            Adequacy score in [0, 1].  Returns 1.0 if there is
            no training data or zero variance.
        """
        if not self._y_train or len(self._y_train) < 2:
            return 1.0

        n = len(self._y_train)
        y_mean = sum(self._y_train) / n
        y_var = sum((y - y_mean) ** 2 for y in self._y_train) / max(n - 1, 1)
        y_std = math.sqrt(y_var)

        if y_std < 1e-15:
            return 1.0

        residuals = self._gp.residuals
        if not residuals:
            return 1.0

        r_mean = sum(residuals) / len(residuals)
        r_var = sum((r - r_mean) ** 2 for r in residuals) / max(len(residuals) - 1, 1)
        r_std = math.sqrt(r_var)

        ratio = r_std / y_std
        return max(0.0, min(1.0, 1.0 - ratio))

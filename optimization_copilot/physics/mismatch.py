"""Theory mismatch detection with auto-degradation recommendations.

Combines four diagnostic signals — chi-squared Q-test, systematic bias,
trend in residuals, and model adequacy — to determine whether a physics
theory model should be kept, revised, or replaced by a data-driven approach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MismatchReport:
    """Result of theory mismatch analysis."""
    q_statistic: float           # sum(r²) / noise_var
    q_over_n: float              # q_statistic / n
    mean_residual: float
    residual_std: float
    has_systematic_bias: bool    # |mean(r)| > 2 * SE
    has_trend: bool              # |pearson_r(y_hat, r)| > 0.5
    trend_correlation: float     # pearson_r(y_hat, residuals)
    adequacy_score: float        # 1 - std(r)/std(y), in [0,1]
    is_mismatched: bool          # overall verdict
    recommendation: str          # "keep_hybrid" | "revise_theory" | "fallback_data_driven"


class MismatchDetector:
    """Detect theory-observation mismatch from residuals.

    Parameters
    ----------
    q_threshold : float
        Q/n threshold for mismatch (default 3.0).
    trend_threshold : float
        |pearson_r| threshold for trend detection (default 0.5).
    adequacy_low : float
        Adequacy score below this triggers concern (default 0.3).
    noise_var : float
        Assumed observation noise variance (default 1.0).
    """

    def __init__(
        self,
        q_threshold: float = 3.0,
        trend_threshold: float = 0.5,
        adequacy_low: float = 0.3,
        noise_var: float = 1.0,
    ) -> None:
        self._q_threshold = q_threshold
        self._trend_threshold = trend_threshold
        self._adequacy_low = adequacy_low
        self._noise_var = noise_var

    def detect(
        self,
        y_observed: list[float],
        y_predicted: list[float],
    ) -> MismatchReport:
        """Analyze residuals and produce a mismatch report.

        Parameters
        ----------
        y_observed : list[float]
            Observed (true) values.
        y_predicted : list[float]
            Theory model predictions.

        Returns
        -------
        MismatchReport
        """
        if len(y_observed) != len(y_predicted):
            raise ValueError("y_observed and y_predicted must have equal length")

        n = len(y_observed)
        if n == 0:
            return MismatchReport(
                q_statistic=0.0, q_over_n=0.0,
                mean_residual=0.0, residual_std=0.0,
                has_systematic_bias=False, has_trend=False,
                trend_correlation=0.0, adequacy_score=1.0,
                is_mismatched=False, recommendation="keep_hybrid",
            )

        # Compute residuals
        residuals = [y_observed[i] - y_predicted[i] for i in range(n)]

        # Signal 1: Q-test (chi-squared goodness of fit)
        q_stat = sum(r * r for r in residuals) / max(self._noise_var, 1e-15)
        q_over_n = q_stat / max(n, 1)

        # Signal 2: Systematic bias
        mean_r = sum(residuals) / n
        var_r = sum((r - mean_r) ** 2 for r in residuals) / max(n - 1, 1)
        std_r = math.sqrt(var_r)
        se = std_r / math.sqrt(n) if n > 0 else 0.0
        has_bias = abs(mean_r) > 2.0 * se if se > 1e-15 else (abs(mean_r) > 1e-15)

        # Signal 3: Trend in residuals (Pearson r between y_predicted and residuals)
        trend_corr = self._pearson_r(y_predicted, residuals)
        has_trend = abs(trend_corr) > self._trend_threshold

        # Signal 4: Adequacy score = 1 - std(residuals) / std(y_observed)
        y_mean = sum(y_observed) / n
        y_var = sum((y - y_mean) ** 2 for y in y_observed) / max(n - 1, 1)
        y_std = math.sqrt(y_var)

        if y_std > 1e-15:
            adequacy = max(0.0, min(1.0, 1.0 - std_r / y_std))
        else:
            adequacy = 1.0  # No variance in observations

        # Overall mismatch verdict
        is_mismatched = (
            q_over_n > self._q_threshold
            or adequacy < self._adequacy_low
            or (has_bias and has_trend)
        )

        # Recommendation logic
        if adequacy < 0.1 or q_over_n > 5.0:
            recommendation = "fallback_data_driven"
        elif has_trend and adequacy > 0.1:
            recommendation = "revise_theory"
        else:
            recommendation = "keep_hybrid"

        return MismatchReport(
            q_statistic=q_stat,
            q_over_n=q_over_n,
            mean_residual=mean_r,
            residual_std=std_r,
            has_systematic_bias=has_bias,
            has_trend=has_trend,
            trend_correlation=trend_corr,
            adequacy_score=adequacy,
            is_mismatched=is_mismatched,
            recommendation=recommendation,
        )

    @staticmethod
    def _pearson_r(x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient between x and y."""
        n = len(x)
        if n < 2:
            return 0.0

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        cov = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        var_x = sum((xi - x_mean) ** 2 for xi in x)
        var_y = sum((yi - y_mean) ** 2 for yi in y)

        denom = math.sqrt(var_x * var_y)
        if denom < 1e-15:
            return 0.0

        return cov / denom

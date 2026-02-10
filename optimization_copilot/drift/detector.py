"""Concept Drift Detection for optimization campaigns.

Detects when the "world has changed" -- same parameters give different KPI
results over time due to instrument drift, batch effects, or environmental
changes.  Uses only stdlib + math (no numpy/scipy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import CampaignSnapshot, Observation, Phase


# ── Data Structures ───────────────────────────────────────


@dataclass
class DriftReport:
    """Result of a drift detection analysis."""

    drift_detected: bool
    drift_score: float  # 0.0 = no drift, 1.0 = severe drift
    drift_type: str  # "none", "gradual", "sudden", "recurring"
    affected_parameters: list[str]
    recommended_action: str  # "continue", "reweight", "re_screen", "re_learn", "restart"
    window_stats: dict[str, Any] = field(default_factory=dict)


# ── Helpers (pure-stdlib statistics) ──────────────────────


def _mean(values: list[float]) -> float:
    """Arithmetic mean.  Returns 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float], ddof: int = 1) -> float:
    """Sample standard deviation.  Returns 0.0 when not enough data."""
    n = len(values)
    if n <= ddof:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (n - ddof)
    return math.sqrt(variance)


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient.  Returns 0.0 on degenerate input."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    sx, sy = _std(xs), _std(ys)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sx * sy)


def _pooled_std(std1: float, n1: int, std2: float, n2: int) -> float:
    """Pooled standard deviation from two groups."""
    if n1 + n2 <= 2:
        return 0.0
    pooled_var = ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)
    return math.sqrt(pooled_var)


# ── Drift Detector ────────────────────────────────────────


class DriftDetector:
    """Multi-strategy concept drift detector for optimization campaigns.

    Uses three complementary strategies:
      1. KPI distribution shift (two-sample mean comparison)
      2. Parameter-KPI relationship drift (correlation stability)
      3. Residual analysis (simple linear model deviation)

    Parameters
    ----------
    reference_window : int
        Number of recent observations for the *reference* (older) window.
    test_window : int
        Number of most-recent observations for the *test* (newer) window.
    significance : float
        Significance level for drift decisions (lower = stricter).
        Controls the test-statistic threshold internally.
    """

    # Test-statistic thresholds mapped from significance levels.
    _SIGNIFICANCE_TO_THRESHOLD = {
        0.01: 2.58,
        0.05: 2.00,
        0.10: 1.65,
    }

    def __init__(
        self,
        reference_window: int = 10,
        test_window: int = 10,
        significance: float = 0.05,
    ) -> None:
        self.reference_window = reference_window
        self.test_window = test_window
        self.significance = significance
        # Pick the closest threshold; default to 2.0 for 0.05.
        self._threshold = self._SIGNIFICANCE_TO_THRESHOLD.get(
            significance, 2.0
        )

    # ── Public API ────────────────────────────────────────

    def detect(self, snapshot: CampaignSnapshot) -> DriftReport:
        """Run all drift detection strategies and return a consolidated report."""
        obs = snapshot.successful_observations
        total = len(obs)
        needed = self.reference_window + self.test_window

        # Not enough data to analyse.
        if total < needed:
            return DriftReport(
                drift_detected=False,
                drift_score=0.0,
                drift_type="none",
                affected_parameters=[],
                recommended_action="continue",
                window_stats={"reason": "insufficient_data", "n_obs": total, "needed": needed},
            )

        ref_obs = obs[total - needed: total - self.test_window]
        test_obs = obs[total - self.test_window:]
        primary_kpi = snapshot.objective_names[0]

        # Strategy (a): KPI distribution shift
        kpi_score, kpi_stats = self._kpi_distribution_shift(
            ref_obs, test_obs, primary_kpi
        )

        # Strategy (b): Parameter-KPI relationship drift
        param_scores, affected = self._parameter_relationship_drift(
            ref_obs, test_obs, primary_kpi, snapshot.parameter_names
        )

        # Strategy (c): Residual analysis
        residual_score, residual_stats = self._residual_analysis(
            ref_obs, test_obs, primary_kpi, snapshot.parameter_names
        )

        # Combine scores (max of the three strategies).
        combined_score = max(kpi_score, max(param_scores) if param_scores else 0.0, residual_score)
        combined_score = min(combined_score, 1.0)

        # Classify drift type using a rolling score history approximation.
        rolling_scores = self._approximate_rolling_scores(obs, primary_kpi)
        drift_type = self.classify_drift_type(rolling_scores)

        drift_detected = combined_score >= 0.3  # mild threshold

        report = DriftReport(
            drift_detected=drift_detected,
            drift_score=combined_score,
            drift_type=drift_type if drift_detected else "none",
            affected_parameters=affected,
            recommended_action="continue",  # filled below
            window_stats={
                "kpi_score": kpi_score,
                "param_scores": dict(zip(snapshot.parameter_names, param_scores)),
                "residual_score": residual_score,
                "combined_score": combined_score,
                **kpi_stats,
                **residual_stats,
            },
        )
        report.recommended_action = self.recommend_action(report, Phase.LEARNING)
        return report

    def classify_drift_type(self, scores: list[float]) -> str:
        """Classify drift pattern from a series of drift scores.

        Parameters
        ----------
        scores : list[float]
            A time-ordered series of per-window drift scores.

        Returns
        -------
        str
            One of ``"none"``, ``"sudden"``, ``"gradual"``, ``"recurring"``.
        """
        if not scores:
            return "none"

        high_threshold = 0.3
        n_high = sum(1 for s in scores if s >= high_threshold)
        ratio_high = n_high / len(scores)

        if ratio_high < 0.15:
            return "none"

        # Check for sudden spike: last score(s) high but earlier ones low.
        n = len(scores)
        if n >= 3:
            tail = scores[-max(1, n // 4):]
            head = scores[: n // 2]
            tail_high = sum(1 for s in tail if s >= high_threshold) / max(len(tail), 1)
            head_high = sum(1 for s in head if s >= high_threshold) / max(len(head), 1)
            if tail_high >= 0.6 and head_high < 0.2:
                # Distinguish sudden spike from gradual ramp: a truly sudden
                # change concentrates most of the increase in one step, while
                # a gradual ramp spreads the increase evenly.
                diffs = [scores[i + 1] - scores[i] for i in range(n - 1)]
                total_increase = scores[-1] - scores[0]
                max_step = max(diffs) if diffs else 0.0
                if total_increase > 0 and max_step / total_increase < 0.5:
                    pass  # Smooth ramp → fall through to gradual check
                else:
                    return "sudden"

        # Check for gradual increase: monotonically rising trend.
        if n >= 4:
            first_half_mean = _mean(scores[: n // 2])
            second_half_mean = _mean(scores[n // 2:])
            if second_half_mean > first_half_mean + 0.1:
                # Additionally check that it isn't oscillating.
                diffs = [scores[i + 1] - scores[i] for i in range(n - 1)]
                sign_changes = sum(
                    1
                    for i in range(len(diffs) - 1)
                    if diffs[i] * diffs[i + 1] < 0
                )
                if sign_changes < len(diffs) * 0.5:
                    return "gradual"

        # Oscillation pattern: multiple rises and falls.
        if n >= 4:
            diffs = [scores[i + 1] - scores[i] for i in range(n - 1)]
            sign_changes = sum(
                1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0
            )
            if sign_changes >= max(2, len(diffs) * 0.4):
                return "recurring"

        # Default to gradual if we saw some high scores.
        if ratio_high >= 0.15:
            return "gradual"

        return "none"

    def recommend_action(self, report: DriftReport, current_phase: Phase) -> str:
        """Recommend a corrective action based on drift severity and phase.

        Returns
        -------
        str
            One of ``"continue"``, ``"reweight"``, ``"re_screen"``,
            ``"re_learn"``, ``"restart"``.
        """
        score = report.drift_score

        if not report.drift_detected or score < 0.3:
            return "continue"

        # Mild drift (0.3 <= score < 0.5).
        if score < 0.5:
            if current_phase in (Phase.LEARNING, Phase.EXPLOITATION):
                return "reweight"
            return "continue"

        # Moderate drift (0.5 <= score < 0.7).
        if score < 0.7:
            return "re_screen"

        # Severe drift (>= 0.7).
        if current_phase == Phase.EXPLOITATION:
            return "restart"
        return "re_learn"

    def compute_drift_aware_window(self, snapshot: CampaignSnapshot) -> int:
        """Return an optimal lookback window size for modelling.

        If drift is detected the window shrinks to focus on recent data;
        otherwise the full history is used.
        """
        report = self.detect(snapshot)
        n = len(snapshot.successful_observations)

        if not report.drift_detected:
            return n

        # Scale window inversely with drift severity.
        # At score=0.3 keep ~80% of data; at score=1.0 keep ~25%.
        keep_ratio = max(0.25, 1.0 - 0.75 * report.drift_score)
        window = max(self.test_window, int(n * keep_ratio))
        return min(window, n)

    # ── Private detection strategies ──────────────────────

    def _extract_kpi(self, obs: list[Observation], kpi_name: str) -> list[float]:
        """Extract a flat list of KPI values from observations."""
        return [o.kpi_values[kpi_name] for o in obs if kpi_name in o.kpi_values]

    def _extract_param(self, obs: list[Observation], param_name: str) -> list[float]:
        """Extract parameter values, coercing to float where possible."""
        values: list[float] = []
        for o in obs:
            v = o.parameters.get(param_name)
            if v is not None:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    pass
        return values

    def _kpi_distribution_shift(
        self,
        ref_obs: list[Observation],
        test_obs: list[Observation],
        kpi_name: str,
    ) -> tuple[float, dict[str, Any]]:
        """Compare KPI distributions between reference and test windows.

        Returns (score_0_to_1, stats_dict).
        """
        ref_kpi = self._extract_kpi(ref_obs, kpi_name)
        test_kpi = self._extract_kpi(test_obs, kpi_name)

        ref_mean = _mean(ref_kpi)
        test_mean = _mean(test_kpi)
        ref_std = _std(ref_kpi)
        test_std = _std(test_kpi)
        pooled = _pooled_std(ref_std, len(ref_kpi), test_std, len(test_kpi))

        if pooled == 0.0:
            # Identical values -- no drift detectable.
            test_stat = 0.0
        else:
            test_stat = abs(ref_mean - test_mean) / pooled

        # Normalise into [0, 1]: 0 at stat=0, 1 at stat>=4.
        score = min(test_stat / 4.0, 1.0)

        # Also compute a Welch-like score (more robust with unequal variances).
        # The pooled t-test can under-report drift when one window has much
        # higher variance (e.g. the reference straddles a change point).
        n1, n2 = len(ref_kpi), len(test_kpi)
        if n1 > 0 and n2 > 0:
            se = math.sqrt(ref_std ** 2 / n1 + test_std ** 2 / n2)
            if se > 0:
                welch_stat = abs(ref_mean - test_mean) / se
                welch_score = min(welch_stat / 4.0, 1.0)
                score = max(score, welch_score)
            elif abs(ref_mean - test_mean) > 1e-12:
                # Both windows have zero variance but different means.
                score = max(score, 0.95)

        stats = {
            "ref_mean": ref_mean,
            "test_mean": test_mean,
            "ref_std": ref_std,
            "test_std": test_std,
            "pooled_std": pooled,
            "kpi_test_statistic": test_stat,
        }
        return score, stats

    def _parameter_relationship_drift(
        self,
        ref_obs: list[Observation],
        test_obs: list[Observation],
        kpi_name: str,
        param_names: list[str],
    ) -> tuple[list[float], list[str]]:
        """Check if parameter-KPI correlations changed between windows.

        Returns (per_param_scores, affected_param_names).
        """
        ref_kpi = self._extract_kpi(ref_obs, kpi_name)
        test_kpi = self._extract_kpi(test_obs, kpi_name)

        scores: list[float] = []
        affected: list[str] = []

        for pname in param_names:
            ref_p = self._extract_param(ref_obs, pname)
            test_p = self._extract_param(test_obs, pname)

            # Need matched lengths.
            ref_len = min(len(ref_p), len(ref_kpi))
            test_len = min(len(test_p), len(test_kpi))

            if ref_len < 3 or test_len < 3:
                scores.append(0.0)
                continue

            r_ref = _pearson_r(ref_p[:ref_len], ref_kpi[:ref_len])
            r_test = _pearson_r(test_p[:test_len], test_kpi[:test_len])

            # Sign flip is a strong signal.
            sign_flip = (r_ref * r_test) < 0 and abs(r_ref) > 0.2 and abs(r_test) > 0.2
            magnitude_change = abs(r_ref - r_test)

            param_score = 0.0
            if sign_flip:
                param_score = min(0.5 + magnitude_change, 1.0)
            else:
                # Magnitude change alone: normalise 0-1 over a 0-1.5 range.
                param_score = min(magnitude_change / 1.5, 1.0)

            scores.append(param_score)
            if param_score >= 0.3:
                affected.append(pname)

        return scores, affected

    def _residual_analysis(
        self,
        ref_obs: list[Observation],
        test_obs: list[Observation],
        kpi_name: str,
        param_names: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """Fit a simple linear model on reference data and check residuals on
        test data.

        The "model" is just: predicted_kpi = ref_mean_kpi (intercept-only) plus
        a slope correction from the first numeric parameter if available.
        Returns (score, stats).
        """
        ref_kpi = self._extract_kpi(ref_obs, kpi_name)
        test_kpi = self._extract_kpi(test_obs, kpi_name)
        ref_mean_kpi = _mean(ref_kpi)

        # Try to build a simple slope from first parameter.
        slope = 0.0
        intercept = ref_mean_kpi
        model_param: str | None = None

        for pname in param_names:
            ref_p = self._extract_param(ref_obs, pname)
            n = min(len(ref_p), len(ref_kpi))
            if n < 3:
                continue
            xs, ys = ref_p[:n], ref_kpi[:n]
            mx, my = _mean(xs), _mean(ys)
            sx = _std(xs)
            if sx == 0.0:
                continue
            # Simple OLS slope.
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            denom = sum((x - mx) ** 2 for x in xs)
            if denom == 0.0:
                continue
            slope = num / denom
            intercept = my - slope * mx
            model_param = pname
            break  # Use first viable parameter.

        # Compute residuals on test window.
        residuals: list[float] = []
        for obs in test_obs:
            actual = obs.kpi_values.get(kpi_name)
            if actual is None:
                continue
            if model_param is not None:
                pval = obs.parameters.get(model_param)
                try:
                    predicted = intercept + slope * float(pval)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    predicted = ref_mean_kpi
            else:
                predicted = ref_mean_kpi
            residuals.append(actual - predicted)

        if not residuals:
            return 0.0, {"residual_mean": 0.0, "residual_std": 0.0}

        res_mean = _mean(residuals)
        res_std = _std(residuals)
        ref_std = _std(ref_kpi)

        # Normalise: if residual mean is large relative to the reference spread.
        denominator = max(ref_std, res_std, 1e-12)
        test_stat = abs(res_mean) / denominator
        score = min(test_stat / 3.0, 1.0)

        stats = {
            "residual_mean": res_mean,
            "residual_std": res_std,
            "residual_test_statistic": test_stat,
            "model_param": model_param,
        }
        return score, stats

    def _approximate_rolling_scores(
        self, obs: list[Observation], kpi_name: str
    ) -> list[float]:
        """Compute a rough per-segment drift score for drift-type classification.

        Splits the observation history into overlapping segments and computes
        a simplified KPI-shift score for each.
        """
        n = len(obs)
        segment_size = max(self.reference_window, 5)
        step = max(segment_size // 2, 2)

        if n < segment_size * 2:
            return []

        scores: list[float] = []
        baseline_obs = obs[:segment_size]
        baseline_kpi = self._extract_kpi(baseline_obs, kpi_name)
        baseline_mean = _mean(baseline_kpi)
        baseline_std = _std(baseline_kpi)

        idx = segment_size
        while idx + segment_size <= n:
            seg_obs = obs[idx: idx + segment_size]
            seg_kpi = self._extract_kpi(seg_obs, kpi_name)
            seg_mean = _mean(seg_kpi)
            seg_std = _std(seg_kpi)
            pooled = _pooled_std(baseline_std, len(baseline_kpi), seg_std, len(seg_kpi))
            if pooled > 0:
                t = abs(baseline_mean - seg_mean) / pooled
                scores.append(min(t / 4.0, 1.0))
            else:
                scores.append(0.0)
            idx += step

        return scores

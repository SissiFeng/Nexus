"""Data Quality Intelligence engine — noise decomposition, batch effects, instrument drift, credibility weights.

All algorithms are pure Python (math module only) and fully deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import CampaignSnapshot, Observation


# ── Dataclasses ───────────────────────────────────────────────


@dataclass
class NoiseDecomposition:
    """Result of noise decomposition into measurement vs. systematic components."""

    measurement_noise: float
    systematic_noise: float
    total_noise: float
    ratio: float  # measurement / total; 0 = all systematic, 1 = all measurement
    n_regions: int
    n_observations_per_region: float


@dataclass
class BatchEffect:
    """Result of temporal batch-effect detection (one-way ANOVA F-test)."""

    detected: bool
    batch_means: list[float]
    batch_sizes: list[int]
    f_statistic: float
    effect_size: float  # eta-squared
    n_batches: int


@dataclass
class InstrumentDrift:
    """Result of instrument-drift detection via residual trend analysis."""

    detected: bool
    drift_slope: float
    drift_intercept: float
    r_squared: float
    residual_trend_direction: str  # "increasing" / "decreasing" / "stable"


@dataclass
class DataQualityReport:
    """Comprehensive data-quality report produced by the engine."""

    noise_decomposition: NoiseDecomposition
    batch_effect: BatchEffect
    instrument_drift: InstrumentDrift
    credibility_weights: dict[int, float]  # observation iteration -> weight
    overall_quality_score: float  # 0-1
    issues: list[str] = field(default_factory=list)


# ── Helper functions (pure Python, no external deps) ─────────


def _mean(values: list[float]) -> float:
    """Arithmetic mean. Returns 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    """Population variance. Returns 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def _std(values: list[float]) -> float:
    """Population standard deviation."""
    return math.sqrt(_variance(values))


def _linreg(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Simple linear regression. Returns (slope, intercept).

    Falls back to (0.0, mean(ys)) when regression is undefined.
    """
    n = len(xs)
    if n < 2 or len(ys) < 2:
        return 0.0, _mean(ys)
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    if ss_xx == 0.0:
        return 0.0, y_mean
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _r_squared(
    xs: list[float], ys: list[float], slope: float, intercept: float
) -> float:
    """Coefficient of determination (R^2) for a linear fit."""
    if len(ys) < 2:
        return 0.0
    y_mean = _mean(ys)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    if ss_tot == 0.0:
        return 0.0
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return max(0.0, 1.0 - ss_res / ss_tot)


# ── Engine ────────────────────────────────────────────────────


class DataQualityEngine:
    """Analyses a *CampaignSnapshot* for noise, batch effects, drift, and
    produces observation-level credibility weights plus an overall quality score.
    """

    def __init__(
        self,
        n_regions: int = 5,
        batch_window: int = 10,
        min_observations: int = 6,
        f_stat_threshold: float = 3.0,
        drift_r2_threshold: float = 0.3,
    ) -> None:
        self.n_regions = n_regions
        self.batch_window = batch_window
        self.min_observations = min_observations
        self.f_stat_threshold = f_stat_threshold
        self.drift_r2_threshold = drift_r2_threshold

    # ── public API ────────────────────────────────────────────

    def analyze(self, snapshot: CampaignSnapshot) -> DataQualityReport:
        """Run every sub-analysis and assemble a *DataQualityReport*.

        Returns a safe default report when the snapshot is empty or has fewer
        observations than *min_observations*.
        """
        obs = snapshot.observations
        if not obs or len(obs) < self.min_observations:
            empty_noise = NoiseDecomposition(
                measurement_noise=0.0,
                systematic_noise=0.0,
                total_noise=0.0,
                ratio=0.5,
                n_regions=0,
                n_observations_per_region=0.0,
            )
            empty_batch = BatchEffect(
                detected=False,
                batch_means=[],
                batch_sizes=[],
                f_statistic=0.0,
                effect_size=0.0,
                n_batches=0,
            )
            empty_drift = InstrumentDrift(
                detected=False,
                drift_slope=0.0,
                drift_intercept=0.0,
                r_squared=0.0,
                residual_trend_direction="stable",
            )
            return DataQualityReport(
                noise_decomposition=empty_noise,
                batch_effect=empty_batch,
                instrument_drift=empty_drift,
                credibility_weights={},
                overall_quality_score=1.0,
                issues=[],
            )

        noise = self._decompose_noise(snapshot)
        batch = self._detect_batch_effects(snapshot)
        drift = self._detect_instrument_drift(snapshot)
        weights = self._compute_credibility_weights(snapshot, noise, batch, drift)
        score = self._compute_quality_score(noise, batch, drift)
        issues = self._collect_issues(noise, batch, drift)

        return DataQualityReport(
            noise_decomposition=noise,
            batch_effect=batch,
            instrument_drift=drift,
            credibility_weights=weights,
            overall_quality_score=score,
            issues=issues,
        )

    # ── noise decomposition ───────────────────────────────────

    def _decompose_noise(self, snapshot: CampaignSnapshot) -> NoiseDecomposition:
        """Decompose KPI variance into measurement (within-bin) and systematic
        (between-bin) noise using parameter-space binning."""
        successful = snapshot.successful_observations
        if not successful or not snapshot.objective_names:
            return NoiseDecomposition(0.0, 0.0, 0.0, 0.5, 0, 0.0)

        kpi_name = snapshot.objective_names[0]

        # Find the first numeric parameter with defined bounds.
        param_spec = None
        for spec in snapshot.parameter_specs:
            if spec.lower is not None and spec.upper is not None:
                param_spec = spec
                break
        if param_spec is None:
            return NoiseDecomposition(0.0, 0.0, 0.0, 0.5, 0, 0.0)

        lower = param_spec.lower
        upper = param_spec.upper
        if upper <= lower:
            return NoiseDecomposition(0.0, 0.0, 0.0, 0.5, 0, 0.0)

        bin_width = (upper - lower) / self.n_regions

        # Assign observations to bins.
        bins: dict[int, list[float]] = {}
        for obs in successful:
            pval = obs.parameters.get(param_spec.name)
            if pval is None:
                continue
            try:
                pval_f = float(pval)
            except (TypeError, ValueError):
                continue
            idx = int((pval_f - lower) / bin_width)
            idx = max(0, min(idx, self.n_regions - 1))
            kpi_val = obs.kpi_values.get(kpi_name)
            if kpi_val is None:
                continue
            bins.setdefault(idx, []).append(kpi_val)

        # Compute within-bin variances (only for bins with >=2 observations).
        within_variances: list[float] = []
        bin_means: list[float] = []
        populated_bin_sizes: list[int] = []
        for kpi_vals in bins.values():
            if len(kpi_vals) >= 2:
                within_variances.append(_variance(kpi_vals))
            if kpi_vals:
                bin_means.append(_mean(kpi_vals))
                populated_bin_sizes.append(len(kpi_vals))

        all_kpi = [
            obs.kpi_values[kpi_name]
            for obs in successful
            if kpi_name in obs.kpi_values
        ]
        grand_mean = _mean(all_kpi)
        total_noise_var = _variance(all_kpi)
        measurement_noise_var = _mean(within_variances) if within_variances else 0.0
        systematic_noise_var = _variance(bin_means) if len(bin_means) >= 2 else 0.0

        # Convert variances to CV (coefficient of variation).
        abs_grand = abs(grand_mean)
        if abs_grand > 0.0:
            measurement_noise = math.sqrt(measurement_noise_var) / abs_grand
            systematic_noise = math.sqrt(systematic_noise_var) / abs_grand
            total_noise = math.sqrt(total_noise_var) / abs_grand
        else:
            measurement_noise = 0.0
            systematic_noise = 0.0
            total_noise = 0.0

        ratio = measurement_noise / total_noise if total_noise > 0.0 else 0.5

        n_pop = len(populated_bin_sizes)
        avg_per_region = (
            sum(populated_bin_sizes) / n_pop if n_pop > 0 else 0.0
        )

        return NoiseDecomposition(
            measurement_noise=measurement_noise,
            systematic_noise=systematic_noise,
            total_noise=total_noise,
            ratio=ratio,
            n_regions=n_pop,
            n_observations_per_region=avg_per_region,
        )

    # ── batch effect detection ────────────────────────────────

    def _detect_batch_effects(self, snapshot: CampaignSnapshot) -> BatchEffect:
        """One-way ANOVA F-test on temporally-ordered observation batches."""
        successful = snapshot.successful_observations
        if not successful or not snapshot.objective_names:
            return BatchEffect(False, [], [], 0.0, 0.0, 0)

        kpi_name = snapshot.objective_names[0]
        sorted_obs = sorted(successful, key=lambda o: o.iteration)
        kpi_values = [
            o.kpi_values[kpi_name]
            for o in sorted_obs
            if kpi_name in o.kpi_values
        ]

        if len(kpi_values) < 2:
            return BatchEffect(False, [], [], 0.0, 0.0, 0)

        # Split into temporal batches.
        batches: list[list[float]] = []
        for i in range(0, len(kpi_values), self.batch_window):
            batch = kpi_values[i : i + self.batch_window]
            batches.append(batch)

        # Last batch must have >= 2 observations; merge with previous if not.
        if len(batches) > 1 and len(batches[-1]) < 2:
            batches[-2].extend(batches[-1])
            batches.pop()

        k = len(batches)
        if k < 2:
            b_means = [_mean(b) for b in batches]
            b_sizes = [len(b) for b in batches]
            return BatchEffect(False, b_means, b_sizes, 0.0, 0.0, k)

        batch_means = [_mean(b) for b in batches]
        batch_sizes = [len(b) for b in batches]
        n_total = sum(batch_sizes)
        grand_mean = _mean(kpi_values)

        ss_between = sum(
            n_j * (m_j - grand_mean) ** 2
            for n_j, m_j in zip(batch_sizes, batch_means)
        )
        ss_within = sum(
            sum((x - m_j) ** 2 for x in batch)
            for batch, m_j in zip(batches, batch_means)
        )
        ss_total = ss_between + ss_within

        if n_total > k and k > 1:
            df_between = k - 1
            df_within = n_total - k
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within if df_within > 0 else 0.0
            f_stat = ms_between / ms_within if ms_within > 0.0 else 0.0
        else:
            f_stat = 0.0

        effect_size = ss_between / ss_total if ss_total > 0.0 else 0.0
        detected = f_stat > self.f_stat_threshold and k > 1

        return BatchEffect(
            detected=detected,
            batch_means=batch_means,
            batch_sizes=batch_sizes,
            f_statistic=f_stat,
            effect_size=effect_size,
            n_batches=k,
        )

    # ── instrument drift detection ────────────────────────────

    def _detect_instrument_drift(self, snapshot: CampaignSnapshot) -> InstrumentDrift:
        """Detect drift by fitting a linear model to KPI residuals over time."""
        successful = snapshot.successful_observations
        if not successful or not snapshot.objective_names:
            return InstrumentDrift(False, 0.0, 0.0, 0.0, "stable")

        kpi_name = snapshot.objective_names[0]

        # Collect (timestamp, kpi) pairs; need >= 3 observations.
        pairs: list[tuple[float, float]] = []
        for obs in successful:
            kpi_val = obs.kpi_values.get(kpi_name)
            if kpi_val is not None:
                pairs.append((obs.timestamp, kpi_val))

        if len(pairs) < 3:
            return InstrumentDrift(False, 0.0, 0.0, 0.0, "stable")

        timestamps = [p[0] for p in pairs]
        kpi_vals = [p[1] for p in pairs]

        # Primary linear fit: KPI ~ timestamp
        slope, intercept = _linreg(timestamps, kpi_vals)

        # Compute residuals.
        residuals = [
            kpi - (slope * t + intercept)
            for t, kpi in zip(timestamps, kpi_vals)
        ]

        # Secondary fit on residuals: residual ~ timestamp
        drift_slope, drift_intercept = _linreg(timestamps, residuals)
        r_sq = _r_squared(timestamps, residuals, drift_slope, drift_intercept)

        detected = r_sq > self.drift_r2_threshold

        if drift_slope > 0.001:
            direction = "increasing"
        elif drift_slope < -0.001:
            direction = "decreasing"
        else:
            direction = "stable"

        return InstrumentDrift(
            detected=detected,
            drift_slope=drift_slope,
            drift_intercept=drift_intercept,
            r_squared=r_sq,
            residual_trend_direction=direction,
        )

    # ── credibility weights ───────────────────────────────────

    def _compute_credibility_weights(
        self,
        snapshot: CampaignSnapshot,
        noise_decomp: NoiseDecomposition,
        batch_effect: BatchEffect,
        instrument_drift: InstrumentDrift,
    ) -> dict[int, float]:
        """Compute per-observation credibility weights in [0.1, 1.0]."""
        observations = snapshot.observations
        if not observations or not snapshot.objective_names:
            return {}

        kpi_name = snapshot.objective_names[0]

        # Gather all KPI values for statistics.
        all_kpi = [
            o.kpi_values[kpi_name]
            for o in observations
            if kpi_name in o.kpi_values
        ]
        grand_mean = _mean(all_kpi) if all_kpi else 0.0
        grand_std = _std(all_kpi) if all_kpi else 0.0

        # Timestamps for drift adjustment.
        timestamps = [o.timestamp for o in observations]
        t_min = min(timestamps) if timestamps else 0.0
        t_max = max(timestamps) if timestamps else 0.0
        t_span = t_max - t_min

        # Batch assignment helpers.
        successful_sorted = sorted(
            snapshot.successful_observations, key=lambda o: o.iteration
        )
        obs_to_batch_idx: dict[int, int] = {}
        batch_idx = 0
        count_in_batch = 0
        for obs in successful_sorted:
            obs_to_batch_idx[obs.iteration] = batch_idx
            count_in_batch += 1
            if count_in_batch >= self.batch_window:
                batch_idx += 1
                count_in_batch = 0

        weights: dict[int, float] = {}
        for obs in observations:
            base = 1.0

            # QC factor.
            qc_factor = 1.0 if obs.qc_passed else 0.5

            # Failure factor.
            if obs.is_failure:
                base *= 0.7

            # Outlier factor (distance from grand mean).
            kpi_val = obs.kpi_values.get(kpi_name)
            outlier_factor = 1.0
            if kpi_val is not None and grand_std > 0.0:
                residual = abs(kpi_val - grand_mean)
                outlier_factor = max(0.3, 1.0 - residual / (3.0 * grand_std))

            # Drift factor.
            drift_factor = 1.0
            if instrument_drift.detected and t_span > 0.0:
                drift_factor = max(
                    0.5,
                    1.0
                    - instrument_drift.r_squared
                    * (t_max - obs.timestamp)
                    / t_span,
                )

            # Batch factor.
            batch_factor = 1.0
            if batch_effect.detected and grand_std > 0.0:
                b_idx = obs_to_batch_idx.get(obs.iteration)
                if (
                    b_idx is not None
                    and 0 <= b_idx < len(batch_effect.batch_means)
                ):
                    b_mean = batch_effect.batch_means[b_idx]
                    batch_factor = max(
                        0.7,
                        1.0 - 0.3 * abs(b_mean - grand_mean) / grand_std,
                    )

            weight = base * qc_factor * outlier_factor * drift_factor * batch_factor
            weight = max(0.1, min(1.0, weight))
            weights[obs.iteration] = weight

        return weights

    # ── quality score ─────────────────────────────────────────

    @staticmethod
    def _compute_quality_score(
        noise_decomp: NoiseDecomposition,
        batch_effect: BatchEffect,
        instrument_drift: InstrumentDrift,
    ) -> float:
        """Overall quality score in [0.0, 1.0]."""
        score = 1.0
        if noise_decomp.total_noise > 0.5:
            score -= 0.2
        if noise_decomp.ratio > 0.7:
            score -= 0.1
        if batch_effect.detected:
            score -= 0.2 * batch_effect.effect_size
        if instrument_drift.detected:
            score -= 0.2 * instrument_drift.r_squared
        return max(0.0, min(1.0, score))

    # ── issue collection ──────────────────────────────────────

    @staticmethod
    def _collect_issues(
        noise_decomp: NoiseDecomposition,
        batch_effect: BatchEffect,
        instrument_drift: InstrumentDrift,
    ) -> list[str]:
        """Collect human-readable issue descriptions."""
        issues: list[str] = []
        if noise_decomp.total_noise > 0.5:
            issues.append(f"High total noise (CV={noise_decomp.total_noise:.2f})")
        if noise_decomp.ratio > 0.7:
            issues.append(
                f"Measurement noise dominates ({noise_decomp.ratio:.0%} of total)"
            )
        if batch_effect.detected:
            issues.append(
                f"Batch effect detected (F={batch_effect.f_statistic:.2f}, "
                f"eta\u00b2={batch_effect.effect_size:.2f})"
            )
        if instrument_drift.detected:
            issues.append(
                f"Instrument drift detected (R\u00b2={instrument_drift.r_squared:.2f}, "
                f"direction={instrument_drift.residual_trend_direction})"
            )
        return issues

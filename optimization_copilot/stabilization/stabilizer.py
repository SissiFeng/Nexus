"""Data conditioning and noise handling for optimization campaigns."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import CampaignSnapshot, Observation, StabilizeSpec


@dataclass
class StabilizedData:
    """Result of data stabilization."""
    observations: list[Observation]
    removed_indices: list[int] = field(default_factory=list)
    smoothed_kpis: dict[str, list[float]] = field(default_factory=dict)
    applied_policies: list[str] = field(default_factory=list)


class Stabilizer:
    """Apply data conditioning policies to campaign observations."""

    def stabilize(
        self, snapshot: CampaignSnapshot, spec: StabilizeSpec
    ) -> StabilizedData:
        obs = list(snapshot.observations)
        removed: list[int] = []
        policies: list[str] = []

        # 1. Failure handling
        obs, fail_removed = self._handle_failures(obs, spec.failure_handling)
        removed.extend(fail_removed)
        if fail_removed:
            policies.append(f"failure_handling:{spec.failure_handling}")

        # 2. Outlier rejection
        obs, outlier_removed = self._reject_outliers(
            obs, snapshot.objective_names, spec.outlier_rejection_sigma
        )
        removed.extend(outlier_removed)
        if outlier_removed:
            policies.append(f"outlier_rejection:sigma={spec.outlier_rejection_sigma}")

        # 3. Reweighting (mark in metadata, not actually remove)
        if spec.reweighting_strategy != "none":
            obs = self._apply_reweighting(obs, spec.reweighting_strategy)
            policies.append(f"reweighting:{spec.reweighting_strategy}")

        # 4. Noise smoothing
        smoothed = {}
        if spec.noise_smoothing_window > 1 and snapshot.objective_names:
            for obj_name in snapshot.objective_names:
                values = [
                    o.kpi_values.get(obj_name, 0.0) for o in obs if not o.is_failure
                ]
                smoothed[obj_name] = self._moving_average(
                    values, spec.noise_smoothing_window
                )
            policies.append(f"noise_smoothing:window={spec.noise_smoothing_window}")

        return StabilizedData(
            observations=obs,
            removed_indices=removed,
            smoothed_kpis=smoothed,
            applied_policies=policies,
        )

    @staticmethod
    def _handle_failures(
        obs: list[Observation], policy: str
    ) -> tuple[list[Observation], list[int]]:
        removed = []
        if policy == "exclude":
            result = []
            for i, o in enumerate(obs):
                if o.is_failure:
                    removed.append(i)
                else:
                    result.append(o)
            return result, removed
        elif policy == "penalize":
            # Keep failures but mark them — no removal
            return obs, []
        elif policy == "impute":
            # Delegate to DeterministicImputer when available
            try:
                from optimization_copilot.imputation.imputer import DeterministicImputer
                from optimization_copilot.imputation.models import ImputationConfig
                imputer = DeterministicImputer(ImputationConfig())
                obj_names = list(
                    {k for o in obs for k in o.kpi_values if not o.is_failure}
                )
                imp_result = imputer.impute(obs, obj_names, [])
                return imp_result.observations, []
            except Exception:
                pass
            # Fallback: replace failure KPIs with worst observed value
            valid_kpis: dict[str, list[float]] = {}
            for o in obs:
                if not o.is_failure:
                    for k, v in o.kpi_values.items():
                        valid_kpis.setdefault(k, []).append(v)
            worst: dict[str, float] = {}
            for k, vals in valid_kpis.items():
                worst[k] = min(vals) if vals else 0.0
            result = []
            for o in obs:
                if o.is_failure and worst:
                    new_obs = Observation(
                        iteration=o.iteration,
                        parameters=o.parameters,
                        kpi_values=dict(worst),
                        qc_passed=o.qc_passed,
                        is_failure=True,
                        failure_reason=o.failure_reason,
                        timestamp=o.timestamp,
                        metadata={**o.metadata, "imputed": True},
                    )
                    result.append(new_obs)
                else:
                    result.append(o)
            return result, []
        return obs, []

    @staticmethod
    def _reject_outliers(
        obs: list[Observation], objective_names: list[str], sigma: float
    ) -> tuple[list[Observation], list[int]]:
        if not objective_names or sigma <= 0:
            return obs, []

        removed = []
        for obj_name in objective_names:
            values = [
                o.kpi_values.get(obj_name, 0.0) for o in obs if not o.is_failure
            ]
            if len(values) < 3:
                continue
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 0.0
            if std == 0:
                continue
            for i, o in enumerate(obs):
                if o.is_failure:
                    continue
                val = o.kpi_values.get(obj_name, 0.0)
                if abs(val - mean) > sigma * std:
                    removed.append(i)

        removed_set = set(removed)
        result = [o for i, o in enumerate(obs) if i not in removed_set]
        return result, sorted(removed_set)

    @staticmethod
    def _apply_reweighting(
        obs: list[Observation], strategy: str
    ) -> list[Observation]:
        n = len(obs)
        if n == 0:
            return obs
        result = []
        for i, o in enumerate(obs):
            weight = 1.0
            if strategy == "recency":
                weight = (i + 1) / n
            elif strategy == "quality":
                weight = 1.0 if o.qc_passed and not o.is_failure else 0.5
            new_meta = {**o.metadata, "weight": weight}
            result.append(Observation(
                iteration=o.iteration,
                parameters=o.parameters,
                kpi_values=o.kpi_values,
                qc_passed=o.qc_passed,
                is_failure=o.is_failure,
                failure_reason=o.failure_reason,
                timestamp=o.timestamp,
                metadata=new_meta,
            ))
        return result

    @staticmethod
    def _moving_average(values: list[float], window: int) -> list[float]:
        if not values:
            return []
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_vals = values[start:i + 1]
            result.append(sum(window_vals) / len(window_vals))
        return result

    # -- Repeat measurement advisor -----------------------------------------

    @staticmethod
    def recommend_repeats(
        noise_estimate: float,
        signal_to_noise_ratio: float,
        n_observations: int,
        *,
        noise_threshold: float = 0.3,
        snr_threshold: float = 3.0,
        max_repeats: int = 5,
    ) -> RepeatRecommendation:
        """Recommend repeat measurements when noise is high.

        Parameters
        ----------
        noise_estimate :
            Coefficient of variation from diagnostics.
        signal_to_noise_ratio :
            SNR from diagnostics.
        n_observations :
            Current number of observations.
        noise_threshold :
            Noise level above which repeats are recommended.
        snr_threshold :
            SNR below which repeats are recommended.
        max_repeats :
            Maximum recommended repeats per setting.

        Returns
        -------
        RepeatRecommendation
        """
        reasons: list[str] = []

        if noise_estimate > noise_threshold:
            reasons.append(
                f"noise_estimate={noise_estimate:.3f} > {noise_threshold}"
            )
        if 0 < signal_to_noise_ratio < snr_threshold:
            reasons.append(
                f"SNR={signal_to_noise_ratio:.2f} < {snr_threshold}"
            )

        if not reasons:
            return RepeatRecommendation(
                should_repeat=False,
                n_repeats=1,
                reason="Noise levels are acceptable; no repeats needed.",
            )

        # Scale repeats with noise severity: higher noise → more repeats
        if noise_estimate > 0:
            severity = min(noise_estimate / noise_threshold, 3.0)
            n_repeats = min(max_repeats, max(2, int(1 + severity)))
        else:
            n_repeats = 2

        # Fewer repeats needed if we already have many observations
        if n_observations > 50:
            n_repeats = max(2, n_repeats - 1)

        return RepeatRecommendation(
            should_repeat=True,
            n_repeats=n_repeats,
            reason="; ".join(reasons),
        )

    # -- Heteroscedastic awareness ------------------------------------------

    @staticmethod
    def detect_heteroscedasticity(
        snapshot: CampaignSnapshot,
        n_bins: int = 5,
    ) -> HeteroscedasticityReport:
        """Detect region-specific noise variation across parameter space.

        Bins successful observations by each parameter dimension, computes
        per-bin noise (coefficient of variation), and reports whether noise
        varies significantly across regions.

        Parameters
        ----------
        snapshot :
            Campaign with observations and parameter specs.
        n_bins :
            Number of bins per parameter dimension.

        Returns
        -------
        HeteroscedasticityReport
        """
        successful = snapshot.successful_observations
        obj_names = snapshot.objective_names

        if not successful or not obj_names:
            return HeteroscedasticityReport(
                is_heteroscedastic=False,
                noise_ratio=1.0,
                region_noise={},
                noisiest_region=None,
                quietest_region=None,
            )

        kpi_name = obj_names[0]
        specs = snapshot.parameter_specs

        if not specs:
            return HeteroscedasticityReport(
                is_heteroscedastic=False,
                noise_ratio=1.0,
                region_noise={},
                noisiest_region=None,
                quietest_region=None,
            )

        # Bin by each parameter dimension independently
        region_noise: dict[str, list[RegionNoise]] = {}

        for spec in specs:
            lo = spec.lower if spec.lower is not None else 0.0
            hi = spec.upper if spec.upper is not None else 1.0
            if hi == lo:
                continue

            bins: dict[int, list[float]] = {}
            for obs in successful:
                val = obs.parameters.get(spec.name)
                if not isinstance(val, (int, float)):
                    continue
                kpi = obs.kpi_values.get(kpi_name)
                if kpi is None:
                    continue
                frac = (float(val) - lo) / (hi - lo)
                frac = max(0.0, min(1.0, frac))
                b = min(int(frac * n_bins), n_bins - 1)
                bins.setdefault(b, []).append(kpi)

            param_regions: list[RegionNoise] = []
            for b, kpis in sorted(bins.items()):
                if len(kpis) < 2:
                    continue
                m = sum(kpis) / len(kpis)
                var = sum((v - m) ** 2 for v in kpis) / len(kpis)
                std = math.sqrt(var)
                cv = std / abs(m) if m != 0 else 0.0
                bin_lo = lo + (hi - lo) * b / n_bins
                bin_hi = lo + (hi - lo) * (b + 1) / n_bins
                param_regions.append(RegionNoise(
                    parameter=spec.name,
                    bin_range=(bin_lo, bin_hi),
                    n_observations=len(kpis),
                    noise_cv=cv,
                ))
            if param_regions:
                region_noise[spec.name] = param_regions

        # Compute noise ratio: max_cv / min_cv across all regions
        all_cvs = [
            rn.noise_cv
            for regions in region_noise.values()
            for rn in regions
            if rn.noise_cv > 0
        ]
        if len(all_cvs) < 2:
            return HeteroscedasticityReport(
                is_heteroscedastic=False,
                noise_ratio=1.0,
                region_noise=region_noise,
                noisiest_region=None,
                quietest_region=None,
            )

        max_cv = max(all_cvs)
        min_cv = min(all_cvs)
        noise_ratio = max_cv / min_cv if min_cv > 0 else max_cv

        # Find noisiest and quietest
        all_regions = [
            rn for regions in region_noise.values() for rn in regions
            if rn.noise_cv > 0
        ]
        noisiest = max(all_regions, key=lambda r: r.noise_cv)
        quietest = min(all_regions, key=lambda r: r.noise_cv)

        return HeteroscedasticityReport(
            is_heteroscedastic=noise_ratio > 3.0,
            noise_ratio=noise_ratio,
            region_noise=region_noise,
            noisiest_region=noisiest,
            quietest_region=quietest,
        )


# ---------------------------------------------------------------------------
# Data structures for repeat advice and heteroscedasticity
# ---------------------------------------------------------------------------

@dataclass
class RepeatRecommendation:
    """Recommendation for repeat measurements at the same parameter setting."""
    should_repeat: bool
    n_repeats: int
    reason: str


@dataclass
class RegionNoise:
    """Noise level in a specific parameter region."""
    parameter: str
    bin_range: tuple[float, float]
    n_observations: int
    noise_cv: float


@dataclass
class HeteroscedasticityReport:
    """Report on region-specific noise variation."""
    is_heteroscedastic: bool
    noise_ratio: float
    region_noise: dict[str, list[RegionNoise]]
    noisiest_region: RegionNoise | None
    quietest_region: RegionNoise | None

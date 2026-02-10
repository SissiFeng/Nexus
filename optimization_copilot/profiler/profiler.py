"""Problem Profiler: classifies a CampaignSnapshot into a ProblemFingerprint.

Uses heuristic rules to infer each fingerprint dimension from the
observable properties of the campaign data.
"""

from __future__ import annotations

import statistics
from typing import Sequence

from optimization_copilot.core.models import (
    CampaignSnapshot,
    CostProfile,
    DataScale,
    Dynamics,
    FailureInformativeness,
    FeasibleRegion,
    NoiseRegime,
    ObjectiveForm,
    Observation,
    ParameterSpec,
    ProblemFingerprint,
    VariableType,
)


class ProblemProfiler:
    """Heuristic classifier that produces a ProblemFingerprint from campaign data."""

    def profile(self, snapshot: CampaignSnapshot) -> ProblemFingerprint:
        """Analyse *snapshot* and return a fully-populated ProblemFingerprint."""
        return ProblemFingerprint(
            variable_types=self._classify_variable_types(snapshot.parameter_specs),
            objective_form=self._classify_objective_form(snapshot),
            noise_regime=self._classify_noise_regime(snapshot),
            cost_profile=self._classify_cost_profile(snapshot.observations),
            failure_informativeness=self._classify_failure_informativeness(snapshot),
            data_scale=self._classify_data_scale(snapshot.n_observations),
            dynamics=self._classify_dynamics(snapshot),
            feasible_region=self._classify_feasible_region(snapshot.failure_rate),
        )

    # ── Individual dimension classifiers ──────────────────

    @staticmethod
    def _classify_variable_types(specs: list[ParameterSpec]) -> VariableType:
        """If all parameters share a single type, return that type; otherwise MIXED."""
        if not specs:
            return VariableType.CONTINUOUS  # default when no parameters

        types = {s.type for s in specs}
        if len(types) == 1:
            return types.pop()
        return VariableType.MIXED

    @staticmethod
    def _classify_objective_form(snapshot: CampaignSnapshot) -> ObjectiveForm:
        """Multi-objective > constrained > single."""
        if len(snapshot.objective_names) > 1:
            return ObjectiveForm.MULTI_OBJECTIVE
        if snapshot.constraints:
            return ObjectiveForm.CONSTRAINED
        return ObjectiveForm.SINGLE

    @staticmethod
    def _classify_noise_regime(snapshot: CampaignSnapshot) -> NoiseRegime:
        """Coefficient of variation of the first KPI among successful observations.

        CV < 0.1 -> LOW, CV < 0.5 -> MEDIUM, else HIGH.
        Falls back to LOW when there are fewer than 2 successful observations.
        """
        successful = snapshot.successful_observations
        if len(successful) < 2:
            return NoiseRegime.LOW

        # Use the first objective name for evaluation.
        kpi_name = snapshot.objective_names[0]
        values = [
            obs.kpi_values[kpi_name]
            for obs in successful
            if kpi_name in obs.kpi_values
        ]

        if len(values) < 2:
            return NoiseRegime.LOW

        mean = statistics.mean(values)
        if mean == 0.0:
            # Cannot compute CV when mean is zero; fall back to stdev-based check.
            sd = statistics.stdev(values)
            if sd < 1e-9:
                return NoiseRegime.LOW
            return NoiseRegime.HIGH

        cv = statistics.stdev(values) / abs(mean)
        if cv < 0.1:
            return NoiseRegime.LOW
        if cv < 0.5:
            return NoiseRegime.MEDIUM
        return NoiseRegime.HIGH

    @staticmethod
    def _classify_cost_profile(observations: list[Observation]) -> CostProfile:
        """Uniform spacing in timestamps -> UNIFORM, else HETEROGENEOUS.

        If fewer than 3 observations, default to UNIFORM (not enough data to judge).
        Spacing is considered uniform when the coefficient of variation of the
        inter-observation time gaps is below 0.3.
        """
        if len(observations) < 3:
            return CostProfile.UNIFORM

        timestamps = [o.timestamp for o in observations]
        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

        # All gaps zero (timestamps not populated) → treat as uniform.
        if all(abs(g) < 1e-12 for g in gaps):
            return CostProfile.UNIFORM

        mean_gap = statistics.mean(gaps)
        if mean_gap == 0.0:
            return CostProfile.UNIFORM

        cv_gap = statistics.stdev(gaps) / abs(mean_gap)
        if cv_gap < 0.3:
            return CostProfile.UNIFORM
        return CostProfile.HETEROGENEOUS

    @staticmethod
    def _classify_failure_informativeness(
        snapshot: CampaignSnapshot,
    ) -> FailureInformativeness:
        """Diverse parameter values among failures -> STRONG, else WEAK.

        Diversity is measured as the fraction of parameter dimensions whose
        unique-value count among failures is > 1.  If that fraction >= 0.5
        the failure region is considered informative (STRONG).
        """
        failed = [o for o in snapshot.observations if o.is_failure]
        if len(failed) < 2:
            return FailureInformativeness.WEAK

        param_names = snapshot.parameter_names
        if not param_names:
            return FailureInformativeness.WEAK

        diverse_count = 0
        for pname in param_names:
            unique_vals = {obs.parameters.get(pname) for obs in failed}
            if len(unique_vals) > 1:
                diverse_count += 1

        diversity_ratio = diverse_count / len(param_names)
        if diversity_ratio >= 0.5:
            return FailureInformativeness.STRONG
        return FailureInformativeness.WEAK

    @staticmethod
    def _classify_data_scale(n_observations: int) -> DataScale:
        """< 10 -> TINY, < 50 -> SMALL, else MODERATE."""
        if n_observations < 10:
            return DataScale.TINY
        if n_observations < 50:
            return DataScale.SMALL
        return DataScale.MODERATE

    @staticmethod
    def _classify_dynamics(snapshot: CampaignSnapshot) -> Dynamics:
        """Temporal autocorrelation of the first KPI -> TIME_SERIES if significant.

        Uses the lag-1 Pearson autocorrelation.  If |r| >= 0.5 the series
        is classified as TIME_SERIES; otherwise STATIC.

        Falls back to STATIC when there are fewer than 4 successful observations.
        """
        successful = snapshot.successful_observations
        if len(successful) < 4:
            return Dynamics.STATIC

        kpi_name = snapshot.objective_names[0]
        values = [
            obs.kpi_values[kpi_name]
            for obs in successful
            if kpi_name in obs.kpi_values
        ]

        if len(values) < 4:
            return Dynamics.STATIC

        autocorr = _lag1_autocorrelation(values)
        if abs(autocorr) >= 0.5:
            return Dynamics.TIME_SERIES
        return Dynamics.STATIC

    @staticmethod
    def _classify_feasible_region(failure_rate: float) -> FeasibleRegion:
        """failure_rate < 0.1 -> WIDE, < 0.3 -> NARROW, else FRAGMENTED."""
        if failure_rate < 0.1:
            return FeasibleRegion.WIDE
        if failure_rate < 0.3:
            return FeasibleRegion.NARROW
        return FeasibleRegion.FRAGMENTED


# ── Helper utilities ──────────────────────────────────────


def _lag1_autocorrelation(values: Sequence[float]) -> float:
    """Compute the lag-1 Pearson autocorrelation for *values*.

    Returns 0.0 if the series has zero variance.
    """
    n = len(values)
    if n < 2:
        return 0.0

    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values)
    if var == 0.0:
        return 0.0

    cov = sum((values[i] - mean) * (values[i + 1] - mean) for i in range(n - 1))
    return cov / var

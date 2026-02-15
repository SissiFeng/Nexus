"""Search-space surgeon: diagnoses and applies dimension-reduction actions."""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.dsl.spec import OptimizationSpec, ParameterDef, ParamType
from optimization_copilot.screening.screener import ScreeningResult
from optimization_copilot.surgery.models import (
    ActionType,
    DerivedType,
    SurgeryAction,
    SurgeryReport,
)


# ── Helper functions ───────────────────────────────────


def _extract_param_values(observations: list[Observation], param_name: str) -> list[float]:
    """Extract numeric parameter values from observations, skipping missing entries."""
    values: list[float] = []
    for obs in observations:
        val = obs.parameters.get(param_name)
        if val is not None:
            try:
                values.append(float(val))
            except (TypeError, ValueError):
                continue
    return values


def _mean(values: list[float]) -> float:
    """Compute arithmetic mean of a list of floats."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient between two equal-length lists.

    Returns 0.0 if either series has zero variance or the lists are too short.
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    x_std = math.sqrt(sum((x - x_mean) ** 2 for x in xs) / n)
    y_std = math.sqrt(sum((y - y_mean) ** 2 for y in ys) / n)

    if x_std < 1e-12 or y_std < 1e-12:
        return 0.0

    cov = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n)) / n
    return cov / (x_std * y_std)


def _abs_correlation(
    observations: list[Observation], param_name: str, obj_name: str
) -> float:
    """Compute absolute Pearson correlation between a parameter and an objective."""
    param_vals: list[float] = []
    obj_vals: list[float] = []
    for obs in observations:
        p_val = obs.parameters.get(param_name)
        o_val = obs.kpi_values.get(obj_name)
        if p_val is not None and o_val is not None:
            try:
                param_vals.append(float(p_val))
                obj_vals.append(float(o_val))
            except (TypeError, ValueError):
                continue
    return abs(_pearson_correlation(param_vals, obj_vals))


def _best_parameter_value(
    observations: list[Observation], spec: ParameterSpec
) -> Any:
    """Return the parameter value from the observation with the best (minimum) first KPI.

    If no valid observation is found, returns None.
    """
    best_obs: Observation | None = None
    best_kpi: float | None = None

    for obs in observations:
        if not obs.kpi_values:
            continue
        # Use the first KPI as the objective (minimize by default)
        first_kpi_name = next(iter(obs.kpi_values))
        kpi_val = obs.kpi_values[first_kpi_name]
        if best_kpi is None or kpi_val < best_kpi:
            best_kpi = kpi_val
            best_obs = obs

    if best_obs is None:
        return None
    return best_obs.parameters.get(spec.name)


# ── Surgeon class ──────────────────────────────────────


class SearchSpaceSurgeon:
    """Diagnoses search-space inefficiencies and prescribes dimension-reduction actions.

    The surgeon analyzes successful observations to detect:
    - Parameters whose effective range is much narrower than specified
    - Parameters with negligible influence on objectives (freeze candidates)
    - Conditional freezing opportunities (parameter irrelevant in part of space)
    - Redundant (highly correlated) parameter pairs
    - Opportunities for derived parameter transformations (log, ratio)
    """

    def __init__(
        self,
        tightening_quantile: float = 0.05,
        correlation_threshold: float = 0.9,
        freeze_importance_threshold: float = 0.05,
        min_observations: int = 10,
    ) -> None:
        self.tightening_quantile = tightening_quantile
        self.correlation_threshold = correlation_threshold
        self.freeze_importance_threshold = freeze_importance_threshold
        self.min_observations = min_observations

    # ── Public API ─────────────────────────────────────

    def diagnose(
        self,
        snapshot: CampaignSnapshot,
        screening_result: ScreeningResult | None = None,
        seed: int = 42,
    ) -> SurgeryReport:
        """Analyze the campaign and produce a surgery report.

        Parameters
        ----------
        snapshot : CampaignSnapshot
            Current state of the optimization campaign.
        screening_result : ScreeningResult | None
            Optional variable-screening output; enables freeze-candidate detection.
        seed : int
            Random seed for reproducibility (reserved for future stochastic methods).

        Returns
        -------
        SurgeryReport
            Prescribed actions with supporting evidence.
        """
        successful = snapshot.successful_observations
        specs = snapshot.parameter_specs
        original_dim = len(specs)

        # Insufficient data guard
        if len(successful) < self.min_observations:
            return SurgeryReport(
                actions=[],
                original_dim=original_dim,
                effective_dim=original_dim,
                space_reduction_ratio=0.0,
                reason_codes=["insufficient_data"],
                metadata={"n_successful": len(successful), "min_required": self.min_observations},
            )

        actions: list[SurgeryAction] = []
        reason_codes: list[str] = []

        # Detection pass 1: range tightening
        actions.extend(self._detect_range_tightening(successful, specs, reason_codes))

        # Detection pass 2: freeze candidates (requires screening result)
        if screening_result is not None:
            actions.extend(self._detect_freeze_candidates(successful, specs, screening_result, reason_codes))

        # Detection pass 3: conditional freezing
        actions.extend(self._detect_conditional_freezing(successful, specs, snapshot, reason_codes))

        # Detection pass 4: redundancy (correlated pairs)
        actions.extend(self._detect_redundancy(successful, specs, reason_codes))

        # Detection pass 5: derived parameter suggestions
        actions.extend(self._suggest_derived_parameters(successful, specs, reason_codes))

        # Compute effective dimension
        frozen_params: set[str] = set()
        merged_params: set[str] = set()
        for action in actions:
            if action.action_type == ActionType.FREEZE_PARAMETER:
                for p in action.target_params:
                    frozen_params.add(p)
            elif action.action_type == ActionType.MERGE_PARAMETERS:
                # Secondary params are frozen; merge_into is kept
                for p in action.target_params:
                    if p != action.merge_into:
                        merged_params.add(p)

        removed_count = len(frozen_params | merged_params)
        effective_dim = max(0, original_dim - removed_count)
        space_reduction_ratio = removed_count / original_dim if original_dim > 0 else 0.0

        return SurgeryReport(
            actions=actions,
            original_dim=original_dim,
            effective_dim=effective_dim,
            space_reduction_ratio=space_reduction_ratio,
            reason_codes=reason_codes,
            metadata={
                "n_successful": len(successful),
                "n_frozen": len(frozen_params),
                "n_merged": len(merged_params),
            },
        )

    def apply(
        self, spec: OptimizationSpec, report: SurgeryReport
    ) -> OptimizationSpec:
        """Apply a surgery report to an OptimizationSpec, returning a new spec.

        The original spec is not mutated. Parameters are deep-copied via
        to_dict / from_dict before modification.

        Parameters
        ----------
        spec : OptimizationSpec
            The original optimization specification.
        report : SurgeryReport
            Surgery report with actions to apply.

        Returns
        -------
        OptimizationSpec
            A new spec with surgery actions applied.
        """
        # Deep copy via serialization round-trip
        new_spec = OptimizationSpec.from_dict(spec.to_dict())

        # Build a lookup for quick parameter access by name
        param_map: dict[str, ParameterDef] = {p.name: p for p in new_spec.parameters}

        for action in report.actions:
            if action.action_type == ActionType.TIGHTEN_RANGE:
                for pname in action.target_params:
                    if pname in param_map:
                        p = param_map[pname]
                        if action.new_lower is not None:
                            p.lower = action.new_lower
                        if action.new_upper is not None:
                            p.upper = action.new_upper

            elif action.action_type == ActionType.FREEZE_PARAMETER:
                for pname in action.target_params:
                    if pname in param_map:
                        p = param_map[pname]
                        p.frozen = True
                        p.frozen_value = action.freeze_value

            elif action.action_type == ActionType.MERGE_PARAMETERS:
                # Freeze secondary parameters (all target_params except merge_into)
                for pname in action.target_params:
                    if pname != action.merge_into and pname in param_map:
                        p = param_map[pname]
                        p.frozen = True
                        p.frozen_value = action.freeze_value

        return new_spec

    # ── Detection methods ──────────────────────────────

    def _detect_range_tightening(
        self,
        successful: list[Observation],
        specs: list[ParameterSpec],
        reason_codes: list[str],
    ) -> list[SurgeryAction]:
        """Detect parameters whose successful observations concentrate in a sub-range."""
        actions: list[SurgeryAction] = []

        for spec in specs:
            if spec.type == VariableType.CATEGORICAL:
                continue
            if spec.lower is None or spec.upper is None:
                continue

            values = _extract_param_values(successful, spec.name)
            n_values = len(values)
            if n_values < 5:
                continue

            values.sort()
            original_range = spec.upper - spec.lower
            if original_range <= 0:
                continue

            # Compute quantile indices
            low_idx = max(0, int(math.floor(self.tightening_quantile * (n_values - 1))))
            high_idx = min(n_values - 1, int(math.ceil((1.0 - self.tightening_quantile) * (n_values - 1))))

            q_low = values[low_idx]
            q_high = values[high_idx]
            new_range = q_high - q_low

            # Only tighten if reduction > 20%
            if new_range >= original_range * 0.8:
                continue

            # Add 10% margin
            margin = new_range * 0.1
            proposed_lower = q_low - margin
            proposed_upper = q_high + margin

            # Clamp to original bounds
            proposed_lower = max(spec.lower, proposed_lower)
            proposed_upper = min(spec.upper, proposed_upper)

            confidence = min(1.0, n_values / 30.0)

            actions.append(SurgeryAction(
                action_type=ActionType.TIGHTEN_RANGE,
                target_params=[spec.name],
                new_lower=proposed_lower,
                new_upper=proposed_upper,
                reason=f"Successful observations concentrate in [{proposed_lower:.4g}, {proposed_upper:.4g}] "
                       f"(original [{spec.lower:.4g}, {spec.upper:.4g}])",
                confidence=confidence,
                evidence={
                    "n_values": n_values,
                    "original_lower": spec.lower,
                    "original_upper": spec.upper,
                    "quantile_lower": q_low,
                    "quantile_upper": q_high,
                    "range_reduction_pct": round((1.0 - (proposed_upper - proposed_lower) / original_range) * 100, 1),
                },
            ))

        if actions:
            reason_codes.append("range_tightening")

        return actions

    def _detect_freeze_candidates(
        self,
        successful: list[Observation],
        specs: list[ParameterSpec],
        screening: ScreeningResult,
        reason_codes: list[str],
    ) -> list[SurgeryAction]:
        """Detect parameters with negligible importance that can be frozen."""
        actions: list[SurgeryAction] = []
        n_obs = len(successful)

        for spec in specs:
            score = screening.importance_scores.get(spec.name, 0.0)
            if score >= self.freeze_importance_threshold:
                continue

            # Find the best observed value for this parameter
            best_val = _best_parameter_value(successful, spec)
            if best_val is None:
                continue

            confidence = min(1.0, n_obs / 20.0)

            actions.append(SurgeryAction(
                action_type=ActionType.FREEZE_PARAMETER,
                target_params=[spec.name],
                freeze_value=best_val,
                reason=f"Parameter '{spec.name}' has importance score {score:.4f} "
                       f"(< threshold {self.freeze_importance_threshold}); "
                       f"freezing at best-observed value {best_val}",
                confidence=confidence,
                evidence={
                    "importance_score": score,
                    "threshold": self.freeze_importance_threshold,
                    "freeze_value": best_val,
                    "n_observations": n_obs,
                },
            ))

        if actions:
            reason_codes.append("freeze_unimportant")

        return actions

    def _detect_conditional_freezing(
        self,
        successful: list[Observation],
        specs: list[ParameterSpec],
        snapshot: CampaignSnapshot,
        reason_codes: list[str],
    ) -> list[SurgeryAction]:
        """Detect parameters that become irrelevant in a sub-region of another parameter."""
        actions: list[SurgeryAction] = []
        n_obs = len(successful)

        if n_obs < 15:
            return actions

        # Need at least one objective
        if not snapshot.objective_names:
            return actions
        obj_name = snapshot.objective_names[0]

        # Get continuous specs only
        continuous_specs = [s for s in specs if s.type != VariableType.CATEGORICAL
                           and s.lower is not None and s.upper is not None]

        # Track which y params already have a conditional freeze
        conditionally_frozen: set[str] = set()

        for x_spec in continuous_specs:
            x_vals = _extract_param_values(successful, x_spec.name)
            if len(x_vals) != n_obs:
                continue

            # Compute median of x
            sorted_x = sorted(x_vals)
            mid = len(sorted_x) // 2
            if len(sorted_x) % 2 == 0:
                x_median = (sorted_x[mid - 1] + sorted_x[mid]) / 2.0
            else:
                x_median = sorted_x[mid]

            # Split observations by median of x
            below_obs: list[Observation] = []
            above_obs: list[Observation] = []
            for i, obs in enumerate(successful):
                x_val = obs.parameters.get(x_spec.name)
                if x_val is not None and float(x_val) <= x_median:
                    below_obs.append(obs)
                else:
                    above_obs.append(obs)

            if len(below_obs) < 3 or len(above_obs) < 3:
                continue

            for y_spec in continuous_specs:
                if y_spec.name == x_spec.name:
                    continue
                if y_spec.name in conditionally_frozen:
                    continue

                # Compute correlations between y and KPI in each subset and full set
                full_corr = _abs_correlation(successful, y_spec.name, obj_name)
                below_corr = _abs_correlation(below_obs, y_spec.name, obj_name)
                above_corr = _abs_correlation(above_obs, y_spec.name, obj_name)

                confidence = min(0.8, n_obs / 40.0)

                if full_corr > 0.1 and below_corr < full_corr - 0.4:
                    # y is unimportant when x <= median
                    best_val = _best_parameter_value(below_obs, y_spec)
                    actions.append(SurgeryAction(
                        action_type=ActionType.CONDITIONAL_FREEZE,
                        target_params=[y_spec.name],
                        freeze_value=best_val,
                        condition_param=x_spec.name,
                        condition_threshold=x_median,
                        condition_direction="below",
                        reason=f"Parameter '{y_spec.name}' becomes irrelevant when "
                               f"'{x_spec.name}' <= {x_median:.4g} "
                               f"(|corr| drops from {full_corr:.3f} to {below_corr:.3f})",
                        confidence=confidence,
                        evidence={
                            "full_correlation": round(full_corr, 4),
                            "below_correlation": round(below_corr, 4),
                            "above_correlation": round(above_corr, 4),
                            "x_median": x_median,
                            "n_below": len(below_obs),
                            "n_above": len(above_obs),
                        },
                    ))
                    conditionally_frozen.add(y_spec.name)
                    break  # Break after first conditional freeze per y_spec

                elif full_corr > 0.1 and above_corr < full_corr - 0.4:
                    # y is unimportant when x > median
                    best_val = _best_parameter_value(above_obs, y_spec)
                    actions.append(SurgeryAction(
                        action_type=ActionType.CONDITIONAL_FREEZE,
                        target_params=[y_spec.name],
                        freeze_value=best_val,
                        condition_param=x_spec.name,
                        condition_threshold=x_median,
                        condition_direction="above",
                        reason=f"Parameter '{y_spec.name}' becomes irrelevant when "
                               f"'{x_spec.name}' > {x_median:.4g} "
                               f"(|corr| drops from {full_corr:.3f} to {above_corr:.3f})",
                        confidence=confidence,
                        evidence={
                            "full_correlation": round(full_corr, 4),
                            "below_correlation": round(below_corr, 4),
                            "above_correlation": round(above_corr, 4),
                            "x_median": x_median,
                            "n_below": len(below_obs),
                            "n_above": len(above_obs),
                        },
                    ))
                    conditionally_frozen.add(y_spec.name)
                    break  # Break after first conditional freeze per y_spec

        if actions:
            reason_codes.append("conditional_freezing")

        return actions

    def _detect_redundancy(
        self,
        successful: list[Observation],
        specs: list[ParameterSpec],
        reason_codes: list[str],
    ) -> list[SurgeryAction]:
        """Detect highly correlated parameter pairs and suggest merging."""
        actions: list[SurgeryAction] = []

        continuous_specs = [s for s in specs if s.type != VariableType.CATEGORICAL
                           and s.lower is not None and s.upper is not None]

        already_merged: set[str] = set()

        for i, s1 in enumerate(continuous_specs):
            if s1.name in already_merged:
                continue
            vals1 = _extract_param_values(successful, s1.name)

            for s2 in continuous_specs[i + 1:]:
                if s2.name in already_merged:
                    continue
                vals2 = _extract_param_values(successful, s2.name)

                # Need paired values
                n_paired = min(len(vals1), len(vals2))
                if n_paired < 3:
                    continue

                corr = abs(_pearson_correlation(vals1[:n_paired], vals2[:n_paired]))
                if corr >= self.correlation_threshold:
                    # Freeze secondary param (s2), keep primary (s1)
                    freeze_val = _mean(vals2)
                    already_merged.add(s2.name)

                    actions.append(SurgeryAction(
                        action_type=ActionType.MERGE_PARAMETERS,
                        target_params=[s1.name, s2.name],
                        merge_into=s1.name,
                        freeze_value=freeze_val,
                        reason=f"Parameters '{s1.name}' and '{s2.name}' are highly correlated "
                               f"(|r| = {corr:.3f} >= {self.correlation_threshold}); "
                               f"freezing '{s2.name}' at mean value {freeze_val:.4g}",
                        confidence=min(1.0, corr),
                        evidence={
                            "correlation": round(corr, 4),
                            "threshold": self.correlation_threshold,
                            "primary_param": s1.name,
                            "secondary_param": s2.name,
                            "freeze_value": freeze_val,
                            "n_paired": n_paired,
                        },
                    ))

        if actions:
            reason_codes.append("redundancy_merge")

        return actions

    def _suggest_derived_parameters(
        self,
        successful: list[Observation],
        specs: list[ParameterSpec],
        reason_codes: list[str],
    ) -> list[SurgeryAction]:
        """Suggest derived parameter transformations (log, ratio)."""
        actions: list[SurgeryAction] = []
        n_obs = len(successful)

        continuous_specs = [s for s in specs if s.type != VariableType.CATEGORICAL
                           and s.lower is not None and s.upper is not None]

        # LOG suggestion: if lower > 0 and upper/lower >= 100
        for spec in continuous_specs:
            if spec.lower is not None and spec.lower > 0 and spec.upper is not None:
                ratio = spec.upper / spec.lower
                if ratio >= 100:
                    actions.append(SurgeryAction(
                        action_type=ActionType.DERIVE_PARAMETER,
                        target_params=[spec.name],
                        derived_type=DerivedType.LOG,
                        derived_name=f"log_{spec.name}",
                        derived_source_params=[spec.name],
                        reason=f"Parameter '{spec.name}' spans {ratio:.0f}x range "
                               f"([{spec.lower:.4g}, {spec.upper:.4g}]); "
                               f"log transform recommended",
                        confidence=min(1.0, n_obs / 20.0),
                        evidence={
                            "lower": spec.lower,
                            "upper": spec.upper,
                            "range_ratio": round(ratio, 1),
                        },
                    ))

        # RATIO suggestion: if 0.5 <= |corr| < correlation_threshold between two params
        if n_obs >= 10:
            for i, s1 in enumerate(continuous_specs):
                vals1 = _extract_param_values(successful, s1.name)
                for s2 in continuous_specs[i + 1:]:
                    vals2 = _extract_param_values(successful, s2.name)

                    n_paired = min(len(vals1), len(vals2))
                    if n_paired < 3:
                        continue

                    corr = abs(_pearson_correlation(vals1[:n_paired], vals2[:n_paired]))
                    if 0.5 <= corr < self.correlation_threshold:
                        actions.append(SurgeryAction(
                            action_type=ActionType.DERIVE_PARAMETER,
                            target_params=[s1.name, s2.name],
                            derived_type=DerivedType.RATIO,
                            derived_name=f"{s1.name}_over_{s2.name}",
                            derived_source_params=[s1.name, s2.name],
                            reason=f"Parameters '{s1.name}' and '{s2.name}' show moderate "
                                   f"correlation (|r| = {corr:.3f}); ratio transform may "
                                   f"capture their joint effect",
                            confidence=min(0.7, corr),
                            evidence={
                                "correlation": round(corr, 4),
                                "n_paired": n_paired,
                            },
                        ))

        if actions:
            reason_codes.append("derived_parameters")

        return actions

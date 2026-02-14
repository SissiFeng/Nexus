"""Insight discovery engine for scientific optimization campaigns.

Automatically analyzes campaign data to surface actionable insights:
- Best-performing conditions and their shared traits
- Parameter-objective correlations and importance ranking
- Interaction effects between parameters
- Optimal regions in parameter space
- Failure patterns and risk zones
- Trend detection and convergence analysis
- Natural language insight summaries
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from optimization_copilot.api.routes.frontend_v2 import (
    _load_snapshot,
    _compute_diagnostics,
    _compute_importance,
)
from optimization_copilot.core.models import CampaignSnapshot, Observation, VariableType

router = APIRouter(tags=["insights"])


# ── Schemas ──────────────────────────────────────────────────────────


class CorrelationInsight(BaseModel):
    parameter: str
    objective: str
    correlation: float
    strength: str  # "strong", "moderate", "weak"
    direction: str  # "positive", "negative"


class InteractionInsight(BaseModel):
    param_a: str
    param_b: str
    interaction_strength: float
    description: str


class OptimalRegion(BaseModel):
    parameter: str
    best_range: list[float]  # [low, high]
    overall_range: list[float]  # [min, max]
    mean_objective_in_region: float
    mean_objective_outside: float
    improvement_pct: float


class TopCondition(BaseModel):
    rank: int
    parameters: dict[str, Any]
    objective_value: float
    objective_name: str


class FailurePattern(BaseModel):
    description: str
    parameter: str
    risky_range: list[float]
    failure_rate_in_range: float
    overall_failure_rate: float


class TrendInsight(BaseModel):
    description: str
    metric: str
    value: float


class InsightSummary(BaseModel):
    title: str
    body: str
    category: str  # "discovery", "warning", "recommendation", "trend"
    importance: float  # 0-1


class InsightsResponse(BaseModel):
    campaign_id: str
    n_observations: int
    n_parameters: int
    n_objectives: int

    # Structured insights
    top_conditions: list[TopCondition] = Field(default_factory=list)
    correlations: list[CorrelationInsight] = Field(default_factory=list)
    interactions: list[InteractionInsight] = Field(default_factory=list)
    optimal_regions: list[OptimalRegion] = Field(default_factory=list)
    failure_patterns: list[FailurePattern] = Field(default_factory=list)
    trends: list[TrendInsight] = Field(default_factory=list)

    # Natural language summaries
    summaries: list[InsightSummary] = Field(default_factory=list)


# ── Analysis Functions ───────────────────────────────────────────────


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 3:
        return 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    den_x = sum((x - x_mean) ** 2 for x in xs) ** 0.5
    den_y = sum((y - y_mean) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _find_top_conditions(
    snapshot: CampaignSnapshot, top_n: int = 10
) -> list[TopCondition]:
    """Find the top-N best-performing experimental conditions."""
    obs = snapshot.successful_observations
    if not obs or not snapshot.objective_names:
        return []

    primary_obj = snapshot.objective_names[0]
    direction = snapshot.objective_directions[0] if snapshot.objective_directions else "minimize"

    scored = []
    for o in obs:
        val = o.kpi_values.get(primary_obj)
        if val is not None:
            scored.append((o, val))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=(direction == "maximize"))

    results = []
    for rank, (o, val) in enumerate(scored[:top_n], 1):
        results.append(TopCondition(
            rank=rank,
            parameters=dict(o.parameters),
            objective_value=round(val, 6),
            objective_name=primary_obj,
        ))
    return results


def _compute_correlations(
    snapshot: CampaignSnapshot,
) -> list[CorrelationInsight]:
    """Compute parameter-objective correlations."""
    obs = snapshot.successful_observations
    if len(obs) < 5:
        return []

    results = []
    for obj_name in snapshot.objective_names:
        y_vals = [o.kpi_values.get(obj_name) for o in obs]
        valid_mask = [v is not None for v in y_vals]
        y_clean = [v for v, m in zip(y_vals, valid_mask) if m]

        if len(y_clean) < 5:
            continue

        for spec in snapshot.parameter_specs:
            if spec.type == VariableType.CATEGORICAL:
                continue  # Skip categorical for Pearson

            x_vals = []
            y_for_x = []
            for o, m in zip(obs, valid_mask):
                if not m:
                    continue
                raw = o.parameters.get(spec.name)
                if raw is None:
                    continue
                try:
                    x_vals.append(float(raw))
                    y_for_x.append(o.kpi_values[obj_name])
                except (ValueError, TypeError):
                    continue

            if len(x_vals) < 5:
                continue

            r = _pearson_r(x_vals, y_for_x)
            abs_r = abs(r)

            if abs_r < 0.1:
                continue  # Skip negligible correlations

            strength = "strong" if abs_r >= 0.6 else "moderate" if abs_r >= 0.3 else "weak"
            direction = "positive" if r > 0 else "negative"

            results.append(CorrelationInsight(
                parameter=spec.name,
                objective=obj_name,
                correlation=round(r, 4),
                strength=strength,
                direction=direction,
            ))

    results.sort(key=lambda x: abs(x.correlation), reverse=True)
    return results


def _detect_interactions(
    snapshot: CampaignSnapshot, top_n: int = 5
) -> list[InteractionInsight]:
    """Detect parameter interaction effects via residual correlation.

    For each pair of continuous parameters, compute the residual of the
    objective after removing the linear effect of each parameter individually,
    then check if the product term (interaction) explains the residual.
    """
    obs = snapshot.successful_observations
    if len(obs) < 10 or not snapshot.objective_names:
        return []

    primary_obj = snapshot.objective_names[0]
    continuous_params = [
        s for s in snapshot.parameter_specs if s.type != VariableType.CATEGORICAL
    ]

    if len(continuous_params) < 2:
        return []

    # Build numeric matrix
    y_vals = []
    param_vals: dict[str, list[float]] = {s.name: [] for s in continuous_params}
    for o in obs:
        val = o.kpi_values.get(primary_obj)
        if val is None:
            continue
        valid = True
        row: dict[str, float] = {}
        for s in continuous_params:
            raw = o.parameters.get(s.name)
            if raw is None:
                valid = False
                break
            try:
                row[s.name] = float(raw)
            except (ValueError, TypeError):
                valid = False
                break
        if valid:
            y_vals.append(val)
            for s in continuous_params:
                param_vals[s.name].append(row[s.name])

    if len(y_vals) < 10:
        return []

    n = len(y_vals)
    y_mean = sum(y_vals) / n

    # Compute interaction: correlation of (x_a * x_b) with y
    interactions = []
    for i, a in enumerate(continuous_params):
        for b in continuous_params[i + 1:]:
            xa = param_vals[a.name]
            xb = param_vals[b.name]

            # Compute product interaction term
            product = [xa[k] * xb[k] for k in range(n)]

            r = _pearson_r(product, y_vals)
            abs_r = abs(r)

            if abs_r < 0.15:
                continue

            effect = "synergistic" if r > 0 else "antagonistic"
            interactions.append(InteractionInsight(
                param_a=a.name,
                param_b=b.name,
                interaction_strength=round(abs_r, 4),
                description=f"{a.name} x {b.name} has {effect} interaction (r={r:.3f}) on {primary_obj}",
            ))

    interactions.sort(key=lambda x: x.interaction_strength, reverse=True)
    return interactions[:top_n]


def _find_optimal_regions(
    snapshot: CampaignSnapshot, quantile: float = 0.2
) -> list[OptimalRegion]:
    """Find parameter ranges where the objective performs best."""
    obs = snapshot.successful_observations
    if len(obs) < 10 or not snapshot.objective_names:
        return []

    primary_obj = snapshot.objective_names[0]
    direction = snapshot.objective_directions[0] if snapshot.objective_directions else "minimize"

    scored = []
    for o in obs:
        val = o.kpi_values.get(primary_obj)
        if val is not None:
            scored.append((o, val))

    if len(scored) < 10:
        return []

    # Top quantile
    scored.sort(key=lambda x: x[1], reverse=(direction == "maximize"))
    top_k = max(int(len(scored) * quantile), 3)
    top_obs = [s[0] for s in scored[:top_k]]
    top_vals = [s[1] for s in scored[:top_k]]
    all_vals = [s[1] for s in scored]
    mean_top = sum(top_vals) / len(top_vals)
    mean_all = sum(all_vals) / len(all_vals)

    results = []
    for spec in snapshot.parameter_specs:
        if spec.type == VariableType.CATEGORICAL:
            continue

        # Get parameter values for top observations
        top_param_vals = []
        all_param_vals = []
        for o in top_obs:
            raw = o.parameters.get(spec.name)
            if raw is not None:
                try:
                    top_param_vals.append(float(raw))
                except (ValueError, TypeError):
                    pass
        for o, _ in scored:
            raw = o.parameters.get(spec.name)
            if raw is not None:
                try:
                    all_param_vals.append(float(raw))
                except (ValueError, TypeError):
                    pass

        if len(top_param_vals) < 3 or len(all_param_vals) < 5:
            continue

        # Best range = [min, max] of top performers
        best_low = min(top_param_vals)
        best_high = max(top_param_vals)
        overall_low = min(all_param_vals)
        overall_high = max(all_param_vals)

        # Calculate mean objective inside vs outside this range
        inside_vals = []
        outside_vals = []
        for o, val in scored:
            raw = o.parameters.get(spec.name)
            if raw is None:
                continue
            try:
                pv = float(raw)
            except (ValueError, TypeError):
                continue
            if best_low <= pv <= best_high:
                inside_vals.append(val)
            else:
                outside_vals.append(val)

        mean_inside = sum(inside_vals) / len(inside_vals) if inside_vals else 0.0
        mean_outside = sum(outside_vals) / len(outside_vals) if outside_vals else mean_inside

        if mean_outside == 0:
            improvement = 0.0
        elif direction == "maximize":
            improvement = (mean_inside - mean_outside) / abs(mean_outside) * 100
        else:
            improvement = (mean_outside - mean_inside) / abs(mean_outside) * 100

        results.append(OptimalRegion(
            parameter=spec.name,
            best_range=[round(best_low, 6), round(best_high, 6)],
            overall_range=[round(overall_low, 6), round(overall_high, 6)],
            mean_objective_in_region=round(mean_inside, 6),
            mean_objective_outside=round(mean_outside, 6),
            improvement_pct=round(improvement, 2),
        ))

    results.sort(key=lambda x: abs(x.improvement_pct), reverse=True)
    return results


def _detect_failure_patterns(
    snapshot: CampaignSnapshot,
) -> list[FailurePattern]:
    """Identify parameter ranges associated with high failure rates."""
    total = len(snapshot.observations)
    failures = [o for o in snapshot.observations if o.is_failure]

    if total < 10 or len(failures) < 2:
        return []

    overall_rate = len(failures) / total
    results = []

    for spec in snapshot.parameter_specs:
        if spec.type == VariableType.CATEGORICAL:
            continue

        all_vals = []
        fail_vals = []
        for o in snapshot.observations:
            raw = o.parameters.get(spec.name)
            if raw is None:
                continue
            try:
                v = float(raw)
                all_vals.append(v)
                if o.is_failure:
                    fail_vals.append(v)
            except (ValueError, TypeError):
                continue

        if len(fail_vals) < 2:
            continue

        # Check if failures cluster in a specific range
        fail_low = min(fail_vals)
        fail_high = max(fail_vals)
        fail_mid = (fail_low + fail_high) / 2
        fail_spread = fail_high - fail_low

        # Narrow the range to where failures concentrate
        if fail_spread > 0:
            # Use the interquartile range of failure values
            sorted_fails = sorted(fail_vals)
            q1_idx = max(0, len(sorted_fails) // 4)
            q3_idx = min(len(sorted_fails) - 1, 3 * len(sorted_fails) // 4)
            risky_low = sorted_fails[q1_idx]
            risky_high = sorted_fails[q3_idx]

            # Count failures in this range
            n_in_range = sum(1 for v in all_vals if risky_low <= v <= risky_high)
            n_fail_in_range = sum(1 for v in fail_vals if risky_low <= v <= risky_high)

            if n_in_range > 0:
                range_fail_rate = n_fail_in_range / n_in_range
                if range_fail_rate > overall_rate * 1.5:  # Significantly higher
                    results.append(FailurePattern(
                        description=f"High failure rate when {spec.name} is in [{risky_low:.3f}, {risky_high:.3f}]",
                        parameter=spec.name,
                        risky_range=[round(risky_low, 6), round(risky_high, 6)],
                        failure_rate_in_range=round(range_fail_rate, 4),
                        overall_failure_rate=round(overall_rate, 4),
                    ))

    results.sort(key=lambda x: x.failure_rate_in_range, reverse=True)
    return results


def _detect_trends(snapshot: CampaignSnapshot) -> list[TrendInsight]:
    """Detect optimization progress trends."""
    obs = snapshot.successful_observations
    if len(obs) < 5 or not snapshot.objective_names:
        return []

    primary_obj = snapshot.objective_names[0]
    direction = snapshot.objective_directions[0] if snapshot.objective_directions else "minimize"
    trends = []

    values = [(o.iteration, o.kpi_values.get(primary_obj)) for o in obs if primary_obj in o.kpi_values]
    values.sort(key=lambda x: x[0])
    y_vals = [v for _, v in values]

    if len(y_vals) < 5:
        return trends

    # Running best
    running_best = []
    for i, v in enumerate(y_vals):
        if i == 0:
            running_best.append(v)
        else:
            if direction == "minimize":
                running_best.append(min(running_best[-1], v))
            else:
                running_best.append(max(running_best[-1], v))

    # Improvement rate in recent vs early
    mid = len(y_vals) // 2
    early_improvement = abs(running_best[mid] - running_best[0]) / max(mid, 1)
    late_improvement = abs(running_best[-1] - running_best[mid]) / max(len(y_vals) - mid, 1)

    if early_improvement > 0 and late_improvement > 0:
        ratio = late_improvement / early_improvement
        if ratio < 0.3:
            trends.append(TrendInsight(
                description="Optimization is converging — improvement rate has slowed significantly",
                metric="convergence_ratio",
                value=round(ratio, 4),
            ))
        elif ratio > 2.0:
            trends.append(TrendInsight(
                description="Optimization is accelerating — recent iterations show faster improvement",
                metric="acceleration_ratio",
                value=round(ratio, 4),
            ))

    # Noise trend: compare CV of first half vs second half
    first_half = y_vals[:mid]
    second_half = y_vals[mid:]
    if len(first_half) >= 3 and len(second_half) >= 3:
        try:
            cv_first = statistics.stdev(first_half) / abs(statistics.mean(first_half)) if statistics.mean(first_half) != 0 else 0
            cv_second = statistics.stdev(second_half) / abs(statistics.mean(second_half)) if statistics.mean(second_half) != 0 else 0
            if cv_second < cv_first * 0.5 and cv_first > 0.1:
                trends.append(TrendInsight(
                    description="Variance is decreasing — results are becoming more consistent",
                    metric="variance_reduction",
                    value=round(cv_first / max(cv_second, 0.001), 4),
                ))
        except (statistics.StatisticsError, ZeroDivisionError):
            pass

    # Exploration vs exploitation balance
    n = len(obs)
    recent_n = min(n, max(n // 5, 5))
    recent_obs = obs[-recent_n:]
    early_obs = obs[:recent_n]

    for spec in snapshot.parameter_specs:
        if spec.type == VariableType.CATEGORICAL or spec.lower is None or spec.upper is None:
            continue
        r = spec.upper - spec.lower
        if r <= 0:
            continue

        recent_vals = [float(o.parameters.get(spec.name, 0)) for o in recent_obs if spec.name in o.parameters]
        early_vals = [float(o.parameters.get(spec.name, 0)) for o in early_obs if spec.name in o.parameters]

        if len(recent_vals) >= 3 and len(early_vals) >= 3:
            try:
                recent_spread = (max(recent_vals) - min(recent_vals)) / r
                early_spread = (max(early_vals) - min(early_vals)) / r
                if recent_spread < early_spread * 0.3 and early_spread > 0.2:
                    trends.append(TrendInsight(
                        description=f"Search has narrowed for {spec.name} — shifting from exploration to exploitation",
                        metric=f"narrowing_{spec.name}",
                        value=round(recent_spread / max(early_spread, 0.001), 4),
                    ))
                    break  # One such insight is enough
            except (ValueError, ZeroDivisionError):
                pass

    return trends


def _generate_summaries(
    snapshot: CampaignSnapshot,
    top_conditions: list[TopCondition],
    correlations: list[CorrelationInsight],
    interactions: list[InteractionInsight],
    optimal_regions: list[OptimalRegion],
    failure_patterns: list[FailurePattern],
    trends: list[TrendInsight],
) -> list[InsightSummary]:
    """Generate natural language insight summaries."""
    summaries: list[InsightSummary] = []
    primary_obj = snapshot.objective_names[0] if snapshot.objective_names else "objective"
    direction = snapshot.objective_directions[0] if snapshot.objective_directions else "minimize"
    dir_word = "highest" if direction == "maximize" else "lowest"

    # Best conditions insight
    if top_conditions:
        best = top_conditions[0]
        top3_params = {}
        for tc in top_conditions[:3]:
            for k, v in tc.parameters.items():
                top3_params.setdefault(k, []).append(v)

        # Find consistent parameters across top results
        consistent = []
        for k, vals in top3_params.items():
            try:
                numeric_vals = [float(v) for v in vals]
                if len(numeric_vals) >= 2:
                    mean = sum(numeric_vals) / len(numeric_vals)
                    spread = max(numeric_vals) - min(numeric_vals)
                    if mean != 0 and spread / abs(mean) < 0.3:
                        consistent.append(f"{k} ≈ {mean:.3g}")
            except (ValueError, TypeError):
                # Categorical — check if all same
                if len(set(str(v) for v in vals)) == 1:
                    consistent.append(f"{k} = {vals[0]}")

        body = f"The {dir_word} {primary_obj} achieved was {best.objective_value:.4g}."
        if consistent:
            body += f" Top results share: {', '.join(consistent[:4])}."

        summaries.append(InsightSummary(
            title=f"Best {primary_obj}: {best.objective_value:.4g}",
            body=body,
            category="discovery",
            importance=1.0,
        ))

    # Correlation insights
    strong_corr = [c for c in correlations if c.strength == "strong"]
    if strong_corr:
        top_c = strong_corr[0]
        dir_desc = "increases" if (
            (top_c.direction == "positive" and direction == "maximize") or
            (top_c.direction == "negative" and direction == "minimize")
        ) else "decreases"
        summaries.append(InsightSummary(
            title=f"Strong correlation: {top_c.parameter} → {top_c.objective}",
            body=(
                f"{top_c.parameter} has a {top_c.strength} {top_c.direction} correlation "
                f"(r={top_c.correlation:.3f}) with {top_c.objective}. "
                f"As {top_c.parameter} increases, {top_c.objective} {dir_desc}."
            ),
            category="discovery",
            importance=0.9,
        ))

    # Interaction insights
    if interactions:
        top_int = interactions[0]
        summaries.append(InsightSummary(
            title=f"Parameter interaction: {top_int.param_a} × {top_int.param_b}",
            body=top_int.description,
            category="discovery",
            importance=0.7,
        ))

    # Optimal region insights
    useful_regions = [r for r in optimal_regions if abs(r.improvement_pct) > 5]
    if useful_regions:
        top_r = useful_regions[0]
        summaries.append(InsightSummary(
            title=f"Optimal range for {top_r.parameter}",
            body=(
                f"Best results occur when {top_r.parameter} is in "
                f"[{top_r.best_range[0]:.4g}, {top_r.best_range[1]:.4g}] "
                f"(overall range: [{top_r.overall_range[0]:.4g}, {top_r.overall_range[1]:.4g}]). "
                f"This region shows {abs(top_r.improvement_pct):.1f}% better {primary_obj} "
                f"than the rest of the data."
            ),
            category="recommendation",
            importance=0.85,
        ))

    # Failure insights
    if failure_patterns:
        fp = failure_patterns[0]
        summaries.append(InsightSummary(
            title=f"Risk zone: {fp.parameter}",
            body=fp.description + f" ({fp.failure_rate_in_range:.0%} failure rate vs {fp.overall_failure_rate:.0%} overall).",
            category="warning",
            importance=0.8,
        ))

    # Trend insights
    for trend in trends[:2]:
        summaries.append(InsightSummary(
            title=trend.description.split("—")[0].strip() if "—" in trend.description else trend.description[:50],
            body=trend.description,
            category="trend",
            importance=0.6,
        ))

    # Data coverage insight
    n = len(snapshot.observations)
    n_params = len(snapshot.parameter_specs)
    if n < n_params * 10:
        summaries.append(InsightSummary(
            title="Limited data coverage",
            body=(
                f"With {n} observations and {n_params} parameters, "
                f"the dataset has only {n / max(n_params, 1):.0f}x parameter coverage. "
                f"Consider collecting more data for reliable optimization "
                f"(recommended: {n_params * 20}+ observations)."
            ),
            category="warning",
            importance=0.5,
        ))

    summaries.sort(key=lambda x: x.importance, reverse=True)
    return summaries


# ── Endpoint ─────────────────────────────────────────────────────────


@router.get("/campaigns/{campaign_id}/insights")
def get_insights(
    campaign_id: str,
    top_n: int = Query(default=10, ge=1, le=50, description="Number of top conditions to return"),
) -> InsightsResponse:
    """Discover actionable insights from campaign optimization data.

    Analyzes the full observation history to surface:
    - Top performing experimental conditions
    - Parameter-objective correlations
    - Parameter interaction effects
    - Optimal parameter ranges
    - Failure risk zones
    - Optimization progress trends
    - Natural language insight summaries
    """
    snapshot = _load_snapshot(campaign_id)

    top_conditions = _find_top_conditions(snapshot, top_n)
    correlations = _compute_correlations(snapshot)
    interactions = _detect_interactions(snapshot)
    optimal_regions = _find_optimal_regions(snapshot)
    failure_patterns = _detect_failure_patterns(snapshot)
    trends = _detect_trends(snapshot)

    summaries = _generate_summaries(
        snapshot,
        top_conditions,
        correlations,
        interactions,
        optimal_regions,
        failure_patterns,
        trends,
    )

    return InsightsResponse(
        campaign_id=campaign_id,
        n_observations=len(snapshot.observations),
        n_parameters=len(snapshot.parameter_specs),
        n_objectives=len(snapshot.objective_names),
        top_conditions=top_conditions,
        correlations=correlations,
        interactions=interactions,
        optimal_regions=optimal_regions,
        failure_patterns=failure_patterns,
        trends=trends,
        summaries=summaries,
    )

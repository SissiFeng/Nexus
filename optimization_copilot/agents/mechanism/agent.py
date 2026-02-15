"""MechanismHypothesisAgent -- matches optimization data against known domain mechanisms.

Uses hypothesis templates to identify plausible physical/chemical mechanisms
that explain observed patterns in the optimization campaign.
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.agents.base import (
    AgentContext,
    AgentMode,
    OptimizationFeedback,
    ScientificAgent,
    TriggerCondition,
)
from optimization_copilot.agents.mechanism.templates import (
    HypothesisTemplate,
    get_all_templates,
    get_templates_for_domain,
)


_MIN_OBSERVATIONS = 5


class MechanismHypothesisAgent(ScientificAgent):
    """Agent that matches optimization data against known mechanism templates.

    For each template, evaluates whether the observed data pattern matches
    the template's expected behaviour. Produces ranked hypotheses with
    confidence scores.

    Parameters
    ----------
    mode : AgentMode
        Operational mode.
    """

    def __init__(self, mode: AgentMode = AgentMode.PRAGMATIC) -> None:
        super().__init__(mode=mode)
        self._trigger_conditions = [
            TriggerCondition(
                name="sufficient_data_for_hypothesis",
                check_fn_name="check_hypothesis_data",
                priority=4,
                description=(
                    f"Activates when >= {_MIN_OBSERVATIONS} observations and "
                    "domain information is available"
                ),
            ),
            TriggerCondition(
                name="anomaly_explanation_needed",
                check_fn_name="check_anomaly_present",
                priority=6,
                description="Activates when anomalies need mechanistic explanation",
            ),
        ]

    def name(self) -> str:
        return "mechanism_hypothesis"

    def should_activate(self, context: AgentContext) -> bool:
        """Activate when sufficient observations exist and domain is known."""
        if len(context.optimization_history) < _MIN_OBSERVATIONS:
            return False
        # Need domain info from config or metadata
        if context.domain_config is not None:
            return True
        if context.metadata.get("domain"):
            return True
        return False

    def validate_context(self, context: AgentContext) -> bool:
        """Validate that we have optimization history with parameters and KPIs."""
        if not context.optimization_history:
            return False
        first = context.optimization_history[0]
        if "parameters" not in first:
            return False
        return True

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        """Match optimization data against hypothesis templates.

        Returns
        -------
        dict[str, Any]
            Keys: ``hypotheses`` (sorted by confidence), ``domain``,
            ``n_templates_checked``, ``n_matches``.
        """
        domain = self._resolve_domain(context)
        if domain is None:
            return {
                "hypotheses": [],
                "domain": None,
                "n_templates_checked": 0,
                "n_matches": 0,
            }

        templates = get_templates_for_domain(domain)
        if not templates:
            # Fallback: try all templates
            templates = get_all_templates()

        history = context.optimization_history
        param_names = context.get_parameter_names()

        # Evaluate each template
        hypotheses: list[dict[str, Any]] = []
        for template in templates:
            evidence_score = self._evaluate_template(template, history, param_names)
            combined_confidence = template.confidence_prior * evidence_score

            if evidence_score > 0.0:
                hypotheses.append({
                    "name": template.name,
                    "domain": template.domain,
                    "pattern": template.pattern,
                    "mechanism": template.mechanism,
                    "evidence_score": round(evidence_score, 3),
                    "combined_confidence": round(combined_confidence, 3),
                    "parameters_involved": template.parameters_involved,
                    "evidence_required": template.evidence_required,
                })

        # Sort by combined confidence descending
        hypotheses.sort(key=lambda h: h["combined_confidence"], reverse=True)

        return {
            "hypotheses": hypotheses,
            "domain": domain,
            "n_templates_checked": len(templates),
            "n_matches": len(hypotheses),
        }

    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        """Convert hypothesis analysis to optimization feedback."""
        hypotheses = analysis_result.get("hypotheses", [])
        if not hypotheses:
            return None

        top = hypotheses[0]
        confidence = top["combined_confidence"]

        if confidence > 0.6:
            # Strong hypothesis -> suggest constraint or warning
            feedback_type = "warning"
            reasoning = (
                f"Mechanism hypothesis '{top['name']}' matched with "
                f"confidence {confidence:.2f}. Pattern: {top['pattern']}. "
                f"Mechanism: {top['mechanism']}"
            )
        else:
            # Weak hypothesis -> just report
            feedback_type = "hypothesis"
            reasoning = (
                f"Possible mechanism '{top['name']}' (confidence {confidence:.2f}). "
                f"Pattern: {top['pattern']}. Further evidence needed: "
                f"{', '.join(top['evidence_required'])}"
            )

        return OptimizationFeedback(
            agent_name=self.name(),
            feedback_type=feedback_type,
            confidence=confidence,
            payload={
                "top_hypothesis": top["name"],
                "mechanism": top["mechanism"],
                "parameters": top["parameters_involved"],
                "n_hypotheses": len(hypotheses),
                "all_hypotheses": [h["name"] for h in hypotheses],
            },
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_domain(context: AgentContext) -> str | None:
        """Determine domain from context."""
        if context.domain_config is not None:
            return context.domain_config.domain_name
        domain = context.metadata.get("domain")
        if isinstance(domain, str) and domain:
            return domain
        return None

    @staticmethod
    def _evaluate_template(
        template: HypothesisTemplate,
        history: list[dict[str, Any]],
        param_names: list[str],
    ) -> float:
        """Evaluate how well a template matches the observed data.

        Returns an evidence score between 0.0 and 1.0.

        Scoring factors:
        1. Parameter overlap: do the template's parameters appear in the data?
        2. Pattern detection: look for the described pattern in the data.
        """
        if not history:
            return 0.0

        score = 0.0
        max_score = 0.0

        # Factor 1: Parameter overlap (weight = 0.4)
        max_score += 0.4
        if template.parameters_involved:
            matched = sum(
                1 for p in template.parameters_involved
                if p in param_names
            )
            overlap = matched / len(template.parameters_involved)
            score += 0.4 * overlap
        else:
            score += 0.2  # no specific parameters required -> partial match

        # Factor 2: Sufficient data volume (weight = 0.2)
        max_score += 0.2
        n_obs = len(history)
        if n_obs >= 20:
            score += 0.2
        elif n_obs >= 10:
            score += 0.15
        elif n_obs >= 5:
            score += 0.1

        # Factor 3: Pattern detection in KPI data (weight = 0.4)
        max_score += 0.4
        pattern_score = _detect_pattern(template, history, param_names)
        score += 0.4 * pattern_score

        if max_score < 1e-10:
            return 0.0
        return min(1.0, score / max_score) if max_score > 0 else 0.0


# ---------------------------------------------------------------------------
# Pattern detection helpers
# ---------------------------------------------------------------------------


def _detect_pattern(
    template: HypothesisTemplate,
    history: list[dict[str, Any]],
    param_names: list[str],
) -> float:
    """Detect whether the described pattern exists in the data.

    Returns a score between 0.0 and 1.0.
    """
    pattern_lower = template.pattern.lower()

    # Extract KPI values
    kpi_values = _extract_kpi_values(history)
    if not kpi_values:
        return 0.0

    score = 0.0

    # Check for "sudden drop" patterns
    if "drop" in pattern_lower or "decrease" in pattern_lower:
        drop_score = _detect_sudden_change(kpi_values, direction="drop")
        score = max(score, drop_score)

    # Check for "increase" patterns
    if "increase" in pattern_lower or "improves" in pattern_lower:
        inc_score = _detect_sudden_change(kpi_values, direction="increase")
        score = max(score, inc_score)

    # Check for "plateau" patterns
    if "plateau" in pattern_lower or "limit" in pattern_lower:
        plat_score = _detect_plateau(kpi_values)
        score = max(score, plat_score)

    # Check for "non-monotonic" or "optimal" patterns
    if "non-monotonic" in pattern_lower or "optimal" in pattern_lower or "peak" in pattern_lower:
        nm_score = _detect_non_monotonic(kpi_values)
        score = max(score, nm_score)

    # Check for "sensitive" patterns (high variance)
    if "sensitive" in pattern_lower or "highly" in pattern_lower:
        var_score = _detect_high_variance(kpi_values)
        score = max(score, var_score)

    # Check parameter-specific trends
    for param in template.parameters_involved:
        if param in param_names:
            param_kpi = _extract_param_kpi_pairs(history, param)
            if param_kpi:
                trend = _detect_trend(param_kpi)
                if "decrease" in pattern_lower and trend < -0.3:
                    score = max(score, abs(trend))
                elif "increase" in pattern_lower and trend > 0.3:
                    score = max(score, abs(trend))
                elif ("non-monotonic" in pattern_lower or "optimal" in pattern_lower):
                    nm = _detect_non_monotonic([v for _, v in param_kpi])
                    score = max(score, nm)

    # If no specific pattern detected but parameters match, give base score
    if score == 0.0 and any(p in param_names for p in template.parameters_involved):
        score = 0.2

    return min(1.0, score)


def _extract_kpi_values(history: list[dict[str, Any]]) -> list[float]:
    """Extract the primary KPI values from history."""
    values: list[float] = []
    for entry in history:
        for key in ("y", "objective", "kpi", "target", "value", "score"):
            if key in entry:
                try:
                    v = float(entry[key])
                    if math.isfinite(v):
                        values.append(v)
                except (TypeError, ValueError):
                    pass
                break
    return values


def _extract_param_kpi_pairs(
    history: list[dict[str, Any]], param: str
) -> list[tuple[float, float]]:
    """Extract (parameter_value, kpi_value) pairs."""
    pairs: list[tuple[float, float]] = []
    for entry in history:
        params = entry.get("parameters", {})
        if param not in params:
            continue
        try:
            pval = float(params[param])
        except (TypeError, ValueError):
            continue

        for key in ("y", "objective", "kpi", "target", "value", "score"):
            if key in entry:
                try:
                    kval = float(entry[key])
                    if math.isfinite(pval) and math.isfinite(kval):
                        pairs.append((pval, kval))
                except (TypeError, ValueError):
                    pass
                break
    return pairs


def _detect_sudden_change(values: list[float], direction: str = "drop") -> float:
    """Detect sudden drops or increases in a value series."""
    if len(values) < 3:
        return 0.0

    max_change = 0.0
    val_range = max(values) - min(values) if max(values) != min(values) else 1.0

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        normalized = abs(change) / val_range if val_range > 1e-10 else 0.0

        if direction == "drop" and change < 0:
            max_change = max(max_change, normalized)
        elif direction == "increase" and change > 0:
            max_change = max(max_change, normalized)

    return min(1.0, max_change * 2.0)


def _detect_plateau(values: list[float]) -> float:
    """Detect plateau regions in a value series."""
    if len(values) < 5:
        return 0.0

    val_range = max(values) - min(values)
    if val_range < 1e-10:
        return 0.8  # constant is a strong plateau

    # Check if the last portion is relatively flat
    n = len(values)
    tail = values[n // 2:]
    tail_range = max(tail) - min(tail)
    flatness = 1.0 - (tail_range / val_range)
    return max(0.0, flatness)


def _detect_non_monotonic(values: list[float]) -> float:
    """Detect non-monotonic behaviour (presence of local extrema)."""
    if len(values) < 3:
        return 0.0

    direction_changes = 0
    for i in range(1, len(values) - 1):
        if (values[i] > values[i - 1] and values[i] > values[i + 1]) or \
           (values[i] < values[i - 1] and values[i] < values[i + 1]):
            direction_changes += 1

    if direction_changes == 0:
        return 0.0
    return min(1.0, direction_changes / (len(values) / 3))


def _detect_high_variance(values: list[float]) -> float:
    """Detect high variance in values."""
    if len(values) < 3:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5

    if abs(mean) < 1e-10:
        return 0.5

    cv = std / abs(mean)
    return min(1.0, cv * 2.0)


def _detect_trend(pairs: list[tuple[float, float]]) -> float:
    """Simple correlation between parameter and KPI values.

    Returns a value between -1 and 1 (Pearson-like).
    """
    if len(pairs) < 3:
        return 0.0

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    n = len(pairs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
    var_x = sum((x - mean_x) ** 2 for x in xs) / n
    var_y = sum((y - mean_y) ** 2 for y in ys) / n

    if var_x < 1e-12 or var_y < 1e-12:
        return 0.0

    r = cov_xy / (var_x ** 0.5 * var_y ** 0.5)
    return max(-1.0, min(1.0, r))

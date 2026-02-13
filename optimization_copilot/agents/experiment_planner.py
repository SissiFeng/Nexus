"""ExperimentPlannerAgent — LLM-driven experiment planning agent.

Generates experiment hypotheses and recommendations based on optimization
campaign state.  Operates in two modes:

- **PRAGMATIC**: Rule-based heuristics that analyze campaign trends and
  produce actionable suggestions without requiring an external API.
- **LLM_ENHANCED**: Calls Claude via the Anthropic API to generate richer
  hypotheses.  Falls back to PRAGMATIC automatically when the API key is
  missing or the call fails.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.agents.base import (
    AgentContext,
    AgentMode,
    OptimizationFeedback,
    ScientificAgent,
    TriggerCondition,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are a scientific experiment planning assistant for Bayesian "
    "optimization campaigns. Given a summary of the campaign so far "
    "(iterations, best result, trend, parameters, objectives, and any "
    "anomalies), propose a concrete hypothesis about which region of the "
    "search space to explore next and why. Keep your response structured "
    "with the following sections:\n"
    "1. SUMMARY — one-sentence assessment of campaign health.\n"
    "2. HYPOTHESIS — a testable hypothesis about where gains may be found.\n"
    "3. RECOMMENDATIONS — 2-4 actionable next steps for the optimizer.\n"
    "Be concise and quantitative where possible."
)

_ACTIVATION_INTERVAL: int = 10  # activate every N iterations


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PlannerConfig:
    """Configuration for the :class:`ExperimentPlannerAgent`.

    Parameters
    ----------
    model_name : str
        Anthropic model to use for LLM-enhanced mode.
    max_tokens : int
        Maximum tokens in the LLM response.
    temperature : float
        Sampling temperature for the LLM call.
    system_prompt : str
        System prompt sent to the LLM for experiment planning.
    """

    model_name: str = "claude-opus-4-6"
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_trend(history: list[dict[str, Any]]) -> str:
    """Classify recent objective trend as improving / stagnating / degrading.

    Uses the last *min(5, len)* entries.  Compares the mean of the first
    half to the mean of the second half (lower objective is better).
    """
    if len(history) < 2:
        return "insufficient_data"

    window = history[-min(5, len(history)):]
    objectives = [
        entry.get("objective", entry.get("value", 0.0))
        for entry in window
    ]

    mid = len(objectives) // 2
    first_half = objectives[:mid] if mid > 0 else objectives[:1]
    second_half = objectives[mid:]

    mean_first = sum(first_half) / len(first_half)
    mean_second = sum(second_half) / len(second_half)

    delta = mean_second - mean_first
    # Use a 1% relative tolerance to distinguish stagnation from change
    tol = 0.01 * (abs(mean_first) + 1e-12)
    if delta < -tol:
        return "improving"
    elif delta > tol:
        return "degrading"
    return "stagnating"


def _best_result(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the entry with the lowest objective value."""
    if not history:
        return {}
    return min(
        history,
        key=lambda e: e.get("objective", e.get("value", float("inf"))),
    )


def _build_campaign_summary(context: AgentContext) -> dict[str, Any]:
    """Distil *context* into a compact campaign summary dict."""
    history = context.optimization_history
    best = _best_result(history)
    trend = _compute_trend(history)

    # Count parameters from first history entry or domain config
    param_names = context.get_parameter_names()
    n_params = len(param_names) if param_names else 0

    # Objectives — assume single unless metadata says otherwise
    n_objectives: int = context.metadata.get("n_objectives", 1)

    anomaly_count = len(context.anomalies) if context.anomalies else 0

    return {
        "total_iterations": context.iteration,
        "history_length": len(history),
        "best_result": best,
        "trend": trend,
        "n_parameters": n_params,
        "parameter_names": param_names,
        "n_objectives": n_objectives,
        "anomaly_count": anomaly_count,
        "event": context.metadata.get("event", ""),
    }


# ---------------------------------------------------------------------------
# Pragmatic (rule-based) analysis
# ---------------------------------------------------------------------------


def _pragmatic_analysis(summary: dict[str, Any]) -> dict[str, Any]:
    """Generate a rule-based experiment plan from the campaign summary."""
    trend = summary["trend"]
    iteration = summary["total_iterations"]
    event = summary.get("event", "")

    # --- Hypothesis ---
    if trend == "stagnating" or event == "stagnation":
        hypothesis = (
            "The optimizer appears stuck in a local basin.  Increasing "
            "exploration diversity (e.g., larger length-scale, random "
            "restarts, or a Latin-hypercube batch in an unexplored region) "
            "may reveal a better optimum."
        )
        recommendations = [
            "Increase acquisition-function exploration weight (e.g., raise "
            "kappa/xi).",
            "Sample a space-filling batch in regions far from current best.",
            "Consider adding a small random perturbation to the GP kernel "
            "length-scales to escape the local basin.",
        ]
    elif trend == "improving":
        hypothesis = (
            "The campaign is making steady progress.  Continue exploiting "
            "the current promising region while maintaining a modest "
            "exploration budget to avoid premature convergence."
        )
        recommendations = [
            "Narrow the search region around the current best by ~20%.",
            "Maintain at least 10-20% exploration budget in the "
            "acquisition function.",
            "Monitor for diminishing returns in the next 5 iterations.",
        ]
    elif trend == "degrading":
        hypothesis = (
            "Recent evaluations are worse than earlier ones — the "
            "optimizer may be exploring a poor region or the surrogate "
            "model may be mis-calibrated.  Re-centring on the historical "
            "best region could recover performance."
        )
        recommendations = [
            "Re-centre the search region around the historical best "
            "parameters.",
            "Re-fit the GP model with fresh hyper-parameter optimization.",
            "Check for data quality issues or constraint violations in "
            "recent evaluations.",
        ]
    elif iteration <= 5:
        # Early-stage: not enough data to judge trend
        hypothesis = (
            "The campaign is in its early stage with limited data.  A "
            "space-filling design will maximise information gain before "
            "switching to model-guided search."
        )
        recommendations = [
            "Run a Latin-hypercube or Sobol batch to fill the design "
            "space.",
            "Defer model-guided acquisition until at least 2*d "
            "evaluations are available.",
        ]
    else:
        # Fallback: insufficient data to classify trend
        hypothesis = (
            "Not enough trend information to form a strong hypothesis.  "
            "Continue the current strategy and re-evaluate after more "
            "data is collected."
        )
        recommendations = [
            "Continue with the current acquisition strategy.",
            "Collect at least 5 more evaluations before re-assessing.",
        ]

    return {
        "summary": summary,
        "hypothesis": hypothesis,
        "recommendations": recommendations,
        "mode_used": "pragmatic",
    }


# ---------------------------------------------------------------------------
# LLM-enhanced analysis
# ---------------------------------------------------------------------------


def _llm_analysis(
    summary: dict[str, Any],
    config: PlannerConfig,
) -> dict[str, Any] | None:
    """Call Claude API and parse a structured experiment plan.

    Returns ``None`` if the call cannot be made (missing key, import error,
    API failure).
    """
    api_key = os.environ.get("MODEL_API_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY"
    )
    if not api_key:
        return None

    try:
        import anthropic  # type: ignore[import-untyped]
    except ImportError:
        return None

    user_message = (
        "Here is the current campaign state:\n"
        f"- Total iterations: {summary['total_iterations']}\n"
        f"- History length: {summary['history_length']}\n"
        f"- Best result: {summary['best_result']}\n"
        f"- Trend: {summary['trend']}\n"
        f"- Parameters ({summary['n_parameters']}): "
        f"{summary['parameter_names']}\n"
        f"- Objectives: {summary['n_objectives']}\n"
        f"- Anomalies detected: {summary['anomaly_count']}\n"
        f"- Current event: {summary['event']}\n\n"
        "Please provide your analysis."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=config.model_name,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=config.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        text: str = response.content[0].text  # type: ignore[union-attr]
    except Exception:
        return None

    # --- Parse structured sections from the LLM response ---
    hypothesis = ""
    recommendations: list[str] = []
    llm_summary = ""

    current_section: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("SUMMARY") or upper.startswith("1."):
            current_section = "summary"
            # Grab inline content after the heading marker
            after = stripped.split("—", 1)[-1].split("-", 1)[-1].strip()
            if after and after != stripped:
                llm_summary = after
            continue
        if upper.startswith("HYPOTHESIS") or upper.startswith("2."):
            current_section = "hypothesis"
            after = stripped.split("—", 1)[-1].split("-", 1)[-1].strip()
            if after and after != stripped:
                hypothesis = after
            continue
        if upper.startswith("RECOMMENDATION") or upper.startswith("3."):
            current_section = "recommendations"
            continue

        if not stripped:
            continue

        if current_section == "summary":
            llm_summary += (" " + stripped) if llm_summary else stripped
        elif current_section == "hypothesis":
            hypothesis += (" " + stripped) if hypothesis else stripped
        elif current_section == "recommendations":
            # Strip leading bullet characters
            clean = stripped.lstrip("-*0123456789.) ").strip()
            if clean:
                recommendations.append(clean)

    # If parsing failed to find sections, use the full text as hypothesis
    if not hypothesis:
        hypothesis = text[:500]
    if not recommendations:
        recommendations = ["(see full LLM response for details)"]

    return {
        "summary": summary,
        "llm_summary": llm_summary or "(not parsed)",
        "hypothesis": hypothesis,
        "recommendations": recommendations,
        "mode_used": "llm_enhanced",
        "raw_response": text,
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ExperimentPlannerAgent(ScientificAgent):
    """LLM-driven (or rule-based) experiment planning agent.

    In **PRAGMATIC** mode the agent uses deterministic heuristics to
    suggest next steps.  In **LLM_ENHANCED** mode it calls the Claude API,
    falling back to pragmatic analysis on failure.

    Parameters
    ----------
    config : PlannerConfig | None
        LLM configuration.  A default is created when ``None``.
    mode : AgentMode
        Operational mode (default ``PRAGMATIC``).
    """

    def __init__(
        self,
        config: PlannerConfig | None = None,
        mode: AgentMode = AgentMode.PRAGMATIC,
    ) -> None:
        super().__init__(mode=mode)
        self.config = config or PlannerConfig()

        # Register trigger conditions
        self._trigger_conditions = [
            TriggerCondition(
                name="milestone_event",
                check_fn_name="check_milestone",
                priority=10,
                description=(
                    "Activates when a milestone event is recorded in the "
                    "campaign metadata."
                ),
            ),
            TriggerCondition(
                name="stagnation_event",
                check_fn_name="check_stagnation",
                priority=20,
                description=(
                    "Activates when stagnation is detected in the "
                    "optimization progress."
                ),
            ),
        ]

    # ── ScientificAgent interface ─────────────────────────────────

    def name(self) -> str:  # noqa: D401
        """Unique agent identifier."""
        return "experiment_planner"

    def should_activate(self, context: AgentContext) -> bool:
        """Decide whether the planner should run for *context*.

        Activates when:
        - The metadata contains a ``"milestone"`` or ``"stagnation"`` event.
        - The current iteration is a multiple of the activation interval.
        - The agent is running in ``LLM_ENHANCED`` mode (always active).
        """
        if self.mode is AgentMode.LLM_ENHANCED:
            return True

        event = context.metadata.get("event", "")
        if event in {"milestone", "stagnation"}:
            return True

        if (
            context.iteration > 0
            and context.iteration % _ACTIVATION_INTERVAL == 0
        ):
            return True

        return False

    def validate_context(self, context: AgentContext) -> bool:
        """Require at least one entry in ``optimization_history``."""
        return len(context.optimization_history) >= 1

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        """Build a campaign summary and generate an experiment plan.

        In LLM_ENHANCED mode the agent attempts a Claude API call first
        and falls back to pragmatic analysis on any failure.
        """
        summary = _build_campaign_summary(context)

        if self.mode is AgentMode.LLM_ENHANCED:
            result = _llm_analysis(summary, self.config)
            if result is not None:
                return result
            # Fallback to pragmatic
        return _pragmatic_analysis(summary)

    def get_optimization_feedback(
        self,
        analysis_result: dict[str, Any],
    ) -> OptimizationFeedback | None:
        """Convert analysis results into an :class:`OptimizationFeedback`.

        Returns feedback with ``feedback_type="hypothesis"`` when the
        analysis contains a hypothesis, otherwise ``None``.
        """
        hypothesis: str = analysis_result.get("hypothesis", "")
        if not hypothesis:
            return None

        # Confidence heuristic: LLM responses get higher base confidence
        mode_used = analysis_result.get("mode_used", "pragmatic")
        base_confidence = 0.6 if mode_used == "llm_enhanced" else 0.4

        # Adjust confidence based on data richness
        summary = analysis_result.get("summary", {})
        history_len = summary.get("history_length", 0)
        if history_len >= 20:
            confidence = min(base_confidence + 0.2, 0.8)
        elif history_len >= 10:
            confidence = min(base_confidence + 0.1, 0.8)
        elif history_len < 3:
            confidence = max(base_confidence - 0.1, 0.3)
        else:
            confidence = base_confidence

        recommendations = analysis_result.get("recommendations", [])
        reasoning = hypothesis
        if recommendations:
            reasoning += " Recommendations: " + "; ".join(recommendations)

        return OptimizationFeedback(
            agent_name=self.name(),
            feedback_type="hypothesis",
            confidence=round(confidence, 2),
            payload={
                "hypothesis": hypothesis,
                "recommendations": recommendations,
                "mode_used": mode_used,
            },
            reasoning=reasoning,
        )

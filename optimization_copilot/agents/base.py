"""ScientificAgent ABC and shared types for the agent layer.

Defines the abstract base class all scientific reasoning agents must implement,
along with shared data types for agent communication: ``AgentContext`` carries
information *into* an agent, ``OptimizationFeedback`` carries recommendations
*out* of an agent back to the optimization loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from optimization_copilot.uncertainty.types import (
    MeasurementWithUncertainty,
    ObservationWithNoise,
)
from optimization_copilot.domain_knowledge.loader import DomainConfig


# ── Enums ──────────────────────────────────────────────────────────────


class AgentMode(str, Enum):
    """Operational mode for an agent."""

    PRAGMATIC = "pragmatic"          # deterministic, rule-based
    LLM_ENHANCED = "llm_enhanced"    # future: LLM-augmented reasoning


# ── Shared Data Types ──────────────────────────────────────────────────


@dataclass
class TriggerCondition:
    """Condition that determines when an agent should activate.

    Parameters
    ----------
    name : str
        Human-readable name for this trigger.
    check_fn_name : str
        Name of the method on the orchestrator that evaluates this condition.
    priority : int
        Higher values are checked first.
    description : str
        Explanation of what this trigger detects.
    """

    name: str
    check_fn_name: str
    priority: int = 0
    description: str = ""


@dataclass
class AgentContext:
    """Context passed to agents when they are invoked.

    Carries all the information an agent needs to perform its analysis,
    including the GP model, optimization history, raw data, anomalies,
    domain configuration, uncertainty-aware measurements, diagnostics,
    and the full campaign snapshot.
    """

    gp_model: Any = None                                          # mod #1: unified name
    optimization_history: list[dict] = field(default_factory=list)
    raw_data: dict | None = None
    anomalies: list | None = None
    domain_config: DomainConfig | None = None                      # mod #2: class, not dict
    measurements: list[MeasurementWithUncertainty] | None = None   # reuse v4 type
    diagnostics: Any = None    # DiagnosticsVector if available
    campaign_snapshot: Any = None  # CampaignSnapshot if available
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_parameter_names(self) -> list[str]:
        """Extract parameter names from DomainConfig or optimization_history.

        Tries the domain configuration first (via ``get_constraints()``),
        falling back to the first entry in ``optimization_history``.
        Returns an empty list if neither source is available.
        """
        if self.domain_config is not None:
            constraints = self.domain_config.get_constraints()
            if constraints:
                return list(constraints.keys())
        if self.optimization_history:
            first = self.optimization_history[0]
            if "parameters" in first:
                return list(first["parameters"].keys())
        return []

    def has_gp_model(self) -> bool:
        """Whether a GP model is available in this context."""
        return self.gp_model is not None

    def has_measurements(self) -> bool:
        """Whether uncertainty-aware measurements are available."""
        return self.measurements is not None and len(self.measurements) > 0

    def has_diagnostics(self) -> bool:
        """Whether a diagnostics vector is available."""
        return self.diagnostics is not None

    def has_campaign_snapshot(self) -> bool:
        """Whether a campaign snapshot is available."""
        return self.campaign_snapshot is not None


@dataclass
class OptimizationFeedback:
    """Feedback from an agent back to the optimization loop.

    Parameters
    ----------
    agent_name : str
        Which agent produced this feedback.
    feedback_type : str
        One of: ``"prior_update"``, ``"constraint_addition"``,
        ``"reweight"``, ``"hypothesis"``, ``"warning"``.
    confidence : float
        Agent's confidence in this recommendation (0.0-1.0).
    payload : dict
        Structured data for the optimization loop to consume.
    reasoning : str
        Human-readable explanation.
    """

    agent_name: str
    feedback_type: str   # "prior_update", "constraint_addition", "reweight", "hypothesis", "warning"
    confidence: float    # 0.0-1.0
    payload: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    ACTIONABLE_TYPES = frozenset({
        "prior_update",
        "constraint_addition",
        "reweight",
    })

    def is_actionable(self) -> bool:
        """Whether this feedback should modify the optimization loop.

        Returns ``True`` when confidence >= 0.5 and the feedback type
        is one of the actionable types (prior_update, constraint_addition,
        reweight).
        """
        return (
            self.confidence >= 0.5
            and self.feedback_type in self.ACTIONABLE_TYPES
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "agent_name": self.agent_name,
            "feedback_type": self.feedback_type,
            "confidence": self.confidence,
            "payload": dict(self.payload),
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationFeedback:
        """Reconstruct from a plain dict."""
        return cls(
            agent_name=data["agent_name"],
            feedback_type=data["feedback_type"],
            confidence=data["confidence"],
            payload=data.get("payload", {}),
            reasoning=data.get("reasoning", ""),
        )


# ── Abstract Base Class ───────────────────────────────────────────────


class ScientificAgent(ABC):
    """Abstract base class for all scientific reasoning agents.

    Subclasses must implement:
    - ``name()`` -- unique identifier string
    - ``analyze(context)`` -- core analysis returning a result dict
    - ``get_optimization_feedback(analysis_result)`` -- convert analysis
      results into optimization feedback (or ``None`` if no action needed)

    Optionally override:
    - ``should_activate(context)`` -- conditional activation (default: True)
    - ``validate_context(context)`` -- check minimum data requirements
    """

    def __init__(self, mode: AgentMode = AgentMode.PRAGMATIC) -> None:
        self.mode = mode
        self._trigger_conditions: list[TriggerCondition] = []

    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier."""
        ...

    @abstractmethod
    def analyze(self, context: AgentContext) -> dict[str, Any]:
        """Run the agent's analysis and return results.

        Parameters
        ----------
        context : AgentContext
            All data the agent needs.

        Returns
        -------
        dict[str, Any]
            Analysis results; structure is agent-specific.
        """
        ...

    @abstractmethod
    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        """Convert analysis results to optimization feedback.

        Parameters
        ----------
        analysis_result : dict[str, Any]
            Output from ``analyze()``.

        Returns
        -------
        OptimizationFeedback | None
            Feedback for the optimization loop, or ``None`` if no action.
        """
        ...

    @property
    def trigger_conditions(self) -> list[TriggerCondition]:
        """Registered trigger conditions for this agent."""
        return self._trigger_conditions

    def should_activate(self, context: AgentContext) -> bool:
        """Determine whether this agent should run for the given context.

        Default implementation always returns ``True``.
        Override in subclasses for conditional activation.
        """
        return True

    def validate_context(self, context: AgentContext) -> bool:
        """Check if context has minimum required data for this agent.

        Default implementation always returns ``True``.
        Override in subclasses to enforce data requirements.
        """
        return True

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name()!r}, "
            f"mode={self.mode.value!r})"
        )

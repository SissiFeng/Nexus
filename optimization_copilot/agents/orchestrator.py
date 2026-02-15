"""Event-driven agent orchestrator.

The ``ScientificOrchestrator`` is the single entry point for all agent
interactions (mod #4).  It manages agent registration, event dispatch,
anomaly detection integration, and maintains an audit trail of all
agent activations and feedback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.agents.base import (
    AgentContext,
    AgentMode,
    OptimizationFeedback,
    ScientificAgent,
)
from optimization_copilot.domain_knowledge.loader import DomainConfig


# ── Orchestrator Data Types ───────────────────────────────────────────


@dataclass
class OrchestratorEvent:
    """An event dispatched through the orchestrator.

    Parameters
    ----------
    event_type : str
        One of: ``"observation"``, ``"anomaly"``, ``"milestone"``,
        ``"stagnation"``, ``"drift"``.
    data : dict
        Event-specific payload.
    timestamp : float
        Time the event was created (epoch seconds).
    """

    event_type: str   # "observation", "anomaly", "milestone", "stagnation", "drift"
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class AuditEntry:
    """Record of a single dispatch cycle.

    Parameters
    ----------
    event : OrchestratorEvent
        The event that triggered the dispatch.
    agents_triggered : list[str]
        Names of agents that were activated.
    feedbacks : list[OptimizationFeedback]
        Feedback produced by the agents.
    elapsed_ms : float
        Wall-clock time for the full dispatch cycle.
    """

    event: OrchestratorEvent
    agents_triggered: list[str] = field(default_factory=list)
    feedbacks: list[OptimizationFeedback] = field(default_factory=list)
    elapsed_ms: float = 0.0
    execution_traces: list[dict] = field(default_factory=list)


# ── Orchestrator ──────────────────────────────────────────────────────


class ScientificOrchestrator:
    """Event-driven dispatcher for scientific reasoning agents.

    The orchestrator is the single entry point (mod #4) for all agent
    interactions.  It:

    1. Manages a registry of ``ScientificAgent`` instances.
    2. Dispatches ``OrchestratorEvent`` objects to registered agents.
    3. Optionally integrates anomaly detection via ``AnomalyDetector``.
    4. Maintains a full audit trail of all dispatch cycles.
    5. Provides convenience methods for common dispatch patterns
       (``on_observation``, ``chain``, ``parallel``).

    Parameters
    ----------
    agents : list[ScientificAgent] | None
        Initial agents to register.
    domain_config : DomainConfig | None
        Domain configuration shared with all agents.
    confidence_threshold : float
        Minimum confidence for feedback to pass validation (default 0.5).
    """

    def __init__(
        self,
        agents: list[ScientificAgent] | None = None,
        domain_config: DomainConfig | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self._agents: list[ScientificAgent] = list(agents) if agents else []
        self._domain_config = domain_config
        self._confidence_threshold = confidence_threshold
        self._audit_trail: list[AuditEntry] = []

        # Lazy-initialised anomaly integrations.  These are set to ``None``
        # by default and can be assigned from the anomaly package when available.
        self._anomaly_detector: Any = None   # AnomalyDetector
        self._anomaly_handler: Any = None    # AnomalyHandler

    # ── Agent Management ──────────────────────────────────────────

    def register_agent(self, agent: ScientificAgent) -> None:
        """Register an agent with the orchestrator.

        Raises ``ValueError`` if an agent with the same name is already
        registered.
        """
        existing_names = {a.name() for a in self._agents}
        if agent.name() in existing_names:
            raise ValueError(
                f"Agent with name {agent.name()!r} is already registered."
            )
        self._agents.append(agent)

    def get_active_agents(self) -> list[str]:
        """Return names of all registered agents."""
        return [a.name() for a in self._agents]

    # ── Core Dispatch ─────────────────────────────────────────────

    def dispatch_event(
        self,
        event: OrchestratorEvent,
        context: AgentContext,
    ) -> list[OptimizationFeedback]:
        """Dispatch an event to all eligible agents.

        For each registered agent:
        1. Check ``should_activate(context)``
        2. Check ``validate_context(context)``
        3. Call ``analyze(context)``
        4. Call ``get_optimization_feedback(result)``
        5. Collect non-None feedbacks

        Parameters
        ----------
        event : OrchestratorEvent
            The event triggering this dispatch.
        context : AgentContext
            Shared context for all agents.

        Returns
        -------
        list[OptimizationFeedback]
            All non-None feedback from activated agents.
        """
        start = time.monotonic()
        feedbacks: list[OptimizationFeedback] = []
        triggered: list[str] = []

        # Inject event data into context metadata
        context.metadata["event_type"] = event.event_type
        context.metadata["event_data"] = event.data

        for agent in self._agents:
            if not agent.should_activate(context):
                continue
            if not agent.validate_context(context):
                continue

            triggered.append(agent.name())
            try:
                result = agent.analyze(context)
                feedback = agent.get_optimization_feedback(result)
                if feedback is not None:
                    feedbacks.append(feedback)
            except Exception:
                # Agents must not crash the orchestrator.
                feedbacks.append(
                    OptimizationFeedback(
                        agent_name=agent.name(),
                        feedback_type="warning",
                        confidence=0.0,
                        reasoning=f"Agent {agent.name()!r} raised an exception.",
                    )
                )

        elapsed = (time.monotonic() - start) * 1000.0
        self._audit_trail.append(
            AuditEntry(
                event=event,
                agents_triggered=triggered,
                feedbacks=feedbacks,
                elapsed_ms=elapsed,
            )
        )
        return feedbacks

    # ── Convenience: on_observation ───────────────────────────────

    def on_observation(
        self,
        x: list[float],
        y: float,
        raw_data: dict | None = None,
        kpi_values: dict | None = None,
        context: AgentContext | None = None,
    ) -> list[OptimizationFeedback]:
        """Handle a new observation.

        1. Create or enrich the context.
        2. Run anomaly detection (if detector is available).
        3. If anomalous, dispatch an ``"anomaly"`` event first.
        4. Dispatch the ``"observation"`` event to all agents.
        5. Return collected feedbacks.

        Parameters
        ----------
        x : list[float]
            Parameter values.
        y : float
            Objective value.
        raw_data : dict | None
            Raw experimental data for signal checks.
        kpi_values : dict | None
            KPI values for KPI validation.
        context : AgentContext | None
            Pre-built context; one is created if ``None``.

        Returns
        -------
        list[OptimizationFeedback]
            Combined feedbacks from anomaly and observation dispatch.
        """
        if context is None:
            context = AgentContext(
                domain_config=self._domain_config,
            )
        context.raw_data = raw_data
        context.metadata["x"] = x
        context.metadata["y"] = y
        if kpi_values is not None:
            context.metadata["kpi_values"] = kpi_values

        all_feedbacks: list[OptimizationFeedback] = []

        # Step 1: Anomaly detection
        if self._anomaly_detector is not None:
            try:
                report = self._anomaly_detector.detect(
                    x=x,
                    y=y,
                    raw_data=raw_data or {},
                    kpi_values=kpi_values or {},
                )
                context.anomalies = _extract_anomaly_list(report)
                context.metadata["anomaly_report"] = report

                if report.is_anomalous:
                    anomaly_event = OrchestratorEvent(
                        event_type="anomaly",
                        data={
                            "severity": report.severity,
                            "summary": report.summary,
                            "is_anomalous": True,
                        },
                    )
                    anomaly_feedbacks = self.dispatch_event(
                        anomaly_event, context
                    )
                    all_feedbacks.extend(anomaly_feedbacks)
            except Exception:
                # Anomaly detection failure should not block observation processing.
                pass

        # Step 2: Dispatch observation event
        obs_event = OrchestratorEvent(
            event_type="observation",
            data={"x": x, "y": y},
        )
        obs_feedbacks = self.dispatch_event(obs_event, context)
        all_feedbacks.extend(obs_feedbacks)

        return all_feedbacks

    # ── Chain Pattern ─────────────────────────────────────────────

    def chain(
        self,
        agents: list[ScientificAgent],
        context: AgentContext,
    ) -> list[OptimizationFeedback]:
        """Run agents sequentially, passing each result to the next.

        Each agent receives the previous agent's analysis result in
        ``context.metadata["previous_result"]``.

        Parameters
        ----------
        agents : list[ScientificAgent]
            Ordered list of agents to chain.
        context : AgentContext
            Shared context (will be mutated with chain results).

        Returns
        -------
        list[OptimizationFeedback]
            All non-None feedbacks from the chain.
        """
        feedbacks: list[OptimizationFeedback] = []
        previous_result: dict[str, Any] = {}

        for agent in agents:
            context.metadata["previous_result"] = previous_result
            if not agent.should_activate(context):
                continue
            if not agent.validate_context(context):
                continue

            try:
                result = agent.analyze(context)
                previous_result = result
                feedback = agent.get_optimization_feedback(result)
                if feedback is not None:
                    feedbacks.append(feedback)
            except Exception:
                feedbacks.append(
                    OptimizationFeedback(
                        agent_name=agent.name(),
                        feedback_type="warning",
                        confidence=0.0,
                        reasoning=f"Agent {agent.name()!r} raised an exception in chain.",
                    )
                )

        return feedbacks

    # ── Parallel Pattern ──────────────────────────────────────────

    def parallel(
        self,
        agents: list[ScientificAgent],
        context: AgentContext,
    ) -> list[OptimizationFeedback]:
        """Run agents independently (no result sharing).

        Unlike ``chain``, each agent gets the original context without
        modifications from other agents.  Runs sequentially in a
        single-threaded environment.

        Parameters
        ----------
        agents : list[ScientificAgent]
            Agents to run in parallel (logically independent).
        context : AgentContext
            Shared context (read-only semantics intended).

        Returns
        -------
        list[OptimizationFeedback]
            All non-None feedbacks from all agents.
        """
        feedbacks: list[OptimizationFeedback] = []

        for agent in agents:
            if not agent.should_activate(context):
                continue
            if not agent.validate_context(context):
                continue

            try:
                result = agent.analyze(context)
                feedback = agent.get_optimization_feedback(result)
                if feedback is not None:
                    feedbacks.append(feedback)
            except Exception:
                feedbacks.append(
                    OptimizationFeedback(
                        agent_name=agent.name(),
                        feedback_type="warning",
                        confidence=0.0,
                        reasoning=f"Agent {agent.name()!r} raised an exception in parallel.",
                    )
                )

        return feedbacks

    # ── Validation ────────────────────────────────────────────────

    def validate(
        self,
        feedbacks: list[OptimizationFeedback],
        context: AgentContext | None = None,
    ) -> list[OptimizationFeedback]:
        """Filter feedbacks by confidence threshold.

        Parameters
        ----------
        feedbacks : list[OptimizationFeedback]
            Raw feedbacks to filter.
        context : AgentContext | None
            Optional context (reserved for future domain-aware validation).

        Returns
        -------
        list[OptimizationFeedback]
            Only feedbacks with confidence >= threshold.
        """
        return [
            f for f in feedbacks
            if f.confidence >= self._confidence_threshold
        ]

    # ── Audit Trail ───────────────────────────────────────────────

    def get_audit_trail(self) -> list[AuditEntry]:
        """Return the full audit trail of all dispatch cycles."""
        return list(self._audit_trail)

    def clear_audit_trail(self) -> None:
        """Clear the audit trail."""
        self._audit_trail.clear()

    @property
    def n_dispatches(self) -> int:
        """Number of dispatch cycles executed."""
        return len(self._audit_trail)

    def __repr__(self) -> str:
        agent_names = [a.name() for a in self._agents]
        return (
            f"ScientificOrchestrator(agents={agent_names}, "
            f"dispatches={self.n_dispatches})"
        )


# ── Module-level helpers ──────────────────────────────────────────────


def _extract_anomaly_list(report: Any) -> list:
    """Extract a flat list of anomaly objects from an AnomalyReport.

    Collects signal_anomalies, kpi_anomalies, gp_anomalies, and
    change_points into a single list.
    """
    items: list = []
    for attr in ("signal_anomalies", "kpi_anomalies", "gp_anomalies", "change_points"):
        val = getattr(report, attr, None)
        if val:
            items.extend(val)
    return items

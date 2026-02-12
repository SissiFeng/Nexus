"""Tests for ScientificOrchestrator, OrchestratorEvent, AuditEntry, and ValidationHarness."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from optimization_copilot.agents.base import (
    AgentContext,
    AgentMode,
    OptimizationFeedback,
    ScientificAgent,
    TriggerCondition,
)
from optimization_copilot.agents.orchestrator import (
    AuditEntry,
    OrchestratorEvent,
    ScientificOrchestrator,
    _extract_anomaly_list,
)
from optimization_copilot.agents.validation_harness import (
    AgentValidationHarness,
    ValidationResult,
    _compute_precision_recall_f1,
)


# ---------------------------------------------------------------------------
# Test helpers: stub agents and mock anomaly types
# ---------------------------------------------------------------------------


class AlwaysAgent(ScientificAgent):
    """Agent that always activates and always produces feedback."""

    def __init__(
        self,
        agent_name: str = "always",
        feedback_type: str = "hypothesis",
        confidence: float = 0.8,
    ) -> None:
        super().__init__()
        self._name = agent_name
        self._feedback_type = feedback_type
        self._confidence = confidence

    def name(self) -> str:
        return self._name

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        return {"agent": self._name, "event_type": context.metadata.get("event_type")}

    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        return OptimizationFeedback(
            agent_name=self._name,
            feedback_type=self._feedback_type,
            confidence=self._confidence,
            reasoning=f"{self._name} completed analysis.",
        )


class NeverActivateAgent(ScientificAgent):
    """Agent that never activates."""

    def name(self) -> str:
        return "never_activate"

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        return {}

    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        return OptimizationFeedback(
            agent_name="never_activate",
            feedback_type="warning",
            confidence=1.0,
        )

    def should_activate(self, context: AgentContext) -> bool:
        return False


class InvalidContextAgent(ScientificAgent):
    """Agent that fails context validation."""

    def name(self) -> str:
        return "invalid_ctx"

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        return {}

    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        return OptimizationFeedback(
            agent_name="invalid_ctx",
            feedback_type="warning",
            confidence=1.0,
        )

    def validate_context(self, context: AgentContext) -> bool:
        return False


class CrashingAgent(ScientificAgent):
    """Agent that raises an exception during analyze."""

    def name(self) -> str:
        return "crashing"

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        raise RuntimeError("Intentional crash for testing")

    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        return None


class NoFeedbackAgent(ScientificAgent):
    """Agent that returns None feedback."""

    def name(self) -> str:
        return "no_feedback"

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        return {"status": "ok"}

    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        return None


class ObservationOnlyAgent(ScientificAgent):
    """Agent that only activates on observation events."""

    def name(self) -> str:
        return "obs_only"

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        return {"x": context.metadata.get("x"), "y": context.metadata.get("y")}

    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        return OptimizationFeedback(
            agent_name="obs_only",
            feedback_type="hypothesis",
            confidence=0.6,
        )

    def should_activate(self, context: AgentContext) -> bool:
        return context.metadata.get("event_type") == "observation"


# Mock anomaly types for testing on_observation with anomaly detection.

@dataclass
class MockAnomalyReport:
    """Minimal mock matching AnomalyReport interface."""

    is_anomalous: bool = False
    severity: str = "none"
    summary: str = "No anomalies"
    signal_anomalies: list = field(default_factory=list)
    kpi_anomalies: list = field(default_factory=list)
    gp_anomalies: list = field(default_factory=list)
    change_points: list = field(default_factory=list)


class MockAnomalyDetector:
    """Mock anomaly detector for orchestrator tests."""

    def __init__(self, report: MockAnomalyReport | None = None) -> None:
        self._report = report or MockAnomalyReport()
        self.call_count = 0

    def detect(
        self,
        x: list[float] | Any = None,
        y: float | Any = None,
        raw_data: dict | Any = None,
        kpi_values: dict | Any = None,
    ) -> MockAnomalyReport:
        self.call_count += 1
        return self._report


class CrashingDetector:
    """Detector that raises an exception."""

    def detect(self, **kwargs: Any) -> MockAnomalyReport:
        raise RuntimeError("Detector crash")


# ---------------------------------------------------------------------------
# OrchestratorEvent
# ---------------------------------------------------------------------------


class TestOrchestratorEvent:
    def test_creation(self):
        event = OrchestratorEvent(event_type="observation", data={"x": [1.0], "y": 2.0})
        assert event.event_type == "observation"
        assert event.data["x"] == [1.0]
        assert event.timestamp > 0

    def test_custom_timestamp(self):
        event = OrchestratorEvent(
            event_type="anomaly", data={}, timestamp=12345.0
        )
        assert event.timestamp == 12345.0

    def test_auto_timestamp(self):
        before = time.time()
        event = OrchestratorEvent(event_type="milestone")
        after = time.time()
        assert before <= event.timestamp <= after + 0.1

    def test_event_types(self):
        for etype in ("observation", "anomaly", "milestone", "stagnation", "drift"):
            event = OrchestratorEvent(event_type=etype)
            assert event.event_type == etype

    def test_data_defaults_to_empty(self):
        event = OrchestratorEvent(event_type="test")
        assert event.data == {}


# ---------------------------------------------------------------------------
# AuditEntry
# ---------------------------------------------------------------------------


class TestAuditEntry:
    def test_creation(self):
        event = OrchestratorEvent(event_type="observation")
        entry = AuditEntry(
            event=event,
            agents_triggered=["agent_a", "agent_b"],
            feedbacks=[],
            elapsed_ms=5.2,
        )
        assert entry.event.event_type == "observation"
        assert len(entry.agents_triggered) == 2
        assert entry.elapsed_ms == 5.2

    def test_defaults(self):
        event = OrchestratorEvent(event_type="test")
        entry = AuditEntry(event=event)
        assert entry.agents_triggered == []
        assert entry.feedbacks == []
        assert entry.elapsed_ms == 0.0


# ---------------------------------------------------------------------------
# ScientificOrchestrator - creation and agent management
# ---------------------------------------------------------------------------


class TestOrchestratorCreation:
    def test_empty_creation(self):
        orch = ScientificOrchestrator()
        assert orch.get_active_agents() == []
        assert orch.n_dispatches == 0

    def test_creation_with_agents(self):
        agents = [AlwaysAgent("a1"), AlwaysAgent("a2")]
        orch = ScientificOrchestrator(agents=agents)
        assert set(orch.get_active_agents()) == {"a1", "a2"}

    def test_register_agent(self):
        orch = ScientificOrchestrator()
        orch.register_agent(AlwaysAgent("new_agent"))
        assert "new_agent" in orch.get_active_agents()

    def test_register_duplicate_raises(self):
        orch = ScientificOrchestrator()
        orch.register_agent(AlwaysAgent("dup"))
        with pytest.raises(ValueError, match="already registered"):
            orch.register_agent(AlwaysAgent("dup"))

    def test_repr(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent("x")])
        r = repr(orch)
        assert "ScientificOrchestrator" in r
        assert "x" in r


# ---------------------------------------------------------------------------
# Dispatch events
# ---------------------------------------------------------------------------


class TestDispatchEvent:
    def test_no_agents(self):
        orch = ScientificOrchestrator()
        event = OrchestratorEvent(event_type="observation")
        ctx = AgentContext()
        feedbacks = orch.dispatch_event(event, ctx)
        assert feedbacks == []
        assert orch.n_dispatches == 1

    def test_with_one_agent(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent()])
        event = OrchestratorEvent(event_type="observation")
        ctx = AgentContext()
        feedbacks = orch.dispatch_event(event, ctx)
        assert len(feedbacks) == 1
        assert feedbacks[0].agent_name == "always"

    def test_with_multiple_agents(self):
        agents = [AlwaysAgent("a1"), AlwaysAgent("a2"), AlwaysAgent("a3")]
        orch = ScientificOrchestrator(agents=agents)
        event = OrchestratorEvent(event_type="observation")
        ctx = AgentContext()
        feedbacks = orch.dispatch_event(event, ctx)
        assert len(feedbacks) == 3
        names = {fb.agent_name for fb in feedbacks}
        assert names == {"a1", "a2", "a3"}

    def test_agent_not_activated(self):
        agents = [AlwaysAgent("active"), NeverActivateAgent()]
        orch = ScientificOrchestrator(agents=agents)
        event = OrchestratorEvent(event_type="observation")
        ctx = AgentContext()
        feedbacks = orch.dispatch_event(event, ctx)
        assert len(feedbacks) == 1
        assert feedbacks[0].agent_name == "active"

    def test_agent_invalid_context(self):
        agents = [AlwaysAgent("valid"), InvalidContextAgent()]
        orch = ScientificOrchestrator(agents=agents)
        event = OrchestratorEvent(event_type="observation")
        ctx = AgentContext()
        feedbacks = orch.dispatch_event(event, ctx)
        assert len(feedbacks) == 1
        assert feedbacks[0].agent_name == "valid"

    def test_crashing_agent_produces_warning(self):
        agents = [CrashingAgent()]
        orch = ScientificOrchestrator(agents=agents)
        event = OrchestratorEvent(event_type="observation")
        ctx = AgentContext()
        feedbacks = orch.dispatch_event(event, ctx)
        assert len(feedbacks) == 1
        assert feedbacks[0].feedback_type == "warning"
        assert feedbacks[0].confidence == 0.0
        assert "exception" in feedbacks[0].reasoning.lower()

    def test_no_feedback_agent(self):
        agents = [NoFeedbackAgent()]
        orch = ScientificOrchestrator(agents=agents)
        event = OrchestratorEvent(event_type="observation")
        ctx = AgentContext()
        feedbacks = orch.dispatch_event(event, ctx)
        assert feedbacks == []

    def test_event_data_in_context_metadata(self):
        class InspectAgent(ScientificAgent):
            def name(self) -> str:
                return "inspect"
            def analyze(self, context: AgentContext) -> dict[str, Any]:
                return {"event_type": context.metadata.get("event_type")}
            def get_optimization_feedback(self, result: dict) -> OptimizationFeedback | None:
                return OptimizationFeedback(
                    agent_name="inspect",
                    feedback_type="hypothesis",
                    confidence=0.5,
                    payload=result,
                )

        orch = ScientificOrchestrator(agents=[InspectAgent()])
        event = OrchestratorEvent(event_type="stagnation", data={"length": 15})
        ctx = AgentContext()
        feedbacks = orch.dispatch_event(event, ctx)
        assert len(feedbacks) == 1
        assert feedbacks[0].payload["event_type"] == "stagnation"

    def test_observation_only_agent(self):
        agents = [ObservationOnlyAgent(), AlwaysAgent("background")]
        orch = ScientificOrchestrator(agents=agents)

        # Observation event -> both activate
        obs_event = OrchestratorEvent(event_type="observation")
        ctx = AgentContext()
        feedbacks = orch.dispatch_event(obs_event, ctx)
        names = {fb.agent_name for fb in feedbacks}
        assert "obs_only" in names
        assert "background" in names

        # Non-observation event -> only background
        other_event = OrchestratorEvent(event_type="stagnation")
        ctx2 = AgentContext()
        feedbacks2 = orch.dispatch_event(other_event, ctx2)
        names2 = {fb.agent_name for fb in feedbacks2}
        assert "obs_only" not in names2
        assert "background" in names2


# ---------------------------------------------------------------------------
# on_observation
# ---------------------------------------------------------------------------


class TestOnObservation:
    def test_basic(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent()])
        feedbacks = orch.on_observation(x=[1.0, 2.0], y=5.0)
        assert len(feedbacks) == 1

    def test_with_context(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent()])
        ctx = AgentContext(iteration=3)
        feedbacks = orch.on_observation(x=[1.0], y=2.0, context=ctx)
        assert len(feedbacks) == 1
        assert ctx.metadata["x"] == [1.0]
        assert ctx.metadata["y"] == 2.0

    def test_with_raw_data(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent()])
        feedbacks = orch.on_observation(
            x=[1.0], y=2.0, raw_data={"voltages": [0.1, 0.2]}
        )
        assert len(feedbacks) == 1

    def test_with_kpi_values(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent()])
        feedbacks = orch.on_observation(
            x=[1.0], y=2.0, kpi_values={"yield": 0.8}
        )
        assert len(feedbacks) == 1

    def test_with_anomaly_detector_no_anomaly(self):
        detector = MockAnomalyDetector(MockAnomalyReport(is_anomalous=False))
        orch = ScientificOrchestrator(agents=[AlwaysAgent()])
        orch._anomaly_detector = detector

        feedbacks = orch.on_observation(x=[1.0], y=2.0)
        assert detector.call_count == 1
        # Only the observation event dispatch
        assert len(feedbacks) >= 1

    def test_with_anomaly_detector_anomaly_found(self):
        report = MockAnomalyReport(
            is_anomalous=True,
            severity="warning",
            summary="KPI out of range",
        )
        detector = MockAnomalyDetector(report)
        orch = ScientificOrchestrator(agents=[AlwaysAgent()])
        orch._anomaly_detector = detector

        feedbacks = orch.on_observation(x=[1.0], y=2.0)
        assert detector.call_count == 1
        # Anomaly event dispatch + observation event dispatch
        assert len(feedbacks) == 2

    def test_anomaly_detector_crash_does_not_block(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent()])
        orch._anomaly_detector = CrashingDetector()

        # Should not raise; observation dispatch still works
        feedbacks = orch.on_observation(x=[1.0], y=2.0)
        assert len(feedbacks) >= 1

    def test_no_agents(self):
        orch = ScientificOrchestrator()
        feedbacks = orch.on_observation(x=[1.0], y=2.0)
        assert feedbacks == []


# ---------------------------------------------------------------------------
# Chain pattern
# ---------------------------------------------------------------------------


class TestChainPattern:
    def test_basic_chain(self):
        a1 = AlwaysAgent("chain_a")
        a2 = AlwaysAgent("chain_b")
        orch = ScientificOrchestrator()
        ctx = AgentContext()
        feedbacks = orch.chain([a1, a2], ctx)
        assert len(feedbacks) == 2
        assert feedbacks[0].agent_name == "chain_a"
        assert feedbacks[1].agent_name == "chain_b"

    def test_chain_passes_previous_result(self):
        class RecordingAgent(ScientificAgent):
            def __init__(self, agent_name: str) -> None:
                super().__init__()
                self._name = agent_name
                self.received_previous: dict = {}

            def name(self) -> str:
                return self._name

            def analyze(self, context: AgentContext) -> dict[str, Any]:
                self.received_previous = context.metadata.get("previous_result", {})
                return {"from": self._name}

            def get_optimization_feedback(self, result: dict) -> OptimizationFeedback | None:
                return OptimizationFeedback(
                    agent_name=self._name,
                    feedback_type="hypothesis",
                    confidence=0.5,
                )

        a1 = RecordingAgent("first")
        a2 = RecordingAgent("second")
        orch = ScientificOrchestrator()
        ctx = AgentContext()
        orch.chain([a1, a2], ctx)

        assert a1.received_previous == {}
        assert a2.received_previous == {"from": "first"}

    def test_chain_with_inactive_agent(self):
        orch = ScientificOrchestrator()
        ctx = AgentContext()
        feedbacks = orch.chain(
            [AlwaysAgent("active"), NeverActivateAgent()], ctx
        )
        assert len(feedbacks) == 1

    def test_chain_with_crashing_agent(self):
        orch = ScientificOrchestrator()
        ctx = AgentContext()
        feedbacks = orch.chain(
            [CrashingAgent(), AlwaysAgent("after_crash")], ctx
        )
        assert len(feedbacks) == 2
        assert feedbacks[0].feedback_type == "warning"
        assert feedbacks[1].agent_name == "after_crash"

    def test_chain_empty(self):
        orch = ScientificOrchestrator()
        feedbacks = orch.chain([], AgentContext())
        assert feedbacks == []


# ---------------------------------------------------------------------------
# Parallel pattern
# ---------------------------------------------------------------------------


class TestParallelPattern:
    def test_basic_parallel(self):
        a1 = AlwaysAgent("par_a")
        a2 = AlwaysAgent("par_b")
        orch = ScientificOrchestrator()
        ctx = AgentContext()
        feedbacks = orch.parallel([a1, a2], ctx)
        assert len(feedbacks) == 2
        names = {fb.agent_name for fb in feedbacks}
        assert names == {"par_a", "par_b"}

    def test_parallel_with_inactive(self):
        orch = ScientificOrchestrator()
        ctx = AgentContext()
        feedbacks = orch.parallel(
            [AlwaysAgent("active"), NeverActivateAgent()], ctx
        )
        assert len(feedbacks) == 1

    def test_parallel_with_crashing(self):
        orch = ScientificOrchestrator()
        ctx = AgentContext()
        feedbacks = orch.parallel(
            [AlwaysAgent("ok"), CrashingAgent()], ctx
        )
        assert len(feedbacks) == 2
        types = {fb.feedback_type for fb in feedbacks}
        assert "warning" in types

    def test_parallel_empty(self):
        orch = ScientificOrchestrator()
        feedbacks = orch.parallel([], AgentContext())
        assert feedbacks == []

    def test_parallel_no_feedback_agents(self):
        orch = ScientificOrchestrator()
        feedbacks = orch.parallel(
            [NoFeedbackAgent(), NoFeedbackAgent()], AgentContext()
        )
        assert feedbacks == []


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_high_confidence_passes(self):
        orch = ScientificOrchestrator(confidence_threshold=0.5)
        feedbacks = [
            OptimizationFeedback(
                agent_name="a", feedback_type="prior_update", confidence=0.8
            )
        ]
        result = orch.validate(feedbacks)
        assert len(result) == 1

    def test_low_confidence_filtered(self):
        orch = ScientificOrchestrator(confidence_threshold=0.5)
        feedbacks = [
            OptimizationFeedback(
                agent_name="a", feedback_type="prior_update", confidence=0.3
            )
        ]
        result = orch.validate(feedbacks)
        assert len(result) == 0

    def test_at_threshold(self):
        orch = ScientificOrchestrator(confidence_threshold=0.5)
        feedbacks = [
            OptimizationFeedback(
                agent_name="a", feedback_type="prior_update", confidence=0.5
            )
        ]
        result = orch.validate(feedbacks)
        assert len(result) == 1

    def test_custom_threshold(self):
        orch = ScientificOrchestrator(confidence_threshold=0.9)
        feedbacks = [
            OptimizationFeedback(
                agent_name="a", feedback_type="prior_update", confidence=0.85
            ),
            OptimizationFeedback(
                agent_name="b", feedback_type="reweight", confidence=0.95
            ),
        ]
        result = orch.validate(feedbacks)
        assert len(result) == 1
        assert result[0].agent_name == "b"

    def test_empty_feedbacks(self):
        orch = ScientificOrchestrator()
        assert orch.validate([]) == []

    def test_mixed_confidence(self):
        orch = ScientificOrchestrator(confidence_threshold=0.5)
        feedbacks = [
            OptimizationFeedback(agent_name="a", feedback_type="x", confidence=0.9),
            OptimizationFeedback(agent_name="b", feedback_type="x", confidence=0.1),
            OptimizationFeedback(agent_name="c", feedback_type="x", confidence=0.6),
        ]
        result = orch.validate(feedbacks)
        assert len(result) == 2
        names = {fb.agent_name for fb in result}
        assert names == {"a", "c"}


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


class TestAuditTrail:
    def test_empty_trail(self):
        orch = ScientificOrchestrator()
        assert orch.get_audit_trail() == []

    def test_trail_after_dispatch(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent()])
        event = OrchestratorEvent(event_type="observation")
        orch.dispatch_event(event, AgentContext())
        trail = orch.get_audit_trail()
        assert len(trail) == 1
        assert trail[0].event.event_type == "observation"
        assert "always" in trail[0].agents_triggered
        assert trail[0].elapsed_ms >= 0

    def test_trail_multiple_dispatches(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent()])
        for etype in ("observation", "anomaly", "stagnation"):
            orch.dispatch_event(
                OrchestratorEvent(event_type=etype), AgentContext()
            )
        trail = orch.get_audit_trail()
        assert len(trail) == 3
        types = [e.event.event_type for e in trail]
        assert types == ["observation", "anomaly", "stagnation"]

    def test_clear_audit_trail(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent()])
        orch.dispatch_event(
            OrchestratorEvent(event_type="test"), AgentContext()
        )
        assert orch.n_dispatches == 1
        orch.clear_audit_trail()
        assert orch.n_dispatches == 0
        assert orch.get_audit_trail() == []

    def test_trail_is_copy(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent()])
        orch.dispatch_event(
            OrchestratorEvent(event_type="test"), AgentContext()
        )
        trail1 = orch.get_audit_trail()
        trail2 = orch.get_audit_trail()
        assert trail1 is not trail2
        assert len(trail1) == len(trail2)

    def test_trail_records_feedbacks(self):
        orch = ScientificOrchestrator(
            agents=[AlwaysAgent("a"), AlwaysAgent("b")]
        )
        orch.dispatch_event(
            OrchestratorEvent(event_type="observation"), AgentContext()
        )
        trail = orch.get_audit_trail()
        assert len(trail[0].feedbacks) == 2


# ---------------------------------------------------------------------------
# _extract_anomaly_list helper
# ---------------------------------------------------------------------------


class TestExtractAnomalyList:
    def test_empty_report(self):
        report = MockAnomalyReport()
        result = _extract_anomaly_list(report)
        assert result == []

    def test_with_signal_anomalies(self):
        report = MockAnomalyReport(signal_anomalies=["sig1", "sig2"])
        result = _extract_anomaly_list(report)
        assert len(result) == 2

    def test_with_all_types(self):
        report = MockAnomalyReport(
            signal_anomalies=["s1"],
            kpi_anomalies=["k1"],
            gp_anomalies=["g1"],
            change_points=["c1"],
        )
        result = _extract_anomaly_list(report)
        assert len(result) == 4

    def test_missing_attributes(self):
        # Object without the expected attributes
        class Minimal:
            pass

        result = _extract_anomaly_list(Minimal())
        assert result == []


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_creation_defaults(self):
        vr = ValidationResult()
        assert vr.precision == 0.0
        assert vr.recall == 0.0
        assert vr.f1_score == 0.0
        assert vr.n_correct == 0
        assert vr.n_total == 0
        assert vr.details == []

    def test_accuracy(self):
        vr = ValidationResult(n_correct=7, n_total=10)
        assert abs(vr.accuracy - 0.7) < 1e-9

    def test_accuracy_zero_total(self):
        vr = ValidationResult(n_correct=0, n_total=0)
        assert vr.accuracy == 0.0

    def test_to_dict(self):
        vr = ValidationResult(
            precision=0.8,
            recall=0.9,
            f1_score=0.847,
            n_correct=8,
            n_total=10,
            details=[{"case": 1}],
        )
        d = vr.to_dict()
        assert d["precision"] == 0.8
        assert d["n_total"] == 10
        assert len(d["details"]) == 1

    def test_perfect_scores(self):
        vr = ValidationResult(
            precision=1.0, recall=1.0, f1_score=1.0, n_correct=10, n_total=10
        )
        assert vr.accuracy == 1.0


# ---------------------------------------------------------------------------
# _compute_precision_recall_f1
# ---------------------------------------------------------------------------


class TestComputePRF1:
    def test_all_correct(self):
        p, r, f1 = _compute_precision_recall_f1(10, 0, 0)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_no_positives(self):
        p, r, f1 = _compute_precision_recall_f1(0, 0, 0)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_only_false_positives(self):
        p, r, f1 = _compute_precision_recall_f1(0, 5, 0)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_only_false_negatives(self):
        p, r, f1 = _compute_precision_recall_f1(0, 0, 5)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_mixed(self):
        p, r, f1 = _compute_precision_recall_f1(8, 2, 3)
        assert abs(p - 0.8) < 1e-9
        assert abs(r - 8 / 11) < 1e-9
        expected_f1 = 2 * 0.8 * (8 / 11) / (0.8 + 8 / 11)
        assert abs(f1 - expected_f1) < 1e-9


# ---------------------------------------------------------------------------
# AgentValidationHarness - anomaly detection
# ---------------------------------------------------------------------------


class TestValidationHarnessAnomaly:
    def test_perfect_detector(self):
        detector = MockAnomalyDetector(
            MockAnomalyReport(is_anomalous=True)
        )
        test_data = [
            {"label": "a", "x": [1.0], "y": 2.0, "raw_data": {}, "kpi_values": {}},
            {"label": "b", "x": [2.0], "y": 3.0, "raw_data": {}, "kpi_values": {}},
        ]
        annotations = {"a": True, "b": True}
        harness = AgentValidationHarness()
        result = harness.evaluate_anomaly_detection(detector, test_data, annotations)
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_all_false_positives(self):
        detector = MockAnomalyDetector(
            MockAnomalyReport(is_anomalous=True)
        )
        test_data = [
            {"label": "a", "x": [1.0], "y": 2.0, "raw_data": {}, "kpi_values": {}},
        ]
        annotations = {"a": False}
        harness = AgentValidationHarness()
        result = harness.evaluate_anomaly_detection(detector, test_data, annotations)
        assert result.precision == 0.0

    def test_all_false_negatives(self):
        detector = MockAnomalyDetector(
            MockAnomalyReport(is_anomalous=False)
        )
        test_data = [
            {"label": "a", "x": [1.0], "y": 2.0, "raw_data": {}, "kpi_values": {}},
        ]
        annotations = {"a": True}
        harness = AgentValidationHarness()
        result = harness.evaluate_anomaly_detection(detector, test_data, annotations)
        assert result.recall == 0.0

    def test_empty_test_data(self):
        detector = MockAnomalyDetector()
        harness = AgentValidationHarness()
        result = harness.evaluate_anomaly_detection(detector, [], {})
        assert result.n_total == 0


# ---------------------------------------------------------------------------
# AgentValidationHarness - agent evaluation
# ---------------------------------------------------------------------------


class TestValidationHarnessAgent:
    def test_agent_produces_expected_feedback(self):
        agent = AlwaysAgent(feedback_type="hypothesis", confidence=0.8)
        contexts = [AgentContext()]
        expected = [{"feedback_type": "hypothesis", "min_confidence": 0.5}]
        harness = AgentValidationHarness()
        result = harness.evaluate_agent(agent, contexts, expected)
        assert result.n_correct >= 1

    def test_agent_wrong_type(self):
        agent = AlwaysAgent(feedback_type="hypothesis", confidence=0.8)
        contexts = [AgentContext()]
        expected = [{"feedback_type": "prior_update"}]
        harness = AgentValidationHarness()
        result = harness.evaluate_agent(agent, contexts, expected)
        # Feedback type doesn't match -> not a correct positive
        assert result.n_total == 1

    def test_agent_no_feedback_expected_none(self):
        agent = NoFeedbackAgent()
        contexts = [AgentContext()]
        expected = [{"should_produce_feedback": False}]
        harness = AgentValidationHarness()
        result = harness.evaluate_agent(agent, contexts, expected)
        assert result.n_correct == 1

    def test_agent_no_feedback_but_expected(self):
        agent = NoFeedbackAgent()
        contexts = [AgentContext()]
        expected = [{"should_produce_feedback": True, "feedback_type": "hypothesis"}]
        harness = AgentValidationHarness()
        result = harness.evaluate_agent(agent, contexts, expected)
        assert result.n_correct == 0

    def test_empty_contexts(self):
        agent = AlwaysAgent()
        harness = AgentValidationHarness()
        result = harness.evaluate_agent(agent, [], [])
        assert result.n_total == 0


# ---------------------------------------------------------------------------
# AgentValidationHarness - orchestrator evaluation
# ---------------------------------------------------------------------------


class TestValidationHarnessOrchestrator:
    def test_correct_activations(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent("agent_a")])
        events = [OrchestratorEvent(event_type="observation")]
        contexts = [AgentContext()]
        expected = {"observation": ["agent_a"]}
        harness = AgentValidationHarness()
        result = harness.evaluate_orchestrator(orch, events, contexts, expected)
        assert result.n_correct == 1
        assert result.precision == 1.0

    def test_unexpected_activation(self):
        orch = ScientificOrchestrator(
            agents=[AlwaysAgent("a"), AlwaysAgent("b")]
        )
        events = [OrchestratorEvent(event_type="observation")]
        contexts = [AgentContext()]
        expected = {"observation": ["a"]}  # "b" is unexpected
        harness = AgentValidationHarness()
        result = harness.evaluate_orchestrator(orch, events, contexts, expected)
        assert result.precision < 1.0

    def test_missing_activation(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent("a")])
        events = [OrchestratorEvent(event_type="observation")]
        contexts = [AgentContext()]
        expected = {"observation": ["a", "b"]}  # "b" not registered
        harness = AgentValidationHarness()
        result = harness.evaluate_orchestrator(orch, events, contexts, expected)
        assert result.recall < 1.0

    def test_no_expected_agents(self):
        orch = ScientificOrchestrator(agents=[AlwaysAgent("a")])
        events = [OrchestratorEvent(event_type="anomaly")]
        contexts = [AgentContext()]
        expected = {"anomaly": []}
        harness = AgentValidationHarness()
        result = harness.evaluate_orchestrator(orch, events, contexts, expected)
        # "a" activated but none expected -> false positives
        assert result.n_total == 1

    def test_empty_events(self):
        orch = ScientificOrchestrator()
        harness = AgentValidationHarness()
        result = harness.evaluate_orchestrator(orch, [], [], {})
        assert result.n_total == 0

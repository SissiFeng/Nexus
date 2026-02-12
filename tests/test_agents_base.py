"""Tests for the ScientificAgent ABC, AgentContext, and OptimizationFeedback."""

from __future__ import annotations

import pytest
from typing import Any

from optimization_copilot.agents.base import (
    AgentContext,
    AgentMode,
    OptimizationFeedback,
    ScientificAgent,
    TriggerCondition,
)
from optimization_copilot.uncertainty.types import MeasurementWithUncertainty
from optimization_copilot.diagnostics.engine import DiagnosticsVector
from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)


# ---------------------------------------------------------------------------
# Concrete agent for testing
# ---------------------------------------------------------------------------


class StubAgent(ScientificAgent):
    """Minimal concrete agent for testing the ABC contract."""

    def __init__(
        self,
        agent_name: str = "stub",
        mode: AgentMode = AgentMode.PRAGMATIC,
        activate: bool = True,
        context_valid: bool = True,
        feedback_type: str = "hypothesis",
        confidence: float = 0.8,
        return_feedback: bool = True,
    ) -> None:
        super().__init__(mode=mode)
        self._agent_name = agent_name
        self._activate = activate
        self._context_valid = context_valid
        self._feedback_type = feedback_type
        self._confidence = confidence
        self._return_feedback = return_feedback

    def name(self) -> str:
        return self._agent_name

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        return {
            "status": "ok",
            "iteration": context.iteration,
            "has_gp": context.has_gp_model(),
        }

    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        if not self._return_feedback:
            return None
        return OptimizationFeedback(
            agent_name=self._agent_name,
            feedback_type=self._feedback_type,
            confidence=self._confidence,
            payload={"result": analysis_result.get("status", "unknown")},
            reasoning="Stub agent analysis complete.",
        )

    def should_activate(self, context: AgentContext) -> bool:
        return self._activate

    def validate_context(self, context: AgentContext) -> bool:
        return self._context_valid


class ConditionalAgent(ScientificAgent):
    """Agent that only activates when iteration > 0."""

    def name(self) -> str:
        return "conditional"

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        return {"iteration": context.iteration}

    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        return OptimizationFeedback(
            agent_name="conditional",
            feedback_type="hypothesis",
            confidence=0.7,
        )

    def should_activate(self, context: AgentContext) -> bool:
        return context.iteration > 0


# ---------------------------------------------------------------------------
# AgentMode enum
# ---------------------------------------------------------------------------


class TestAgentMode:
    def test_pragmatic_value(self):
        assert AgentMode.PRAGMATIC.value == "pragmatic"

    def test_llm_enhanced_value(self):
        assert AgentMode.LLM_ENHANCED.value == "llm_enhanced"

    def test_is_str_enum(self):
        assert isinstance(AgentMode.PRAGMATIC, str)
        assert AgentMode.PRAGMATIC == "pragmatic"

    def test_from_string(self):
        assert AgentMode("pragmatic") is AgentMode.PRAGMATIC
        assert AgentMode("llm_enhanced") is AgentMode.LLM_ENHANCED


# ---------------------------------------------------------------------------
# TriggerCondition
# ---------------------------------------------------------------------------


class TestTriggerCondition:
    def test_creation(self):
        tc = TriggerCondition(
            name="stagnation",
            check_fn_name="check_stagnation",
            priority=5,
            description="Fires when plateau > 10",
        )
        assert tc.name == "stagnation"
        assert tc.check_fn_name == "check_stagnation"
        assert tc.priority == 5
        assert tc.description == "Fires when plateau > 10"

    def test_defaults(self):
        tc = TriggerCondition(name="basic", check_fn_name="check_basic")
        assert tc.priority == 0
        assert tc.description == ""

    def test_priority_comparison(self):
        high = TriggerCondition(name="high", check_fn_name="h", priority=10)
        low = TriggerCondition(name="low", check_fn_name="l", priority=1)
        assert high.priority > low.priority


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------


class TestAgentContext:
    def test_creation_defaults(self):
        ctx = AgentContext()
        assert ctx.gp_model is None
        assert ctx.optimization_history == []
        assert ctx.raw_data is None
        assert ctx.anomalies is None
        assert ctx.domain_config is None
        assert ctx.measurements is None
        assert ctx.diagnostics is None
        assert ctx.campaign_snapshot is None
        assert ctx.iteration == 0
        assert ctx.metadata == {}

    def test_creation_with_values(self):
        ctx = AgentContext(
            gp_model="mock_gp",
            iteration=5,
            metadata={"key": "value"},
        )
        assert ctx.gp_model == "mock_gp"
        assert ctx.iteration == 5
        assert ctx.metadata["key"] == "value"

    def test_get_parameter_names_from_history(self):
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"temp": 100, "pressure": 1.0}, "y": 0.5},
                {"parameters": {"temp": 200, "pressure": 2.0}, "y": 0.8},
            ]
        )
        names = ctx.get_parameter_names()
        assert "temp" in names
        assert "pressure" in names

    def test_get_parameter_names_empty(self):
        ctx = AgentContext()
        assert ctx.get_parameter_names() == []

    def test_get_parameter_names_no_parameters_key(self):
        ctx = AgentContext(
            optimization_history=[{"y": 0.5}]
        )
        assert ctx.get_parameter_names() == []

    def test_has_gp_model_false(self):
        ctx = AgentContext()
        assert ctx.has_gp_model() is False

    def test_has_gp_model_true(self):
        ctx = AgentContext(gp_model="mock")
        assert ctx.has_gp_model() is True

    def test_has_measurements_false_none(self):
        ctx = AgentContext()
        assert ctx.has_measurements() is False

    def test_has_measurements_false_empty(self):
        ctx = AgentContext(measurements=[])
        assert ctx.has_measurements() is False

    def test_has_measurements_true(self):
        m = MeasurementWithUncertainty(
            value=1.0, variance=0.01, confidence=0.9, source="test"
        )
        ctx = AgentContext(measurements=[m])
        assert ctx.has_measurements() is True

    def test_has_diagnostics(self):
        ctx = AgentContext()
        assert ctx.has_diagnostics() is False
        ctx.diagnostics = DiagnosticsVector()
        assert ctx.has_diagnostics() is True

    def test_has_campaign_snapshot(self):
        ctx = AgentContext()
        assert ctx.has_campaign_snapshot() is False
        ctx.campaign_snapshot = "mock_snapshot"
        assert ctx.has_campaign_snapshot() is True

    def test_metadata_mutation(self):
        ctx = AgentContext()
        ctx.metadata["key1"] = "val1"
        ctx.metadata["key2"] = 42
        assert len(ctx.metadata) == 2

    def test_context_with_measurements(self):
        m1 = MeasurementWithUncertainty(
            value=1.0, variance=0.01, confidence=0.9, source="eis"
        )
        m2 = MeasurementWithUncertainty(
            value=2.0, variance=0.05, confidence=0.7, source="xrd"
        )
        ctx = AgentContext(measurements=[m1, m2])
        assert len(ctx.measurements) == 2
        assert ctx.measurements[0].source == "eis"
        assert ctx.measurements[1].value == 2.0

    def test_context_with_raw_data(self):
        ctx = AgentContext(raw_data={"voltages": [1.0, 2.0, 3.0]})
        assert ctx.raw_data is not None
        assert len(ctx.raw_data["voltages"]) == 3

    def test_context_with_anomalies(self):
        ctx = AgentContext(anomalies=["anomaly1", "anomaly2"])
        assert ctx.anomalies is not None
        assert len(ctx.anomalies) == 2


# ---------------------------------------------------------------------------
# OptimizationFeedback
# ---------------------------------------------------------------------------


class TestOptimizationFeedback:
    def test_creation(self):
        fb = OptimizationFeedback(
            agent_name="test_agent",
            feedback_type="prior_update",
            confidence=0.9,
            payload={"mean": 5.0},
            reasoning="Strong evidence for prior shift.",
        )
        assert fb.agent_name == "test_agent"
        assert fb.feedback_type == "prior_update"
        assert fb.confidence == 0.9
        assert fb.payload["mean"] == 5.0
        assert "Strong evidence" in fb.reasoning

    def test_defaults(self):
        fb = OptimizationFeedback(
            agent_name="a", feedback_type="warning", confidence=0.1
        )
        assert fb.payload == {}
        assert fb.reasoning == ""

    def test_is_actionable_high_confidence_prior_update(self):
        fb = OptimizationFeedback(
            agent_name="a", feedback_type="prior_update", confidence=0.8
        )
        assert fb.is_actionable() is True

    def test_is_actionable_constraint_addition(self):
        fb = OptimizationFeedback(
            agent_name="a", feedback_type="constraint_addition", confidence=0.6
        )
        assert fb.is_actionable() is True

    def test_is_actionable_reweight(self):
        fb = OptimizationFeedback(
            agent_name="a", feedback_type="reweight", confidence=0.5
        )
        assert fb.is_actionable() is True

    def test_is_actionable_low_confidence(self):
        fb = OptimizationFeedback(
            agent_name="a", feedback_type="prior_update", confidence=0.4
        )
        assert fb.is_actionable() is False

    def test_is_actionable_exactly_threshold(self):
        fb = OptimizationFeedback(
            agent_name="a", feedback_type="prior_update", confidence=0.5
        )
        assert fb.is_actionable() is True

    def test_non_actionable_type_hypothesis(self):
        fb = OptimizationFeedback(
            agent_name="a", feedback_type="hypothesis", confidence=0.9
        )
        assert fb.is_actionable() is False

    def test_non_actionable_type_warning(self):
        fb = OptimizationFeedback(
            agent_name="a", feedback_type="warning", confidence=0.9
        )
        assert fb.is_actionable() is False

    def test_to_dict(self):
        fb = OptimizationFeedback(
            agent_name="a",
            feedback_type="prior_update",
            confidence=0.7,
            payload={"x": 1},
            reasoning="test",
        )
        d = fb.to_dict()
        assert d["agent_name"] == "a"
        assert d["feedback_type"] == "prior_update"
        assert d["confidence"] == 0.7
        assert d["payload"] == {"x": 1}
        assert d["reasoning"] == "test"

    def test_from_dict(self):
        d = {
            "agent_name": "b",
            "feedback_type": "reweight",
            "confidence": 0.6,
            "payload": {"w": 2.0},
            "reasoning": "reweight needed",
        }
        fb = OptimizationFeedback.from_dict(d)
        assert fb.agent_name == "b"
        assert fb.feedback_type == "reweight"
        assert fb.confidence == 0.6
        assert fb.payload["w"] == 2.0

    def test_from_dict_missing_optional(self):
        d = {
            "agent_name": "c",
            "feedback_type": "warning",
            "confidence": 0.3,
        }
        fb = OptimizationFeedback.from_dict(d)
        assert fb.payload == {}
        assert fb.reasoning == ""

    def test_round_trip(self):
        original = OptimizationFeedback(
            agent_name="round",
            feedback_type="prior_update",
            confidence=0.85,
            payload={"mu": 3.14},
            reasoning="pi prior",
        )
        restored = OptimizationFeedback.from_dict(original.to_dict())
        assert restored.agent_name == original.agent_name
        assert restored.feedback_type == original.feedback_type
        assert restored.confidence == original.confidence
        assert restored.payload == original.payload
        assert restored.reasoning == original.reasoning


# ---------------------------------------------------------------------------
# ScientificAgent ABC
# ---------------------------------------------------------------------------


class TestScientificAgentABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ScientificAgent()  # type: ignore[abstract]

    def test_cannot_instantiate_partial_impl(self):
        class Partial(ScientificAgent):
            def name(self) -> str:
                return "partial"

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Concrete agent tests
# ---------------------------------------------------------------------------


class TestConcreteAgent:
    def test_name(self):
        agent = StubAgent(agent_name="my_agent")
        assert agent.name() == "my_agent"

    def test_default_mode(self):
        agent = StubAgent()
        assert agent.mode == AgentMode.PRAGMATIC

    def test_llm_mode(self):
        agent = StubAgent(mode=AgentMode.LLM_ENHANCED)
        assert agent.mode == AgentMode.LLM_ENHANCED

    def test_analyze_returns_dict(self):
        agent = StubAgent()
        ctx = AgentContext(iteration=3)
        result = agent.analyze(ctx)
        assert isinstance(result, dict)
        assert result["status"] == "ok"
        assert result["iteration"] == 3

    def test_analyze_with_gp(self):
        agent = StubAgent()
        ctx = AgentContext(gp_model="mock_gp")
        result = agent.analyze(ctx)
        assert result["has_gp"] is True

    def test_analyze_without_gp(self):
        agent = StubAgent()
        ctx = AgentContext()
        result = agent.analyze(ctx)
        assert result["has_gp"] is False

    def test_feedback_returned(self):
        agent = StubAgent()
        result = {"status": "ok"}
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert feedback.agent_name == "stub"
        assert feedback.confidence == 0.8

    def test_feedback_none_when_disabled(self):
        agent = StubAgent(return_feedback=False)
        result = {"status": "ok"}
        feedback = agent.get_optimization_feedback(result)
        assert feedback is None

    def test_should_activate_true(self):
        agent = StubAgent(activate=True)
        ctx = AgentContext()
        assert agent.should_activate(ctx) is True

    def test_should_activate_false(self):
        agent = StubAgent(activate=False)
        ctx = AgentContext()
        assert agent.should_activate(ctx) is False

    def test_validate_context_true(self):
        agent = StubAgent(context_valid=True)
        ctx = AgentContext()
        assert agent.validate_context(ctx) is True

    def test_validate_context_false(self):
        agent = StubAgent(context_valid=False)
        ctx = AgentContext()
        assert agent.validate_context(ctx) is False

    def test_trigger_conditions_empty_default(self):
        agent = StubAgent()
        assert agent.trigger_conditions == []

    def test_trigger_conditions_can_be_set(self):
        agent = StubAgent()
        tc = TriggerCondition(name="test", check_fn_name="check_test")
        agent._trigger_conditions.append(tc)
        assert len(agent.trigger_conditions) == 1
        assert agent.trigger_conditions[0].name == "test"

    def test_repr(self):
        agent = StubAgent(agent_name="repr_test")
        r = repr(agent)
        assert "StubAgent" in r
        assert "repr_test" in r
        assert "pragmatic" in r


class TestConditionalAgent:
    def test_activates_when_iteration_positive(self):
        agent = ConditionalAgent()
        ctx = AgentContext(iteration=1)
        assert agent.should_activate(ctx) is True

    def test_does_not_activate_at_iteration_zero(self):
        agent = ConditionalAgent()
        ctx = AgentContext(iteration=0)
        assert agent.should_activate(ctx) is False

    def test_name(self):
        agent = ConditionalAgent()
        assert agent.name() == "conditional"

    def test_analyze(self):
        agent = ConditionalAgent()
        ctx = AgentContext(iteration=5)
        result = agent.analyze(ctx)
        assert result["iteration"] == 5

    def test_feedback(self):
        agent = ConditionalAgent()
        result = {"iteration": 5}
        fb = agent.get_optimization_feedback(result)
        assert fb is not None
        assert fb.confidence == 0.7

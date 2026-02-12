"""Tests for agents/traced_agent.py — TracedScientificAgent base class."""

from __future__ import annotations

from typing import Any

import pytest

from optimization_copilot.agents.base import (
    AgentContext,
    AgentMode,
    OptimizationFeedback,
)
from optimization_copilot.agents.execution_trace import (
    ExecutionTag,
    ExecutionTrace,
    TracedResult,
    trace_call,
)
from optimization_copilot.agents.traced_agent import TracedScientificAgent


# ---------------------------------------------------------------------------
# Concrete test agent
# ---------------------------------------------------------------------------


class _TopKAgent(TracedScientificAgent):
    """Test agent that uses pipeline.run_top_k."""

    def name(self) -> str:
        return "top_k_test_agent"

    def analyze_traced(self, context: AgentContext) -> dict[str, Any]:
        values = context.metadata.get("values", [])
        names = context.metadata.get("names", [])
        k = context.metadata.get("k", 3)
        return {"top_k_result": self.pipeline.run_top_k(values, names, k)}

    def _build_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        top_k = analysis_result.get("top_k_result")
        if top_k is None:
            return None
        return OptimizationFeedback(
            agent_name=self.name(),
            feedback_type="hypothesis",
            confidence=0.9,
            payload={"top_k": top_k},
            reasoning="Top-K analysis computed",
        )


class _FailingAgent(TracedScientificAgent):
    """Test agent that produces a failed TracedResult."""

    def name(self) -> str:
        return "failing_test_agent"

    def analyze_traced(self, context: AgentContext) -> dict[str, Any]:
        # Deliberately create a failed trace
        failed = trace_call(
            module="test", method="bad_fn",
            fn=lambda: (_ for _ in ()).throw(ValueError("boom")),
        )
        return {"bad_result": failed}

    def _build_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        return OptimizationFeedback(
            agent_name=self.name(),
            feedback_type="warning",
            confidence=0.3,
            payload={"error": "something failed"},
            reasoning="Analysis failed",
        )


class _NoFeedbackAgent(TracedScientificAgent):
    """Test agent that returns None from _build_feedback."""

    def name(self) -> str:
        return "no_feedback_agent"

    def analyze_traced(self, context: AgentContext) -> dict[str, Any]:
        return {"plain_value": 42}

    def _build_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        return None


class _MixedAgent(TracedScientificAgent):
    """Test agent with both TracedResult and plain values."""

    def name(self) -> str:
        return "mixed_agent"

    def analyze_traced(self, context: AgentContext) -> dict[str, Any]:
        traced = self.pipeline.run_top_k([1.0, 2.0, 3.0], ["a", "b", "c"], k=2)
        return {
            "computed": traced,
            "config": {"seed": 42},
        }

    def _build_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        return OptimizationFeedback(
            agent_name=self.name(),
            feedback_type="hypothesis",
            confidence=0.8,
            payload={
                "top_items": analysis_result.get("computed"),
                "config": analysis_result.get("config"),
            },
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTracedScientificAgent:
    def _make_context(self, **metadata):
        return AgentContext(metadata=metadata)

    def test_is_subclass_of_scientific_agent(self):
        from optimization_copilot.agents.base import ScientificAgent
        assert issubclass(TracedScientificAgent, ScientificAgent)

    def test_pipeline_is_available(self):
        agent = _TopKAgent()
        assert agent.pipeline is not None
        assert hasattr(agent.pipeline, "run_top_k")

    def test_mode_defaults_to_pragmatic(self):
        agent = _TopKAgent()
        assert agent.mode == AgentMode.PRAGMATIC

    def test_traces_empty_before_analyze(self):
        agent = _TopKAgent()
        assert agent.collected_traces == []


class TestAnalyze:
    def _make_context(self, **metadata):
        return AgentContext(metadata=metadata)

    def test_analyze_collects_traces(self):
        agent = _TopKAgent()
        ctx = self._make_context(values=[0.5, 0.9, 0.3], names=["A", "B", "C"], k=2)
        result = agent.analyze(ctx)

        # Traces should be collected
        assert len(agent.collected_traces) >= 1
        assert agent.collected_traces[0].tag == ExecutionTag.COMPUTED

    def test_analyze_unwraps_traced_result(self):
        agent = _TopKAgent()
        ctx = self._make_context(values=[0.5, 0.9, 0.3], names=["A", "B", "C"], k=2)
        result = agent.analyze(ctx)

        # The TracedResult should be unwrapped to its .value
        assert "top_k_result" in result
        # Should be the actual value dict, not a TracedResult
        assert not isinstance(result["top_k_result"], TracedResult)

    def test_analyze_injects_execution_traces(self):
        agent = _TopKAgent()
        ctx = self._make_context(values=[0.5, 0.9], names=["A", "B"], k=1)
        result = agent.analyze(ctx)

        assert "_execution_traces" in result
        assert "_execution_tag" in result
        assert isinstance(result["_execution_traces"], list)
        assert len(result["_execution_traces"]) >= 1

    def test_analyze_overall_tag_computed(self):
        agent = _TopKAgent()
        ctx = self._make_context(values=[0.5, 0.9], names=["A", "B"], k=1)
        result = agent.analyze(ctx)
        assert result["_execution_tag"] == "computed"

    def test_analyze_overall_tag_failed(self):
        agent = _FailingAgent()
        ctx = self._make_context()
        result = agent.analyze(ctx)
        assert result["_execution_tag"] == "failed"

    def test_analyze_preserves_plain_values(self):
        agent = _MixedAgent()
        ctx = self._make_context()
        result = agent.analyze(ctx)

        # Plain value should be preserved
        assert result["config"] == {"seed": 42}
        # Traced value should be unwrapped
        assert not isinstance(result["computed"], TracedResult)

    def test_analyze_resets_traces_each_call(self):
        agent = _TopKAgent()
        ctx1 = self._make_context(values=[0.5, 0.9], names=["A", "B"], k=1)
        agent.analyze(ctx1)
        n1 = len(agent.collected_traces)

        ctx2 = self._make_context(values=[0.3, 0.7, 0.1], names=["X", "Y", "Z"], k=2)
        agent.analyze(ctx2)
        n2 = len(agent.collected_traces)

        # Traces are reset, not accumulated
        assert n1 >= 1
        assert n2 >= 1


class TestGetOptimizationFeedback:
    def _make_context(self, **metadata):
        return AgentContext(metadata=metadata)

    def test_feedback_has_traces_in_payload(self):
        agent = _TopKAgent()
        ctx = self._make_context(values=[0.5, 0.9, 0.3], names=["A", "B", "C"], k=2)
        result = agent.analyze(ctx)
        feedback = agent.get_optimization_feedback(result)

        assert feedback is not None
        assert "_execution_traces" in feedback.payload
        assert "_execution_tag" in feedback.payload
        assert feedback.payload["_execution_tag"] == "computed"

    def test_feedback_none_when_build_returns_none(self):
        agent = _NoFeedbackAgent()
        ctx = self._make_context()
        result = agent.analyze(ctx)
        feedback = agent.get_optimization_feedback(result)
        assert feedback is None

    def test_feedback_preserves_agent_name(self):
        agent = _TopKAgent()
        ctx = self._make_context(values=[0.1, 0.2], names=["X", "Y"], k=1)
        result = agent.analyze(ctx)
        feedback = agent.get_optimization_feedback(result)
        assert feedback.agent_name == "top_k_test_agent"

    def test_feedback_preserves_confidence(self):
        agent = _TopKAgent()
        ctx = self._make_context(values=[0.1], names=["X"], k=1)
        result = agent.analyze(ctx)
        feedback = agent.get_optimization_feedback(result)
        assert feedback.confidence == 0.9

    def test_failed_agent_feedback_has_failed_tag(self):
        agent = _FailingAgent()
        ctx = self._make_context()
        result = agent.analyze(ctx)
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert feedback.payload["_execution_tag"] == "failed"


class TestOrchestratorIntegration:
    def test_with_orchestrator_dispatch(self):
        from optimization_copilot.agents.orchestrator import (
            OrchestratorEvent,
            ScientificOrchestrator,
        )

        agent = _TopKAgent()
        orchestrator = ScientificOrchestrator(agents=[agent])
        ctx = AgentContext(
            metadata={"values": [0.3, 0.8, 0.5], "names": ["P1", "P2", "P3"], "k": 2}
        )
        event = OrchestratorEvent(event_type="observation")
        feedbacks = orchestrator.dispatch_event(event, ctx)

        assert len(feedbacks) == 1
        fb = feedbacks[0]
        assert fb.agent_name == "top_k_test_agent"
        assert "_execution_traces" in fb.payload
        assert fb.payload["_execution_tag"] == "computed"

    def test_with_guard_validation(self):
        from optimization_copilot.agents.execution_guard import (
            ExecutionGuard,
            GuardMode,
        )

        agent = _TopKAgent()
        ctx = AgentContext(
            metadata={"values": [0.3, 0.8, 0.5], "names": ["P1", "P2", "P3"], "k": 2}
        )
        result = agent.analyze(ctx)
        feedback = agent.get_optimization_feedback(result)

        guard = ExecutionGuard(mode=GuardMode.STRICT)
        is_valid, issues = guard.validate_feedback(feedback)
        # The feedback has traced top_k data → should pass
        assert is_valid, f"Guard issues: {issues}"

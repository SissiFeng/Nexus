"""Tests for the optimization_copilot.agents.experiment_planner module.

Covers ExperimentPlannerAgent, PlannerConfig, activation logic,
context validation, pragmatic analysis, feedback generation,
and LLM-enhanced fallback behaviour.
"""

from __future__ import annotations

from typing import Any

import pytest

from optimization_copilot.agents import ExperimentPlannerAgent, PlannerConfig
from optimization_copilot.agents.base import AgentContext, AgentMode, OptimizationFeedback


# ── Helpers ───────────────────────────────────────────────────────────


def _make_context(
    history: list[dict[str, Any]] | None = None,
    iteration: int = 0,
    metadata: dict[str, Any] | None = None,
) -> AgentContext:
    """Build a minimal AgentContext for testing."""
    return AgentContext(
        optimization_history=history or [],
        iteration=iteration,
        metadata=metadata or {},
    )


def _make_history(n: int, improving: bool = True) -> list[dict[str, Any]]:
    """Generate a synthetic optimization history of length *n*.

    If *improving* is True, objectives decrease (improve) over iterations.
    Otherwise they increase (stagnate/degrade).
    """
    entries: list[dict[str, Any]] = []
    for i in range(1, n + 1):
        obj = 1.0 - 0.05 * i if improving else 0.5 + 0.01 * i
        entries.append({"parameters": {"x": float(i)}, "objective": obj, "iteration": i})
    return entries


# ── PlannerConfig defaults ────────────────────────────────────────────


class TestPlannerConfig:
    """Test default values of PlannerConfig."""

    def test_defaults(self):
        cfg = PlannerConfig()
        assert cfg.model_name == "claude-opus-4-6"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == pytest.approx(0.7)


# ── ExperimentPlannerAgent.name ───────────────────────────────────────


class TestAgentName:
    def test_name_returns_experiment_planner(self):
        agent = ExperimentPlannerAgent()
        assert agent.name() == "experiment_planner"


# ── should_activate ──────────────────────────────────────────────────


class TestShouldActivate:
    """Test the conditional activation logic of the planner."""

    def test_stagnation_event_activates(self):
        agent = ExperimentPlannerAgent()
        ctx = _make_context(iteration=3, metadata={"event": "stagnation"})
        assert agent.should_activate(ctx) is True

    def test_milestone_event_activates(self):
        agent = ExperimentPlannerAgent()
        ctx = _make_context(iteration=3, metadata={"event": "milestone"})
        assert agent.should_activate(ctx) is True

    def test_iteration_multiple_of_10_activates(self):
        agent = ExperimentPlannerAgent()
        ctx = _make_context(iteration=10, metadata={})
        assert agent.should_activate(ctx) is True

    def test_iteration_3_no_event_does_not_activate(self):
        agent = ExperimentPlannerAgent()
        ctx = _make_context(iteration=3, metadata={})
        assert agent.should_activate(ctx) is False


# ── validate_context ─────────────────────────────────────────────────


class TestValidateContext:
    """Test context validation (requires >= 1 history entry)."""

    def test_empty_history_returns_false(self):
        agent = ExperimentPlannerAgent()
        ctx = _make_context(history=[])
        assert agent.validate_context(ctx) is False

    def test_one_entry_returns_true(self):
        agent = ExperimentPlannerAgent()
        ctx = _make_context(
            history=[{"parameters": {"x": 1.0}, "objective": 0.5, "iteration": 1}],
        )
        assert agent.validate_context(ctx) is True


# ── analyze (PRAGMATIC mode) ─────────────────────────────────────────


class TestAnalyzePragmatic:
    """Test the deterministic pragmatic analysis paths."""

    def test_stagnating_trend(self):
        """Flat or rising objectives should trigger the stagnation hypothesis."""
        history = _make_history(10, improving=False)
        ctx = _make_context(history=history, iteration=10)
        agent = ExperimentPlannerAgent(mode=AgentMode.PRAGMATIC)
        result = agent.analyze(ctx)

        assert result["mode_used"] == "pragmatic"
        # Stagnating/degrading trend should mention exploration or re-centre
        assert "hypothesis" in result
        assert len(result["recommendations"]) >= 2

    def test_improving_trend(self):
        """Improving objectives should trigger the 'steady progress' hypothesis."""
        history = _make_history(10, improving=True)
        ctx = _make_context(history=history, iteration=10)
        agent = ExperimentPlannerAgent(mode=AgentMode.PRAGMATIC)
        result = agent.analyze(ctx)

        assert result["mode_used"] == "pragmatic"
        assert "progress" in result["hypothesis"].lower() or "exploit" in result["hypothesis"].lower()

    def test_early_stage(self):
        """With iteration <= 5 and insufficient trend data, agent suggests space-filling."""
        # Use a single history entry so _compute_trend returns "insufficient_data",
        # which lets the analysis fall through to the iteration <= 5 branch.
        history = [{"parameters": {"x": 1.0}, "objective": 0.5, "iteration": 1}]
        ctx = _make_context(history=history, iteration=3)
        agent = ExperimentPlannerAgent(mode=AgentMode.PRAGMATIC)
        result = agent.analyze(ctx)

        assert result["mode_used"] == "pragmatic"
        hyp_lower = result["hypothesis"].lower()
        assert "early" in hyp_lower or "space" in hyp_lower or "limited" in hyp_lower


# ── get_optimization_feedback ─────────────────────────────────────────


class TestGetOptimizationFeedback:
    """Test conversion of analysis results to OptimizationFeedback."""

    def test_returns_hypothesis_feedback(self):
        agent = ExperimentPlannerAgent()
        analysis = {
            "hypothesis": "Test hypothesis",
            "recommendations": ["Do A", "Do B"],
            "mode_used": "pragmatic",
            "summary": {"history_length": 10},
        }
        fb = agent.get_optimization_feedback(analysis)
        assert fb is not None
        assert isinstance(fb, OptimizationFeedback)
        assert fb.feedback_type == "hypothesis"
        assert fb.agent_name == "experiment_planner"
        assert fb.payload["hypothesis"] == "Test hypothesis"

    def test_confidence_varies_with_history_length(self):
        agent = ExperimentPlannerAgent()

        # Short history -> lower confidence
        short = {
            "hypothesis": "h",
            "recommendations": [],
            "mode_used": "pragmatic",
            "summary": {"history_length": 2},
        }
        fb_short = agent.get_optimization_feedback(short)

        # Long history -> higher confidence
        long = {
            "hypothesis": "h",
            "recommendations": [],
            "mode_used": "pragmatic",
            "summary": {"history_length": 25},
        }
        fb_long = agent.get_optimization_feedback(long)

        assert fb_short is not None and fb_long is not None
        assert fb_long.confidence > fb_short.confidence


# ── LLM_ENHANCED fallback ────────────────────────────────────────────


class TestLLMEnhancedFallback:
    """Test that LLM_ENHANCED mode gracefully falls back to pragmatic."""

    def test_falls_back_when_no_api_key(self, monkeypatch):
        """Without MODEL_API_KEY / ANTHROPIC_API_KEY, should fall back to pragmatic."""
        monkeypatch.delenv("MODEL_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        agent = ExperimentPlannerAgent(mode=AgentMode.LLM_ENHANCED)
        history = _make_history(5, improving=True)
        ctx = _make_context(history=history, iteration=5)
        result = agent.analyze(ctx)

        # Should have fallen back to pragmatic mode
        assert result["mode_used"] == "pragmatic"


# ── __repr__ ──────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_string(self):
        agent = ExperimentPlannerAgent()
        r = repr(agent)
        assert "ExperimentPlannerAgent" in r
        assert "experiment_planner" in r
        assert "pragmatic" in r

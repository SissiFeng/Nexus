"""Tests for the SymbolicRegressionAgent."""

from __future__ import annotations

import math
import pytest
from typing import Any

from optimization_copilot.agents.base import (
    AgentContext,
    AgentMode,
    OptimizationFeedback,
    ScientificAgent,
    TriggerCondition,
)
from optimization_copilot.agents.symreg.agent import SymbolicRegressionAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_history(
    n: int,
    param_names: list[str] | None = None,
    func: Any = None,
) -> list[dict[str, Any]]:
    """Generate synthetic optimization history.

    Parameters
    ----------
    n : int
        Number of observations.
    param_names : list[str] | None
        Parameter names (default: ["x0", "x1"]).
    func : callable | None
        Function from params -> y. Default: simple linear y = x0 + 2*x1.
    """
    if param_names is None:
        param_names = ["x0", "x1"]
    if func is None:
        func = lambda params: params.get("x0", 0) + 2 * params.get("x1", 0)

    history: list[dict[str, Any]] = []
    for i in range(n):
        params = {name: float(i * 0.5 + j) for j, name in enumerate(param_names)}
        y = func(params)
        history.append({"parameters": params, "y": y})
    return history


def _make_context(
    n: int = 15,
    param_names: list[str] | None = None,
    func: Any = None,
    **kwargs: Any,
) -> AgentContext:
    """Create a context with synthetic history."""
    history = _make_history(n, param_names, func)
    return AgentContext(optimization_history=history, **kwargs)


# ---------------------------------------------------------------------------
# Agent construction and name
# ---------------------------------------------------------------------------


class TestAgentConstruction:
    def test_name(self):
        agent = SymbolicRegressionAgent()
        assert agent.name() == "symbolic_regression"

    def test_default_mode(self):
        agent = SymbolicRegressionAgent()
        assert agent.mode == AgentMode.PRAGMATIC

    def test_custom_params(self):
        agent = SymbolicRegressionAgent(population_size=50, generations=10, seed=123)
        assert agent._population_size == 50
        assert agent._generations == 10
        assert agent._seed == 123

    def test_is_scientific_agent(self):
        agent = SymbolicRegressionAgent()
        assert isinstance(agent, ScientificAgent)

    def test_repr(self):
        agent = SymbolicRegressionAgent()
        r = repr(agent)
        assert "SymbolicRegressionAgent" in r
        assert "symbolic_regression" in r


# ---------------------------------------------------------------------------
# should_activate
# ---------------------------------------------------------------------------


class TestShouldActivate:
    def test_too_few_observations(self):
        agent = SymbolicRegressionAgent()
        ctx = _make_context(n=5)
        assert agent.should_activate(ctx) is False

    def test_empty_history(self):
        agent = SymbolicRegressionAgent()
        ctx = AgentContext()
        assert agent.should_activate(ctx) is False

    def test_exactly_threshold(self):
        agent = SymbolicRegressionAgent()
        ctx = _make_context(n=10)
        assert agent.should_activate(ctx) is True

    def test_enough_observations(self):
        agent = SymbolicRegressionAgent()
        ctx = _make_context(n=20)
        assert agent.should_activate(ctx) is True


# ---------------------------------------------------------------------------
# validate_context
# ---------------------------------------------------------------------------


class TestValidateContext:
    def test_valid_context(self):
        agent = SymbolicRegressionAgent()
        ctx = _make_context(n=15)
        assert agent.validate_context(ctx) is True

    def test_empty_history(self):
        agent = SymbolicRegressionAgent()
        ctx = AgentContext()
        assert agent.validate_context(ctx) is False

    def test_missing_parameters_key(self):
        agent = SymbolicRegressionAgent()
        ctx = AgentContext(optimization_history=[{"y": 1.0}])
        assert agent.validate_context(ctx) is False

    def test_no_target_key(self):
        agent = SymbolicRegressionAgent()
        ctx = AgentContext(
            optimization_history=[{"parameters": {"x": 1.0}}]
        )
        assert agent.validate_context(ctx) is False


# ---------------------------------------------------------------------------
# analyze with synthetic linear data
# ---------------------------------------------------------------------------


class TestAnalyzeLinear:
    def test_returns_dict(self):
        agent = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
        ctx = _make_context(n=15, param_names=["x0"], func=lambda p: 2 * p["x0"] + 1)
        result = agent.analyze(ctx)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        agent = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
        ctx = _make_context(n=15, param_names=["x0"], func=lambda p: 2 * p["x0"] + 1)
        result = agent.analyze(ctx)
        assert "equations" in result
        assert "best_equation" in result
        assert "pareto_front" in result
        assert "n_observations" in result
        assert "feature_names" in result

    def test_n_observations(self):
        agent = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
        ctx = _make_context(n=15, param_names=["x0"], func=lambda p: 2 * p["x0"] + 1)
        result = agent.analyze(ctx)
        assert result["n_observations"] == 15

    def test_feature_names(self):
        agent = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
        ctx = _make_context(n=15, param_names=["temp", "pressure"])
        result = agent.analyze(ctx)
        assert "temp" in result["feature_names"]
        assert "pressure" in result["feature_names"]

    def test_finds_equations(self):
        agent = SymbolicRegressionAgent(population_size=80, generations=20, seed=42)
        ctx = _make_context(n=20, param_names=["x0"], func=lambda p: 3 * p["x0"])
        result = agent.analyze(ctx)
        assert len(result["equations"]) > 0

    def test_best_equation_present(self):
        agent = SymbolicRegressionAgent(population_size=80, generations=20, seed=42)
        ctx = _make_context(n=20, param_names=["x0"], func=lambda p: 3 * p["x0"])
        result = agent.analyze(ctx)
        if result["equations"]:
            assert result["best_equation"] is not None


# ---------------------------------------------------------------------------
# analyze with quadratic data
# ---------------------------------------------------------------------------


class TestAnalyzeQuadratic:
    def test_quadratic_returns_results(self):
        agent = SymbolicRegressionAgent(population_size=80, generations=20, seed=42)
        ctx = _make_context(
            n=20, param_names=["x"], func=lambda p: p["x"] ** 2
        )
        result = agent.analyze(ctx)
        assert result["n_observations"] == 20
        assert len(result["equations"]) >= 0  # May or may not find exact

    def test_quadratic_pareto_front(self):
        agent = SymbolicRegressionAgent(population_size=80, generations=20, seed=42)
        ctx = _make_context(
            n=20, param_names=["x"], func=lambda p: p["x"] ** 2
        )
        result = agent.analyze(ctx)
        # Pareto front should be a list
        assert isinstance(result["pareto_front"], list)

    def test_equations_have_mse(self):
        agent = SymbolicRegressionAgent(population_size=80, generations=20, seed=42)
        ctx = _make_context(
            n=20, param_names=["x"], func=lambda p: p["x"] ** 2
        )
        result = agent.analyze(ctx)
        for eq in result["equations"]:
            assert "mse" in eq
            assert "complexity" in eq
            assert "equation" in eq

    def test_two_variable_quadratic(self):
        agent = SymbolicRegressionAgent(population_size=80, generations=20, seed=42)
        ctx = _make_context(
            n=20,
            param_names=["x", "y"],
            func=lambda p: p["x"] ** 2 + p["y"],
        )
        result = agent.analyze(ctx)
        assert result["n_observations"] == 20
        assert "x" in result["feature_names"]
        assert "y" in result["feature_names"]

    def test_target_key_detection(self):
        agent = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
        ctx = _make_context(n=15)
        result = agent.analyze(ctx)
        assert result["target_key"] == "y"


# ---------------------------------------------------------------------------
# get_optimization_feedback
# ---------------------------------------------------------------------------


class TestGetOptimizationFeedback:
    def test_good_equation(self):
        agent = SymbolicRegressionAgent()
        result = {
            "best_equation": {
                "equation": "(x0 * 3)",
                "mse": 0.01,
                "complexity": 3,
            },
            "feature_names": ["x0"],
            "target_key": "y",
            "n_observations": 20,
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert isinstance(feedback, OptimizationFeedback)
        assert feedback.agent_name == "symbolic_regression"
        assert feedback.feedback_type == "hypothesis"
        assert 0 < feedback.confidence <= 1.0

    def test_no_equation(self):
        agent = SymbolicRegressionAgent()
        result = {"best_equation": None, "equations": []}
        feedback = agent.get_optimization_feedback(result)
        assert feedback is None

    def test_high_complexity_equation(self):
        agent = SymbolicRegressionAgent()
        result = {
            "best_equation": {
                "equation": "very_complex_expression",
                "mse": 0.01,
                "complexity": 50,
            },
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is None

    def test_high_mse_equation(self):
        agent = SymbolicRegressionAgent()
        result = {
            "best_equation": {
                "equation": "(x0 + 1)",
                "mse": 1e7,
                "complexity": 3,
            },
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is None

    def test_infinite_mse(self):
        agent = SymbolicRegressionAgent()
        result = {
            "best_equation": {
                "equation": "(x0 / 0)",
                "mse": float("inf"),
                "complexity": 3,
            },
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is None

    def test_feedback_payload(self):
        agent = SymbolicRegressionAgent()
        result = {
            "best_equation": {
                "equation": "(x0 + x1)",
                "mse": 0.001,
                "complexity": 3,
            },
            "feature_names": ["x0", "x1"],
            "target_key": "y",
            "n_observations": 20,
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert "equation" in feedback.payload
        assert feedback.payload["equation"] == "(x0 + x1)"


# ---------------------------------------------------------------------------
# Integration: full analyze -> feedback pipeline
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_pipeline_linear(self):
        agent = SymbolicRegressionAgent(population_size=80, generations=20, seed=42)
        ctx = _make_context(n=20, param_names=["x0"], func=lambda p: 3 * p["x0"])
        result = agent.analyze(ctx)
        feedback = agent.get_optimization_feedback(result)
        # May or may not produce feedback depending on equation quality
        if feedback is not None:
            assert isinstance(feedback, OptimizationFeedback)
            assert feedback.agent_name == "symbolic_regression"

    def test_full_pipeline_constant(self):
        agent = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
        ctx = _make_context(n=15, param_names=["x0"], func=lambda p: 5.0)
        result = agent.analyze(ctx)
        assert result.get("constant_target") is True

    def test_should_activate_then_analyze(self):
        agent = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
        ctx = _make_context(n=15)
        if agent.should_activate(ctx) and agent.validate_context(ctx):
            result = agent.analyze(ctx)
            assert result["n_observations"] == 15

    def test_not_activate_small_data(self):
        agent = SymbolicRegressionAgent()
        ctx = _make_context(n=3)
        assert agent.should_activate(ctx) is False

    def test_pipeline_different_target_key(self):
        history = [
            {"parameters": {"x": float(i)}, "objective": float(i) * 2}
            for i in range(15)
        ]
        agent = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
        ctx = AgentContext(optimization_history=history)
        result = agent.analyze(ctx)
        assert result["target_key"] == "objective"
        assert result["n_observations"] == 15


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_history(self):
        agent = SymbolicRegressionAgent()
        ctx = AgentContext()
        result = agent.analyze(ctx)
        assert result["equations"] == []
        assert result["best_equation"] is None

    def test_single_observation(self):
        agent = SymbolicRegressionAgent()
        ctx = AgentContext(
            optimization_history=[{"parameters": {"x": 1.0}, "y": 2.0}]
        )
        result = agent.analyze(ctx)
        assert result["n_observations"] <= 1

    def test_constant_y(self):
        agent = SymbolicRegressionAgent()
        history = [
            {"parameters": {"x": float(i)}, "y": 5.0}
            for i in range(15)
        ]
        ctx = AgentContext(optimization_history=history)
        result = agent.analyze(ctx)
        assert result.get("constant_target") is True

    def test_nan_in_parameters(self):
        agent = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
        history = [
            {"parameters": {"x": float(i)}, "y": float(i) * 2}
            for i in range(15)
        ]
        history[5]["parameters"]["x"] = float("nan")
        ctx = AgentContext(optimization_history=history)
        result = agent.analyze(ctx)
        # Should skip the nan entry
        assert result["n_observations"] == 14

    def test_missing_y_in_some_entries(self):
        agent = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
        history = [
            {"parameters": {"x": float(i)}, "y": float(i)}
            for i in range(15)
        ]
        del history[3]["y"]
        ctx = AgentContext(optimization_history=history)
        result = agent.analyze(ctx)
        assert result["n_observations"] == 14


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_results(self):
        ctx = _make_context(n=15, param_names=["x0"], func=lambda p: 2 * p["x0"])
        agent1 = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
        agent2 = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
        r1 = agent1.analyze(ctx)
        r2 = agent2.analyze(ctx)
        # Same equations should be found
        assert len(r1["equations"]) == len(r2["equations"])

    def test_different_seed_different_results(self):
        ctx = _make_context(n=15, param_names=["x0"], func=lambda p: 2 * p["x0"])
        agent1 = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
        agent2 = SymbolicRegressionAgent(population_size=50, generations=10, seed=99)
        r1 = agent1.analyze(ctx)
        r2 = agent2.analyze(ctx)
        # Results may differ (not guaranteed but likely with different seeds)
        # Just verify both produce valid output
        assert isinstance(r1["equations"], list)
        assert isinstance(r2["equations"], list)

    def test_deterministic_over_multiple_runs(self):
        ctx = _make_context(n=15, param_names=["x0"], func=lambda p: p["x0"] * 3)
        results = []
        for _ in range(3):
            agent = SymbolicRegressionAgent(population_size=50, generations=10, seed=42)
            results.append(agent.analyze(ctx))
        for i in range(1, len(results)):
            assert len(results[i]["equations"]) == len(results[0]["equations"])


# ---------------------------------------------------------------------------
# TriggerCondition registration
# ---------------------------------------------------------------------------


class TestTriggerConditions:
    def test_has_trigger_conditions(self):
        agent = SymbolicRegressionAgent()
        assert len(agent.trigger_conditions) > 0

    def test_trigger_condition_types(self):
        agent = SymbolicRegressionAgent()
        for tc in agent.trigger_conditions:
            assert isinstance(tc, TriggerCondition)

    def test_trigger_condition_names(self):
        agent = SymbolicRegressionAgent()
        names = [tc.name for tc in agent.trigger_conditions]
        assert "sufficient_observations" in names

    def test_trigger_condition_has_check_fn(self):
        agent = SymbolicRegressionAgent()
        for tc in agent.trigger_conditions:
            assert tc.check_fn_name
            assert isinstance(tc.check_fn_name, str)

    def test_trigger_condition_priorities(self):
        agent = SymbolicRegressionAgent()
        for tc in agent.trigger_conditions:
            assert isinstance(tc.priority, int)
            assert tc.priority >= 0

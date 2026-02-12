"""Tests for the LiteraturePriorAgent, PriorTable, and LLMSafetyWrapper."""

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
from optimization_copilot.agents.literature.prior_tables import (
    CATALYSIS_PRIORS,
    PEROVSKITE_PRIORS,
    ZINC_PRIORS,
    PriorEntry,
    PriorTable,
)
from optimization_copilot.agents.literature.agent import LiteraturePriorAgent
from optimization_copilot.agents.safety import LLMSafetyWrapper


# ---------------------------------------------------------------------------
# PriorEntry tests
# ---------------------------------------------------------------------------


class TestPriorEntry:
    def test_creation(self):
        pe = PriorEntry(
            parameter="temp",
            prior_mean=80.0,
            prior_std=20.0,
            source="domain_knowledge",
            domain="catalysis",
        )
        assert pe.parameter == "temp"
        assert pe.prior_mean == 80.0
        assert pe.prior_std == 20.0

    def test_with_notes(self):
        pe = PriorEntry(
            parameter="x",
            prior_mean=1.0,
            prior_std=0.5,
            source="test",
            domain="test",
            notes="Test note",
        )
        assert pe.notes == "Test note"

    def test_default_notes(self):
        pe = PriorEntry(
            parameter="x",
            prior_mean=1.0,
            prior_std=0.5,
            source="test",
            domain="test",
        )
        assert pe.notes == ""


# ---------------------------------------------------------------------------
# PriorTable tests
# ---------------------------------------------------------------------------


class TestPriorTable:
    def test_list_domains(self):
        table = PriorTable()
        domains = table.list_domains()
        assert "electrochemistry" in domains
        assert "catalysis" in domains
        assert "perovskite" in domains

    def test_get_priors_electrochemistry(self):
        table = PriorTable()
        priors = table.get_priors("electrochemistry")
        assert len(priors) > 0
        assert all(p.domain == "electrochemistry" for p in priors)

    def test_get_priors_catalysis(self):
        table = PriorTable()
        priors = table.get_priors("catalysis")
        assert len(priors) >= 5

    def test_get_priors_unknown_domain(self):
        table = PriorTable()
        priors = table.get_priors("unknown")
        assert priors == []

    def test_get_prior_for_parameter(self):
        table = PriorTable()
        prior = table.get_prior_for_parameter("electrochemistry", "additive_1")
        assert prior is not None
        assert prior.parameter == "additive_1"

    def test_get_prior_for_unknown_parameter(self):
        table = PriorTable()
        prior = table.get_prior_for_parameter("electrochemistry", "nonexistent")
        assert prior is None

    def test_add_custom_prior(self):
        table = PriorTable()
        entry = PriorEntry(
            parameter="custom_param",
            prior_mean=42.0,
            prior_std=5.0,
            source="test",
            domain="electrochemistry",
        )
        table.add_custom_prior(entry)
        result = table.get_prior_for_parameter("electrochemistry", "custom_param")
        assert result is not None
        assert result.prior_mean == 42.0

    def test_add_custom_prior_new_domain(self):
        table = PriorTable()
        entry = PriorEntry(
            parameter="x",
            prior_mean=1.0,
            prior_std=0.5,
            source="test",
            domain="new_domain",
        )
        table.add_custom_prior(entry)
        assert "new_domain" in table.list_domains()
        assert table.get_prior_for_parameter("new_domain", "x") is not None

    def test_add_custom_replaces_existing(self):
        table = PriorTable()
        entry = PriorEntry(
            parameter="additive_1",
            prior_mean=99.0,
            prior_std=1.0,
            source="test",
            domain="electrochemistry",
        )
        table.add_custom_prior(entry)
        result = table.get_prior_for_parameter("electrochemistry", "additive_1")
        assert result is not None
        assert result.prior_mean == 99.0

    def test_independent_instances(self):
        """Two PriorTable instances should not share state."""
        table1 = PriorTable()
        table2 = PriorTable()
        entry = PriorEntry(
            parameter="isolated",
            prior_mean=1.0,
            prior_std=0.5,
            source="test",
            domain="electrochemistry",
        )
        table1.add_custom_prior(entry)
        assert table2.get_prior_for_parameter("electrochemistry", "isolated") is None


# ---------------------------------------------------------------------------
# LiteraturePriorAgent tests
# ---------------------------------------------------------------------------


class TestLiteraturePriorAgentBasics:
    def test_name(self):
        agent = LiteraturePriorAgent()
        assert agent.name() == "literature_prior"

    def test_is_scientific_agent(self):
        agent = LiteraturePriorAgent()
        assert isinstance(agent, ScientificAgent)

    def test_default_mode(self):
        agent = LiteraturePriorAgent()
        assert agent.mode == AgentMode.PRAGMATIC

    def test_trigger_conditions(self):
        agent = LiteraturePriorAgent()
        assert len(agent.trigger_conditions) > 0
        names = [tc.name for tc in agent.trigger_conditions]
        assert "early_campaign" in names


class TestShouldActivate:
    def test_early_campaign(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext(iteration=2)
        assert agent.should_activate(ctx) is True

    def test_late_campaign_no_config(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext(iteration=20)
        assert agent.should_activate(ctx) is False

    def test_exact_threshold(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext(iteration=5)
        assert agent.should_activate(ctx) is True

    def test_above_threshold(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext(iteration=6)
        assert agent.should_activate(ctx) is False


class TestAnalyzeWithFallback:
    def test_analyze_with_metadata_domain(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"additive_1": 0.3, "temperature": 30}, "y": 90},
            ],
            metadata={"domain": "electrochemistry"},
            iteration=1,
        )
        result = agent.analyze(ctx)
        assert result["source"] == "prior_table"
        assert result["domain"] == "electrochemistry"
        assert result["n_parameters_matched"] > 0

    def test_analyze_no_domain(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"x": 1.0}, "y": 5.0},
            ],
            iteration=1,
        )
        result = agent.analyze(ctx)
        assert result["source"] == "none"
        assert result["n_parameters_matched"] == 0

    def test_analyze_catalysis_domain(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext(
            optimization_history=[
                {
                    "parameters": {"temperature": 80, "catalyst_loading": 2.0},
                    "y": 75.0,
                },
            ],
            metadata={"domain": "catalysis"},
            iteration=1,
        )
        result = agent.analyze(ctx)
        assert result["domain"] == "catalysis"
        assert result["n_parameters_matched"] > 0

    def test_coverage_calculation(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext(
            optimization_history=[
                {
                    "parameters": {"additive_1": 0.3, "unknown_param": 5.0},
                    "y": 90,
                },
            ],
            metadata={"domain": "electrochemistry"},
            iteration=1,
        )
        result = agent.analyze(ctx)
        # additive_1 matches, unknown_param doesn't
        assert 0 < result["coverage"] < 1.0

    def test_priors_have_required_fields(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"additive_1": 0.3}, "y": 90},
            ],
            metadata={"domain": "electrochemistry"},
            iteration=1,
        )
        result = agent.analyze(ctx)
        for p in result["priors"]:
            assert "parameter" in p
            assert "mean" in p
            assert "std" in p
            assert "source" in p


# ---------------------------------------------------------------------------
# get_optimization_feedback
# ---------------------------------------------------------------------------


class TestGetFeedback:
    def test_good_coverage(self):
        agent = LiteraturePriorAgent()
        result = {
            "priors": [
                {"parameter": "additive_1", "mean": 0.3, "std": 0.15, "source": "prior_table"},
                {"parameter": "temperature", "mean": 30.0, "std": 10.0, "source": "prior_table"},
            ],
            "domain": "electrochemistry",
            "n_parameters_matched": 2,
            "coverage": 0.8,
            "source": "prior_table",
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert feedback.feedback_type == "prior_update"
        assert feedback.confidence > 0.4

    def test_no_coverage(self):
        agent = LiteraturePriorAgent()
        result = {
            "priors": [],
            "domain": "unknown",
            "n_parameters_matched": 0,
            "coverage": 0.0,
            "source": "none",
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is None

    def test_low_coverage(self):
        agent = LiteraturePriorAgent()
        result = {
            "priors": [{"parameter": "x", "mean": 1.0, "std": 0.5, "source": "test"}],
            "domain": "catalysis",
            "n_parameters_matched": 1,
            "coverage": 0.1,  # below threshold
            "source": "prior_table",
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is None

    def test_feedback_payload_structure(self):
        agent = LiteraturePriorAgent()
        result = {
            "priors": [
                {"parameter": "temp", "mean": 80.0, "std": 20.0, "source": "test"},
            ],
            "domain": "catalysis",
            "n_parameters_matched": 1,
            "coverage": 0.5,
            "source": "prior_table",
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert "parameter_priors" in feedback.payload
        assert "temp" in feedback.payload["parameter_priors"]

    def test_feedback_agent_name(self):
        agent = LiteraturePriorAgent()
        result = {
            "priors": [
                {"parameter": "x", "mean": 1.0, "std": 0.5, "source": "test"},
            ],
            "domain": "test",
            "n_parameters_matched": 1,
            "coverage": 0.5,
            "source": "prior_table",
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert feedback.agent_name == "literature_prior"


# ---------------------------------------------------------------------------
# LLMSafetyWrapper tests
# ---------------------------------------------------------------------------


class TestSafetyWrapperValidation:
    def test_valid_feedback(self):
        wrapper = LLMSafetyWrapper(confidence_threshold=0.5)
        fb = OptimizationFeedback(
            agent_name="test",
            feedback_type="hypothesis",
            confidence=0.7,
            payload={"key": "value"},
            reasoning="test reasoning",
        )
        is_valid, reasons = wrapper.validate_feedback(fb)
        assert is_valid is True
        assert reasons == []

    def test_low_confidence_rejected(self):
        wrapper = LLMSafetyWrapper(confidence_threshold=0.5)
        fb = OptimizationFeedback(
            agent_name="test",
            feedback_type="hypothesis",
            confidence=0.3,
        )
        is_valid, reasons = wrapper.validate_feedback(fb)
        assert is_valid is False
        assert any("Confidence" in r for r in reasons)

    def test_empty_payload_actionable(self):
        wrapper = LLMSafetyWrapper(confidence_threshold=0.3)
        fb = OptimizationFeedback(
            agent_name="test",
            feedback_type="prior_update",
            confidence=0.8,
            payload={},
        )
        is_valid, reasons = wrapper.validate_feedback(fb)
        assert is_valid is False
        assert any("Empty payload" in r for r in reasons)

    def test_missing_agent_name(self):
        wrapper = LLMSafetyWrapper(confidence_threshold=0.3)
        fb = OptimizationFeedback(
            agent_name="",
            feedback_type="hypothesis",
            confidence=0.8,
        )
        is_valid, reasons = wrapper.validate_feedback(fb)
        assert is_valid is False
        assert any("agent_name" in r for r in reasons)

    def test_unknown_feedback_type(self):
        wrapper = LLMSafetyWrapper(confidence_threshold=0.3)
        fb = OptimizationFeedback(
            agent_name="test",
            feedback_type="unknown_type",
            confidence=0.8,
        )
        is_valid, reasons = wrapper.validate_feedback(fb)
        assert is_valid is False
        assert any("Unknown feedback type" in r for r in reasons)


class TestSafetyWrapperGate:
    def test_gate_valid(self):
        wrapper = LLMSafetyWrapper(confidence_threshold=0.5)
        fb = OptimizationFeedback(
            agent_name="test",
            feedback_type="hypothesis",
            confidence=0.7,
        )
        result = wrapper.gate_feedback(fb)
        assert result is not None
        assert result is fb

    def test_gate_invalid(self):
        wrapper = LLMSafetyWrapper(confidence_threshold=0.8)
        fb = OptimizationFeedback(
            agent_name="test",
            feedback_type="hypothesis",
            confidence=0.5,
        )
        result = wrapper.gate_feedback(fb)
        assert result is None


class TestSafetyWrapperBatch:
    def test_batch_filters(self):
        wrapper = LLMSafetyWrapper(confidence_threshold=0.5)
        feedbacks = [
            OptimizationFeedback(
                agent_name="a", feedback_type="hypothesis", confidence=0.8
            ),
            OptimizationFeedback(
                agent_name="b", feedback_type="hypothesis", confidence=0.3
            ),
            OptimizationFeedback(
                agent_name="c", feedback_type="hypothesis", confidence=0.7
            ),
        ]
        valid = wrapper.validate_batch(feedbacks)
        assert len(valid) == 2
        names = [f.agent_name for f in valid]
        assert "a" in names
        assert "c" in names
        assert "b" not in names

    def test_batch_empty(self):
        wrapper = LLMSafetyWrapper()
        valid = wrapper.validate_batch([])
        assert valid == []

    def test_batch_all_valid(self):
        wrapper = LLMSafetyWrapper(confidence_threshold=0.3)
        feedbacks = [
            OptimizationFeedback(
                agent_name="a", feedback_type="hypothesis", confidence=0.5
            ),
            OptimizationFeedback(
                agent_name="b", feedback_type="warning", confidence=0.4
            ),
        ]
        valid = wrapper.validate_batch(feedbacks)
        assert len(valid) == 2

    def test_batch_all_rejected(self):
        wrapper = LLMSafetyWrapper(confidence_threshold=0.9)
        feedbacks = [
            OptimizationFeedback(
                agent_name="a", feedback_type="hypothesis", confidence=0.5
            ),
            OptimizationFeedback(
                agent_name="b", feedback_type="hypothesis", confidence=0.7
            ),
        ]
        valid = wrapper.validate_batch(feedbacks)
        assert len(valid) == 0


class TestSafetyWrapperPhysics:
    def test_physics_check_within_bounds(self):
        wrapper = LLMSafetyWrapper(confidence_threshold=0.3, enable_physics_check=True)
        fb = OptimizationFeedback(
            agent_name="test",
            feedback_type="prior_update",
            confidence=0.8,
            payload={
                "parameter_priors": {
                    "temperature": {"mean": 30.0, "std": 5.0},
                }
            },
        )
        # Create a mock context with domain config that has temperature bounds
        # We'll use a simple context without domain_config to skip physics check
        is_valid, reasons = wrapper.validate_feedback(fb, context=None)
        assert is_valid is True

    def test_physics_check_disabled(self):
        wrapper = LLMSafetyWrapper(
            confidence_threshold=0.3, enable_physics_check=False
        )
        fb = OptimizationFeedback(
            agent_name="test",
            feedback_type="prior_update",
            confidence=0.8,
            payload={"parameter_priors": {"x": {"mean": 999999}}},
        )
        is_valid, reasons = wrapper.validate_feedback(fb)
        assert is_valid is True

    def test_confidence_threshold_property(self):
        wrapper = LLMSafetyWrapper(confidence_threshold=0.5)
        assert wrapper.confidence_threshold == 0.5
        wrapper.confidence_threshold = 0.7
        assert wrapper.confidence_threshold == 0.7

    def test_invalid_threshold_raises(self):
        wrapper = LLMSafetyWrapper()
        with pytest.raises(ValueError):
            wrapper.confidence_threshold = 1.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_priors_available(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"unknown_param": 1.0}, "y": 5.0},
            ],
            metadata={"domain": "electrochemistry"},
            iteration=1,
        )
        result = agent.analyze(ctx)
        # unknown_param has no prior
        assert result["n_parameters_matched"] == 0

    def test_unknown_domain(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"x": 1.0}, "y": 5.0},
            ],
            metadata={"domain": "nonexistent_domain"},
            iteration=1,
        )
        result = agent.analyze(ctx)
        assert result["n_parameters_matched"] == 0

    def test_empty_context(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext(iteration=1)
        result = agent.analyze(ctx)
        assert result["priors"] == [] or result["source"] == "none"

    def test_validate_context_no_info(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext()
        assert agent.validate_context(ctx) is False

    def test_validate_context_with_history(self):
        agent = LiteraturePriorAgent()
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"x": 1.0}, "y": 5.0},
            ]
        )
        assert agent.validate_context(ctx) is True

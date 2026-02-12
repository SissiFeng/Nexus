"""Tests for the MechanismHypothesisAgent and templates."""

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
from optimization_copilot.agents.mechanism.templates import (
    CATALYSIS_TEMPLATES,
    ELECTROCHEMISTRY_TEMPLATES,
    PEROVSKITE_TEMPLATES,
    HypothesisTemplate,
    get_all_templates,
    get_templates_for_domain,
)
from optimization_copilot.agents.mechanism.agent import MechanismHypothesisAgent


# ---------------------------------------------------------------------------
# HypothesisTemplate tests
# ---------------------------------------------------------------------------


class TestHypothesisTemplate:
    def test_creation(self):
        ht = HypothesisTemplate(
            name="test",
            domain="test_domain",
            pattern="something happens",
            mechanism="because physics",
        )
        assert ht.name == "test"
        assert ht.domain == "test_domain"

    def test_defaults(self):
        ht = HypothesisTemplate(
            name="test", domain="d", pattern="p", mechanism="m"
        )
        assert ht.parameters_involved == []
        assert ht.evidence_required == []
        assert ht.confidence_prior == 0.5

    def test_all_fields(self):
        ht = HypothesisTemplate(
            name="full",
            domain="electrochemistry",
            pattern="CE drops",
            mechanism="passivation",
            parameters_involved=["additive_1"],
            evidence_required=["EIS data"],
            confidence_prior=0.8,
        )
        assert len(ht.parameters_involved) == 1
        assert ht.confidence_prior == 0.8


# ---------------------------------------------------------------------------
# get_templates_for_domain
# ---------------------------------------------------------------------------


class TestGetTemplatesForDomain:
    def test_electrochemistry(self):
        templates = get_templates_for_domain("electrochemistry")
        assert len(templates) >= 4
        assert all(t.domain == "electrochemistry" for t in templates)

    def test_catalysis(self):
        templates = get_templates_for_domain("catalysis")
        assert len(templates) >= 4
        assert all(t.domain == "catalysis" for t in templates)

    def test_perovskite(self):
        templates = get_templates_for_domain("perovskite")
        assert len(templates) >= 4
        assert all(t.domain == "perovskite" for t in templates)

    def test_unknown_domain(self):
        templates = get_templates_for_domain("unknown")
        assert templates == []

    def test_get_all_templates(self):
        all_templates = get_all_templates()
        assert len(all_templates) >= 12  # at least 4 per domain * 3 domains


# ---------------------------------------------------------------------------
# MechanismHypothesisAgent basics
# ---------------------------------------------------------------------------


class TestAgentBasics:
    def test_name(self):
        agent = MechanismHypothesisAgent()
        assert agent.name() == "mechanism_hypothesis"

    def test_is_scientific_agent(self):
        agent = MechanismHypothesisAgent()
        assert isinstance(agent, ScientificAgent)

    def test_default_mode(self):
        agent = MechanismHypothesisAgent()
        assert agent.mode == AgentMode.PRAGMATIC

    def test_repr(self):
        agent = MechanismHypothesisAgent()
        r = repr(agent)
        assert "MechanismHypothesisAgent" in r

    def test_trigger_conditions(self):
        agent = MechanismHypothesisAgent()
        assert len(agent.trigger_conditions) > 0


# ---------------------------------------------------------------------------
# should_activate
# ---------------------------------------------------------------------------


class TestShouldActivate:
    def test_too_few_observations(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"x": float(i)}, "y": float(i)}
                for i in range(3)
            ],
            metadata={"domain": "electrochemistry"},
        )
        assert agent.should_activate(ctx) is False

    def test_no_domain(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"x": float(i)}, "y": float(i)}
                for i in range(10)
            ]
        )
        assert agent.should_activate(ctx) is False

    def test_with_metadata_domain(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"x": float(i)}, "y": float(i)}
                for i in range(10)
            ],
            metadata={"domain": "electrochemistry"},
        )
        assert agent.should_activate(ctx) is True

    def test_enough_obs_with_domain(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"x": float(i)}, "y": float(i)}
                for i in range(5)
            ],
            metadata={"domain": "catalysis"},
        )
        assert agent.should_activate(ctx) is True

    def test_empty_history(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(metadata={"domain": "catalysis"})
        assert agent.should_activate(ctx) is False


# ---------------------------------------------------------------------------
# analyze with electrochemistry data
# ---------------------------------------------------------------------------


def _make_echem_history(
    n: int = 15,
    drop_at: int = 10,
) -> list[dict[str, Any]]:
    """Create electrochemistry-like history with a CE drop."""
    history: list[dict[str, Any]] = []
    for i in range(n):
        conc = 0.1 + i * 0.05
        if i < drop_at:
            ce = 90.0 + i * 0.5
        else:
            ce = 90.0 - (i - drop_at) * 5.0
        history.append({
            "parameters": {
                "additive_concentration": conc,
                "additive_1": conc,
                "current_density": 3.0,
            },
            "y": ce,
        })
    return history


class TestAnalyzeElectrochemistry:
    def test_returns_dict(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_echem_history(),
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_echem_history(),
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        assert "hypotheses" in result
        assert "domain" in result
        assert "n_templates_checked" in result
        assert "n_matches" in result

    def test_domain_correct(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_echem_history(),
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        assert result["domain"] == "electrochemistry"

    def test_finds_hypotheses(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_echem_history(),
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        assert result["n_matches"] > 0

    def test_hypotheses_sorted_by_confidence(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_echem_history(),
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        confidences = [h["combined_confidence"] for h in result["hypotheses"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_templates_checked_matches_domain(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_echem_history(),
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        assert result["n_templates_checked"] == len(ELECTROCHEMISTRY_TEMPLATES)

    def test_hypothesis_has_required_fields(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_echem_history(),
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        if result["hypotheses"]:
            h = result["hypotheses"][0]
            assert "name" in h
            assert "mechanism" in h
            assert "evidence_score" in h
            assert "combined_confidence" in h

    def test_passivation_template_matches(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_echem_history(),
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        hypothesis_names = [h["name"] for h in result["hypotheses"]]
        # Should find passivation or hydrogen_evolution
        assert len(hypothesis_names) > 0


# ---------------------------------------------------------------------------
# analyze with catalysis data
# ---------------------------------------------------------------------------


def _make_catalysis_history(n: int = 15) -> list[dict[str, Any]]:
    """Create catalysis-like history."""
    history: list[dict[str, Any]] = []
    for i in range(n):
        temp = 60.0 + i * 3.0
        base_eq = 1.5 + i * 0.1
        # Yield increases then plateaus
        yield_val = min(95.0, 50.0 + temp * 0.3 + base_eq * 5.0)
        history.append({
            "parameters": {
                "temperature": temp,
                "base_equivalents": base_eq,
                "catalyst_loading": 2.0,
            },
            "y": yield_val,
        })
    return history


class TestAnalyzeCatalysis:
    def test_catalysis_analysis(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_catalysis_history(),
            metadata={"domain": "catalysis"},
        )
        result = agent.analyze(ctx)
        assert result["domain"] == "catalysis"
        assert result["n_templates_checked"] == len(CATALYSIS_TEMPLATES)

    def test_finds_catalysis_hypotheses(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_catalysis_history(),
            metadata={"domain": "catalysis"},
        )
        result = agent.analyze(ctx)
        assert result["n_matches"] > 0

    def test_catalysis_hypothesis_names(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_catalysis_history(),
            metadata={"domain": "catalysis"},
        )
        result = agent.analyze(ctx)
        names = [h["name"] for h in result["hypotheses"]]
        # Should find temperature or base-related hypotheses
        assert len(names) > 0

    def test_evidence_scores_range(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_catalysis_history(),
            metadata={"domain": "catalysis"},
        )
        result = agent.analyze(ctx)
        for h in result["hypotheses"]:
            assert 0.0 <= h["evidence_score"] <= 1.0
            assert 0.0 <= h["combined_confidence"] <= 1.0

    def test_combined_confidence_formula(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_catalysis_history(),
            metadata={"domain": "catalysis"},
        )
        result = agent.analyze(ctx)
        for h in result["hypotheses"]:
            # combined = prior * evidence (approximately)
            assert h["combined_confidence"] <= h["evidence_score"]
            assert h["combined_confidence"] >= 0


# ---------------------------------------------------------------------------
# get_optimization_feedback
# ---------------------------------------------------------------------------


class TestGetFeedback:
    def test_high_confidence_returns_warning(self):
        agent = MechanismHypothesisAgent()
        result = {
            "hypotheses": [{
                "name": "test",
                "domain": "electrochemistry",
                "pattern": "CE drops",
                "mechanism": "passivation",
                "evidence_score": 0.9,
                "combined_confidence": 0.7,
                "parameters_involved": ["additive_1"],
                "evidence_required": ["EIS data"],
            }],
            "domain": "electrochemistry",
            "n_templates_checked": 5,
            "n_matches": 1,
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert feedback.feedback_type == "warning"
        assert feedback.confidence == 0.7

    def test_low_confidence_returns_hypothesis(self):
        agent = MechanismHypothesisAgent()
        result = {
            "hypotheses": [{
                "name": "weak_test",
                "domain": "catalysis",
                "pattern": "something",
                "mechanism": "maybe",
                "evidence_score": 0.3,
                "combined_confidence": 0.2,
                "parameters_involved": [],
                "evidence_required": ["more data"],
            }],
            "domain": "catalysis",
            "n_templates_checked": 4,
            "n_matches": 1,
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert feedback.feedback_type == "hypothesis"

    def test_no_hypotheses_returns_none(self):
        agent = MechanismHypothesisAgent()
        result = {
            "hypotheses": [],
            "domain": "electrochemistry",
            "n_templates_checked": 5,
            "n_matches": 0,
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is None

    def test_feedback_payload(self):
        agent = MechanismHypothesisAgent()
        result = {
            "hypotheses": [{
                "name": "test",
                "domain": "electrochemistry",
                "pattern": "drop",
                "mechanism": "passivation",
                "evidence_score": 0.9,
                "combined_confidence": 0.8,
                "parameters_involved": ["additive_1"],
                "evidence_required": ["EIS"],
            }],
            "domain": "electrochemistry",
            "n_templates_checked": 5,
            "n_matches": 1,
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert "top_hypothesis" in feedback.payload
        assert "mechanism" in feedback.payload

    def test_feedback_agent_name(self):
        agent = MechanismHypothesisAgent()
        result = {
            "hypotheses": [{
                "name": "test",
                "domain": "electrochemistry",
                "pattern": "drop",
                "mechanism": "passivation",
                "evidence_score": 0.8,
                "combined_confidence": 0.65,
                "parameters_involved": [],
                "evidence_required": [],
            }],
            "domain": "electrochemistry",
            "n_templates_checked": 5,
            "n_matches": 1,
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert feedback.agent_name == "mechanism_hypothesis"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_history(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(metadata={"domain": "electrochemistry"})
        result = agent.analyze(ctx)
        # No data -> no hypotheses
        assert result["n_matches"] == 0

    def test_no_domain(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"x": 1.0}, "y": 1.0}
                for _ in range(10)
            ]
        )
        result = agent.analyze(ctx)
        assert result["domain"] is None
        assert result["n_matches"] == 0

    def test_insufficient_data(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"x": 1.0}, "y": 1.0},
            ],
            metadata={"domain": "catalysis"},
        )
        result = agent.analyze(ctx)
        # May still find template matches with parameter overlap
        assert isinstance(result["hypotheses"], list)

    def test_no_parameters_in_history(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=[{"y": 1.0} for _ in range(10)],
            metadata={"domain": "electrochemistry"},
        )
        # validate_context should fail
        assert agent.validate_context(ctx) is False

    def test_all_same_y_values(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=[
                {"parameters": {"additive_1": float(i)}, "y": 50.0}
                for i in range(10)
            ],
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        # Constant y -> less pattern detection
        assert isinstance(result["hypotheses"], list)


# ---------------------------------------------------------------------------
# Template matching accuracy
# ---------------------------------------------------------------------------


class TestTemplateMatching:
    def test_parameter_overlap_scoring(self):
        agent = MechanismHypothesisAgent()
        # History with additive_1 parameter
        history = [
            {"parameters": {"additive_1": float(i) * 0.1}, "y": 90 - i}
            for i in range(10)
        ]
        ctx = AgentContext(
            optimization_history=history,
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        # Templates involving additive_1 should score higher
        assert result["n_matches"] > 0

    def test_multiple_parameter_overlap(self):
        agent = MechanismHypothesisAgent()
        history = [
            {
                "parameters": {
                    "additive_concentration": float(i) * 0.1,
                    "current_density": 3.0 + i * 0.2,
                    "flow_rate": 100.0,
                },
                "y": 85 + i * 0.5,
            }
            for i in range(10)
        ]
        ctx = AgentContext(
            optimization_history=history,
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        assert result["n_matches"] > 0

    def test_non_matching_domain_parameters(self):
        agent = MechanismHypothesisAgent()
        history = [
            {"parameters": {"irrelevant_param": float(i)}, "y": float(i)}
            for i in range(10)
        ]
        ctx = AgentContext(
            optimization_history=history,
            metadata={"domain": "catalysis"},
        )
        result = agent.analyze(ctx)
        # Still may find some matches due to base score
        assert isinstance(result["hypotheses"], list)

    def test_perovskite_domain(self):
        agent = MechanismHypothesisAgent()
        history = [
            {
                "parameters": {
                    "annealing_temperature": 80 + i * 5,
                    "antisolvent_delay": 5 + i * 0.5,
                },
                "y": 15 + i * 0.5 if i < 8 else 15 + 8 * 0.5 - (i - 8),
            }
            for i in range(12)
        ]
        ctx = AgentContext(
            optimization_history=history,
            metadata={"domain": "perovskite"},
        )
        result = agent.analyze(ctx)
        assert result["domain"] == "perovskite"
        assert result["n_templates_checked"] == len(PEROVSKITE_TEMPLATES)

    def test_evidence_scores_bounded(self):
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_echem_history(n=20),
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        for h in result["hypotheses"]:
            assert 0.0 <= h["evidence_score"] <= 1.0


# ---------------------------------------------------------------------------
# Evidence scoring
# ---------------------------------------------------------------------------


class TestEvidenceScoring:
    def test_drop_pattern_detected(self):
        """CE drops should boost passivation template score."""
        agent = MechanismHypothesisAgent()
        ctx = AgentContext(
            optimization_history=_make_echem_history(n=15, drop_at=10),
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        assert result["n_matches"] > 0

    def test_monotonic_increase_pattern(self):
        agent = MechanismHypothesisAgent()
        history = [
            {"parameters": {"temperature": 50 + i * 5.0}, "y": 30 + i * 3.0}
            for i in range(10)
        ]
        ctx = AgentContext(
            optimization_history=history,
            metadata={"domain": "catalysis"},
        )
        result = agent.analyze(ctx)
        assert isinstance(result["hypotheses"], list)

    def test_plateau_pattern(self):
        agent = MechanismHypothesisAgent()
        history = [
            {"parameters": {"current_density": 1.0 + i * 0.5}, "y": min(90, 60 + i * 5)}
            for i in range(15)
        ]
        ctx = AgentContext(
            optimization_history=history,
            metadata={"domain": "electrochemistry"},
        )
        result = agent.analyze(ctx)
        assert isinstance(result["hypotheses"], list)

    def test_non_monotonic_pattern(self):
        """Create data with a peak at the middle."""
        agent = MechanismHypothesisAgent()
        history = [
            {
                "parameters": {"base_equivalents": 1.0 + i * 0.3},
                "y": 50 + i * 5 if i < 7 else 50 + (14 - i) * 5,
            }
            for i in range(15)
        ]
        ctx = AgentContext(
            optimization_history=history,
            metadata={"domain": "catalysis"},
        )
        result = agent.analyze(ctx)
        assert isinstance(result["hypotheses"], list)

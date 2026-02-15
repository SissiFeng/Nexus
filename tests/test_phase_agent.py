"""Tests for the PhaseStructureAgent and ReferenceDB."""

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
from optimization_copilot.agents.phase_structure.reference_db import (
    REFERENCE_PEAKS,
    ReferenceDB,
)
from optimization_copilot.agents.phase_structure.agent import PhaseStructureAgent


# ---------------------------------------------------------------------------
# ReferenceDB tests
# ---------------------------------------------------------------------------


class TestReferenceDBGetReference:
    def test_get_known_phase(self):
        db = ReferenceDB()
        ref = db.get_reference("Zn")
        assert ref is not None
        assert ref["crystal_system"] == "hexagonal"

    def test_get_unknown_phase(self):
        db = ReferenceDB()
        ref = db.get_reference("UnknownPhase")
        assert ref is None

    def test_get_zno(self):
        db = ReferenceDB()
        ref = db.get_reference("ZnO")
        assert ref is not None
        assert "peaks_2theta" in ref
        assert len(ref["peaks_2theta"]) > 0

    def test_get_znoh2(self):
        db = ReferenceDB()
        ref = db.get_reference("ZnOH2")
        assert ref is not None
        assert ref["crystal_system"] == "orthorhombic"

    def test_get_znso4(self):
        db = ReferenceDB()
        ref = db.get_reference("ZnSO4")
        assert ref is not None
        assert ref["space_group"] == "Pnma"


class TestReferenceDBListPhases:
    def test_list_phases(self):
        db = ReferenceDB()
        phases = db.list_phases()
        assert "Zn" in phases
        assert "ZnO" in phases
        assert len(phases) == 4

    def test_list_is_sorted(self):
        db = ReferenceDB()
        phases = db.list_phases()
        assert phases == sorted(phases)

    def test_custom_db(self):
        db = ReferenceDB(references={"Custom": {"peaks_2theta": [10.0]}})
        phases = db.list_phases()
        assert phases == ["Custom"]


class TestReferenceDBMatchPeaks:
    def test_match_zn_peaks(self):
        db = ReferenceDB()
        # Use exact Zn reference peaks
        zn_ref = REFERENCE_PEAKS["Zn"]["peaks_2theta"]
        matches = db.match_peaks(zn_ref)
        assert len(matches) > 0
        # Zn should be the best match
        best = matches[0]
        assert best[0] == "Zn"
        assert best[2] > 0.8  # High score

    def test_match_zno_peaks(self):
        db = ReferenceDB()
        zno_ref = REFERENCE_PEAKS["ZnO"]["peaks_2theta"]
        matches = db.match_peaks(zno_ref)
        # ZnO should be among top matches
        phase_names = [m[0] for m in matches]
        assert "ZnO" in phase_names

    def test_match_empty_peaks(self):
        db = ReferenceDB()
        matches = db.match_peaks([])
        assert matches == []

    def test_match_no_matching_peaks(self):
        db = ReferenceDB()
        matches = db.match_peaks([1.0, 2.0, 3.0])
        # No reference should match peaks at 1, 2, 3 degrees
        assert len(matches) == 0

    def test_tolerance_effect(self):
        db = ReferenceDB()
        # Shifted peaks
        shifted = [p + 0.2 for p in REFERENCE_PEAKS["Zn"]["peaks_2theta"]]
        matches_tight = db.match_peaks(shifted, tolerance=0.1)
        matches_loose = db.match_peaks(shifted, tolerance=0.5)
        # Loose tolerance should find more matches
        tight_zn = [m for m in matches_tight if m[0] == "Zn"]
        loose_zn = [m for m in matches_loose if m[0] == "Zn"]
        if tight_zn and loose_zn:
            assert loose_zn[0][2] >= tight_zn[0][2]

    def test_match_returns_sorted_by_score(self):
        db = ReferenceDB()
        zn_ref = REFERENCE_PEAKS["Zn"]["peaks_2theta"]
        matches = db.match_peaks(zn_ref)
        scores = [m[2] for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_match_score_range(self):
        db = ReferenceDB()
        zn_ref = REFERENCE_PEAKS["Zn"]["peaks_2theta"]
        matches = db.match_peaks(zn_ref)
        for _, _, score in matches:
            assert 0.0 <= score <= 1.0


class TestReferenceDBTextureCoefficient:
    def test_tc_zn_002_100(self):
        db = ReferenceDB()
        # Construct observed data with Zn peaks
        zn_ref = REFERENCE_PEAKS["Zn"]
        peaks = zn_ref["peaks_2theta"]
        # Give (002) double the expected intensity relative to (100)
        intensities = list(zn_ref["relative_intensity"])
        tc = db.get_texture_coefficient(peaks, intensities, "Zn")
        assert tc is not None
        # With reference intensities, TC should be ~1.0
        assert abs(tc - 1.0) < 0.01

    def test_tc_preferred_orientation(self):
        db = ReferenceDB()
        zn_ref = REFERENCE_PEAKS["Zn"]
        peaks = zn_ref["peaks_2theta"]
        intensities = list(zn_ref["relative_intensity"])
        # Enhance (002) peak (index 0, 2theta=36.3): double it
        intensities[0] = 200  # was 100
        tc = db.get_texture_coefficient(peaks, intensities, "Zn")
        assert tc is not None
        assert tc > 1.5  # Enhanced (002) orientation

    def test_tc_unknown_phase(self):
        db = ReferenceDB()
        tc = db.get_texture_coefficient([36.3], [100], "Unknown")
        assert tc is None

    def test_tc_missing_peaks(self):
        db = ReferenceDB()
        # Peaks that don't include (100) for Zn
        tc = db.get_texture_coefficient([36.3], [100], "Zn")
        # (100) is at 39.0, not in our observed peaks
        assert tc is None

    def test_tc_mismatched_lengths(self):
        db = ReferenceDB()
        tc = db.get_texture_coefficient([36.3, 39.0], [100], "Zn")
        assert tc is None


# ---------------------------------------------------------------------------
# PhaseStructureAgent tests
# ---------------------------------------------------------------------------


class TestPhaseAgentBasics:
    def test_name(self):
        agent = PhaseStructureAgent()
        assert agent.name() == "phase_structure"

    def test_is_scientific_agent(self):
        agent = PhaseStructureAgent()
        assert isinstance(agent, ScientificAgent)

    def test_default_mode(self):
        agent = PhaseStructureAgent()
        assert agent.mode == AgentMode.PRAGMATIC

    def test_custom_tolerance(self):
        agent = PhaseStructureAgent(tolerance=0.5)
        assert agent._tolerance == 0.5

    def test_repr(self):
        agent = PhaseStructureAgent()
        r = repr(agent)
        assert "PhaseStructureAgent" in r


class TestShouldActivate:
    def test_no_raw_data(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext()
        assert agent.should_activate(ctx) is False

    def test_with_xrd_key(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(raw_data={"xrd": {"two_theta": [36.3]}})
        assert agent.should_activate(ctx) is True

    def test_with_peaks_key(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(raw_data={"peaks": [36.3, 39.0]})
        assert agent.should_activate(ctx) is True

    def test_with_two_theta_key(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(raw_data={"two_theta": [36.3]})
        assert agent.should_activate(ctx) is True

    def test_with_unrelated_data(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(raw_data={"voltages": [1.0, 2.0]})
        assert agent.should_activate(ctx) is False


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


class TestAnalyzeZnPeaks:
    def test_analyze_zn(self):
        agent = PhaseStructureAgent()
        zn_peaks = REFERENCE_PEAKS["Zn"]["peaks_2theta"]
        zn_ints = REFERENCE_PEAKS["Zn"]["relative_intensity"]
        ctx = AgentContext(
            raw_data={
                "two_theta": zn_peaks,
                "intensities": zn_ints,
            }
        )
        result = agent.analyze(ctx)
        assert result["n_peaks"] == len(zn_peaks)
        assert len(result["matched_phases"]) > 0
        phase_names = [m["phase"] for m in result["matched_phases"]]
        assert "Zn" in phase_names

    def test_quality_score_positive(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(
            raw_data={
                "two_theta": REFERENCE_PEAKS["Zn"]["peaks_2theta"],
                "intensities": REFERENCE_PEAKS["Zn"]["relative_intensity"],
            }
        )
        result = agent.analyze(ctx)
        assert result["quality_score"] > 0.0

    def test_texture_coefficients_present(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(
            raw_data={
                "two_theta": REFERENCE_PEAKS["Zn"]["peaks_2theta"],
                "intensities": REFERENCE_PEAKS["Zn"]["relative_intensity"],
            }
        )
        result = agent.analyze(ctx)
        assert isinstance(result["texture_coefficients"], dict)


class TestAnalyzeZnOPeaks:
    def test_analyze_zno(self):
        agent = PhaseStructureAgent()
        zno_peaks = REFERENCE_PEAKS["ZnO"]["peaks_2theta"]
        zno_ints = REFERENCE_PEAKS["ZnO"]["relative_intensity"]
        ctx = AgentContext(
            raw_data={
                "two_theta": zno_peaks,
                "intensities": zno_ints,
            }
        )
        result = agent.analyze(ctx)
        phase_names = [m["phase"] for m in result["matched_phases"]]
        assert "ZnO" in phase_names


class TestAnalyzeMixedPhases:
    def test_mixed_zn_zno(self):
        agent = PhaseStructureAgent()
        # Combine Zn and ZnO peaks
        peaks = REFERENCE_PEAKS["Zn"]["peaks_2theta"][:5] + \
                REFERENCE_PEAKS["ZnO"]["peaks_2theta"][:5]
        ctx = AgentContext(raw_data={"peaks": peaks})
        result = agent.analyze(ctx)
        phase_names = [m["phase"] for m in result["matched_phases"]]
        # Should find multiple phases
        assert len(result["matched_phases"]) >= 1


class TestAnalyzeFromPeaksDicts:
    def test_peaks_as_dicts(self):
        agent = PhaseStructureAgent()
        peaks_data = [
            {"two_theta": 36.3, "intensity": 100},
            {"two_theta": 39.0, "intensity": 28},
            {"two_theta": 43.2, "intensity": 60},
        ]
        ctx = AgentContext(raw_data={"peaks": peaks_data})
        result = agent.analyze(ctx)
        assert result["n_peaks"] == 3

    def test_xrd_subdicts(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(
            raw_data={
                "xrd": {
                    "two_theta": [36.3, 39.0, 43.2, 54.3],
                    "intensities": [100, 28, 60, 40],
                }
            }
        )
        result = agent.analyze(ctx)
        assert result["n_peaks"] == 4


# ---------------------------------------------------------------------------
# get_optimization_feedback
# ---------------------------------------------------------------------------


class TestFeedback:
    def test_with_insights(self):
        agent = PhaseStructureAgent()
        result = {
            "matched_phases": [{"phase": "Zn", "score": 0.8}],
            "texture_coefficients": {"Zn_(002)/(100)": 2.5},
            "structural_insights": ["Strong preferred orientation detected"],
            "quality_score": 0.8,
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert feedback.feedback_type == "hypothesis"
        assert feedback.confidence > 0.3

    def test_no_insights(self):
        agent = PhaseStructureAgent()
        result = {
            "matched_phases": [],
            "texture_coefficients": {},
            "structural_insights": [],
            "quality_score": 0.0,
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is None

    def test_feedback_payload_has_phases(self):
        agent = PhaseStructureAgent()
        result = {
            "matched_phases": [{"phase": "Zn", "score": 0.9}],
            "texture_coefficients": {},
            "structural_insights": ["ZnO detected"],
            "quality_score": 0.7,
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert "phases" in feedback.payload

    def test_feedback_reasoning(self):
        agent = PhaseStructureAgent()
        result = {
            "matched_phases": [{"phase": "Zn", "score": 0.9}],
            "texture_coefficients": {},
            "structural_insights": ["Important finding"],
            "quality_score": 0.7,
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert "Zn" in feedback.reasoning

    def test_feedback_confidence_range(self):
        agent = PhaseStructureAgent()
        result = {
            "matched_phases": [{"phase": "Zn", "score": 0.5}],
            "texture_coefficients": {},
            "structural_insights": ["Found something"],
            "quality_score": 0.5,
        }
        feedback = agent.get_optimization_feedback(result)
        assert feedback is not None
        assert 0.0 <= feedback.confidence <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_peaks(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(raw_data={"peaks": []})
        result = agent.analyze(ctx)
        assert result["n_peaks"] == 0
        assert result["matched_phases"] == []

    def test_no_matches(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(raw_data={"peaks": [1.0, 2.0, 3.0]})
        result = agent.analyze(ctx)
        assert result["matched_phases"] == []

    def test_single_peak(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(raw_data={"peaks": [36.3]})
        result = agent.analyze(ctx)
        # May match Zn with low score
        assert result["n_peaks"] == 1

    def test_null_raw_data(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(raw_data=None)
        result = agent.analyze(ctx)
        assert result["n_peaks"] == 0

    def test_empty_raw_data(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(raw_data={})
        result = agent.analyze(ctx)
        assert result["n_peaks"] == 0


# ---------------------------------------------------------------------------
# Texture coefficient interpretation
# ---------------------------------------------------------------------------


class TestTextureInterpretation:
    def test_high_tc_insight(self):
        agent = PhaseStructureAgent()
        zn_ref = REFERENCE_PEAKS["Zn"]
        peaks = zn_ref["peaks_2theta"]
        ints = list(zn_ref["relative_intensity"])
        # Enhance (002) to get high TC
        ints[0] = 300  # (002) much stronger
        ctx = AgentContext(
            raw_data={"two_theta": peaks, "intensities": ints}
        )
        result = agent.analyze(ctx)
        insights = result["structural_insights"]
        # Should mention preferred orientation
        assert any("orientation" in i.lower() or "texture" in i.lower() for i in insights)

    def test_zno_passivation_insight(self):
        agent = PhaseStructureAgent()
        # Use ZnO peaks
        ctx = AgentContext(
            raw_data={
                "two_theta": REFERENCE_PEAKS["ZnO"]["peaks_2theta"],
                "intensities": REFERENCE_PEAKS["ZnO"]["relative_intensity"],
            }
        )
        result = agent.analyze(ctx)
        insights = result["structural_insights"]
        assert any("passivation" in i.lower() or "zno" in i.lower() for i in insights)

    def test_znoh2_poor_quality_insight(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(
            raw_data={
                "two_theta": REFERENCE_PEAKS["ZnOH2"]["peaks_2theta"],
                "intensities": REFERENCE_PEAKS["ZnOH2"]["relative_intensity"],
            }
        )
        result = agent.analyze(ctx)
        phase_names = [m["phase"] for m in result["matched_phases"]]
        if "ZnOH2" in phase_names:
            insights = result["structural_insights"]
            assert any("poor" in i.lower() or "zn(oh)2" in i.lower() for i in insights)

    def test_znso4_inclusion_insight(self):
        agent = PhaseStructureAgent()
        ctx = AgentContext(
            raw_data={
                "two_theta": REFERENCE_PEAKS["ZnSO4"]["peaks_2theta"],
                "intensities": REFERENCE_PEAKS["ZnSO4"]["relative_intensity"],
            }
        )
        result = agent.analyze(ctx)
        phase_names = [m["phase"] for m in result["matched_phases"]]
        if "ZnSO4" in phase_names:
            insights = result["structural_insights"]
            assert any("inclusion" in i.lower() or "znso4" in i.lower() for i in insights)

    def test_trigger_conditions(self):
        agent = PhaseStructureAgent()
        assert len(agent.trigger_conditions) > 0
        names = [tc.name for tc in agent.trigger_conditions]
        assert "xrd_data_available" in names

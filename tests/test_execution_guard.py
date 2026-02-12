"""Tests for agents/execution_guard.py -- execution trace validation.

Covers:
- GuardMode enum values
- Claim detection from payload keys
- Trace matching logic
- STRICT mode validation and enforcement
- LENIENT mode validation and enforcement
- Custom requirements
- Integration with pipeline TracedResult flow
- Robustness with malformed payloads
"""

from __future__ import annotations

import pytest

from optimization_copilot.agents.base import OptimizationFeedback
from optimization_copilot.agents.execution_guard import (
    CLAIM_TRACE_REQUIREMENTS,
    ExecutionGuard,
    GuardMode,
)
from optimization_copilot.agents.execution_trace import (
    ExecutionTag,
    ExecutionTrace,
    TracedResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace_dict(
    method: str = "run_top_k",
    module: str = "data_pipeline",
    tag: str = "computed",
    **extra: object,
) -> dict:
    """Build a minimal trace dict suitable for embedding in a payload."""
    d = {
        "module": module,
        "method": method,
        "input_summary": {},
        "output_summary": {},
        "tag": tag,
        "timestamp": 1000.0,
        "duration_ms": 3.5,
        "error": None,
    }
    d.update(extra)
    return d


def _make_feedback(
    payload: dict | None = None,
    agent_name: str = "test_agent",
    feedback_type: str = "hypothesis",
    confidence: float = 0.8,
    reasoning: str = "test reasoning",
) -> OptimizationFeedback:
    """Build an OptimizationFeedback with the given payload."""
    return OptimizationFeedback(
        agent_name=agent_name,
        feedback_type=feedback_type,
        confidence=confidence,
        payload=payload if payload is not None else {},
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# TestGuardMode
# ---------------------------------------------------------------------------


class TestGuardMode:
    """Verify enum values and behaviour."""

    def test_strict_value(self):
        assert GuardMode.STRICT.value == "strict"

    def test_lenient_value(self):
        assert GuardMode.LENIENT.value == "lenient"

    def test_is_str_subclass(self):
        assert isinstance(GuardMode.STRICT, str)
        assert isinstance(GuardMode.LENIENT, str)

    def test_from_string(self):
        assert GuardMode("strict") is GuardMode.STRICT
        assert GuardMode("lenient") is GuardMode.LENIENT


# ---------------------------------------------------------------------------
# TestClaimDetection
# ---------------------------------------------------------------------------


class TestClaimDetection:
    """Test _detect_quantitative_claims identifies the right payload keys."""

    def setup_method(self):
        self.guard = ExecutionGuard(mode=GuardMode.STRICT)

    def test_top_5_polymers_triggers_top_claim(self):
        payload = {"top_5_polymers": [1, 2, 3]}
        claims = self.guard._detect_quantitative_claims(payload)
        assert len(claims) == 1
        key, claim_type = claims[0]
        assert key == "top_5_polymers"
        # "top_" is a substring of the key, so it should match
        assert "top_" in claim_type

    def test_pareto_front_triggers_pareto_front_claim(self):
        payload = {"pareto_front": [{"x": 1}]}
        claims = self.guard._detect_quantitative_claims(payload)
        assert len(claims) == 1
        key, claim_type = claims[0]
        assert key == "pareto_front"
        assert claim_type == "pareto_front"

    def test_pearson_correlation_triggers_correlation_claim(self):
        payload = {"pearson_correlation": 0.95}
        claims = self.guard._detect_quantitative_claims(payload)
        assert len(claims) == 1
        key, claim_type = claims[0]
        assert key == "pearson_correlation"
        # Should match one of the correlation-related claim types
        assert claim_type in (
            "pearson", "correlation", "pearson_correlation",
        )

    def test_underscore_keys_are_skipped(self):
        payload = {
            "_execution_traces": [],
            "_execution_tag": "computed",
            "_internal_data": [1, 2, 3],
        }
        claims = self.guard._detect_quantitative_claims(payload)
        assert claims == []

    def test_non_matching_keys_ignored(self):
        payload = {
            "agent_config": {"mode": "fast"},
            "description": "some text",
            "iteration_count": 42,
        }
        claims = self.guard._detect_quantitative_claims(payload)
        assert claims == []

    def test_outlier_key_detected(self):
        payload = {"outlier_scores": [0.1, 0.9, 0.3]}
        claims = self.guard._detect_quantitative_claims(payload)
        assert len(claims) == 1
        assert claims[0][1] == "outlier"

    def test_fanova_key_detected(self):
        payload = {"fanova_results": {"x1": 0.6}}
        claims = self.guard._detect_quantitative_claims(payload)
        assert len(claims) == 1
        assert claims[0][1] == "fanova"

    def test_main_effect_key_detected(self):
        payload = {"main_effect_x1": 0.42}
        claims = self.guard._detect_quantitative_claims(payload)
        assert len(claims) == 1
        assert claims[0][1] == "main_effect"

    def test_best_prefix_detected(self):
        payload = {"best_candidate": {"name": "A", "score": 0.99}}
        claims = self.guard._detect_quantitative_claims(payload)
        assert len(claims) == 1
        assert claims[0][1] == "best_"

    def test_screening_key_detected(self):
        payload = {"screening_results": [1, 2]}
        claims = self.guard._detect_quantitative_claims(payload)
        assert len(claims) == 1
        assert claims[0][1] == "screening"

    def test_molecular_key_detected(self):
        payload = {"molecular_descriptors": [0.1, 0.2]}
        claims = self.guard._detect_quantitative_claims(payload)
        assert len(claims) == 1
        assert claims[0][1] == "molecular"

    def test_multiple_claims_in_one_payload(self):
        payload = {
            "top_3_materials": [1, 2, 3],
            "pareto_front": [],
            "outlier_report": {},
        }
        claims = self.guard._detect_quantitative_claims(payload)
        assert len(claims) == 3
        claim_types = {ct for _, ct in claims}
        # Each key should match a distinct claim type
        assert len(claim_types) == 3

    def test_case_insensitive_matching(self):
        payload = {"Top_3_Polymers": [1, 2]}
        claims = self.guard._detect_quantitative_claims(payload)
        assert len(claims) == 1

    def test_longest_match_wins(self):
        """'pareto_front' should match 'pareto_front' not just 'pareto'."""
        payload = {"pareto_front": []}
        claims = self.guard._detect_quantitative_claims(payload)
        assert len(claims) == 1
        assert claims[0][1] == "pareto_front"


# ---------------------------------------------------------------------------
# TestTraceMatching
# ---------------------------------------------------------------------------


class TestTraceMatching:
    """Test _find_matching_trace logic."""

    def setup_method(self):
        self.guard = ExecutionGuard(mode=GuardMode.STRICT)

    def test_matching_trace_found(self):
        traces = [_make_trace_dict(method="run_top_k")]
        assert self.guard._find_matching_trace("top_k", traces) is True

    def test_matching_trace_found_alternate_method(self):
        traces = [_make_trace_dict(method="run_ranking")]
        assert self.guard._find_matching_trace("top_k", traces) is True

    def test_no_matching_trace(self):
        traces = [_make_trace_dict(method="run_diagnostics")]
        assert self.guard._find_matching_trace("top_k", traces) is False

    def test_empty_traces_list(self):
        assert self.guard._find_matching_trace("top_k", []) is False

    def test_method_substring_matching(self):
        """Trace method 'DataAnalysisPipeline.run_top_k' should still match."""
        traces = [_make_trace_dict(method="DataAnalysisPipeline.run_top_k")]
        assert self.guard._find_matching_trace("top_k", traces) is True

    def test_pareto_trace_matches_pareto_front_claim(self):
        traces = [_make_trace_dict(method="run_pareto_analysis")]
        assert self.guard._find_matching_trace("pareto_front", traces) is True

    def test_correlation_trace_matches(self):
        traces = [_make_trace_dict(method="run_correlation")]
        assert self.guard._find_matching_trace("correlation", traces) is True
        assert self.guard._find_matching_trace("pearson", traces) is True

    def test_unknown_claim_type_returns_false(self):
        traces = [_make_trace_dict(method="run_top_k")]
        assert self.guard._find_matching_trace("nonexistent_claim", traces) is False

    def test_malformed_trace_dict_is_skipped(self):
        """Trace dicts without 'method' should be silently skipped."""
        traces = [{"module": "test"}, {"not_method": "foo"}]
        assert self.guard._find_matching_trace("top_k", traces) is False

    def test_non_string_method_is_skipped(self):
        traces = [{"method": 12345}]
        assert self.guard._find_matching_trace("top_k", traces) is False

    def test_multiple_traces_one_matches(self):
        traces = [
            _make_trace_dict(method="run_diagnostics"),
            _make_trace_dict(method="run_top_k"),
        ]
        assert self.guard._find_matching_trace("top_k", traces) is True


# ---------------------------------------------------------------------------
# TestStrictMode
# ---------------------------------------------------------------------------


class TestStrictMode:
    """Test STRICT mode validation and enforcement."""

    def setup_method(self):
        self.guard = ExecutionGuard(mode=GuardMode.STRICT)

    def test_traced_claims_pass(self):
        """Feedback with proper traces for all claims is valid."""
        payload = {
            "top_3_polymers": [
                {"name": "A", "value": 0.9},
                {"name": "B", "value": 0.8},
                {"name": "C", "value": 0.7},
            ],
            "_execution_traces": [_make_trace_dict(method="run_top_k")],
            "_execution_tag": "computed",
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = self.guard.validate_feedback(fb)
        assert is_valid is True
        assert issues == []

    def test_untraced_claims_fail(self):
        """Feedback with quantitative claims but no traces fails."""
        payload = {
            "top_5_polymers": [1, 2, 3, 4, 5],
            # No _execution_traces at all
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = self.guard.validate_feedback(fb)
        assert is_valid is False
        assert len(issues) == 1
        assert "top_5_polymers" in issues[0]

    def test_no_quantitative_claims_passes(self):
        """Feedback without any quantitative keywords is always valid."""
        payload = {
            "agent_notes": "everything is fine",
            "config": {"mode": "turbo"},
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = self.guard.validate_feedback(fb)
        assert is_valid is True
        assert issues == []

    def test_empty_payload_passes(self):
        fb = _make_feedback(payload={})
        is_valid, issues = self.guard.validate_feedback(fb)
        assert is_valid is True
        assert issues == []

    def test_wrong_trace_for_claim_fails(self):
        """A trace for diagnostics does not satisfy a top_k claim."""
        payload = {
            "top_3_materials": [1, 2, 3],
            "_execution_traces": [_make_trace_dict(method="run_diagnostics")],
            "_execution_tag": "computed",
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = self.guard.validate_feedback(fb)
        assert is_valid is False
        assert len(issues) == 1

    def test_multiple_claims_partial_trace_fails(self):
        """One traced claim + one untraced claim = invalid."""
        payload = {
            "top_3_materials": [1, 2, 3],
            "outlier_report": {"outliers": ["X"]},
            "_execution_traces": [_make_trace_dict(method="run_top_k")],
            "_execution_tag": "computed",
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = self.guard.validate_feedback(fb)
        assert is_valid is False
        assert len(issues) == 1
        assert "outlier" in issues[0].lower()

    def test_multiple_claims_all_traced_passes(self):
        """All claims traced = valid."""
        payload = {
            "top_3_materials": [1, 2, 3],
            "outlier_report": {"outliers": []},
            "_execution_traces": [
                _make_trace_dict(method="run_top_k"),
                _make_trace_dict(method="run_outlier_detection"),
            ],
            "_execution_tag": "computed",
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = self.guard.validate_feedback(fb)
        assert is_valid is True
        assert issues == []

    def test_enforce_raises_for_untraced_claims(self):
        """enforce() in STRICT mode raises ValueError."""
        payload = {
            "top_5_polymers": [1, 2, 3, 4, 5],
        }
        fb = _make_feedback(payload=payload)
        with pytest.raises(ValueError, match="STRICT"):
            self.guard.enforce(fb)

    def test_enforce_returns_feedback_when_valid(self):
        """enforce() returns the original feedback when all claims are traced."""
        payload = {
            "top_3": [1, 2, 3],
            "_execution_traces": [_make_trace_dict(method="run_top_k")],
            "_execution_tag": "computed",
        }
        fb = _make_feedback(payload=payload)
        result = self.guard.enforce(fb)
        assert result is fb  # Same object, not modified

    def test_enforce_returns_clean_feedback_when_no_claims(self):
        """enforce() with no quantitative claims returns the same feedback."""
        payload = {"notes": "nothing quantitative here"}
        fb = _make_feedback(payload=payload)
        result = self.guard.enforce(fb)
        assert result is fb


# ---------------------------------------------------------------------------
# TestLenientMode
# ---------------------------------------------------------------------------


class TestLenientMode:
    """Test LENIENT mode validation and enforcement."""

    def setup_method(self):
        self.guard = ExecutionGuard(mode=GuardMode.LENIENT)

    def test_always_valid_even_with_untraced_claims(self):
        payload = {
            "top_5_polymers": [1, 2, 3, 4, 5],
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = self.guard.validate_feedback(fb)
        assert is_valid is True
        assert len(issues) == 1

    def test_issues_populated_for_untraced_claims(self):
        payload = {
            "top_5": [1, 2, 3],
            "pareto_front": [],
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = self.guard.validate_feedback(fb)
        assert is_valid is True
        assert len(issues) == 2

    def test_no_issues_when_fully_traced(self):
        payload = {
            "top_3": [1, 2, 3],
            "_execution_traces": [_make_trace_dict(method="run_top_k")],
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = self.guard.validate_feedback(fb)
        assert is_valid is True
        assert issues == []

    def test_enforce_wraps_numeric_value(self):
        payload = {
            "correlation": 0.95,
        }
        fb = _make_feedback(payload=payload)
        result = self.guard.enforce(fb)
        assert "[ESTIMATED" in str(result.payload["correlation"])

    def test_enforce_wraps_list_value(self):
        payload = {
            "top_5_polymers": [1, 2, 3, 4, 5],
        }
        fb = _make_feedback(payload=payload)
        result = self.guard.enforce(fb)
        tagged_val = result.payload["top_5_polymers"]
        assert "[ESTIMATED" in str(tagged_val)

    def test_enforce_wraps_string_value(self):
        payload = {
            "best_equation": "y = 2*x + 1",
        }
        fb = _make_feedback(payload=payload)
        result = self.guard.enforce(fb)
        assert result.payload["best_equation"].startswith("[ESTIMATED")

    def test_enforce_wraps_dict_value(self):
        payload = {
            "outlier_info": {"name": "X", "z_score": 3.1},
        }
        fb = _make_feedback(payload=payload)
        result = self.guard.enforce(fb)
        tagged = result.payload["outlier_info"]
        assert isinstance(tagged, dict)
        assert tagged.get("_estimated") is True

    def test_enforce_preserves_metadata_keys(self):
        """Metadata keys (_execution_traces, etc.) should be untouched."""
        traces = [_make_trace_dict(method="run_diagnostics")]
        payload = {
            "top_5": [1, 2, 3],
            "_execution_traces": traces,
            "_execution_tag": "computed",
        }
        fb = _make_feedback(payload=payload)
        result = self.guard.enforce(fb)
        # Metadata keys are preserved.
        assert result.payload["_execution_traces"] == traces
        assert result.payload["_execution_tag"] == "computed"
        # The untraced claim is tagged.
        assert "[ESTIMATED" in str(result.payload["top_5"])

    def test_enforce_does_not_modify_original_feedback(self):
        payload = {"top_5": [1, 2, 3]}
        fb = _make_feedback(payload=payload)
        _ = self.guard.enforce(fb)
        # Original payload should be unchanged.
        assert fb.payload["top_5"] == [1, 2, 3]

    def test_enforce_returns_new_feedback_object(self):
        payload = {"top_5": [1, 2, 3]}
        fb = _make_feedback(payload=payload)
        result = self.guard.enforce(fb)
        assert result is not fb
        assert result.agent_name == fb.agent_name
        assert result.feedback_type == fb.feedback_type
        assert result.confidence == fb.confidence
        assert result.reasoning == fb.reasoning


# ---------------------------------------------------------------------------
# TestCustomRequirements
# ---------------------------------------------------------------------------


class TestCustomRequirements:
    """Test adding custom claim -> trace requirements."""

    def test_custom_requirement_detected_and_validated(self):
        custom = {"polymer_score": ["run_polymer_evaluation"]}
        guard = ExecutionGuard(
            mode=GuardMode.STRICT, custom_requirements=custom
        )
        payload = {
            "polymer_score_top3": [0.9, 0.8, 0.7],
            "_execution_traces": [
                _make_trace_dict(method="run_polymer_evaluation")
            ],
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = guard.validate_feedback(fb)
        assert is_valid is True
        assert issues == []

    def test_custom_requirement_fails_without_trace(self):
        custom = {"polymer_score": ["run_polymer_evaluation"]}
        guard = ExecutionGuard(
            mode=GuardMode.STRICT, custom_requirements=custom
        )
        payload = {"polymer_score_top3": [0.9, 0.8, 0.7]}
        fb = _make_feedback(payload=payload)
        is_valid, issues = guard.validate_feedback(fb)
        assert is_valid is False
        assert len(issues) == 1

    def test_custom_requirement_merged_with_defaults(self):
        """Default requirements should still work alongside custom ones."""
        custom = {"custom_metric": ["run_custom"]}
        guard = ExecutionGuard(
            mode=GuardMode.STRICT, custom_requirements=custom
        )
        # A default claim type should still be recognized.
        payload = {
            "top_5": [1, 2, 3],
            "_execution_traces": [_make_trace_dict(method="run_top_k")],
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = guard.validate_feedback(fb)
        assert is_valid is True


# ---------------------------------------------------------------------------
# TestRobustness
# ---------------------------------------------------------------------------


class TestRobustness:
    """Test that the guard handles malformed inputs gracefully."""

    def setup_method(self):
        self.guard = ExecutionGuard(mode=GuardMode.STRICT)

    def test_none_traces_in_payload(self):
        payload = {
            "top_3": [1, 2, 3],
            "_execution_traces": None,
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = self.guard.validate_feedback(fb)
        assert is_valid is False
        assert len(issues) == 1

    def test_non_list_traces_in_payload(self):
        payload = {
            "top_3": [1, 2, 3],
            "_execution_traces": "not a list",
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = self.guard.validate_feedback(fb)
        assert is_valid is False

    def test_traces_with_mixed_valid_invalid_entries(self):
        payload = {
            "top_3": [1, 2, 3],
            "_execution_traces": [
                None,
                42,
                _make_trace_dict(method="run_top_k"),
            ],
        }
        fb = _make_feedback(payload=payload)
        is_valid, issues = self.guard.validate_feedback(fb)
        # The valid trace should match, so it passes.
        assert is_valid is True
        assert issues == []

    def test_empty_payload(self):
        fb = _make_feedback(payload={})
        is_valid, issues = self.guard.validate_feedback(fb)
        assert is_valid is True
        assert issues == []


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests: pipeline -> feedback -> guard flow."""

    def test_pipeline_traced_result_passes_guard(self):
        """Simulate: pipeline.run_top_k() -> build feedback -> guard validates."""
        # 1. Simulate a TracedResult from run_top_k.
        trace = ExecutionTrace(
            module="data_pipeline",
            method="run_top_k",
            input_summary={"n_items": 10, "k": 3},
            output_summary={"n_returned": 3},
            tag=ExecutionTag.COMPUTED,
            timestamp=1000.0,
            duration_ms=2.5,
        )
        traced_result = TracedResult(
            value=[
                {"name": "A", "value": 0.99, "rank": 1},
                {"name": "B", "value": 0.95, "rank": 2},
                {"name": "C", "value": 0.90, "rank": 3},
            ],
            traces=[trace],
            tag=ExecutionTag.COMPUTED,
        )

        # 2. Build feedback payload (as pipeline code would).
        payload = {
            "top_3": traced_result.value,
            **traced_result.to_payload_dict(),
        }
        fb = OptimizationFeedback(
            agent_name="data_agent",
            feedback_type="hypothesis",
            confidence=0.9,
            payload=payload,
            reasoning="Top 3 candidates identified by computed ranking.",
        )

        # 3. Guard validates -- should pass.
        guard = ExecutionGuard(mode=GuardMode.STRICT)
        is_valid, issues = guard.validate_feedback(fb)
        assert is_valid is True
        assert issues == []

    def test_feedback_without_traces_fails_guard(self):
        """Feedback with a 'top_5' key but no traces should fail STRICT."""
        payload = {
            "top_5": [
                {"name": "X", "value": 0.5},
                {"name": "Y", "value": 0.4},
            ],
            "notes": "I estimated these values.",
        }
        fb = OptimizationFeedback(
            agent_name="llm_agent",
            feedback_type="hypothesis",
            confidence=0.7,
            payload=payload,
            reasoning="Top candidates based on my estimation.",
        )

        guard = ExecutionGuard(mode=GuardMode.STRICT)
        is_valid, issues = guard.validate_feedback(fb)
        assert is_valid is False
        assert len(issues) >= 1
        assert "top_5" in issues[0]

    def test_lenient_tags_untraced_integration(self):
        """LENIENT mode should tag untraced values but remain valid."""
        payload = {
            "correlation": 0.95,
            "pareto_front": [{"x": 1, "y": 2}],
        }
        fb = OptimizationFeedback(
            agent_name="analysis_agent",
            feedback_type="hypothesis",
            confidence=0.6,
            payload=payload,
            reasoning="Estimated correlation and pareto front.",
        )

        guard = ExecutionGuard(mode=GuardMode.LENIENT)
        result = guard.enforce(fb)

        # Both values should be tagged.
        assert "[ESTIMATED" in str(result.payload["correlation"])
        assert "[ESTIMATED" in str(result.payload["pareto_front"])

    def test_mixed_traced_and_untraced_claims(self):
        """Only untraced claims should be tagged; traced ones left alone."""
        trace = ExecutionTrace(
            module="data_pipeline",
            method="run_top_k",
            input_summary={},
            output_summary={},
            tag=ExecutionTag.COMPUTED,
            timestamp=1000.0,
            duration_ms=1.0,
        )
        traced_result = TracedResult(
            value=[{"name": "A", "value": 0.99}],
            traces=[trace],
            tag=ExecutionTag.COMPUTED,
        )

        payload = {
            "top_3": traced_result.value,
            "outlier_count": 5,  # Untraced claim
            **traced_result.to_payload_dict(),
        }
        fb = OptimizationFeedback(
            agent_name="hybrid_agent",
            feedback_type="hypothesis",
            confidence=0.8,
            payload=payload,
        )

        # STRICT: fails because outlier_count is untraced.
        guard_strict = ExecutionGuard(mode=GuardMode.STRICT)
        is_valid, issues = guard_strict.validate_feedback(fb)
        assert is_valid is False
        assert any("outlier" in issue.lower() for issue in issues)

        # LENIENT: valid, but outlier_count is tagged.
        guard_lenient = ExecutionGuard(mode=GuardMode.LENIENT)
        result = guard_lenient.enforce(fb)
        assert "[ESTIMATED" in str(result.payload["outlier_count"])
        # top_3 should remain unchanged (it was traced).
        assert result.payload["top_3"] == traced_result.value

    def test_enforce_strict_raises_with_message_details(self):
        """The ValueError from enforce() should contain useful details."""
        payload = {
            "top_5": [1, 2, 3, 4, 5],
            "pareto_front": [],
        }
        fb = _make_feedback(payload=payload)
        guard = ExecutionGuard(mode=GuardMode.STRICT)
        with pytest.raises(ValueError) as exc_info:
            guard.enforce(fb)
        msg = str(exc_info.value)
        assert "2 untraced" in msg
        assert "STRICT" in msg

    def test_full_roundtrip_with_traced_result_to_payload_dict(self):
        """Verify TracedResult.to_payload_dict() format is compatible with guard."""
        # Simulate what DataAnalysisPipeline produces.
        trace1 = ExecutionTrace(
            module="data_pipeline",
            method="run_top_k",
            input_summary={"n_items": 20, "k": 5},
            output_summary={"n_returned": 5},
            tag=ExecutionTag.COMPUTED,
            timestamp=1000.0,
            duration_ms=3.0,
        )
        trace2 = ExecutionTrace(
            module="data_pipeline",
            method="run_outlier_detection",
            input_summary={"n_items": 20, "n_sigma": 2.0},
            output_summary={"n_outliers": 2},
            tag=ExecutionTag.COMPUTED,
            timestamp=1001.0,
            duration_ms=2.0,
        )
        top_result = TracedResult(
            value=[{"name": "A", "rank": 1}],
            traces=[trace1],
            tag=ExecutionTag.COMPUTED,
        )
        outlier_result = TracedResult(
            value={"outliers": [{"name": "Z", "z_score": 3.5}]},
            traces=[trace2],
            tag=ExecutionTag.COMPUTED,
        )

        # Merge traces for the combined payload.
        all_traces = TracedResult.merge([top_result, outlier_result])
        overall_tag = TracedResult.overall_tag([top_result, outlier_result])

        payload = {
            "top_5": top_result.value,
            "outlier_report": outlier_result.value,
            "_execution_traces": [t.to_dict() for t in all_traces],
            "_execution_tag": overall_tag.value,
        }

        fb = OptimizationFeedback(
            agent_name="pipeline_agent",
            feedback_type="hypothesis",
            confidence=0.95,
            payload=payload,
        )

        guard = ExecutionGuard(mode=GuardMode.STRICT)
        is_valid, issues = guard.validate_feedback(fb)
        assert is_valid is True
        assert issues == []

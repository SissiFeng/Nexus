"""Tests for agents/execution_trace.py — execution tracing primitives."""

from __future__ import annotations

import time

import pytest

from optimization_copilot.agents.execution_trace import (
    ExecutionTag,
    ExecutionTrace,
    TracedResult,
    trace_call,
)


# ---------------------------------------------------------------------------
# ExecutionTag
# ---------------------------------------------------------------------------


class TestExecutionTag:
    def test_values(self):
        assert ExecutionTag.COMPUTED.value == "computed"
        assert ExecutionTag.ESTIMATED.value == "estimated"
        assert ExecutionTag.FAILED.value == "failed"

    def test_is_str_subclass(self):
        assert isinstance(ExecutionTag.COMPUTED, str)

    def test_from_string(self):
        assert ExecutionTag("computed") is ExecutionTag.COMPUTED
        assert ExecutionTag("failed") is ExecutionTag.FAILED


# ---------------------------------------------------------------------------
# ExecutionTrace
# ---------------------------------------------------------------------------


class TestExecutionTrace:
    def _make_trace(self, **overrides):
        defaults = dict(
            module="test.module",
            method="TestClass.run",
            input_summary={"n": 10},
            output_summary={"result": 42},
            tag=ExecutionTag.COMPUTED,
            timestamp=1000.0,
            duration_ms=5.5,
            error=None,
        )
        defaults.update(overrides)
        return ExecutionTrace(**defaults)

    def test_basic_creation(self):
        t = self._make_trace()
        assert t.module == "test.module"
        assert t.method == "TestClass.run"
        assert t.tag == ExecutionTag.COMPUTED
        assert t.error is None

    def test_failed_trace_has_error(self):
        t = self._make_trace(tag=ExecutionTag.FAILED, error="division by zero")
        assert t.tag == ExecutionTag.FAILED
        assert t.error == "division by zero"

    def test_to_dict(self):
        t = self._make_trace()
        d = t.to_dict()
        assert d["module"] == "test.module"
        assert d["tag"] == "computed"
        assert d["duration_ms"] == 5.5
        assert isinstance(d["input_summary"], dict)
        assert isinstance(d["output_summary"], dict)

    def test_from_dict_roundtrip(self):
        original = self._make_trace(tag=ExecutionTag.FAILED, error="oops")
        d = original.to_dict()
        restored = ExecutionTrace.from_dict(d)
        assert restored.module == original.module
        assert restored.method == original.method
        assert restored.tag == original.tag
        assert restored.error == original.error
        assert restored.timestamp == original.timestamp
        assert restored.duration_ms == original.duration_ms

    def test_from_dict_missing_optional_fields(self):
        d = {"module": "m", "method": "f", "tag": "computed"}
        t = ExecutionTrace.from_dict(d)
        assert t.input_summary == {}
        assert t.output_summary == {}
        assert t.timestamp == 0.0
        assert t.duration_ms == 0.0
        assert t.error is None


# ---------------------------------------------------------------------------
# TracedResult
# ---------------------------------------------------------------------------


class TestTracedResult:
    def test_is_computed(self):
        r = TracedResult(value=42, traces=[], tag=ExecutionTag.COMPUTED)
        assert r.is_computed is True

    def test_is_not_computed_when_failed(self):
        r = TracedResult(value=None, traces=[], tag=ExecutionTag.FAILED)
        assert r.is_computed is False

    def test_is_not_computed_when_estimated(self):
        r = TracedResult(value=None, traces=[], tag=ExecutionTag.ESTIMATED)
        assert r.is_computed is False

    def test_to_payload_dict(self):
        trace = ExecutionTrace(
            module="m", method="f",
            input_summary={}, output_summary={},
            tag=ExecutionTag.COMPUTED, timestamp=0.0, duration_ms=1.0,
        )
        r = TracedResult(value=99, traces=[trace], tag=ExecutionTag.COMPUTED)
        payload = r.to_payload_dict()
        assert payload["_execution_tag"] == "computed"
        assert len(payload["_execution_traces"]) == 1
        assert payload["_execution_traces"][0]["method"] == "f"

    def test_to_payload_dict_empty_traces(self):
        r = TracedResult(value=None, traces=[], tag=ExecutionTag.ESTIMATED)
        payload = r.to_payload_dict()
        assert payload["_execution_tag"] == "estimated"
        assert payload["_execution_traces"] == []

    def test_merge_traces(self):
        t1 = ExecutionTrace(
            module="a", method="a.f",
            input_summary={}, output_summary={},
            tag=ExecutionTag.COMPUTED, timestamp=0.0, duration_ms=1.0,
        )
        t2 = ExecutionTrace(
            module="b", method="b.g",
            input_summary={}, output_summary={},
            tag=ExecutionTag.COMPUTED, timestamp=0.0, duration_ms=2.0,
        )
        r1 = TracedResult(value=1, traces=[t1], tag=ExecutionTag.COMPUTED)
        r2 = TracedResult(value=2, traces=[t2], tag=ExecutionTag.COMPUTED)
        merged = TracedResult.merge([r1, r2])
        assert len(merged) == 2
        assert merged[0].method == "a.f"
        assert merged[1].method == "b.g"

    def test_merge_empty(self):
        assert TracedResult.merge([]) == []

    def test_overall_tag_all_computed(self):
        r1 = TracedResult(value=1, tag=ExecutionTag.COMPUTED)
        r2 = TracedResult(value=2, tag=ExecutionTag.COMPUTED)
        assert TracedResult.overall_tag([r1, r2]) == ExecutionTag.COMPUTED

    def test_overall_tag_any_failed(self):
        r1 = TracedResult(value=1, tag=ExecutionTag.COMPUTED)
        r2 = TracedResult(value=None, tag=ExecutionTag.FAILED)
        assert TracedResult.overall_tag([r1, r2]) == ExecutionTag.FAILED

    def test_overall_tag_estimated(self):
        r1 = TracedResult(value=1, tag=ExecutionTag.COMPUTED)
        r2 = TracedResult(value=None, tag=ExecutionTag.ESTIMATED)
        assert TracedResult.overall_tag([r1, r2]) == ExecutionTag.ESTIMATED

    def test_overall_tag_empty(self):
        assert TracedResult.overall_tag([]) == ExecutionTag.ESTIMATED


# ---------------------------------------------------------------------------
# trace_call
# ---------------------------------------------------------------------------


class TestTraceCall:
    def test_successful_call(self):
        result = trace_call(
            module="test",
            method="add",
            fn=lambda: 2 + 3,
            input_summary={"a": 2, "b": 3},
        )
        assert result.value == 5
        assert result.tag == ExecutionTag.COMPUTED
        assert result.is_computed
        assert len(result.traces) == 1
        assert result.traces[0].module == "test"
        assert result.traces[0].method == "add"
        assert result.traces[0].tag == ExecutionTag.COMPUTED
        assert result.traces[0].error is None

    def test_failed_call(self):
        def _failing():
            raise ValueError("test error")

        result = trace_call(module="test", method="fail", fn=_failing)
        assert result.value is None
        assert result.tag == ExecutionTag.FAILED
        assert not result.is_computed
        assert len(result.traces) == 1
        assert result.traces[0].tag == ExecutionTag.FAILED
        assert "test error" in result.traces[0].error

    def test_duration_is_positive(self):
        def _slow():
            time.sleep(0.01)
            return True

        result = trace_call(module="test", method="slow", fn=_slow)
        assert result.traces[0].duration_ms >= 5.0  # at least 5ms

    def test_timestamp_is_recent(self):
        before = time.time()
        result = trace_call(module="test", method="ts", fn=lambda: 1)
        after = time.time()
        ts = result.traces[0].timestamp
        assert before <= ts <= after

    def test_with_args_and_kwargs(self):
        def _add(a, b, c=0):
            return a + b + c

        result = trace_call(
            module="test", method="add",
            fn=_add, args=(1, 2), kwargs={"c": 3},
        )
        assert result.value == 6

    def test_input_summary_preserved(self):
        summary = {"param": "temperature", "range": [0, 100]}
        result = trace_call(
            module="test", method="f", fn=lambda: 42,
            input_summary=summary,
        )
        assert result.traces[0].input_summary == summary

    def test_output_summarizer_called(self):
        def _summarize(val):
            return {"length": len(val)}

        result = trace_call(
            module="test", method="f",
            fn=lambda: [1, 2, 3],
            output_summarizer=_summarize,
        )
        assert result.traces[0].output_summary == {"length": 3}

    def test_output_summarizer_failure_is_safe(self):
        def _bad_summarizer(val):
            raise RuntimeError("summarizer crashed")

        result = trace_call(
            module="test", method="f",
            fn=lambda: 42,
            output_summarizer=_bad_summarizer,
        )
        # Should still succeed — summarizer failure doesn't break trace_call
        assert result.value == 42
        assert result.tag == ExecutionTag.COMPUTED
        assert result.traces[0].output_summary == {}

    def test_defaults_for_none_kwargs(self):
        result = trace_call(
            module="test", method="f",
            fn=lambda: "ok",
            kwargs=None,
            input_summary=None,
        )
        assert result.value == "ok"
        assert result.traces[0].input_summary == {}

    def test_trace_call_with_complex_return(self):
        def _complex():
            return {
                "pareto_front": [("A", 0.9), ("B", 0.8)],
                "n_pareto": 2,
            }

        result = trace_call(
            module="multi_objective.pareto",
            method="MultiObjectiveAnalyzer.analyze",
            fn=_complex,
            input_summary={"n_observations": 50},
            output_summarizer=lambda v: {"n_pareto": v["n_pareto"]},
        )
        assert result.value["n_pareto"] == 2
        assert result.traces[0].output_summary["n_pareto"] == 2

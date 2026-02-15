"""Execution tracing primitives for code-execution enforcement.

Every quantitative result produced by the agent layer carries an
:class:`ExecutionTrace` proving that code actually ran.  The
:func:`trace_call` helper wraps any callable with timing, error
handling, and tag assignment so pipeline methods stay concise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ExecutionTag(str, Enum):
    """Tag indicating whether a result was actually computed."""

    COMPUTED = "computed"     # Code ran and produced this result
    ESTIMATED = "estimated"  # NOT computed; explicitly marked as estimate
    FAILED = "failed"        # Code attempted execution but raised an error


@dataclass
class ExecutionTrace:
    """Record of a single code-execution step.

    Parameters
    ----------
    module : str
        Dotted module path (e.g. ``"explain.interaction_map"``).
    method : str
        Class.method or function name (e.g. ``"InteractionMap.fit"``).
    input_summary : dict
        Compact summary of inputs (e.g. ``{"n_samples": 99}``).
    output_summary : dict
        Compact summary of outputs (e.g. ``{"n_effects": 4}``).
    tag : ExecutionTag
        Whether this step completed successfully.
    timestamp : float
        ``time.time()`` when execution started.
    duration_ms : float
        Wall-clock milliseconds for this step.
    error : str | None
        Error message if ``tag == FAILED``.
    """

    module: str
    method: str
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    tag: ExecutionTag
    timestamp: float
    duration_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "module": self.module,
            "method": self.method,
            "input_summary": dict(self.input_summary),
            "output_summary": dict(self.output_summary),
            "tag": self.tag.value,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionTrace:
        """Reconstruct from a plain dict."""
        return cls(
            module=data["module"],
            method=data["method"],
            input_summary=data.get("input_summary", {}),
            output_summary=data.get("output_summary", {}),
            tag=ExecutionTag(data["tag"]),
            timestamp=data.get("timestamp", 0.0),
            duration_ms=data.get("duration_ms", 0.0),
            error=data.get("error"),
        )


@dataclass
class TracedResult:
    """A computed value bundled with its execution traces.

    Parameters
    ----------
    value : Any
        The actual result of the computation.
    traces : list[ExecutionTrace]
        All execution traces that contributed to this result.
    tag : ExecutionTag
        Overall tag — ``COMPUTED`` only when **all** traces are ``COMPUTED``.
    """

    value: Any
    traces: list[ExecutionTrace] = field(default_factory=list)
    tag: ExecutionTag = ExecutionTag.COMPUTED

    @property
    def is_computed(self) -> bool:
        """Whether the result was fully computed (no failures)."""
        return self.tag == ExecutionTag.COMPUTED

    def to_payload_dict(self) -> dict[str, Any]:
        """Return trace metadata suitable for ``OptimizationFeedback.payload``.

        Returns a dict with ``_execution_traces`` (list of trace dicts)
        and ``_execution_tag`` (string tag).
        """
        return {
            "_execution_traces": [t.to_dict() for t in self.traces],
            "_execution_tag": self.tag.value,
        }

    @staticmethod
    def merge(results: list[TracedResult]) -> list[ExecutionTrace]:
        """Collect all traces from multiple :class:`TracedResult` instances."""
        traces: list[ExecutionTrace] = []
        for r in results:
            traces.extend(r.traces)
        return traces

    @staticmethod
    def overall_tag(results: list[TracedResult]) -> ExecutionTag:
        """Compute the overall tag for a collection of results.

        ``COMPUTED`` only if every result is ``COMPUTED``.
        ``FAILED`` if any result is ``FAILED``.
        ``ESTIMATED`` otherwise.
        """
        if not results:
            return ExecutionTag.ESTIMATED
        tags = {r.tag for r in results}
        if tags == {ExecutionTag.COMPUTED}:
            return ExecutionTag.COMPUTED
        if ExecutionTag.FAILED in tags:
            return ExecutionTag.FAILED
        return ExecutionTag.ESTIMATED


def trace_call(
    module: str,
    method: str,
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    input_summary: dict[str, Any] | None = None,
    output_summarizer: Callable[[Any], dict[str, Any]] | None = None,
) -> TracedResult:
    """Execute *fn* and wrap the result in a :class:`TracedResult`.

    On success the trace is tagged ``COMPUTED``; on exception it is
    tagged ``FAILED`` with the error message preserved.

    Parameters
    ----------
    module : str
        Dotted module path for the trace record.
    method : str
        Method/function name for the trace record.
    fn : Callable
        The callable to execute.
    args : tuple
        Positional arguments for *fn*.
    kwargs : dict | None
        Keyword arguments for *fn*.
    input_summary : dict | None
        Compact description of the inputs.
    output_summarizer : Callable | None
        If provided, called on the result to produce ``output_summary``.

    Returns
    -------
    TracedResult
        Contains the computed value (or ``None`` on failure) and the trace.
    """
    if kwargs is None:
        kwargs = {}
    if input_summary is None:
        input_summary = {}

    start = time.monotonic()
    ts = time.time()

    try:
        result = fn(*args, **kwargs)
        duration = (time.monotonic() - start) * 1000.0

        output_summary: dict[str, Any] = {}
        if output_summarizer is not None:
            try:
                output_summary = output_summarizer(result)
            except Exception:
                output_summary = {}

        trace = ExecutionTrace(
            module=module,
            method=method,
            input_summary=input_summary,
            output_summary=output_summary,
            tag=ExecutionTag.COMPUTED,
            timestamp=ts,
            duration_ms=duration,
        )
        return TracedResult(value=result, traces=[trace], tag=ExecutionTag.COMPUTED)

    except Exception as exc:
        duration = (time.monotonic() - start) * 1000.0
        trace = ExecutionTrace(
            module=module,
            method=method,
            input_summary=input_summary,
            output_summary={},
            tag=ExecutionTag.FAILED,
            timestamp=ts,
            duration_ms=duration,
            error=str(exc),
        )
        return TracedResult(value=None, traces=[trace], tag=ExecutionTag.FAILED)

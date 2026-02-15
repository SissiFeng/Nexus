"""TracedScientificAgent — optional base class with built-in pipeline and trace collection.

Subclasses implement :meth:`analyze_traced` instead of :meth:`analyze`
and :meth:`_build_feedback` instead of :meth:`get_optimization_feedback`.
The base class automatically:

- Provides ``self.pipeline`` — a :class:`DataAnalysisPipeline` instance
- Collects execution traces from any :class:`TracedResult` values
- Injects traces into the feedback payload

Existing agents that extend :class:`ScientificAgent` directly are NOT
affected — this is opt-in only.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from optimization_copilot.agents.base import (
    AgentContext,
    AgentMode,
    OptimizationFeedback,
    ScientificAgent,
)
from optimization_copilot.agents.execution_trace import (
    ExecutionTag,
    ExecutionTrace,
    TracedResult,
)
from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline


class TracedScientificAgent(ScientificAgent):
    """ScientificAgent subclass with built-in pipeline and trace collection.

    Provides:
    - ``self.pipeline``: :class:`DataAnalysisPipeline` instance
    - ``self._traces``: traces collected during the latest ``analyze()`` call
    - Automatic trace injection into feedback payloads

    Subclasses implement :meth:`analyze_traced` and :meth:`_build_feedback`
    instead of the raw ``analyze`` / ``get_optimization_feedback`` pair.

    Parameters
    ----------
    mode : AgentMode
        Operational mode (default ``PRAGMATIC``).
    """

    def __init__(self, mode: AgentMode = AgentMode.PRAGMATIC) -> None:
        super().__init__(mode=mode)
        self.pipeline = DataAnalysisPipeline()
        self._traces: list[ExecutionTrace] = []

    # ── Wrapping analyze → analyze_traced ─────────────────────────

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        """Call :meth:`analyze_traced`, collect traces from any ``TracedResult`` values.

        After this method returns, ``self._traces`` contains all traces
        from the call, and the result dict has ``_execution_traces``
        (list of trace dicts) and ``_execution_tag`` (overall tag string).
        """
        self._traces = []
        result = self.analyze_traced(context)

        # Extract traces from TracedResult values and unwrap them
        unwrapped: dict[str, Any] = {}
        for key, val in result.items():
            if isinstance(val, TracedResult):
                self._traces.extend(val.traces)
                unwrapped[key] = val.value
            else:
                unwrapped[key] = val

        # Inject trace metadata
        unwrapped["_execution_traces"] = [t.to_dict() for t in self._traces]
        tags = {t.tag for t in self._traces}
        if tags == {ExecutionTag.COMPUTED}:
            overall = ExecutionTag.COMPUTED
        elif ExecutionTag.FAILED in tags:
            overall = ExecutionTag.FAILED
        else:
            overall = ExecutionTag.ESTIMATED if tags else ExecutionTag.ESTIMATED
        unwrapped["_execution_tag"] = overall.value

        return unwrapped

    @abstractmethod
    def analyze_traced(self, context: AgentContext) -> dict[str, Any]:
        """Implement analysis using ``self.pipeline`` methods.

        Return values can be :class:`TracedResult` instances — they are
        automatically unwrapped and their traces collected.

        Parameters
        ----------
        context : AgentContext
            All data the agent needs.

        Returns
        -------
        dict[str, Any]
            Analysis results; values may be ``TracedResult`` or plain values.
        """
        ...

    # ── Wrapping get_optimization_feedback → _build_feedback ──────

    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        """Build feedback and inject execution traces into the payload.

        Calls :meth:`_build_feedback` and then enriches the payload with
        ``_execution_traces`` and ``_execution_tag`` from the analysis result.
        """
        feedback = self._build_feedback(analysis_result)
        if feedback is not None:
            # Inject traces into payload
            traces = analysis_result.get("_execution_traces", [])
            tag = analysis_result.get("_execution_tag", ExecutionTag.ESTIMATED.value)
            feedback.payload["_execution_traces"] = traces
            feedback.payload["_execution_tag"] = tag
        return feedback

    @abstractmethod
    def _build_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        """Build the feedback without trace injection.

        Subclasses implement this to convert analysis results into
        an :class:`OptimizationFeedback` (or return ``None``).

        Parameters
        ----------
        analysis_result : dict[str, Any]
            Output from :meth:`analyze_traced` (with ``TracedResult``
            values already unwrapped).

        Returns
        -------
        OptimizationFeedback | None
        """
        ...

    @property
    def collected_traces(self) -> list[ExecutionTrace]:
        """Traces collected during the most recent ``analyze()`` call."""
        return list(self._traces)

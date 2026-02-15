"""Execution guard: validates that feedback payloads carry execution traces.

The :class:`ExecutionGuard` scans :class:`OptimizationFeedback` payloads for
quantitative claim keywords (e.g. ``"top_5_polymers"``, ``"pareto_front"``)
and verifies that a matching :class:`ExecutionTrace` exists in the payload's
``_execution_traces`` list.  This prevents LLM-hallucinated numbers from
entering the optimization loop without having been computed by real code.

Two modes are supported:

- **STRICT**: Rejects feedback containing untraced quantitative claims.
- **LENIENT**: Accepts all feedback but tags untraced values with an
  ``[ESTIMATED - not computed]`` marker so downstream consumers can
  distinguish computed results from unverified claims.
"""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any

from optimization_copilot.agents.base import OptimizationFeedback
from optimization_copilot.agents.execution_trace import ExecutionTag, ExecutionTrace


# ---------------------------------------------------------------------------
# Guard mode
# ---------------------------------------------------------------------------


class GuardMode(str, Enum):
    """Operating mode for the execution guard."""

    STRICT = "strict"    # Reject feedback with untraced quantitative claims
    LENIENT = "lenient"  # Tag untraced claims as [ESTIMATED - not computed]


# ---------------------------------------------------------------------------
# Claim -> trace requirement registry
# ---------------------------------------------------------------------------

# If a payload key contains one of these substrings, a matching trace is
# required.  The value lists contain method-name substrings that satisfy
# the requirement (i.e. if *any* trace's ``method`` field contains one of
# these substrings, the claim is considered traced).

CLAIM_TRACE_REQUIREMENTS: dict[str, list[str]] = {
    "top_k":               ["run_top_k", "run_ranking"],
    "top_":                ["run_top_k", "run_ranking"],
    "ranking":             ["run_top_k", "run_ranking"],
    "rank":                ["run_top_k", "run_ranking"],
    "outlier":             ["run_outlier_detection"],
    "pareto_front":        ["run_pareto_analysis"],
    "pareto":              ["run_pareto_analysis"],
    "non_dominated":       ["run_pareto_analysis"],
    "correlation":         ["run_correlation", "run_confounder_detection"],
    "pearson":             ["run_correlation"],
    "confounder":          ["run_confounder_detection", "run_correlation"],
    "fanova":              ["run_fanova", "run_insight_report"],
    "main_effect":         ["run_fanova", "run_insight_report"],
    "feature_importance":  ["run_fanova", "run_screening"],
    "importance":          ["run_fanova", "run_screening"],
    "interaction":         ["run_fanova", "run_insight_report"],
    "equation":            ["run_symreg", "run_insight_report"],
    "symbolic_regression": ["run_symreg", "run_insight_report"],
    "symreg":              ["run_symreg", "run_insight_report"],
    "diagnostics":         ["run_diagnostics"],
    "convergence":         ["run_diagnostics"],
    "noise_estimate":      ["run_diagnostics"],
    "screening":           ["run_screening"],
    "molecular":           ["run_molecular_pipeline"],
    "fingerprint":         ["run_molecular_pipeline"],
    "acquisition":         ["run_molecular_pipeline"],
    "best_":               ["run_top_k", "run_ranking", "run_pareto_analysis"],
    "worst_":              ["run_top_k", "run_ranking"],
    "highest_":            ["run_top_k", "run_ranking"],
    "lowest_":             ["run_top_k", "run_ranking"],
}

# Sort claim keywords longest-first so that ``"pareto_front"`` is matched
# before ``"pareto"`` when scanning payload keys.
_SORTED_CLAIM_KEYS: list[str] = sorted(
    CLAIM_TRACE_REQUIREMENTS.keys(), key=len, reverse=True
)


# ---------------------------------------------------------------------------
# ExecutionGuard
# ---------------------------------------------------------------------------


class ExecutionGuard:
    """Validates that :class:`OptimizationFeedback` payloads carry traces.

    Parameters
    ----------
    mode : GuardMode
        ``STRICT``: ``is_valid=False`` if any quantitative claim lacks a
        matching execution trace.
        ``LENIENT``: ``is_valid=True`` always, but the issues list is
        populated for untraced claims.
    custom_requirements : dict | None
        Additional ``{claim_keyword: [method_patterns]}`` mappings merged
        on top of the default :data:`CLAIM_TRACE_REQUIREMENTS`.
    """

    def __init__(
        self,
        mode: GuardMode = GuardMode.STRICT,
        custom_requirements: dict[str, list[str]] | None = None,
    ) -> None:
        self.mode = mode

        # Build the effective requirements map.
        self._requirements: dict[str, list[str]] = dict(CLAIM_TRACE_REQUIREMENTS)
        if custom_requirements:
            self._requirements.update(custom_requirements)

        # Pre-sort longest-first for deterministic matching.
        self._sorted_keys: list[str] = sorted(
            self._requirements.keys(), key=len, reverse=True
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def validate_feedback(
        self, feedback: OptimizationFeedback
    ) -> tuple[bool, list[str]]:
        """Validate *feedback* and return ``(is_valid, issues)``.

        Scans payload keys for quantitative claim patterns.  For each
        claim found, checks whether a matching execution trace exists
        in ``payload["_execution_traces"]``.

        In ``STRICT`` mode, ``is_valid`` is ``False`` when any untraced
        claim is detected.  In ``LENIENT`` mode, ``is_valid`` is always
        ``True`` but the issues list is still populated.

        Parameters
        ----------
        feedback : OptimizationFeedback
            The feedback to validate.

        Returns
        -------
        tuple[bool, list[str]]
            ``(is_valid, issues)`` where *issues* is a list of
            human-readable descriptions of untraced claims.
        """
        issues: list[str] = []

        try:
            payload = feedback.payload
            if not isinstance(payload, dict):
                return (self.mode != GuardMode.STRICT, [])
        except Exception:
            # Malformed feedback -- be robust.
            return (self.mode != GuardMode.STRICT, [])

        traces = self._extract_traces(payload)
        claims = self._detect_quantitative_claims(payload)

        for key, claim_type in claims:
            if not self._find_matching_trace(claim_type, traces):
                issues.append(
                    f"Payload key '{key}' implies quantitative claim "
                    f"'{claim_type}' but no matching execution trace was "
                    f"found (expected method containing one of "
                    f"{self._requirements.get(claim_type, [])})"
                )

        if self.mode == GuardMode.STRICT:
            return (len(issues) == 0, issues)
        else:
            # LENIENT: always valid, but issues are informational.
            return (True, issues)

    def enforce(
        self, feedback: OptimizationFeedback
    ) -> OptimizationFeedback:
        """Return a corrected copy of *feedback*.

        In ``STRICT`` mode, raises :class:`ValueError` if untraced
        quantitative claims are found.

        In ``LENIENT`` mode, wraps untraced numeric/list values with an
        ``"[ESTIMATED - not computed]"`` prefix and returns a new
        :class:`OptimizationFeedback` instance.

        Parameters
        ----------
        feedback : OptimizationFeedback
            The feedback to enforce.

        Returns
        -------
        OptimizationFeedback
            A (possibly modified) copy of *feedback*.

        Raises
        ------
        ValueError
            In ``STRICT`` mode when untraced quantitative claims exist.
        """
        is_valid, issues = self.validate_feedback(feedback)

        if is_valid and not issues:
            # No problems found; return as-is.
            return feedback

        if self.mode == GuardMode.STRICT:
            raise ValueError(
                f"ExecutionGuard (STRICT): {len(issues)} untraced "
                f"quantitative claim(s) found:\n"
                + "\n".join(f"  - {issue}" for issue in issues)
            )

        # LENIENT: tag untraced values.
        try:
            payload = feedback.payload
            if not isinstance(payload, dict):
                return feedback
        except Exception:
            return feedback

        new_payload = deepcopy(payload)
        claims = self._detect_quantitative_claims(payload)
        traces = self._extract_traces(payload)

        for key, claim_type in claims:
            if not self._find_matching_trace(claim_type, traces):
                if key in new_payload:
                    new_payload[key] = self._tag_untraced_value(
                        key, new_payload[key]
                    )

        return OptimizationFeedback(
            agent_name=feedback.agent_name,
            feedback_type=feedback.feedback_type,
            confidence=feedback.confidence,
            payload=new_payload,
            reasoning=feedback.reasoning,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _extract_traces(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract the ``_execution_traces`` list from *payload*.

        Returns an empty list if the key is missing or the value is not
        a list.
        """
        try:
            traces = payload.get("_execution_traces", [])
            if isinstance(traces, list):
                return traces
            return []
        except Exception:
            return []

    def _detect_quantitative_claims(
        self, payload: dict[str, Any]
    ) -> list[tuple[str, str]]:
        """Identify payload keys that imply quantitative claims.

        Scans all payload keys (excluding those starting with ``_``,
        which are considered internal metadata) and checks whether any
        substring from the requirements registry matches.

        Returns
        -------
        list[tuple[str, str]]
            ``(payload_key, claim_type)`` pairs.  Each payload key
            appears at most once, matched against the *longest*
            matching claim keyword.
        """
        claims: list[tuple[str, str]] = []
        try:
            keys = list(payload.keys())
        except Exception:
            return claims

        for key in keys:
            # Skip internal/metadata keys.
            if key.startswith("_"):
                continue
            # Convert key to lowercase for matching.
            lower_key = key.lower()
            for claim_keyword in self._sorted_keys:
                if claim_keyword in lower_key:
                    claims.append((key, claim_keyword))
                    break  # Longest match wins; move to next key.

        return claims

    def _find_matching_trace(
        self, claim_type: str, traces: list[dict[str, Any]]
    ) -> bool:
        """Check whether any trace satisfies the requirement for *claim_type*.

        A trace matches if its ``method`` field contains any of the
        required method patterns for the given claim type.

        Parameters
        ----------
        claim_type : str
            The claim keyword (key into ``_requirements``).
        traces : list[dict]
            List of trace dicts (each with a ``method`` key).

        Returns
        -------
        bool
            ``True`` if at least one trace matches.
        """
        required_patterns = self._requirements.get(claim_type, [])
        if not required_patterns:
            return False

        for trace in traces:
            try:
                method = trace.get("method", "")
                if not isinstance(method, str):
                    continue
                for pattern in required_patterns:
                    if pattern in method:
                        return True
            except Exception:
                continue

        return False

    def _tag_untraced_value(self, key: str, value: Any) -> Any:
        """Wrap an untraced value with an ``[ESTIMATED]`` marker.

        - **str**: Prepends ``"[ESTIMATED - not computed] "``.
        - **int / float**: Converts to a string with the marker prefix.
        - **list**: Wraps the entire list in a string representation
          with the marker prefix.
        - **dict**: Adds an ``"_estimated"`` key set to ``True``.
        - **Other types**: Converts to string with the marker prefix.

        Parameters
        ----------
        key : str
            The payload key (for context, unused currently).
        value : Any
            The original value to tag.

        Returns
        -------
        Any
            The tagged value.
        """
        marker = "[ESTIMATED - not computed]"

        if isinstance(value, str):
            return f"{marker} {value}"
        elif isinstance(value, (int, float)):
            return f"{marker} {value}"
        elif isinstance(value, list):
            return f"{marker} {value}"
        elif isinstance(value, dict):
            tagged = dict(value)
            tagged["_estimated"] = True
            return tagged
        else:
            return f"{marker} {value}"

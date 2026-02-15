"""Validation harness for evaluating agents against ground-truth annotations.

Provides a structured way to measure agent quality using precision, recall,
and F1 score against v5 case-study annotations (mod #8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.agents.base import (
    AgentContext,
    OptimizationFeedback,
    ScientificAgent,
)
from optimization_copilot.agents.orchestrator import (
    OrchestratorEvent,
    ScientificOrchestrator,
)


# ── Validation Result ─────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Result of a validation evaluation.

    Parameters
    ----------
    precision : float
        True positives / (true positives + false positives).
    recall : float
        True positives / (true positives + false negatives).
    f1_score : float
        Harmonic mean of precision and recall.
    n_correct : int
        Number of correct predictions.
    n_total : int
        Total number of test cases.
    details : list[dict]
        Per-case details for debugging.
    """

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    n_correct: int = 0
    n_total: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Simple accuracy: n_correct / n_total."""
        if self.n_total == 0:
            return 0.0
        return self.n_correct / self.n_total

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "n_correct": self.n_correct,
            "n_total": self.n_total,
            "details": list(self.details),
        }


# ── Helpers ───────────────────────────────────────────────────────────


def _compute_precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 from raw counts."""
    if true_positives + false_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return precision, recall, f1


# ── Validation Harness ────────────────────────────────────────────────


class AgentValidationHarness:
    """Evaluate agents, detectors, and orchestrators against ground truth.

    Provides three evaluation methods:

    1. ``evaluate_anomaly_detection`` -- compare detector output to
       annotated ground-truth anomaly labels.
    2. ``evaluate_agent`` -- compare agent feedback to expected feedback
       types and confidence ranges.
    3. ``evaluate_orchestrator`` -- verify that the correct agents
       activate for each event type.
    """

    def evaluate_anomaly_detection(
        self,
        detector: Any,
        test_data: list[dict],
        annotations: dict[str, bool],
    ) -> ValidationResult:
        """Evaluate an anomaly detector against ground-truth annotations.

        Parameters
        ----------
        detector : AnomalyDetector
            Detector with a ``detect(x, y, raw_data, kpi_values)`` method.
        test_data : list[dict]
            Each dict has keys ``"x"``, ``"y"``, ``"raw_data"``,
            ``"kpi_values"``, and a ``"label"`` key used for lookup
            in *annotations*.
        annotations : dict[str, bool]
            Mapping from label to ground-truth anomaly flag.

        Returns
        -------
        ValidationResult
            Precision, recall, F1 comparing detector to annotations.
        """
        tp = 0
        fp = 0
        fn = 0
        details: list[dict[str, Any]] = []
        n_total = len(test_data)

        for entry in test_data:
            label = entry.get("label", "")
            expected = annotations.get(label, False)

            try:
                report = detector.detect(
                    x=entry.get("x", []),
                    y=entry.get("y", 0.0),
                    raw_data=entry.get("raw_data", {}),
                    kpi_values=entry.get("kpi_values", {}),
                )
                predicted = report.is_anomalous
            except Exception:
                predicted = False

            if predicted and expected:
                tp += 1
            elif predicted and not expected:
                fp += 1
            elif not predicted and expected:
                fn += 1

            details.append({
                "label": label,
                "expected": expected,
                "predicted": predicted,
                "correct": predicted == expected,
            })

        precision, recall, f1 = _compute_precision_recall_f1(tp, fp, fn)
        n_correct = sum(1 for d in details if d["correct"])

        return ValidationResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            n_correct=n_correct,
            n_total=n_total,
            details=details,
        )

    def evaluate_agent(
        self,
        agent: ScientificAgent,
        test_contexts: list[AgentContext],
        expected_feedbacks: list[dict],
    ) -> ValidationResult:
        """Evaluate an agent against expected feedback specifications.

        Parameters
        ----------
        agent : ScientificAgent
            The agent under test.
        test_contexts : list[AgentContext]
            Contexts to feed to the agent.
        expected_feedbacks : list[dict]
            Expected feedback specifications.  Each dict may have:
            - ``"feedback_type"`` (str): expected feedback type
            - ``"min_confidence"`` (float): minimum confidence
            - ``"max_confidence"`` (float): maximum confidence
            - ``"should_produce_feedback"`` (bool): whether feedback
              is expected at all (default True)

        Returns
        -------
        ValidationResult
            Precision/recall based on feedback type matching.
        """
        tp = 0
        fp = 0
        fn = 0
        details: list[dict[str, Any]] = []
        n_total = len(test_contexts)

        for i, ctx in enumerate(test_contexts):
            expected = expected_feedbacks[i] if i < len(expected_feedbacks) else {}
            should_produce = expected.get("should_produce_feedback", True)
            expected_type = expected.get("feedback_type")
            min_conf = expected.get("min_confidence", 0.0)
            max_conf = expected.get("max_confidence", 1.0)

            try:
                result = agent.analyze(ctx)
                feedback = agent.get_optimization_feedback(result)
            except Exception:
                feedback = None

            produced = feedback is not None
            correct = False

            if should_produce and produced:
                type_ok = (
                    expected_type is None
                    or feedback.feedback_type == expected_type
                )
                conf_ok = min_conf <= feedback.confidence <= max_conf
                if type_ok and conf_ok:
                    tp += 1
                    correct = True
                else:
                    fp += 1
            elif should_produce and not produced:
                fn += 1
            elif not should_produce and produced:
                fp += 1
            else:
                # Not expected, not produced -> true negative
                tp += 1
                correct = True

            details.append({
                "index": i,
                "expected_type": expected_type,
                "produced": produced,
                "feedback_type": feedback.feedback_type if feedback else None,
                "confidence": feedback.confidence if feedback else None,
                "correct": correct,
            })

        precision, recall, f1 = _compute_precision_recall_f1(tp, fp, fn)
        n_correct = sum(1 for d in details if d["correct"])

        return ValidationResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            n_correct=n_correct,
            n_total=n_total,
            details=details,
        )

    def evaluate_orchestrator(
        self,
        orchestrator: ScientificOrchestrator,
        events: list[OrchestratorEvent],
        contexts: list[AgentContext],
        expected_agent_activations: dict[str, list[str]],
    ) -> ValidationResult:
        """Evaluate that correct agents activate for each event type.

        Parameters
        ----------
        orchestrator : ScientificOrchestrator
            The orchestrator under test.
        events : list[OrchestratorEvent]
            Events to dispatch.
        contexts : list[AgentContext]
            Corresponding contexts (one per event).
        expected_agent_activations : dict[str, list[str]]
            Mapping from event_type to expected agent names that should
            be triggered.

        Returns
        -------
        ValidationResult
            How well the orchestrator's activation matches expectations.
        """
        tp = 0
        fp = 0
        fn = 0
        details: list[dict[str, Any]] = []
        n_total = len(events)

        for i, event in enumerate(events):
            ctx = contexts[i] if i < len(contexts) else AgentContext()

            # Clear audit trail to isolate this dispatch
            initial_dispatches = orchestrator.n_dispatches
            orchestrator.dispatch_event(event, ctx)

            # Find the audit entry for this dispatch
            trail = orchestrator.get_audit_trail()
            if len(trail) > initial_dispatches:
                entry = trail[initial_dispatches]
                actual_agents = set(entry.agents_triggered)
            else:
                actual_agents = set()

            expected_agents = set(
                expected_agent_activations.get(event.event_type, [])
            )

            # Compute per-agent hits
            for agent_name in expected_agents | actual_agents:
                in_expected = agent_name in expected_agents
                in_actual = agent_name in actual_agents

                if in_expected and in_actual:
                    tp += 1
                elif in_actual and not in_expected:
                    fp += 1
                elif in_expected and not in_actual:
                    fn += 1

            details.append({
                "event_type": event.event_type,
                "expected_agents": sorted(expected_agents),
                "actual_agents": sorted(actual_agents),
                "match": actual_agents == expected_agents,
            })

        precision, recall, f1 = _compute_precision_recall_f1(tp, fp, fn)
        n_correct = sum(1 for d in details if d["match"])

        return ValidationResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            n_correct=n_correct,
            n_total=n_total,
            details=details,
        )

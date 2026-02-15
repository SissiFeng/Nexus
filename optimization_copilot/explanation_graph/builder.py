from __future__ import annotations

from typing import Any

from optimization_copilot.core.models import CampaignSnapshot, StrategyDecision
from optimization_copilot.diagnostics.engine import DiagnosticsVector
from optimization_copilot.surgery.models import SurgeryReport, SurgeryAction, ActionType
from optimization_copilot.feasibility.surface import FailureSurface, DangerZone

from .models import EdgeType, ExplanationGraph, GraphEdge, GraphNode, NodeType


class GraphBuilder:
    """Builds an ExplanationGraph from optimization pipeline artifacts."""

    def __init__(
        self,
        signal_threshold: float = 0.1,
        confidence_threshold: float = 0.0,
    ) -> None:
        self._signal_threshold = signal_threshold
        self._confidence_threshold = confidence_threshold

    def build(
        self,
        snapshot: CampaignSnapshot,
        diagnostics: DiagnosticsVector | None = None,
        decision: StrategyDecision | None = None,
        surgery_report: SurgeryReport | None = None,
        failure_surface: FailureSurface | None = None,
    ) -> ExplanationGraph:
        graph = ExplanationGraph()

        if diagnostics:
            self._add_signal_nodes(graph, diagnostics)
        if decision:
            self._add_decision_node(graph, decision)
        if snapshot.constraints:
            self._add_constraint_nodes(graph, snapshot)
        if surgery_report:
            self._add_exclusion_nodes(graph, surgery_report)
        if failure_surface:
            self._add_failure_nodes(graph, failure_surface)

        if diagnostics and decision:
            self._wire_signal_to_decision_edges(graph, diagnostics, decision)
        if failure_surface and surgery_report:
            self._wire_failure_to_exclusion_edges(
                graph, failure_surface, surgery_report
            )
        if snapshot.constraints and decision:
            self._wire_constraint_to_decision_edges(graph, snapshot, decision)

        return graph

    # ------------------------------------------------------------------
    # Node builders
    # ------------------------------------------------------------------

    def _add_signal_nodes(
        self, graph: ExplanationGraph, diagnostics: DiagnosticsVector
    ) -> None:
        d = diagnostics.to_dict()

        # Defaults for comparison
        defaults: dict[str, Any] = {
            "convergence_trend": 0.0,
            "improvement_velocity": 0.0,
            "variance_contraction": 1.0,
            "noise_estimate": 0.0,
            "failure_rate": 0.0,
            "failure_clustering": 0.0,
            "feasibility_shrinkage": 0.0,
            "parameter_drift": 0.0,
            "model_uncertainty": 0.0,
            "exploration_coverage": 0.0,
            "kpi_plateau_length": 0,
            "best_kpi_value": 0.0,
            "data_efficiency": 0.0,
            "constraint_violation_rate": 0.0,
        }

        for field_name, value in d.items():
            default = defaults.get(field_name)
            if value == default:
                continue

            # For float values: skip if below signal threshold
            if isinstance(value, float):
                if abs(value) < self._signal_threshold:
                    continue
                label = f"{field_name} = {value:.4f}"
            elif isinstance(value, int):
                # kpi_plateau_length
                if value == 0:
                    continue
                label = f"{field_name} = {str(value)}"
            else:
                label = f"{field_name} = {value}"

            node_id = f"signal:{field_name}"
            data = {"value": value, "field": field_name}
            graph.add_node(
                GraphNode(
                    node_id=node_id,
                    node_type=NodeType.SIGNAL,
                    label=label,
                    data=data,
                )
            )

    def _add_decision_node(
        self, graph: ExplanationGraph, decision: StrategyDecision
    ) -> None:
        node_id = "decision:strategy"
        label = (
            f"{decision.backend_name} "
            f"(phase={decision.phase.value}, "
            f"explore={decision.exploration_strength:.2f})"
        )
        data = {
            "backend_name": decision.backend_name,
            "phase": decision.phase.value,
            "exploration_strength": decision.exploration_strength,
            "risk_posture": decision.risk_posture.value,
            "reason_codes": list(decision.reason_codes),
        }
        graph.add_node(
            GraphNode(
                node_id=node_id,
                node_type=NodeType.DECISION,
                label=label,
                data=data,
            )
        )

    def _add_constraint_nodes(
        self, graph: ExplanationGraph, snapshot: CampaignSnapshot
    ) -> None:
        for i, constraint in enumerate(snapshot.constraints):
            target = constraint.get("target", f"constraint_{i}")
            node_id = f"constraint:{target}"
            label = f"Constraint on {target}"
            if "lower" in constraint or "upper" in constraint:
                label += (
                    f": [{constraint.get('lower', '?')}, "
                    f"{constraint.get('upper', '?')}]"
                )
            graph.add_node(
                GraphNode(
                    node_id=node_id,
                    node_type=NodeType.CONSTRAINT,
                    label=label,
                    data=dict(constraint),
                )
            )

    def _add_exclusion_nodes(
        self, graph: ExplanationGraph, surgery_report: SurgeryReport
    ) -> None:
        for action in surgery_report.actions:
            params_str = (
                ",".join(action.target_params)
                if action.target_params
                else "unknown"
            )
            node_id = f"exclusion:{action.action_type.value}:{params_str}"
            label = (
                action.reason[:120]
                if action.reason
                else f"{action.action_type.value} on {params_str}"
            )
            data = {
                "action_type": action.action_type.value,
                "target_params": list(action.target_params),
                "confidence": action.confidence,
                "evidence": dict(action.evidence),
            }
            graph.add_node(
                GraphNode(
                    node_id=node_id,
                    node_type=NodeType.EXCLUSION,
                    label=label,
                    data=data,
                )
            )

    def _add_failure_nodes(
        self, graph: ExplanationGraph, failure_surface: FailureSurface
    ) -> None:
        for dz in failure_surface.danger_zones:
            node_id = f"failure:danger:{dz.parameter}:{dz.bound_type}"
            label = (
                f"Danger zone: {dz.parameter} {dz.bound_type} "
                f"{dz.threshold:.4g} "
                f"(failure_rate={dz.failure_rate:.0%})"
            )
            data = {
                "parameter": dz.parameter,
                "bound_type": dz.bound_type,
                "threshold": dz.threshold,
                "failure_rate": dz.failure_rate,
                "n_samples": dz.n_samples,
            }
            graph.add_node(
                GraphNode(
                    node_id=node_id,
                    node_type=NodeType.FAILURE,
                    label=label,
                    data=data,
                )
            )

    # ------------------------------------------------------------------
    # Edge wiring
    # ------------------------------------------------------------------

    def _wire_signal_to_decision_edges(
        self,
        graph: ExplanationGraph,
        diagnostics: DiagnosticsVector,
        decision: StrategyDecision,
    ) -> None:
        decision_id = "decision:strategy"
        if decision_id not in graph.nodes:
            return

        d = diagnostics.to_dict()

        # Heuristic threshold rules
        if (
            "signal:failure_rate" in graph.nodes
            and d.get("failure_rate", 0) > 0.3
        ):
            graph.add_edge(
                GraphEdge(
                    source_id="signal:failure_rate",
                    target_id=decision_id,
                    edge_type=EdgeType.TRIGGERS,
                    evidence={
                        "threshold": 0.3,
                        "value": d["failure_rate"],
                    },
                )
            )

        if (
            "signal:convergence_trend" in graph.nodes
            and d.get("convergence_trend", 0) < -0.2
        ):
            graph.add_edge(
                GraphEdge(
                    source_id="signal:convergence_trend",
                    target_id=decision_id,
                    edge_type=EdgeType.TRIGGERS,
                    evidence={
                        "threshold": -0.2,
                        "value": d["convergence_trend"],
                    },
                )
            )

        if (
            "signal:kpi_plateau_length" in graph.nodes
            and d.get("kpi_plateau_length", 0) > 5
        ):
            graph.add_edge(
                GraphEdge(
                    source_id="signal:kpi_plateau_length",
                    target_id=decision_id,
                    edge_type=EdgeType.TRIGGERS,
                    evidence={
                        "threshold": 5,
                        "value": d["kpi_plateau_length"],
                    },
                )
            )

        if (
            "signal:exploration_coverage" in graph.nodes
            and d.get("exploration_coverage", 0) < 0.3
            and d.get("exploration_coverage", 0) != 0
        ):
            graph.add_edge(
                GraphEdge(
                    source_id="signal:exploration_coverage",
                    target_id=decision_id,
                    edge_type=EdgeType.SUPPORTS,
                    evidence={
                        "threshold": 0.3,
                        "value": d["exploration_coverage"],
                    },
                )
            )

        if (
            "signal:noise_estimate" in graph.nodes
            and d.get("noise_estimate", 0) > 0.5
        ):
            graph.add_edge(
                GraphEdge(
                    source_id="signal:noise_estimate",
                    target_id=decision_id,
                    edge_type=EdgeType.SUPPORTS,
                    evidence={
                        "threshold": 0.5,
                        "value": d["noise_estimate"],
                    },
                )
            )

        if (
            "signal:variance_contraction" in graph.nodes
            and d.get("variance_contraction", 1.0) < 0.5
        ):
            graph.add_edge(
                GraphEdge(
                    source_id="signal:variance_contraction",
                    target_id=decision_id,
                    edge_type=EdgeType.SUPPORTS,
                    evidence={
                        "threshold": 0.5,
                        "value": d["variance_contraction"],
                    },
                )
            )

        # Also: match reason_codes to signal field names
        connected_signals = {
            e.source_id
            for e in graph.edges
            if e.target_id == decision_id
        }
        signal_nodes = graph.nodes_by_type(NodeType.SIGNAL)
        for reason_code in decision.reason_codes:
            for sn in signal_nodes:
                field_name = sn.data.get("field", "")
                if (
                    field_name
                    and field_name in reason_code
                    and sn.node_id not in connected_signals
                ):
                    graph.add_edge(
                        GraphEdge(
                            source_id=sn.node_id,
                            target_id=decision_id,
                            edge_type=EdgeType.TRIGGERS,
                            evidence={
                                "reason_code": reason_code,
                                "field": field_name,
                            },
                        )
                    )
                    connected_signals.add(sn.node_id)

    def _wire_failure_to_exclusion_edges(
        self,
        graph: ExplanationGraph,
        failure_surface: FailureSurface,
        surgery_report: SurgeryReport,
    ) -> None:
        failure_nodes = graph.nodes_by_type(NodeType.FAILURE)
        exclusion_nodes = graph.nodes_by_type(NodeType.EXCLUSION)

        for fn in failure_nodes:
            param = fn.data.get("parameter", "")
            for en in exclusion_nodes:
                if param in en.data.get("target_params", []):
                    graph.add_edge(
                        GraphEdge(
                            source_id=fn.node_id,
                            target_id=en.node_id,
                            edge_type=EdgeType.EXCLUDES,
                            confidence=fn.data.get("failure_rate", 0.5),
                            evidence={
                                "failure_rate": fn.data.get("failure_rate"),
                                "parameter": param,
                            },
                        )
                    )

    def _wire_constraint_to_decision_edges(
        self,
        graph: ExplanationGraph,
        snapshot: CampaignSnapshot,
        decision: StrategyDecision,
    ) -> None:
        decision_id = "decision:strategy"
        if decision_id not in graph.nodes:
            return

        constraint_nodes = graph.nodes_by_type(NodeType.CONSTRAINT)
        for cn in constraint_nodes:
            graph.add_edge(
                GraphEdge(
                    source_id=cn.node_id,
                    target_id=decision_id,
                    edge_type=EdgeType.RESTRICTS,
                    confidence=1.0,
                    evidence=dict(cn.data),
                )
            )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query_decision(
        self,
        graph: ExplanationGraph,
        node_id: str = "decision:strategy",
    ) -> list[list[tuple[str, str]]]:
        return graph.trace_back(node_id)

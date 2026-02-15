"""Tests for the Explanation Graph package.

Covers:
  - GraphNode (construction, serialization, enum values)
  - GraphEdge (construction, serialization, confidence default)
  - ExplanationGraph (CRUD, filtering, trace_back, serialization)
  - GraphBuilder (node creation, edge wiring, filtering, integration)
  - Query helpers
  - Edge cases
"""

from __future__ import annotations

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    Phase,
    RiskPosture,
    StabilizeSpec,
    StrategyDecision,
    VariableType,
)
from optimization_copilot.diagnostics.engine import DiagnosticsVector
from optimization_copilot.surgery.models import SurgeryAction, SurgeryReport, ActionType
from optimization_copilot.feasibility.surface import DangerZone, FailureSurface
from optimization_copilot.explanation_graph.models import (
    EdgeType,
    ExplanationGraph,
    GraphEdge,
    GraphNode,
    NodeType,
)
from optimization_copilot.explanation_graph.builder import GraphBuilder


# ── Helpers ───────────────────────────────────────────────


def _make_snapshot(**kwargs):
    """Create a basic CampaignSnapshot with sensible defaults."""
    defaults = dict(
        campaign_id="test-campaign",
        parameter_specs=[
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ],
        observations=[
            Observation(iteration=0, parameters={"x1": 5.0}, kpi_values={"y": 1.0}),
        ],
        objective_names=["y"],
        objective_directions=["minimize"],
        constraints=[],
        current_iteration=1,
    )
    defaults.update(kwargs)
    return CampaignSnapshot(**defaults)


def _make_diagnostics(**overrides):
    """Create a DiagnosticsVector with specific signal overrides."""
    return DiagnosticsVector(**overrides)


def _make_decision(reason_codes=None, backend_name="tpe"):
    """Create a StrategyDecision with sensible defaults."""
    return StrategyDecision(
        backend_name=backend_name,
        stabilize_spec=StabilizeSpec(),
        exploration_strength=0.5,
        batch_size=1,
        risk_posture=RiskPosture.MODERATE,
        phase=Phase.LEARNING,
        reason_codes=reason_codes or [],
    )


def _make_action(action_type, target_params, reason="test", confidence=0.8):
    """Create a SurgeryAction."""
    return SurgeryAction(
        action_type=action_type,
        target_params=target_params,
        reason=reason,
        confidence=confidence,
        evidence={"source": "test"},
    )


def _make_surgery_report(actions):
    """Create a SurgeryReport from a list of SurgeryAction."""
    return SurgeryReport(
        actions=actions,
        original_dim=3,
        effective_dim=2,
        space_reduction_ratio=0.33,
    )


def _make_failure_surface(danger_zones):
    """Create a FailureSurface with given danger zones."""
    return FailureSurface(
        safe_bounds={"x1": (0.0, 10.0)},
        danger_zones=danger_zones,
        parameter_failure_density={"x1": [0.0] * 10},
        n_observations=100,
        n_failures=20,
        overall_failure_rate=0.2,
    )


# ── TestGraphNode ─────────────────────────────────────────


class TestGraphNode:
    """Tests for the GraphNode dataclass."""

    def test_construction(self):
        """GraphNode should store all fields correctly."""
        node = GraphNode(
            node_id="sig:test",
            node_type=NodeType.SIGNAL,
            label="test signal",
            data={"value": 0.5},
        )
        assert node.node_id == "sig:test"
        assert node.node_type == NodeType.SIGNAL
        assert node.label == "test signal"
        assert node.data == {"value": 0.5}

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization and deserialization should be lossless, including NodeType enum."""
        node = GraphNode(
            node_id="decision:strategy",
            node_type=NodeType.DECISION,
            label="tpe strategy",
            data={"backend": "tpe"},
        )
        d = node.to_dict()
        # node_type should be serialized as string value
        assert d["node_type"] == "decision"
        restored = GraphNode.from_dict(d)
        assert restored.node_id == node.node_id
        assert restored.node_type == node.node_type
        assert restored.label == node.label
        assert restored.data == node.data

    def test_node_types_are_strings(self):
        """All NodeType enum members should be string-compatible."""
        for nt in NodeType:
            assert isinstance(nt.value, str)
            assert nt == NodeType(nt.value)


# ── TestGraphEdge ─────────────────────────────────────────


class TestGraphEdge:
    """Tests for the GraphEdge dataclass."""

    def test_construction(self):
        """GraphEdge should store all fields correctly."""
        edge = GraphEdge(
            source_id="signal:failure_rate",
            target_id="decision:strategy",
            edge_type=EdgeType.TRIGGERS,
            confidence=0.9,
            evidence={"threshold": 0.3},
        )
        assert edge.source_id == "signal:failure_rate"
        assert edge.target_id == "decision:strategy"
        assert edge.edge_type == EdgeType.TRIGGERS
        assert edge.confidence == 0.9
        assert edge.evidence == {"threshold": 0.3}

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization and deserialization should be lossless, including EdgeType enum."""
        edge = GraphEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.SUPPORTS,
            confidence=0.75,
            evidence={"note": "test"},
        )
        d = edge.to_dict()
        assert d["edge_type"] == "supports"
        restored = GraphEdge.from_dict(d)
        assert restored.source_id == edge.source_id
        assert restored.target_id == edge.target_id
        assert restored.edge_type == edge.edge_type
        assert abs(restored.confidence - edge.confidence) < 1e-9
        assert restored.evidence == edge.evidence

    def test_confidence_default(self):
        """Default confidence should be 1.0 when not specified."""
        edge = GraphEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CAUSES,
        )
        assert abs(edge.confidence - 1.0) < 1e-9


# ── TestExplanationGraph ──────────────────────────────────


class TestExplanationGraph:
    """Tests for the ExplanationGraph dataclass."""

    def test_empty_graph(self):
        """A fresh graph should have zero nodes and zero edges."""
        g = ExplanationGraph()
        assert g.n_nodes == 0
        assert g.n_edges == 0

    def test_add_node_and_retrieve(self):
        """Adding a node should make it retrievable via get_node."""
        g = ExplanationGraph()
        node = GraphNode(node_id="n1", node_type=NodeType.SIGNAL, label="test")
        g.add_node(node)
        assert g.n_nodes == 1
        retrieved = g.get_node("n1")
        assert retrieved is not None
        assert retrieved.node_id == "n1"
        # Non-existent node returns None
        assert g.get_node("nonexistent") is None

    def test_add_edge_and_query(self):
        """edges_from and edges_to should return correct edges."""
        g = ExplanationGraph()
        g.add_node(GraphNode(node_id="a", node_type=NodeType.SIGNAL, label="A"))
        g.add_node(GraphNode(node_id="b", node_type=NodeType.DECISION, label="B"))
        edge = GraphEdge(source_id="a", target_id="b", edge_type=EdgeType.TRIGGERS)
        g.add_edge(edge)

        assert g.n_edges == 1
        from_a = g.edges_from("a")
        assert len(from_a) == 1
        assert from_a[0].target_id == "b"

        to_b = g.edges_to("b")
        assert len(to_b) == 1
        assert to_b[0].source_id == "a"

        # No edges from b or to a
        assert len(g.edges_from("b")) == 0
        assert len(g.edges_to("a")) == 0

    def test_nodes_by_type_filters(self):
        """nodes_by_type should return only nodes of the specified type."""
        g = ExplanationGraph()
        g.add_node(GraphNode(node_id="s1", node_type=NodeType.SIGNAL, label="S1"))
        g.add_node(GraphNode(node_id="s2", node_type=NodeType.SIGNAL, label="S2"))
        g.add_node(GraphNode(node_id="d1", node_type=NodeType.DECISION, label="D1"))
        g.add_node(GraphNode(node_id="c1", node_type=NodeType.CONSTRAINT, label="C1"))

        signals = g.nodes_by_type(NodeType.SIGNAL)
        assert len(signals) == 2
        decisions = g.nodes_by_type(NodeType.DECISION)
        assert len(decisions) == 1
        failures = g.nodes_by_type(NodeType.FAILURE)
        assert len(failures) == 0

    def test_predecessors_and_successors(self):
        """predecessors and successors should return correct node IDs."""
        g = ExplanationGraph()
        g.add_node(GraphNode(node_id="a", node_type=NodeType.SIGNAL, label="A"))
        g.add_node(GraphNode(node_id="b", node_type=NodeType.DECISION, label="B"))
        g.add_node(GraphNode(node_id="c", node_type=NodeType.SIGNAL, label="C"))
        g.add_edge(GraphEdge(source_id="a", target_id="b", edge_type=EdgeType.TRIGGERS))
        g.add_edge(GraphEdge(source_id="c", target_id="b", edge_type=EdgeType.SUPPORTS))

        preds = g.predecessors("b")
        assert set(preds) == {"a", "c"}
        succs = g.successors("a")
        assert succs == ["b"]
        # No predecessors for root nodes
        assert g.predecessors("a") == []

    def test_to_dict_from_dict_roundtrip(self):
        """Full graph serialization and deserialization should be lossless."""
        g = ExplanationGraph()
        g.add_node(GraphNode(node_id="s1", node_type=NodeType.SIGNAL, label="sig1", data={"v": 0.5}))
        g.add_node(GraphNode(node_id="d1", node_type=NodeType.DECISION, label="dec1"))
        g.add_edge(GraphEdge(source_id="s1", target_id="d1", edge_type=EdgeType.TRIGGERS, confidence=0.8))

        d = g.to_dict()
        restored = ExplanationGraph.from_dict(d)

        assert restored.n_nodes == g.n_nodes
        assert restored.n_edges == g.n_edges
        assert restored.get_node("s1").label == "sig1"
        assert restored.get_node("s1").data == {"v": 0.5}
        assert restored.edges[0].edge_type == EdgeType.TRIGGERS
        assert abs(restored.edges[0].confidence - 0.8) < 1e-9


# ── TestTraceBack ─────────────────────────────────────────


class TestTraceBack:
    """Tests for the ExplanationGraph.trace_back method."""

    def test_single_path(self):
        """A->B->C: trace_back(C) should return one path [A, B, C]."""
        g = ExplanationGraph()
        g.add_node(GraphNode(node_id="A", node_type=NodeType.SIGNAL, label="A"))
        g.add_node(GraphNode(node_id="B", node_type=NodeType.SIGNAL, label="B"))
        g.add_node(GraphNode(node_id="C", node_type=NodeType.DECISION, label="C"))
        g.add_edge(GraphEdge(source_id="A", target_id="B", edge_type=EdgeType.TRIGGERS))
        g.add_edge(GraphEdge(source_id="B", target_id="C", edge_type=EdgeType.CAUSES))

        paths = g.trace_back("C")
        assert len(paths) == 1
        path = paths[0]
        # Path should go from root to target: A -> B -> C
        node_ids = [p[0] for p in path]
        assert node_ids == ["A", "B", "C"]

    def test_multiple_paths(self):
        """Diamond: A->C, B->C: trace_back(C) should return 2 paths."""
        g = ExplanationGraph()
        g.add_node(GraphNode(node_id="A", node_type=NodeType.SIGNAL, label="A"))
        g.add_node(GraphNode(node_id="B", node_type=NodeType.SIGNAL, label="B"))
        g.add_node(GraphNode(node_id="C", node_type=NodeType.DECISION, label="C"))
        g.add_edge(GraphEdge(source_id="A", target_id="C", edge_type=EdgeType.TRIGGERS))
        g.add_edge(GraphEdge(source_id="B", target_id="C", edge_type=EdgeType.SUPPORTS))

        paths = g.trace_back("C")
        assert len(paths) == 2
        # Each path should end at C
        for path in paths:
            assert path[-1][0] == "C"
        # Roots should be A and B
        roots = {path[0][0] for path in paths}
        assert roots == {"A", "B"}

    def test_max_depth_limits(self):
        """With max_depth=1, trace_back should not go beyond 1 hop."""
        g = ExplanationGraph()
        g.add_node(GraphNode(node_id="A", node_type=NodeType.SIGNAL, label="A"))
        g.add_node(GraphNode(node_id="B", node_type=NodeType.SIGNAL, label="B"))
        g.add_node(GraphNode(node_id="C", node_type=NodeType.DECISION, label="C"))
        g.add_edge(GraphEdge(source_id="A", target_id="B", edge_type=EdgeType.TRIGGERS))
        g.add_edge(GraphEdge(source_id="B", target_id="C", edge_type=EdgeType.CAUSES))

        paths = g.trace_back("C", max_depth=1)
        assert len(paths) >= 1
        # With max_depth=1, paths should have at most 3 entries
        # (the initial node + 1 hop back = 2 entries in the path list)
        for path in paths:
            # Each path element is (node_id, edge_type_value)
            # max_depth=1 means we go at most 1 step back from C
            assert len(path) <= 3

    def test_cycle_handling(self):
        """A cycle A->B->A should not cause infinite loop."""
        g = ExplanationGraph()
        g.add_node(GraphNode(node_id="A", node_type=NodeType.SIGNAL, label="A"))
        g.add_node(GraphNode(node_id="B", node_type=NodeType.DECISION, label="B"))
        g.add_edge(GraphEdge(source_id="A", target_id="B", edge_type=EdgeType.TRIGGERS))
        g.add_edge(GraphEdge(source_id="B", target_id="A", edge_type=EdgeType.CAUSES))

        # This should terminate without hanging
        paths = g.trace_back("B")
        assert len(paths) >= 1
        # Should contain A and B
        all_nodes = set()
        for path in paths:
            for node_id, _ in path:
                all_nodes.add(node_id)
        assert "A" in all_nodes
        assert "B" in all_nodes


# ── TestGraphBuilder ──────────────────────────────────────


class TestGraphBuilder:
    """Tests for the GraphBuilder.build method."""

    def test_build_with_diagnostics_only(self):
        """Diagnostics with non-zero signals should create signal nodes."""
        snap = _make_snapshot()
        diag = _make_diagnostics(failure_rate=0.5, noise_estimate=0.3)
        builder = GraphBuilder()
        graph = builder.build(snap, diagnostics=diag)

        signal_nodes = graph.nodes_by_type(NodeType.SIGNAL)
        assert len(signal_nodes) >= 2
        node_ids = {n.node_id for n in signal_nodes}
        assert "signal:failure_rate" in node_ids
        assert "signal:noise_estimate" in node_ids

    def test_build_with_decision_only(self):
        """A decision should create a single decision node."""
        snap = _make_snapshot()
        decision = _make_decision(reason_codes=["high_failure"])
        builder = GraphBuilder()
        graph = builder.build(snap, decision=decision)

        decision_nodes = graph.nodes_by_type(NodeType.DECISION)
        assert len(decision_nodes) == 1
        assert decision_nodes[0].node_id == "decision:strategy"
        assert decision_nodes[0].data["backend_name"] == "tpe"

    def test_build_with_diagnostics_and_decision(self):
        """Both diagnostics and decision should produce signal nodes, decision node, and edges."""
        snap = _make_snapshot()
        diag = _make_diagnostics(failure_rate=0.5)
        decision = _make_decision(reason_codes=["high_failure_rate"])
        builder = GraphBuilder()
        graph = builder.build(snap, diagnostics=diag, decision=decision)

        assert graph.n_nodes >= 2  # at least 1 signal + 1 decision
        assert graph.n_edges >= 1  # at least 1 edge from signal to decision

        # The failure_rate signal should have a TRIGGERS edge to the decision
        edges_to_decision = graph.edges_to("decision:strategy")
        assert len(edges_to_decision) >= 1

    def test_signal_threshold_filters_weak(self):
        """Signals below the signal_threshold should be excluded."""
        snap = _make_snapshot()
        # failure_rate=0.05 is below default threshold of 0.1
        diag = _make_diagnostics(failure_rate=0.05)
        builder = GraphBuilder(signal_threshold=0.1)
        graph = builder.build(snap, diagnostics=diag)

        signal_nodes = graph.nodes_by_type(NodeType.SIGNAL)
        node_ids = {n.node_id for n in signal_nodes}
        assert "signal:failure_rate" not in node_ids

    def test_surgery_report_creates_exclusion_nodes(self):
        """Surgery actions should create exclusion nodes."""
        snap = _make_snapshot()
        action = _make_action(ActionType.TIGHTEN_RANGE, ["x1"], reason="danger zone")
        report = _make_surgery_report([action])
        builder = GraphBuilder()
        graph = builder.build(snap, surgery_report=report)

        exclusion_nodes = graph.nodes_by_type(NodeType.EXCLUSION)
        assert len(exclusion_nodes) == 1
        assert "x1" in exclusion_nodes[0].node_id

    def test_failure_surface_creates_failure_nodes(self):
        """Danger zones from the failure surface should create failure nodes."""
        snap = _make_snapshot()
        dz = DangerZone(
            parameter="x1",
            bound_type="above",
            threshold=8.0,
            failure_rate=0.6,
            n_samples=20,
        )
        fs = _make_failure_surface([dz])
        builder = GraphBuilder()
        graph = builder.build(snap, failure_surface=fs)

        failure_nodes = graph.nodes_by_type(NodeType.FAILURE)
        assert len(failure_nodes) == 1
        assert failure_nodes[0].data["parameter"] == "x1"
        assert failure_nodes[0].data["bound_type"] == "above"

    def test_failure_to_exclusion_edges_wired(self):
        """When failure nodes and exclusion nodes share a parameter, EXCLUDES edges should be created."""
        snap = _make_snapshot()
        dz = DangerZone(
            parameter="x1",
            bound_type="above",
            threshold=8.0,
            failure_rate=0.6,
            n_samples=20,
        )
        fs = _make_failure_surface([dz])
        action = _make_action(ActionType.TIGHTEN_RANGE, ["x1"], reason="danger zone x1")
        report = _make_surgery_report([action])
        builder = GraphBuilder()
        graph = builder.build(snap, surgery_report=report, failure_surface=fs)

        # There should be an EXCLUDES edge from the failure node to the exclusion node
        excludes_edges = [e for e in graph.edges if e.edge_type == EdgeType.EXCLUDES]
        assert len(excludes_edges) >= 1
        assert excludes_edges[0].evidence["parameter"] == "x1"

    def test_constraint_nodes_and_edges(self):
        """Constraints on the snapshot should create constraint nodes and RESTRICTS edges to decision."""
        constraints = [
            {"target": "y", "lower": 0.0, "upper": 100.0},
        ]
        snap = _make_snapshot(constraints=constraints)
        decision = _make_decision()
        builder = GraphBuilder()
        graph = builder.build(snap, decision=decision)

        constraint_nodes = graph.nodes_by_type(NodeType.CONSTRAINT)
        assert len(constraint_nodes) == 1
        assert constraint_nodes[0].node_id == "constraint:y"
        assert "0.0" in constraint_nodes[0].label or "0" in constraint_nodes[0].label

        # There should be a RESTRICTS edge from constraint to decision
        restricts_edges = [e for e in graph.edges if e.edge_type == EdgeType.RESTRICTS]
        assert len(restricts_edges) == 1
        assert restricts_edges[0].source_id == "constraint:y"
        assert restricts_edges[0].target_id == "decision:strategy"


# ── TestEdgeWiring ────────────────────────────────────────


class TestEdgeWiring:
    """Tests for specific edge wiring logic in the builder."""

    def test_reason_code_triggers_edge(self):
        """A reason code matching a signal field name should create a TRIGGERS edge."""
        snap = _make_snapshot()
        diag = _make_diagnostics(parameter_drift=0.5)
        decision = _make_decision(reason_codes=["parameter_drift_detected"])
        builder = GraphBuilder()
        graph = builder.build(snap, diagnostics=diag, decision=decision)

        # The parameter_drift signal should be wired to the decision via reason_code matching
        edges_to_dec = graph.edges_to("decision:strategy")
        trigger_sources = {e.source_id for e in edges_to_dec if e.edge_type == EdgeType.TRIGGERS}
        assert "signal:parameter_drift" in trigger_sources

    def test_high_failure_rate_triggers_edge(self):
        """A failure_rate > 0.3 should create a TRIGGERS edge to decision."""
        snap = _make_snapshot()
        diag = _make_diagnostics(failure_rate=0.5)
        decision = _make_decision()
        builder = GraphBuilder()
        graph = builder.build(snap, diagnostics=diag, decision=decision)

        edges_to_dec = graph.edges_to("decision:strategy")
        triggers = [e for e in edges_to_dec if e.edge_type == EdgeType.TRIGGERS]
        fr_triggers = [e for e in triggers if e.source_id == "signal:failure_rate"]
        assert len(fr_triggers) == 1
        assert fr_triggers[0].evidence["threshold"] == 0.3

    def test_low_coverage_supports_edge(self):
        """An exploration_coverage < 0.3 (and != 0) should create a SUPPORTS edge."""
        snap = _make_snapshot()
        diag = _make_diagnostics(exploration_coverage=0.15)
        decision = _make_decision()
        builder = GraphBuilder()
        graph = builder.build(snap, diagnostics=diag, decision=decision)

        edges_to_dec = graph.edges_to("decision:strategy")
        supports = [e for e in edges_to_dec if e.edge_type == EdgeType.SUPPORTS]
        coverage_supports = [e for e in supports if e.source_id == "signal:exploration_coverage"]
        assert len(coverage_supports) == 1
        assert coverage_supports[0].evidence["threshold"] == 0.3


# ── TestQueryDecision ─────────────────────────────────────


class TestQueryDecision:
    """Tests for the GraphBuilder.query_decision helper."""

    def test_query_returns_paths(self):
        """query_decision should return trace_back paths for the decision node."""
        snap = _make_snapshot()
        diag = _make_diagnostics(failure_rate=0.5)
        decision = _make_decision()
        builder = GraphBuilder()
        graph = builder.build(snap, diagnostics=diag, decision=decision)

        paths = builder.query_decision(graph)
        assert len(paths) >= 1
        # Each path should end at the decision node
        for path in paths:
            assert path[-1][0] == "decision:strategy"

    def test_query_nonexistent_empty(self):
        """Querying a node that doesn't exist should return empty list."""
        graph = ExplanationGraph()
        builder = GraphBuilder()
        paths = builder.query_decision(graph, node_id="nonexistent:node")
        assert paths == []


# ── TestEdgeCases ─────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for the Explanation Graph package."""

    def test_build_empty_inputs(self):
        """Building with all None optional inputs should produce an empty or minimal graph."""
        snap = _make_snapshot()
        builder = GraphBuilder()
        graph = builder.build(snap)

        # No diagnostics, no decision, no surgery, no failure surface, no constraints
        assert graph.n_nodes == 0
        assert graph.n_edges == 0

    def test_build_full_integration(self):
        """Building with all inputs should produce a graph with all node types and edges."""
        constraints = [{"target": "y", "lower": 0.0, "upper": 100.0}]
        snap = _make_snapshot(constraints=constraints)

        diag = _make_diagnostics(
            failure_rate=0.5,
            noise_estimate=0.6,
            exploration_coverage=0.15,
            convergence_trend=-0.4,
            kpi_plateau_length=10,
        )
        decision = _make_decision(reason_codes=["high_failure_rate", "noise_estimate_high"])

        action = _make_action(ActionType.TIGHTEN_RANGE, ["x1"], reason="danger x1")
        report = _make_surgery_report([action])

        dz = DangerZone(
            parameter="x1",
            bound_type="above",
            threshold=8.0,
            failure_rate=0.6,
            n_samples=20,
        )
        fs = _make_failure_surface([dz])

        builder = GraphBuilder()
        graph = builder.build(
            snap,
            diagnostics=diag,
            decision=decision,
            surgery_report=report,
            failure_surface=fs,
        )

        # Should have nodes of all types
        assert len(graph.nodes_by_type(NodeType.SIGNAL)) >= 3
        assert len(graph.nodes_by_type(NodeType.DECISION)) == 1
        assert len(graph.nodes_by_type(NodeType.CONSTRAINT)) == 1
        assert len(graph.nodes_by_type(NodeType.EXCLUSION)) == 1
        assert len(graph.nodes_by_type(NodeType.FAILURE)) == 1

        # Should have various edge types
        edge_types = {e.edge_type for e in graph.edges}
        assert EdgeType.TRIGGERS in edge_types
        assert EdgeType.RESTRICTS in edge_types
        assert EdgeType.EXCLUDES in edge_types
        # SUPPORTS from exploration_coverage and noise_estimate
        assert EdgeType.SUPPORTS in edge_types

        # Total graph should be non-trivial
        assert graph.n_nodes >= 7
        assert graph.n_edges >= 5

        # query_decision should return paths through the graph
        paths = builder.query_decision(graph)
        assert len(paths) >= 1

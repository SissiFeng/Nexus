"""Comprehensive tests for the fidelity graph and simulator modules.

Covers GateCondition, StageGate, GateDecision, FidelityStage,
FidelityGraph, CandidateTrajectory, FidelitySimulator, and all
edge cases.
"""

from __future__ import annotations

import math

import pytest

from optimization_copilot.workflow.fidelity_graph import (
    CandidateTrajectory,
    FidelityGraph,
    FidelityStage,
    GateCondition,
    GateDecision,
    StageGate,
)
from optimization_copilot.workflow.simulator import (
    FidelitySimulator,
    PerGateStats,
    SimulationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_graph() -> FidelityGraph:
    """Create a 3-stage linear fidelity graph: screening -> validation -> production."""
    graph = FidelityGraph()
    graph.add_stage(FidelityStage(
        name="screening", fidelity_level=1, cost=1.0,
        kpis=["quick_score"],
    ))
    graph.add_stage(FidelityStage(
        name="validation", fidelity_level=2, cost=10.0,
        kpis=["validated_score"],
    ))
    graph.add_stage(FidelityStage(
        name="production", fidelity_level=3, cost=100.0,
        kpis=["final_score"],
    ))

    # Gate: screening -> validation
    graph.add_gate(StageGate(
        from_stage="screening",
        to_stage="validation",
        conditions=[
            GateCondition(kpi_name="quick_score", operator=">=", threshold=0.5),
        ],
    ))

    # Gate: validation -> production
    graph.add_gate(StageGate(
        from_stage="validation",
        to_stage="production",
        conditions=[
            GateCondition(kpi_name="validated_score", operator=">=", threshold=0.7),
        ],
    ))

    return graph


def _make_linear_evaluator(
    quality: float,
) -> callable:
    """Create an evaluator where all KPIs are proportional to *quality*.

    Returns an evaluator(stage_name, parameters) -> dict.
    *quality* is read from ``parameters["quality"]``.
    """
    def evaluator(stage_name: str, parameters: dict) -> dict[str, float]:
        q = parameters["quality"]
        if stage_name == "screening":
            return {"quick_score": q * 0.9}
        elif stage_name == "validation":
            return {"validated_score": q * 0.95}
        elif stage_name == "production":
            return {"final_score": q}
        return {}
    return evaluator


def _make_candidates(
    n: int,
    quality_range: tuple[float, float] = (0.0, 1.0),
) -> list[dict]:
    """Create *n* candidates with linearly spaced quality values."""
    step = (quality_range[1] - quality_range[0]) / max(n - 1, 1)
    return [
        {
            "id": f"cand_{i}",
            "parameters": {
                "quality": quality_range[0] + i * step,
            },
        }
        for i in range(n)
    ]


# ===========================================================================
# GateCondition tests
# ===========================================================================


class TestGateCondition:
    """Tests for GateCondition.evaluate with all operators."""

    def test_gte_pass(self) -> None:
        cond = GateCondition("score", ">=", 0.5)
        assert cond.evaluate({"score": 0.5}) is True

    def test_gte_fail(self) -> None:
        cond = GateCondition("score", ">=", 0.5)
        assert cond.evaluate({"score": 0.49}) is False

    def test_gte_above(self) -> None:
        cond = GateCondition("score", ">=", 0.5)
        assert cond.evaluate({"score": 0.99}) is True

    def test_lte_pass(self) -> None:
        cond = GateCondition("error", "<=", 0.1)
        assert cond.evaluate({"error": 0.1}) is True

    def test_lte_fail(self) -> None:
        cond = GateCondition("error", "<=", 0.1)
        assert cond.evaluate({"error": 0.11}) is False

    def test_lte_below(self) -> None:
        cond = GateCondition("error", "<=", 0.1)
        assert cond.evaluate({"error": 0.0}) is True

    def test_gt_pass(self) -> None:
        cond = GateCondition("value", ">", 10.0)
        assert cond.evaluate({"value": 10.1}) is True

    def test_gt_fail_equal(self) -> None:
        cond = GateCondition("value", ">", 10.0)
        assert cond.evaluate({"value": 10.0}) is False

    def test_gt_fail_below(self) -> None:
        cond = GateCondition("value", ">", 10.0)
        assert cond.evaluate({"value": 9.9}) is False

    def test_lt_pass(self) -> None:
        cond = GateCondition("cost", "<", 50.0)
        assert cond.evaluate({"cost": 49.9}) is True

    def test_lt_fail_equal(self) -> None:
        cond = GateCondition("cost", "<", 50.0)
        assert cond.evaluate({"cost": 50.0}) is False

    def test_lt_fail_above(self) -> None:
        cond = GateCondition("cost", "<", 50.0)
        assert cond.evaluate({"cost": 50.1}) is False

    def test_missing_kpi_returns_false(self) -> None:
        cond = GateCondition("missing_kpi", ">=", 0.0)
        assert cond.evaluate({"other_kpi": 1.0}) is False

    def test_empty_kpi_dict(self) -> None:
        cond = GateCondition("score", ">=", 0.0)
        assert cond.evaluate({}) is False

    def test_invalid_operator_returns_false(self) -> None:
        cond = GateCondition("score", "==", 1.0)
        assert cond.evaluate({"score": 1.0}) is False

    def test_negative_threshold(self) -> None:
        cond = GateCondition("delta", ">=", -5.0)
        assert cond.evaluate({"delta": -4.0}) is True
        assert cond.evaluate({"delta": -6.0}) is False


# ===========================================================================
# StageGate tests
# ===========================================================================


class TestStageGate:
    """Tests for StageGate evaluation with 'all' and 'any' modes."""

    def test_all_mode_all_pass(self) -> None:
        gate = StageGate(
            from_stage="a", to_stage="b",
            conditions=[
                GateCondition("x", ">=", 1.0),
                GateCondition("y", ">=", 2.0),
            ],
            gate_mode="all",
        )
        result = gate.evaluate({"x": 1.0, "y": 2.0})
        assert result.passed is True
        assert len(result.condition_results) == 2

    def test_all_mode_one_fails(self) -> None:
        gate = StageGate(
            from_stage="a", to_stage="b",
            conditions=[
                GateCondition("x", ">=", 1.0),
                GateCondition("y", ">=", 2.0),
            ],
            gate_mode="all",
        )
        result = gate.evaluate({"x": 1.0, "y": 1.0})
        assert result.passed is False

    def test_all_mode_all_fail(self) -> None:
        gate = StageGate(
            from_stage="a", to_stage="b",
            conditions=[
                GateCondition("x", ">=", 10.0),
                GateCondition("y", ">=", 20.0),
            ],
            gate_mode="all",
        )
        result = gate.evaluate({"x": 1.0, "y": 1.0})
        assert result.passed is False

    def test_any_mode_one_passes(self) -> None:
        gate = StageGate(
            from_stage="a", to_stage="b",
            conditions=[
                GateCondition("x", ">=", 100.0),
                GateCondition("y", ">=", 2.0),
            ],
            gate_mode="any",
        )
        result = gate.evaluate({"x": 1.0, "y": 2.0})
        assert result.passed is True

    def test_any_mode_none_pass(self) -> None:
        gate = StageGate(
            from_stage="a", to_stage="b",
            conditions=[
                GateCondition("x", ">=", 100.0),
                GateCondition("y", ">=", 200.0),
            ],
            gate_mode="any",
        )
        result = gate.evaluate({"x": 1.0, "y": 1.0})
        assert result.passed is False

    def test_any_mode_all_pass(self) -> None:
        gate = StageGate(
            from_stage="a", to_stage="b",
            conditions=[
                GateCondition("x", ">=", 1.0),
                GateCondition("y", ">=", 1.0),
            ],
            gate_mode="any",
        )
        result = gate.evaluate({"x": 5.0, "y": 5.0})
        assert result.passed is True

    def test_any_mode_empty_conditions(self) -> None:
        gate = StageGate(
            from_stage="a", to_stage="b",
            conditions=[],
            gate_mode="any",
        )
        result = gate.evaluate({"x": 5.0})
        assert result.passed is False

    def test_gate_name_property(self) -> None:
        gate = StageGate(
            from_stage="alpha", to_stage="beta",
            conditions=[],
        )
        assert gate.name == "alpha->beta"


# ===========================================================================
# GateDecision tests
# ===========================================================================


class TestGateDecision:
    """Tests for GateDecision properties."""

    def test_summary_passed(self) -> None:
        gd = GateDecision(
            gate_name="a->b", passed=True,
            condition_results=[("x>=1.0", True), ("y>=2.0", True)],
        )
        summary = gd.summary
        assert "PASSED" in summary
        assert "a->b" in summary
        assert "x>=1.0=pass" in summary
        assert "y>=2.0=pass" in summary

    def test_summary_failed(self) -> None:
        gd = GateDecision(
            gate_name="a->b", passed=False,
            condition_results=[("x>=1.0", True), ("y>=2.0", False)],
        )
        summary = gd.summary
        assert "FAILED" in summary
        assert "y>=2.0=fail" in summary

    def test_summary_empty_conditions(self) -> None:
        gd = GateDecision(
            gate_name="a->b", passed=True,
            condition_results=[],
        )
        summary = gd.summary
        assert "PASSED" in summary
        assert "()" in summary


# ===========================================================================
# FidelityGraph tests
# ===========================================================================


class TestFidelityGraph:
    """Tests for FidelityGraph construction and queries."""

    def test_add_stage(self) -> None:
        graph = FidelityGraph()
        stage = FidelityStage("s1", fidelity_level=1, cost=1.0, kpis=["k1"])
        graph.add_stage(stage)
        assert len(graph) == 1
        assert "s1" in graph

    def test_add_duplicate_stage_raises(self) -> None:
        graph = FidelityGraph()
        stage = FidelityStage("s1", fidelity_level=1, cost=1.0, kpis=["k1"])
        graph.add_stage(stage)
        with pytest.raises(ValueError, match="already exists"):
            graph.add_stage(stage)

    def test_add_gate(self) -> None:
        graph = FidelityGraph()
        graph.add_stage(FidelityStage("a", 1, 1.0, ["k"]))
        graph.add_stage(FidelityStage("b", 2, 2.0, ["k"]))
        gate = StageGate("a", "b", [GateCondition("k", ">=", 0.5)])
        graph.add_gate(gate)
        assert len(graph.gates) == 1

    def test_add_gate_missing_source_raises(self) -> None:
        graph = FidelityGraph()
        graph.add_stage(FidelityStage("b", 2, 2.0, ["k"]))
        gate = StageGate("a", "b", [])
        with pytest.raises(ValueError, match="Source stage 'a' not in graph"):
            graph.add_gate(gate)

    def test_add_gate_missing_target_raises(self) -> None:
        graph = FidelityGraph()
        graph.add_stage(FidelityStage("a", 1, 1.0, ["k"]))
        gate = StageGate("a", "b", [])
        with pytest.raises(ValueError, match="Target stage 'b' not in graph"):
            graph.add_gate(gate)

    def test_get_stage(self) -> None:
        graph = FidelityGraph()
        stage = FidelityStage("s1", 1, 1.0, ["k1"])
        graph.add_stage(stage)
        assert graph.get_stage("s1") is stage

    def test_get_stage_missing_raises(self) -> None:
        graph = FidelityGraph()
        with pytest.raises(KeyError, match="not found"):
            graph.get_stage("nonexistent")

    def test_topological_order_sorts_by_fidelity_level(self) -> None:
        graph = FidelityGraph()
        graph.add_stage(FidelityStage("high", 3, 100.0, []))
        graph.add_stage(FidelityStage("low", 1, 1.0, []))
        graph.add_stage(FidelityStage("mid", 2, 10.0, []))
        order = graph.topological_order()
        assert order == ["low", "mid", "high"]

    def test_topological_order_empty_graph(self) -> None:
        graph = FidelityGraph()
        assert graph.topological_order() == []

    def test_stages_property(self) -> None:
        graph = FidelityGraph()
        s1 = FidelityStage("a", 1, 1.0, [])
        s2 = FidelityStage("b", 2, 2.0, [])
        graph.add_stage(s1)
        graph.add_stage(s2)
        assert graph.stages == [s1, s2]

    def test_gates_property(self) -> None:
        graph = _make_simple_graph()
        assert len(graph.gates) == 2

    def test_evaluate_gate(self) -> None:
        graph = _make_simple_graph()
        gate = graph.gates[0]
        decision = graph.evaluate_gate(gate, {"quick_score": 0.6})
        assert decision.passed is True

    def test_contains(self) -> None:
        graph = _make_simple_graph()
        assert "screening" in graph
        assert "nonexistent" not in graph

    def test_len(self) -> None:
        graph = _make_simple_graph()
        assert len(graph) == 3


# ===========================================================================
# FidelityGraph.run_candidate tests
# ===========================================================================


class TestRunCandidate:
    """Tests for running candidates through the fidelity graph."""

    def test_all_gates_pass_complete_trajectory(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        # quality=1.0 -> quick_score=0.9 (>=0.5), validated=0.95 (>=0.7)
        traj = graph.run_candidate("c1", {"quality": 1.0}, evaluator)
        assert traj.reached_final_stage is True
        assert traj.stages_completed == ["screening", "validation", "production"]
        assert traj.stages_skipped == []
        assert traj.total_cost == pytest.approx(111.0)

    def test_first_gate_fails_skips_remaining(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        # quality=0.3 -> quick_score=0.27 (<0.5)
        traj = graph.run_candidate("c2", {"quality": 0.3}, evaluator)
        assert traj.reached_final_stage is False
        assert "screening" in traj.stages_completed
        assert "validation" in traj.stages_skipped
        assert "production" in traj.stages_skipped

    def test_second_gate_fails(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        # quality=0.6 -> quick_score=0.54 (>=0.5) ok
        # validated_score=0.57 (<0.7) fail
        traj = graph.run_candidate("c3", {"quality": 0.6}, evaluator)
        assert traj.reached_final_stage is False
        assert "screening" in traj.stages_completed
        assert "validation" in traj.stages_completed
        assert "production" in traj.stages_skipped

    def test_first_stage_always_evaluated(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        # Even quality=0.0 should evaluate screening
        traj = graph.run_candidate("c4", {"quality": 0.0}, evaluator)
        assert "screening" in traj.stages_completed
        assert traj.total_cost >= 1.0

    def test_cost_accumulates_correctly(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        # quality=0.3 -> only screening (cost=1.0)
        traj = graph.run_candidate("c5", {"quality": 0.3}, evaluator)
        assert traj.total_cost == pytest.approx(1.0)

        # quality=0.6 -> screening + validation (cost=1+10=11)
        traj2 = graph.run_candidate("c6", {"quality": 0.6}, evaluator)
        assert traj2.total_cost == pytest.approx(11.0)

    def test_kpi_values_by_stage_populated(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        traj = graph.run_candidate("c7", {"quality": 1.0}, evaluator)
        assert "screening" in traj.kpi_values_by_stage
        assert "quick_score" in traj.kpi_values_by_stage["screening"]
        assert "validation" in traj.kpi_values_by_stage
        assert "production" in traj.kpi_values_by_stage

    def test_kpi_values_not_populated_for_skipped(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        traj = graph.run_candidate("c8", {"quality": 0.3}, evaluator)
        assert "validation" not in traj.kpi_values_by_stage
        assert "production" not in traj.kpi_values_by_stage

    def test_gate_decisions_recorded(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        traj = graph.run_candidate("c9", {"quality": 1.0}, evaluator)
        assert len(traj.gate_decisions) >= 2
        gate_names = [gd.gate_name for gd in traj.gate_decisions]
        assert "screening->validation" in gate_names
        assert "validation->production" in gate_names

    def test_candidate_id_preserved(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        traj = graph.run_candidate("my-id", {"quality": 0.5}, evaluator)
        assert traj.candidate_id == "my-id"

    def test_parameters_preserved(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        params = {"quality": 0.8, "extra": "data"}
        traj = graph.run_candidate("c10", params, evaluator)
        assert traj.parameters == params

    def test_single_stage_graph(self) -> None:
        graph = FidelityGraph()
        graph.add_stage(FidelityStage("only", 1, 5.0, ["k"]))

        def evaluator(stage_name: str, params: dict) -> dict:
            return {"k": 1.0}

        traj = graph.run_candidate("c11", {}, evaluator)
        assert traj.reached_final_stage is True
        assert traj.stages_completed == ["only"]
        assert traj.total_cost == pytest.approx(5.0)


# ===========================================================================
# FidelitySimulator tests
# ===========================================================================


class TestFidelitySimulator:
    """Tests for FidelitySimulator."""

    def test_basic_simulation(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        candidates = _make_candidates(10, (0.0, 1.0))
        sim = FidelitySimulator()
        result = sim.simulate(
            graph=graph,
            candidates=candidates,
            evaluator=evaluator,
            final_kpi_name="final_score",
            final_kpi_threshold=0.8,
        )
        assert result.n_candidates == 10
        assert isinstance(result.n_passed_all, int)
        assert isinstance(result.n_truly_good, int)

    def test_simulator_cost_savings(self) -> None:
        """KEY ACCEPTANCE TEST: gated screening must save cost."""
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        # 20 candidates: most will be filtered at early gates
        candidates = _make_candidates(20, (0.0, 1.0))
        sim = FidelitySimulator()
        result = sim.simulate(
            graph=graph,
            candidates=candidates,
            evaluator=evaluator,
            final_kpi_name="final_score",
            final_kpi_threshold=0.8,
        )
        # All 20 candidates through all 3 stages = 20*(1+10+100) = 2220
        assert result.cost_without_gates == pytest.approx(2220.0)
        # With gates, bad candidates are filtered early
        assert result.cost_with_gates < result.cost_without_gates
        assert result.cost_savings_fraction > 0.0

    def test_false_reject_rate(self) -> None:
        """KEY ACCEPTANCE TEST: false reject rate computed correctly."""
        # Build a graph where the screening gate is somewhat strict:
        # the gate uses quick_score >= 0.5, but quick_score = quality * 0.9
        # so quality >= 0.556 passes screening.
        # "Truly good" defined as final_score >= 0.5, where final_score = quality
        # So quality in [0.5, 0.556) are truly good but get rejected at screening.
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)

        # Make candidates with specific quality values to guarantee some
        # are truly good but gate-rejected.
        candidates = [
            {"id": "good_pass", "parameters": {"quality": 0.9}},  # truly good, passes
            {"id": "good_pass2", "parameters": {"quality": 0.8}},  # truly good, passes
            {"id": "good_reject", "parameters": {"quality": 0.52}},  # truly good (>=0.5), but quick_score=0.468 (<0.5) fails gate
            {"id": "bad", "parameters": {"quality": 0.2}},  # not truly good
            {"id": "bad2", "parameters": {"quality": 0.1}},  # not truly good
        ]

        sim = FidelitySimulator()
        result = sim.simulate(
            graph=graph,
            candidates=candidates,
            evaluator=evaluator,
            final_kpi_name="final_score",
            final_kpi_threshold=0.5,
        )

        # 3 truly good candidates (quality >= 0.5): good_pass, good_pass2, good_reject
        assert result.n_truly_good == 3
        # good_reject has quick_score = 0.52 * 0.9 = 0.468, fails screening gate
        # so 1 out of 3 truly good candidates is rejected
        assert result.false_reject_rate == pytest.approx(1.0 / 3.0)
        # Rate must be > 0 and < 1.0
        assert 0.0 < result.false_reject_rate < 1.0

    def test_per_gate_stats_tracking(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        candidates = _make_candidates(10, (0.0, 1.0))
        sim = FidelitySimulator()
        result = sim.simulate(
            graph=graph,
            candidates=candidates,
            evaluator=evaluator,
            final_kpi_name="final_score",
            final_kpi_threshold=0.8,
        )
        assert len(result.per_gate_stats) >= 1
        for gs in result.per_gate_stats:
            assert gs.n_evaluated == gs.n_passed + gs.n_failed
            if gs.n_evaluated > 0:
                assert gs.pass_rate == pytest.approx(
                    gs.n_passed / gs.n_evaluated
                )

    def test_simulation_result_to_dict(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        candidates = _make_candidates(5, (0.0, 1.0))
        sim = FidelitySimulator()
        result = sim.simulate(
            graph=graph,
            candidates=candidates,
            evaluator=evaluator,
            final_kpi_name="final_score",
            final_kpi_threshold=0.8,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "n_candidates" in d
        assert "n_passed_all" in d
        assert "n_truly_good" in d
        assert "false_reject_rate" in d
        assert "cost_with_gates" in d
        assert "cost_without_gates" in d
        assert "cost_savings_fraction" in d
        assert "per_gate_stats" in d
        assert isinstance(d["per_gate_stats"], list)
        # Trajectories should NOT be in to_dict output
        assert "trajectories" not in d

    def test_no_candidates(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        sim = FidelitySimulator()
        result = sim.simulate(
            graph=graph,
            candidates=[],
            evaluator=evaluator,
            final_kpi_name="final_score",
            final_kpi_threshold=0.8,
        )
        assert result.n_candidates == 0
        assert result.n_passed_all == 0
        assert result.n_truly_good == 0
        assert result.false_reject_rate == 0.0
        assert result.cost_with_gates == 0.0
        assert result.cost_without_gates == 0.0
        assert result.cost_savings_fraction == 0.0

    def test_all_candidates_pass(self) -> None:
        """When all candidates are high quality, all should pass gates."""
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        # All candidates have quality=1.0 -> all pass everything
        candidates = [
            {"id": f"c{i}", "parameters": {"quality": 1.0}}
            for i in range(5)
        ]
        sim = FidelitySimulator()
        result = sim.simulate(
            graph=graph,
            candidates=candidates,
            evaluator=evaluator,
            final_kpi_name="final_score",
            final_kpi_threshold=0.8,
        )
        assert result.n_passed_all == 5
        assert result.n_truly_good == 5
        assert result.false_reject_rate == 0.0
        # No savings when all pass
        assert result.cost_with_gates == result.cost_without_gates

    def test_no_candidates_truly_good(self) -> None:
        """When no candidate is truly good, false_reject_rate should be 0."""
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        # All candidates have quality=0.1 -> none are truly good (threshold 0.8)
        candidates = [
            {"id": f"c{i}", "parameters": {"quality": 0.1}}
            for i in range(5)
        ]
        sim = FidelitySimulator()
        result = sim.simulate(
            graph=graph,
            candidates=candidates,
            evaluator=evaluator,
            final_kpi_name="final_score",
            final_kpi_threshold=0.8,
        )
        assert result.n_truly_good == 0
        assert result.false_reject_rate == 0.0

    def test_minimize_direction(self) -> None:
        """Test with minimize direction for the final KPI."""
        graph = FidelityGraph()
        graph.add_stage(FidelityStage("fast", 1, 1.0, ["error_est"]))
        graph.add_stage(FidelityStage("precise", 2, 10.0, ["error"]))
        graph.add_gate(StageGate(
            from_stage="fast", to_stage="precise",
            conditions=[GateCondition("error_est", "<=", 0.5)],
        ))

        def evaluator(stage_name: str, params: dict) -> dict:
            q = params["quality"]
            if stage_name == "fast":
                return {"error_est": 1.0 - q}
            return {"error": 1.0 - q}

        candidates = [
            {"id": "good", "parameters": {"quality": 0.9}},  # error=0.1, truly good
            {"id": "bad", "parameters": {"quality": 0.2}},   # error=0.8, not good
        ]
        sim = FidelitySimulator()
        result = sim.simulate(
            graph=graph,
            candidates=candidates,
            evaluator=evaluator,
            final_kpi_name="error",
            final_kpi_threshold=0.3,
            final_kpi_direction="minimize",
        )
        assert result.n_truly_good == 1  # only "good" has error <= 0.3

    def test_trajectories_returned(self) -> None:
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        candidates = _make_candidates(3, (0.0, 1.0))
        sim = FidelitySimulator()
        result = sim.simulate(
            graph=graph,
            candidates=candidates,
            evaluator=evaluator,
            final_kpi_name="final_score",
            final_kpi_threshold=0.8,
        )
        assert len(result.trajectories) == 3
        for traj in result.trajectories:
            assert isinstance(traj, CandidateTrajectory)


# ===========================================================================
# PerGateStats tests
# ===========================================================================


class TestPerGateStats:
    """Tests for PerGateStats dataclass."""

    def test_construction(self) -> None:
        stats = PerGateStats(
            gate_name="a->b",
            n_evaluated=10,
            n_passed=7,
            n_failed=3,
            pass_rate=0.7,
        )
        assert stats.gate_name == "a->b"
        assert stats.n_evaluated == 10
        assert stats.n_passed == 7
        assert stats.n_failed == 3
        assert stats.pass_rate == pytest.approx(0.7)


# ===========================================================================
# CandidateTrajectory tests
# ===========================================================================


class TestCandidateTrajectory:
    """Tests for CandidateTrajectory dataclass."""

    def test_construction(self) -> None:
        traj = CandidateTrajectory(
            candidate_id="t1",
            parameters={"a": 1},
            stages_completed=["s1", "s2"],
            stages_skipped=["s3"],
            gate_decisions=[],
            total_cost=5.0,
            reached_final_stage=False,
        )
        assert traj.candidate_id == "t1"
        assert traj.reached_final_stage is False

    def test_kpi_values_default(self) -> None:
        traj = CandidateTrajectory(
            candidate_id="t2",
            parameters={},
            stages_completed=[],
            stages_skipped=[],
            gate_decisions=[],
            total_cost=0.0,
            reached_final_stage=False,
        )
        assert traj.kpi_values_by_stage == {}


# ===========================================================================
# FidelityStage tests
# ===========================================================================


class TestFidelityStage:
    """Tests for FidelityStage dataclass."""

    def test_construction(self) -> None:
        stage = FidelityStage(
            name="test", fidelity_level=2, cost=5.0, kpis=["k1", "k2"],
        )
        assert stage.name == "test"
        assert stage.fidelity_level == 2
        assert stage.cost == 5.0
        assert stage.kpis == ["k1", "k2"]
        assert stage.gates_out == []

    def test_gates_out_populated_by_graph(self) -> None:
        graph = _make_simple_graph()
        screening = graph.get_stage("screening")
        assert len(screening.gates_out) == 1
        assert screening.gates_out[0].to_stage == "validation"


# ===========================================================================
# Integration tests
# ===========================================================================


class TestIntegration:
    """End-to-end integration tests combining graph and simulator."""

    def test_full_pipeline_deterministic(self) -> None:
        """Results are deterministic given the same inputs."""
        graph = _make_simple_graph()
        evaluator = _make_linear_evaluator(1.0)
        candidates = _make_candidates(15, (0.0, 1.0))
        sim = FidelitySimulator()

        result1 = sim.simulate(
            graph=graph,
            candidates=candidates,
            evaluator=evaluator,
            final_kpi_name="final_score",
            final_kpi_threshold=0.8,
            seed=42,
        )
        result2 = sim.simulate(
            graph=graph,
            candidates=candidates,
            evaluator=evaluator,
            final_kpi_name="final_score",
            final_kpi_threshold=0.8,
            seed=42,
        )

        assert result1.n_passed_all == result2.n_passed_all
        assert result1.n_truly_good == result2.n_truly_good
        assert result1.false_reject_rate == result2.false_reject_rate
        assert result1.cost_with_gates == result2.cost_with_gates
        assert result1.cost_without_gates == result2.cost_without_gates

    def test_multi_condition_gate(self) -> None:
        """Test a graph with multi-condition gates."""
        graph = FidelityGraph()
        graph.add_stage(FidelityStage("s1", 1, 1.0, ["a", "b"]))
        graph.add_stage(FidelityStage("s2", 2, 10.0, ["c"]))

        graph.add_gate(StageGate(
            from_stage="s1", to_stage="s2",
            conditions=[
                GateCondition("a", ">=", 0.5),
                GateCondition("b", "<=", 0.3),
            ],
            gate_mode="all",
        ))

        def evaluator(stage_name: str, params: dict) -> dict:
            q = params["quality"]
            if stage_name == "s1":
                return {"a": q, "b": 1.0 - q}
            return {"c": q}

        # quality=0.8 -> a=0.8 (>=0.5 ok), b=0.2 (<=0.3 ok) -> passes
        traj = graph.run_candidate("multi1", {"quality": 0.8}, evaluator)
        assert traj.reached_final_stage is True

        # quality=0.3 -> a=0.3 (<0.5 fail) -> rejected
        traj2 = graph.run_candidate("multi2", {"quality": 0.3}, evaluator)
        assert traj2.reached_final_stage is False

    def test_any_mode_gate_integration(self) -> None:
        """Test a graph with an 'any' mode gate."""
        graph = FidelityGraph()
        graph.add_stage(FidelityStage("s1", 1, 1.0, ["a", "b"]))
        graph.add_stage(FidelityStage("s2", 2, 10.0, ["c"]))

        graph.add_gate(StageGate(
            from_stage="s1", to_stage="s2",
            conditions=[
                GateCondition("a", ">=", 0.9),
                GateCondition("b", ">=", 0.9),
            ],
            gate_mode="any",
        ))

        def evaluator(stage_name: str, params: dict) -> dict:
            if stage_name == "s1":
                return {"a": params["a"], "b": params["b"]}
            return {"c": 1.0}

        # a=0.95, b=0.1 -> a passes (>=0.9), 'any' mode -> gate passes
        traj = graph.run_candidate(
            "any1", {"a": 0.95, "b": 0.1}, evaluator,
        )
        assert traj.reached_final_stage is True

        # a=0.1, b=0.1 -> neither passes -> gate fails
        traj2 = graph.run_candidate(
            "any2", {"a": 0.1, "b": 0.1}, evaluator,
        )
        assert traj2.reached_final_stage is False

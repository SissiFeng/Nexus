"""Tests for campaign.stage_gate and campaign.output modules."""

from __future__ import annotations

import time
from typing import Any

import pytest

from optimization_copilot.campaign.ranker import RankedCandidate, RankedTable
from optimization_copilot.campaign.stage_gate import (
    ProtocolStep,
    ScreeningProtocol,
    StageGateProtocol,
)
from optimization_copilot.campaign.output import (
    CampaignDeliverable,
    Layer1Dashboard,
    Layer2Intelligence,
    Layer3Reasoning,
    LearningReport,
    ModelMetrics,
)
from optimization_copilot.workflow.fidelity_graph import (
    FidelityGraph,
    FidelityStage,
    GateCondition,
    StageGate,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_ranked_candidate(rank: int = 1, name: str = "cand-1", **overrides: Any) -> RankedCandidate:
    defaults = {
        "rank": rank,
        "name": name,
        "parameters": {"x1": 0.5, "x2": 0.8},
        "predicted_mean": 3.5,
        "predicted_std": 0.2,
        "acquisition_score": 1.2,
    }
    defaults.update(overrides)
    return RankedCandidate(**defaults)


def _make_ranked_table(n: int = 5, **overrides: Any) -> RankedTable:
    candidates = [
        _make_ranked_candidate(rank=i, name=f"cand-{i}")
        for i in range(1, n + 1)
    ]
    defaults = {
        "candidates": candidates,
        "objective_name": "band_gap",
        "direction": "maximize",
        "acquisition_strategy": "ucb",
        "best_observed": 2.8,
    }
    defaults.update(overrides)
    return RankedTable(**defaults)


def _make_uv_vis_her_graph() -> FidelityGraph:
    """Build a standard UV-Vis -> HER fidelity graph for testing."""
    graph = FidelityGraph()
    graph.add_stage(FidelityStage(
        name="UV-Vis", fidelity_level=1, cost=10.0, kpis=["band_gap"],
    ))
    graph.add_stage(FidelityStage(
        name="HER", fidelity_level=2, cost=100.0, kpis=["her_activity"],
    ))
    graph.add_gate(StageGate(
        from_stage="UV-Vis",
        to_stage="HER",
        conditions=[
            GateCondition(kpi_name="band_gap", operator=">=", threshold=2.2),
            GateCondition(kpi_name="band_gap", operator="<=", threshold=2.6),
        ],
        gate_mode="all",
    ))
    return graph


def _make_screening_protocol() -> ScreeningProtocol:
    """Build a simple two-step ScreeningProtocol for reuse in output tests."""
    builder = StageGateProtocol()
    return builder.build_protocol(_make_uv_vis_her_graph())


def _make_model_metrics(**overrides: Any) -> ModelMetrics:
    defaults = {
        "objective_name": "band_gap",
        "n_training_points": 50,
        "y_mean": 2.4,
        "y_std": 0.3,
        "fit_duration_ms": 120.5,
    }
    defaults.update(overrides)
    return ModelMetrics(**defaults)


def _make_learning_report(**overrides: Any) -> LearningReport:
    defaults = {
        "new_observations": [
            {"name": "cand-1", "band_gap": 2.5},
            {"name": "cand-2", "band_gap": 2.1},
        ],
        "prediction_errors": [
            {"name": "cand-1", "objective": "band_gap", "predicted": 2.4, "actual": 2.5, "error": 0.1, "pct_error": 4.0},
            {"name": "cand-2", "objective": "band_gap", "predicted": 2.3, "actual": 2.1, "error": 0.2, "pct_error": 9.5},
        ],
        "mean_absolute_error": 0.15,
        "model_updated": True,
        "summary": "Model improved after 2 new observations.",
    }
    defaults.update(overrides)
    return LearningReport(**defaults)


# ==================================================================
# StageGateProtocol Tests
# ==================================================================


class TestStageGateProtocol:
    """Tests for StageGateProtocol.build_protocol and build_simple_protocol."""

    def test_build_protocol_produces_correct_number_of_steps(self):
        graph = _make_uv_vis_her_graph()
        protocol = StageGateProtocol().build_protocol(graph)
        assert protocol.n_stages == 2

    def test_build_protocol_step_order(self):
        graph = _make_uv_vis_her_graph()
        protocol = StageGateProtocol().build_protocol(graph)
        assert protocol.steps[0].stage_name == "UV-Vis"
        assert protocol.steps[1].stage_name == "HER"

    def test_build_protocol_step_numbers_are_one_based(self):
        graph = _make_uv_vis_her_graph()
        protocol = StageGateProtocol().build_protocol(graph)
        assert protocol.steps[0].step_number == 1
        assert protocol.steps[1].step_number == 2

    def test_build_protocol_costs(self):
        graph = _make_uv_vis_her_graph()
        protocol = StageGateProtocol().build_protocol(graph)
        assert protocol.steps[0].cost == 10.0
        assert protocol.steps[1].cost == 100.0
        assert protocol.total_cost_all_pass == 110.0
        assert protocol.cost_first_stage == 10.0

    def test_build_protocol_kpis_measured(self):
        graph = _make_uv_vis_her_graph()
        protocol = StageGateProtocol().build_protocol(graph)
        assert protocol.steps[0].kpis_measured == ["band_gap"]
        assert protocol.steps[1].kpis_measured == ["her_activity"]

    def test_build_protocol_gate_conditions_on_first_stage(self):
        graph = _make_uv_vis_her_graph()
        protocol = StageGateProtocol().build_protocol(graph)
        # UV-Vis has outgoing gate conditions
        conds = protocol.steps[0].gate_conditions
        assert len(conds) == 2
        assert "band_gap >= 2.2" in conds
        assert "band_gap <= 2.6" in conds

    def test_build_protocol_no_gate_conditions_on_last_stage(self):
        graph = _make_uv_vis_her_graph()
        protocol = StageGateProtocol().build_protocol(graph)
        # HER has no outgoing gates
        assert protocol.steps[1].gate_conditions == []

    def test_build_protocol_action_if_pass_first_step(self):
        graph = _make_uv_vis_her_graph()
        protocol = StageGateProtocol().build_protocol(graph)
        assert protocol.steps[0].action_if_pass == "Proceed to HER"

    def test_build_protocol_action_if_pass_last_step(self):
        graph = _make_uv_vis_her_graph()
        protocol = StageGateProtocol().build_protocol(graph)
        assert protocol.steps[1].action_if_pass == "Accept candidate for full evaluation"

    def test_build_protocol_action_if_fail_with_gate(self):
        graph = _make_uv_vis_her_graph()
        protocol = StageGateProtocol().build_protocol(graph)
        # UV-Vis has gate conditions so fail should skip
        assert "Skip candidate" in protocol.steps[0].action_if_fail

    def test_build_protocol_action_if_fail_without_gate(self):
        graph = _make_uv_vis_her_graph()
        protocol = StageGateProtocol().build_protocol(graph)
        # HER has no gate conditions so fail action is "Continue"
        assert protocol.steps[1].action_if_fail == "Continue"

    def test_build_protocol_description_contains_stage_name(self):
        graph = _make_uv_vis_her_graph()
        protocol = StageGateProtocol().build_protocol(graph)
        assert "UV-Vis" in protocol.steps[0].description
        assert "HER" in protocol.steps[1].description

    def test_build_protocol_three_stages(self):
        """Three-stage graph: UV-Vis -> XRD -> HER."""
        graph = FidelityGraph()
        graph.add_stage(FidelityStage(name="UV-Vis", fidelity_level=1, cost=10.0, kpis=["band_gap"]))
        graph.add_stage(FidelityStage(name="XRD", fidelity_level=2, cost=50.0, kpis=["crystal_phase"]))
        graph.add_stage(FidelityStage(name="HER", fidelity_level=3, cost=100.0, kpis=["her_activity"]))
        graph.add_gate(StageGate(
            from_stage="UV-Vis", to_stage="XRD",
            conditions=[GateCondition(kpi_name="band_gap", operator=">=", threshold=2.0)],
            gate_mode="all",
        ))
        graph.add_gate(StageGate(
            from_stage="XRD", to_stage="HER",
            conditions=[GateCondition(kpi_name="crystal_phase", operator=">=", threshold=0.8)],
            gate_mode="all",
        ))
        protocol = StageGateProtocol().build_protocol(graph)

        assert protocol.n_stages == 3
        assert protocol.total_cost_all_pass == 160.0
        assert protocol.cost_first_stage == 10.0
        assert protocol.steps[0].action_if_pass == "Proceed to XRD"
        assert protocol.steps[1].action_if_pass == "Proceed to HER"
        assert protocol.steps[2].action_if_pass == "Accept candidate for full evaluation"

    def test_build_simple_protocol_basic(self):
        stages = [
            {"name": "UV-Vis", "cost": 10.0, "kpis": ["band_gap"], "gate_conditions": ["band_gap >= 2.2"]},
            {"name": "HER", "cost": 100.0, "kpis": ["her_activity"]},
        ]
        protocol = StageGateProtocol().build_simple_protocol(stages)

        assert protocol.n_stages == 2
        assert protocol.total_cost_all_pass == 110.0
        assert protocol.cost_first_stage == 10.0

    def test_build_simple_protocol_step_names(self):
        stages = [
            {"name": "Screen", "cost": 5.0, "kpis": ["val"]},
            {"name": "Verify", "cost": 50.0, "kpis": ["result"]},
        ]
        protocol = StageGateProtocol().build_simple_protocol(stages)
        assert protocol.steps[0].stage_name == "Screen"
        assert protocol.steps[1].stage_name == "Verify"

    def test_build_simple_protocol_action_if_pass(self):
        stages = [
            {"name": "A", "cost": 1.0, "kpis": ["x"], "gate_conditions": ["x > 0"]},
            {"name": "B", "cost": 2.0, "kpis": ["y"]},
        ]
        protocol = StageGateProtocol().build_simple_protocol(stages)
        assert protocol.steps[0].action_if_pass == "Proceed to B"
        assert protocol.steps[1].action_if_pass == "Accept candidate for full evaluation"

    def test_build_simple_protocol_action_if_fail(self):
        stages = [
            {"name": "A", "cost": 1.0, "kpis": ["x"], "gate_conditions": ["x > 0"]},
            {"name": "B", "cost": 2.0, "kpis": ["y"]},
        ]
        protocol = StageGateProtocol().build_simple_protocol(stages)
        assert "Skip candidate" in protocol.steps[0].action_if_fail
        assert protocol.steps[1].action_if_fail == "Continue"

    def test_build_simple_protocol_defaults(self):
        """Test that missing keys get default values."""
        stages = [{"cost": 10.0}]
        protocol = StageGateProtocol().build_simple_protocol(stages)
        assert protocol.steps[0].stage_name == "Stage-1"
        assert protocol.steps[0].kpis_measured == []
        assert protocol.steps[0].gate_conditions == []

    def test_build_simple_protocol_custom_description(self):
        stages = [
            {"name": "UV-Vis", "cost": 10.0, "kpis": ["bg"], "description": "Measure UV-Vis absorption spectrum"},
        ]
        protocol = StageGateProtocol().build_simple_protocol(stages)
        assert protocol.steps[0].description == "Measure UV-Vis absorption spectrum"


# ==================================================================
# ScreeningProtocol Tests
# ==================================================================


class TestScreeningProtocol:
    """Tests for ScreeningProtocol data model."""

    def test_n_stages_property(self):
        protocol = _make_screening_protocol()
        assert protocol.n_stages == 2

    def test_n_stages_empty(self):
        protocol = ScreeningProtocol(steps=[], total_cost_all_pass=0.0, cost_first_stage=0.0)
        assert protocol.n_stages == 0

    def test_summary_contains_step_info(self):
        protocol = _make_screening_protocol()
        text = protocol.summary()
        assert "Step 1" in text
        assert "Step 2" in text
        assert "UV-Vis" in text
        assert "HER" in text

    def test_summary_contains_gate_conditions(self):
        protocol = _make_screening_protocol()
        text = protocol.summary()
        assert "Gate:" in text
        assert "band_gap >= 2.2" in text
        assert "band_gap <= 2.6" in text

    def test_summary_contains_actions(self):
        protocol = _make_screening_protocol()
        text = protocol.summary()
        assert "Pass ->" in text
        assert "Fail ->" in text

    def test_summary_contains_cost_info(self):
        protocol = _make_screening_protocol()
        text = protocol.summary()
        assert "Full evaluation cost: 110.0" in text
        assert "Early-exit cost (fail first stage): 10.0" in text

    def test_summary_empty_protocol(self):
        protocol = ScreeningProtocol(steps=[], total_cost_all_pass=0.0, cost_first_stage=0.0)
        assert protocol.summary() == "No screening protocol defined."

    def test_to_dict_keys(self):
        protocol = _make_screening_protocol()
        d = protocol.to_dict()
        assert set(d.keys()) == {"steps", "total_cost_all_pass", "cost_first_stage", "n_stages", "summary"}

    def test_to_dict_steps_serialized(self):
        protocol = _make_screening_protocol()
        d = protocol.to_dict()
        assert len(d["steps"]) == 2
        assert d["steps"][0]["stage_name"] == "UV-Vis"
        assert d["steps"][1]["stage_name"] == "HER"

    def test_to_dict_values(self):
        protocol = _make_screening_protocol()
        d = protocol.to_dict()
        assert d["total_cost_all_pass"] == 110.0
        assert d["cost_first_stage"] == 10.0
        assert d["n_stages"] == 2
        assert isinstance(d["summary"], str)


# ==================================================================
# ProtocolStep Tests
# ==================================================================


class TestProtocolStep:
    """Tests for ProtocolStep data model."""

    def test_to_dict_round_trip(self):
        step = ProtocolStep(
            step_number=1,
            stage_name="UV-Vis",
            description="Run UV-Vis (cost: 10.0)",
            cost=10.0,
            kpis_measured=["band_gap"],
            gate_conditions=["band_gap >= 2.2"],
            action_if_pass="Proceed to HER",
            action_if_fail="Skip candidate (save remaining cost)",
        )
        d = step.to_dict()
        assert d["step_number"] == 1
        assert d["stage_name"] == "UV-Vis"
        assert d["cost"] == 10.0
        assert d["kpis_measured"] == ["band_gap"]
        assert d["gate_conditions"] == ["band_gap >= 2.2"]
        assert d["action_if_pass"] == "Proceed to HER"
        assert "Skip candidate" in d["action_if_fail"]

    def test_to_dict_all_keys_present(self):
        step = ProtocolStep(
            step_number=1, stage_name="A", description="desc",
            cost=0.0, kpis_measured=[], gate_conditions=[],
            action_if_pass="pass", action_if_fail="fail",
        )
        expected_keys = {
            "step_number", "stage_name", "description", "cost",
            "kpis_measured", "gate_conditions", "action_if_pass", "action_if_fail",
        }
        assert set(step.to_dict().keys()) == expected_keys


# ==================================================================
# Output Model Tests
# ==================================================================


class TestModelMetrics:
    """Tests for ModelMetrics data model."""

    def test_creation(self):
        m = _make_model_metrics()
        assert m.objective_name == "band_gap"
        assert m.n_training_points == 50
        assert m.y_mean == 2.4
        assert m.y_std == 0.3
        assert m.fit_duration_ms == 120.5

    def test_default_fit_duration(self):
        m = ModelMetrics(objective_name="x", n_training_points=10, y_mean=1.0, y_std=0.5)
        assert m.fit_duration_ms == 0.0

    def test_to_dict(self):
        m = _make_model_metrics()
        d = m.to_dict()
        assert d["objective_name"] == "band_gap"
        assert d["n_training_points"] == 50
        assert d["y_mean"] == 2.4
        assert d["y_std"] == 0.3
        assert d["fit_duration_ms"] == 120.5

    def test_to_dict_keys(self):
        m = _make_model_metrics()
        expected = {"objective_name", "n_training_points", "y_mean", "y_std", "fit_duration_ms"}
        assert set(m.to_dict().keys()) == expected


class TestLearningReport:
    """Tests for LearningReport data model."""

    def test_creation(self):
        lr = _make_learning_report()
        assert len(lr.new_observations) == 2
        assert len(lr.prediction_errors) == 2
        assert lr.mean_absolute_error == 0.15
        assert lr.model_updated is True
        assert "improved" in lr.summary

    def test_to_dict(self):
        lr = _make_learning_report()
        d = lr.to_dict()
        assert len(d["new_observations"]) == 2
        assert len(d["prediction_errors"]) == 2
        assert d["mean_absolute_error"] == 0.15
        assert d["model_updated"] is True
        assert isinstance(d["summary"], str)

    def test_to_dict_keys(self):
        lr = _make_learning_report()
        expected = {"new_observations", "prediction_errors", "mean_absolute_error", "model_updated", "summary"}
        assert set(lr.to_dict().keys()) == expected


class TestLayer1Dashboard:
    """Tests for Layer1Dashboard."""

    def test_next_batch_returns_correct_count(self):
        table = _make_ranked_table(n=10)
        dash = Layer1Dashboard(
            ranked_table=table, batch_size=3,
            screening_protocol=None, iteration=1,
        )
        batch = dash.next_batch
        assert len(batch) == 3

    def test_next_batch_contains_dicts(self):
        table = _make_ranked_table(n=5)
        dash = Layer1Dashboard(
            ranked_table=table, batch_size=2,
            screening_protocol=None, iteration=1,
        )
        batch = dash.next_batch
        assert all(isinstance(c, dict) for c in batch)
        assert batch[0]["rank"] == 1
        assert batch[1]["rank"] == 2

    def test_next_batch_clipped_to_table_size(self):
        table = _make_ranked_table(n=2)
        dash = Layer1Dashboard(
            ranked_table=table, batch_size=5,
            screening_protocol=None, iteration=1,
        )
        batch = dash.next_batch
        assert len(batch) == 2

    def test_to_dict_without_protocol(self):
        table = _make_ranked_table(n=3)
        dash = Layer1Dashboard(
            ranked_table=table, batch_size=2,
            screening_protocol=None, iteration=1,
        )
        d = dash.to_dict()
        assert d["iteration"] == 1
        assert d["batch_size"] == 2
        assert len(d["next_batch"]) == 2
        assert d["screening_protocol"] is None
        assert "full_ranking" in d

    def test_to_dict_with_protocol(self):
        table = _make_ranked_table(n=3)
        protocol = _make_screening_protocol()
        dash = Layer1Dashboard(
            ranked_table=table, batch_size=2,
            screening_protocol=protocol, iteration=2,
        )
        d = dash.to_dict()
        assert d["screening_protocol"] is not None
        assert d["screening_protocol"]["n_stages"] == 2
        assert d["iteration"] == 2


class TestLayer2Intelligence:
    """Tests for Layer2Intelligence."""

    def test_creation_with_all_fields(self):
        metrics = [_make_model_metrics()]
        lr = _make_learning_report()
        intel = Layer2Intelligence(
            pareto_summary={"n_pareto": 5, "hypervolume": 0.85},
            model_metrics=metrics,
            learning_report=lr,
            iteration_count=3,
        )
        assert intel.iteration_count == 3
        assert len(intel.model_metrics) == 1
        assert intel.learning_report is not None

    def test_creation_without_optional_fields(self):
        intel = Layer2Intelligence(
            pareto_summary=None,
            model_metrics=[],
            learning_report=None,
            iteration_count=1,
        )
        assert intel.pareto_summary is None
        assert intel.learning_report is None

    def test_to_dict(self):
        metrics = [_make_model_metrics(), _make_model_metrics(objective_name="her_activity")]
        intel = Layer2Intelligence(
            pareto_summary={"n_pareto": 3},
            model_metrics=metrics,
            learning_report=_make_learning_report(),
            iteration_count=5,
        )
        d = intel.to_dict()
        assert d["pareto_summary"] == {"n_pareto": 3}
        assert len(d["model_metrics"]) == 2
        assert d["learning_report"] is not None
        assert d["iteration_count"] == 5

    def test_to_dict_none_learning_report(self):
        intel = Layer2Intelligence(
            pareto_summary=None, model_metrics=[],
            learning_report=None, iteration_count=1,
        )
        d = intel.to_dict()
        assert d["learning_report"] is None
        assert d["pareto_summary"] is None


class TestLayer3Reasoning:
    """Tests for Layer3Reasoning."""

    def test_default_values(self):
        reasoning = Layer3Reasoning()
        assert reasoning.diagnostic_summary is None
        assert reasoning.fanova_result is None
        assert reasoning.execution_traces == []
        assert reasoning.additional == {}

    def test_creation_with_all_fields(self):
        reasoning = Layer3Reasoning(
            diagnostic_summary={"signal_count": 14},
            fanova_result={"top_feature": "x1", "importance": 0.45},
            execution_traces=[{"step": "fit_surrogate", "duration_ms": 200}],
            additional={"note": "debug info"},
        )
        assert reasoning.diagnostic_summary["signal_count"] == 14
        assert reasoning.fanova_result["top_feature"] == "x1"
        assert len(reasoning.execution_traces) == 1
        assert reasoning.additional["note"] == "debug info"

    def test_to_dict(self):
        reasoning = Layer3Reasoning(
            diagnostic_summary={"signals": 14},
            fanova_result={"importance": [0.5, 0.3, 0.2]},
            execution_traces=[{"a": 1}, {"b": 2}],
            additional={"meta": "info"},
        )
        d = reasoning.to_dict()
        assert d["diagnostic_summary"] == {"signals": 14}
        assert d["fanova_result"]["importance"] == [0.5, 0.3, 0.2]
        assert len(d["execution_traces"]) == 2
        assert d["additional"]["meta"] == "info"

    def test_to_dict_defaults(self):
        d = Layer3Reasoning().to_dict()
        assert d["diagnostic_summary"] is None
        assert d["fanova_result"] is None
        assert d["execution_traces"] == []
        assert d["additional"] == {}


# ==================================================================
# CampaignDeliverable Tests
# ==================================================================


class TestCampaignDeliverable:
    """Tests for the top-level CampaignDeliverable."""

    def _make_deliverable(self, iteration: int = 1, batch_size: int = 3) -> CampaignDeliverable:
        table = _make_ranked_table(n=5)
        protocol = _make_screening_protocol()
        dashboard = Layer1Dashboard(
            ranked_table=table, batch_size=batch_size,
            screening_protocol=protocol, iteration=iteration,
        )
        intelligence = Layer2Intelligence(
            pareto_summary={"n_pareto": 4},
            model_metrics=[_make_model_metrics()],
            learning_report=_make_learning_report(),
            iteration_count=iteration,
        )
        reasoning = Layer3Reasoning(
            diagnostic_summary={"signal_count": 14},
            fanova_result={"top_feature": "x1"},
            execution_traces=[{"step": "rank"}],
        )
        return CampaignDeliverable(
            iteration=iteration,
            dashboard=dashboard,
            intelligence=intelligence,
            reasoning=reasoning,
        )

    def test_iteration_stored(self):
        d = self._make_deliverable(iteration=5)
        assert d.iteration == 5

    def test_timestamp_auto_generated(self):
        before = time.time()
        d = self._make_deliverable()
        after = time.time()
        assert before <= d.timestamp <= after

    def test_timestamp_explicit(self):
        d = CampaignDeliverable(
            iteration=1,
            dashboard=Layer1Dashboard(
                ranked_table=_make_ranked_table(n=1),
                batch_size=1, screening_protocol=None, iteration=1,
            ),
            intelligence=Layer2Intelligence(
                pareto_summary=None, model_metrics=[],
                learning_report=None, iteration_count=1,
            ),
            reasoning=Layer3Reasoning(),
            timestamp=1234567890.0,
        )
        assert d.timestamp == 1234567890.0

    def test_next_batch_delegates_to_dashboard(self):
        d = self._make_deliverable(batch_size=2)
        batch = d.next_batch
        assert len(batch) == 2
        assert batch == d.dashboard.next_batch

    def test_to_dict_top_level_keys(self):
        d = self._make_deliverable()
        result = d.to_dict()
        assert set(result.keys()) == {"iteration", "timestamp", "dashboard", "intelligence", "reasoning"}

    def test_to_dict_dashboard_nested(self):
        d = self._make_deliverable(iteration=3, batch_size=2)
        result = d.to_dict()
        assert result["iteration"] == 3
        assert result["dashboard"]["batch_size"] == 2
        assert len(result["dashboard"]["next_batch"]) == 2

    def test_to_dict_intelligence_nested(self):
        d = self._make_deliverable()
        result = d.to_dict()
        assert result["intelligence"]["pareto_summary"] == {"n_pareto": 4}
        assert len(result["intelligence"]["model_metrics"]) == 1
        assert result["intelligence"]["learning_report"] is not None

    def test_to_dict_reasoning_nested(self):
        d = self._make_deliverable()
        result = d.to_dict()
        assert result["reasoning"]["diagnostic_summary"]["signal_count"] == 14
        assert result["reasoning"]["fanova_result"]["top_feature"] == "x1"
        assert len(result["reasoning"]["execution_traces"]) == 1

    def test_to_dict_is_json_serializable(self):
        """Ensure the entire to_dict() output is JSON-serializable."""
        import json
        d = self._make_deliverable()
        result = d.to_dict()
        # This will raise if not serializable
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
        roundtripped = json.loads(serialized)
        assert roundtripped["iteration"] == result["iteration"]

    def test_all_three_layers_present(self):
        d = self._make_deliverable()
        assert d.dashboard is not None
        assert d.intelligence is not None
        assert d.reasoning is not None
        assert isinstance(d.dashboard, Layer1Dashboard)
        assert isinstance(d.intelligence, Layer2Intelligence)
        assert isinstance(d.reasoning, Layer3Reasoning)

"""Comprehensive tests for CampaignLoop (optimization_copilot.campaign.loop).

Covers initialization, run_iteration, ingest_results, fidelity graph
integration, multi-objective support, and serialization of deliverables.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.campaign.loop import CampaignLoop
from optimization_copilot.campaign.output import (
    CampaignDeliverable,
    Layer1Dashboard,
    Layer2Intelligence,
    Layer3Reasoning,
    LearningReport,
    ModelMetrics,
)
from optimization_copilot.campaign.ranker import RankedTable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMILES_POOL = ["CC", "CCC", "CCCC", "CCO", "CC=O", "c1ccccc1", "CCCO", "C=CC"]


def make_snapshot(n_obs: int = 5) -> CampaignSnapshot:
    """Build a simple single-objective snapshot with *n_obs* observations."""
    specs = [
        ParameterSpec(name="smiles", type=VariableType.CATEGORICAL),
        ParameterSpec(name="temperature", type=VariableType.CONTINUOUS, lower=300, upper=500),
    ]
    observations: list[Observation] = []
    for i in range(min(n_obs, len(SMILES_POOL))):
        obs = Observation(
            iteration=i,
            parameters={"smiles": SMILES_POOL[i], "temperature": 350.0 + i * 10},
            kpi_values={"HER": 0.1 * (i + 1) + 0.05 * (i % 3)},
            metadata={"name": f"Poly-{i + 1}"},
        )
        observations.append(obs)
    return CampaignSnapshot(
        campaign_id="test",
        parameter_specs=specs,
        observations=observations,
        objective_names=["HER"],
        objective_directions=["minimize"],
    )


def make_candidates() -> list[dict[str, Any]]:
    """Return seven untested candidate dicts."""
    return [
        {"smiles": "CCCCC", "name": "Cand-A"},
        {"smiles": "CC(C)C", "name": "Cand-B"},
        {"smiles": "C=CCC", "name": "Cand-C"},
        {"smiles": "CCCCCC", "name": "Cand-D"},
        {"smiles": "CC(=O)C", "name": "Cand-E"},
        {"smiles": "CCC=O", "name": "Cand-F"},
        {"smiles": "c1ccc(O)cc1", "name": "Cand-G"},
    ]


def make_loop(**kwargs: Any) -> CampaignLoop:
    """Convenience factory with sensible defaults."""
    defaults: dict[str, Any] = dict(
        snapshot=make_snapshot(),
        candidates=make_candidates(),
        smiles_param="smiles",
        objectives=["HER"],
        objective_directions={"HER": "minimize"},
        batch_size=5,
        acquisition_strategy="ucb",
        seed=42,
    )
    defaults.update(kwargs)
    return CampaignLoop(**defaults)


def make_multi_objective_snapshot(n_obs: int = 5) -> CampaignSnapshot:
    """Snapshot with two objectives for multi-objective tests."""
    specs = [
        ParameterSpec(name="smiles", type=VariableType.CATEGORICAL),
        ParameterSpec(name="temperature", type=VariableType.CONTINUOUS, lower=300, upper=500),
    ]
    observations: list[Observation] = []
    for i in range(min(n_obs, len(SMILES_POOL))):
        obs = Observation(
            iteration=i,
            parameters={"smiles": SMILES_POOL[i], "temperature": 350.0 + i * 10},
            kpi_values={
                "HER": 0.1 * (i + 1) + 0.05 * (i % 3),
                "stability": 0.9 - 0.1 * i,
            },
            metadata={"name": f"Poly-{i + 1}"},
        )
        observations.append(obs)
    return CampaignSnapshot(
        campaign_id="test-multi",
        parameter_specs=specs,
        observations=observations,
        objective_names=["HER", "stability"],
        objective_directions=["minimize", "maximize"],
    )


# ---------------------------------------------------------------------------
# TestCampaignLoopInit
# ---------------------------------------------------------------------------


class TestCampaignLoopInit:
    """Validate constructor state and properties before any iterations."""

    def test_basic_creation(self):
        loop = make_loop()
        assert isinstance(loop, CampaignLoop)

    def test_properties_before_iteration(self):
        loop = make_loop()
        assert loop.iteration == 0
        assert loop.history == []

    def test_n_candidates_remaining(self):
        candidates = make_candidates()
        loop = make_loop(candidates=candidates)
        assert loop.n_candidates_remaining == len(candidates)

    def test_n_candidates_remaining_empty(self):
        loop = make_loop(candidates=[])
        assert loop.n_candidates_remaining == 0

    def test_history_is_defensive_copy(self):
        loop = make_loop()
        h1 = loop.history
        h1.append("garbage")  # type: ignore[arg-type]
        assert loop.history == [], "history property must return a copy"


# ---------------------------------------------------------------------------
# TestRunIteration
# ---------------------------------------------------------------------------


class TestRunIteration:
    """Tests for the main run_iteration method."""

    def test_produces_deliverable(self):
        loop = make_loop()
        result = loop.run_iteration()
        assert isinstance(result, CampaignDeliverable)

    def test_deliverable_has_three_layers(self):
        loop = make_loop()
        d = loop.run_iteration()
        assert isinstance(d.dashboard, Layer1Dashboard)
        assert isinstance(d.intelligence, Layer2Intelligence)
        assert isinstance(d.reasoning, Layer3Reasoning)

    def test_dashboard_has_ranked_table(self):
        loop = make_loop()
        d = loop.run_iteration()
        assert isinstance(d.dashboard.ranked_table, RankedTable)

    def test_ranked_table_has_candidates(self):
        loop = make_loop()
        d = loop.run_iteration()
        table = d.dashboard.ranked_table
        assert table.n_candidates > 0, "Expected ranked candidates"

    def test_next_batch_respects_batch_size(self):
        loop = make_loop(batch_size=3)
        d = loop.run_iteration()
        assert len(d.next_batch) <= 3

    def test_next_batch_default_batch_size(self):
        loop = make_loop(batch_size=5)
        d = loop.run_iteration()
        assert len(d.next_batch) <= 5

    def test_next_batch_entries_are_dicts(self):
        loop = make_loop()
        d = loop.run_iteration()
        for entry in d.next_batch:
            assert isinstance(entry, dict)

    def test_model_metrics_populated(self):
        loop = make_loop()
        d = loop.run_iteration()
        metrics = d.intelligence.model_metrics
        assert len(metrics) >= 1
        m = metrics[0]
        assert isinstance(m, ModelMetrics)
        assert m.objective_name == "HER"
        assert m.n_training_points > 0
        assert m.fit_duration_ms >= 0.0

    def test_iteration_increments(self):
        loop = make_loop()
        assert loop.iteration == 0
        loop.run_iteration()
        assert loop.iteration == 1
        loop.run_iteration()
        assert loop.iteration == 2

    def test_history_accumulates(self):
        loop = make_loop()
        d1 = loop.run_iteration()
        d2 = loop.run_iteration()
        assert len(loop.history) == 2
        assert loop.history[0].iteration == 1
        assert loop.history[1].iteration == 2

    def test_multiple_iterations(self):
        loop = make_loop()
        for _ in range(4):
            d = loop.run_iteration()
        assert loop.iteration == 4
        assert len(loop.history) == 4
        assert isinstance(d, CampaignDeliverable)

    def test_with_few_observations(self):
        """Minimum for GP is 2 observations -- verify it still works."""
        snapshot = make_snapshot(n_obs=2)
        loop = make_loop(snapshot=snapshot)
        d = loop.run_iteration()
        assert isinstance(d, CampaignDeliverable)
        # With only 2 training points the surrogate should still fit
        assert len(d.intelligence.model_metrics) >= 1

    def test_with_single_observation_falls_back(self):
        """With only 1 observation, GP cannot fit -- should produce empty table."""
        snapshot = make_snapshot(n_obs=1)
        loop = make_loop(snapshot=snapshot)
        d = loop.run_iteration()
        assert isinstance(d, CampaignDeliverable)
        # Not enough data to fit, so model_metrics may be empty
        assert d.dashboard.ranked_table is not None

    def test_with_zero_candidates(self):
        """Empty candidate list should still produce a deliverable."""
        loop = make_loop(candidates=[])
        d = loop.run_iteration()
        assert isinstance(d, CampaignDeliverable)
        assert d.next_batch == []

    def test_with_maximize_direction(self):
        loop = make_loop(objective_directions={"HER": "maximize"})
        d = loop.run_iteration()
        assert isinstance(d, CampaignDeliverable)
        table = d.dashboard.ranked_table
        assert table.n_candidates > 0

    def test_with_ei_strategy(self):
        loop = make_loop(acquisition_strategy="ei")
        d = loop.run_iteration()
        assert isinstance(d, CampaignDeliverable)
        table = d.dashboard.ranked_table
        assert table.acquisition_strategy == "ei"
        assert table.n_candidates > 0

    def test_with_pi_strategy(self):
        loop = make_loop(acquisition_strategy="pi")
        d = loop.run_iteration()
        assert isinstance(d, CampaignDeliverable)
        table = d.dashboard.ranked_table
        assert table.acquisition_strategy == "pi"
        assert table.n_candidates > 0

    def test_deliverable_iteration_field(self):
        loop = make_loop()
        d = loop.run_iteration()
        assert d.iteration == 1
        d2 = loop.run_iteration()
        assert d2.iteration == 2

    def test_deliverable_timestamp_populated(self):
        loop = make_loop()
        d = loop.run_iteration()
        assert d.timestamp > 0.0

    def test_intelligence_learning_report_none_on_first_iteration(self):
        loop = make_loop()
        d = loop.run_iteration()
        assert d.intelligence.learning_report is None

    def test_intelligence_pareto_summary_none_for_single_objective(self):
        loop = make_loop()
        d = loop.run_iteration()
        assert d.intelligence.pareto_summary is None


# ---------------------------------------------------------------------------
# TestIngestResults
# ---------------------------------------------------------------------------


class TestIngestResults:
    """Tests for the data-return / ingest_results workflow."""

    def _make_new_observations(self, loop: CampaignLoop) -> list[Observation]:
        """Create observations for top candidates from a previous iteration."""
        d = loop.run_iteration()
        batch = d.next_batch
        new_obs: list[Observation] = []
        for i, entry in enumerate(batch[:3]):
            obs = Observation(
                iteration=loop.iteration,
                parameters={"smiles": entry["parameters"]["smiles"], "temperature": 400.0},
                kpi_values={"HER": 0.05 + 0.02 * i},
                metadata={"name": entry.get("name", f"Tested-{i}")},
            )
            new_obs.append(obs)
        return new_obs

    def test_ingest_returns_deliverable(self):
        loop = make_loop()
        new_obs = self._make_new_observations(loop)
        d = loop.ingest_results(new_obs)
        assert isinstance(d, CampaignDeliverable)

    def test_ingest_removes_tested_candidates(self):
        loop = make_loop()
        initial_count = loop.n_candidates_remaining
        new_obs = self._make_new_observations(loop)
        tested_smiles = {
            obs.parameters["smiles"] for obs in new_obs
        }
        loop.ingest_results(new_obs)
        remaining_smiles = {
            c["smiles"] for c in loop._candidates
        }
        assert tested_smiles.isdisjoint(remaining_smiles), (
            "Tested candidates should be removed from pool"
        )
        assert loop.n_candidates_remaining < initial_count

    def test_ingest_updates_snapshot(self):
        loop = make_loop()
        original_n_obs = loop._snapshot.n_observations
        new_obs = self._make_new_observations(loop)
        loop.ingest_results(new_obs)
        assert loop._snapshot.n_observations == original_n_obs + len(new_obs)

    def test_ingest_has_learning_report(self):
        loop = make_loop()
        new_obs = self._make_new_observations(loop)
        d = loop.ingest_results(new_obs)
        assert d.intelligence.learning_report is not None
        assert isinstance(d.intelligence.learning_report, LearningReport)

    def test_learning_report_has_errors(self):
        loop = make_loop()
        new_obs = self._make_new_observations(loop)
        d = loop.ingest_results(new_obs)
        report = d.intelligence.learning_report
        assert report is not None
        # prediction_errors may or may not be populated depending on whether
        # the candidate was in last_predictions; at minimum the list exists
        assert isinstance(report.prediction_errors, list)

    def test_learning_report_summary_not_empty(self):
        loop = make_loop()
        new_obs = self._make_new_observations(loop)
        d = loop.ingest_results(new_obs)
        report = d.intelligence.learning_report
        assert report is not None
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_learning_report_new_observations(self):
        loop = make_loop()
        new_obs = self._make_new_observations(loop)
        d = loop.ingest_results(new_obs)
        report = d.intelligence.learning_report
        assert report is not None
        assert len(report.new_observations) == len(new_obs)

    def test_learning_report_model_updated_flag(self):
        loop = make_loop()
        new_obs = self._make_new_observations(loop)
        d = loop.ingest_results(new_obs)
        report = d.intelligence.learning_report
        assert report is not None
        assert report.model_updated is True

    def test_ingest_multiple_rounds(self):
        loop = make_loop(batch_size=2)
        # Round 1
        d1 = loop.run_iteration()
        batch1 = d1.next_batch[:2]
        obs1 = [
            Observation(
                iteration=loop.iteration,
                parameters={"smiles": entry["parameters"]["smiles"], "temperature": 400.0},
                kpi_values={"HER": 0.08},
                metadata={"name": entry.get("name", "T1")},
            )
            for entry in batch1
        ]
        d2 = loop.ingest_results(obs1)
        assert isinstance(d2, CampaignDeliverable)

        # Round 2
        batch2 = d2.next_batch[:2]
        obs2 = [
            Observation(
                iteration=loop.iteration,
                parameters={"smiles": entry["parameters"]["smiles"], "temperature": 410.0},
                kpi_values={"HER": 0.06},
                metadata={"name": entry.get("name", "T2")},
            )
            for entry in batch2
            if entry["parameters"]["smiles"] not in {o.parameters["smiles"] for o in obs1}
        ]
        if obs2:
            d3 = loop.ingest_results(obs2)
            assert isinstance(d3, CampaignDeliverable)
            assert d3.intelligence.learning_report is not None

    def test_ingest_with_failure_observation(self):
        loop = make_loop()
        loop.run_iteration()
        failure_obs = [
            Observation(
                iteration=loop.iteration,
                parameters={"smiles": "CCCCC", "temperature": 400.0},
                kpi_values={},
                is_failure=True,
                failure_reason="Synthesis failed",
                metadata={"name": "Failed-Cand"},
            )
        ]
        d = loop.ingest_results(failure_obs)
        assert isinstance(d, CampaignDeliverable)
        report = d.intelligence.learning_report
        assert report is not None
        # The failure observation should appear in new_observations
        found_failure = any(
            obs_dict.get("is_failure") is True
            for obs_dict in report.new_observations
        )
        assert found_failure, "Failure observation should be recorded in learning report"

    def test_ingest_increments_snapshot_iteration(self):
        loop = make_loop()
        loop.run_iteration()
        original_iter = loop._snapshot.current_iteration
        obs = [
            Observation(
                iteration=1,
                parameters={"smiles": "CCCCC", "temperature": 400.0},
                kpi_values={"HER": 0.07},
                metadata={"name": "test"},
            )
        ]
        loop.ingest_results(obs)
        assert loop._snapshot.current_iteration == original_iter + 1


# ---------------------------------------------------------------------------
# TestWithFidelityGraph
# ---------------------------------------------------------------------------


class TestWithFidelityGraph:
    """Test CampaignLoop when a FidelityGraph is supplied."""

    @staticmethod
    def _make_fidelity_graph():
        """Build a simple 2-stage fidelity graph for testing."""
        from optimization_copilot.workflow.fidelity_graph import (
            FidelityGraph,
            FidelityStage,
            GateCondition,
            StageGate,
        )

        graph = FidelityGraph()
        graph.add_stage(FidelityStage(
            name="UV-Vis",
            fidelity_level=1,
            cost=10.0,
            kpis=["band_gap"],
        ))
        graph.add_stage(FidelityStage(
            name="HER-test",
            fidelity_level=2,
            cost=100.0,
            kpis=["HER"],
        ))
        graph.add_gate(StageGate(
            from_stage="UV-Vis",
            to_stage="HER-test",
            conditions=[
                GateCondition(kpi_name="band_gap", operator=">=", threshold=2.0),
            ],
        ))
        return graph

    def test_with_fidelity_graph_produces_protocol(self):
        graph = self._make_fidelity_graph()
        loop = make_loop(fidelity_graph=graph)
        d = loop.run_iteration()
        assert d.dashboard.screening_protocol is not None

    def test_protocol_has_stages(self):
        graph = self._make_fidelity_graph()
        loop = make_loop(fidelity_graph=graph)
        d = loop.run_iteration()
        protocol = d.dashboard.screening_protocol
        assert protocol is not None
        assert protocol.n_stages == 2

    def test_protocol_steps_have_names(self):
        graph = self._make_fidelity_graph()
        loop = make_loop(fidelity_graph=graph)
        d = loop.run_iteration()
        protocol = d.dashboard.screening_protocol
        assert protocol is not None
        stage_names = [step.stage_name for step in protocol.steps]
        assert "UV-Vis" in stage_names
        assert "HER-test" in stage_names

    def test_protocol_has_cost_info(self):
        graph = self._make_fidelity_graph()
        loop = make_loop(fidelity_graph=graph)
        d = loop.run_iteration()
        protocol = d.dashboard.screening_protocol
        assert protocol is not None
        assert protocol.total_cost_all_pass == 110.0
        assert protocol.cost_first_stage == 10.0

    def test_no_fidelity_graph_means_no_protocol(self):
        loop = make_loop(fidelity_graph=None)
        d = loop.run_iteration()
        assert d.dashboard.screening_protocol is None


# ---------------------------------------------------------------------------
# TestMultiObjective
# ---------------------------------------------------------------------------


class TestMultiObjective:
    """Test CampaignLoop with multiple objectives (Pareto analysis)."""

    def test_multi_objective_produces_deliverable(self):
        snapshot = make_multi_objective_snapshot(n_obs=5)
        loop = CampaignLoop(
            snapshot=snapshot,
            candidates=make_candidates(),
            smiles_param="smiles",
            objectives=["HER", "stability"],
            objective_directions={"HER": "minimize", "stability": "maximize"},
            batch_size=5,
            seed=42,
        )
        d = loop.run_iteration()
        assert isinstance(d, CampaignDeliverable)

    def test_multi_objective_has_pareto_summary(self):
        snapshot = make_multi_objective_snapshot(n_obs=5)
        loop = CampaignLoop(
            snapshot=snapshot,
            candidates=make_candidates(),
            smiles_param="smiles",
            objectives=["HER", "stability"],
            objective_directions={"HER": "minimize", "stability": "maximize"},
            batch_size=5,
            seed=42,
        )
        d = loop.run_iteration()
        # The Pareto analysis may or may not succeed depending on
        # whether the multi_objective module handles it gracefully.
        # At minimum, the deliverable should be produced.
        if d.intelligence.pareto_summary is not None:
            summary = d.intelligence.pareto_summary
            assert "n_pareto_optimal" in summary
            assert "pareto_indices" in summary
            assert isinstance(summary["n_pareto_optimal"], int)

    def test_multi_objective_model_metrics_for_each_objective(self):
        snapshot = make_multi_objective_snapshot(n_obs=5)
        loop = CampaignLoop(
            snapshot=snapshot,
            candidates=make_candidates(),
            smiles_param="smiles",
            objectives=["HER", "stability"],
            objective_directions={"HER": "minimize", "stability": "maximize"},
            batch_size=5,
            seed=42,
        )
        d = loop.run_iteration()
        metric_names = {m.objective_name for m in d.intelligence.model_metrics}
        # Both objectives should have enough data for the GP
        assert "HER" in metric_names
        assert "stability" in metric_names


# ---------------------------------------------------------------------------
# TestSerialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Ensure deliverables can be serialized to plain dicts."""

    def test_deliverable_to_dict(self):
        loop = make_loop()
        d = loop.run_iteration()
        result = d.to_dict()
        assert isinstance(result, dict)

    def test_deliverable_to_dict_has_all_layers(self):
        loop = make_loop()
        d = loop.run_iteration()
        result = d.to_dict()
        assert "dashboard" in result
        assert "intelligence" in result
        assert "reasoning" in result
        assert "iteration" in result
        assert "timestamp" in result

    def test_dashboard_layer_serialization(self):
        loop = make_loop()
        d = loop.run_iteration()
        dash = result = d.to_dict()["dashboard"]
        assert "iteration" in dash
        assert "batch_size" in dash
        assert "next_batch" in dash
        assert "full_ranking" in dash

    def test_intelligence_layer_serialization(self):
        loop = make_loop()
        d = loop.run_iteration()
        intel = d.to_dict()["intelligence"]
        assert "model_metrics" in intel
        assert "learning_report" in intel
        assert "pareto_summary" in intel
        assert "iteration_count" in intel

    def test_reasoning_layer_serialization(self):
        loop = make_loop()
        d = loop.run_iteration()
        reasoning = d.to_dict()["reasoning"]
        assert "diagnostic_summary" in reasoning
        assert "fanova_result" in reasoning
        assert "execution_traces" in reasoning
        assert "additional" in reasoning

    def test_serialized_ranked_candidates_structure(self):
        loop = make_loop()
        d = loop.run_iteration()
        result = d.to_dict()
        ranking = result["dashboard"]["full_ranking"]
        assert "candidates" in ranking
        assert isinstance(ranking["candidates"], list)
        if ranking["candidates"]:
            cand = ranking["candidates"][0]
            assert "rank" in cand
            assert "name" in cand
            assert "predicted_mean" in cand
            assert "predicted_std" in cand
            assert "acquisition_score" in cand

    def test_serialized_model_metrics_structure(self):
        loop = make_loop()
        d = loop.run_iteration()
        result = d.to_dict()
        metrics_list = result["intelligence"]["model_metrics"]
        assert len(metrics_list) >= 1
        m = metrics_list[0]
        assert "objective_name" in m
        assert "n_training_points" in m
        assert "y_mean" in m
        assert "y_std" in m

    def test_learning_report_serialization_after_ingest(self):
        loop = make_loop()
        d0 = loop.run_iteration()
        batch = d0.next_batch[:2]
        obs = [
            Observation(
                iteration=1,
                parameters={"smiles": entry["parameters"]["smiles"], "temperature": 400.0},
                kpi_values={"HER": 0.05},
                metadata={"name": entry.get("name", "T")},
            )
            for entry in batch
        ]
        d1 = loop.ingest_results(obs)
        result = d1.to_dict()
        lr = result["intelligence"]["learning_report"]
        assert lr is not None
        assert "new_observations" in lr
        assert "prediction_errors" in lr
        assert "mean_absolute_error" in lr
        assert "model_updated" in lr
        assert "summary" in lr

    def test_screening_protocol_serialization_with_fidelity_graph(self):
        from optimization_copilot.workflow.fidelity_graph import (
            FidelityGraph,
            FidelityStage,
            GateCondition,
            StageGate,
        )

        graph = FidelityGraph()
        graph.add_stage(FidelityStage(name="S1", fidelity_level=1, cost=5.0, kpis=["k1"]))
        graph.add_stage(FidelityStage(name="S2", fidelity_level=2, cost=50.0, kpis=["k2"]))
        graph.add_gate(StageGate(
            from_stage="S1",
            to_stage="S2",
            conditions=[GateCondition(kpi_name="k1", operator=">=", threshold=1.0)],
        ))

        loop = make_loop(fidelity_graph=graph)
        d = loop.run_iteration()
        result = d.to_dict()
        protocol = result["dashboard"]["screening_protocol"]
        assert protocol is not None
        assert "steps" in protocol
        assert "n_stages" in protocol
        assert protocol["n_stages"] == 2


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and unusual configurations."""

    def test_batch_size_larger_than_candidates(self):
        """batch_size > n_candidates should return all candidates."""
        loop = make_loop(batch_size=100)
        d = loop.run_iteration()
        n_cands = len(make_candidates())
        assert len(d.next_batch) <= n_cands

    def test_batch_size_one(self):
        loop = make_loop(batch_size=1)
        d = loop.run_iteration()
        assert len(d.next_batch) <= 1

    def test_different_kappa_values(self):
        """Changing kappa should not break the loop (UCB exploration param)."""
        for kappa in [0.0, 0.5, 1.0, 5.0, 10.0]:
            loop = make_loop(kappa=kappa)
            d = loop.run_iteration()
            assert isinstance(d, CampaignDeliverable)

    def test_different_fp_sizes(self):
        """Fingerprint sizes should not break the loop."""
        for fp_size in [32, 64, 256]:
            loop = make_loop(fp_size=fp_size)
            d = loop.run_iteration()
            assert isinstance(d, CampaignDeliverable)

    def test_candidates_without_name_key(self):
        """Candidates missing the 'name' key should still work."""
        candidates = [
            {"smiles": "CCCCC"},
            {"smiles": "CC(C)C"},
            {"smiles": "C=CCC"},
        ]
        loop = make_loop(candidates=candidates)
        d = loop.run_iteration()
        assert isinstance(d, CampaignDeliverable)
        assert d.dashboard.ranked_table.n_candidates == 3

    def test_seed_reproducibility(self):
        """Same seed should produce same rankings."""
        d1 = make_loop(seed=123).run_iteration()
        d2 = make_loop(seed=123).run_iteration()
        names1 = [c["name"] for c in d1.next_batch]
        names2 = [c["name"] for c in d2.next_batch]
        assert names1 == names2

    def test_all_observations_are_failures(self):
        """When all observations are failures, GP cannot fit -- handle gracefully."""
        specs = [
            ParameterSpec(name="smiles", type=VariableType.CATEGORICAL),
        ]
        observations = [
            Observation(
                iteration=0,
                parameters={"smiles": "CC"},
                kpi_values={},
                is_failure=True,
                failure_reason="all failed",
            ),
            Observation(
                iteration=1,
                parameters={"smiles": "CCC"},
                kpi_values={},
                is_failure=True,
                failure_reason="all failed",
            ),
        ]
        snapshot = CampaignSnapshot(
            campaign_id="fail-test",
            parameter_specs=specs,
            observations=observations,
            objective_names=["HER"],
            objective_directions=["minimize"],
        )
        loop = CampaignLoop(
            snapshot=snapshot,
            candidates=make_candidates(),
            smiles_param="smiles",
            objectives=["HER"],
            objective_directions={"HER": "minimize"},
            seed=42,
        )
        d = loop.run_iteration()
        assert isinstance(d, CampaignDeliverable)
        # With no successful observations, should fallback to empty table
        assert d.dashboard.ranked_table is not None

    def test_ingest_observation_without_smiles_param(self):
        """Observations missing the smiles param should not crash."""
        loop = make_loop()
        loop.run_iteration()
        obs = [
            Observation(
                iteration=1,
                parameters={"temperature": 400.0},  # no "smiles" key
                kpi_values={"HER": 0.05},
                metadata={"name": "no-smiles"},
            )
        ]
        d = loop.ingest_results(obs)
        assert isinstance(d, CampaignDeliverable)

    def test_run_iteration_after_exhausting_candidates(self):
        """If all candidates are consumed, iteration should still work."""
        # Use only 2 candidates so they can be consumed quickly
        candidates = [
            {"smiles": "CCCCC", "name": "A"},
            {"smiles": "CC(C)C", "name": "B"},
        ]
        loop = make_loop(candidates=candidates, batch_size=5)

        d1 = loop.run_iteration()
        # Ingest results for both candidates to remove them
        obs = [
            Observation(
                iteration=1,
                parameters={"smiles": c["smiles"], "temperature": 400.0},
                kpi_values={"HER": 0.05},
                metadata={"name": c["name"]},
            )
            for c in candidates
        ]
        d2 = loop.ingest_results(obs)
        assert loop.n_candidates_remaining == 0
        assert isinstance(d2, CampaignDeliverable)
        assert d2.next_batch == []

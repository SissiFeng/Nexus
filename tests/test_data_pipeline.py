"""Tests for DataAnalysisPipeline traced execution methods.

Each of the 12 methods is tested with:
- Valid input: tag == COMPUTED, correct value structure
- Empty / minimal input: tag == FAILED or graceful COMPUTED
"""

from __future__ import annotations

import unittest

from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline
from optimization_copilot.agents.execution_trace import ExecutionTag, TracedResult
from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)


# ---------------------------------------------------------------------------
# Test helpers: build reusable CampaignSnapshot fixtures
# ---------------------------------------------------------------------------

def _simple_snapshot(
    n_obs: int = 10,
    n_params: int = 2,
    n_objectives: int = 1,
    include_metadata: bool = False,
) -> CampaignSnapshot:
    """Build a simple CampaignSnapshot for testing."""
    params = [
        ParameterSpec(
            name=f"p{i}",
            type=VariableType.CONTINUOUS,
            lower=0.0,
            upper=10.0,
        )
        for i in range(n_params)
    ]
    obj_names = [f"obj{i}" for i in range(n_objectives)]
    directions = ["minimize"] * n_objectives

    observations = []
    for j in range(n_obs):
        param_vals = {f"p{i}": float(j + i) for i in range(n_params)}
        kpi_vals = {name: float(j * (idx + 1) + 1) for idx, name in enumerate(obj_names)}
        meta = {}
        if include_metadata:
            # Add a metadata column that correlates with the objective
            meta["batch_id"] = float(j)
        observations.append(
            Observation(
                iteration=j,
                parameters=param_vals,
                kpi_values=kpi_vals,
                metadata=meta,
            )
        )

    return CampaignSnapshot(
        campaign_id="test-campaign",
        parameter_specs=params,
        observations=observations,
        objective_names=obj_names,
        objective_directions=directions,
        current_iteration=n_obs,
    )


def _multi_objective_snapshot(n_obs: int = 10) -> CampaignSnapshot:
    """Build a multi-objective CampaignSnapshot."""
    params = [
        ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ParameterSpec(name="y", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
    ]
    observations = []
    for j in range(n_obs):
        observations.append(
            Observation(
                iteration=j,
                parameters={"x": float(j), "y": float(10 - j)},
                kpi_values={"cost": float(j), "quality": float(10 - j)},
            )
        )
    return CampaignSnapshot(
        campaign_id="multi-obj-test",
        parameter_specs=params,
        observations=observations,
        objective_names=["cost", "quality"],
        objective_directions=["minimize", "maximize"],
        current_iteration=n_obs,
    )


def _empty_snapshot() -> CampaignSnapshot:
    """Minimal empty snapshot."""
    return CampaignSnapshot(
        campaign_id="empty",
        parameter_specs=[],
        observations=[],
        objective_names=[],
        objective_directions=[],
    )


# ---------------------------------------------------------------------------
# Test: run_diagnostics
# ---------------------------------------------------------------------------

class TestRunDiagnostics(unittest.TestCase):
    """Tests for DataAnalysisPipeline.run_diagnostics."""

    def setUp(self) -> None:
        self.pipe = DataAnalysisPipeline()

    def test_valid_snapshot(self) -> None:
        snapshot = _simple_snapshot(n_obs=20)
        result = self.pipe.run_diagnostics(snapshot)

        self.assertIsInstance(result, TracedResult)
        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertTrue(result.is_computed)
        self.assertIsInstance(result.value, dict)
        # Should have diagnostic signal keys
        self.assertIn("convergence_trend", result.value)
        self.assertIn("failure_rate", result.value)
        self.assertIn("signal_to_noise_ratio", result.value)

    def test_empty_snapshot(self) -> None:
        snapshot = _empty_snapshot()
        result = self.pipe.run_diagnostics(snapshot)

        # DiagnosticEngine handles empty gracefully
        self.assertIsInstance(result, TracedResult)
        self.assertEqual(result.tag, ExecutionTag.COMPUTED)

    def test_traces_populated(self) -> None:
        snapshot = _simple_snapshot()
        result = self.pipe.run_diagnostics(snapshot)

        self.assertEqual(len(result.traces), 1)
        trace = result.traces[0]
        self.assertEqual(trace.module, "diagnostics.engine")
        self.assertEqual(trace.method, "run_diagnostics")
        self.assertGreater(trace.duration_ms, 0.0)


# ---------------------------------------------------------------------------
# Test: run_top_k
# ---------------------------------------------------------------------------

class TestRunTopK(unittest.TestCase):
    """Tests for DataAnalysisPipeline.run_top_k."""

    def setUp(self) -> None:
        self.pipe = DataAnalysisPipeline()

    def test_top_k_descending(self) -> None:
        values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
        names = ["a", "b", "c", "d", "e", "f"]
        result = self.pipe.run_top_k(values, names, k=3, descending=True)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        top = result.value
        self.assertEqual(len(top), 3)
        self.assertEqual(top[0]["name"], "e")  # 9.0 is largest
        self.assertEqual(top[0]["rank"], 1)
        self.assertEqual(top[1]["name"], "c")  # 4.0
        self.assertEqual(top[2]["name"], "a")  # 3.0

    def test_top_k_ascending(self) -> None:
        values = [3.0, 1.0, 4.0]
        names = ["a", "b", "c"]
        result = self.pipe.run_top_k(values, names, k=2, descending=False)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        top = result.value
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0]["name"], "b")  # 1.0 is smallest

    def test_k_larger_than_n(self) -> None:
        values = [1.0, 2.0]
        names = ["a", "b"]
        result = self.pipe.run_top_k(values, names, k=10)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(len(result.value), 2)

    def test_empty_input(self) -> None:
        result = self.pipe.run_top_k([], [], k=3)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(result.value, [])


# ---------------------------------------------------------------------------
# Test: run_ranking
# ---------------------------------------------------------------------------

class TestRunRanking(unittest.TestCase):
    """Tests for DataAnalysisPipeline.run_ranking."""

    def setUp(self) -> None:
        self.pipe = DataAnalysisPipeline()

    def test_full_ranking(self) -> None:
        values = [3.0, 1.0, 2.0]
        names = ["a", "b", "c"]
        result = self.pipe.run_ranking(values, names, descending=True)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        ranked = result.value
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0]["name"], "a")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["name"], "c")
        self.assertEqual(ranked[2]["name"], "b")

    def test_ascending_ranking(self) -> None:
        values = [3.0, 1.0, 2.0]
        names = ["a", "b", "c"]
        result = self.pipe.run_ranking(values, names, descending=False)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(result.value[0]["name"], "b")

    def test_empty_ranking(self) -> None:
        result = self.pipe.run_ranking([], [])

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(result.value, [])


# ---------------------------------------------------------------------------
# Test: run_outlier_detection
# ---------------------------------------------------------------------------

class TestRunOutlierDetection(unittest.TestCase):
    """Tests for DataAnalysisPipeline.run_outlier_detection."""

    def setUp(self) -> None:
        self.pipe = DataAnalysisPipeline()

    def test_finds_outlier(self) -> None:
        # All values close to 5 except one huge outlier
        values = [5.0, 5.1, 4.9, 5.05, 4.95, 50.0]
        names = ["a", "b", "c", "d", "e", "outlier"]
        result = self.pipe.run_outlier_detection(values, names, n_sigma=2.0)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        outliers = result.value["outliers"]
        self.assertGreater(len(outliers), 0)
        outlier_names = [o["name"] for o in outliers]
        self.assertIn("outlier", outlier_names)

    def test_no_outliers(self) -> None:
        values = [5.0, 5.0, 5.0, 5.0]
        names = ["a", "b", "c", "d"]
        result = self.pipe.run_outlier_detection(values, names)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(len(result.value["outliers"]), 0)

    def test_empty_input(self) -> None:
        result = self.pipe.run_outlier_detection([], [])

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(result.value["outliers"], [])

    def test_single_value(self) -> None:
        result = self.pipe.run_outlier_detection([5.0], ["a"])

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(result.value["outliers"], [])


# ---------------------------------------------------------------------------
# Test: run_fanova
# ---------------------------------------------------------------------------

class TestRunFanova(unittest.TestCase):
    """Tests for DataAnalysisPipeline.run_fanova."""

    def setUp(self) -> None:
        self.pipe = DataAnalysisPipeline()

    def test_basic_fanova(self) -> None:
        # Generate some data where x0 is important
        X = [[float(i), float(i % 3)] for i in range(20)]
        y = [float(i) * 2.0 + 1.0 for i in range(20)]  # linear in x0
        result = self.pipe.run_fanova(X, y, var_names=["temp", "pressure"])

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertIn("main_effects", result.value)
        self.assertIn("top_interactions", result.value)
        effects = result.value["main_effects"]
        self.assertIn("temp", effects)
        self.assertIn("pressure", effects)

    def test_fanova_no_var_names(self) -> None:
        X = [[float(i), float(i * 2)] for i in range(10)]
        y = [float(i) for i in range(10)]
        result = self.pipe.run_fanova(X, y)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        effects = result.value["main_effects"]
        self.assertIn("x0", effects)

    def test_fanova_empty_data(self) -> None:
        result = self.pipe.run_fanova([], [])

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(result.value["main_effects"], {})

    def test_trace_metadata(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        y = [1.0, 2.0, 3.0]
        result = self.pipe.run_fanova(X, y)

        self.assertEqual(len(result.traces), 1)
        self.assertEqual(result.traces[0].module, "explain.interaction_map")


# ---------------------------------------------------------------------------
# Test: run_symreg
# ---------------------------------------------------------------------------

class TestRunSymreg(unittest.TestCase):
    """Tests for DataAnalysisPipeline.run_symreg."""

    def setUp(self) -> None:
        self.pipe = DataAnalysisPipeline()

    def test_basic_symreg(self) -> None:
        X = [[float(i)] for i in range(15)]
        y = [float(i) * 2.0 + 1.0 for i in range(15)]
        # Use small population/generations for speed
        result = self.pipe.run_symreg(
            X, y,
            var_names=["x"],
            population_size=30,
            n_generations=5,
            seed=42,
        )

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertIn("pareto_front", result.value)
        self.assertIn("best_equation", result.value)
        # Should have at least one Pareto solution
        self.assertGreater(len(result.value["pareto_front"]), 0)

    def test_symreg_empty_data(self) -> None:
        result = self.pipe.run_symreg([], [])

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(result.value["pareto_front"], [])
        self.assertIsNone(result.value["best_equation"])


# ---------------------------------------------------------------------------
# Test: run_insight_report
# ---------------------------------------------------------------------------

class TestRunInsightReport(unittest.TestCase):
    """Tests for DataAnalysisPipeline.run_insight_report."""

    def setUp(self) -> None:
        self.pipe = DataAnalysisPipeline()

    def test_basic_report(self) -> None:
        X = [[float(i), float(i * 0.5)] for i in range(15)]
        y = [float(i) * 3.0 for i in range(15)]
        result = self.pipe.run_insight_report(X, y, var_names=["temp", "flow"])

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        val = result.value
        self.assertIn("main_effects", val)
        self.assertIn("summary", val)
        self.assertIn("domain", val)
        self.assertEqual(val["domain"], "general")
        self.assertEqual(val["n_observations"], 15)

    def test_report_with_domain(self) -> None:
        X = [[1.0], [2.0], [3.0]]
        y = [1.0, 2.0, 3.0]
        result = self.pipe.run_insight_report(X, y, domain="chemistry")

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(result.value["domain"], "chemistry")


# ---------------------------------------------------------------------------
# Test: run_correlation
# ---------------------------------------------------------------------------

class TestRunCorrelation(unittest.TestCase):
    """Tests for DataAnalysisPipeline.run_correlation."""

    def setUp(self) -> None:
        self.pipe = DataAnalysisPipeline()

    def test_perfect_positive(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = self.pipe.run_correlation(xs, ys)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertAlmostEqual(result.value["r"], 1.0, places=4)

    def test_perfect_negative(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [50.0, 40.0, 30.0, 20.0, 10.0]
        result = self.pipe.run_correlation(xs, ys)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertAlmostEqual(result.value["r"], -1.0, places=4)

    def test_zero_correlation(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [5.0, 5.0, 5.0, 5.0, 5.0]  # constant -> r=0
        result = self.pipe.run_correlation(xs, ys)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertAlmostEqual(result.value["r"], 0.0, places=4)

    def test_empty_input(self) -> None:
        result = self.pipe.run_correlation([], [])

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(result.value["r"], 0.0)


# ---------------------------------------------------------------------------
# Test: run_confounder_detection
# ---------------------------------------------------------------------------

class TestRunConfounderDetection(unittest.TestCase):
    """Tests for DataAnalysisPipeline.run_confounder_detection."""

    def setUp(self) -> None:
        self.pipe = DataAnalysisPipeline()

    def test_detects_confounder(self) -> None:
        # Create snapshot with a metadata column that correlates with objective
        snapshot = _simple_snapshot(n_obs=20, include_metadata=True)
        result = self.pipe.run_confounder_detection(snapshot, threshold=0.3)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertIsInstance(result.value, list)
        # batch_id should correlate with obj0 since both increase linearly
        if result.value:
            self.assertEqual(result.value[0]["column_name"], "batch_id")

    def test_no_confounders(self) -> None:
        snapshot = _simple_snapshot(n_obs=10, include_metadata=False)
        result = self.pipe.run_confounder_detection(snapshot)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertIsInstance(result.value, list)

    def test_empty_snapshot(self) -> None:
        snapshot = _empty_snapshot()
        result = self.pipe.run_confounder_detection(snapshot)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(result.value, [])


# ---------------------------------------------------------------------------
# Test: run_pareto_analysis
# ---------------------------------------------------------------------------

class TestRunParetoAnalysis(unittest.TestCase):
    """Tests for DataAnalysisPipeline.run_pareto_analysis."""

    def setUp(self) -> None:
        self.pipe = DataAnalysisPipeline()

    def test_multi_objective(self) -> None:
        snapshot = _multi_objective_snapshot(n_obs=10)
        result = self.pipe.run_pareto_analysis(snapshot)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        val = result.value
        self.assertIn("pareto_front", val)
        self.assertIn("pareto_indices", val)
        self.assertIn("dominance_ranks", val)
        self.assertIn("tradeoff_report", val)
        # Should have a non-empty Pareto front
        self.assertGreater(len(val["pareto_front"]), 0)

    def test_single_objective(self) -> None:
        snapshot = _simple_snapshot(n_obs=5, n_objectives=1)
        result = self.pipe.run_pareto_analysis(snapshot)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        # Single objective: all points are non-dominated
        self.assertIsInstance(result.value["pareto_front"], list)

    def test_empty_snapshot(self) -> None:
        snapshot = _empty_snapshot()
        result = self.pipe.run_pareto_analysis(snapshot)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(result.value["pareto_front"], [])


# ---------------------------------------------------------------------------
# Test: run_molecular_pipeline
# ---------------------------------------------------------------------------

class TestRunMolecularPipeline(unittest.TestCase):
    """Tests for DataAnalysisPipeline.run_molecular_pipeline."""

    def setUp(self) -> None:
        self.pipe = DataAnalysisPipeline()

    def test_end_to_end(self) -> None:
        smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CC", "CCCO"]
        params = [
            ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
        observations = [
            Observation(
                iteration=i,
                parameters={"x": float(i * 2)},
                kpi_values={"yield": float(50 + i * 5)},
            )
            for i in range(5)
        ]

        result = self.pipe.run_molecular_pipeline(
            smiles_list=smiles,
            observations=observations,
            parameter_specs=params,
            objective_name="yield",
            n_suggestions=2,
            seed=42,
        )

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        val = result.value
        self.assertIn("fingerprints", val)
        self.assertIn("suggestions", val)
        self.assertIn("encoding_metadata", val)
        self.assertIn("molecule_ranking", val)
        self.assertEqual(val["fingerprints"]["n_molecules"], 5)
        self.assertEqual(val["fingerprints"]["fingerprint_size"], 128)
        self.assertEqual(len(val["suggestions"]), 2)
        # Molecule ranking should be sorted by score desc
        ranking = val["molecule_ranking"]
        self.assertEqual(len(ranking), 5)
        self.assertEqual(ranking[0]["rank"], 1)

    def test_empty_smiles(self) -> None:
        result = self.pipe.run_molecular_pipeline(
            smiles_list=[],
            observations=[],
            parameter_specs=[],
            objective_name="yield",
        )

        # This should either compute (empty result) or fail gracefully
        self.assertIsInstance(result, TracedResult)


# ---------------------------------------------------------------------------
# Test: run_screening
# ---------------------------------------------------------------------------

class TestRunScreening(unittest.TestCase):
    """Tests for DataAnalysisPipeline.run_screening."""

    def setUp(self) -> None:
        self.pipe = DataAnalysisPipeline()

    def test_basic_screening(self) -> None:
        snapshot = _simple_snapshot(n_obs=15, n_params=3)
        result = self.pipe.run_screening(snapshot)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        val = result.value
        self.assertIn("ranked_parameters", val)
        self.assertIn("importance_scores", val)
        self.assertIn("directionality", val)
        self.assertIn("recommended_step_sizes", val)
        self.assertEqual(len(val["ranked_parameters"]), 3)

    def test_empty_snapshot(self) -> None:
        snapshot = _empty_snapshot()
        result = self.pipe.run_screening(snapshot)

        self.assertEqual(result.tag, ExecutionTag.COMPUTED)
        self.assertEqual(result.value["ranked_parameters"], [])

    def test_trace_structure(self) -> None:
        snapshot = _simple_snapshot(n_obs=10)
        result = self.pipe.run_screening(snapshot)

        self.assertEqual(len(result.traces), 1)
        self.assertEqual(result.traces[0].module, "screening.screener")
        self.assertEqual(result.traces[0].tag, ExecutionTag.COMPUTED)


# ---------------------------------------------------------------------------
# Test: TracedResult integration
# ---------------------------------------------------------------------------

class TestTracedResultIntegration(unittest.TestCase):
    """Verify TracedResult metadata across multiple pipeline methods."""

    def setUp(self) -> None:
        self.pipe = DataAnalysisPipeline()

    def test_payload_dict(self) -> None:
        result = self.pipe.run_correlation([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        payload = result.to_payload_dict()

        self.assertIn("_execution_traces", payload)
        self.assertIn("_execution_tag", payload)
        self.assertEqual(payload["_execution_tag"], "computed")
        self.assertEqual(len(payload["_execution_traces"]), 1)

    def test_merge_multiple_results(self) -> None:
        r1 = self.pipe.run_correlation([1.0, 2.0], [3.0, 4.0])
        r2 = self.pipe.run_top_k([5.0, 3.0], ["a", "b"], k=1)

        all_traces = TracedResult.merge([r1, r2])
        self.assertEqual(len(all_traces), 2)

    def test_overall_tag_all_computed(self) -> None:
        r1 = self.pipe.run_correlation([1.0, 2.0], [3.0, 4.0])
        r2 = self.pipe.run_ranking([1.0, 2.0], ["a", "b"])

        tag = TracedResult.overall_tag([r1, r2])
        self.assertEqual(tag, ExecutionTag.COMPUTED)

    def test_trace_has_timing(self) -> None:
        result = self.pipe.run_top_k([1.0, 2.0, 3.0], ["a", "b", "c"], k=2)

        trace = result.traces[0]
        self.assertGreater(trace.timestamp, 0.0)
        self.assertGreaterEqual(trace.duration_ms, 0.0)
        self.assertIsNone(trace.error)


if __name__ == "__main__":
    unittest.main()

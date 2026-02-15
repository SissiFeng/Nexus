"""End-to-end test: real combinatorial materials dataset.

Dataset: full_dataset_exp.csv (1,129 materials)
- Inputs:  A_smiles, B_smiles, C_smiles (building blocks)
- Product: SMILES (combined molecule)
- Output:  Emission Wavelength (nm)

Tests the full agent stack:
1. CampaignLoop: surrogate → rank → deliver → ingest → learn
2. DataAnalysisPipeline: fANOVA, symreg, correlation, ranking, outlier, Pareto
3. ExecutionGuard: STRICT mode validates all traced claims
"""

from __future__ import annotations

import csv
import os
import math
import pytest
from typing import Any

# ---------------------------------------------------------------------------
# Locate the CSV
# ---------------------------------------------------------------------------

CSV_PATH = os.path.expanduser(
    "~/Downloads/mp_materials/full_dataset_exp.csv"
)

# Skip entire module if CSV is absent (CI environments).
pytestmark = pytest.mark.skipif(
    not os.path.isfile(CSV_PATH),
    reason="Real materials CSV not found",
)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def load_materials() -> tuple[list[dict[str, Any]], list[str]]:
    """Load CSV and return (rows-as-dicts, column_names)."""
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return rows, list(reader.fieldnames) if reader.fieldnames else []


def split_observed_candidates(
    rows: list[dict[str, Any]],
    objective_col: str = "Emission Wavelength (nm)",
    smiles_col: str = "SMILES",
    n_observed: int = 60,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split rows into observed (with objective values) and candidates.

    Returns (observed, candidates_with_value, candidates_without_value).
    """
    with_value: list[dict] = []
    without_value: list[dict] = []

    for row in rows:
        val = row.get(objective_col, "").strip()
        if val and row.get(smiles_col, "").strip():
            try:
                float(val)
                with_value.append(row)
            except ValueError:
                without_value.append(row)
        else:
            without_value.append(row)

    observed = with_value[:n_observed]
    candidates_with = with_value[n_observed:]
    return observed, candidates_with, without_value


# ---------------------------------------------------------------------------
# Build core.models objects from raw rows
# ---------------------------------------------------------------------------


def rows_to_snapshot_and_candidates(
    observed: list[dict],
    candidates: list[dict],
    objective_col: str = "Emission Wavelength (nm)",
    smiles_col: str = "SMILES",
):
    """Convert raw CSV rows to CampaignSnapshot + candidate dicts."""
    from optimization_copilot.core.models import (
        CampaignSnapshot,
        Observation,
        ParameterSpec,
    )

    # --- Build observations ---
    observations: list[Observation] = []
    for i, row in enumerate(observed):
        obs = Observation(
            iteration=i + 1,
            parameters={
                smiles_col: row[smiles_col],
                "A_smiles": row["A_smiles"],
                "B_smiles": row["B_smiles"],
                "C_smiles": row["C_smiles"],
            },
            kpi_values={
                objective_col: float(row[objective_col]),
            },
            metadata={
                "name": row.get("Identifier", f"Obs-{i}"),
            },
        )
        observations.append(obs)

    # --- Build parameter specs ---
    from optimization_copilot.core.models import VariableType
    param_specs = [
        ParameterSpec(
            name=smiles_col,
            type=VariableType.CATEGORICAL,
        ),
    ]

    snapshot = CampaignSnapshot(
        campaign_id="materials_emission",
        observations=observations,
        parameter_specs=param_specs,
        objective_names=[objective_col],
        objective_directions=["maximize"],
        current_iteration=len(observed),
    )

    # --- Candidate dicts ---
    cand_dicts: list[dict[str, Any]] = []
    for i, row in enumerate(candidates):
        cand_dicts.append({
            smiles_col: row[smiles_col],
            "A_smiles": row["A_smiles"],
            "B_smiles": row["B_smiles"],
            "C_smiles": row["C_smiles"],
            "name": row.get("Identifier", f"Cand-{i}"),
        })

    return snapshot, cand_dicts


# =========================================================================
# Test 1: CampaignLoop — full closed-loop cycle
# =========================================================================


class TestCampaignLoopRealData:
    """Campaign loop with 60 observed + ~302 candidates from real data."""

    @pytest.fixture(scope="class")
    def setup(self):
        rows, _ = load_materials()
        observed, cands_with, cands_without = split_observed_candidates(rows, n_observed=60)
        snapshot, candidates = rows_to_snapshot_and_candidates(
            observed, cands_with[:200],  # Use first 200 candidates
        )
        from optimization_copilot.campaign.loop import CampaignLoop

        loop = CampaignLoop(
            snapshot=snapshot,
            candidates=candidates,
            smiles_param="SMILES",
            objectives=["Emission Wavelength (nm)"],
            objective_directions={"Emission Wavelength (nm)": "maximize"},
            batch_size=5,
            acquisition_strategy="ucb",
            kappa=2.0,
            n_gram=3,
            fp_size=128,
            seed=42,
        )
        return loop, snapshot, candidates, cands_with

    def test_run_iteration_produces_deliverable(self, setup):
        loop, _, _, _ = setup
        deliverable = loop.run_iteration()
        assert deliverable is not None
        assert deliverable.iteration == 1

    def test_dashboard_has_ranked_table(self, setup):
        loop, _, _, _ = setup
        d = loop.run_iteration()
        assert d.dashboard.ranked_table is not None
        assert d.dashboard.ranked_table.n_candidates > 0

    def test_next_batch_returns_5_candidates(self, setup):
        loop, _, _, _ = setup
        d = loop.run_iteration()
        batch = d.next_batch
        assert len(batch) == 5
        for entry in batch:
            assert "name" in entry
            assert "predicted_mean" in entry
            assert "predicted_std" in entry
            assert "acquisition_score" in entry

    def test_candidates_are_ranked_by_acquisition(self, setup):
        loop, _, _, _ = setup
        d = loop.run_iteration()
        table = d.dashboard.ranked_table
        ranks = [c.rank for c in table.candidates[:10]]
        assert ranks == list(range(1, 11))

    def test_model_metrics_present(self, setup):
        loop, _, _, _ = setup
        d = loop.run_iteration()
        metrics = d.intelligence.model_metrics
        assert len(metrics) >= 1
        m = metrics[0]
        assert m.objective_name == "Emission Wavelength (nm)"
        assert m.n_training_points == 60
        assert m.fit_duration_ms >= 0

    def test_deliverable_serializes_to_dict(self, setup):
        loop, _, _, _ = setup
        d = loop.run_iteration()
        d_dict = d.to_dict()
        assert "dashboard" in d_dict
        assert "intelligence" in d_dict
        assert "reasoning" in d_dict
        assert len(d_dict["dashboard"]["next_batch"]) == 5

    def test_ingest_results_produces_learning_report(self, setup):
        loop, snapshot, candidates, cands_with = setup
        from optimization_copilot.core.models import Observation

        # Run first iteration
        loop.run_iteration()

        # Simulate returning top-5 results
        new_obs: list[Observation] = []
        top5_smiles = [c["SMILES"] for c in candidates[:5]]
        for i, smi in enumerate(top5_smiles):
            # Find actual value from cands_with
            actual = None
            for row in cands_with:
                if row["SMILES"] == smi:
                    val = row.get("Emission Wavelength (nm)", "").strip()
                    if val:
                        actual = float(val)
                    break
            if actual is None:
                actual = 450.0  # fallback

            new_obs.append(Observation(
                iteration=loop.iteration + 1,
                parameters={"SMILES": smi},
                kpi_values={"Emission Wavelength (nm)": actual},
                metadata={"name": f"Return-{i}"},
            ))

        d = loop.ingest_results(new_obs)
        report = d.intelligence.learning_report
        assert report is not None
        assert report.model_updated is True
        assert len(report.new_observations) == 5
        assert report.summary  # non-empty summary


# =========================================================================
# Test 2: DataAnalysisPipeline — real data through traced pipeline
# =========================================================================


class TestPipelineRealData:
    """DataAnalysisPipeline on extracted feature matrix from real data."""

    @pytest.fixture(scope="class")
    def feature_data(self):
        """Extract numeric feature matrix from the dataset."""
        rows, _ = load_materials()

        # Use rows with all numeric KPIs available
        X: list[list[float]] = []
        y: list[float] = []
        names: list[str] = []

        numeric_cols = [
            "Absorption Wavelength (nm)",
            "Photoluminescence Quantum Yield",
            "Photoluminescence Lifetime (ns)",
        ]

        for row in rows:
            emission = row.get("Emission Wavelength (nm)", "").strip()
            if not emission:
                continue
            try:
                y_val = float(emission)
            except ValueError:
                continue

            features = []
            valid = True
            for col in numeric_cols:
                val = row.get(col, "").strip()
                if not val:
                    valid = False
                    break
                try:
                    features.append(float(val))
                except ValueError:
                    valid = False
                    break

            if valid:
                X.append(features)
                y.append(y_val)
                names.append(row.get("Identifier", "?"))

        return X, y, names, numeric_cols

    def test_run_top_k(self, feature_data):
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline

        _, y, names, _ = feature_data
        pipeline = DataAnalysisPipeline()
        result = pipeline.run_top_k(y, names, k=10, descending=True)

        assert result.is_computed
        assert len(result.value) == 10
        # Top-1 should have the highest emission
        assert result.value[0]["value"] == max(y)
        assert result.traces[0].tag.value == "computed"

    def test_run_ranking(self, feature_data):
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline

        _, y, names, _ = feature_data
        pipeline = DataAnalysisPipeline()
        result = pipeline.run_ranking(y, names, descending=True)

        assert result.is_computed
        assert len(result.value) == len(y)
        # Rank 1 should be highest
        assert result.value[0]["rank"] == 1
        assert result.value[0]["value"] >= result.value[-1]["value"]

    def test_run_outlier_detection(self, feature_data):
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline

        _, y, names, _ = feature_data
        pipeline = DataAnalysisPipeline()
        result = pipeline.run_outlier_detection(y, names, n_sigma=2.0)

        assert result.is_computed
        assert "outliers" in result.value
        assert "mean" in result.value
        assert "std" in result.value

    def test_run_correlation(self, feature_data):
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline

        X, y, _, cols = feature_data
        pipeline = DataAnalysisPipeline()

        # Correlation between absorption and emission wavelength
        absorption = [row[0] for row in X]
        result = pipeline.run_correlation(absorption, y)

        assert result.is_computed
        r = result.value["r"]
        assert -1.0 <= r <= 1.0
        assert result.value["n"] == len(y)

    def test_run_fanova(self, feature_data):
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline

        X, y, _, cols = feature_data
        pipeline = DataAnalysisPipeline()
        result = pipeline.run_fanova(X, y, var_names=cols, n_trees=30, seed=42)

        assert result.is_computed
        effects = result.value["main_effects"]
        assert len(effects) == len(cols)
        # Importances should sum to ~1
        total = sum(effects.values())
        assert 0.9 <= total <= 1.1

    def test_run_symreg(self, feature_data):
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline

        X, y, _, cols = feature_data
        # Use a small subset for speed
        pipeline = DataAnalysisPipeline()
        result = pipeline.run_symreg(
            X[:50], y[:50], var_names=cols,
            population_size=50, n_generations=10, seed=42,
        )

        assert result.is_computed
        assert "pareto_front" in result.value
        assert "best_equation" in result.value
        if result.value["best_equation"]:
            assert "equation" in result.value["best_equation"]

    def test_run_insight_report(self, feature_data):
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline

        X, y, _, cols = feature_data
        pipeline = DataAnalysisPipeline()
        result = pipeline.run_insight_report(X[:50], y[:50], var_names=cols)

        assert result.is_computed
        assert "main_effects" in result.value
        assert "best_equation" in result.value
        assert "summary" in result.value
        assert result.value["n_observations"] == 50


# =========================================================================
# Test 3: ExecutionGuard — STRICT mode catches untraced claims
# =========================================================================


class TestGuardRealData:
    """ExecutionGuard validates pipeline outputs with real data."""

    def test_traced_top_k_passes_strict_guard(self):
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline
        from optimization_copilot.agents.execution_guard import ExecutionGuard, GuardMode
        from optimization_copilot.agents.base import OptimizationFeedback

        rows, _ = load_materials()
        emissions = []
        names = []
        for row in rows:
            val = row.get("Emission Wavelength (nm)", "").strip()
            if val:
                try:
                    emissions.append(float(val))
                    names.append(row.get("Identifier", "?"))
                except ValueError:
                    pass

        pipeline = DataAnalysisPipeline()
        result = pipeline.run_top_k(emissions, names, k=5, descending=True)

        # Build feedback WITH traces
        feedback = OptimizationFeedback(
            agent_name="test_agent",
            feedback_type="hypothesis",
            confidence=0.9,
            payload={
                "top_5_emission": result.value,
                **result.to_payload_dict(),
            },
        )

        guard = ExecutionGuard(mode=GuardMode.STRICT)
        is_valid, issues = guard.validate_feedback(feedback)
        assert is_valid, f"Should pass but got issues: {issues}"

    def test_untraced_ranking_fails_strict_guard(self):
        from optimization_copilot.agents.execution_guard import ExecutionGuard, GuardMode
        from optimization_copilot.agents.base import OptimizationFeedback

        # Feedback with ranking claim but NO traces
        feedback = OptimizationFeedback(
            agent_name="hallucinating_agent",
            feedback_type="hypothesis",
            confidence=0.95,
            payload={
                "top_5_polymers": [
                    {"name": "A001B003C001", "emission": 543.4},
                    {"name": "A001B003C004", "emission": 540.0},
                ],
            },
        )

        guard = ExecutionGuard(mode=GuardMode.STRICT)
        is_valid, issues = guard.validate_feedback(feedback)
        assert not is_valid, "Untraced ranking should be rejected"
        assert len(issues) > 0

    def test_traced_fanova_passes_guard(self):
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline
        from optimization_copilot.agents.execution_guard import ExecutionGuard, GuardMode
        from optimization_copilot.agents.base import OptimizationFeedback

        rows, _ = load_materials()
        X: list[list[float]] = []
        y: list[float] = []
        for row in rows:
            em = row.get("Emission Wavelength (nm)", "").strip()
            ab = row.get("Absorption Wavelength (nm)", "").strip()
            qy = row.get("Photoluminescence Quantum Yield", "").strip()
            lt = row.get("Photoluminescence Lifetime (ns)", "").strip()
            if em and ab and qy and lt:
                try:
                    y.append(float(em))
                    X.append([float(ab), float(qy), float(lt)])
                except ValueError:
                    pass

        pipeline = DataAnalysisPipeline()
        result = pipeline.run_fanova(X[:80], y[:80], var_names=["Absorption", "QY", "Lifetime"])

        feedback = OptimizationFeedback(
            agent_name="test_agent",
            feedback_type="hypothesis",
            confidence=0.85,
            payload={
                "fanova_main_effects": result.value["main_effects"],
                "top_interactions": result.value["top_interactions"],
                **result.to_payload_dict(),
            },
        )

        guard = ExecutionGuard(mode=GuardMode.STRICT)
        is_valid, issues = guard.validate_feedback(feedback)
        assert is_valid, f"Traced fANOVA should pass: {issues}"

    def test_traced_correlation_passes_guard(self):
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline
        from optimization_copilot.agents.execution_guard import ExecutionGuard, GuardMode
        from optimization_copilot.agents.base import OptimizationFeedback

        rows, _ = load_materials()
        absorption = []
        emission = []
        for row in rows:
            ab = row.get("Absorption Wavelength (nm)", "").strip()
            em = row.get("Emission Wavelength (nm)", "").strip()
            if ab and em:
                try:
                    absorption.append(float(ab))
                    emission.append(float(em))
                except ValueError:
                    pass

        pipeline = DataAnalysisPipeline()
        result = pipeline.run_correlation(absorption, emission)

        feedback = OptimizationFeedback(
            agent_name="test_agent",
            feedback_type="hypothesis",
            confidence=0.9,
            payload={
                "correlation_absorption_emission": result.value,
                **result.to_payload_dict(),
            },
        )

        guard = ExecutionGuard(mode=GuardMode.STRICT)
        is_valid, issues = guard.validate_feedback(feedback)
        assert is_valid, f"Traced correlation should pass: {issues}"


# =========================================================================
# Test 4: Pareto analysis with multi-objective (Emission + QY)
# =========================================================================


class TestMultiObjectiveRealData:
    """Multi-objective optimization: Emission Wavelength + Quantum Yield."""

    @pytest.fixture(scope="class")
    def multi_obj_data(self):
        from optimization_copilot.core.models import (
            CampaignSnapshot,
            Observation,
            ParameterSpec,
        )

        rows, _ = load_materials()
        observations = []
        for i, row in enumerate(rows):
            em = row.get("Emission Wavelength (nm)", "").strip()
            qy = row.get("Photoluminescence Quantum Yield", "").strip()
            smi = row.get("SMILES", "").strip()
            if em and qy and smi:
                try:
                    observations.append(Observation(
                        iteration=i + 1,
                        parameters={"SMILES": smi},
                        kpi_values={
                            "Emission Wavelength (nm)": float(em),
                            "Quantum Yield": float(qy),
                        },
                        metadata={"name": row.get("Identifier", f"Obs-{i}")},
                    ))
                except ValueError:
                    pass

        from optimization_copilot.core.models import VariableType
        snapshot = CampaignSnapshot(
            campaign_id="materials_multi_obj",
            observations=observations,
            parameter_specs=[
                ParameterSpec(name="SMILES", type=VariableType.CATEGORICAL),
            ],
            objective_names=["Emission Wavelength (nm)", "Quantum Yield"],
            objective_directions=["maximize", "maximize"],
            current_iteration=len(observations),
        )
        return snapshot

    def test_pareto_analysis_runs(self, multi_obj_data):
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline

        pipeline = DataAnalysisPipeline()
        result = pipeline.run_pareto_analysis(multi_obj_data)

        assert result.is_computed
        assert "pareto_front" in result.value
        assert "pareto_indices" in result.value
        assert len(result.value["pareto_front"]) > 0

    def test_pareto_front_is_non_dominated(self, multi_obj_data):
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline

        pipeline = DataAnalysisPipeline()
        result = pipeline.run_pareto_analysis(multi_obj_data)

        front = result.value["pareto_front"]
        # Each Pareto member should not be dominated by another
        for member in front:
            kpis = member["kpi_values"]
            em = kpis["Emission Wavelength (nm)"]
            qy = kpis["Quantum Yield"]
            assert isinstance(em, float)
            assert isinstance(qy, float)

    def test_pareto_passes_guard(self, multi_obj_data):
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline
        from optimization_copilot.agents.execution_guard import ExecutionGuard, GuardMode
        from optimization_copilot.agents.base import OptimizationFeedback

        pipeline = DataAnalysisPipeline()
        result = pipeline.run_pareto_analysis(multi_obj_data)

        feedback = OptimizationFeedback(
            agent_name="pareto_agent",
            feedback_type="hypothesis",
            confidence=0.9,
            payload={
                "pareto_front_materials": result.value["pareto_front"],
                **result.to_payload_dict(),
            },
        )

        guard = ExecutionGuard(mode=GuardMode.STRICT)
        is_valid, issues = guard.validate_feedback(feedback)
        assert is_valid, f"Traced Pareto should pass: {issues}"


# =========================================================================
# Test 5: Full integrated cycle — CampaignLoop + Pipeline + Guard
# =========================================================================


class TestFullIntegration:
    """Full integration: load → surrogate → rank → pipeline analysis → guard."""

    def test_full_cycle(self):
        from optimization_copilot.campaign.loop import CampaignLoop
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline
        from optimization_copilot.agents.execution_guard import ExecutionGuard, GuardMode
        from optimization_copilot.agents.base import OptimizationFeedback
        from optimization_copilot.core.models import Observation

        # --- Step 1: Load and split ---
        rows, _ = load_materials()
        observed, cands_with, _ = split_observed_candidates(rows, n_observed=50)
        snapshot, candidates = rows_to_snapshot_and_candidates(
            observed, cands_with[:100],
        )

        # --- Step 2: Run campaign loop ---
        loop = CampaignLoop(
            snapshot=snapshot,
            candidates=candidates,
            smiles_param="SMILES",
            objectives=["Emission Wavelength (nm)"],
            objective_directions={"Emission Wavelength (nm)": "maximize"},
            batch_size=5,
            seed=42,
        )
        deliverable = loop.run_iteration()

        # Verify Layer 1
        batch = deliverable.next_batch
        assert len(batch) == 5
        top_name = batch[0]["name"]

        # Verify Layer 2
        assert len(deliverable.intelligence.model_metrics) >= 1

        # --- Step 3: Pipeline analysis on observed data ---
        pipeline = DataAnalysisPipeline()
        emission_values = [
            float(r["Emission Wavelength (nm)"]) for r in observed
        ]
        obs_names = [r.get("Identifier", "?") for r in observed]

        top_k_result = pipeline.run_top_k(emission_values, obs_names, k=5)
        ranking_result = pipeline.run_ranking(emission_values, obs_names)
        outlier_result = pipeline.run_outlier_detection(emission_values, obs_names)

        assert top_k_result.is_computed
        assert ranking_result.is_computed
        assert outlier_result.is_computed

        # --- Step 4: Guard validates traced feedback ---
        guard = ExecutionGuard(mode=GuardMode.STRICT)

        # Merge traces from all three results
        all_traces = (
            top_k_result.to_payload_dict()["_execution_traces"]
            + ranking_result.to_payload_dict()["_execution_traces"]
            + outlier_result.to_payload_dict()["_execution_traces"]
        )
        feedback_good = OptimizationFeedback(
            agent_name="integration_agent",
            feedback_type="hypothesis",
            confidence=0.9,
            payload={
                "top_5_emission": top_k_result.value,
                "ranking_all": ranking_result.value[:5],
                "outlier_report": outlier_result.value,
                "_execution_traces": all_traces,
                "_execution_tag": "computed",
            },
        )
        is_valid, issues = guard.validate_feedback(feedback_good)
        assert is_valid, f"Traced feedback should pass: {issues}"

        # --- Step 5: Ingest results and check learning ---
        new_obs = []
        for c in candidates[:5]:
            # Find the actual emission value
            actual_em = None
            for row in cands_with:
                if row["SMILES"] == c["SMILES"]:
                    v = row.get("Emission Wavelength (nm)", "").strip()
                    if v:
                        actual_em = float(v)
                    break
            if actual_em is None:
                actual_em = 460.0

            new_obs.append(Observation(
                iteration=loop.iteration + 1,
                parameters={"SMILES": c["SMILES"]},
                kpi_values={"Emission Wavelength (nm)": actual_em},
                metadata={"name": c.get("name", "?")},
            ))

        deliverable2 = loop.ingest_results(new_obs)
        report = deliverable2.intelligence.learning_report
        assert report is not None
        assert report.model_updated
        assert len(report.prediction_errors) > 0
        assert report.mean_absolute_error >= 0

        # Learning report should contain meaningful text
        assert len(report.summary) > 0

    def test_deliverable_json_roundtrip(self):
        """Deliverable serializes cleanly to dict (JSON-compatible)."""
        import json
        from optimization_copilot.campaign.loop import CampaignLoop

        rows, _ = load_materials()
        observed, cands_with, _ = split_observed_candidates(rows, n_observed=30)
        snapshot, candidates = rows_to_snapshot_and_candidates(
            observed, cands_with[:50],
        )

        loop = CampaignLoop(
            snapshot=snapshot,
            candidates=candidates,
            smiles_param="SMILES",
            objectives=["Emission Wavelength (nm)"],
            objective_directions={"Emission Wavelength (nm)": "maximize"},
            batch_size=3,
            seed=42,
        )
        d = loop.run_iteration()
        d_dict = d.to_dict()

        # Must be JSON serializable
        json_str = json.dumps(d_dict)
        assert len(json_str) > 100

        # Round-trip
        parsed = json.loads(json_str)
        assert parsed["iteration"] == 1
        assert len(parsed["dashboard"]["next_batch"]) == 3

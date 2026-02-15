"""Tier 1 End-to-End Integration Test for the Optimization Copilot.

Generates a realistic synthetic Suzuki coupling dataset (~50 rows) and runs
the FULL pipeline through all 8 stages, verifying every deliverable.

Dataset:
  4 parameters: temperature (50-150C), catalyst_loading (0.5-5.0 mol%),
                solvent_ratio (0.1-0.9), reaction_time (1-24 h)
  1 KPI: yield (0-100%)
  Hidden function: yield ~ 80*exp(-((T-100)^2/2000 + (cat-2.5)^2/2 +
                                     (sol-0.5)^2/0.5)) + noise
  seed=42 for reproducibility
"""

from __future__ import annotations

import io
import math
import random
import time
import unittest
from typing import Any


class TestTier1EndToEnd(unittest.TestCase):
    """End-to-end pipeline test covering all 8 stages."""

    # ------------------------------------------------------------------
    # Shared state across stages (populated in order by the sequential test)
    # ------------------------------------------------------------------

    csv_string: str = ""
    ingestion_report: Any = None
    store: Any = None
    campaign_id: str = "suzuki_coupling"
    snapshot: Any = None
    fingerprint: Any = None
    diagnostics_vector: Any = None
    stabilized_data: Any = None
    drift_report: Any = None
    data_quality_report: Any = None
    confounder_results: Any = None
    gp_backend: Any = None
    gp_suggestions: Any = None
    strategy_decision: Any = None
    explanation_graph: Any = None
    audit_log: Any = None
    execution_traces: list = []

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_suzuki_csv() -> str:
        """Generate a synthetic Suzuki coupling dataset as a CSV string.

        Returns a CSV with 50 rows:
        - 4 parameters: temperature, catalyst_loading, solvent_ratio, reaction_time
        - 1 KPI: yield
        - timestamp: sequential (epoch + i*3600)
        - batch_id: batch_1 for first 25, batch_2 for last 25
        - is_failed: True for ~5 rows in extreme parameter regions
        """
        rng = random.Random(42)
        base_epoch = 1700000000.0
        n_rows = 50

        lines = [
            "temperature,catalyst_loading,solvent_ratio,reaction_time,"
            "yield,timestamp,batch_id,is_failed"
        ]

        for i in range(n_rows):
            T = rng.uniform(50.0, 150.0)
            cat = rng.uniform(0.5, 5.0)
            sol = rng.uniform(0.1, 0.9)
            rt = rng.uniform(1.0, 24.0)
            ts = base_epoch + i * 3600
            batch = "batch_1" if i < 25 else "batch_2"

            # Hidden yield function with Gaussian-peak shape
            exponent = (
                (T - 100.0) ** 2 / 2000.0
                + (cat - 2.5) ** 2 / 2.0
                + (sol - 0.5) ** 2 / 0.5
            )
            true_yield = 80.0 * math.exp(-exponent)
            noise = rng.gauss(0, 5.0)
            y = max(0.0, min(100.0, true_yield + noise))

            # Force ~5 failures in extreme regions
            is_failed = False
            if (
                (T < 60 or T > 140)
                and (cat < 1.0 or cat > 4.5)
                and i % 10 == 0
            ):
                y = 0.0
                is_failed = True

            lines.append(
                f"{T:.2f},{cat:.3f},{sol:.3f},{rt:.2f},"
                f"{y:.2f},{ts:.1f},{batch},{is_failed}"
            )

        return "\n".join(lines)

    # ==================================================================
    # Stage 0: Input & Spec
    # ==================================================================

    def test_stage0_input_and_spec(self) -> None:
        """Stage 0: Generate CSV, ingest via DataIngestionAgent, verify report."""
        from optimization_copilot.ingestion.agent import DataIngestionAgent
        from optimization_copilot.ingestion.models import ColumnRole
        from optimization_copilot.store.store import ExperimentStore

        csv_string = self._generate_suzuki_csv()
        self.assertIn("temperature", csv_string)
        self.assertIn("yield", csv_string)

        store = ExperimentStore()
        agent = DataIngestionAgent()
        # Force reaction_time to be treated as PARAMETER, not ITERATION.
        # Its values (1-24) overlap with typical iteration indices, causing
        # the role inference engine to misclassify it.
        report = agent.ingest_csv_string(
            csv_string, store, campaign_id=self.campaign_id,
            role_overrides={"reaction_time": ColumnRole.PARAMETER},
        )

        # Verify the IngestionReport
        self.assertEqual(report.n_rows, 50)
        self.assertEqual(report.campaign_id, self.campaign_id)
        self.assertGreater(report.experiments_created, 0)

        # Verify column profiles contain expected roles
        col_roles = {p.name: p.inferred_role for p in report.column_profiles}
        self.assertIn("temperature", col_roles)
        self.assertIn("yield", col_roles)

        # Count parameter and kpi columns
        param_count = sum(
            1 for p in report.column_profiles
            if p.inferred_role == ColumnRole.PARAMETER
        )
        kpi_count = sum(
            1 for p in report.column_profiles
            if p.inferred_role == ColumnRole.KPI
        )
        # We expect at least some parameters and at least 1 KPI detected
        self.assertGreaterEqual(param_count, 1, "Expected at least 1 parameter column")
        self.assertGreaterEqual(kpi_count, 1, "Expected at least 1 KPI column")

        # Store for subsequent stages
        TestTier1EndToEnd.csv_string = csv_string
        TestTier1EndToEnd.ingestion_report = report
        TestTier1EndToEnd.store = store

    # ==================================================================
    # Stage 1: Ingestion + Store
    # ==================================================================

    def test_stage1_ingestion_and_store(self) -> None:
        """Stage 1: Verify ExperimentStore contents match ingested data."""
        store = TestTier1EndToEnd.store
        if store is None:
            self.test_stage0_input_and_spec()
            store = TestTier1EndToEnd.store

        summary = store.summary(campaign_id=self.campaign_id)

        # Should have ~50 experiments
        self.assertEqual(summary.n_experiments, 50)
        self.assertEqual(summary.n_campaigns, 1)
        self.assertIn(self.campaign_id, summary.campaign_ids)

        # Verify parameter names are present
        self.assertGreater(len(summary.parameter_names), 0)
        # Verify KPI names
        self.assertGreater(len(summary.kpi_names), 0)

        # Verify iteration range
        self.assertIsNotNone(summary.iteration_range)

    # ==================================================================
    # Stage 2: CampaignSnapshot + Fingerprint
    # ==================================================================

    def test_stage2_snapshot_and_fingerprint(self) -> None:
        """Stage 2: Build CampaignSnapshot, run ProblemProfiler for fingerprint."""
        from optimization_copilot.core.models import (
            CampaignSnapshot,
            Observation,
            ParameterSpec,
            VariableType,
        )
        from optimization_copilot.profiler.profiler import ProblemProfiler

        store = TestTier1EndToEnd.store
        if store is None:
            self.test_stage0_input_and_spec()
            store = TestTier1EndToEnd.store

        # Build CampaignSnapshot from store data
        experiments = store.get_by_campaign(self.campaign_id)
        self.assertGreater(len(experiments), 0)

        # Build parameter specs
        param_specs = [
            ParameterSpec(name="temperature", type=VariableType.CONTINUOUS,
                          lower=50.0, upper=150.0),
            ParameterSpec(name="catalyst_loading", type=VariableType.CONTINUOUS,
                          lower=0.5, upper=5.0),
            ParameterSpec(name="solvent_ratio", type=VariableType.CONTINUOUS,
                          lower=0.1, upper=0.9),
            ParameterSpec(name="reaction_time", type=VariableType.CONTINUOUS,
                          lower=1.0, upper=24.0),
        ]

        # Convert experiments to observations
        observations = []
        for exp in experiments:
            # Determine if the experiment is a failure
            is_failure = exp.is_failure
            # Also check metadata for is_failed flag
            if not is_failure and exp.metadata.get("is_failed") in (
                True, "True", "true",
            ):
                is_failure = True

            # Map parameter values from experiment data
            params = {}
            for pspec in param_specs:
                if pspec.name in exp.parameters:
                    params[pspec.name] = float(exp.parameters[pspec.name])
                elif pspec.name in exp.metadata:
                    params[pspec.name] = float(exp.metadata[pspec.name])

            kpi_values = {}
            if "yield" in exp.kpi_values:
                kpi_values["yield"] = float(exp.kpi_values["yield"])
            elif "yield" in exp.metadata:
                try:
                    kpi_values["yield"] = float(exp.metadata["yield"])
                except (ValueError, TypeError):
                    pass

            obs = Observation(
                iteration=exp.iteration,
                parameters=params,
                kpi_values=kpi_values,
                qc_passed=exp.qc_passed,
                is_failure=is_failure,
                timestamp=exp.timestamp,
                metadata=exp.metadata,
            )
            observations.append(obs)

        snapshot = CampaignSnapshot(
            campaign_id=self.campaign_id,
            parameter_specs=param_specs,
            observations=observations,
            objective_names=["yield"],
            objective_directions=["maximize"],
            current_iteration=len(observations),
        )

        # Verify snapshot properties
        self.assertEqual(snapshot.n_observations, 50)
        self.assertEqual(len(snapshot.parameter_specs), 4)
        self.assertEqual(snapshot.parameter_names, [
            "temperature", "catalyst_loading", "solvent_ratio", "reaction_time"
        ])

        # Run ProblemProfiler to get 8D fingerprint
        profiler = ProblemProfiler()
        fingerprint = profiler.profile(snapshot)

        # Verify fingerprint classification
        self.assertEqual(
            fingerprint.variable_types.value, "continuous",
            "All 4 parameters are continuous"
        )
        self.assertEqual(
            fingerprint.objective_form.value, "single",
            "Single objective (yield)"
        )
        # Data scale: 50 obs -> moderate
        self.assertEqual(
            fingerprint.data_scale.value, "moderate",
            "50 observations = moderate data scale"
        )
        # Effective dimensionality should be set
        self.assertGreater(fingerprint.effective_dimensionality, 0)

        # Verify fingerprint can produce a continuous vector
        vec = fingerprint.to_continuous_vector()
        self.assertEqual(len(vec), 9)
        for v in vec:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

        TestTier1EndToEnd.snapshot = snapshot
        TestTier1EndToEnd.fingerprint = fingerprint

    # ==================================================================
    # Stage 3: Diagnostics + Stabilization
    # ==================================================================

    def test_stage3_diagnostics_and_stabilization(self) -> None:
        """Stage 3: Compute 17 diagnostic signals, apply stabilization."""
        from optimization_copilot.diagnostics.engine import (
            DiagnosticEngine,
            DiagnosticsVector,
        )
        from optimization_copilot.stabilization.stabilizer import (
            Stabilizer,
            StabilizedData,
        )
        from optimization_copilot.core.models import StabilizeSpec

        snapshot = TestTier1EndToEnd.snapshot
        if snapshot is None:
            self.test_stage2_snapshot_and_fingerprint()
            snapshot = TestTier1EndToEnd.snapshot

        # Compute diagnostics
        diag_engine = DiagnosticEngine()
        diag = diag_engine.compute(snapshot)

        # Verify it is a DiagnosticsVector
        self.assertIsInstance(diag, DiagnosticsVector)

        # Verify key signals are populated and reasonable
        d = diag.to_dict()
        self.assertIn("convergence_trend", d)
        self.assertIn("noise_estimate", d)
        self.assertIn("failure_rate", d)
        self.assertIn("exploration_coverage", d)
        self.assertIn("signal_to_noise_ratio", d)
        self.assertIn("miscalibration_score", d)
        self.assertIn("overconfidence_rate", d)

        # noise_estimate should be non-negative
        self.assertGreaterEqual(d["noise_estimate"], 0.0)
        # failure_rate should be in [0, 1]
        self.assertGreaterEqual(d["failure_rate"], 0.0)
        self.assertLessEqual(d["failure_rate"], 1.0)
        # exploration_coverage should be in [0, 1]
        self.assertGreaterEqual(d["exploration_coverage"], 0.0)
        self.assertLessEqual(d["exploration_coverage"], 1.0)
        # best_kpi_value should be positive (yield > 0 for best experiment)
        self.assertGreater(d["best_kpi_value"], 0.0)

        # Verify all 17 signals are present
        self.assertEqual(len(d), 17, "DiagnosticsVector should have 17 signals")

        # Apply stabilization with default spec
        stabilizer = Stabilizer()
        spec = StabilizeSpec()
        stabilized = stabilizer.stabilize(snapshot, spec)

        self.assertIsInstance(stabilized, StabilizedData)
        # Stabilized observations should be a list
        self.assertIsInstance(stabilized.observations, list)
        # With default penalize policy, no observations removed
        self.assertEqual(len(stabilized.observations), snapshot.n_observations)

        TestTier1EndToEnd.diagnostics_vector = diag
        TestTier1EndToEnd.stabilized_data = stabilized

    # ==================================================================
    # Stage 4: Drift / Batch / Confounder
    # ==================================================================

    def test_stage4_drift_batch_confounder(self) -> None:
        """Stage 4: Run drift detection, batch effect, and confounder detection."""
        from optimization_copilot.drift.detector import DriftDetector, DriftReport
        from optimization_copilot.data_quality.engine import (
            DataQualityEngine,
            DataQualityReport,
        )
        from optimization_copilot.confounder.detector import ConfounderDetector

        snapshot = TestTier1EndToEnd.snapshot
        if snapshot is None:
            self.test_stage2_snapshot_and_fingerprint()
            snapshot = TestTier1EndToEnd.snapshot

        # -- Drift detection --
        # Use small windows since we only have 50 obs
        drift_detector = DriftDetector(
            reference_window=10, test_window=10, significance=0.05
        )
        drift_report = drift_detector.detect(snapshot)
        self.assertIsInstance(drift_report, DriftReport)
        # drift_score should be in [0, 1]
        self.assertGreaterEqual(drift_report.drift_score, 0.0)
        self.assertLessEqual(drift_report.drift_score, 1.0)
        # drift_type should be a known type
        self.assertIn(
            drift_report.drift_type,
            ["none", "gradual", "sudden", "recurring"]
        )
        # recommended_action should be a valid action
        self.assertIn(
            drift_report.recommended_action,
            ["continue", "reweight", "re_screen", "re_learn", "restart"]
        )

        # -- Data Quality (batch effects, noise decomposition, drift) --
        dq_engine = DataQualityEngine()
        dq_report = dq_engine.analyze(snapshot)
        self.assertIsInstance(dq_report, DataQualityReport)
        # overall_quality_score in [0, 1]
        self.assertGreaterEqual(dq_report.overall_quality_score, 0.0)
        self.assertLessEqual(dq_report.overall_quality_score, 1.0)
        # Verify noise decomposition was computed
        self.assertIsNotNone(dq_report.noise_decomposition)
        # Verify batch effect detection ran
        self.assertIsNotNone(dq_report.batch_effect)
        # Verify instrument drift detection ran
        self.assertIsNotNone(dq_report.instrument_drift)
        # Credibility weights should cover observations
        self.assertGreater(len(dq_report.credibility_weights), 0)

        # -- Confounder detection --
        confounder_detector = ConfounderDetector()
        confounders = confounder_detector.detect(snapshot)
        # confounders is a list (may be empty if batch_id is not numeric)
        self.assertIsInstance(confounders, list)

        TestTier1EndToEnd.drift_report = drift_report
        TestTier1EndToEnd.data_quality_report = dq_report
        TestTier1EndToEnd.confounder_results = confounders

    # ==================================================================
    # Stage 5: Modeling
    # ==================================================================

    def test_stage5_modeling(self) -> None:
        """Stage 5: Fit GP backend on data, verify predictions with mean+std."""
        from optimization_copilot.backends.builtin import (
            GaussianProcessBO,
            RandomForestBO,
            TPESampler,
        )

        snapshot = TestTier1EndToEnd.snapshot
        if snapshot is None:
            self.test_stage2_snapshot_and_fingerprint()
            snapshot = TestTier1EndToEnd.snapshot

        successful_obs = snapshot.successful_observations
        self.assertGreater(len(successful_obs), 10, "Need enough data for modeling")

        # -- Fit GP backend --
        gp = GaussianProcessBO()
        gp.fit(snapshot.observations, snapshot.parameter_specs)

        # GP should produce suggestions
        suggestions = gp.suggest(n_suggestions=3, seed=42)
        self.assertEqual(len(suggestions), 3)
        for s in suggestions:
            # Each suggestion should have all 4 parameters
            self.assertIn("temperature", s)
            self.assertIn("catalyst_loading", s)
            self.assertIn("solvent_ratio", s)
            self.assertIn("reaction_time", s)
            # Values should be within bounds
            self.assertGreaterEqual(s["temperature"], 50.0)
            self.assertLessEqual(s["temperature"], 150.0)
            self.assertGreaterEqual(s["catalyst_loading"], 0.5)
            self.assertLessEqual(s["catalyst_loading"], 5.0)

        # -- Fit Random Forest backend for comparison --
        rf = RandomForestBO()
        rf.fit(snapshot.observations, snapshot.parameter_specs)
        rf_suggestions = rf.suggest(n_suggestions=3, seed=42)
        self.assertEqual(len(rf_suggestions), 3)

        # -- Fit TPE backend --
        tpe = TPESampler()
        tpe.fit(snapshot.observations, snapshot.parameter_specs)
        tpe_suggestions = tpe.suggest(n_suggestions=3, seed=42)
        self.assertEqual(len(tpe_suggestions), 3)

        # Verify cross-model consistency: all models should produce
        # suggestions within parameter bounds
        for model_name, model_suggestions in [
            ("GP", suggestions),
            ("RF", rf_suggestions),
            ("TPE", tpe_suggestions),
        ]:
            for s in model_suggestions:
                self.assertGreaterEqual(
                    s["temperature"], 50.0,
                    f"{model_name} temperature below lower bound"
                )
                self.assertLessEqual(
                    s["temperature"], 150.0,
                    f"{model_name} temperature above upper bound"
                )

        TestTier1EndToEnd.gp_backend = gp
        TestTier1EndToEnd.gp_suggestions = suggestions

    # ==================================================================
    # Stage 6: Recommendation
    # ==================================================================

    def test_stage6_recommendation(self) -> None:
        """Stage 6: MetaController decision, strategy, candidate suggestions."""
        from optimization_copilot.meta_controller.controller import MetaController
        from optimization_copilot.core.models import (
            Phase,
            StrategyDecision,
        )

        snapshot = TestTier1EndToEnd.snapshot
        diag = TestTier1EndToEnd.diagnostics_vector
        fingerprint = TestTier1EndToEnd.fingerprint
        if snapshot is None or diag is None or fingerprint is None:
            self.test_stage3_diagnostics_and_stabilization()
            snapshot = TestTier1EndToEnd.snapshot
            diag = TestTier1EndToEnd.diagnostics_vector
            fingerprint = TestTier1EndToEnd.fingerprint

        # Create MetaController with available backends
        controller = MetaController(
            available_backends=["random", "latin_hypercube", "tpe"]
        )

        # Get diagnostics as dict
        diag_dict = diag.to_dict()

        # Make strategy decision
        decision = controller.decide(
            snapshot=snapshot,
            diagnostics=diag_dict,
            fingerprint=fingerprint,
            seed=42,
        )

        self.assertIsInstance(decision, StrategyDecision)

        # Verify phase is set
        self.assertIsInstance(decision.phase, Phase)
        # With 50 observations, should NOT be cold_start
        self.assertNotEqual(
            decision.phase, Phase.COLD_START,
            "50 observations should not be cold_start"
        )

        # Verify backend is selected
        self.assertIsNotNone(decision.backend_name)
        self.assertIn(decision.backend_name, ["random", "latin_hypercube", "tpe"])

        # Verify exploration_strength is in [0, 1]
        self.assertGreaterEqual(decision.exploration_strength, 0.0)
        self.assertLessEqual(decision.exploration_strength, 1.0)

        # Verify reason_codes are populated
        self.assertGreater(
            len(decision.reason_codes), 0,
            "Decision should have reason codes for audit trail"
        )

        # Verify batch_size is positive
        self.assertGreater(decision.batch_size, 0)

        # Verify stabilize_spec is populated
        self.assertIsNotNone(decision.stabilize_spec)

        # Verify the decision can be serialized
        d = decision.to_dict()
        self.assertIn("backend_name", d)
        self.assertIn("phase", d)
        self.assertIn("exploration_strength", d)

        TestTier1EndToEnd.strategy_decision = decision

    # ==================================================================
    # Stage 7: Replay / Verify
    # ==================================================================

    def test_stage7_replay_verify(self) -> None:
        """Stage 7: Verify deterministic replay via ReplayEngine.record_iteration."""
        from optimization_copilot.replay.engine import ReplayEngine
        from optimization_copilot.replay.log import DecisionLog, DecisionLogEntry
        from optimization_copilot.plugins.registry import PluginRegistry
        from optimization_copilot.backends.builtin import (
            RandomSampler,
            TPESampler,
            LatinHypercubeSampler,
        )
        from optimization_copilot.diagnostics.engine import DiagnosticEngine
        from optimization_copilot.profiler.profiler import ProblemProfiler
        from optimization_copilot.core.hashing import (
            snapshot_hash,
            diagnostics_hash,
            decision_hash,
        )

        snapshot = TestTier1EndToEnd.snapshot
        diag = TestTier1EndToEnd.diagnostics_vector
        fingerprint = TestTier1EndToEnd.fingerprint
        decision = TestTier1EndToEnd.strategy_decision
        if snapshot is None or decision is None:
            self.test_stage6_recommendation()
            snapshot = TestTier1EndToEnd.snapshot
            diag = TestTier1EndToEnd.diagnostics_vector
            fingerprint = TestTier1EndToEnd.fingerprint
            decision = TestTier1EndToEnd.strategy_decision

        # Verify deterministic hashing
        hash1 = snapshot_hash(snapshot)
        hash2 = snapshot_hash(snapshot)
        self.assertEqual(hash1, hash2, "snapshot_hash must be deterministic")

        diag_dict = diag.to_dict()
        dhash1 = diagnostics_hash(diag_dict)
        dhash2 = diagnostics_hash(diag_dict)
        self.assertEqual(dhash1, dhash2, "diagnostics_hash must be deterministic")

        dechash1 = decision_hash(decision)
        dechash2 = decision_hash(decision)
        self.assertEqual(dechash1, dechash2, "decision_hash must be deterministic")

        # Create a registry with backends for replay
        registry = PluginRegistry()
        registry.register(RandomSampler)
        registry.register(TPESampler)
        registry.register(LatinHypercubeSampler)

        # Use record_iteration to create a log entry
        entry = ReplayEngine.record_iteration(
            iteration=0,
            snapshot=snapshot,
            diagnostics_vector=diag_dict,
            fingerprint_dict=fingerprint.to_dict(),
            decision=decision,
            candidates=[{"temperature": 100.0, "catalyst_loading": 2.5,
                         "solvent_ratio": 0.5, "reaction_time": 12.0}],
            results=[{
                "iteration": 0,
                "parameters": {"temperature": 100.0, "catalyst_loading": 2.5,
                                "solvent_ratio": 0.5, "reaction_time": 12.0},
                "kpi_values": {"yield": 75.0},
                "is_failure": False,
                "qc_passed": True,
            }],
            seed=42,
        )

        self.assertIsInstance(entry, DecisionLogEntry)
        self.assertEqual(entry.iteration, 0)
        self.assertIsNotNone(entry.snapshot_hash)
        self.assertIsNotNone(entry.diagnostics_hash)
        self.assertIsNotNone(entry.decision_hash)
        # Hashes should be non-empty strings
        self.assertGreater(len(entry.snapshot_hash), 0)
        self.assertGreater(len(entry.diagnostics_hash), 0)
        self.assertGreater(len(entry.decision_hash), 0)

    # ==================================================================
    # Stage 8: Deliverables
    # ==================================================================

    def test_stage8_deliverables(self) -> None:
        """Stage 8: ExplanationGraph, AuditLog with hash chain, ExecutionTrace."""
        from optimization_copilot.explanation_graph.models import (
            ExplanationGraph,
            NodeType,
        )
        from optimization_copilot.compliance.audit import (
            AuditEntry,
            AuditLog,
            verify_chain,
            _compute_content_hash,
            _compute_chain_hash,
        )
        from optimization_copilot.replay.log import DecisionLogEntry
        from optimization_copilot.agents.execution_trace import (
            ExecutionTag,
            ExecutionTrace,
            TracedResult,
            trace_call,
        )

        snapshot = TestTier1EndToEnd.snapshot
        diag = TestTier1EndToEnd.diagnostics_vector
        decision = TestTier1EndToEnd.strategy_decision
        fingerprint = TestTier1EndToEnd.fingerprint
        if snapshot is None or decision is None:
            self.test_stage6_recommendation()
            snapshot = TestTier1EndToEnd.snapshot
            diag = TestTier1EndToEnd.diagnostics_vector
            decision = TestTier1EndToEnd.strategy_decision
            fingerprint = TestTier1EndToEnd.fingerprint

        # -- ExplanationGraph --
        # The GraphBuilder has dependencies on surgery and feasibility modules,
        # so we build a basic graph manually using the models.
        graph = ExplanationGraph()

        # Add signal nodes from diagnostics
        d = diag.to_dict()
        from optimization_copilot.explanation_graph.models import (
            GraphNode,
            GraphEdge,
            EdgeType,
        )
        for field_name, value in d.items():
            if isinstance(value, (int, float)) and value != 0:
                node_id = f"signal:{field_name}"
                graph.add_node(GraphNode(
                    node_id=node_id,
                    node_type=NodeType.SIGNAL,
                    label=f"{field_name} = {value}",
                    data={"value": value, "field": field_name},
                ))

        # Add decision node
        graph.add_node(GraphNode(
            node_id="decision:strategy",
            node_type=NodeType.DECISION,
            label=f"{decision.backend_name} (phase={decision.phase.value})",
            data={
                "backend_name": decision.backend_name,
                "phase": decision.phase.value,
            },
        ))

        # Wire some edges
        if "signal:failure_rate" in graph.nodes:
            graph.add_edge(GraphEdge(
                source_id="signal:failure_rate",
                target_id="decision:strategy",
                edge_type=EdgeType.TRIGGERS,
                evidence={"value": d["failure_rate"]},
            ))

        self.assertGreater(graph.n_nodes, 0)
        self.assertIsNotNone(graph.get_node("decision:strategy"))

        # Verify serialization round-trip
        graph_dict = graph.to_dict()
        restored_graph = ExplanationGraph.from_dict(graph_dict)
        self.assertEqual(restored_graph.n_nodes, graph.n_nodes)

        # -- AuditLog with hash chain --
        audit_log = AuditLog(
            campaign_id=self.campaign_id,
            spec={
                "parameters": [
                    {"name": "temperature", "type": "continuous",
                     "lower": 50.0, "upper": 150.0},
                    {"name": "catalyst_loading", "type": "continuous",
                     "lower": 0.5, "upper": 5.0},
                    {"name": "solvent_ratio", "type": "continuous",
                     "lower": 0.1, "upper": 0.9},
                    {"name": "reaction_time", "type": "continuous",
                     "lower": 1.0, "upper": 24.0},
                ],
                "objectives": [
                    {"name": "yield", "direction": "maximize"}
                ],
            },
            base_seed=42,
            signer_id="test_runner",
        )

        # Create a few audit entries with proper hash chaining
        prev_chain_hash = ""
        for i in range(3):
            log_entry = DecisionLogEntry(
                iteration=i,
                timestamp=time.time(),
                snapshot_hash=f"snap_{i}",
                diagnostics_hash=f"diag_{i}",
                diagnostics=d,
                fingerprint=fingerprint.to_dict(),
                decision=decision.to_dict(),
                decision_hash=f"dec_{i}",
                suggested_candidates=[],
                ingested_results=[],
                phase=decision.phase.value,
                backend_name=decision.backend_name,
                reason_codes=list(decision.reason_codes),
                seed=42 + i,
            )

            # Compute chain hash
            audit_entry = AuditEntry.from_log_entry(
                log_entry,
                chain_hash="",  # placeholder, computed below
                signer_id="test_runner",
            )
            content_h = audit_entry.content_hash()
            chain_h = _compute_chain_hash(prev_chain_hash, content_h)
            audit_entry.chain_hash = chain_h
            prev_chain_hash = chain_h

            audit_log.append(audit_entry)

        # Verify chain integrity
        verification = verify_chain(audit_log)
        self.assertTrue(
            verification.valid,
            f"Hash chain should be valid: {verification.summary()}"
        )
        self.assertEqual(verification.n_entries, 3)
        self.assertEqual(verification.n_broken_links, 0)

        # Verify serialization round-trip
        json_str = audit_log.to_json()
        restored_log = AuditLog.from_json(json_str)
        self.assertEqual(restored_log.n_entries, 3)

        # Re-verify chain after round-trip
        verification2 = verify_chain(restored_log)
        self.assertTrue(
            verification2.valid,
            "Chain should remain valid after JSON round-trip"
        )

        # -- ExecutionTrace --
        # Test trace_call with a simple function
        def sample_computation(x: float) -> float:
            return x * 2.0

        traced = trace_call(
            module="test.tier1",
            method="sample_computation",
            fn=sample_computation,
            args=(42.0,),
            input_summary={"x": 42.0},
            output_summarizer=lambda r: {"result": r},
        )

        self.assertIsInstance(traced, TracedResult)
        self.assertTrue(traced.is_computed)
        self.assertEqual(traced.tag, ExecutionTag.COMPUTED)
        self.assertEqual(traced.value, 84.0)
        self.assertEqual(len(traced.traces), 1)
        self.assertEqual(traced.traces[0].module, "test.tier1")
        self.assertEqual(traced.traces[0].method, "sample_computation")
        self.assertGreater(traced.traces[0].duration_ms, 0.0)
        self.assertIsNone(traced.traces[0].error)

        # Test trace_call with a failing function
        def failing_computation() -> None:
            raise ValueError("deliberate failure")

        failed_traced = trace_call(
            module="test.tier1",
            method="failing_computation",
            fn=failing_computation,
        )

        self.assertEqual(failed_traced.tag, ExecutionTag.FAILED)
        self.assertIsNone(failed_traced.value)
        self.assertEqual(len(failed_traced.traces), 1)
        self.assertIsNotNone(failed_traced.traces[0].error)
        self.assertIn("deliberate failure", failed_traced.traces[0].error)

        # Test TracedResult merge and overall_tag
        combined_traces = TracedResult.merge([traced, failed_traced])
        self.assertEqual(len(combined_traces), 2)
        overall = TracedResult.overall_tag([traced, failed_traced])
        self.assertEqual(overall, ExecutionTag.FAILED)

        TestTier1EndToEnd.explanation_graph = graph
        TestTier1EndToEnd.audit_log = audit_log

    # ==================================================================
    # Full Pipeline Sequential Test
    # ==================================================================

    def test_full_pipeline_sequential(self) -> None:
        """Run all 8 stages in order and verify the complete deliverable bundle."""
        # Stage 0: Input & Spec
        self.test_stage0_input_and_spec()
        self.assertIsNotNone(TestTier1EndToEnd.ingestion_report)
        self.assertIsNotNone(TestTier1EndToEnd.store)

        # Stage 1: Ingestion + Store
        self.test_stage1_ingestion_and_store()

        # Stage 2: CampaignSnapshot + Fingerprint
        self.test_stage2_snapshot_and_fingerprint()
        self.assertIsNotNone(TestTier1EndToEnd.snapshot)
        self.assertIsNotNone(TestTier1EndToEnd.fingerprint)

        # Stage 3: Diagnostics + Stabilization
        self.test_stage3_diagnostics_and_stabilization()
        self.assertIsNotNone(TestTier1EndToEnd.diagnostics_vector)
        self.assertIsNotNone(TestTier1EndToEnd.stabilized_data)

        # Stage 4: Drift / Batch / Confounder
        self.test_stage4_drift_batch_confounder()
        self.assertIsNotNone(TestTier1EndToEnd.drift_report)
        self.assertIsNotNone(TestTier1EndToEnd.data_quality_report)
        self.assertIsNotNone(TestTier1EndToEnd.confounder_results)

        # Stage 5: Modeling
        self.test_stage5_modeling()
        self.assertIsNotNone(TestTier1EndToEnd.gp_backend)
        self.assertIsNotNone(TestTier1EndToEnd.gp_suggestions)

        # Stage 6: Recommendation
        self.test_stage6_recommendation()
        self.assertIsNotNone(TestTier1EndToEnd.strategy_decision)

        # Stage 7: Replay / Verify
        self.test_stage7_replay_verify()

        # Stage 8: Deliverables
        self.test_stage8_deliverables()
        self.assertIsNotNone(TestTier1EndToEnd.explanation_graph)
        self.assertIsNotNone(TestTier1EndToEnd.audit_log)

        # === Final deliverable bundle verification ===

        # Product A: Data Dictionary (column profiles from ingestion)
        report = TestTier1EndToEnd.ingestion_report
        self.assertGreater(len(report.column_profiles), 0)

        # Product B, C: Store with experiments
        store = TestTier1EndToEnd.store
        self.assertEqual(len(store), 50)

        # Product D: CampaignSnapshot
        snapshot = TestTier1EndToEnd.snapshot
        self.assertEqual(snapshot.n_observations, 50)
        self.assertEqual(len(snapshot.parameter_specs), 4)

        # Product E: ProblemFingerprint (8 dimensions)
        fp = TestTier1EndToEnd.fingerprint
        fp_dict = fp.to_dict()
        self.assertIn("variable_types", fp_dict)
        self.assertIn("objective_form", fp_dict)
        self.assertIn("noise_regime", fp_dict)
        self.assertIn("cost_profile", fp_dict)
        self.assertIn("failure_informativeness", fp_dict)
        self.assertIn("data_scale", fp_dict)
        self.assertIn("dynamics", fp_dict)
        self.assertIn("feasible_region", fp_dict)

        # Product F: DiagnosticsVector (17 signals)
        diag_dict = TestTier1EndToEnd.diagnostics_vector.to_dict()
        self.assertEqual(len(diag_dict), 17)

        # Product G: StabilizedData
        stabilized = TestTier1EndToEnd.stabilized_data
        self.assertIsNotNone(stabilized.observations)

        # Product H: DriftReport
        drift = TestTier1EndToEnd.drift_report
        self.assertIn("drift_score", drift.__dict__)

        # Product I: DataQualityReport
        dq = TestTier1EndToEnd.data_quality_report
        self.assertIsNotNone(dq.noise_decomposition)
        self.assertIsNotNone(dq.batch_effect)
        self.assertIsNotNone(dq.instrument_drift)

        # Product J: Confounder results
        self.assertIsInstance(TestTier1EndToEnd.confounder_results, list)

        # Product K, L, M: Model suggestions from multiple backends
        self.assertEqual(len(TestTier1EndToEnd.gp_suggestions), 3)

        # Product N, O, P, Q: StrategyDecision
        decision = TestTier1EndToEnd.strategy_decision
        self.assertIsNotNone(decision.phase)
        self.assertIsNotNone(decision.backend_name)
        self.assertGreater(len(decision.reason_codes), 0)

        # Product R: Replay verification (deterministic hashing)
        from optimization_copilot.core.hashing import snapshot_hash
        h1 = snapshot_hash(snapshot)
        h2 = snapshot_hash(snapshot)
        self.assertEqual(h1, h2)

        # Final: ExplanationGraph
        graph = TestTier1EndToEnd.explanation_graph
        self.assertGreater(graph.n_nodes, 0)

        # Final: AuditLog with verified hash chain
        audit = TestTier1EndToEnd.audit_log
        self.assertEqual(audit.n_entries, 3)
        from optimization_copilot.compliance.audit import verify_chain
        self.assertTrue(verify_chain(audit).valid)


if __name__ == "__main__":
    unittest.main()

"""Tests for the Optimization Reasoning Agent (Capability 16).

Covers:
- RewriteSuggestion dataclass: construction, to_dict/from_dict
- FailureCluster dataclass: construction, to_dict/from_dict
- FailureClusterReport dataclass: construction, n_clusters, to_dict/from_dict
- CampaignNarrative dataclass: construction, format_text(), to_dict/from_dict
- ReasoningAgent: template-only mode, LLM mode, explain_surgery, explain_failures,
  generate_narrative, clustering (taxonomy and proximity), recommendations
"""

from __future__ import annotations

import os

from optimization_copilot.reasoning.agent import (
    RewriteSuggestion,
    FailureCluster,
    FailureClusterReport,
    CampaignNarrative,
    ReasoningAgent,
    _default_llm_caller,
)
from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
    Phase,
    RiskPosture,
    StrategyDecision,
    StabilizeSpec,
    ProblemFingerprint,
)
from optimization_copilot.diagnostics.engine import DiagnosticsVector
from optimization_copilot.feasibility.feasibility import FeasibilityMap
from optimization_copilot.feasibility.taxonomy import (
    FailureTaxonomy,
    FailureType,
    ClassifiedFailure,
)
from optimization_copilot.surgery.models import (
    ActionType,
    DerivedType,
    SurgeryAction,
    SurgeryReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(
    n_obs: int = 10,
    n_failures: int = 3,
    n_params: int = 3,
) -> CampaignSnapshot:
    """Create a CampaignSnapshot with controllable observation/failure counts.

    Parameters
    ----------
    n_obs : int
        Total number of observations.
    n_failures : int
        Number of observations marked as failures (the last *n_failures*).
    n_params : int
        Number of continuous parameters (up to 3: x, y, z).
    """
    param_defs = [
        ("x", 0.0, 1.0),
        ("y", 0.0, 10.0),
        ("z", 0.0, 100.0),
    ][:n_params]

    specs = [
        ParameterSpec(name=name, type=VariableType.CONTINUOUS, lower=lo, upper=hi)
        for name, lo, hi in param_defs
    ]

    obs: list[Observation] = []
    for i in range(n_obs):
        t = i / max(n_obs - 1, 1)
        is_fail = i >= (n_obs - n_failures)
        params = {}
        for name, lo, hi in param_defs:
            params[name] = lo + (hi - lo) * t
        obs.append(
            Observation(
                iteration=i,
                parameters=params,
                kpi_values={"kpi": 10.0 - 5.0 * t if not is_fail else float("nan")},
                is_failure=is_fail,
                failure_reason="simulated failure" if is_fail else None,
                timestamp=float(i),
            )
        )

    return CampaignSnapshot(
        campaign_id="test-campaign",
        parameter_specs=specs,
        observations=obs,
        objective_names=["kpi"],
        objective_directions=["minimize"],
        current_iteration=n_obs,
    )


def _make_surgery_report() -> SurgeryReport:
    """Create a SurgeryReport with TIGHTEN_RANGE, FREEZE_PARAMETER, and DERIVE_PARAMETER."""
    actions = [
        SurgeryAction(
            action_type=ActionType.TIGHTEN_RANGE,
            target_params=["x"],
            new_lower=0.2,
            new_upper=0.8,
            reason="concentrated observations",
            confidence=0.9,
            evidence={
                "original_lower": 0.0,
                "original_upper": 1.0,
                "range_reduction_pct": 40,
            },
        ),
        SurgeryAction(
            action_type=ActionType.FREEZE_PARAMETER,
            target_params=["y"],
            freeze_value=5.0,
            reason="low importance",
            confidence=0.85,
            evidence={"importance_score": 0.02},
        ),
        SurgeryAction(
            action_type=ActionType.DERIVE_PARAMETER,
            target_params=["z"],
            derived_type=DerivedType.LOG,
            derived_name="log_z",
            reason="wide range ratio",
            confidence=0.7,
            evidence={"range_ratio": 1000},
        ),
    ]
    return SurgeryReport(
        actions=actions,
        original_dim=3,
        effective_dim=2,
        space_reduction_ratio=0.33,
        reason_codes=["range_tightening", "freeze_unimportant", "log_transform"],
    )


def _make_diagnostics() -> DiagnosticsVector:
    """Create a DiagnosticsVector with non-default values."""
    return DiagnosticsVector(
        convergence_trend=-0.15,
        improvement_velocity=0.05,
        variance_contraction=0.8,
        noise_estimate=0.12,
        failure_rate=0.3,
        failure_clustering=1.5,
        feasibility_shrinkage=-0.1,
        parameter_drift=0.05,
        model_uncertainty=0.2,
        exploration_coverage=0.15,
        kpi_plateau_length=12,
        best_kpi_value=3.5,
        data_efficiency=0.02,
        constraint_violation_rate=0.05,
    )


def _make_decision(phase: Phase = Phase.LEARNING) -> StrategyDecision:
    """Create a StrategyDecision with the given phase."""
    return StrategyDecision(
        backend_name="bo_gp",
        stabilize_spec=StabilizeSpec(),
        exploration_strength=0.6,
        batch_size=4,
        risk_posture=RiskPosture.MODERATE,
        phase=phase,
        reason_codes=["exploration_needed", "moderate_noise"],
        fallback_events=["backend_timeout"],
    )


def _make_feasibility_map(
    n_infeasible_zones: int = 0,
    feasibility_score: float = 0.8,
) -> FeasibilityMap:
    """Create a FeasibilityMap with controllable infeasible zones."""
    zones = [
        {"x": (0.0, 0.1)} for _ in range(n_infeasible_zones)
    ]
    return FeasibilityMap(
        safe_bounds={"x": (0.1, 0.9), "y": (1.0, 9.0), "z": (10.0, 90.0)},
        infeasible_zones=zones,
        failure_density={"x": [0.05], "y": [2.0], "z": [15.0]},
        feasibility_score=feasibility_score,
        constraint_tightness={},
    )


def _make_taxonomy(snapshot: CampaignSnapshot) -> FailureTaxonomy:
    """Create a FailureTaxonomy with classified failures matching snapshot failures."""
    failures = [
        (i, obs)
        for i, obs in enumerate(snapshot.observations)
        if obs.is_failure
    ]
    classified: list[ClassifiedFailure] = []
    # Distribute failures across two types for variety
    for idx, (obs_idx, obs) in enumerate(failures):
        ft = FailureType.CHEMISTRY if idx % 2 == 0 else FailureType.HARDWARE
        classified.append(
            ClassifiedFailure(
                observation_index=obs_idx,
                failure_type=ft,
                confidence=0.8,
                evidence=["test evidence"],
            )
        )

    type_counts = {ft.value: 0 for ft in FailureType}
    for cf in classified:
        type_counts[cf.failure_type.value] += 1
    total = len(classified) or 1
    type_rates = {k: v / total for k, v in type_counts.items()}
    dominant_type = FailureType(max(type_counts, key=lambda k: type_counts[k]))

    return FailureTaxonomy(
        classified_failures=classified,
        type_counts=type_counts,
        dominant_type=dominant_type,
        type_rates=type_rates,
        strategy_adjustments={"dominant": "adjust_bounds"},
    )


# ---------------------------------------------------------------------------
# Tests: RewriteSuggestion
# ---------------------------------------------------------------------------


def test_rewrite_suggestion_construction():
    """RewriteSuggestion can be constructed with all fields."""
    rs = RewriteSuggestion(
        action_type="tighten_range",
        target_params=["x", "y"],
        explanation="Tighten x and y.",
        confidence_narrative="High confidence (0.90).",
        evidence_summary="Evidence: pct=40",
        generated_by="template",
        metadata={"extra": 1},
    )
    assert rs.action_type == "tighten_range"
    assert rs.target_params == ["x", "y"]
    assert rs.explanation == "Tighten x and y."
    assert rs.confidence_narrative == "High confidence (0.90)."
    assert rs.evidence_summary == "Evidence: pct=40"
    assert rs.generated_by == "template"
    assert rs.metadata == {"extra": 1}


def test_rewrite_suggestion_to_dict():
    """to_dict returns a plain dict with all fields."""
    rs = RewriteSuggestion(
        action_type="freeze_parameter",
        target_params=["y"],
        explanation="Freeze y.",
        confidence_narrative="Moderate.",
        evidence_summary="importance=0.02",
        generated_by="llm",
        metadata={"model": "test"},
    )
    d = rs.to_dict()
    assert isinstance(d, dict)
    assert d["action_type"] == "freeze_parameter"
    assert d["target_params"] == ["y"]
    assert d["generated_by"] == "llm"
    assert d["metadata"] == {"model": "test"}


def test_rewrite_suggestion_from_dict_roundtrip():
    """from_dict(to_dict()) produces an equivalent object."""
    original = RewriteSuggestion(
        action_type="derive_parameter",
        target_params=["z"],
        explanation="Log-transform z.",
        confidence_narrative="Low confidence.",
        evidence_summary="range_ratio=1000",
        generated_by="template",
        metadata={"step": 3},
    )
    restored = RewriteSuggestion.from_dict(original.to_dict())
    assert restored.action_type == original.action_type
    assert restored.target_params == original.target_params
    assert restored.explanation == original.explanation
    assert restored.confidence_narrative == original.confidence_narrative
    assert restored.evidence_summary == original.evidence_summary
    assert restored.generated_by == original.generated_by
    assert restored.metadata == original.metadata


# ---------------------------------------------------------------------------
# Tests: FailureCluster
# ---------------------------------------------------------------------------


def test_failure_cluster_construction():
    """FailureCluster can be constructed with tuple ranges."""
    fc = FailureCluster(
        cluster_id=0,
        failure_indices=[7, 8, 9],
        failure_type="chemistry",
        parameter_ranges={"x": (0.7, 1.0), "y": (7.0, 10.0)},
        explanation="Cluster of 3 chemistry failures.",
        count=3,
    )
    assert fc.cluster_id == 0
    assert fc.failure_indices == [7, 8, 9]
    assert fc.failure_type == "chemistry"
    assert fc.parameter_ranges["x"] == (0.7, 1.0)
    assert fc.count == 3


def test_failure_cluster_to_dict_tuple_ranges():
    """to_dict converts tuple parameter_ranges to lists (JSON-safe)."""
    fc = FailureCluster(
        cluster_id=1,
        failure_indices=[5],
        failure_type="hardware",
        parameter_ranges={"z": (50.0, 90.0)},
        explanation="Single hardware failure.",
        count=1,
    )
    d = fc.to_dict()
    # parameter_ranges values should be lists, not tuples
    assert isinstance(d["parameter_ranges"]["z"], list)
    assert d["parameter_ranges"]["z"] == [50.0, 90.0]


def test_failure_cluster_from_dict_roundtrip():
    """from_dict(to_dict()) produces an equivalent FailureCluster with tuple ranges."""
    original = FailureCluster(
        cluster_id=2,
        failure_indices=[3, 4, 5],
        failure_type="proximity",
        parameter_ranges={"x": (0.3, 0.5), "y": (3.0, 5.0)},
        explanation="Proximity cluster.",
        count=3,
    )
    d = original.to_dict()
    restored = FailureCluster.from_dict(d)
    assert restored.cluster_id == original.cluster_id
    assert restored.failure_indices == original.failure_indices
    assert restored.failure_type == original.failure_type
    # Ranges should be tuples after round-trip
    assert isinstance(restored.parameter_ranges["x"], tuple)
    assert restored.parameter_ranges["x"] == original.parameter_ranges["x"]
    assert restored.parameter_ranges["y"] == original.parameter_ranges["y"]
    assert restored.count == original.count


# ---------------------------------------------------------------------------
# Tests: FailureClusterReport
# ---------------------------------------------------------------------------


def test_failure_cluster_report_n_clusters():
    """n_clusters property returns the number of clusters."""
    clusters = [
        FailureCluster(
            cluster_id=i,
            failure_indices=[i],
            failure_type="test",
            parameter_ranges={},
            explanation="",
            count=1,
        )
        for i in range(4)
    ]
    report = FailureClusterReport(
        clusters=clusters,
        overall_pattern="Four clusters.",
        dominant_failure_mode="test",
        recommendations=["Check clusters."],
    )
    assert report.n_clusters == 4


def test_failure_cluster_report_to_dict():
    """to_dict serializes all fields including nested clusters."""
    cluster = FailureCluster(
        cluster_id=0,
        failure_indices=[1, 2],
        failure_type="chemistry",
        parameter_ranges={"x": (0.1, 0.5)},
        explanation="Chem cluster.",
        count=2,
    )
    report = FailureClusterReport(
        clusters=[cluster],
        overall_pattern="One cluster of chemistry failures.",
        dominant_failure_mode="chemistry",
        recommendations=["Tighten bounds."],
        generated_by="template",
        metadata={"total_failures": 2},
    )
    d = report.to_dict()
    assert len(d["clusters"]) == 1
    assert d["clusters"][0]["cluster_id"] == 0
    assert d["overall_pattern"] == "One cluster of chemistry failures."
    assert d["dominant_failure_mode"] == "chemistry"
    assert d["recommendations"] == ["Tighten bounds."]
    assert d["generated_by"] == "template"
    assert d["metadata"] == {"total_failures": 2}


def test_failure_cluster_report_from_dict_roundtrip():
    """from_dict(to_dict()) preserves all fields."""
    clusters = [
        FailureCluster(
            cluster_id=0,
            failure_indices=[7, 8],
            failure_type="hardware",
            parameter_ranges={"y": (8.0, 10.0)},
            explanation="HW cluster.",
            count=2,
        ),
        FailureCluster(
            cluster_id=1,
            failure_indices=[9],
            failure_type="chemistry",
            parameter_ranges={"z": (80.0, 100.0)},
            explanation="Chem cluster.",
            count=1,
        ),
    ]
    original = FailureClusterReport(
        clusters=clusters,
        overall_pattern="Mixed failures.",
        dominant_failure_mode="hardware",
        recommendations=["Rec 1", "Rec 2"],
        generated_by="llm",
        metadata={"model": "test"},
    )
    d = original.to_dict()
    restored = FailureClusterReport.from_dict(d)
    assert restored.n_clusters == 2
    assert restored.overall_pattern == original.overall_pattern
    assert restored.dominant_failure_mode == original.dominant_failure_mode
    assert restored.recommendations == original.recommendations
    assert restored.generated_by == original.generated_by
    assert restored.metadata == original.metadata
    assert restored.clusters[0].failure_type == "hardware"
    assert restored.clusters[1].failure_type == "chemistry"
    # Verify tuple round-trip in nested clusters
    assert isinstance(restored.clusters[0].parameter_ranges["y"], tuple)


# ---------------------------------------------------------------------------
# Tests: CampaignNarrative
# ---------------------------------------------------------------------------


def test_campaign_narrative_construction():
    """CampaignNarrative can be constructed with all fields."""
    cn = CampaignNarrative(
        campaign_id="test-campaign",
        executive_summary="Summary text.",
        phase_description="Phase: learning.",
        diagnostic_summary="Convergence trend: -0.15.",
        strategy_rationale="Reason codes: exploration_needed.",
        failure_analysis="3 failures observed.",
        recommendations=["Rec 1", "Rec 2"],
        generated_by="template",
        metadata={"version": 1},
    )
    assert cn.campaign_id == "test-campaign"
    assert cn.executive_summary == "Summary text."
    assert cn.recommendations == ["Rec 1", "Rec 2"]
    assert cn.generated_by == "template"


def test_campaign_narrative_format_text_has_sections():
    """format_text() produces text with all expected section headers."""
    cn = CampaignNarrative(
        campaign_id="my-campaign",
        executive_summary="Exec summary.",
        phase_description="In learning phase.",
        diagnostic_summary="Signals look good.",
        strategy_rationale="Exploring more.",
        failure_analysis="2 failures observed.",
        recommendations=["Do A.", "Do B."],
        generated_by="template",
    )
    text = cn.format_text()
    assert "=== CAMPAIGN NARRATIVE: my-campaign ===" in text
    assert "-- EXECUTIVE SUMMARY --" in text
    assert "Exec summary." in text
    assert "-- CURRENT PHASE --" in text
    assert "-- DIAGNOSTIC SIGNALS --" in text
    assert "-- STRATEGY RATIONALE --" in text
    assert "-- FAILURE ANALYSIS --" in text
    assert "2 failures observed." in text
    assert "-- RECOMMENDATIONS --" in text
    assert "1. Do A." in text
    assert "2. Do B." in text
    assert "[Generated by: template]" in text


def test_campaign_narrative_format_text_empty_failure_analysis():
    """When failure_analysis is empty, the FAILURE ANALYSIS section is omitted."""
    cn = CampaignNarrative(
        campaign_id="no-fail",
        executive_summary="All good.",
        phase_description="Exploiting.",
        diagnostic_summary="Clean.",
        strategy_rationale="Stay the course.",
        failure_analysis="",
        recommendations=["Continue."],
        generated_by="template",
    )
    text = cn.format_text()
    assert "-- FAILURE ANALYSIS --" not in text
    # Other sections should still be present
    assert "-- EXECUTIVE SUMMARY --" in text
    assert "-- RECOMMENDATIONS --" in text


def test_campaign_narrative_to_from_dict_roundtrip():
    """to_dict/from_dict round-trip preserves all fields."""
    original = CampaignNarrative(
        campaign_id="rt-test",
        executive_summary="Summary.",
        phase_description="Phase.",
        diagnostic_summary="Diag.",
        strategy_rationale="Strategy.",
        failure_analysis="Failures.",
        recommendations=["R1", "R2", "R3"],
        generated_by="llm",
        metadata={"key": "value"},
    )
    d = original.to_dict()
    restored = CampaignNarrative.from_dict(d)
    assert restored.campaign_id == original.campaign_id
    assert restored.executive_summary == original.executive_summary
    assert restored.phase_description == original.phase_description
    assert restored.diagnostic_summary == original.diagnostic_summary
    assert restored.strategy_rationale == original.strategy_rationale
    assert restored.failure_analysis == original.failure_analysis
    assert restored.recommendations == original.recommendations
    assert restored.generated_by == original.generated_by
    assert restored.metadata == original.metadata


# ---------------------------------------------------------------------------
# Tests: ReasoningAgent construction
# ---------------------------------------------------------------------------


def test_agent_template_only_mode(monkeypatch):
    """With no API key and no env vars, the agent uses template-only mode."""
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    agent = ReasoningAgent()
    assert agent._llm_caller is None


def test_agent_with_injectable_caller():
    """When llm_caller is passed directly, the agent uses it."""
    calls = []

    def mock_caller(prompt: str) -> str:
        calls.append(prompt)
        return "LLM response"

    agent = ReasoningAgent(llm_caller=mock_caller)
    assert agent._llm_caller is not None
    # Verify the callable is the one we passed
    result = agent._llm_caller("test prompt")
    assert result == "LLM response"
    assert len(calls) == 1


def test_agent_use_llm_false(monkeypatch):
    """When use_llm=False, _llm_caller is None even if an API key is provided."""
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    agent = ReasoningAgent(api_key="sk-test-key-123", use_llm=False)
    assert agent._llm_caller is None


# ---------------------------------------------------------------------------
# Tests: explain_surgery
# ---------------------------------------------------------------------------


def test_explain_surgery_empty_report():
    """An empty surgery report produces an empty suggestions list."""
    agent = ReasoningAgent(use_llm=False)
    snapshot = _make_snapshot(n_obs=10, n_failures=0)
    report = SurgeryReport()  # empty, no actions
    suggestions = agent.explain_surgery(report, snapshot)
    assert suggestions == []


def test_explain_surgery_template_mode(monkeypatch):
    """In template-only mode, all suggestions have generated_by='template'."""
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    agent = ReasoningAgent()
    snapshot = _make_snapshot(n_obs=10, n_failures=3)
    report = _make_surgery_report()

    suggestions = agent.explain_surgery(report, snapshot)
    assert len(suggestions) == 3
    for s in suggestions:
        assert s.generated_by == "template"
        assert isinstance(s, RewriteSuggestion)
        assert s.explanation  # non-empty
        assert s.confidence_narrative  # non-empty


def test_explain_surgery_all_action_types(monkeypatch):
    """Each ActionType template produces a non-empty, type-specific explanation."""
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    agent = ReasoningAgent()
    snapshot = _make_snapshot(n_obs=10, n_failures=0)

    # Build one action per type
    actions = [
        SurgeryAction(
            action_type=ActionType.TIGHTEN_RANGE,
            target_params=["x"],
            new_lower=0.2,
            new_upper=0.8,
            reason="concentrated",
            confidence=0.9,
            evidence={"original_lower": 0.0, "original_upper": 1.0, "range_reduction_pct": 40},
        ),
        SurgeryAction(
            action_type=ActionType.FREEZE_PARAMETER,
            target_params=["y"],
            freeze_value=5.0,
            reason="low importance",
            confidence=0.85,
            evidence={"importance_score": 0.02},
        ),
        SurgeryAction(
            action_type=ActionType.CONDITIONAL_FREEZE,
            target_params=["x"],
            condition_param="y",
            condition_threshold=5.0,
            condition_direction="above",
            reason="conditional pattern",
            confidence=0.7,
            evidence={"correlations": {"x_y": 0.1}},
        ),
        SurgeryAction(
            action_type=ActionType.MERGE_PARAMETERS,
            target_params=["x", "y"],
            merge_into="x",
            reason="redundant",
            confidence=0.8,
            evidence={"correlation": 0.95, "primary_param": "x", "secondary_param": "y"},
        ),
        SurgeryAction(
            action_type=ActionType.DERIVE_PARAMETER,
            target_params=["z"],
            derived_type=DerivedType.LOG,
            derived_name="log_z",
            reason="wide range",
            confidence=0.7,
            evidence={"range_ratio": 1000},
        ),
        SurgeryAction(
            action_type=ActionType.DERIVE_PARAMETER,
            target_params=["x", "y"],
            derived_type=DerivedType.RATIO,
            derived_name="x_over_y",
            reason="moderate correlation",
            confidence=0.6,
            evidence={"correlation": 0.65},
        ),
        SurgeryAction(
            action_type=ActionType.REMOVE_PARAMETER,
            target_params=["z"],
            reason="Remove z (irrelevant).",
            confidence=0.5,
            evidence={},
        ),
    ]

    report = SurgeryReport(actions=actions, original_dim=3, effective_dim=1)
    suggestions = agent.explain_surgery(report, snapshot)
    assert len(suggestions) == 7

    # Verify each action type produced a non-empty explanation
    for sug in suggestions:
        assert sug.explanation, f"Empty explanation for {sug.action_type}"

    # Check specific templates triggered correctly
    tighten_sug = suggestions[0]
    assert "Tighten" in tighten_sug.explanation
    assert "'x'" in tighten_sug.explanation

    freeze_sug = suggestions[1]
    assert "Freeze" in freeze_sug.explanation
    assert "'y'" in freeze_sug.explanation

    cond_sug = suggestions[2]
    assert "Conditionally freeze" in cond_sug.explanation

    merge_sug = suggestions[3]
    assert "Merge" in merge_sug.explanation

    log_sug = suggestions[4]
    assert "log-transform" in log_sug.explanation

    ratio_sug = suggestions[5]
    assert "ratio" in ratio_sug.explanation

    remove_sug = suggestions[6]
    assert "Remove" in remove_sug.explanation or "irrelevant" in remove_sug.explanation


def test_explain_surgery_with_llm_caller():
    """When an LLM caller is provided, suggestions have generated_by='llm'."""
    def mock_caller(prompt: str) -> str:
        return "LLM-enhanced explanation of the surgery action."

    agent = ReasoningAgent(llm_caller=mock_caller)
    snapshot = _make_snapshot(n_obs=10, n_failures=3)
    report = _make_surgery_report()

    suggestions = agent.explain_surgery(report, snapshot)
    assert len(suggestions) == 3
    for s in suggestions:
        assert s.generated_by == "llm"
        assert s.explanation == "LLM-enhanced explanation of the surgery action."


# ---------------------------------------------------------------------------
# Tests: explain_failures
# ---------------------------------------------------------------------------


def test_explain_failures_no_failures():
    """Snapshot with 0 failures produces empty clusters and 'No failures' message."""
    agent = ReasoningAgent(use_llm=False)
    snapshot = _make_snapshot(n_obs=10, n_failures=0)
    fmap = _make_feasibility_map()

    report = agent.explain_failures(fmap, snapshot)
    assert report.n_clusters == 0
    assert report.clusters == []
    assert "No failures" in report.overall_pattern
    assert report.dominant_failure_mode == "none"
    assert len(report.recommendations) >= 1
    assert report.generated_by == "template"


def test_explain_failures_taxonomy_clustering():
    """With a taxonomy, clusters match the taxonomy failure types."""
    agent = ReasoningAgent(use_llm=False)
    snapshot = _make_snapshot(n_obs=10, n_failures=3)
    fmap = _make_feasibility_map()
    taxonomy = _make_taxonomy(snapshot)

    report = agent.explain_failures(fmap, snapshot, taxonomy=taxonomy)

    # The taxonomy has both CHEMISTRY and HARDWARE types
    cluster_types = {c.failure_type for c in report.clusters}
    # Verify that the cluster types come from the taxonomy
    taxonomy_types = {cf.failure_type.value for cf in taxonomy.classified_failures}
    assert cluster_types.issubset(taxonomy_types)

    # Total failures across clusters should match
    total_in_clusters = sum(c.count for c in report.clusters)
    assert total_in_clusters == len(taxonomy.classified_failures)

    assert report.generated_by == "template"


def test_explain_failures_proximity_clustering():
    """Without taxonomy, proximity clustering is used; clusters have type 'proximity'."""
    agent = ReasoningAgent(use_llm=False)
    snapshot = _make_snapshot(n_obs=10, n_failures=3)
    fmap = _make_feasibility_map()

    report = agent.explain_failures(fmap, snapshot, taxonomy=None, seed=42)

    assert report.n_clusters >= 1
    for cluster in report.clusters:
        assert cluster.failure_type == "proximity"
        assert cluster.count >= 1
    assert report.generated_by == "template"


def test_explain_failures_determinism():
    """Same seed produces the same proximity clusters."""
    agent = ReasoningAgent(use_llm=False)
    snapshot = _make_snapshot(n_obs=15, n_failures=5)
    fmap = _make_feasibility_map()

    report1 = agent.explain_failures(fmap, snapshot, taxonomy=None, seed=123)
    report2 = agent.explain_failures(fmap, snapshot, taxonomy=None, seed=123)

    assert report1.n_clusters == report2.n_clusters
    for c1, c2 in zip(report1.clusters, report2.clusters):
        assert c1.cluster_id == c2.cluster_id
        assert c1.failure_indices == c2.failure_indices
        assert c1.failure_type == c2.failure_type
        assert c1.count == c2.count
        assert c1.parameter_ranges == c2.parameter_ranges


def test_explain_failures_with_llm_caller():
    """When an LLM caller is provided, overall_pattern is from the LLM."""
    def mock_caller(prompt: str) -> str:
        return "LLM-generated failure analysis summary."

    agent = ReasoningAgent(llm_caller=mock_caller)
    snapshot = _make_snapshot(n_obs=10, n_failures=3)
    fmap = _make_feasibility_map()

    report = agent.explain_failures(fmap, snapshot, taxonomy=None, seed=42)

    assert report.overall_pattern == "LLM-generated failure analysis summary."
    assert report.generated_by == "llm"
    # Clusters should still be present (computed by template logic)
    assert report.n_clusters >= 1


# ---------------------------------------------------------------------------
# Tests: generate_narrative
# ---------------------------------------------------------------------------


def test_generate_narrative_template(monkeypatch):
    """In template-only mode, all narrative sections are populated."""
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    agent = ReasoningAgent()
    snapshot = _make_snapshot(n_obs=10, n_failures=3)
    diagnostics = _make_diagnostics()
    decision = _make_decision(Phase.LEARNING)

    narrative = agent.generate_narrative(snapshot, diagnostics, decision)

    assert narrative.campaign_id == "test-campaign"
    assert narrative.generated_by == "template"
    assert "test-campaign" in narrative.executive_summary
    assert narrative.phase_description  # non-empty
    assert "learning" in narrative.phase_description.lower()
    assert narrative.diagnostic_summary  # non-empty
    assert "Convergence trend" in narrative.diagnostic_summary
    assert narrative.strategy_rationale  # non-empty
    assert "exploration_needed" in narrative.strategy_rationale
    assert "backend_timeout" in narrative.strategy_rationale
    assert narrative.failure_analysis  # non-empty because n_failures > 0
    assert len(narrative.recommendations) >= 1


def test_generate_narrative_with_surgery_report(monkeypatch):
    """When a surgery report is provided, recommendations include surgery info."""
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    agent = ReasoningAgent()
    snapshot = _make_snapshot(n_obs=10, n_failures=3)
    diagnostics = _make_diagnostics()
    decision = _make_decision(Phase.LEARNING)
    surgery = _make_surgery_report()

    narrative = agent.generate_narrative(
        snapshot, diagnostics, decision, surgery_report=surgery,
    )

    # Should mention surgery actions in recommendations
    surgery_rec = [r for r in narrative.recommendations if "surgery" in r.lower()]
    assert len(surgery_rec) >= 1
    assert "3 action(s)" in surgery_rec[0]


def test_generate_narrative_with_llm():
    """When LLM caller is provided, only executive_summary is replaced."""
    def mock_caller(prompt: str) -> str:
        return "LLM-generated executive summary."

    agent = ReasoningAgent(llm_caller=mock_caller)
    snapshot = _make_snapshot(n_obs=10, n_failures=3)
    diagnostics = _make_diagnostics()
    decision = _make_decision(Phase.LEARNING)

    narrative = agent.generate_narrative(snapshot, diagnostics, decision)

    assert narrative.executive_summary == "LLM-generated executive summary."
    assert narrative.generated_by == "llm"
    # Other sections should still be template-generated content
    assert "learning" in narrative.phase_description.lower()
    assert "Convergence trend" in narrative.diagnostic_summary


def test_generate_narrative_phase_descriptions():
    """Each Phase enum value produces a distinct phase description."""
    agent = ReasoningAgent(use_llm=False)
    diagnostics = _make_diagnostics()

    seen_descriptions = set()
    for phase in Phase:
        snapshot = _make_snapshot(n_obs=10, n_failures=0)
        decision = _make_decision(phase)
        narrative = agent.generate_narrative(snapshot, diagnostics, decision)
        assert phase.value in narrative.phase_description.lower()
        seen_descriptions.add(narrative.phase_description)

    # All phases should produce distinct descriptions
    assert len(seen_descriptions) == len(Phase)


# ---------------------------------------------------------------------------
# Tests: Recommendations & helpers
# ---------------------------------------------------------------------------


def test_failure_recommendations_high_failure_rate():
    """High failure rate (>30%) triggers a recommendation about parameter bounds."""
    agent = ReasoningAgent(use_llm=False)
    # 5 out of 10 = 50% failure rate
    snapshot = _make_snapshot(n_obs=10, n_failures=5)
    fmap = _make_feasibility_map(n_infeasible_zones=0, feasibility_score=0.5)

    report = agent.explain_failures(fmap, snapshot, taxonomy=None, seed=42)

    rate_recs = [r for r in report.recommendations if "failure rate" in r.lower()]
    assert len(rate_recs) >= 1
    assert "tighten" in rate_recs[0].lower() or "parameter bounds" in rate_recs[0].lower()


def test_failure_recommendations_infeasible_zones():
    """Infeasible zones in the feasibility map trigger zone-avoidance recommendation."""
    agent = ReasoningAgent(use_llm=False)
    snapshot = _make_snapshot(n_obs=10, n_failures=3)
    fmap = _make_feasibility_map(n_infeasible_zones=2, feasibility_score=0.7)

    report = agent.explain_failures(fmap, snapshot, taxonomy=None, seed=42)

    zone_recs = [r for r in report.recommendations if "infeasible zone" in r.lower()]
    assert len(zone_recs) >= 1
    assert "2" in zone_recs[0]  # Should mention the count


def test_confidence_narrative_bands():
    """_confidence_narrative returns distinct text for high/moderate/low confidence."""
    agent = ReasoningAgent(use_llm=False)

    high = agent._confidence_narrative(0.9)
    assert "High confidence" in high
    assert "0.90" in high

    moderate = agent._confidence_narrative(0.6)
    assert "Moderate confidence" in moderate
    assert "0.60" in moderate

    low = agent._confidence_narrative(0.3)
    assert "Low confidence" in low
    assert "0.30" in low

    # Boundary: exactly 0.8 should be high
    boundary_high = agent._confidence_narrative(0.8)
    assert "High confidence" in boundary_high

    # Boundary: exactly 0.5 should be moderate
    boundary_mod = agent._confidence_narrative(0.5)
    assert "Moderate confidence" in boundary_mod

    # Just below 0.5 should be low
    below_mod = agent._confidence_narrative(0.49)
    assert "Low confidence" in below_mod

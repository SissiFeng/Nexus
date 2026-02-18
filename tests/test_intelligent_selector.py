"""Tests for intelligent algorithm selector."""

import pytest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    DataScale,
    NoiseRegime,
    Observation,
    ParameterSpec,
    Phase,
    ProblemFingerprint,
    VariableType,
)
from optimization_copilot.meta_controller.intelligent_selector import (
    IntelligentAlgorithmSelector,
    AlgorithmRecommendation,
    AlgorithmSwitchExplanation,
)


def test_analyze_data_characteristics():
    """Test data characteristics extraction."""
    selector = IntelligentAlgorithmSelector()
    
    snapshot = CampaignSnapshot(
        campaign_id="test",
        parameter_specs=[
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0, upper=1),
            ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0, upper=1),
        ],
        observations=[],
        objective_names=["obj"],
        objective_directions=["maximize"],
    )
    
    diagnostics = {
        "noise_estimate": 0.15,
        "snr": 8.0,
    }
    
    fingerprint = ProblemFingerprint(
        variable_types=VariableType.CONTINUOUS,
        noise_regime=NoiseRegime.MEDIUM,
        data_scale=DataScale.TINY,
    )
    
    chars = selector.analyze_data_characteristics(snapshot, diagnostics, fingerprint)
    
    assert chars["n_dimensions"] == 2
    assert chars["variable_type"] == "continuous"
    assert chars["noise_regime"] == "medium"
    assert chars["n_constraints"] == 0


def test_score_algorithm_fit():
    """Test algorithm scoring."""
    selector = IntelligentAlgorithmSelector()
    
    characteristics = {
        "noise_regime": "low",
        "n_dimensions": 5,
        "variable_type": "continuous",
        "n_constraints": 0,
    }
    
    # TPE should score well for low noise, continuous, medium dim
    tpe_score = selector.score_algorithm_fit("tpe", characteristics, Phase.LEARNING)
    assert 0 < tpe_score <= 1
    
    # CMA-ES should score well for continuous, low noise
    cma_score = selector.score_algorithm_fit("cma_es", characteristics, Phase.LEARNING)
    assert 0 < cma_score <= 1
    
    # Random should have moderate score (works everywhere)
    random_score = selector.score_algorithm_fit("random", characteristics, Phase.LEARNING)
    assert 0 < random_score <= 1


def test_recommend_algorithm():
    """Test full algorithm recommendation."""
    selector = IntelligentAlgorithmSelector()
    
    snapshot = CampaignSnapshot(
        campaign_id="test",
        parameter_specs=[
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0, upper=1),
            ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0, upper=1),
        ],
        observations=[
            Observation(
                iteration=i,
                parameters={"x1": i * 0.1, "x2": i * 0.1},
                kpi_values={"obj": i * 0.5},
            )
            for i in range(15)
        ],
        objective_names=["obj"],
        objective_directions=["maximize"],
    )
    
    diagnostics = {
        "noise_estimate": 0.1,
        "snr": 10.0,
        "exploration_coverage": 0.5,
        "convergence_trend": 0.3,
    }
    
    fingerprint = ProblemFingerprint(
        variable_types=VariableType.CONTINUOUS,
        noise_regime=NoiseRegime.LOW,
        data_scale=DataScale.SMALL,
    )
    
    rec = selector.recommend_algorithm(
        snapshot=snapshot,
        diagnostics=diagnostics,
        fingerprint=fingerprint,
        phase=Phase.LEARNING,
    )
    
    assert isinstance(rec, AlgorithmRecommendation)
    assert rec.algorithm in selector.ALGORITHM_PROFILES
    assert 0 <= rec.confidence <= 1
    assert len(rec.reason) > 0
    assert len(rec.trade_offs) > 0
    assert len(rec.expected_performance) > 0
    assert len(rec.when_to_switch) > 0


def test_explain_switch():
    """Test algorithm switch explanation."""
    selector = IntelligentAlgorithmSelector()
    
    snapshot = CampaignSnapshot(
        campaign_id="test",
        parameter_specs=[
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0, upper=1),
        ],
        observations=[],
        objective_names=["obj"],
        objective_directions=["maximize"],
    )
    
    # High noise scenario - should recommend switching from CMA-ES
    diagnostics = {
        "noise_estimate": 0.6,
        "snr": 1.5,
    }
    
    fingerprint = ProblemFingerprint(
        variable_types=VariableType.CONTINUOUS,
        noise_regime=NoiseRegime.HIGH,
        data_scale=DataScale.TINY,
    )
    
    switch = selector.explain_switch(
        current_algorithm="cma_es",
        snapshot=snapshot,
        diagnostics=diagnostics,
        fingerprint=fingerprint,
        phase=Phase.LEARNING,
    )
    
    # Should recommend a switch due to high noise
    assert switch is not None
    assert switch.from_algorithm == "cma_es"
    assert switch.to_algorithm != "cma_es"
    assert len(switch.explanation) > 0
    assert switch.confidence_gain >= 0


def test_no_switch_when_optimal():
    """Test that no switch is recommended when already using best algorithm."""
    selector = IntelligentAlgorithmSelector()
    
    snapshot = CampaignSnapshot(
        campaign_id="test",
        parameter_specs=[
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0, upper=1),
        ],
        observations=[
            Observation(iteration=i, parameters={"x1": i * 0.1}, kpi_values={"obj": i})
            for i in range(20)
        ],
        objective_names=["obj"],
        objective_directions=["maximize"],
    )
    
    diagnostics = {
        "noise_estimate": 0.1,
        "snr": 10.0,
    }
    
    fingerprint = ProblemFingerprint(
        variable_types=VariableType.CONTINUOUS,
        noise_regime=NoiseRegime.LOW,
        data_scale=DataScale.SMALL,
    )
    
    # Get recommendation
    rec = selector.recommend_algorithm(
        snapshot=snapshot,
        diagnostics=diagnostics,
        fingerprint=fingerprint,
        phase=Phase.LEARNING,
    )
    
    # Check if switch is recommended when using the recommended algorithm
    switch = selector.explain_switch(
        current_algorithm=rec.algorithm,
        snapshot=snapshot,
        diagnostics=diagnostics,
        fingerprint=fingerprint,
        phase=Phase.LEARNING,
    )
    
    # No switch should be needed
    assert switch is None


def test_selection_history():
    """Test that selection history is tracked."""
    selector = IntelligentAlgorithmSelector()
    
    snapshot = CampaignSnapshot(
        campaign_id="test",
        parameter_specs=[
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0, upper=1),
        ],
        observations=[
            Observation(iteration=i, parameters={"x1": i * 0.1}, kpi_values={"obj": i})
            for i in range(20)
        ],
        objective_names=["obj"],
        objective_directions=["maximize"],
    )
    
    diagnostics = {"noise_estimate": 0.1, "snr": 10.0}
    fingerprint = ProblemFingerprint(
        variable_types=VariableType.CONTINUOUS,
        noise_regime=NoiseRegime.LOW,
    )
    
    # Make a recommendation
    selector.recommend_algorithm(
        snapshot=snapshot,
        diagnostics=diagnostics,
        fingerprint=fingerprint,
        phase=Phase.LEARNING,
    )
    
    # Check history
    history = selector.get_selection_history()
    assert len(history) == 1
    assert "algorithm" in history[0]
    assert "score" in history[0]

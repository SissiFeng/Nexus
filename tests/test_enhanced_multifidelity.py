"""Tests for enhanced multi-fidelity optimization."""

import pytest

from optimization_copilot.core.models import CampaignSnapshot, ParameterSpec
from optimization_copilot.multi_fidelity.planner import FidelityLevel
from optimization_copilot.multi_fidelity.enhanced_planner import (
    EnhancedMultiFidelityOptimizer,
    AdaptiveFidelityState,
    FidelityGateResult,
)


def test_initialization():
    """Test optimizer initialization."""
    optimizer = EnhancedMultiFidelityOptimizer(cost_budget=100.0)
    
    assert optimizer.cost_budget == 100.0
    assert optimizer._state.budget_remaining == 100.0


def test_initialize_campaign():
    """Test campaign initialization with plan generation."""
    fidelity_levels = [
        FidelityLevel(name="simulation", cost_multiplier=1.0, noise_multiplier=2.0, correlation_with_truth=0.6),
        FidelityLevel(name="experiment", cost_multiplier=10.0, noise_multiplier=1.0, correlation_with_truth=1.0),
    ]
    
    optimizer = EnhancedMultiFidelityOptimizer(
        fidelity_levels=fidelity_levels,
        cost_budget=100.0,
    )
    
    plan = optimizer.initialize_campaign(n_candidates=20)
    
    assert len(plan.stages) == 2
    assert plan.stages[0].stage == "screening"
    assert plan.stages[1].stage == "refinement"
    assert plan.total_estimated_cost > 0
    assert 0 <= plan.efficiency_gain <= 1


def test_budget_constrained_planning():
    """Test that planning respects budget constraints."""
    fidelity_levels = [
        FidelityLevel(name="cheap", cost_multiplier=1.0, noise_multiplier=2.0, correlation_with_truth=0.6),
        FidelityLevel(name="expensive", cost_multiplier=10.0, noise_multiplier=1.0, correlation_with_truth=1.0),
    ]
    
    optimizer = EnhancedMultiFidelityOptimizer(
        fidelity_levels=fidelity_levels,
        cost_budget=50.0,  # Limited budget
    )
    
    plan = optimizer.initialize_campaign(n_candidates=100)  # Request many candidates
    
    # Should fit within budget
    assert plan.total_estimated_cost <= 50.0


def test_fidelity_gate_evaluation():
    """Test evaluation at a fidelity level."""
    fidelity_levels = [
        FidelityLevel(name="cheap", cost_multiplier=1.0, noise_multiplier=2.0, correlation_with_truth=0.6),
        FidelityLevel(name="expensive", cost_multiplier=10.0, noise_multiplier=1.0, correlation_with_truth=1.0),
    ]
    
    optimizer = EnhancedMultiFidelityOptimizer(fidelity_levels=fidelity_levels)
    optimizer.initialize_campaign(n_candidates=10)
    
    def mock_evaluator(params, fidelity_idx):
        return params.get("x", 0.5) * 10  # Simple scoring
    
    candidate = {"x": 0.8}
    result = optimizer.evaluate_at_fidelity(candidate, 0, mock_evaluator)
    
    assert isinstance(result, FidelityGateResult)
    assert result.candidate == candidate
    assert result.cost_incurred == 1.0  # cheap fidelity cost
    assert result.score == 8.0  # 0.8 * 10


def test_adaptive_promotion():
    """Test adaptive candidate promotion strategy."""
    fidelity_levels = [
        FidelityLevel(name="cheap", cost_multiplier=1.0, noise_multiplier=2.0, correlation_with_truth=0.6),
        FidelityLevel(name="expensive", cost_multiplier=10.0, noise_multiplier=1.0, correlation_with_truth=1.0),
    ]
    
    optimizer = EnhancedMultiFidelityOptimizer(fidelity_levels=fidelity_levels)
    optimizer.initialize_campaign(n_candidates=10)
    
    candidates = [{"x": i * 0.1} for i in range(10)]
    
    # Mock observations at fidelity 0
    observations_by_fidelity = {
        0: [(c, c["x"] * 10) for c in candidates],
    }
    
    promoted, discarded = optimizer.adaptive_promotion_strategy(
        candidates=candidates,
        observations_by_fidelity=observations_by_fidelity,
        current_stage=0,
    )
    
    # Should promote some, discard others
    assert len(promoted) > 0
    assert len(discarded) > 0
    assert len(promoted) + len(discarded) == len(candidates)


def test_fidelity_correlation_estimation():
    """Test correlation estimation between fidelity levels."""
    optimizer = EnhancedMultiFidelityOptimizer()
    
    # Perfect correlation
    low_fidelity = [
        ({"x": 0.1}, 1.0),
        ({"x": 0.2}, 2.0),
        ({"x": 0.3}, 3.0),
    ]
    high_fidelity = [
        ({"x": 0.1}, 1.0),
        ({"x": 0.2}, 2.0),
        ({"x": 0.3}, 3.0),
    ]
    
    corr = optimizer._estimate_fidelity_correlation(low_fidelity, high_fidelity)
    assert corr > 0.9  # Should be very high


def test_high_fidelity_score_estimation():
    """Test estimation of high fidelity scores from low fidelity."""
    fidelity_levels = [
        FidelityLevel(name="cheap", cost_multiplier=1.0, noise_multiplier=2.0, correlation_with_truth=0.6),
        FidelityLevel(name="expensive", cost_multiplier=10.0, noise_multiplier=1.0, correlation_with_truth=1.0),
    ]
    
    optimizer = EnhancedMultiFidelityOptimizer(fidelity_levels=fidelity_levels)
    optimizer._state.fidelity_correlation[(0, 1)] = 0.8
    
    estimated, uncertainty = optimizer._estimate_high_fidelity_score(
        low_fidelity_score=5.0,
        fidelity_idx=0,
    )
    
    # Should be correlated with low fidelity score
    assert estimated != 0
    assert uncertainty >= 0


def test_cost_efficiency_report():
    """Test cost efficiency report generation."""
    fidelity_levels = [
        FidelityLevel(name="cheap", cost_multiplier=1.0, noise_multiplier=2.0, correlation_with_truth=0.6),
        FidelityLevel(name="expensive", cost_multiplier=10.0, noise_multiplier=1.0, correlation_with_truth=1.0),
    ]
    
    optimizer = EnhancedMultiFidelityOptimizer(
        fidelity_levels=fidelity_levels,
        cost_budget=100.0,
    )
    optimizer.initialize_campaign(n_candidates=10)
    
    # Simulate some evaluations
    def mock_evaluator(params, fidelity_idx):
        return params.get("x", 0.5) * 10
    
    for i in range(5):
        optimizer.evaluate_at_fidelity({"x": i * 0.1}, 0, mock_evaluator)
    
    report = optimizer.get_cost_efficiency_report()
    
    assert "budget_total" in report
    assert "budget_spent" in report
    assert report["budget_spent"] == 5.0  # 5 evaluations at cost 1.0
    assert report["budget_remaining"] == 95.0
    assert report["candidates_evaluated"] == 5


def test_parameter_distance():
    """Test parameter distance calculation."""
    optimizer = EnhancedMultiFidelityOptimizer()
    
    params1 = {"x": 0.0, "y": 0.0}
    params2 = {"x": 1.0, "y": 0.0}
    
    dist = optimizer._parameter_distance(params1, params2)
    
    assert dist > 0
    assert dist < 2  # Should be normalized


def test_score_with_uncertainty():
    """Test scoring with uncertainty."""
    optimizer = EnhancedMultiFidelityOptimizer()
    
    observations = [
        ({"x": 0.0}, 0.0),
        ({"x": 0.5}, 5.0),
        ({"x": 1.0}, 10.0),
    ]
    
    candidate = {"x": 0.5}  # Exact match
    score, uncertainty = optimizer._score_with_uncertainty(candidate, observations)
    
    assert score == 5.0  # Exact match
    assert uncertainty == 0.0  # No uncertainty for exact match

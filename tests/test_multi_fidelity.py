"""Tests for the multi-fidelity planner module."""

import math

import pytest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.multi_fidelity.planner import (
    DEFAULT_FIDELITY_LEVELS,
    FidelityLevel,
    FidelityPlan,
    MultiFidelityPlan,
    MultiFidelityPlanner,
)


# ── Fixtures / helpers ───────────────────────────────────


def _make_snapshot(n_obs: int = 5) -> CampaignSnapshot:
    """Build a minimal campaign snapshot for testing."""
    specs = [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=-1.0, upper=1.0),
    ]
    obs = [
        Observation(
            iteration=i,
            parameters={"x1": i * 0.1, "x2": i * -0.1},
            kpi_values={"y": float(i)},
            timestamp=float(i),
        )
        for i in range(n_obs)
    ]
    return CampaignSnapshot(
        campaign_id="mf-test-001",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
    )


def _make_observations(values: list[float], obj_name: str = "y") -> list[Observation]:
    """Create observations with specified KPI values."""
    return [
        Observation(
            iteration=i,
            parameters={"x": float(i)},
            kpi_values={obj_name: v},
            timestamp=float(i),
        )
        for i, v in enumerate(values)
    ]


# ── Test default fidelity levels ─────────────────────────


class TestDefaultFidelityLevels:
    def test_default_levels_exist(self):
        assert len(DEFAULT_FIDELITY_LEVELS) == 2

    def test_low_fidelity_properties(self):
        low = DEFAULT_FIDELITY_LEVELS[0]
        assert low.name == "low"
        assert low.cost_multiplier == 1.0
        assert low.noise_multiplier == 2.0
        assert low.correlation_with_truth == 0.6

    def test_high_fidelity_properties(self):
        high = DEFAULT_FIDELITY_LEVELS[1]
        assert high.name == "high"
        assert high.cost_multiplier == 10.0
        assert high.noise_multiplier == 1.0
        assert high.correlation_with_truth == 1.0

    def test_planner_uses_defaults_when_none(self):
        planner = MultiFidelityPlanner()
        assert len(planner.fidelity_levels) == 2
        assert planner.fidelity_levels[0].name == "low"
        assert planner.fidelity_levels[1].name == "high"


class TestFidelityLevelValidation:
    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="cost_multiplier"):
            FidelityLevel("bad", cost_multiplier=-1.0, noise_multiplier=1.0, correlation_with_truth=0.5)

    def test_zero_cost_raises(self):
        with pytest.raises(ValueError, match="cost_multiplier"):
            FidelityLevel("bad", cost_multiplier=0.0, noise_multiplier=1.0, correlation_with_truth=0.5)

    def test_negative_noise_raises(self):
        with pytest.raises(ValueError, match="noise_multiplier"):
            FidelityLevel("bad", cost_multiplier=1.0, noise_multiplier=-1.0, correlation_with_truth=0.5)

    def test_correlation_out_of_range_raises(self):
        with pytest.raises(ValueError, match="correlation_with_truth"):
            FidelityLevel("bad", cost_multiplier=1.0, noise_multiplier=1.0, correlation_with_truth=1.5)

        with pytest.raises(ValueError, match="correlation_with_truth"):
            FidelityLevel("bad", cost_multiplier=1.0, noise_multiplier=1.0, correlation_with_truth=-0.1)


# ── Test plan generation with budget ─────────────────────


class TestPlanWithBudget:
    def test_plan_fits_within_budget(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        budget = 50.0
        plan = planner.plan(snapshot, budget=budget, n_total=100)
        assert plan.total_estimated_cost <= budget

    def test_plan_uses_full_budget_approximately(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        budget = 200.0
        plan = planner.plan(snapshot, budget=budget, n_total=100)
        # Should use a reasonable fraction of the budget
        assert plan.total_estimated_cost <= budget
        assert plan.total_estimated_cost > 0

    def test_budget_scales_candidates_down(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        # Very tight budget should reduce candidates
        plan_tight = planner.plan(snapshot, budget=30.0, n_total=100)
        plan_loose = planner.plan(snapshot, budget=500.0, n_total=100)
        assert plan_tight.stages[0].n_candidates <= plan_loose.stages[0].n_candidates

    def test_minimum_two_candidates_under_budget(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        # Extremely tight budget
        plan = planner.plan(snapshot, budget=1.0, n_total=100)
        assert plan.stages[0].n_candidates >= 2


# ── Test plan generation without budget ──────────────────


class TestPlanWithoutBudget:
    def test_plan_has_two_stages_by_default(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        assert len(plan.stages) == 2

    def test_first_stage_is_screening(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        assert plan.stages[0].stage == "screening"
        assert plan.stages[0].fidelity_level.name == "low"

    def test_second_stage_is_refinement(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        assert plan.stages[1].stage == "refinement"
        assert plan.stages[1].fidelity_level.name == "high"

    def test_screening_uses_broad_search_hint(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        assert plan.stages[0].backend_hint in ("random", "latin_hypercube")

    def test_refinement_uses_tpe_hint(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        assert plan.stages[1].backend_hint == "tpe"

    def test_successive_halving_reduces_candidates(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        assert plan.stages[0].n_candidates == 20
        # Top 50% promoted => 10 candidates in refinement
        assert plan.stages[1].n_candidates == 10

    def test_screening_promotion_threshold_is_half(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        assert plan.stages[0].promotion_threshold == 0.5

    def test_refinement_promotion_threshold_is_one(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        assert plan.stages[1].promotion_threshold == 1.0

    def test_plan_has_positive_efficiency_gain(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        assert plan.efficiency_gain > 0.0

    def test_small_n_total_uses_random_hint(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=5)
        assert plan.stages[0].backend_hint == "random"

    def test_large_n_total_uses_latin_hypercube_hint(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        assert plan.stages[0].backend_hint == "latin_hypercube"

    def test_stages_have_reasons(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        for stage in plan.stages:
            assert stage.reason
            assert len(stage.reason) > 10


# ── Test promotion threshold computation ─────────────────


class TestPromotionThreshold:
    def test_top_50_percent_maximize(self):
        obs = _make_observations([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        planner = MultiFidelityPlanner()
        threshold = planner.compute_promotion_threshold(obs, "y", top_fraction=0.5, maximize=True)
        # Top 50% of 10 values (maximizing): 10,9,8,7,6 => threshold is 6.0
        assert threshold == 6.0

    def test_top_50_percent_minimize(self):
        obs = _make_observations([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        planner = MultiFidelityPlanner()
        threshold = planner.compute_promotion_threshold(obs, "y", top_fraction=0.5, maximize=False)
        # Top 50% of 10 values (minimizing): 1,2,3,4,5 => threshold is 5.0
        assert threshold == 5.0

    def test_top_fraction_one_returns_worst(self):
        obs = _make_observations([1.0, 5.0, 3.0])
        planner = MultiFidelityPlanner()
        threshold = planner.compute_promotion_threshold(obs, "y", top_fraction=1.0, maximize=True)
        assert threshold == 1.0  # All promoted, threshold at worst value

    def test_small_top_fraction(self):
        obs = _make_observations([10.0, 20.0, 30.0, 40.0, 50.0])
        planner = MultiFidelityPlanner()
        # top 20% of 5 = ceil(1) = 1 candidate
        threshold = planner.compute_promotion_threshold(obs, "y", top_fraction=0.2, maximize=True)
        assert threshold == 50.0

    def test_empty_observations_raises(self):
        planner = MultiFidelityPlanner()
        with pytest.raises(ValueError, match="No observations"):
            planner.compute_promotion_threshold([], "y", top_fraction=0.5, maximize=True)

    def test_missing_kpi_raises(self):
        obs = _make_observations([1.0, 2.0])
        planner = MultiFidelityPlanner()
        with pytest.raises(ValueError, match="No observations contain KPI"):
            planner.compute_promotion_threshold(obs, "nonexistent", top_fraction=0.5, maximize=True)

    def test_invalid_top_fraction_raises(self):
        obs = _make_observations([1.0])
        planner = MultiFidelityPlanner()
        with pytest.raises(ValueError, match="top_fraction"):
            planner.compute_promotion_threshold(obs, "y", top_fraction=0.0, maximize=True)
        with pytest.raises(ValueError, match="top_fraction"):
            planner.compute_promotion_threshold(obs, "y", top_fraction=1.5, maximize=True)


# ── Test should_promote ──────────────────────────────────


class TestShouldPromote:
    def test_promote_when_above_threshold_maximize(self):
        obs = Observation(iteration=0, parameters={"x": 1.0}, kpi_values={"y": 8.0})
        planner = MultiFidelityPlanner()
        assert planner.should_promote(obs, threshold_kpi=5.0, maximize=True) is True

    def test_reject_when_below_threshold_maximize(self):
        obs = Observation(iteration=0, parameters={"x": 1.0}, kpi_values={"y": 3.0})
        planner = MultiFidelityPlanner()
        assert planner.should_promote(obs, threshold_kpi=5.0, maximize=True) is False

    def test_promote_when_at_threshold_maximize(self):
        obs = Observation(iteration=0, parameters={"x": 1.0}, kpi_values={"y": 5.0})
        planner = MultiFidelityPlanner()
        assert planner.should_promote(obs, threshold_kpi=5.0, maximize=True) is True

    def test_promote_when_below_threshold_minimize(self):
        obs = Observation(iteration=0, parameters={"x": 1.0}, kpi_values={"y": 3.0})
        planner = MultiFidelityPlanner()
        assert planner.should_promote(obs, threshold_kpi=5.0, maximize=False) is True

    def test_reject_when_above_threshold_minimize(self):
        obs = Observation(iteration=0, parameters={"x": 1.0}, kpi_values={"y": 8.0})
        planner = MultiFidelityPlanner()
        assert planner.should_promote(obs, threshold_kpi=5.0, maximize=False) is False

    def test_promote_when_at_threshold_minimize(self):
        obs = Observation(iteration=0, parameters={"x": 1.0}, kpi_values={"y": 5.0})
        planner = MultiFidelityPlanner()
        assert planner.should_promote(obs, threshold_kpi=5.0, maximize=False) is True

    def test_empty_kpi_values_not_promoted(self):
        obs = Observation(iteration=0, parameters={"x": 1.0}, kpi_values={})
        planner = MultiFidelityPlanner()
        assert planner.should_promote(obs, threshold_kpi=5.0, maximize=True) is False


# ── Test efficiency estimation ───────────────────────────


class TestEfficiencyEstimation:
    def test_positive_savings(self):
        planner = MultiFidelityPlanner()
        plan = MultiFidelityPlan(stages=[], total_estimated_cost=40.0, efficiency_gain=0.0)
        eff = planner.estimate_efficiency(plan, single_fidelity_cost=100.0)
        assert eff == pytest.approx(0.6)

    def test_no_savings_when_equal_cost(self):
        planner = MultiFidelityPlanner()
        plan = MultiFidelityPlan(stages=[], total_estimated_cost=100.0, efficiency_gain=0.0)
        eff = planner.estimate_efficiency(plan, single_fidelity_cost=100.0)
        assert eff == pytest.approx(0.0)

    def test_no_savings_when_more_expensive(self):
        planner = MultiFidelityPlanner()
        plan = MultiFidelityPlan(stages=[], total_estimated_cost=150.0, efficiency_gain=0.0)
        eff = planner.estimate_efficiency(plan, single_fidelity_cost=100.0)
        assert eff == 0.0  # Clamped to 0

    def test_zero_single_fidelity_cost(self):
        planner = MultiFidelityPlanner()
        plan = MultiFidelityPlan(stages=[], total_estimated_cost=10.0, efficiency_gain=0.0)
        eff = planner.estimate_efficiency(plan, single_fidelity_cost=0.0)
        assert eff == 0.0

    def test_full_plan_efficiency(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        # 20 candidates at cost 1.0 = 20 (screening)
        # 10 candidates at cost 10.0 = 100 (refinement)
        # Total: 120
        # All-high-fidelity: 20 * 10 = 200
        # Efficiency: 1 - 120/200 = 0.4
        assert plan.total_estimated_cost == pytest.approx(120.0)
        assert plan.efficiency_gain == pytest.approx(0.4)


# ── Test cost calculation ────────────────────────────────


class TestCostCalculation:
    def test_default_two_level_cost(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        # Stage 1: 20 * 1.0 = 20
        # Stage 2: 10 * 10.0 = 100
        expected_cost = 20 * 1.0 + 10 * 10.0
        assert plan.total_estimated_cost == pytest.approx(expected_cost)

    def test_single_candidate_cost(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=2)
        # Stage 1: 2 * 1.0 = 2
        # Stage 2: 1 * 10.0 = 10
        assert plan.total_estimated_cost == pytest.approx(12.0)

    def test_cost_increases_with_n_total(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan_small = planner.plan(snapshot, n_total=10)
        plan_large = planner.plan(snapshot, n_total=50)
        assert plan_large.total_estimated_cost > plan_small.total_estimated_cost

    def test_custom_fidelity_costs(self):
        levels = [
            FidelityLevel("cheap", cost_multiplier=0.5, noise_multiplier=3.0, correlation_with_truth=0.4),
            FidelityLevel("expensive", cost_multiplier=20.0, noise_multiplier=0.5, correlation_with_truth=0.95),
        ]
        planner = MultiFidelityPlanner(fidelity_levels=levels)
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=10)
        # Stage 1: 10 * 0.5 = 5
        # Stage 2: 5 * 20.0 = 100
        assert plan.total_estimated_cost == pytest.approx(105.0)


# ── Test edge case: single fidelity level ────────────────


class TestSingleFidelityLevel:
    def test_single_level_produces_one_stage(self):
        levels = [FidelityLevel("only", cost_multiplier=5.0, noise_multiplier=1.0, correlation_with_truth=1.0)]
        planner = MultiFidelityPlanner(fidelity_levels=levels)
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=10)
        assert len(plan.stages) == 1

    def test_single_level_stage_is_refinement(self):
        levels = [FidelityLevel("only", cost_multiplier=5.0, noise_multiplier=1.0, correlation_with_truth=1.0)]
        planner = MultiFidelityPlanner(fidelity_levels=levels)
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=10)
        assert plan.stages[0].stage == "refinement"

    def test_single_level_no_efficiency_gain(self):
        levels = [FidelityLevel("only", cost_multiplier=5.0, noise_multiplier=1.0, correlation_with_truth=1.0)]
        planner = MultiFidelityPlanner(fidelity_levels=levels)
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=10)
        assert plan.efficiency_gain == 0.0

    def test_single_level_cost_is_n_times_multiplier(self):
        levels = [FidelityLevel("only", cost_multiplier=5.0, noise_multiplier=1.0, correlation_with_truth=1.0)]
        planner = MultiFidelityPlanner(fidelity_levels=levels)
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=10)
        assert plan.total_estimated_cost == pytest.approx(50.0)

    def test_single_level_all_candidates_evaluated(self):
        levels = [FidelityLevel("only", cost_multiplier=5.0, noise_multiplier=1.0, correlation_with_truth=1.0)]
        planner = MultiFidelityPlanner(fidelity_levels=levels)
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=15)
        assert plan.stages[0].n_candidates == 15

    def test_single_level_promotion_threshold_is_one(self):
        levels = [FidelityLevel("only", cost_multiplier=5.0, noise_multiplier=1.0, correlation_with_truth=1.0)]
        planner = MultiFidelityPlanner(fidelity_levels=levels)
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=10)
        assert plan.stages[0].promotion_threshold == 1.0


class TestPlannerEdgeCases:
    def test_empty_fidelity_levels_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            MultiFidelityPlanner(fidelity_levels=[])

    def test_levels_sorted_by_cost(self):
        high = FidelityLevel("high", cost_multiplier=10.0, noise_multiplier=1.0, correlation_with_truth=1.0)
        low = FidelityLevel("low", cost_multiplier=1.0, noise_multiplier=2.0, correlation_with_truth=0.6)
        planner = MultiFidelityPlanner(fidelity_levels=[high, low])
        # Should be sorted cheapest first
        assert planner.fidelity_levels[0].name == "low"
        assert planner.fidelity_levels[1].name == "high"

    def test_three_fidelity_levels(self):
        levels = [
            FidelityLevel("low", cost_multiplier=1.0, noise_multiplier=3.0, correlation_with_truth=0.4),
            FidelityLevel("mid", cost_multiplier=5.0, noise_multiplier=1.5, correlation_with_truth=0.8),
            FidelityLevel("high", cost_multiplier=20.0, noise_multiplier=0.5, correlation_with_truth=1.0),
        ]
        planner = MultiFidelityPlanner(fidelity_levels=levels)
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=20)
        assert len(plan.stages) == 3
        assert plan.stages[0].stage == "screening"
        assert plan.stages[1].stage == "screening"  # intermediate
        assert plan.stages[2].stage == "refinement"
        # 20 -> 10 -> 5
        assert plan.stages[0].n_candidates == 20
        assert plan.stages[1].n_candidates == 10
        assert plan.stages[2].n_candidates == 5

    def test_odd_n_total_halving(self):
        planner = MultiFidelityPlanner()
        snapshot = _make_snapshot()
        plan = planner.plan(snapshot, n_total=7)
        # ceil(7 * 0.5) = 4
        assert plan.stages[0].n_candidates == 7
        assert plan.stages[1].n_candidates == 4

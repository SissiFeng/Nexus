"""Comprehensive tests for the infrastructure package.

Covers all four modules:
- cost_tracker (TrialCost, CostTracker)
- stopping_rule (StoppingDecision, StoppingRule)
- auto_sampler (SelectionResult, AutoSampler)
- constraint_engine (ConstraintType, ConstraintStatus, Constraint,
                     ConstraintEvaluation, ConstraintEngine)
"""

from __future__ import annotations

import math

import pytest

from optimization_copilot.infrastructure import (
    AutoSampler,
    Constraint,
    ConstraintEngine,
    ConstraintEvaluation,
    ConstraintStatus,
    ConstraintType,
    CostTracker,
    SelectionResult,
    StoppingDecision,
    StoppingRule,
    TrialCost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trial(
    trial_id: str = "t1",
    wall: float = 1.0,
    resource: float = 2.0,
    compute: float = 3.0,
    opportunity: float = 4.0,
    fidelity: int = 0,
) -> TrialCost:
    return TrialCost(
        trial_id=trial_id,
        wall_time_seconds=wall,
        resource_cost=resource,
        compute_cost=compute,
        opportunity_cost=opportunity,
        fidelity_level=fidelity,
        timestamp=0.0,
    )


# ===================================================================
# 1. cost_tracker.py
# ===================================================================


class TestTrialCost:
    """Tests for the TrialCost dataclass."""

    def test_total_cost_property(self):
        tc = _make_trial(wall=1.0, resource=2.0, compute=3.0, opportunity=4.0)
        assert tc.total_cost == pytest.approx(10.0)

    def test_total_cost_zero(self):
        tc = TrialCost(trial_id="z", timestamp=0.0)
        assert tc.total_cost == pytest.approx(0.0)

    def test_to_dict_contains_all_fields(self):
        tc = _make_trial()
        d = tc.to_dict()
        assert d["trial_id"] == "t1"
        assert d["wall_time_seconds"] == pytest.approx(1.0)
        assert d["resource_cost"] == pytest.approx(2.0)
        assert d["compute_cost"] == pytest.approx(3.0)
        assert d["opportunity_cost"] == pytest.approx(4.0)
        assert d["fidelity_level"] == 0
        assert isinstance(d["metadata"], dict)

    def test_to_dict_from_dict_roundtrip(self):
        tc = _make_trial()
        tc.metadata["key"] = "value"
        d = tc.to_dict()
        restored = TrialCost.from_dict(d)
        assert restored.trial_id == tc.trial_id
        assert restored.total_cost == pytest.approx(tc.total_cost)
        assert restored.metadata == tc.metadata

    def test_from_dict_defaults(self):
        d = {"trial_id": "min"}
        restored = TrialCost.from_dict(d)
        assert restored.total_cost == pytest.approx(0.0)
        assert restored.fidelity_level == 0


class TestCostTracker:
    """Tests for CostTracker."""

    def test_empty_tracker(self):
        ct = CostTracker(budget=100.0)
        assert ct.total_spent == pytest.approx(0.0)
        assert ct.remaining_budget == pytest.approx(100.0)
        assert ct.n_trials == 0
        assert ct.average_cost_per_trial == pytest.approx(0.0)

    def test_record_trial_updates_counts(self):
        ct = CostTracker(budget=100.0)
        ct.record_trial(_make_trial(trial_id="a"))
        ct.record_trial(_make_trial(trial_id="b"))
        assert ct.n_trials == 2
        assert ct.total_spent == pytest.approx(20.0)

    def test_remaining_budget_decreases(self):
        ct = CostTracker(budget=50.0)
        ct.record_trial(_make_trial())  # total_cost = 10
        assert ct.remaining_budget == pytest.approx(40.0)

    def test_remaining_budget_clamps_to_zero(self):
        ct = CostTracker(budget=5.0)
        ct.record_trial(_make_trial())  # total_cost = 10 > budget
        assert ct.remaining_budget == pytest.approx(0.0)

    def test_no_budget_returns_none(self):
        ct = CostTracker(budget=None)
        assert ct.budget is None
        assert ct.remaining_budget is None

    def test_estimated_remaining_trials(self):
        ct = CostTracker(budget=100.0)
        ct.record_trial(_make_trial())  # cost = 10, avg = 10
        # remaining = 90, avg = 10 => 9
        assert ct.estimated_remaining_trials() == 9

    def test_estimated_remaining_trials_no_budget(self):
        ct = CostTracker(budget=None)
        ct.record_trial(_make_trial())
        assert ct.estimated_remaining_trials() is None

    def test_estimated_remaining_trials_no_trials(self):
        ct = CostTracker(budget=100.0)
        # avg_cost = 0 => None
        assert ct.estimated_remaining_trials() is None

    def test_cost_by_fidelity(self):
        ct = CostTracker()
        ct.record_trial(_make_trial(trial_id="a", fidelity=0, wall=1, resource=0, compute=0, opportunity=0))
        ct.record_trial(_make_trial(trial_id="b", fidelity=1, wall=5, resource=0, compute=0, opportunity=0))
        ct.record_trial(_make_trial(trial_id="c", fidelity=0, wall=2, resource=0, compute=0, opportunity=0))
        by_fid = ct.cost_by_fidelity()
        assert by_fid[0] == pytest.approx(3.0)
        assert by_fid[1] == pytest.approx(5.0)

    def test_cost_adjusted_acquisition_positive_values(self):
        ct = CostTracker()
        acq = [1.0, math.e, math.e**2]
        costs = [1.0, 1.0, 2.0]
        result = ct.cost_adjusted_acquisition(acq, costs)
        assert result[0] == pytest.approx(0.0)           # log(1)/1 = 0
        assert result[1] == pytest.approx(1.0)           # log(e)/1 = 1
        assert result[2] == pytest.approx(2.0 / 2.0)     # log(e^2)/2 = 1

    def test_cost_adjusted_acquisition_zero_acq(self):
        ct = CostTracker()
        result = ct.cost_adjusted_acquisition([0.0], [1.0])
        assert result[0] == -float("inf")

    def test_cost_adjusted_acquisition_negative_acq(self):
        ct = CostTracker()
        result = ct.cost_adjusted_acquisition([-1.0], [1.0])
        assert result[0] == -float("inf")

    def test_cost_adjusted_acquisition_zero_cost_uses_epsilon(self):
        ct = CostTracker()
        result = ct.cost_adjusted_acquisition([1.0], [0.0])
        # cost clamped to 1e-8, log(1)/1e-8 = 0
        assert result[0] == pytest.approx(0.0)

    def test_cost_adjusted_regret(self):
        ct = CostTracker(budget=100.0)
        ct.record_trial(_make_trial(trial_id="a", wall=5, resource=0, compute=0, opportunity=0))
        ct.record_trial(_make_trial(trial_id="b", wall=5, resource=0, compute=0, opportunity=0))
        best_values = [3.0, 4.0]
        regrets = ct.cost_adjusted_regret(best_values, optimal=5.0)
        assert len(regrets) == 2
        # First: regret = |5-3|=2, cum_cost=5 => 2/5
        assert regrets[0] == pytest.approx(2.0 / 5.0)
        # Second: regret = |5-4|=1, cum_cost=10 => 1/10
        assert regrets[1] == pytest.approx(1.0 / 10.0)

    def test_cost_adjusted_regret_no_optimal(self):
        ct = CostTracker()
        ct.record_trial(_make_trial())
        assert ct.cost_adjusted_regret([1.0], optimal=None) == []

    def test_cost_adjusted_regret_no_history(self):
        ct = CostTracker()
        assert ct.cost_adjusted_regret([], optimal=5.0) == []

    def test_gittins_stopping_index_positive(self):
        ct = CostTracker()
        # posterior_mean > current_best, so expected improvement is high
        idx = ct.gittins_stopping_index(
            current_best=0.0,
            posterior_mean=5.0,
            posterior_std=1.0,
            expected_cost=1.0,
        )
        assert idx > 0.0, "Index should be positive (continue exploring)"

    def test_gittins_stopping_index_zero_std(self):
        ct = CostTracker()
        idx = ct.gittins_stopping_index(
            current_best=5.0,
            posterior_mean=5.0,
            posterior_std=0.0,
            expected_cost=1.0,
        )
        assert idx == pytest.approx(0.0)

    def test_gittins_stopping_index_zero_cost(self):
        ct = CostTracker()
        idx = ct.gittins_stopping_index(
            current_best=5.0,
            posterior_mean=6.0,
            posterior_std=1.0,
            expected_cost=0.0,
        )
        assert idx == pytest.approx(0.0)

    def test_cumulative_cost_series(self):
        ct = CostTracker()
        ct.record_trial(_make_trial(trial_id="a", wall=2, resource=0, compute=0, opportunity=0))
        ct.record_trial(_make_trial(trial_id="b", wall=3, resource=0, compute=0, opportunity=0))
        ct.record_trial(_make_trial(trial_id="c", wall=5, resource=0, compute=0, opportunity=0))
        series = ct.cumulative_cost_series()
        assert series == pytest.approx([2.0, 5.0, 10.0])

    def test_cumulative_cost_series_empty(self):
        ct = CostTracker()
        assert ct.cumulative_cost_series() == []

    def test_to_dict_from_dict_roundtrip(self):
        ct = CostTracker(budget=200.0)
        ct.record_trial(_make_trial(trial_id="a"))
        ct.record_trial(_make_trial(trial_id="b", fidelity=1))
        d = ct.to_dict()
        restored = CostTracker.from_dict(d)
        assert restored.budget == pytest.approx(200.0)
        assert restored.n_trials == 2
        assert restored.total_spent == pytest.approx(ct.total_spent)

    def test_custom_cost_field(self):
        ct = CostTracker(cost_field="compute_cost")
        ct.record_trial(_make_trial(trial_id="a", compute=7.0))
        assert ct.total_spent == pytest.approx(7.0)

    def test_average_cost_per_trial(self):
        ct = CostTracker()
        ct.record_trial(_make_trial(trial_id="a", wall=2, resource=0, compute=0, opportunity=0))
        ct.record_trial(_make_trial(trial_id="b", wall=8, resource=0, compute=0, opportunity=0))
        assert ct.average_cost_per_trial == pytest.approx(5.0)


# ===================================================================
# 2. stopping_rule.py
# ===================================================================


class TestStoppingDecision:
    """Tests for StoppingDecision dataclass."""

    def test_to_dict(self):
        sd = StoppingDecision(
            should_stop=True,
            reason="max trials",
            criterion="max_trials",
            details={"n_trials": 100},
        )
        d = sd.to_dict()
        assert d["should_stop"] is True
        assert d["criterion"] == "max_trials"
        assert d["details"]["n_trials"] == 100


class TestStoppingRule:
    """Tests for StoppingRule."""

    def test_max_trials_triggers(self):
        rule = StoppingRule(max_trials=10)
        decision = rule.should_stop(n_trials=10)
        assert decision.should_stop is True
        assert decision.criterion == "max_trials"

    def test_max_trials_below_limit(self):
        rule = StoppingRule(max_trials=10)
        decision = rule.should_stop(n_trials=5)
        assert decision.should_stop is False

    def test_budget_exhausted_triggers(self):
        rule = StoppingRule(max_cost=50.0)
        decision = rule.should_stop(total_cost=55.0)
        assert decision.should_stop is True
        assert decision.criterion == "budget_exhausted"

    def test_stagnation_triggers(self):
        rule = StoppingRule(improvement_patience=5, improvement_threshold=1e-4)
        # 5 identical values => improvement = 0 < 1e-4
        flat_values = [3.0] * 5
        decision = rule.should_stop(best_values=flat_values)
        assert decision.should_stop is True
        assert decision.criterion == "stagnation"

    def test_stagnation_not_triggered_with_improvement(self):
        rule = StoppingRule(improvement_patience=5, improvement_threshold=1e-4)
        values = [1.0, 1.5, 2.0, 2.5, 3.0]
        decision = rule.should_stop(best_values=values)
        assert decision.should_stop is False

    def test_stagnation_not_triggered_insufficient_data(self):
        rule = StoppingRule(improvement_patience=5)
        decision = rule.should_stop(best_values=[1.0, 1.0, 1.0])
        assert decision.should_stop is False

    def test_gittins_stop_triggers(self):
        rule = StoppingRule()
        decision = rule.should_stop(gittins_index=0.0)
        assert decision.should_stop is True
        assert decision.criterion == "gittins"

    def test_gittins_negative_triggers(self):
        rule = StoppingRule()
        decision = rule.should_stop(gittins_index=-0.5)
        assert decision.should_stop is True
        assert decision.criterion == "gittins"

    def test_gittins_positive_no_trigger(self):
        rule = StoppingRule()
        decision = rule.should_stop(gittins_index=1.0)
        assert decision.should_stop is False

    def test_convergence_triggers(self):
        rule = StoppingRule(min_uncertainty=0.01)
        decision = rule.should_stop(current_uncertainty=0.005)
        assert decision.should_stop is True
        assert decision.criterion == "convergence"

    def test_convergence_no_trigger_above_threshold(self):
        rule = StoppingRule(min_uncertainty=0.01)
        decision = rule.should_stop(current_uncertainty=0.1)
        assert decision.should_stop is False

    def test_pareto_stable_triggers(self):
        rule = StoppingRule(pareto_stability_window=3)
        # Same front size for 3 iterations
        decision = rule.should_stop(pareto_sizes=[5, 5, 5])
        assert decision.should_stop is True
        assert decision.criterion == "pareto_stable"

    def test_pareto_stable_not_triggered_varying_sizes(self):
        rule = StoppingRule(pareto_stability_window=3)
        decision = rule.should_stop(pareto_sizes=[3, 4, 5])
        assert decision.should_stop is False

    def test_no_criteria_met_returns_continue(self):
        rule = StoppingRule(max_trials=100, max_cost=1000.0, min_uncertainty=0.001)
        decision = rule.should_stop(
            n_trials=5,
            total_cost=10.0,
            best_values=[1.0, 2.0, 3.0],
            current_uncertainty=0.5,
            gittins_index=1.0,
        )
        assert decision.should_stop is False
        assert decision.criterion == "none"

    def test_first_criterion_wins_priority(self):
        rule = StoppingRule(max_trials=5, max_cost=10.0)
        # Both max_trials AND budget met; max_trials is checked first
        decision = rule.should_stop(n_trials=5, total_cost=20.0)
        assert decision.should_stop is True
        assert decision.criterion == "max_trials"

    def test_active_criteria(self):
        rule = StoppingRule(max_trials=10, max_cost=50.0, min_uncertainty=0.01)
        criteria = rule.active_criteria()
        assert "max_trials" in criteria
        assert "budget" in criteria
        assert "convergence" in criteria
        assert "stagnation" in criteria
        assert "gittins" in criteria
        assert "pareto_stable" in criteria

    def test_active_criteria_minimal_config(self):
        rule = StoppingRule()
        criteria = rule.active_criteria()
        assert "max_trials" not in criteria
        assert "budget" not in criteria
        assert "convergence" not in criteria
        # Always present
        assert "stagnation" in criteria
        assert "gittins" in criteria
        assert "pareto_stable" in criteria

    def test_to_dict_from_dict_roundtrip(self):
        rule = StoppingRule(
            max_trials=50,
            max_cost=200.0,
            improvement_patience=8,
            improvement_threshold=0.001,
            min_uncertainty=0.01,
            pareto_stability_window=4,
        )
        d = rule.to_dict()
        restored = StoppingRule.from_dict(d)
        d2 = restored.to_dict()
        assert d == d2


# ===================================================================
# 3. auto_sampler.py
# ===================================================================


class TestSelectionResult:
    """Tests for SelectionResult dataclass."""

    def test_to_dict(self):
        sr = SelectionResult(
            backend_name="gp_bo",
            score=3.5,
            scores_breakdown={"phase": 2.0, "data": 1.5},
            reason="test",
            alternatives=["rf_bo"],
        )
        d = sr.to_dict()
        assert d["backend_name"] == "gp_bo"
        assert d["score"] == pytest.approx(3.5)
        assert d["alternatives"] == ["rf_bo"]


class TestAutoSampler:
    """Tests for AutoSampler."""

    def test_cold_start_prefers_space_filling(self):
        sampler = AutoSampler()
        result = sampler.select(phase="cold_start", n_observations=0)
        # sobol or lhs should win in cold_start
        assert result.backend_name in ("sobol_sampler", "latin_hypercube_sampler")

    def test_learning_phase_prefers_model_based(self):
        sampler = AutoSampler()
        result = sampler.select(phase="learning", n_observations=25)
        assert result.backend_name in ("gaussian_process_bo", "random_forest_bo", "tpe_sampler")

    def test_exploitation_phase_prefers_local_search(self):
        sampler = AutoSampler()
        result = sampler.select(phase="exploitation", n_observations=30)
        assert result.backend_name in ("cmaes_sampler", "gaussian_process_bo", "turbo_sampler")

    def test_multi_objective_selects_compatible(self):
        sampler = AutoSampler()
        result = sampler.select(phase="learning", n_observations=25, is_multi_objective=True)
        # Must be in MULTI_OBJ_BACKENDS or the other compatible ones
        from optimization_copilot.infrastructure.auto_sampler import MULTI_OBJ_BACKENDS
        compatible = MULTI_OBJ_BACKENDS | {"tpe_sampler", "random_sampler", "sobol_sampler",
                                            "latin_hypercube_sampler", "differential_evolution"}
        assert result.backend_name in compatible

    def test_high_noise_prefers_robust(self):
        sampler = AutoSampler()
        result = sampler.select(
            phase="learning",
            n_observations=25,
            noise_level="high",
        )
        # random_forest_bo has highest noise robustness
        assert result.backend_name in ("random_forest_bo", "gaussian_process_bo", "tpe_sampler")

    def test_tight_budget_prefers_efficient(self):
        sampler = AutoSampler()
        result = sampler.select(
            phase="learning",
            n_observations=10,
            budget_remaining=3.0,
        )
        # Efficiency score should be factored in
        assert result.score > 0

    def test_selection_history_tracks(self):
        sampler = AutoSampler()
        assert len(sampler.selection_history) == 0
        sampler.select(phase="cold_start")
        sampler.select(phase="learning", n_observations=10)
        assert len(sampler.selection_history) == 2

    def test_reset_history(self):
        sampler = AutoSampler()
        sampler.select(phase="cold_start")
        sampler.reset_history()
        assert len(sampler.selection_history) == 0

    def test_available_backends_property(self):
        backends = ["sobol_sampler", "tpe_sampler"]
        sampler = AutoSampler(available_backends=backends)
        assert sampler.available_backends == backends

    def test_to_dict_from_dict_roundtrip(self):
        sampler = AutoSampler(
            available_backends=["sobol_sampler", "tpe_sampler"],
            weights={"phase": 5.0, "data": 1.0, "noise": 1.0,
                     "efficiency": 1.0, "portfolio": 1.0, "capability": 1.0},
        )
        d = sampler.to_dict()
        restored = AutoSampler.from_dict(d)
        assert restored.available_backends == sampler.available_backends

    def test_custom_weights_override(self):
        # Heavily weight efficiency: tight budget should dominate
        sampler = AutoSampler(weights={
            "phase": 0.0, "data": 0.0, "noise": 0.0,
            "efficiency": 100.0, "portfolio": 0.0, "capability": 0.0,
        })
        result = sampler.select(phase="learning", n_observations=10, budget_remaining=3.0)
        # With 100x weight on efficiency and tight budget, random_sampler (eff=1.0) should win
        assert result.backend_name == "random_sampler"

    def test_portfolio_scores_influence(self):
        sampler = AutoSampler(
            available_backends=["tpe_sampler", "random_sampler"],
            weights={
                "phase": 0.0, "data": 0.0, "noise": 0.0,
                "efficiency": 0.0, "portfolio": 100.0, "capability": 0.0,
            },
        )
        result = sampler.select(
            phase="learning",
            n_observations=10,
            portfolio_scores={"random_sampler": 10.0, "tpe_sampler": 0.1},
        )
        assert result.backend_name == "random_sampler"

    def test_fallback_when_no_compatible(self):
        # Only provide a backend that's incompatible with multi-objective
        sampler = AutoSampler(available_backends=["cmaes_sampler"])
        result = sampler.select(phase="learning", n_observations=10, is_multi_objective=True)
        # cmaes_sampler is not compatible; fallback to tpe_sampler
        assert result.backend_name == "tpe_sampler"
        assert result.score == pytest.approx(0.0)

    def test_high_dimensions_bonus(self):
        sampler = AutoSampler()
        result = sampler.select(
            phase="learning",
            n_observations=25,
            n_dimensions=15,
        )
        # High-dim backends (turbo, rf_bo, de) get a capability bonus
        assert result.score > 0


# ===================================================================
# 4. constraint_engine.py
# ===================================================================


class TestConstraintTypes:
    """Tests for ConstraintType and ConstraintStatus enums."""

    def test_constraint_type_values(self):
        assert ConstraintType.KNOWN_HARD.value == "known_hard"
        assert ConstraintType.KNOWN_SOFT.value == "known_soft"
        assert ConstraintType.UNKNOWN.value == "unknown"

    def test_constraint_status_values(self):
        assert ConstraintStatus.FEASIBLE.value == "feasible"
        assert ConstraintStatus.VIOLATED.value == "violated"
        assert ConstraintStatus.UNKNOWN.value == "unknown"


class TestConstraintDataclasses:
    """Tests for Constraint and ConstraintEvaluation dataclasses."""

    def test_constraint_to_dict(self):
        c = Constraint(
            name="temp_limit",
            constraint_type=ConstraintType.KNOWN_HARD,
            tolerance=0.1,
            safety_probability=0.99,
        )
        d = c.to_dict()
        assert d["name"] == "temp_limit"
        assert d["constraint_type"] == "known_hard"
        assert d["tolerance"] == pytest.approx(0.1)

    def test_constraint_evaluation_to_dict(self):
        ev = ConstraintEvaluation(
            candidate={"x": 1.0},
            is_feasible=True,
            constraint_results={"c1": ConstraintStatus.FEASIBLE},
            feasibility_probabilities={"c1": 1.0},
            overall_feasibility_probability=1.0,
        )
        d = ev.to_dict()
        assert d["is_feasible"] is True
        assert d["constraint_results"]["c1"] == "feasible"


class TestConstraintEngine:
    """Tests for ConstraintEngine."""

    def _hard_constraint(self, name: str = "x_positive") -> Constraint:
        return Constraint(
            name=name,
            constraint_type=ConstraintType.KNOWN_HARD,
            evaluate=lambda x: x.get("x", 0) > 0,
        )

    def _soft_constraint(self, name: str = "x_small", tolerance: float = 0.5) -> Constraint:
        return Constraint(
            name=name,
            constraint_type=ConstraintType.KNOWN_SOFT,
            evaluate=lambda x: x.get("x", 0) < 10,
            tolerance=tolerance,
        )

    def _unknown_constraint(self, name: str = "unknown_c") -> Constraint:
        return Constraint(
            name=name,
            constraint_type=ConstraintType.UNKNOWN,
        )

    def test_filter_candidates_hard_constraint(self):
        engine = ConstraintEngine(constraints=[self._hard_constraint()])
        candidates = [{"x": -1}, {"x": 5}, {"x": 0}, {"x": 3}]
        filtered = engine.filter_candidates(candidates)
        # x > 0 keeps {"x": 5} and {"x": 3}
        assert len(filtered) == 2
        assert all(c["x"] > 0 for c in filtered)

    def test_filter_candidates_no_constraints(self):
        engine = ConstraintEngine()
        candidates = [{"x": -1}, {"x": 5}]
        filtered = engine.filter_candidates(candidates)
        assert len(filtered) == 2

    def test_filter_candidates_soft_constraint_passes_all(self):
        """Soft constraints do not filter, only hard ones do."""
        engine = ConstraintEngine(constraints=[self._soft_constraint()])
        candidates = [{"x": 100}, {"x": 5}]
        filtered = engine.filter_candidates(candidates)
        # Soft constraints don't filter
        assert len(filtered) == 2

    def test_constraint_weighted_acquisition_soft(self):
        engine = ConstraintEngine(constraints=[self._soft_constraint("soft", tolerance=0.5)])
        acq = [10.0, 10.0]
        candidates = [{"x": 5}, {"x": 100}]  # first passes, second violates
        weighted = engine.constraint_weighted_acquisition(acq, candidates)
        # First candidate: passes soft constraint => acq unchanged
        assert weighted[0] == pytest.approx(10.0)
        # Second: violates => penalty = max(0.01, 1.0 - 0.5) = 0.5
        assert weighted[1] == pytest.approx(10.0 * 0.5)

    def test_constraint_weighted_acquisition_unknown_no_model(self):
        engine = ConstraintEngine(constraints=[self._unknown_constraint()])
        acq = [10.0]
        candidates = [{"x": 5}]
        weighted = engine.constraint_weighted_acquisition(acq, candidates)
        # No model => p_feasible = 0.5
        assert weighted[0] == pytest.approx(10.0 * 0.5)

    def test_evaluate_candidate_known_hard_feasible(self):
        engine = ConstraintEngine(constraints=[self._hard_constraint()])
        ev = engine.evaluate_candidate({"x": 5})
        assert ev.is_feasible is True
        assert ev.constraint_results["x_positive"] == ConstraintStatus.FEASIBLE

    def test_evaluate_candidate_known_hard_violated(self):
        engine = ConstraintEngine(constraints=[self._hard_constraint()])
        ev = engine.evaluate_candidate({"x": -1})
        assert ev.is_feasible is False
        assert ev.constraint_results["x_positive"] == ConstraintStatus.VIOLATED

    def test_evaluate_candidate_unknown_no_model(self):
        engine = ConstraintEngine(constraints=[self._unknown_constraint()])
        ev = engine.evaluate_candidate({"x": 5.0})
        # No model => p = 0.5, below safety_probability 0.95
        # and above 1 - 0.95 = 0.05, so status is UNKNOWN
        assert ev.constraint_results["unknown_c"] == ConstraintStatus.UNKNOWN
        assert ev.feasibility_probabilities["unknown_c"] == pytest.approx(0.5)

    def test_update_unknown_constraints_and_predict(self):
        """After enough observations, GP model should learn simple boundary."""
        engine = ConstraintEngine(constraints=[self._unknown_constraint("boundary")])

        # Add observations: x < 5 feasible, x >= 5 infeasible
        for i in range(10):
            x_val = float(i)
            feasible = x_val < 5
            engine.update_unknown_constraints(
                [x_val],
                {"boundary": feasible},
            )

        # Test prediction: low x should have higher feasibility
        ev_low = engine.evaluate_candidate({"x": 1.0})
        ev_high = engine.evaluate_candidate({"x": 9.0})
        p_low = ev_low.feasibility_probabilities["boundary"]
        p_high = ev_high.feasibility_probabilities["boundary"]
        assert p_low > p_high, "Low x should be more likely feasible"

    def test_suggest_constraint_exploration_selects_uncertain(self):
        engine = ConstraintEngine(constraints=[self._unknown_constraint("uc")])
        # Train model with some data
        for i in range(10):
            engine.update_unknown_constraints([float(i)], {"uc": i < 5})

        candidates = [{"x": 0.0}, {"x": 5.0}, {"x": 9.0}]
        selected = engine.suggest_constraint_exploration(candidates, budget=1)
        assert len(selected) == 1
        # The point near the boundary (x=5) should have highest entropy
        assert selected[0]["x"] == pytest.approx(5.0)

    def test_suggest_constraint_exploration_no_unknown(self):
        engine = ConstraintEngine(constraints=[self._hard_constraint()])
        candidates = [{"x": 1}, {"x": 2}]
        selected = engine.suggest_constraint_exploration(candidates, budget=1)
        # No unknown constraints => returns first budget candidates
        assert len(selected) == 1

    def test_feasibility_summary(self):
        engine = ConstraintEngine(constraints=[
            self._hard_constraint(),
            self._soft_constraint(),
            self._unknown_constraint(),
        ])
        summary = engine.feasibility_summary()
        assert summary["n_constraints"] == 3
        assert summary["n_hard"] == 1
        assert summary["n_soft"] == 1
        assert summary["n_unknown"] == 1
        assert "x_positive" in summary["constraints"]

    def test_feasibility_rate(self):
        engine = ConstraintEngine(constraints=[self._hard_constraint()])
        engine.evaluate_candidate({"x": 5})
        engine.evaluate_candidate({"x": -1})
        engine.evaluate_candidate({"x": 3})
        rate = engine.feasibility_rate()
        assert rate == pytest.approx(2.0 / 3.0)

    def test_feasibility_rate_no_evaluations(self):
        engine = ConstraintEngine()
        assert engine.feasibility_rate() is None

    def test_add_constraint(self):
        engine = ConstraintEngine()
        assert engine.n_constraints == 0
        engine.add_constraint(self._hard_constraint())
        assert engine.n_constraints == 1

    def test_remove_constraint(self):
        engine = ConstraintEngine(constraints=[self._hard_constraint("c1")])
        assert engine.n_constraints == 1
        removed = engine.remove_constraint("c1")
        assert removed is True
        assert engine.n_constraints == 0

    def test_remove_nonexistent_constraint(self):
        engine = ConstraintEngine()
        removed = engine.remove_constraint("nope")
        assert removed is False

    def test_has_unknown_constraints_property(self):
        engine = ConstraintEngine(constraints=[self._unknown_constraint()])
        assert engine.has_unknown_constraints is True

    def test_has_hard_constraints_property(self):
        engine = ConstraintEngine(constraints=[self._hard_constraint()])
        assert engine.has_hard_constraints is True

    def test_no_unknown_constraints_property(self):
        engine = ConstraintEngine(constraints=[self._hard_constraint()])
        assert engine.has_unknown_constraints is False

    def test_no_hard_constraints_property(self):
        engine = ConstraintEngine(constraints=[self._soft_constraint()])
        assert engine.has_hard_constraints is False

    def test_from_constraints_deserializes(self):
        defs = [
            {"name": "c1", "constraint_type": "known_hard", "tolerance": 0.0},
            {"name": "c2", "constraint_type": "unknown", "safety_probability": 0.9},
        ]
        engine = ConstraintEngine.from_constraints(defs)
        assert engine.n_constraints == 2
        assert engine.constraints[0].name == "c1"
        assert engine.constraints[0].constraint_type == ConstraintType.KNOWN_HARD
        assert engine.constraints[1].constraint_type == ConstraintType.UNKNOWN

    def test_to_dict(self):
        engine = ConstraintEngine(constraints=[self._hard_constraint()])
        d = engine.to_dict()
        assert "constraints" in d
        assert "feasibility_summary" in d
        assert len(d["constraints"]) == 1

"""Tests for cost analyzer and batch diversifier modules."""

import math

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.cost.cost_analyzer import CostAnalyzer, CostSignals
from optimization_copilot.batch.diversifier import (
    AdaptiveBatchSizer,
    BatchDiversifier,
    BatchFailureReplanner,
    BatchPolicy,
)
from optimization_copilot.core.models import Phase


# ── helpers ───────────────────────────────────────────────


def _make_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
    ]


def _make_snapshot_with_timestamps(
    n_obs: int = 10,
    timestamp_gap: float = 1.0,
) -> CampaignSnapshot:
    """Create a snapshot where cost is inferred from timestamp gaps."""
    specs = _make_specs()
    obs = []
    for i in range(n_obs):
        obs.append(Observation(
            iteration=i,
            parameters={"x1": i / max(n_obs - 1, 1), "x2": 0.5},
            kpi_values={"y": float(i)},  # monotonically improving
            timestamp=i * timestamp_gap,
        ))
    return CampaignSnapshot(
        campaign_id="ts-test",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n_obs,
    )


def _make_snapshot_with_metadata_cost(
    n_obs: int = 10,
    cost_per_obs: float = 5.0,
) -> CampaignSnapshot:
    """Create a snapshot with explicit cost in metadata."""
    specs = _make_specs()
    obs = []
    for i in range(n_obs):
        obs.append(Observation(
            iteration=i,
            parameters={"x1": i / max(n_obs - 1, 1), "x2": 0.5},
            kpi_values={"y": float(i)},
            timestamp=float(i),
            metadata={"cost": cost_per_obs},
        ))
    return CampaignSnapshot(
        campaign_id="meta-cost-test",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n_obs,
    )


def _make_empty_snapshot() -> CampaignSnapshot:
    return CampaignSnapshot(
        campaign_id="empty",
        parameter_specs=_make_specs(),
        observations=[],
        objective_names=["y"],
        objective_directions=["maximize"],
    )


# ── CostAnalyzer tests ───────────────────────────────────


class TestCostAnalyzerTimestampCosts:
    """Test cost extraction from timestamp gaps."""

    def test_basic_signals(self):
        snap = _make_snapshot_with_timestamps(n_obs=10, timestamp_gap=2.0)
        analyzer = CostAnalyzer(total_budget=100.0)
        signals = analyzer.analyze(snap)

        # First observation cost = its timestamp = 0.0
        # Subsequent costs = gap = 2.0 each (9 gaps)
        # cumulative = 0 + 2*9 = 18.0
        assert signals.cumulative_cost == 18.0
        assert signals.time_budget_pressure == 18.0 / 100.0
        assert signals.estimated_remaining_budget == 82.0

    def test_cost_per_improvement(self):
        snap = _make_snapshot_with_timestamps(n_obs=10, timestamp_gap=1.0)
        analyzer = CostAnalyzer()
        signals = analyzer.analyze(snap)

        # KPI goes from 0 to 9, improvement = 9
        # Cumulative cost from timestamps: 0 + 1*9 = 9.0
        assert signals.cost_per_improvement == 9.0 / 9.0  # 1.0

    def test_no_improvement(self):
        """When KPI is flat, cost_per_improvement should be inf."""
        specs = _make_specs()
        obs = [
            Observation(
                iteration=i,
                parameters={"x1": 0.5, "x2": 0.5},
                kpi_values={"y": 1.0},  # no improvement
                timestamp=float(i),
            )
            for i in range(5)
        ]
        snap = CampaignSnapshot(
            campaign_id="flat",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=5,
        )
        analyzer = CostAnalyzer()
        signals = analyzer.analyze(snap)
        assert signals.cost_per_improvement == math.inf


class TestCostAnalyzerMetadataCosts:
    """Test cost extraction from observation metadata."""

    def test_metadata_cost_extraction(self):
        snap = _make_snapshot_with_metadata_cost(n_obs=10, cost_per_obs=5.0)
        analyzer = CostAnalyzer(total_budget=200.0, cost_field="cost")
        signals = analyzer.analyze(snap)

        # 10 observations * 5.0 each = 50.0
        assert signals.cumulative_cost == 50.0
        assert signals.time_budget_pressure == 50.0 / 200.0
        assert signals.estimated_remaining_budget == 150.0

    def test_custom_cost_field(self):
        specs = _make_specs()
        obs = [
            Observation(
                iteration=i,
                parameters={"x1": 0.5, "x2": 0.5},
                kpi_values={"y": float(i)},
                timestamp=float(i),
                metadata={"dollars": 10.0},
            )
            for i in range(5)
        ]
        snap = CampaignSnapshot(
            campaign_id="custom-field",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=5,
        )
        analyzer = CostAnalyzer(total_budget=100.0, cost_field="dollars")
        signals = analyzer.analyze(snap)
        assert signals.cumulative_cost == 50.0  # 5 * 10.0


class TestBudgetPressure:
    """Test budget pressure calculation."""

    def test_no_budget(self):
        snap = _make_snapshot_with_timestamps(n_obs=5)
        analyzer = CostAnalyzer(total_budget=None)
        signals = analyzer.analyze(snap)
        assert signals.time_budget_pressure == 0.0
        assert signals.estimated_remaining_budget == math.inf

    def test_zero_budget(self):
        snap = _make_snapshot_with_timestamps(n_obs=5)
        analyzer = CostAnalyzer(total_budget=0.0)
        signals = analyzer.analyze(snap)
        # Division by zero protected
        assert signals.time_budget_pressure == 0.0

    def test_half_spent(self):
        snap = _make_snapshot_with_metadata_cost(n_obs=5, cost_per_obs=10.0)
        analyzer = CostAnalyzer(total_budget=100.0)
        signals = analyzer.analyze(snap)
        assert signals.time_budget_pressure == 0.5
        assert signals.estimated_remaining_budget == 50.0

    def test_over_budget(self):
        snap = _make_snapshot_with_metadata_cost(n_obs=10, cost_per_obs=15.0)
        analyzer = CostAnalyzer(total_budget=100.0)
        signals = analyzer.analyze(snap)
        # cumulative = 150, budget = 100 -> pressure capped at 1.0
        assert signals.time_budget_pressure == 1.0
        assert signals.estimated_remaining_budget == 0.0

    def test_empty_snapshot(self):
        snap = _make_empty_snapshot()
        analyzer = CostAnalyzer(total_budget=100.0)
        signals = analyzer.analyze(snap)
        assert signals.cumulative_cost == 0.0
        assert signals.time_budget_pressure == 0.0
        assert signals.estimated_remaining_budget == 100.0
        assert signals.cost_optimal_batch_size == 1


class TestExplorationAdjustment:
    """Test exploration strength adjustment based on cost signals."""

    def test_no_pressure_no_change(self):
        analyzer = CostAnalyzer()
        signals = CostSignals(
            cost_per_improvement=1.0,
            time_budget_pressure=0.0,
            cost_efficiency_trend=0.0,
            cumulative_cost=10.0,
            estimated_remaining_budget=math.inf,
            cost_optimal_batch_size=4,
        )
        adjusted = analyzer.adjust_exploration(0.5, signals)
        # No pressure, no trend -> minimal change
        assert abs(adjusted - 0.5) < 0.01

    def test_high_pressure_reduces_exploration(self):
        analyzer = CostAnalyzer()
        signals = CostSignals(
            cost_per_improvement=1.0,
            time_budget_pressure=0.9,
            cost_efficiency_trend=0.0,
            cumulative_cost=90.0,
            estimated_remaining_budget=10.0,
            cost_optimal_batch_size=1,
        )
        adjusted = analyzer.adjust_exploration(0.8, signals)
        # Should be significantly reduced
        assert adjusted < 0.5

    def test_full_pressure_reduces_to_near_zero(self):
        analyzer = CostAnalyzer()
        signals = CostSignals(
            cost_per_improvement=10.0,
            time_budget_pressure=1.0,
            cost_efficiency_trend=-0.5,
            cumulative_cost=100.0,
            estimated_remaining_budget=0.0,
            cost_optimal_batch_size=1,
        )
        adjusted = analyzer.adjust_exploration(0.5, signals)
        assert adjusted < 0.1

    def test_improving_trend_allows_more_exploration(self):
        analyzer = CostAnalyzer()
        signals_improving = CostSignals(
            cost_per_improvement=1.0,
            time_budget_pressure=0.3,
            cost_efficiency_trend=0.8,
            cumulative_cost=30.0,
            estimated_remaining_budget=math.inf,
            cost_optimal_batch_size=4,
        )
        signals_worsening = CostSignals(
            cost_per_improvement=1.0,
            time_budget_pressure=0.3,
            cost_efficiency_trend=-0.8,
            cumulative_cost=30.0,
            estimated_remaining_budget=math.inf,
            cost_optimal_batch_size=4,
        )
        adj_improving = analyzer.adjust_exploration(0.5, signals_improving)
        adj_worsening = analyzer.adjust_exploration(0.5, signals_worsening)
        assert adj_improving > adj_worsening

    def test_clamped_to_zero(self):
        analyzer = CostAnalyzer()
        signals = CostSignals(
            cost_per_improvement=10.0,
            time_budget_pressure=1.0,
            cost_efficiency_trend=-1.0,
            cumulative_cost=100.0,
            estimated_remaining_budget=0.0,
            cost_optimal_batch_size=1,
        )
        adjusted = analyzer.adjust_exploration(0.1, signals)
        assert adjusted >= 0.0

    def test_clamped_to_one(self):
        analyzer = CostAnalyzer()
        signals = CostSignals(
            cost_per_improvement=0.01,
            time_budget_pressure=0.0,
            cost_efficiency_trend=1.0,
            cumulative_cost=1.0,
            estimated_remaining_budget=999.0,
            cost_optimal_batch_size=8,
        )
        adjusted = analyzer.adjust_exploration(0.95, signals)
        assert adjusted <= 1.0


# ── BatchDiversifier tests ───────────────────────────────


def _make_candidates(n: int = 20, seed: int = 42) -> list[dict]:
    """Generate random candidates in [0, 1]^2."""
    import random as _rng
    r = _rng.Random(seed)
    return [{"x1": r.random(), "x2": r.random()} for _ in range(n)]


def _make_clustered_candidates(n: int = 20, seed: int = 42) -> list[dict]:
    """Generate candidates clustered in two corners."""
    import random as _rng
    r = _rng.Random(seed)
    candidates = []
    for i in range(n):
        if i < n // 2:
            # Cluster near (0, 0)
            candidates.append({"x1": r.random() * 0.1, "x2": r.random() * 0.1})
        else:
            # Cluster near (1, 1)
            candidates.append({"x1": 0.9 + r.random() * 0.1, "x2": 0.9 + r.random() * 0.1})
    return candidates


class TestMaximinStrategy:
    """Test the maximin diversification strategy."""

    def test_selects_correct_count(self):
        specs = _make_specs()
        candidates = _make_candidates(20)
        div = BatchDiversifier(strategy="maximin")
        result = div.diversify(candidates, specs, n_select=5)
        assert len(result.points) == 5
        assert result.strategy == "maximin"

    def test_selects_diverse_from_clusters(self):
        specs = _make_specs()
        candidates = _make_clustered_candidates(20)
        div = BatchDiversifier(strategy="maximin")
        result = div.diversify(candidates, specs, n_select=4)

        # Should pick from both clusters
        x1_values = [p["x1"] for p in result.points]
        has_low = any(v < 0.2 for v in x1_values)
        has_high = any(v > 0.8 for v in x1_values)
        assert has_low and has_high, (
            f"Expected points from both clusters, got x1={x1_values}"
        )

    def test_diversity_score_positive(self):
        specs = _make_specs()
        candidates = _make_candidates(20)
        div = BatchDiversifier(strategy="maximin")
        result = div.diversify(candidates, specs, n_select=5)
        assert result.diversity_score > 0.0

    def test_reproducible_with_same_seed(self):
        specs = _make_specs()
        candidates = _make_candidates(20)
        div = BatchDiversifier(strategy="maximin")
        r1 = div.diversify(candidates, specs, n_select=5, seed=123)
        r2 = div.diversify(candidates, specs, n_select=5, seed=123)
        assert r1.points == r2.points


class TestCoverageStrategy:
    """Test the coverage diversification strategy."""

    def test_selects_correct_count(self):
        specs = _make_specs()
        candidates = _make_candidates(20)
        div = BatchDiversifier(strategy="coverage")
        result = div.diversify(candidates, specs, n_select=5)
        assert len(result.points) == 5
        assert result.strategy == "coverage"

    def test_avoids_existing_observations(self):
        specs = _make_specs()
        # Existing observations near (0, 0)
        existing = [
            Observation(
                iteration=0,
                parameters={"x1": 0.05, "x2": 0.05},
                kpi_values={"y": 1.0},
            ),
        ]
        # Candidates spread across space
        candidates = [
            {"x1": 0.05, "x2": 0.05},  # Same bin as existing
            {"x1": 0.5, "x2": 0.5},    # New region
            {"x1": 0.95, "x2": 0.95},  # New region
        ]
        div = BatchDiversifier(strategy="coverage")
        result = div.diversify(
            candidates, specs, n_select=2, existing_obs=existing,
        )

        # Should prefer (0.5, 0.5) and (0.95, 0.95) over (0.05, 0.05)
        selected_x1 = [p["x1"] for p in result.points]
        assert 0.5 in selected_x1 or 0.95 in selected_x1

    def test_coverage_gain_computed(self):
        specs = _make_specs()
        candidates = _make_candidates(20)
        div = BatchDiversifier(strategy="coverage")
        result = div.diversify(candidates, specs, n_select=5)
        # With no existing observations, coverage gain should be positive
        assert result.coverage_gain >= 0.0


class TestDiversityScore:
    """Test that diversity_score increases with diverse points."""

    def test_identical_points_zero_diversity(self):
        specs = _make_specs()
        div = BatchDiversifier(strategy="maximin")
        identical = [{"x1": 0.5, "x2": 0.5}] * 10
        result = div.diversify(identical, specs, n_select=3)
        assert result.diversity_score == 0.0

    def test_spread_points_higher_diversity(self):
        specs = _make_specs()
        div = BatchDiversifier(strategy="maximin")

        # Clustered candidates
        clustered = [{"x1": 0.5 + i * 0.001, "x2": 0.5 + i * 0.001} for i in range(10)]
        result_clustered = div.diversify(clustered, specs, n_select=5)

        # Well-spread candidates
        spread = [
            {"x1": 0.0, "x2": 0.0},
            {"x1": 0.0, "x2": 1.0},
            {"x1": 1.0, "x2": 0.0},
            {"x1": 1.0, "x2": 1.0},
            {"x1": 0.5, "x2": 0.5},
        ]
        result_spread = div.diversify(spread, specs, n_select=5)

        assert result_spread.diversity_score > result_clustered.diversity_score

    def test_two_points_positive_diversity(self):
        specs = _make_specs()
        div = BatchDiversifier()
        candidates = [
            {"x1": 0.0, "x2": 0.0},
            {"x1": 1.0, "x2": 1.0},
        ]
        result = div.diversify(candidates, specs, n_select=2)
        assert result.diversity_score > 0.0


class TestEdgeCases:
    """Test edge cases for both modules."""

    def test_single_point_batch(self):
        specs = _make_specs()
        candidates = [{"x1": 0.5, "x2": 0.5}]
        div = BatchDiversifier(strategy="maximin")
        result = div.diversify(candidates, specs, n_select=1)
        assert len(result.points) == 1
        assert result.diversity_score == 0.0

    def test_no_candidates(self):
        specs = _make_specs()
        div = BatchDiversifier(strategy="maximin")
        result = div.diversify([], specs, n_select=5)
        assert len(result.points) == 0
        assert result.diversity_score == 0.0
        assert result.coverage_gain == 0.0

    def test_n_select_exceeds_candidates(self):
        specs = _make_specs()
        candidates = [{"x1": 0.1, "x2": 0.2}, {"x1": 0.8, "x2": 0.9}]
        div = BatchDiversifier(strategy="hybrid")
        result = div.diversify(candidates, specs, n_select=10)
        assert len(result.points) == 2  # Clamped to available

    def test_invalid_strategy_raises(self):
        try:
            BatchDiversifier(strategy="nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_categorical_parameter(self):
        specs = [
            ParameterSpec(
                name="color",
                type=VariableType.CATEGORICAL,
                categories=["red", "green", "blue"],
            ),
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ]
        candidates = [
            {"color": "red", "x1": 0.1},
            {"color": "green", "x1": 0.5},
            {"color": "blue", "x1": 0.9},
        ]
        div = BatchDiversifier(strategy="maximin")
        result = div.diversify(candidates, specs, n_select=3)
        assert len(result.points) == 3
        assert result.diversity_score > 0.0

    def test_cost_analyzer_empty_snapshot(self):
        snap = _make_empty_snapshot()
        analyzer = CostAnalyzer(total_budget=100.0)
        signals = analyzer.analyze(snap)
        assert signals.cumulative_cost == 0.0
        assert signals.cost_per_improvement == 0.0
        assert signals.cost_optimal_batch_size == 1

    def test_cost_analyzer_single_observation(self):
        specs = _make_specs()
        obs = [Observation(
            iteration=0,
            parameters={"x1": 0.5, "x2": 0.5},
            kpi_values={"y": 1.0},
            timestamp=5.0,
        )]
        snap = CampaignSnapshot(
            campaign_id="single",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=1,
        )
        analyzer = CostAnalyzer()
        signals = analyzer.analyze(snap)
        # Single observation: improvement = 0 (need at least 2 for improvement)
        assert signals.cumulative_cost == 5.0
        assert signals.cost_per_improvement == math.inf

    def test_hybrid_strategy_works(self):
        specs = _make_specs()
        candidates = _make_candidates(20)
        div = BatchDiversifier(strategy="hybrid")
        result = div.diversify(candidates, specs, n_select=5)
        assert len(result.points) == 5
        assert result.strategy == "hybrid"
        assert result.diversity_score > 0.0

    def test_minimize_direction(self):
        """Test cost analyzer with minimize objective direction."""
        specs = _make_specs()
        obs = [
            Observation(
                iteration=i,
                parameters={"x1": 0.5, "x2": 0.5},
                kpi_values={"y": 10.0 - float(i)},  # Improving (decreasing)
                timestamp=float(i),
            )
            for i in range(10)
        ]
        snap = CampaignSnapshot(
            campaign_id="minimize",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["minimize"],
            current_iteration=10,
        )
        analyzer = CostAnalyzer()
        signals = analyzer.analyze(snap)
        # KPI goes from 10 to 1, improvement = 10 - 1 = 9
        assert signals.cost_per_improvement > 0.0
        assert signals.cost_per_improvement != math.inf


# ── Batch Failure Replanning tests ─────────────────────


class TestBatchFailureReplanner:

    def test_no_failures_no_replacements(self):
        specs = _make_specs()
        batch = [{"x1": 0.1, "x2": 0.2}, {"x1": 0.5, "x2": 0.6}]
        results = [True, True]
        replanner = BatchFailureReplanner()
        result = replanner.replan(batch, results, specs)
        assert result.n_failed == 0
        assert result.n_replaced == 0
        assert result.replacement_points == []

    def test_all_failures_generates_replacements(self):
        specs = _make_specs()
        batch = [{"x1": 0.1, "x2": 0.2}, {"x1": 0.5, "x2": 0.6}]
        results = [False, False]
        replanner = BatchFailureReplanner()
        result = replanner.replan(batch, results, specs, seed=42)
        assert result.n_failed == 2
        assert result.n_replaced == 2
        assert len(result.replacement_points) == 2
        assert result.strategy == "random"

    def test_partial_failures(self):
        specs = _make_specs()
        batch = [
            {"x1": 0.1, "x2": 0.2},
            {"x1": 0.5, "x2": 0.6},
            {"x1": 0.9, "x2": 0.1},
        ]
        results = [True, False, True]
        replanner = BatchFailureReplanner()
        result = replanner.replan(batch, results, specs, seed=42)
        assert result.n_failed == 1
        assert result.n_replaced == 1
        assert result.strategy == "perturb_successful"

    def test_replacements_within_bounds(self):
        specs = _make_specs()
        batch = [{"x1": 0.5, "x2": 0.5}] * 3
        results = [True, False, False]
        replanner = BatchFailureReplanner()
        result = replanner.replan(batch, results, specs, seed=42)
        for pt in result.replacement_points:
            for spec in specs:
                lo = spec.lower or 0.0
                hi = spec.upper or 1.0
                assert lo <= pt[spec.name] <= hi

    def test_mismatched_sizes_raises(self):
        specs = _make_specs()
        replanner = BatchFailureReplanner()
        try:
            replanner.replan([{"x1": 0.5}], [True, False], specs)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ── Adaptive Batch Sizer tests ──────────────────────────


class TestAdaptiveBatchSizer:

    def test_cold_start_large_batch(self):
        sizer = AdaptiveBatchSizer()
        rec = sizer.compute_size(Phase.COLD_START, n_params=3)
        assert rec.batch_size >= 4  # 3 * 2 = 6

    def test_exploitation_small_batch(self):
        sizer = AdaptiveBatchSizer()
        rec = sizer.compute_size(Phase.EXPLOITATION, n_params=5)
        assert rec.batch_size <= 3

    def test_high_noise_increases_batch(self):
        sizer = AdaptiveBatchSizer()
        low_noise = sizer.compute_size(Phase.LEARNING, n_params=3, noise_estimate=0.1)
        high_noise = sizer.compute_size(Phase.LEARNING, n_params=3, noise_estimate=0.8)
        assert high_noise.batch_size >= low_noise.batch_size
        assert high_noise.noise_adjustment > 0

    def test_high_failure_rate_increases_batch(self):
        sizer = AdaptiveBatchSizer()
        low_fail = sizer.compute_size(Phase.LEARNING, n_params=3, failure_rate=0.05)
        high_fail = sizer.compute_size(Phase.LEARNING, n_params=3, failure_rate=0.5)
        assert high_fail.batch_size >= low_fail.batch_size
        assert high_fail.failure_adjustment > 0

    def test_capped_at_max(self):
        sizer = AdaptiveBatchSizer(max_batch=10)
        rec = sizer.compute_size(
            Phase.COLD_START, n_params=20, noise_estimate=0.9, failure_rate=0.5,
        )
        assert rec.batch_size <= 10

    def test_at_least_min(self):
        sizer = AdaptiveBatchSizer(min_batch=2)
        rec = sizer.compute_size(Phase.EXPLOITATION, n_params=1)
        assert rec.batch_size >= 2

    def test_stagnation_restarts_exploration(self):
        sizer = AdaptiveBatchSizer()
        rec = sizer.compute_size(Phase.STAGNATION, n_params=3)
        assert rec.batch_size >= 3
        assert "stagnation" in rec.reason

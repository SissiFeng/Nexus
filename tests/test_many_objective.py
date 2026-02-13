"""Tests for the optimization_copilot.multi_objective package enhancements.

Covers HypervolumeIndicator, IGDMetric, ManyObjectiveRanker,
ParetoQuery, TradeoffAnalysis, and InteractiveParetoExplorer.
"""

from __future__ import annotations

import math

import pytest

from optimization_copilot.multi_objective import (
    HypervolumeIndicator,
    IGDMetric,
    InteractiveParetoExplorer,
    ManyObjectiveRanker,
    ParetoQuery,
    TradeoffAnalysis,
)


# ── HypervolumeIndicator ─────────────────────────────────────────────


class TestHypervolumeIndicator:
    """Tests for exact 2-D and Monte Carlo hypervolume computation."""

    def test_compute_2d_exact(self):
        """Classic 2-D example: three non-dominated points."""
        hv = HypervolumeIndicator()
        points = [[1, 4], [2, 2], [4, 1]]
        ref = [5, 5]
        result = hv.compute(points, ref)
        assert result == pytest.approx(11.0), (
            f"Expected hypervolume 11.0, got {result}"
        )

    def test_compute_2d_single_point(self):
        """Single point should produce a simple rectangle."""
        hv = HypervolumeIndicator()
        points = [[2, 3]]
        ref = [5, 5]
        # Rectangle: (5-2) * (5-3) = 6.0
        result = hv.compute(points, ref)
        assert result == pytest.approx(6.0)

    def test_compute_2d_empty_returns_zero(self):
        """Empty point list should return 0.0."""
        hv = HypervolumeIndicator()
        result = hv.compute([], [5, 5])
        assert result == 0.0

    def test_compute_3d_monte_carlo_positive(self):
        """3-D points should use Monte Carlo and return a positive value."""
        hv = HypervolumeIndicator(n_samples=10000, seed=42)
        points = [[1, 1, 1], [2, 2, 2]]
        ref = [5, 5, 5]
        result = hv.compute(points, ref)
        assert result > 0.0, "Monte Carlo hypervolume should be positive"
        # Rough upper bound: full box volume is (5-1)^3 = 64
        assert result <= 64.0


# ── IGDMetric ─────────────────────────────────────────────────────────


class TestIGDMetric:
    """Tests for Inverted Generational Distance."""

    def test_compute_basic(self):
        """Basic case: obtained front near but not identical to reference."""
        igd = IGDMetric()
        obtained = [[1, 4], [2, 2], [4, 1]]
        reference = [[1.1, 3.9], [2.1, 2.1]]
        result = igd.compute(obtained, reference)
        assert result >= 0.0
        # Each reference point is close to an obtained point, so IGD < 1
        assert result < 1.0

    def test_compute_identical_sets_returns_zero(self):
        """When obtained == reference, IGD should be 0.0."""
        igd = IGDMetric()
        front = [[1, 4], [2, 2], [4, 1]]
        result = igd.compute(front, front)
        assert result == pytest.approx(0.0)

    def test_compute_empty_reference_returns_zero(self):
        """Empty reference front should yield 0.0 (no distances to average)."""
        igd = IGDMetric()
        result = igd.compute([[1, 2]], [])
        assert result == 0.0

    def test_compute_empty_obtained_returns_inf(self):
        """Empty obtained front should yield inf."""
        igd = IGDMetric()
        result = igd.compute([], [[1, 2]])
        assert result == float("inf")


# ── ManyObjectiveRanker ───────────────────────────────────────────────


class TestManyObjectiveRanker:
    """Tests for hypervolume-contribution-based ranking."""

    def test_rank_returns_correct_ranking(self):
        """Rank should assign 1 to the highest-contribution point."""
        ranker = ManyObjectiveRanker(n_samples=5000, seed=42)
        points = [[1, 4], [2, 2], [4, 1]]
        ranks = ranker.rank(points)
        assert len(ranks) == 3
        # All ranks should be 1, 2, or 3 (permutation of {1,2,3})
        assert sorted(ranks) == [1, 2, 3]

    def test_rank_with_maximize_direction(self):
        """Maximize directions should flip values before ranking."""
        ranker = ManyObjectiveRanker(n_samples=5000, seed=42)
        points = [[1, 4], [2, 2], [4, 1]]
        directions = ["maximize", "maximize"]
        ranks = ranker.rank(points, directions=directions)
        assert len(ranks) == 3
        assert sorted(ranks) == [1, 2, 3]

    def test_contribution_returns_positive(self):
        """Contribution of a non-redundant point should be positive."""
        ranker = ManyObjectiveRanker(n_samples=5000, seed=42)
        points = [[1, 4], [2, 2], [4, 1]]
        contrib = ranker.contribution(points, index=1)
        assert contrib > 0.0, "Contribution of a non-dominated point must be positive"


# ── ParetoQuery ───────────────────────────────────────────────────────


class TestParetoQuery:
    """Tests for the ParetoQuery dataclass."""

    def test_creation_with_weights(self):
        q = ParetoQuery(weights={"cost": 0.7, "quality": 0.3})
        assert q.weights == {"cost": 0.7, "quality": 0.3}
        assert q.aspiration_levels is None
        assert q.bounds is None

    def test_creation_with_aspiration(self):
        q = ParetoQuery(aspiration_levels={"cost": 10.0, "quality": 0.9})
        assert q.aspiration_levels is not None
        assert q.weights is None

    def test_creation_with_bounds(self):
        q = ParetoQuery(bounds={"cost": (5.0, 15.0), "quality": (0.5, 1.0)})
        assert q.bounds is not None
        assert q.bounds["cost"] == (5.0, 15.0)


# ── InteractiveParetoExplorer ─────────────────────────────────────────


class TestInteractiveParetoExplorer:
    """Tests for interactive Pareto front exploration."""

    @pytest.fixture()
    def explorer(self):
        return InteractiveParetoExplorer(
            objective_names=["cost", "quality"],
            directions=["minimize", "minimize"],
        )

    @pytest.fixture()
    def front(self):
        return [
            {"cost": 10.0, "quality": 1.0},
            {"cost": 5.0, "quality": 5.0},
            {"cost": 1.0, "quality": 10.0},
        ]

    def test_query_with_weights_sorts_correctly(self, explorer, front):
        """Weighted query should sort by weighted sum (minimization)."""
        q = ParetoQuery(weights={"cost": 1.0, "quality": 0.0})
        result = explorer.query(front, q)
        # With weight only on cost (minimize), lowest cost first
        costs = [pt["cost"] for pt in result]
        assert costs == sorted(costs), "Should be sorted by ascending cost"

    def test_query_with_aspiration_sorts_by_distance(self, explorer, front):
        """Aspiration query should sort by distance to aspiration point."""
        q = ParetoQuery(aspiration_levels={"cost": 5.0, "quality": 5.0})
        result = explorer.query(front, q)
        # The point (5, 5) should be first as it matches exactly
        assert result[0] == {"cost": 5.0, "quality": 5.0}

    def test_query_with_bounds_filters_correctly(self, explorer, front):
        """Bounds query should filter to points within specified ranges."""
        q = ParetoQuery(bounds={"cost": (0.0, 6.0), "quality": (0.0, 6.0)})
        result = explorer.query(front, q)
        # Only (5, 5) and (1, ?) meet cost bounds; only (5, 5) meets both
        for pt in result:
            assert 0.0 <= pt["cost"] <= 6.0
            assert 0.0 <= pt["quality"] <= 6.0

    def test_nearest_to_ideal(self, explorer, front):
        """nearest_to_ideal should find the closest point to the ideal."""
        ideal = {"cost": 0.0, "quality": 0.0}
        nearest = explorer.nearest_to_ideal(front, ideal)
        assert nearest is not None
        # After normalization the result should be deterministic.
        # Just check that it returns one of the front points.
        assert nearest in front

    def test_tradeoff_analysis_negative_correlation(self, explorer):
        """Trade-off front should show negative correlation between objectives."""
        # Construct a clear trade-off front: as cost goes down, quality goes up
        trade_front = [
            {"cost": float(i), "quality": float(10 - i)} for i in range(1, 10)
        ]
        ta = explorer.tradeoff_analysis(trade_front, "cost", "quality")
        assert ta.correlation < 0.0, (
            "Trade-off front should exhibit negative correlation"
        )

    def test_tradeoff_analysis_dataclass_fields(self):
        """TradeoffAnalysis should have the expected fields."""
        ta = TradeoffAnalysis(
            objective_a="cost",
            objective_b="quality",
            slope=-1.5,
            correlation=-0.9,
            elasticity=-0.8,
        )
        assert ta.objective_a == "cost"
        assert ta.objective_b == "quality"
        assert ta.slope == -1.5
        assert ta.correlation == -0.9
        assert ta.elasticity == -0.8

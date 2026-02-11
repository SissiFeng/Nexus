"""Comprehensive tests for MultiFidelityManager and ParameterImportanceAnalyzer.

Tests the infrastructure modules:
- optimization_copilot.infrastructure.multi_fidelity
- optimization_copilot.infrastructure.parameter_importance
"""

import math

import pytest

from optimization_copilot.infrastructure.multi_fidelity import (
    FidelityLevel,
    MultiFidelityManager,
)
from optimization_copilot.infrastructure.parameter_importance import (
    ImportanceResult,
    ParameterImportanceAnalyzer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _three_level_fidelities() -> list[FidelityLevel]:
    """Standard three-level fidelity setup for testing."""
    return [
        FidelityLevel(0, "coarse_dft", 1.0, 0.6),
        FidelityLevel(1, "fine_dft", 5.0, 0.9),
        FidelityLevel(2, "experimental", 50.0, 1.0),
    ]


def _two_level_fidelities() -> list[FidelityLevel]:
    """Simple two-level fidelity setup for testing."""
    return [
        FidelityLevel(0, "low", 1.0, 0.7),
        FidelityLevel(1, "high", 10.0, 1.0),
    ]


def _single_level_fidelity() -> list[FidelityLevel]:
    """Single-level fidelity setup for testing edge cases."""
    return [FidelityLevel(0, "only", 1.0, 1.0)]


def _make_observations(
    n: int,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
) -> list[dict]:
    """Create n observations with linearly spaced x, y, and objective."""
    obs = []
    for i in range(n):
        frac = i / max(n - 1, 1)
        obs.append({
            "x": x_range[0] + frac * (x_range[1] - x_range[0]),
            "y": y_range[0] + frac * (y_range[1] - y_range[0]),
            "objective": frac,
        })
    return obs


def _make_param_specs(*names: str, ptype: str = "continuous") -> list[dict]:
    """Create parameter specs for the given names."""
    return [{"name": n, "type": ptype} for n in names]


# ===================================================================
# Part 1: FidelityLevel dataclass
# ===================================================================


class TestFidelityLevelDataclass:
    """Tests for FidelityLevel creation and serialization."""

    def test_create_with_defaults(self):
        fl = FidelityLevel(level=0, name="low", cost_multiplier=1.0)
        assert fl.level == 0
        assert fl.name == "low"
        assert fl.cost_multiplier == 1.0
        assert fl.correlation == 0.8  # default

    def test_create_with_custom_correlation(self):
        fl = FidelityLevel(level=2, name="high", cost_multiplier=50.0, correlation=0.95)
        assert fl.correlation == 0.95

    def test_to_dict(self):
        fl = FidelityLevel(level=1, name="mid", cost_multiplier=5.0, correlation=0.85)
        d = fl.to_dict()
        assert d == {
            "level": 1,
            "name": "mid",
            "cost_multiplier": 5.0,
            "correlation": 0.85,
        }

    def test_from_dict(self):
        data = {"level": 0, "name": "low", "cost_multiplier": 1.0, "correlation": 0.6}
        fl = FidelityLevel.from_dict(data)
        assert fl.level == 0
        assert fl.name == "low"
        assert fl.cost_multiplier == 1.0
        assert fl.correlation == 0.6

    def test_from_dict_default_correlation(self):
        data = {"level": 0, "name": "low", "cost_multiplier": 1.0}
        fl = FidelityLevel.from_dict(data)
        assert fl.correlation == 0.8

    def test_to_dict_from_dict_roundtrip(self):
        original = FidelityLevel(level=2, name="exp", cost_multiplier=50.0, correlation=0.99)
        restored = FidelityLevel.from_dict(original.to_dict())
        assert restored.level == original.level
        assert restored.name == original.name
        assert restored.cost_multiplier == original.cost_multiplier
        assert restored.correlation == original.correlation


# ===================================================================
# Part 2: MultiFidelityManager
# ===================================================================


class TestMultiFidelityManagerInit:
    """Tests for MultiFidelityManager initialization."""

    def test_init_with_three_levels(self):
        levels = _three_level_fidelities()
        mgr = MultiFidelityManager(levels)
        assert mgr.n_levels == 3

    def test_init_sorts_by_level(self):
        # Provide levels out of order
        levels = [
            FidelityLevel(2, "high", 50.0, 1.0),
            FidelityLevel(0, "low", 1.0, 0.6),
            FidelityLevel(1, "mid", 5.0, 0.9),
        ]
        mgr = MultiFidelityManager(levels)
        assert mgr.lowest_fidelity.level == 0
        assert mgr.highest_fidelity.level == 2

    def test_init_empty_raises(self):
        with pytest.raises(ValueError, match="At least one fidelity level"):
            MultiFidelityManager([])

    def test_init_duplicate_levels_raises(self):
        levels = [
            FidelityLevel(0, "low", 1.0, 0.6),
            FidelityLevel(0, "also_low", 2.0, 0.7),
        ]
        with pytest.raises(ValueError, match="Duplicate fidelity level"):
            MultiFidelityManager(levels)

    def test_lowest_and_highest_properties(self):
        mgr = MultiFidelityManager(_three_level_fidelities())
        assert mgr.lowest_fidelity.name == "coarse_dft"
        assert mgr.highest_fidelity.name == "experimental"


class TestAddObservation:
    """Tests for MultiFidelityManager.add_observation."""

    def test_add_observation_basic(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        mgr.add_observation({"x": 1.0, "objective": 0.5}, fidelity=0)
        obs = mgr.get_observations(fidelity=0)
        assert len(obs) == 1
        assert obs[0]["objective"] == 0.5

    def test_observation_tagged_with_fidelity(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        mgr.add_observation({"x": 1.0, "objective": 0.5}, fidelity=0)
        obs = mgr.get_observations(fidelity=0)
        assert obs[0]["_fidelity_level"] == 0

    def test_add_observation_unknown_fidelity_raises(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        with pytest.raises(ValueError, match="Unknown fidelity level"):
            mgr.add_observation({"x": 1.0, "objective": 0.5}, fidelity=99)

    def test_add_multiple_observations_same_level(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        for i in range(5):
            mgr.add_observation({"x": float(i), "objective": float(i)}, fidelity=0)
        assert len(mgr.get_observations(fidelity=0)) == 5
        assert len(mgr.get_observations(fidelity=1)) == 0

    def test_add_observations_different_levels(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        mgr.add_observation({"x": 1.0, "objective": 0.5}, fidelity=0)
        mgr.add_observation({"x": 2.0, "objective": 0.9}, fidelity=1)
        assert len(mgr.get_observations(fidelity=0)) == 1
        assert len(mgr.get_observations(fidelity=1)) == 1

    def test_get_all_observations(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        mgr.add_observation({"x": 1.0, "objective": 0.5}, fidelity=0)
        mgr.add_observation({"x": 2.0, "objective": 0.9}, fidelity=1)
        all_obs = mgr.get_observations()
        assert len(all_obs) == 2

    def test_observation_does_not_mutate_original(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        original = {"x": 1.0, "objective": 0.5}
        mgr.add_observation(original, fidelity=0)
        # Original dict should not have _fidelity_level added
        assert "_fidelity_level" not in original


class TestSuggestFidelity:
    """Tests for MultiFidelityManager.suggest_fidelity."""

    def test_single_level_always_returns_that_level(self):
        mgr = MultiFidelityManager(_single_level_fidelity())
        result = mgr.suggest_fidelity({"x": 0.5})
        assert result.level == 0

    def test_budget_too_tight_returns_lowest(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        # Budget is less than cost of next level (10.0)
        result = mgr.suggest_fidelity({"x": 0.5}, budget_remaining=3.0)
        assert result.level == 0

    def test_no_observations_returns_lowest(self):
        mgr = MultiFidelityManager(_three_level_fidelities())
        result = mgr.suggest_fidelity({"x": 0.5})
        assert result.level == 0

    def test_unlimited_budget_not_constrained(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        result = mgr.suggest_fidelity({"x": 0.5}, budget_remaining=None)
        assert result.level == 0  # No observations, starts at lowest

    def test_suggest_returns_fidelity_level_object(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        result = mgr.suggest_fidelity({"x": 0.5})
        assert isinstance(result, FidelityLevel)


class TestPromotionCandidates:
    """Tests for MultiFidelityManager.promotion_candidates."""

    def test_no_candidates_when_no_observations(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        candidates = mgr.promotion_candidates()
        assert candidates == []

    def test_single_level_returns_empty(self):
        mgr = MultiFidelityManager(_single_level_fidelity())
        mgr.add_observation({"x": 1.0, "objective": 0.9}, fidelity=0)
        candidates = mgr.promotion_candidates()
        assert candidates == []

    def test_promotion_candidates_with_data(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        # Add low-fidelity observations with varied quality
        for i in range(10):
            mgr.add_observation(
                {"x": float(i), "objective": float(i) / 10.0}, fidelity=0
            )
        candidates = mgr.promotion_candidates(top_fraction=0.2)
        # Should return some candidates from fidelity 0
        # (those meeting the promotion threshold)
        assert isinstance(candidates, list)

    def test_promotion_respects_top_fraction(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        # Add 10 observations
        for i in range(10):
            mgr.add_observation(
                {"x": float(i) * 10, "objective": float(i)}, fidelity=0
            )
        # With top_fraction=1.0, considers all observations
        candidates_all = mgr.promotion_candidates(top_fraction=1.0)
        # With top_fraction=0.1, considers fewer
        candidates_few = mgr.promotion_candidates(top_fraction=0.1)
        assert len(candidates_all) >= len(candidates_few)


class TestBuildMultiFidelityDataset:
    """Tests for MultiFidelityManager.build_multi_fidelity_dataset."""

    def test_empty_dataset(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        dataset = mgr.build_multi_fidelity_dataset()
        assert dataset == []

    def test_dataset_includes_weights(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        mgr.add_observation({"x": 1.0, "objective": 0.5}, fidelity=0)
        mgr.add_observation({"x": 2.0, "objective": 0.9}, fidelity=1)
        dataset = mgr.build_multi_fidelity_dataset()
        assert len(dataset) == 2
        assert "_weight" in dataset[0]
        assert "_weight" in dataset[1]

    def test_higher_fidelity_gets_higher_weight(self):
        levels = _two_level_fidelities()
        mgr = MultiFidelityManager(levels)
        mgr.add_observation({"x": 1.0, "objective": 0.5}, fidelity=0)
        mgr.add_observation({"x": 2.0, "objective": 0.9}, fidelity=1)
        dataset = mgr.build_multi_fidelity_dataset()
        low_weight = [d for d in dataset if d["_fidelity_level"] == 0][0]["_weight"]
        high_weight = [d for d in dataset if d["_fidelity_level"] == 1][0]["_weight"]
        assert high_weight > low_weight


class TestFidelitySummary:
    """Tests for MultiFidelityManager.fidelity_summary."""

    def test_summary_empty(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        summary = mgr.fidelity_summary()
        assert summary["n_levels"] == 2
        assert summary["total_observations"] == 0
        assert summary["total_estimated_cost"] == 0.0

    def test_summary_with_observations(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        mgr.add_observation({"x": 1.0, "objective": 0.5}, fidelity=0)
        mgr.add_observation({"x": 2.0, "objective": 0.9}, fidelity=0)
        mgr.add_observation({"x": 3.0, "objective": 0.8}, fidelity=1)
        summary = mgr.fidelity_summary()
        assert summary["total_observations"] == 3
        # Cost: 2 * 1.0 + 1 * 10.0 = 12.0
        assert summary["total_estimated_cost"] == pytest.approx(12.0)

    def test_summary_per_level_stats(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        mgr.add_observation({"x": 1.0, "objective": 0.3}, fidelity=0)
        mgr.add_observation({"x": 2.0, "objective": 0.7}, fidelity=0)
        summary = mgr.fidelity_summary()
        level_0_stats = summary["per_level"]["0"]
        assert level_0_stats["n_observations"] == 2
        assert level_0_stats["best_objective"] == pytest.approx(0.7)
        assert level_0_stats["mean_objective"] == pytest.approx(0.5)


class TestMultiFidelityManagerSerialization:
    """Tests for MultiFidelityManager.to_dict and from_dict."""

    def test_to_dict_structure(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        d = mgr.to_dict()
        assert "levels" in d
        assert "observations" in d
        assert "summary" in d
        assert len(d["levels"]) == 2

    def test_from_dict_restores_levels(self):
        mgr = MultiFidelityManager(_three_level_fidelities())
        d = mgr.to_dict()
        restored = MultiFidelityManager.from_dict(d)
        assert restored.n_levels == 3
        assert restored.lowest_fidelity.name == "coarse_dft"
        assert restored.highest_fidelity.name == "experimental"

    def test_roundtrip_with_observations(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        mgr.add_observation({"x": 1.0, "objective": 0.5}, fidelity=0)
        mgr.add_observation({"x": 2.0, "objective": 0.9}, fidelity=1)
        d = mgr.to_dict()
        restored = MultiFidelityManager.from_dict(d)
        assert len(restored.get_observations(fidelity=0)) == 1
        assert len(restored.get_observations(fidelity=1)) == 1
        assert restored.get_observations(fidelity=0)[0]["objective"] == 0.5

    def test_roundtrip_preserves_summary(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        mgr.add_observation({"x": 1.0, "objective": 0.5}, fidelity=0)
        d = mgr.to_dict()
        restored = MultiFidelityManager.from_dict(d)
        summary = restored.fidelity_summary()
        assert summary["total_observations"] == 1


class TestMultiFidelityEdgeCases:
    """Edge cases for MultiFidelityManager."""

    def test_single_fidelity_manager(self):
        mgr = MultiFidelityManager(_single_level_fidelity())
        assert mgr.n_levels == 1
        assert mgr.lowest_fidelity == mgr.highest_fidelity

    def test_no_observations_promotion_threshold_is_zero(self):
        """With no observations, the internal promotion threshold should be 0.0."""
        mgr = MultiFidelityManager(_two_level_fidelities())
        # Calling promotion_candidates should not crash
        candidates = mgr.promotion_candidates()
        assert candidates == []

    def test_observations_without_objective_key(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        mgr.add_observation({"x": 1.0}, fidelity=0)  # no 'objective' key
        obs = mgr.get_observations(fidelity=0)
        assert len(obs) == 1
        # Summary should still work
        summary = mgr.fidelity_summary()
        assert summary["total_observations"] == 1

    def test_many_observations_all_same_fidelity(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        for i in range(100):
            mgr.add_observation({"x": float(i), "objective": float(i)}, fidelity=0)
        assert len(mgr.get_observations(fidelity=0)) == 100
        assert len(mgr.get_observations(fidelity=1)) == 0

    def test_negative_objective_values(self):
        mgr = MultiFidelityManager(_two_level_fidelities())
        mgr.add_observation({"x": 1.0, "objective": -5.0}, fidelity=0)
        mgr.add_observation({"x": 2.0, "objective": -1.0}, fidelity=0)
        summary = mgr.fidelity_summary()
        assert summary["per_level"]["0"]["best_objective"] == pytest.approx(-1.0)

    def test_percentile_single_value(self):
        """Static method _percentile with a single value returns that value."""
        result = MultiFidelityManager._percentile([42.0], 80.0)
        assert result == 42.0

    def test_percentile_two_values(self):
        result = MultiFidelityManager._percentile([10.0, 20.0], 50.0)
        assert result == pytest.approx(15.0)


# ===================================================================
# Part 3: ImportanceResult dataclass
# ===================================================================


class TestImportanceResult:
    """Tests for ImportanceResult creation and serialization."""

    def test_create_basic(self):
        result = ImportanceResult(
            scores={"x": 0.8, "y": 0.3},
            method="correlation",
        )
        assert result.scores["x"] == 0.8
        assert result.method == "correlation"
        assert result.details == {}

    def test_create_with_details(self):
        result = ImportanceResult(
            scores={"x": 0.9},
            method="variance",
            details={"raw_scores": {"x": 5.0}},
        )
        assert result.details["raw_scores"]["x"] == 5.0

    def test_to_dict(self):
        result = ImportanceResult(
            scores={"x": 0.7, "y": 0.4},
            method="pedanova",
            details={"n_observations": 50},
        )
        d = result.to_dict()
        assert d == {
            "scores": {"x": 0.7, "y": 0.4},
            "method": "pedanova",
            "details": {"n_observations": 50},
        }

    def test_to_dict_preserves_empty_details(self):
        result = ImportanceResult(scores={"a": 0.5}, method="variance")
        d = result.to_dict()
        assert d["details"] == {}


# ===================================================================
# Part 4: ParameterImportanceAnalyzer
# ===================================================================


class TestAnalyzerInit:
    """Tests for ParameterImportanceAnalyzer initialization."""

    def test_default_method_is_auto(self):
        analyzer = ParameterImportanceAnalyzer()
        assert analyzer.method == "auto"

    def test_explicit_method(self):
        for m in ("variance", "correlation", "pedanova", "auto"):
            analyzer = ParameterImportanceAnalyzer(method=m)
            assert analyzer.method == m

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            ParameterImportanceAnalyzer(method="invalid")

    def test_to_dict(self):
        analyzer = ParameterImportanceAnalyzer(method="correlation")
        d = analyzer.to_dict()
        assert d == {"method": "correlation"}


class TestAnalyzeVariance:
    """Tests for ParameterImportanceAnalyzer with variance method."""

    def test_variance_basic(self):
        analyzer = ParameterImportanceAnalyzer(method="variance")
        # x is strongly correlated with objective (important)
        # y is random noise (less important)
        observations = []
        for i in range(20):
            x_val = float(i) / 20.0
            observations.append({
                "x": x_val,
                "y": (i % 5) * 0.2,  # noisy
                "objective": x_val,  # objective = x
            })
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        assert result.method == "variance"
        # x should be scored higher (top performers have tightly clustered x values)
        assert result.scores["x"] >= result.scores["y"]

    def test_variance_all_scores_in_range(self):
        analyzer = ParameterImportanceAnalyzer(method="variance")
        observations = _make_observations(20)
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        for score in result.scores.values():
            assert 0.0 <= score <= 1.0

    def test_variance_returns_importance_result(self):
        analyzer = ParameterImportanceAnalyzer(method="variance")
        observations = _make_observations(10)
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        assert isinstance(result, ImportanceResult)


class TestAnalyzeCorrelation:
    """Tests for ParameterImportanceAnalyzer with correlation method."""

    def test_correlation_perfect_positive(self):
        analyzer = ParameterImportanceAnalyzer(method="correlation")
        # x is perfectly correlated with objective
        observations = [{"x": float(i), "objective": float(i)} for i in range(20)]
        specs = _make_param_specs("x")
        result = analyzer.analyze(observations, specs)
        assert result.scores["x"] == pytest.approx(1.0, abs=0.01)

    def test_correlation_perfect_negative(self):
        analyzer = ParameterImportanceAnalyzer(method="correlation")
        # x is perfectly negatively correlated with objective
        observations = [{"x": float(i), "objective": -float(i)} for i in range(20)]
        specs = _make_param_specs("x")
        result = analyzer.analyze(observations, specs)
        # Uses absolute correlation, so should still be high
        assert result.scores["x"] == pytest.approx(1.0, abs=0.01)

    def test_correlation_uncorrelated(self):
        analyzer = ParameterImportanceAnalyzer(method="correlation")
        # x is unrelated to objective
        observations = [
            {"x": float(i), "y": float(i), "objective": float(i)}
            for i in range(20)
        ]
        # Override y values with something uncorrelated
        import random
        rng = random.Random(42)
        for obs in observations:
            obs["z"] = rng.random()
        specs = _make_param_specs("x", "z")
        result = analyzer.analyze(observations, specs)
        # x should be more important than z
        assert result.scores["x"] > result.scores["z"]

    def test_correlation_categorical_gets_zero(self):
        analyzer = ParameterImportanceAnalyzer(method="correlation")
        observations = [
            {"x": float(i), "cat": "a" if i < 10 else "b", "objective": float(i)}
            for i in range(20)
        ]
        specs = [
            {"name": "x", "type": "continuous"},
            {"name": "cat", "type": "categorical"},
        ]
        result = analyzer.analyze(observations, specs)
        assert result.scores["cat"] == 0.0

    def test_correlation_known_ranking(self):
        """With known linear relationship, most correlated parameter ranks highest."""
        analyzer = ParameterImportanceAnalyzer(method="correlation")
        observations = []
        for i in range(30):
            x = float(i)
            y = float(i) * 0.5 + 10.0  # weaker correlation with objective
            objective = x * 2.0  # objective = 2*x
            observations.append({"x": x, "y": y, "objective": objective})
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        # Both should have high scores since y = 0.5*x + 10 correlates with 2*x
        # but x should be at least as high
        assert result.scores["x"] >= result.scores["y"] - 0.01


class TestAnalyzePedanova:
    """Tests for ParameterImportanceAnalyzer with pedanova method."""

    def test_pedanova_basic(self):
        analyzer = ParameterImportanceAnalyzer(method="pedanova")
        observations = []
        for i in range(20):
            x_val = float(i) / 20.0
            observations.append({
                "x": x_val,
                "y": 0.5,  # constant, should be unimportant
                "objective": x_val,
            })
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        assert result.method == "pedanova"
        # x should have higher importance than constant y
        assert result.scores["x"] > result.scores["y"]

    def test_pedanova_f_statistic_constant_parameter(self):
        """A constant parameter should have 0 importance."""
        analyzer = ParameterImportanceAnalyzer(method="pedanova")
        observations = [
            {"x": 5.0, "y": float(i), "objective": float(i)}
            for i in range(15)
        ]
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        assert result.scores["x"] == pytest.approx(0.0)
        assert result.scores["y"] == pytest.approx(1.0, abs=0.01)

    def test_pedanova_requires_sufficient_data(self):
        """Auto method falls back to correlation when < 10 observations."""
        analyzer = ParameterImportanceAnalyzer(method="auto")
        observations = _make_observations(5)
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        assert result.method == "correlation"

    def test_pedanova_auto_uses_pedanova_with_enough_data(self):
        analyzer = ParameterImportanceAnalyzer(method="auto")
        observations = _make_observations(15)
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        assert result.method == "pedanova"

    def test_pedanova_all_scores_in_range(self):
        analyzer = ParameterImportanceAnalyzer(method="pedanova")
        observations = _make_observations(20)
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        for score in result.scores.values():
            assert 0.0 <= score <= 1.0


class TestAnalyzeCombined:
    """Tests for the combined analyze() method."""

    def test_no_observations_raises(self):
        analyzer = ParameterImportanceAnalyzer()
        with pytest.raises(ValueError, match="No observations"):
            analyzer.analyze([], _make_param_specs("x"))

    def test_no_parameter_specs_raises(self):
        analyzer = ParameterImportanceAnalyzer()
        with pytest.raises(ValueError, match="No parameter specs"):
            analyzer.analyze([{"x": 1.0, "objective": 0.5}], [])

    def test_observations_missing_objective(self):
        """All observations missing objective should produce zero scores."""
        analyzer = ParameterImportanceAnalyzer(method="correlation")
        observations = [{"x": 1.0}, {"x": 2.0}]  # no 'objective' key
        specs = _make_param_specs("x")
        result = analyzer.analyze(observations, specs)
        assert result.scores["x"] == 0.0
        assert "no valid observations" in result.details.get("reason", "")

    def test_result_details_contain_raw_scores(self):
        analyzer = ParameterImportanceAnalyzer(method="variance")
        observations = _make_observations(10)
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        assert "raw_scores" in result.details
        assert "n_observations" in result.details

    def test_method_is_recorded_in_result(self):
        for method in ("variance", "correlation", "pedanova"):
            analyzer = ParameterImportanceAnalyzer(method=method)
            observations = _make_observations(20)
            specs = _make_param_specs("x")
            result = analyzer.analyze(observations, specs)
            assert result.method == method


class TestRankParameters:
    """Tests that verify parameter ranking from importance scores."""

    def test_rank_by_score(self):
        """Parameters should be rankable from the importance scores."""
        analyzer = ParameterImportanceAnalyzer(method="correlation")
        observations = []
        for i in range(30):
            observations.append({
                "x": float(i),
                "y": float(i) * 0.1,
                "z": 5.0,  # constant
                "objective": float(i) * 2.0,
            })
        specs = _make_param_specs("x", "y", "z")
        result = analyzer.analyze(observations, specs)
        ranked = sorted(result.scores.items(), key=lambda kv: kv[1], reverse=True)
        # x and y should be ranked higher than z (constant)
        ranked_names = [name for name, _ in ranked]
        assert ranked_names[-1] == "z"

    def test_single_parameter_gets_score_1(self):
        """A single important parameter should be normalized to 1.0."""
        analyzer = ParameterImportanceAnalyzer(method="correlation")
        observations = [{"x": float(i), "objective": float(i)} for i in range(20)]
        specs = _make_param_specs("x")
        result = analyzer.analyze(observations, specs)
        assert result.scores["x"] == pytest.approx(1.0, abs=0.01)


class TestEdgeCasesImportance:
    """Edge cases for ParameterImportanceAnalyzer."""

    def test_constant_objective(self):
        """When objective is constant, no parameter should be important."""
        analyzer = ParameterImportanceAnalyzer(method="correlation")
        observations = [
            {"x": float(i), "y": float(i) * 2.0, "objective": 5.0}
            for i in range(20)
        ]
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        for score in result.scores.values():
            assert score == pytest.approx(0.0)

    def test_all_parameters_identical(self):
        """When all parameter values are identical, importance should be 0."""
        analyzer = ParameterImportanceAnalyzer(method="variance")
        observations = [
            {"x": 1.0, "y": 2.0, "objective": float(i)}
            for i in range(20)
        ]
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        for score in result.scores.values():
            assert score == pytest.approx(0.0)

    def test_single_observation(self):
        """Single observation should not crash."""
        analyzer = ParameterImportanceAnalyzer(method="variance")
        observations = [{"x": 1.0, "y": 2.0, "objective": 0.5}]
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        assert isinstance(result, ImportanceResult)

    def test_nan_and_inf_values_skipped(self):
        """NaN and inf parameter values should be safely handled."""
        analyzer = ParameterImportanceAnalyzer(method="correlation")
        observations = [
            {"x": float("nan"), "y": 1.0, "objective": 0.5},
            {"x": float("inf"), "y": 2.0, "objective": 0.7},
            {"x": 3.0, "y": 3.0, "objective": 0.9},
            {"x": 4.0, "y": 4.0, "objective": 0.8},
        ]
        specs = _make_param_specs("x", "y")
        result = analyzer.analyze(observations, specs)
        # Should not crash and y should have a valid score
        assert isinstance(result, ImportanceResult)

    def test_custom_objective_key(self):
        """Analyzer should respect a custom objective key."""
        analyzer = ParameterImportanceAnalyzer(method="correlation")
        observations = [
            {"x": float(i), "my_kpi": float(i) * 2.0}
            for i in range(20)
        ]
        specs = _make_param_specs("x")
        result = analyzer.analyze(observations, specs, objective_key="my_kpi")
        assert result.scores["x"] == pytest.approx(1.0, abs=0.01)

    def test_categorical_parameter_in_pedanova(self):
        """Categorical parameters should get a chi-squared-like score in pedanova."""
        analyzer = ParameterImportanceAnalyzer(method="pedanova")
        observations = []
        for i in range(20):
            cat_val = "good" if i >= 15 else "bad"
            observations.append({
                "cat": cat_val,
                "x": float(i),
                "objective": float(i),
            })
        specs = [
            {"name": "cat", "type": "categorical"},
            {"name": "x", "type": "continuous"},
        ]
        result = analyzer.analyze(observations, specs)
        assert "cat" in result.scores
        assert "x" in result.scores

    def test_two_observations_correlation(self):
        """With exactly 2 observations, correlation should work."""
        analyzer = ParameterImportanceAnalyzer(method="correlation")
        observations = [
            {"x": 0.0, "objective": 0.0},
            {"x": 1.0, "objective": 1.0},
        ]
        specs = _make_param_specs("x")
        result = analyzer.analyze(observations, specs)
        assert result.scores["x"] == pytest.approx(1.0, abs=0.01)


class TestStaticHelpers:
    """Tests for static helper methods on ParameterImportanceAnalyzer."""

    def test_pearson_correlation_perfect(self):
        r = ParameterImportanceAnalyzer._pearson_correlation(
            [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]
        )
        assert r == pytest.approx(1.0)

    def test_pearson_correlation_zero_variance(self):
        r = ParameterImportanceAnalyzer._pearson_correlation(
            [5.0, 5.0, 5.0], [1.0, 2.0, 3.0]
        )
        assert r == pytest.approx(0.0)

    def test_pearson_correlation_single_value(self):
        r = ParameterImportanceAnalyzer._pearson_correlation([1.0], [2.0])
        assert r == pytest.approx(0.0)

    def test_f_statistic_identical_groups(self):
        f = ParameterImportanceAnalyzer._f_statistic(
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]
        )
        assert f == pytest.approx(0.0, abs=1e-10)

    def test_f_statistic_distinct_groups(self):
        f = ParameterImportanceAnalyzer._f_statistic(
            [1.0, 1.0, 1.0], [10.0, 10.0, 10.0]
        )
        assert f == float("inf")

    def test_f_statistic_empty_group(self):
        f = ParameterImportanceAnalyzer._f_statistic([], [1.0, 2.0])
        assert f == 0.0

    def test_variance_basic(self):
        v = ParameterImportanceAnalyzer._variance([1.0, 2.0, 3.0, 4.0, 5.0])
        assert v == pytest.approx(2.5)

    def test_variance_single_value(self):
        v = ParameterImportanceAnalyzer._variance([42.0])
        assert v == 0.0

    def test_extract_numeric_skips_nan(self):
        obs = [
            {"x": 1.0},
            {"x": float("nan")},
            {"x": 3.0},
            {"x": float("inf")},
        ]
        vals = ParameterImportanceAnalyzer._extract_numeric(obs, "x")
        assert vals == [1.0, 3.0]

    def test_extract_numeric_missing_key(self):
        obs = [{"y": 1.0}, {"y": 2.0}]
        vals = ParameterImportanceAnalyzer._extract_numeric(obs, "x")
        assert vals == []

    def test_normalize_scores_max_normalization(self):
        analyzer = ParameterImportanceAnalyzer()
        normalized = analyzer._normalize_scores({"a": 10.0, "b": 5.0, "c": 0.0})
        assert normalized["a"] == pytest.approx(1.0)
        assert normalized["b"] == pytest.approx(0.5)
        assert normalized["c"] == pytest.approx(0.0)

    def test_normalize_scores_all_zeros(self):
        analyzer = ParameterImportanceAnalyzer()
        normalized = analyzer._normalize_scores({"a": 0.0, "b": 0.0})
        assert normalized["a"] == 0.0
        assert normalized["b"] == 0.0

    def test_normalize_scores_empty(self):
        analyzer = ParameterImportanceAnalyzer()
        normalized = analyzer._normalize_scores({})
        assert normalized == {}

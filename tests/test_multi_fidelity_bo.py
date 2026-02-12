"""Tests for full multi-fidelity Bayesian optimization backend."""

from __future__ import annotations

import math
import pytest

from optimization_copilot.core.models import Observation, ParameterSpec, VariableType
from optimization_copilot.fidelity.config import FidelityLevel, FidelityConfig, CostModel
from optimization_copilot.backends.multi_fidelity import (
    MultiFidelityBackend,
    MFObservation,
)
from optimization_copilot.backends.mf_acquisition import (
    cost_aware_ei,
    fidelity_weighted_ei,
    multi_fidelity_knowledge_gradient,
    entropy_search_multi_fidelity,
)
from optimization_copilot.backends.mf_diagnostics import (
    cross_fidelity_correlation,
    render_fidelity_comparison,
    render_cost_allocation,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _default_config() -> FidelityConfig:
    """Create a default multi-fidelity config for testing."""
    return FidelityConfig(
        levels=[
            FidelityLevel(name="low", fidelity=0.3, cost=1.0, noise_multiplier=2.0),
            FidelityLevel(name="medium", fidelity=0.6, cost=5.0, noise_multiplier=1.5),
            FidelityLevel(name="high", fidelity=1.0, cost=10.0, noise_multiplier=1.0),
        ],
        target_fidelity="high",
        cost_budget=100.0,
    )


def _two_level_config() -> FidelityConfig:
    """Create a simple two-level config."""
    return FidelityConfig(
        levels=[
            FidelityLevel(name="low", fidelity=0.3, cost=1.0, noise_multiplier=2.0),
            FidelityLevel(name="high", fidelity=1.0, cost=10.0, noise_multiplier=1.0),
        ],
        target_fidelity="high",
        cost_budget=50.0,
    )


def _default_specs() -> list[ParameterSpec]:
    """Create default parameter specs."""
    return [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
    ]


def _make_observations(n: int = 5) -> list[Observation]:
    """Create standard observations."""
    return [
        Observation(
            iteration=i,
            parameters={"x1": i * 0.2, "x2": 1.0 - i * 0.2},
            kpi_values={"y": (i * 0.2 - 0.5) ** 2},
            timestamp=float(i),
        )
        for i in range(n)
    ]


# ── FidelityLevel Tests ─────────────────────────────────────────────


class TestFidelityLevel:
    """Test FidelityLevel dataclass."""

    def test_valid_construction(self):
        lv = FidelityLevel(name="low", fidelity=0.3, cost=1.0)
        assert lv.name == "low"
        assert lv.fidelity == 0.3
        assert lv.cost == 1.0
        assert lv.noise_multiplier == 1.0

    def test_custom_noise_multiplier(self):
        lv = FidelityLevel(name="low", fidelity=0.3, cost=1.0, noise_multiplier=2.0)
        assert lv.noise_multiplier == 2.0

    def test_fidelity_out_of_range_raises(self):
        with pytest.raises(ValueError):
            FidelityLevel(name="bad", fidelity=1.5, cost=1.0)

    def test_negative_fidelity_raises(self):
        with pytest.raises(ValueError):
            FidelityLevel(name="bad", fidelity=-0.1, cost=1.0)

    def test_zero_cost_raises(self):
        with pytest.raises(ValueError):
            FidelityLevel(name="bad", fidelity=0.5, cost=0.0)

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError):
            FidelityLevel(name="bad", fidelity=0.5, cost=-1.0)

    def test_negative_noise_raises(self):
        with pytest.raises(ValueError):
            FidelityLevel(name="bad", fidelity=0.5, cost=1.0, noise_multiplier=-1.0)

    def test_metadata_default_empty(self):
        lv = FidelityLevel(name="test", fidelity=0.5, cost=1.0)
        assert lv.metadata == {}


# ── FidelityConfig Tests ─────────────────────────────────────────────


class TestFidelityConfig:
    """Test FidelityConfig dataclass."""

    def test_n_levels(self):
        config = _default_config()
        assert config.n_levels == 3

    def test_target_level(self):
        config = _default_config()
        assert config.target_level.name == "high"
        assert config.target_level.fidelity == 1.0

    def test_get_level(self):
        config = _default_config()
        lv = config.get_level("medium")
        assert lv.name == "medium"
        assert lv.cost == 5.0

    def test_get_level_missing_raises(self):
        config = _default_config()
        with pytest.raises(KeyError):
            config.get_level("nonexistent")

    def test_get_cost_ratio(self):
        config = _default_config()
        assert config.get_cost_ratio("low") == pytest.approx(0.1)  # 1/10
        assert config.get_cost_ratio("high") == pytest.approx(1.0)  # 10/10

    def test_empty_levels_raises(self):
        with pytest.raises(ValueError):
            FidelityConfig(levels=[], target_fidelity="high")

    def test_target_not_in_levels_raises(self):
        levels = [FidelityLevel(name="low", fidelity=0.3, cost=1.0)]
        with pytest.raises(ValueError):
            FidelityConfig(levels=levels, target_fidelity="high")


# ── CostModel Tests ──────────────────────────────────────────────────


class TestCostModel:
    """Test CostModel budget tracking."""

    def test_initial_state(self):
        cm = CostModel(budget=100.0)
        assert cm.remaining == 100.0
        assert cm.spent == 0.0

    def test_spend(self):
        cm = CostModel(budget=100.0)
        cm.spend(30.0)
        assert cm.spent == pytest.approx(30.0)
        assert cm.remaining == pytest.approx(70.0)

    def test_multiple_spends(self):
        cm = CostModel(budget=100.0)
        cm.spend(20.0)
        cm.spend(30.0)
        assert cm.spent == pytest.approx(50.0)
        assert cm.remaining == pytest.approx(50.0)

    def test_can_afford_true(self):
        cm = CostModel(budget=100.0)
        assert cm.can_afford(50.0) is True

    def test_can_afford_false(self):
        cm = CostModel(budget=100.0)
        cm.spend(90.0)
        assert cm.can_afford(20.0) is False

    def test_can_afford_exact(self):
        cm = CostModel(budget=100.0)
        assert cm.can_afford(100.0) is True

    def test_negative_spend_raises(self):
        cm = CostModel(budget=100.0)
        with pytest.raises(ValueError):
            cm.spend(-5.0)

    def test_overspend_remaining_zero(self):
        cm = CostModel(budget=100.0)
        cm.spend(120.0)
        assert cm.remaining == 0.0
        assert cm.spent == 120.0


# ── MFObservation and Backend Construction ───────────────────────────


class TestMFObservationAndConstruction:
    """Test MFObservation and MultiFidelityBackend construction."""

    def test_mf_observation_construction(self):
        obs = MFObservation(
            parameters={"x1": 0.5, "x2": 0.3},
            kpi_value=1.23,
            fidelity_level="low",
            cost=1.0,
        )
        assert obs.kpi_value == 1.23
        assert obs.fidelity_level == "low"

    def test_backend_construction(self):
        config = _default_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        assert backend.name() == "multi_fidelity_bo"

    def test_backend_capabilities(self):
        config = _default_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        caps = backend.capabilities()
        assert caps["supports_multi_fidelity"] is True
        assert caps["n_fidelity_levels"] == 3
        assert "high" in caps["fidelity_levels"]

    def test_add_mf_observation(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        backend._parameter_specs = _default_specs()

        obs = MFObservation(
            parameters={"x1": 0.5, "x2": 0.3},
            kpi_value=1.0,
            fidelity_level="low",
            cost=1.0,
        )
        backend.add_mf_observation(obs)
        report = backend.get_cost_report()
        assert report["spent"] == 1.0

    def test_add_multiple_observations(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        backend._parameter_specs = _default_specs()

        for i in range(5):
            obs = MFObservation(
                parameters={"x1": i * 0.2, "x2": 0.5},
                kpi_value=float(i),
                fidelity_level="low",
                cost=1.0,
            )
            backend.add_mf_observation(obs)

        report = backend.get_cost_report()
        assert report["spent"] == 5.0
        assert report["per_level"]["low"]["n_evaluations"] == 5


# ── Cost-Aware EI Tests ──────────────────────────────────────────────


class TestCostAwareEI:
    """Test cost-aware expected improvement."""

    def test_positive_value(self):
        val = cost_aware_ei(mean=0.5, variance=0.25, best_y=0.0, cost=1.0)
        assert val > 0

    def test_scales_inversely_with_cost(self):
        val_cheap = cost_aware_ei(mean=0.5, variance=0.25, best_y=0.0, cost=1.0)
        val_expensive = cost_aware_ei(mean=0.5, variance=0.25, best_y=0.0, cost=10.0)
        assert val_cheap > val_expensive

    def test_zero_variance_returns_zero(self):
        val = cost_aware_ei(mean=0.5, variance=0.0, best_y=0.0, cost=1.0)
        assert val == 0.0

    def test_zero_cost_returns_zero(self):
        val = cost_aware_ei(mean=0.5, variance=0.25, best_y=0.0, cost=0.0)
        assert val == 0.0

    def test_worse_mean_lower_ei(self):
        val_good = cost_aware_ei(mean=-0.5, variance=0.25, best_y=0.0, cost=1.0)
        val_bad = cost_aware_ei(mean=2.0, variance=0.25, best_y=0.0, cost=1.0)
        assert val_good > val_bad

    def test_higher_variance_can_increase_ei(self):
        val_low_var = cost_aware_ei(mean=0.5, variance=0.01, best_y=0.0, cost=1.0)
        val_high_var = cost_aware_ei(mean=0.5, variance=1.0, best_y=0.0, cost=1.0)
        assert val_high_var > val_low_var

    def test_xi_parameter_effect(self):
        val_no_xi = cost_aware_ei(mean=0.0, variance=0.25, best_y=0.0, cost=1.0, xi=0.0)
        val_xi = cost_aware_ei(mean=0.0, variance=0.25, best_y=0.0, cost=1.0, xi=0.5)
        # xi shifts the threshold, potentially reducing EI
        assert isinstance(val_no_xi, float)
        assert isinstance(val_xi, float)

    def test_nonnegative(self):
        val = cost_aware_ei(mean=10.0, variance=0.01, best_y=0.0, cost=1.0)
        assert val >= 0.0


# ── Fidelity Weighted EI Tests ───────────────────────────────────────


class TestFidelityWeightedEI:
    """Test fidelity-weighted EI."""

    def test_positive_value(self):
        val = fidelity_weighted_ei(
            mean=0.5, variance=0.25, best_y=0.0,
            fidelity=1.0, cost=1.0,
        )
        assert val > 0

    def test_higher_fidelity_higher_value(self):
        val_low = fidelity_weighted_ei(
            mean=0.5, variance=0.25, best_y=0.0,
            fidelity=0.3, cost=1.0,
        )
        val_high = fidelity_weighted_ei(
            mean=0.5, variance=0.25, best_y=0.0,
            fidelity=1.0, cost=1.0,
        )
        assert val_high > val_low

    def test_zero_fidelity_returns_zero(self):
        val = fidelity_weighted_ei(
            mean=0.5, variance=0.25, best_y=0.0,
            fidelity=0.0, cost=1.0,
        )
        assert val == 0.0


# ── Multi-Fidelity Knowledge Gradient Tests ──────────────────────────


class TestMultiFidelityKnowledgeGradient:
    """Test multi-fidelity knowledge gradient."""

    def test_returns_tuple(self):
        result = multi_fidelity_knowledge_gradient(
            means_by_fidelity=[0.5, 0.3],
            variances_by_fidelity=[0.25, 0.1],
            costs=[1.0, 10.0],
            current_best=0.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_selects_best_fidelity(self):
        # Cheap fidelity with same stats should be preferred
        value, idx = multi_fidelity_knowledge_gradient(
            means_by_fidelity=[0.5, 0.5],
            variances_by_fidelity=[0.25, 0.25],
            costs=[1.0, 100.0],
            current_best=0.0,
        )
        assert idx == 0  # Cheaper one

    def test_nonnegative_value(self):
        value, idx = multi_fidelity_knowledge_gradient(
            means_by_fidelity=[0.5],
            variances_by_fidelity=[0.25],
            costs=[1.0],
            current_best=0.0,
        )
        assert value >= 0

    def test_empty_returns_zero(self):
        value, idx = multi_fidelity_knowledge_gradient(
            means_by_fidelity=[],
            variances_by_fidelity=[],
            costs=[],
            current_best=0.0,
        )
        assert value == 0.0

    def test_high_variance_preferred(self):
        # Higher variance means more to learn => higher KG per cost if equal cost
        value, idx = multi_fidelity_knowledge_gradient(
            means_by_fidelity=[0.5, 0.5],
            variances_by_fidelity=[0.01, 1.0],
            costs=[1.0, 1.0],
            current_best=0.0,
        )
        assert idx == 1  # Higher variance

    def test_zero_cost_fidelity_skipped(self):
        value, idx = multi_fidelity_knowledge_gradient(
            means_by_fidelity=[0.5, 0.3],
            variances_by_fidelity=[0.25, 0.25],
            costs=[0.0, 1.0],
            current_best=0.0,
        )
        assert idx == 1  # Skip zero-cost

    def test_zero_variance_fidelity_skipped(self):
        value, idx = multi_fidelity_knowledge_gradient(
            means_by_fidelity=[0.5, 0.3],
            variances_by_fidelity=[0.0, 0.25],
            costs=[1.0, 1.0],
            current_best=0.0,
        )
        assert idx == 1


# ── Entropy Search Tests ─────────────────────────────────────────────


class TestEntropySearch:
    """Test entropy search multi-fidelity."""

    def test_returns_tuple(self):
        result = entropy_search_multi_fidelity(
            means=[0.5, 0.3],
            variances=[0.25, 0.1],
            costs=[1.0, 10.0],
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_prefers_cheap_high_variance(self):
        value, idx = entropy_search_multi_fidelity(
            means=[0.5, 0.5],
            variances=[1.0, 1.0],
            costs=[1.0, 100.0],
        )
        assert idx == 0  # Cheaper

    def test_nonnegative(self):
        value, idx = entropy_search_multi_fidelity(
            means=[0.5],
            variances=[0.25],
            costs=[1.0],
        )
        assert value >= 0

    def test_empty_returns_zero(self):
        value, idx = entropy_search_multi_fidelity(
            means=[], variances=[], costs=[],
        )
        assert value == 0.0

    def test_higher_variance_more_info(self):
        val_lo, _ = entropy_search_multi_fidelity(
            means=[0.5], variances=[0.01], costs=[1.0],
        )
        val_hi, _ = entropy_search_multi_fidelity(
            means=[0.5], variances=[1.0], costs=[1.0],
        )
        assert val_hi > val_lo


# ── MultiFidelityBackend: fit, suggest, predict, capabilities ────────


class TestMultiFidelityBackendFitSuggest:
    """Test MultiFidelityBackend fit/suggest/predict cycle."""

    def test_fit_with_standard_observations(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        obs = _make_observations(5)
        specs = _default_specs()
        backend.fit(obs, specs)
        # Should not raise

    def test_suggest_after_fit(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        obs = _make_observations(5)
        specs = _default_specs()
        backend.fit(obs, specs)
        suggestions = backend.suggest(n_suggestions=2)
        assert len(suggestions) == 2
        for s in suggestions:
            assert "x1" in s
            assert "x2" in s

    def test_suggest_without_fit_returns_random(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        backend._parameter_specs = _default_specs()
        suggestions = backend.suggest(n_suggestions=3)
        assert len(suggestions) == 3

    def test_predict_before_fit(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        means, variances = backend.predict([[0.5, 0.5]])
        assert len(means) == 1
        assert len(variances) == 1
        assert variances[0] > 0  # Prior variance

    def test_predict_after_fit(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        obs = _make_observations(5)
        specs = _default_specs()
        backend.fit(obs, specs)
        means, variances = backend.predict([[0.5, 0.5]])
        assert len(means) == 1
        assert math.isfinite(means[0])
        assert variances[0] > 0

    def test_predict_at_different_fidelities(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        obs = _make_observations(5)
        specs = _default_specs()
        backend.fit(obs, specs)
        m_high, v_high = backend.predict([[0.5, 0.5]], fidelity="high")
        m_low, v_low = backend.predict([[0.5, 0.5]], fidelity="low")
        # Both should produce valid results
        assert math.isfinite(m_high[0])
        assert math.isfinite(m_low[0])

    def test_suggest_returns_within_bounds(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        obs = _make_observations(5)
        specs = _default_specs()
        backend.fit(obs, specs)
        suggestions = backend.suggest(n_suggestions=5, seed=123)
        for s in suggestions:
            assert 0.0 <= s["x1"] <= 1.0
            assert 0.0 <= s["x2"] <= 1.0

    def test_capabilities_correct(self):
        config = _default_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        caps = backend.capabilities()
        assert caps["supports_continuous"] is True
        assert caps["supports_multi_fidelity"] is True
        assert caps["n_fidelity_levels"] == 3

    def test_fit_with_mf_observations(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        backend._parameter_specs = _default_specs()

        # Add observations at different fidelities
        for i in range(3):
            backend.add_mf_observation(MFObservation(
                parameters={"x1": i * 0.3, "x2": 0.5},
                kpi_value=float(i),
                fidelity_level="low",
                cost=1.0,
            ))
        for i in range(2):
            backend.add_mf_observation(MFObservation(
                parameters={"x1": i * 0.5, "x2": 0.5},
                kpi_value=float(i) * 0.9,
                fidelity_level="high",
                cost=10.0,
            ))

        means, variances = backend.predict([[0.3, 0.5]])
        assert math.isfinite(means[0])

    def test_predict_multiple_points(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        obs = _make_observations(5)
        specs = _default_specs()
        backend.fit(obs, specs)
        X_test = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]
        means, variances = backend.predict(X_test)
        assert len(means) == 3
        assert len(variances) == 3
        for m, v in zip(means, variances):
            assert math.isfinite(m)
            assert v > 0


# ── suggest_with_fidelity Tests ──────────────────────────────────────


class TestSuggestWithFidelity:
    """Test fidelity-aware suggestion."""

    def test_returns_tuples(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        backend._parameter_specs = _default_specs()

        for i in range(3):
            backend.add_mf_observation(MFObservation(
                parameters={"x1": i * 0.3, "x2": 0.5},
                kpi_value=float(i),
                fidelity_level="low",
                cost=1.0,
            ))

        results = backend.suggest_with_fidelity(n_suggestions=2)
        assert len(results) == 2
        for params, fidelity in results:
            assert isinstance(params, dict)
            assert fidelity in ("low", "high")

    def test_fidelity_selection_valid(self):
        config = _default_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        backend._parameter_specs = _default_specs()

        for i in range(5):
            backend.add_mf_observation(MFObservation(
                parameters={"x1": i * 0.2, "x2": 0.5},
                kpi_value=(i * 0.2 - 0.5) ** 2,
                fidelity_level="low",
                cost=1.0,
            ))

        results = backend.suggest_with_fidelity(n_suggestions=3)
        valid_levels = {"low", "medium", "high"}
        for _, fidelity in results:
            assert fidelity in valid_levels

    def test_budget_respects_can_afford(self):
        config = FidelityConfig(
            levels=[
                FidelityLevel(name="low", fidelity=0.3, cost=1.0),
                FidelityLevel(name="high", fidelity=1.0, cost=100.0),
            ],
            target_fidelity="high",
            cost_budget=5.0,
        )
        backend = MultiFidelityBackend(fidelity_config=config)
        backend._parameter_specs = _default_specs()

        for i in range(3):
            backend.add_mf_observation(MFObservation(
                parameters={"x1": i * 0.3, "x2": 0.5},
                kpi_value=float(i),
                fidelity_level="low",
                cost=1.0,
            ))

        # Budget is 5.0, spent 3.0 already; high costs 100 so cannot afford
        results = backend.suggest_with_fidelity(n_suggestions=1)
        _, fidelity = results[0]
        assert fidelity == "low"  # Can only afford low

    def test_empty_specs_fallback(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        results = backend.suggest_with_fidelity(n_suggestions=1)
        assert len(results) == 1

    def test_suggest_with_fidelity_deterministic(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config, seed=42)
        backend._parameter_specs = _default_specs()

        for i in range(3):
            backend.add_mf_observation(MFObservation(
                parameters={"x1": i * 0.3, "x2": 0.5},
                kpi_value=float(i),
                fidelity_level="low",
                cost=1.0,
            ))

        r1 = backend.suggest_with_fidelity(n_suggestions=1)
        # Create a fresh backend with same data
        backend2 = MultiFidelityBackend(fidelity_config=_two_level_config(), seed=42)
        backend2._parameter_specs = _default_specs()
        for i in range(3):
            backend2.add_mf_observation(MFObservation(
                parameters={"x1": i * 0.3, "x2": 0.5},
                kpi_value=float(i),
                fidelity_level="low",
                cost=1.0,
            ))
        r2 = backend2.suggest_with_fidelity(n_suggestions=1)
        assert r1[0][1] == r2[0][1]  # Same fidelity selection


# ── Cost Tracking and Budget Management ──────────────────────────────


class TestCostTracking:
    """Test cost tracking in the backend."""

    def test_cost_report_structure(self):
        config = _default_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        report = backend.get_cost_report()
        assert "budget" in report
        assert "spent" in report
        assert "remaining" in report
        assert "per_level" in report

    def test_cost_tracks_observations(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        backend._parameter_specs = _default_specs()

        backend.add_mf_observation(MFObservation(
            parameters={"x1": 0.5, "x2": 0.5},
            kpi_value=1.0,
            fidelity_level="low",
            cost=1.0,
        ))
        backend.add_mf_observation(MFObservation(
            parameters={"x1": 0.3, "x2": 0.7},
            kpi_value=0.5,
            fidelity_level="high",
            cost=10.0,
        ))

        report = backend.get_cost_report()
        assert report["spent"] == pytest.approx(11.0)
        assert report["remaining"] == pytest.approx(39.0)

    def test_per_level_breakdown(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        backend._parameter_specs = _default_specs()

        for i in range(3):
            backend.add_mf_observation(MFObservation(
                parameters={"x1": i * 0.3, "x2": 0.5},
                kpi_value=float(i),
                fidelity_level="low",
                cost=1.0,
            ))
        backend.add_mf_observation(MFObservation(
            parameters={"x1": 0.5, "x2": 0.5},
            kpi_value=0.5,
            fidelity_level="high",
            cost=10.0,
        ))

        report = backend.get_cost_report()
        assert report["per_level"]["low"]["n_evaluations"] == 3
        assert report["per_level"]["low"]["total_cost"] == pytest.approx(3.0)
        assert report["per_level"]["high"]["n_evaluations"] == 1
        assert report["per_level"]["high"]["total_cost"] == pytest.approx(10.0)

    def test_budget_initial_value(self):
        config = _default_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        report = backend.get_cost_report()
        assert report["budget"] == 100.0

    def test_cost_accumulates_correctly(self):
        config = _two_level_config()
        backend = MultiFidelityBackend(fidelity_config=config)
        backend._parameter_specs = _default_specs()

        total = 0.0
        for i in range(5):
            cost = 1.0 if i < 3 else 10.0
            fid = "low" if i < 3 else "high"
            backend.add_mf_observation(MFObservation(
                parameters={"x1": i * 0.2, "x2": 0.5},
                kpi_value=float(i),
                fidelity_level=fid,
                cost=cost,
            ))
            total += cost

        assert backend.get_cost_report()["spent"] == pytest.approx(total)


# ── Diagnostics Tests ────────────────────────────────────────────────


class TestDiagnostics:
    """Test cross-fidelity diagnostics."""

    def test_cross_fidelity_correlation_self(self):
        obs = {
            "low": [([0.1], 1.0), ([0.5], 2.0), ([0.9], 3.0)],
            "high": [([0.1], 1.1), ([0.5], 2.1), ([0.9], 3.1)],
        }
        corr = cross_fidelity_correlation(obs)
        assert corr[("low", "low")] == 1.0
        assert corr[("high", "high")] == 1.0

    def test_cross_fidelity_correlation_matched(self):
        obs = {
            "low": [([0.1], 1.0), ([0.5], 2.0), ([0.9], 3.0)],
            "high": [([0.1], 1.5), ([0.5], 2.5), ([0.9], 3.5)],
        }
        corr = cross_fidelity_correlation(obs)
        # Perfect rank correlation (same ordering)
        assert corr[("low", "high")] == pytest.approx(1.0, abs=0.01)

    def test_render_fidelity_comparison_returns_svg(self):
        obs = {
            "low": [([0.1], 1.0), ([0.5], 2.0)],
            "high": [([0.2], 1.5), ([0.6], 2.5)],
        }
        svg = render_fidelity_comparison(obs, ["x1"])
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_cost_allocation_returns_svg(self):
        report = {
            "budget": 100.0,
            "spent": 30.0,
            "remaining": 70.0,
            "per_level": {
                "low": {"n_evaluations": 20, "total_cost": 20.0},
                "high": {"n_evaluations": 1, "total_cost": 10.0},
            },
        }
        svg = render_cost_allocation(report)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_render_empty_data(self):
        svg = render_fidelity_comparison({}, [])
        assert "<svg" in svg

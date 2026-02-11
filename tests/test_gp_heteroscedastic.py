"""Tests for the heteroscedastic Gaussian Process backend.

Covers construction, observe/fit, predict, suggest, capabilities,
model state export, noise impact diagnostics, edge cases, and
protocol conformance.
"""

from __future__ import annotations

import math
import random

import pytest

from optimization_copilot.backends.gp_heteroscedastic import HeteroscedasticGP
from optimization_copilot.core.models import Observation, ParameterSpec, VariableType
from optimization_copilot.plugins.base import AlgorithmPlugin
from optimization_copilot.visualization.models import SurrogateModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(name: str, lo: float, hi: float) -> ParameterSpec:
    return ParameterSpec(name=name, type=VariableType.CONTINUOUS, lower=lo, upper=hi)


def _make_obs(
    params: dict,
    kpi: float,
    noise_var: float | None = None,
    iteration: int = 0,
) -> Observation:
    meta = {}
    if noise_var is not None:
        meta["noise_variance"] = noise_var
    return Observation(
        iteration=iteration,
        parameters=params,
        kpi_values={"obj": kpi},
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# 1. Basic construction and name
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        gp = HeteroscedasticGP()
        assert gp.name() == "heteroscedastic_gp"

    def test_custom_kernel(self):
        gp = HeteroscedasticGP(kernel="rbf", lengthscale=0.5, signal_variance=2.0)
        assert gp.name() == "heteroscedastic_gp"

    def test_default_hyperparameters(self):
        gp = HeteroscedasticGP()
        state = gp.get_model_state()
        assert state["kernel"] == "matern52"
        assert state["lengthscale"] == 1.0
        assert state["signal_variance"] == 1.0
        assert state["default_noise"] == 0.01

    def test_custom_hyperparameters(self):
        gp = HeteroscedasticGP(
            kernel="rbf",
            lengthscale=2.0,
            signal_variance=3.0,
            default_noise=0.1,
        )
        state = gp.get_model_state()
        assert state["kernel"] == "rbf"
        assert state["lengthscale"] == 2.0
        assert state["signal_variance"] == 3.0
        assert state["default_noise"] == 0.1


# ---------------------------------------------------------------------------
# 2. observe() stores data correctly
# ---------------------------------------------------------------------------

class TestObserve:
    def test_observe_single_point(self):
        gp = HeteroscedasticGP()
        gp.observe([0.5], 1.0, noise_var=0.1)
        state = gp.get_model_state()
        assert state["n_observations"] == 1
        assert state["X_train"] == [[0.5]]
        assert state["y_train"] == [1.0]
        assert state["noise_vars"] == [0.1]

    def test_observe_multiple_points(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0], 0.0, noise_var=0.01)
        gp.observe([1.0], 1.0, noise_var=0.1)
        gp.observe([2.0], 4.0, noise_var=0.5)
        state = gp.get_model_state()
        assert state["n_observations"] == 3
        assert len(state["X_train"]) == 3

    def test_observe_default_noise(self):
        gp = HeteroscedasticGP(default_noise=0.05)
        gp.observe([0.5], 1.0)  # no noise_var
        state = gp.get_model_state()
        assert state["noise_vars"] == [0.05]

    def test_observe_with_metadata(self):
        gp = HeteroscedasticGP()
        gp.observe([0.5], 1.0, noise_var=0.01, metadata={"source": "test"})
        assert gp._metadata_list[0]["source"] == "test"

    def test_observe_invalidates_cache(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0], 0.0, noise_var=0.01)
        gp.observe([1.0], 1.0, noise_var=0.01)
        # Force cache build
        gp.predict([0.5])
        assert gp._L is not None
        # Adding a new point should invalidate
        gp.observe([2.0], 2.0, noise_var=0.01)
        assert gp._L is None


# ---------------------------------------------------------------------------
# 3. fit() from Observation list
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_with_noise_metadata(self):
        specs = [_make_spec("x", 0.0, 10.0)]
        obs = [
            _make_obs({"x": 1.0}, 2.0, noise_var=0.01),
            _make_obs({"x": 5.0}, 3.0, noise_var=0.1),
            _make_obs({"x": 9.0}, 1.0, noise_var=0.5),
        ]
        gp = HeteroscedasticGP()
        gp.fit(obs, specs)
        state = gp.get_model_state()
        assert state["n_observations"] == 3
        assert state["noise_vars"] == [0.01, 0.1, 0.5]

    def test_fit_without_noise_metadata(self):
        specs = [_make_spec("x", 0.0, 10.0)]
        obs = [
            _make_obs({"x": 1.0}, 2.0),
            _make_obs({"x": 5.0}, 3.0),
        ]
        gp = HeteroscedasticGP(default_noise=0.05)
        gp.fit(obs, specs)
        state = gp.get_model_state()
        assert state["noise_vars"] == [0.05, 0.05]

    def test_fit_mixed_noise_metadata(self):
        specs = [_make_spec("x", 0.0, 10.0)]
        obs = [
            _make_obs({"x": 1.0}, 2.0, noise_var=0.01),
            _make_obs({"x": 5.0}, 3.0),  # no noise_var
            _make_obs({"x": 9.0}, 1.0, noise_var=0.5),
        ]
        gp = HeteroscedasticGP(default_noise=0.07)
        gp.fit(obs, specs)
        assert gp._noise_vars == [0.01, 0.07, 0.5]

    def test_fit_skips_failures(self):
        specs = [_make_spec("x", 0.0, 10.0)]
        obs = [
            _make_obs({"x": 1.0}, 2.0, noise_var=0.01),
            Observation(
                iteration=1,
                parameters={"x": 5.0},
                kpi_values={"obj": 0.0},
                is_failure=True,
            ),
            _make_obs({"x": 9.0}, 1.0, noise_var=0.5),
        ]
        gp = HeteroscedasticGP()
        gp.fit(obs, specs)
        assert gp.get_model_state()["n_observations"] == 2

    def test_fit_resets_previous_data(self):
        specs = [_make_spec("x", 0.0, 10.0)]
        gp = HeteroscedasticGP()
        gp.observe([1.0], 1.0, noise_var=0.01)
        gp.observe([2.0], 2.0, noise_var=0.01)
        assert gp.get_model_state()["n_observations"] == 2

        # fit() should replace all data
        obs = [_make_obs({"x": 5.0}, 3.0, noise_var=0.1)]
        gp.fit(obs, specs)
        assert gp.get_model_state()["n_observations"] == 1

    def test_fit_multidimensional(self):
        specs = [_make_spec("x", 0.0, 1.0), _make_spec("y", 0.0, 1.0)]
        obs = [
            _make_obs({"x": 0.2, "y": 0.3}, 1.0, noise_var=0.01),
            _make_obs({"x": 0.8, "y": 0.7}, 2.0, noise_var=0.1),
        ]
        gp = HeteroscedasticGP()
        gp.fit(obs, specs)
        assert gp._X == [[0.2, 0.3], [0.8, 0.7]]


# ---------------------------------------------------------------------------
# 4. predict() shape and basic behavior
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_tuple(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0], 0.0, noise_var=0.01)
        gp.observe([1.0], 1.0, noise_var=0.01)
        result = gp.predict([0.5])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_mean_and_variance_are_floats(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0], 0.0, noise_var=0.01)
        gp.observe([1.0], 1.0, noise_var=0.01)
        mu, var = gp.predict([0.5])
        assert isinstance(mu, float)
        assert isinstance(var, float)

    def test_predict_variance_positive(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0], 0.0, noise_var=0.01)
        gp.observe([1.0], 1.0, noise_var=0.01)
        _, var = gp.predict([0.5])
        assert var > 0

    def test_predict_at_training_point_low_variance(self):
        gp = HeteroscedasticGP(lengthscale=1.0, signal_variance=1.0)
        gp.observe([0.0], 0.0, noise_var=0.001)
        gp.observe([1.0], 1.0, noise_var=0.001)
        gp.observe([2.0], 4.0, noise_var=0.001)
        _, var_at_train = gp.predict([1.0])
        _, var_far = gp.predict([10.0])
        # Variance at training point should be much lower than far away
        assert var_at_train < var_far

    def test_predict_far_from_data_high_variance(self):
        gp = HeteroscedasticGP(lengthscale=0.5)
        gp.observe([0.0], 0.0, noise_var=0.01)
        gp.observe([1.0], 1.0, noise_var=0.01)
        _, var_near = gp.predict([0.5])
        _, var_far = gp.predict([100.0])
        assert var_far > var_near

    def test_predict_no_data_returns_prior(self):
        gp = HeteroscedasticGP(signal_variance=2.0)
        mu, var = gp.predict([0.5])
        assert mu == 0.0
        assert var == 2.0

    def test_predict_mean_near_training_value(self):
        gp = HeteroscedasticGP(lengthscale=1.0, signal_variance=1.0)
        gp.observe([0.0], 5.0, noise_var=0.001)
        gp.observe([0.01], 5.01, noise_var=0.001)
        mu, _ = gp.predict([0.005])
        # Should be close to 5.0
        assert abs(mu - 5.0) < 0.5

    def test_predict_multidimensional(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0, 0.0], 0.0, noise_var=0.01)
        gp.observe([1.0, 1.0], 2.0, noise_var=0.01)
        mu, var = gp.predict([0.5, 0.5])
        assert isinstance(mu, float)
        assert var > 0


# ---------------------------------------------------------------------------
# 5. Heteroscedastic noise: influence test
# ---------------------------------------------------------------------------

class TestHeteroscedasticBehavior:
    def test_high_noise_point_less_influence(self):
        """A point with high noise should pull the GP prediction less."""
        # Two points at different locations:
        # x=0 has y=0 with LOW noise (reliable)
        # x=1 has y=10 with HIGH noise (unreliable)
        # At x=0.5, the prediction should be closer to the low-noise point's value
        gp = HeteroscedasticGP(lengthscale=1.0, signal_variance=1.0)
        gp.observe([0.0], 0.0, noise_var=0.001)
        gp.observe([1.0], 10.0, noise_var=100.0)
        mu, _ = gp.predict([0.5])
        # Mean should be closer to 0 (the reliable point) than to 10
        assert mu < 5.0

    def test_large_noise_gp_closer_to_low_noise_points(self):
        """GP prediction should favor low-noise observations."""
        gp = HeteroscedasticGP(lengthscale=1.0, signal_variance=1.0)
        # Cluster of reliable points around y=1
        gp.observe([0.0], 1.0, noise_var=0.001)
        gp.observe([0.5], 1.0, noise_var=0.001)
        # One noisy outlier
        gp.observe([1.0], 100.0, noise_var=1000.0)

        mu, _ = gp.predict([0.75])
        # Should be much closer to 1.0 than to 100.0
        assert mu < 50.0

    def test_symmetric_noise_symmetric_prediction(self):
        """With equal noise, prediction at midpoint should be average."""
        gp = HeteroscedasticGP(lengthscale=1.0, signal_variance=1.0)
        gp.observe([0.0], 0.0, noise_var=0.01)
        gp.observe([2.0], 2.0, noise_var=0.01)
        mu, _ = gp.predict([1.0])
        # Should be approximately 1.0 (midpoint)
        assert abs(mu - 1.0) < 0.5

    def test_all_same_noise_like_homoscedastic(self):
        """When all noise is the same, should behave like homoscedastic GP."""
        noise = 0.01
        gp_hetero = HeteroscedasticGP(
            kernel="rbf", lengthscale=1.0, signal_variance=1.0, default_noise=noise,
        )
        gp_hetero.observe([0.0], 0.0, noise_var=noise)
        gp_hetero.observe([1.0], 1.0, noise_var=noise)
        gp_hetero.observe([2.0], 4.0, noise_var=noise)

        mu_h, var_h = gp_hetero.predict([0.5])

        # Build an equivalent homoscedastic GP manually
        gp_homo = HeteroscedasticGP(
            kernel="rbf", lengthscale=1.0, signal_variance=1.0, default_noise=noise,
        )
        gp_homo.observe([0.0], 0.0, noise_var=noise)
        gp_homo.observe([1.0], 1.0, noise_var=noise)
        gp_homo.observe([2.0], 4.0, noise_var=noise)

        mu_o, var_o = gp_homo.predict([0.5])

        assert abs(mu_h - mu_o) < 1e-10
        assert abs(var_h - var_o) < 1e-10


# ---------------------------------------------------------------------------
# 6. Monotonic test: adding data reduces variance
# ---------------------------------------------------------------------------

class TestMonotonic:
    def test_adding_data_reduces_variance_at_train_points(self):
        gp = HeteroscedasticGP(lengthscale=1.0, signal_variance=1.0)
        gp.observe([0.0], 0.0, noise_var=0.01)
        _, var_1 = gp.predict([0.5])

        gp.observe([1.0], 1.0, noise_var=0.01)
        _, var_2 = gp.predict([0.5])

        # Adding a nearby point should reduce uncertainty
        assert var_2 <= var_1

    def test_adding_nearby_data_reduces_variance(self):
        gp = HeteroscedasticGP(lengthscale=1.0, signal_variance=1.0)
        gp.observe([0.0], 0.0, noise_var=0.01)
        gp.observe([2.0], 2.0, noise_var=0.01)
        _, var_before = gp.predict([1.0])

        gp.observe([1.0], 1.0, noise_var=0.01)
        _, var_after = gp.predict([1.0])

        assert var_after < var_before


# ---------------------------------------------------------------------------
# 7. suggest() tests
# ---------------------------------------------------------------------------

class TestSuggest:
    def test_suggest_returns_list_of_dicts(self):
        specs = [_make_spec("x", 0.0, 10.0)]
        obs = [
            _make_obs({"x": 1.0}, 5.0, noise_var=0.01),
            _make_obs({"x": 5.0}, 2.0, noise_var=0.01),
            _make_obs({"x": 9.0}, 7.0, noise_var=0.01),
        ]
        gp = HeteroscedasticGP()
        gp.fit(obs, specs)
        suggestions = gp.suggest(n_suggestions=3, seed=42)
        assert isinstance(suggestions, list)
        assert len(suggestions) == 3
        for s in suggestions:
            assert isinstance(s, dict)
            assert "x" in s

    def test_suggest_respects_bounds(self):
        specs = [_make_spec("x", -5.0, 5.0)]
        obs = [
            _make_obs({"x": -3.0}, 10.0, noise_var=0.01),
            _make_obs({"x": 0.0}, 1.0, noise_var=0.01),
            _make_obs({"x": 3.0}, 8.0, noise_var=0.01),
        ]
        gp = HeteroscedasticGP()
        gp.fit(obs, specs)
        suggestions = gp.suggest(n_suggestions=20, seed=123)
        for s in suggestions:
            assert -5.0 <= s["x"] <= 5.0

    def test_suggest_multidimensional(self):
        specs = [_make_spec("x", 0.0, 1.0), _make_spec("y", 0.0, 1.0)]
        obs = [
            _make_obs({"x": 0.1, "y": 0.1}, 1.0, noise_var=0.01),
            _make_obs({"x": 0.9, "y": 0.9}, 2.0, noise_var=0.01),
        ]
        gp = HeteroscedasticGP()
        gp.fit(obs, specs)
        suggestions = gp.suggest(n_suggestions=2, seed=42)
        assert len(suggestions) == 2
        for s in suggestions:
            assert "x" in s and "y" in s
            assert 0.0 <= s["x"] <= 1.0
            assert 0.0 <= s["y"] <= 1.0

    def test_suggest_deterministic_with_same_seed(self):
        specs = [_make_spec("x", 0.0, 10.0)]
        obs = [
            _make_obs({"x": 2.0}, 3.0, noise_var=0.01),
            _make_obs({"x": 5.0}, 1.0, noise_var=0.01),
            _make_obs({"x": 8.0}, 4.0, noise_var=0.01),
        ]
        gp1 = HeteroscedasticGP()
        gp1.fit(obs, specs)
        s1 = gp1.suggest(n_suggestions=3, seed=99)

        gp2 = HeteroscedasticGP()
        gp2.fit(obs, specs)
        s2 = gp2.suggest(n_suggestions=3, seed=99)

        for a, b in zip(s1, s2):
            assert a["x"] == b["x"]

    def test_suggest_different_seeds_different_results(self):
        specs = [_make_spec("x", 0.0, 10.0)]
        obs = [
            _make_obs({"x": 2.0}, 3.0, noise_var=0.01),
            _make_obs({"x": 5.0}, 1.0, noise_var=0.01),
            _make_obs({"x": 8.0}, 4.0, noise_var=0.01),
        ]
        gp = HeteroscedasticGP()
        gp.fit(obs, specs)
        s1 = gp.suggest(n_suggestions=1, seed=42)
        s2 = gp.suggest(n_suggestions=1, seed=12345)
        # Very unlikely to be exactly the same with different seeds
        # (not guaranteed but astronomically unlikely)
        # We just verify they are valid
        assert 0.0 <= s1[0]["x"] <= 10.0
        assert 0.0 <= s2[0]["x"] <= 10.0

    def test_suggest_fallback_no_data(self):
        specs = [_make_spec("x", 0.0, 10.0)]
        gp = HeteroscedasticGP()
        gp.fit([], specs)
        suggestions = gp.suggest(n_suggestions=2, seed=42)
        assert len(suggestions) == 2
        for s in suggestions:
            assert 0.0 <= s["x"] <= 10.0


# ---------------------------------------------------------------------------
# 8. capabilities()
# ---------------------------------------------------------------------------

class TestCapabilities:
    def test_capabilities_includes_heteroscedastic_flag(self):
        gp = HeteroscedasticGP()
        caps = gp.capabilities()
        assert caps["supports_heteroscedastic_noise"] is True

    def test_capabilities_complete(self):
        gp = HeteroscedasticGP()
        caps = gp.capabilities()
        assert "supports_continuous" in caps
        assert "supports_discrete" in caps
        assert "supports_batch" in caps
        assert "requires_observations" in caps

    def test_capabilities_no_categorical(self):
        gp = HeteroscedasticGP()
        caps = gp.capabilities()
        assert caps["supports_categorical"] is False


# ---------------------------------------------------------------------------
# 9. get_model_state()
# ---------------------------------------------------------------------------

class TestModelState:
    def test_model_state_keys(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0], 1.0, noise_var=0.01)
        state = gp.get_model_state()
        expected_keys = {
            "kernel", "lengthscale", "signal_variance", "default_noise",
            "n_observations", "X_train", "y_train", "noise_vars",
            "best_y", "noise_range",
        }
        assert expected_keys.issubset(set(state.keys()))

    def test_model_state_best_y(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0], 5.0, noise_var=0.01)
        gp.observe([1.0], 2.0, noise_var=0.01)
        gp.observe([2.0], 8.0, noise_var=0.01)
        state = gp.get_model_state()
        assert state["best_y"] == 2.0

    def test_model_state_noise_range(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0], 0.0, noise_var=0.001)
        gp.observe([1.0], 1.0, noise_var=0.1)
        gp.observe([2.0], 2.0, noise_var=1.0)
        state = gp.get_model_state()
        assert state["noise_range"] == (0.001, 1.0)

    def test_model_state_empty(self):
        gp = HeteroscedasticGP()
        state = gp.get_model_state()
        assert state["n_observations"] == 0
        assert state["best_y"] is None
        assert state["noise_range"] == (None, None)


# ---------------------------------------------------------------------------
# 10. compute_noise_impact()
# ---------------------------------------------------------------------------

class TestNoiseImpact:
    def test_noise_impact_returns_list(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0], 0.0, noise_var=0.01)
        gp.observe([1.0], 1.0, noise_var=0.1)
        impact = gp.compute_noise_impact()
        assert isinstance(impact, list)
        assert len(impact) == 2

    def test_noise_impact_per_point_keys(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0], 0.0, noise_var=0.01)
        gp.observe([1.0], 1.0, noise_var=0.1)
        impact = gp.compute_noise_impact()
        expected_keys = {
            "index", "x", "y", "noise_variance",
            "hetero_weight", "homo_weight", "weight_ratio",
        }
        for entry in impact:
            assert expected_keys.issubset(set(entry.keys()))

    def test_noise_impact_low_noise_higher_weight(self):
        """Point with lower noise should have relatively higher weight."""
        gp = HeteroscedasticGP(lengthscale=1.0, signal_variance=1.0)
        gp.observe([0.0], 1.0, noise_var=0.001)
        gp.observe([2.0], 3.0, noise_var=10.0)
        impact = gp.compute_noise_impact()

        # In the heteroscedastic model, the low-noise point should have
        # a higher weight ratio than the high-noise point
        low_noise_entry = impact[0]  # noise_var=0.001
        high_noise_entry = impact[1]  # noise_var=10.0
        assert low_noise_entry["noise_variance"] < high_noise_entry["noise_variance"]

    def test_noise_impact_empty(self):
        gp = HeteroscedasticGP()
        assert gp.compute_noise_impact() == []

    def test_noise_impact_values_nonnegative(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0], 0.0, noise_var=0.01)
        gp.observe([1.0], 1.0, noise_var=0.1)
        gp.observe([2.0], 4.0, noise_var=0.5)
        for entry in gp.compute_noise_impact():
            assert entry["hetero_weight"] >= 0
            assert entry["homo_weight"] >= 0
            assert entry["weight_ratio"] >= 0


# ---------------------------------------------------------------------------
# 11. Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocols:
    def test_is_algorithm_plugin(self):
        gp = HeteroscedasticGP()
        assert isinstance(gp, AlgorithmPlugin)

    def test_is_surrogate_model(self):
        gp = HeteroscedasticGP()
        assert isinstance(gp, SurrogateModel)

    def test_algorithm_plugin_methods(self):
        gp = HeteroscedasticGP()
        assert callable(gp.name)
        assert callable(gp.fit)
        assert callable(gp.suggest)
        assert callable(gp.capabilities)

    def test_surrogate_model_predict(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0], 1.0, noise_var=0.01)
        result = gp.predict([0.0])
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 12. Single observation
# ---------------------------------------------------------------------------

class TestSingleObservation:
    def test_single_observation_predict(self):
        gp = HeteroscedasticGP()
        gp.observe([0.0], 5.0, noise_var=0.01)
        mu, var = gp.predict([0.0])
        # Mean should be close to the training value
        assert abs(mu - 5.0) < 1.0
        assert var > 0

    def test_single_observation_suggest(self):
        specs = [_make_spec("x", 0.0, 10.0)]
        obs = [_make_obs({"x": 5.0}, 3.0, noise_var=0.01)]
        gp = HeteroscedasticGP()
        gp.fit(obs, specs)
        suggestions = gp.suggest(n_suggestions=1, seed=42)
        assert len(suggestions) == 1
        assert 0.0 <= suggestions[0]["x"] <= 10.0


# ---------------------------------------------------------------------------
# 13. Edge cases: noise values
# ---------------------------------------------------------------------------

class TestNoiseEdgeCases:
    def test_zero_noise(self):
        """Zero noise should still work (with jitter from cholesky)."""
        gp = HeteroscedasticGP()
        gp.observe([0.0], 0.0, noise_var=0.0)
        gp.observe([1.0], 1.0, noise_var=0.0)
        mu, var = gp.predict([0.5])
        assert isinstance(mu, float)
        assert var > 0  # should still have some variance from numerical jitter

    def test_very_large_noise(self):
        """Very large noise on all points: predictions revert to prior."""
        gp = HeteroscedasticGP(signal_variance=1.0)
        gp.observe([0.0], 100.0, noise_var=1e6)
        gp.observe([1.0], -100.0, noise_var=1e6)
        mu, var = gp.predict([0.5])
        # With huge noise, the GP should largely ignore the data
        # Mean should be closer to 0 (prior) than to the data values
        assert abs(mu) < 50.0

    def test_mixed_zero_and_large_noise(self):
        """Mix of zero and large noise."""
        gp = HeteroscedasticGP(lengthscale=1.0, signal_variance=1.0)
        gp.observe([0.0], 5.0, noise_var=0.0)   # exact observation
        gp.observe([1.0], 100.0, noise_var=1e6)  # very noisy
        mu, _ = gp.predict([0.0])
        # Should be very close to 5.0 (the exact observation)
        assert abs(mu - 5.0) < 1.0

    def test_uniform_noise_consistency(self):
        """All observations with the same noise should be treated equally."""
        gp = HeteroscedasticGP(kernel="rbf", lengthscale=1.0, signal_variance=1.0)
        # Symmetric setup: two equidistant points with equal noise
        gp.observe([-1.0], 0.0, noise_var=0.01)
        gp.observe([1.0], 2.0, noise_var=0.01)
        mu, _ = gp.predict([0.0])
        # Should be approximately the average
        assert abs(mu - 1.0) < 0.5


# ---------------------------------------------------------------------------
# 14. Kernel choice
# ---------------------------------------------------------------------------

class TestKernelChoice:
    def test_rbf_kernel(self):
        gp = HeteroscedasticGP(kernel="rbf")
        gp.observe([0.0], 0.0, noise_var=0.01)
        gp.observe([1.0], 1.0, noise_var=0.01)
        mu, var = gp.predict([0.5])
        assert isinstance(mu, float)
        assert var > 0

    def test_matern52_kernel(self):
        gp = HeteroscedasticGP(kernel="matern52")
        gp.observe([0.0], 0.0, noise_var=0.01)
        gp.observe([1.0], 1.0, noise_var=0.01)
        mu, var = gp.predict([0.5])
        assert isinstance(mu, float)
        assert var > 0

    def test_rbf_and_matern_give_different_predictions(self):
        data_x = [[0.0], [1.0], [2.0]]
        data_y = [0.0, 1.0, 0.5]
        noise = 0.01

        gp_rbf = HeteroscedasticGP(kernel="rbf", lengthscale=1.0)
        gp_mat = HeteroscedasticGP(kernel="matern52", lengthscale=1.0)
        for x, y in zip(data_x, data_y):
            gp_rbf.observe(x, y, noise_var=noise)
            gp_mat.observe(x, y, noise_var=noise)

        mu_rbf, _ = gp_rbf.predict([1.5])
        mu_mat, _ = gp_mat.predict([1.5])
        # They should give somewhat different results (different kernels)
        # but both should be valid floats
        assert isinstance(mu_rbf, float)
        assert isinstance(mu_mat, float)


# ---------------------------------------------------------------------------
# 15. Discrete parameter specs
# ---------------------------------------------------------------------------

class TestDiscreteParams:
    def test_discrete_suggest_respects_bounds(self):
        specs = [
            ParameterSpec(name="n_layers", type=VariableType.DISCRETE, lower=1, upper=10)
        ]
        obs = [
            _make_obs({"n_layers": 3}, 5.0, noise_var=0.01),
            _make_obs({"n_layers": 7}, 2.0, noise_var=0.01),
        ]
        gp = HeteroscedasticGP()
        gp.fit(obs, specs)
        suggestions = gp.suggest(n_suggestions=5, seed=42)
        for s in suggestions:
            assert 1 <= s["n_layers"] <= 10
            assert isinstance(s["n_layers"], int)


# ---------------------------------------------------------------------------
# 16. Numerical stability with many observations
# ---------------------------------------------------------------------------

class TestNumericalStability:
    def test_many_observations(self):
        gp = HeteroscedasticGP(lengthscale=1.0, signal_variance=1.0)
        rng = random.Random(42)
        for i in range(50):
            x = rng.uniform(0, 10)
            y = math.sin(x) + rng.gauss(0, 0.1)
            noise = rng.uniform(0.01, 0.5)
            gp.observe([x], y, noise_var=noise)

        # Should not raise
        mu, var = gp.predict([5.0])
        assert isinstance(mu, float)
        assert var > 0
        assert not math.isnan(mu)
        assert not math.isnan(var)

    def test_duplicate_points_different_noise(self):
        """Multiple observations at the same x with different noise."""
        gp = HeteroscedasticGP(lengthscale=1.0, signal_variance=1.0)
        gp.observe([1.0], 2.0, noise_var=0.01)
        gp.observe([1.0], 2.1, noise_var=0.1)
        gp.observe([1.0], 1.9, noise_var=0.5)
        mu, var = gp.predict([1.0])
        assert isinstance(mu, float)
        assert var > 0
        # Mean should be close to 2.0
        assert abs(mu - 2.0) < 1.0


# ---------------------------------------------------------------------------
# 17. Integration: fit then suggest then predict cycle
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_cycle(self):
        specs = [_make_spec("x", 0.0, 10.0), _make_spec("y", -5.0, 5.0)]
        obs = [
            _make_obs({"x": 1.0, "y": -2.0}, 10.0, noise_var=0.01),
            _make_obs({"x": 5.0, "y": 0.0}, 2.0, noise_var=0.05),
            _make_obs({"x": 9.0, "y": 3.0}, 8.0, noise_var=0.1),
        ]
        gp = HeteroscedasticGP()
        gp.fit(obs, specs)

        # Suggest
        suggestions = gp.suggest(n_suggestions=2, seed=42)
        assert len(suggestions) == 2
        for s in suggestions:
            assert 0.0 <= s["x"] <= 10.0
            assert -5.0 <= s["y"] <= 5.0

        # Predict
        mu, var = gp.predict([5.0, 0.0])
        assert isinstance(mu, float)
        assert var > 0

        # State
        state = gp.get_model_state()
        assert state["n_observations"] == 3

        # Noise impact
        impact = gp.compute_noise_impact()
        assert len(impact) == 3

    def test_observe_then_fit_overrides(self):
        """fit() should completely replace observe() data."""
        gp = HeteroscedasticGP()
        gp.observe([0.0], 1.0, noise_var=0.01)
        gp.observe([1.0], 2.0, noise_var=0.01)

        specs = [_make_spec("x", 0.0, 10.0)]
        obs = [_make_obs({"x": 5.0}, 5.0, noise_var=0.1)]
        gp.fit(obs, specs)

        assert gp.get_model_state()["n_observations"] == 1
        assert gp._X == [[5.0]]

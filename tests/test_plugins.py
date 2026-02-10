"""Tests for the plugin architecture and built-in backends."""

from __future__ import annotations

import pytest

from optimization_copilot.core import Observation, ParameterSpec, VariableType
from optimization_copilot.plugins.base import AlgorithmPlugin
from optimization_copilot.plugins.registry import BackendPolicy, PluginRegistry
from optimization_copilot.backends.builtin import (
    LatinHypercubeSampler,
    RandomSampler,
    TPESampler,
)


# ── helpers ───────────────────────────────────────────────────────────

CONTINUOUS_SPECS = [
    ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
    ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
]

MIXED_SPECS = [
    ParameterSpec(name="lr", type=VariableType.CONTINUOUS, lower=1e-4, upper=1.0),
    ParameterSpec(name="n_layers", type=VariableType.DISCRETE, lower=1, upper=10),
    ParameterSpec(name="activation", type=VariableType.CATEGORICAL, categories=["relu", "tanh", "sigmoid"]),
]


def _make_observations(specs: list[ParameterSpec], n: int = 20) -> list[Observation]:
    """Generate synthetic observations for testing."""
    import random as _rng

    r = _rng.Random(99)
    obs: list[Observation] = []
    for i in range(n):
        params: dict = {}
        for s in specs:
            if s.type == VariableType.CATEGORICAL:
                params[s.name] = r.choice(s.categories)
            elif s.type == VariableType.DISCRETE:
                params[s.name] = r.randint(int(s.lower), int(s.upper))
            else:
                params[s.name] = r.uniform(s.lower, s.upper)
        kpi = sum(v for v in params.values() if isinstance(v, (int, float)))
        obs.append(
            Observation(
                iteration=i,
                parameters=params,
                kpi_values={"objective": kpi},
                timestamp=float(i),
            )
        )
    return obs


def _assert_within_bounds(
    suggestions: list[dict],
    specs: list[ParameterSpec],
) -> None:
    """Verify every suggestion respects the parameter bounds."""
    for point in suggestions:
        for spec in specs:
            val = point[spec.name]
            if spec.type == VariableType.CATEGORICAL:
                assert val in spec.categories, (
                    f"{spec.name}: {val!r} not in {spec.categories}"
                )
            elif spec.type == VariableType.DISCRETE:
                assert isinstance(val, int), f"{spec.name}: expected int, got {type(val)}"
                assert int(spec.lower) <= val <= int(spec.upper), (
                    f"{spec.name}: {val} out of [{spec.lower}, {spec.upper}]"
                )
            else:
                assert spec.lower <= val <= spec.upper, (
                    f"{spec.name}: {val} out of [{spec.lower}, {spec.upper}]"
                )


# ── Registry tests ────────────────────────────────────────────────────

class TestPluginRegistry:
    def test_register_and_list(self):
        reg = PluginRegistry()
        reg.register(RandomSampler)
        reg.register(LatinHypercubeSampler)
        names = reg.list_plugins()
        assert "random_sampler" in names
        assert "latin_hypercube_sampler" in names

    def test_get_returns_instance(self):
        reg = PluginRegistry()
        reg.register(RandomSampler)
        plugin = reg.get("random_sampler")
        assert isinstance(plugin, AlgorithmPlugin)
        assert isinstance(plugin, RandomSampler)

    def test_get_unknown_raises(self):
        reg = PluginRegistry()
        with pytest.raises(KeyError, match="Unknown plugin"):
            reg.get("nonexistent")

    def test_duplicate_register_raises(self):
        reg = PluginRegistry()
        reg.register(RandomSampler)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(RandomSampler)

    def test_register_non_plugin_raises(self):
        reg = PluginRegistry()
        with pytest.raises(TypeError, match="subclass of AlgorithmPlugin"):
            reg.register(dict)  # type: ignore[arg-type]

    def test_all_builtins_register(self):
        reg = PluginRegistry()
        for cls in [RandomSampler, LatinHypercubeSampler, TPESampler]:
            reg.register(cls)
        assert len(reg.list_plugins()) == 3


class TestBackendPolicy:
    def test_allowlist(self):
        policy = BackendPolicy(allowlist=["random_sampler"])
        reg = PluginRegistry(policy=policy)
        reg.register(RandomSampler)
        reg.register(LatinHypercubeSampler)
        assert reg.list_plugins() == ["random_sampler"]

    def test_denylist(self):
        policy = BackendPolicy(denylist=["random_sampler"])
        reg = PluginRegistry(policy=policy)
        reg.register(RandomSampler)
        reg.register(LatinHypercubeSampler)
        assert "random_sampler" not in reg.list_plugins()
        assert "latin_hypercube_sampler" in reg.list_plugins()

    def test_denylist_blocks_get(self):
        policy = BackendPolicy(denylist=["random_sampler"])
        reg = PluginRegistry(policy=policy)
        reg.register(RandomSampler)
        with pytest.raises(PermissionError, match="blocked"):
            reg.get("random_sampler")


class TestCapabilityMatching:
    def test_match_requires_observations(self):
        reg = PluginRegistry()
        reg.register(RandomSampler)
        reg.register(TPESampler)
        # Only TPE requires observations.
        matches = reg.match_capabilities({"requires_observations": True})
        assert "tpe_sampler" in matches
        assert "random_sampler" not in matches

    def test_match_categorical_support(self):
        reg = PluginRegistry()
        reg.register(RandomSampler)
        reg.register(LatinHypercubeSampler)
        reg.register(TPESampler)
        matches = reg.match_capabilities({"supports_categorical": True})
        assert len(matches) == 3

    def test_match_no_requirements(self):
        reg = PluginRegistry()
        reg.register(RandomSampler)
        matches = reg.match_capabilities({})
        assert matches == ["random_sampler"]


# ── RandomSampler tests ──────────────────────────────────────────────

class TestRandomSampler:
    def test_name(self):
        assert RandomSampler().name() == "random_sampler"

    def test_suggest_within_bounds_continuous(self):
        sampler = RandomSampler()
        sampler.fit([], CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=0)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_suggest_within_bounds_mixed(self):
        sampler = RandomSampler()
        sampler.fit([], MIXED_SPECS)
        suggestions = sampler.suggest(n_suggestions=20, seed=7)
        _assert_within_bounds(suggestions, MIXED_SPECS)

    def test_deterministic_with_seed(self):
        sampler = RandomSampler()
        sampler.fit([], CONTINUOUS_SPECS)
        a = sampler.suggest(n_suggestions=5, seed=123)
        b = sampler.suggest(n_suggestions=5, seed=123)
        assert a == b

    def test_different_seeds_differ(self):
        sampler = RandomSampler()
        sampler.fit([], CONTINUOUS_SPECS)
        a = sampler.suggest(n_suggestions=5, seed=1)
        b = sampler.suggest(n_suggestions=5, seed=2)
        assert a != b

    def test_capabilities(self):
        caps = RandomSampler().capabilities()
        assert caps["supports_categorical"] is True
        assert caps["supports_continuous"] is True
        assert caps["requires_observations"] is False


# ── LatinHypercubeSampler tests ───────────────────────────────────────

class TestLatinHypercubeSampler:
    def test_name(self):
        assert LatinHypercubeSampler().name() == "latin_hypercube_sampler"

    def test_suggest_within_bounds(self):
        sampler = LatinHypercubeSampler()
        sampler.fit([], CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=42)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_suggest_mixed_within_bounds(self):
        sampler = LatinHypercubeSampler()
        sampler.fit([], MIXED_SPECS)
        suggestions = sampler.suggest(n_suggestions=15, seed=7)
        _assert_within_bounds(suggestions, MIXED_SPECS)

    def test_strata_coverage(self):
        """Each stratum in each dimension should have exactly one sample."""
        sampler = LatinHypercubeSampler()
        specs = [
            ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ]
        sampler.fit([], specs)
        n = 10
        suggestions = sampler.suggest(n_suggestions=n, seed=42)
        values = [s["x"] for s in suggestions]
        # Each stratum spans width 0.1.  Verify exactly one value per stratum.
        strata_hit = [False] * n
        for v in values:
            stratum = min(int(v * n), n - 1)
            strata_hit[stratum] = True
        assert all(strata_hit), f"Not all strata hit: {strata_hit}"

    def test_deterministic(self):
        sampler = LatinHypercubeSampler()
        sampler.fit([], CONTINUOUS_SPECS)
        a = sampler.suggest(n_suggestions=5, seed=99)
        b = sampler.suggest(n_suggestions=5, seed=99)
        assert a == b


# ── TPESampler tests ──────────────────────────────────────────────────

class TestTPESampler:
    def test_name(self):
        assert TPESampler().name() == "tpe_sampler"

    def test_fallback_to_random_with_few_obs(self):
        sampler = TPESampler()
        sampler.fit([], CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=5, seed=42)
        assert len(suggestions) == 5
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_suggest_with_observations(self):
        sampler = TPESampler()
        obs = _make_observations(CONTINUOUS_SPECS, n=30)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=42)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_suggest_mixed_with_observations(self):
        sampler = TPESampler()
        obs = _make_observations(MIXED_SPECS, n=30)
        sampler.fit(obs, MIXED_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=7)
        _assert_within_bounds(suggestions, MIXED_SPECS)

    def test_deterministic(self):
        sampler = TPESampler()
        obs = _make_observations(CONTINUOUS_SPECS, n=20)
        sampler.fit(obs, CONTINUOUS_SPECS)
        a = sampler.suggest(n_suggestions=5, seed=55)
        b = sampler.suggest(n_suggestions=5, seed=55)
        assert a == b

    def test_capabilities(self):
        caps = TPESampler().capabilities()
        assert caps["requires_observations"] is True
        assert caps["supports_batch"] is True

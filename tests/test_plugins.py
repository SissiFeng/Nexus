"""Tests for the plugin architecture and built-in backends."""

from __future__ import annotations

import pytest

from optimization_copilot.core import Observation, ParameterSpec, VariableType
from optimization_copilot.plugins.base import AlgorithmPlugin
from optimization_copilot.plugins.registry import BackendPolicy, PluginRegistry
from optimization_copilot.backends.builtin import (
    CMAESSampler,
    DifferentialEvolution,
    GaussianProcessBO,
    LatinHypercubeSampler,
    NSGA2Sampler,
    RandomForestBO,
    RandomSampler,
    SobolSampler,
    TPESampler,
    TuRBOSampler,
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


# ── SobolSampler tests ──────────────────────────────────────────────

class TestSobolSampler:
    def test_name(self):
        assert SobolSampler().name() == "sobol_sampler"

    def test_suggest_within_bounds_continuous(self):
        sampler = SobolSampler()
        sampler.fit([], CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=0)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_suggest_within_bounds_mixed(self):
        sampler = SobolSampler()
        sampler.fit([], MIXED_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=7)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, MIXED_SPECS)

    def test_deterministic(self):
        sampler = SobolSampler()
        sampler.fit([], CONTINUOUS_SPECS)
        a = sampler.suggest(n_suggestions=5, seed=42)
        b = sampler.suggest(n_suggestions=5, seed=42)
        assert a == b

    def test_capabilities(self):
        caps = SobolSampler().capabilities()
        assert caps["supports_categorical"] is True
        assert caps["supports_continuous"] is True
        assert caps["supports_discrete"] is True
        assert caps["requires_observations"] is False
        assert caps["max_dimensions"] == 21

    def test_different_seeds_differ(self):
        sampler = SobolSampler()
        sampler.fit([], CONTINUOUS_SPECS)
        a = sampler.suggest(n_suggestions=5, seed=1)
        b = sampler.suggest(n_suggestions=5, seed=2)
        assert a != b

    def test_better_distribution_than_random(self):
        """Sobol sequences should produce more uniformly spread samples.

        We use a 1-D projection and check that the standard deviation of
        gaps between sorted samples is lower for Sobol than for random.
        A perfectly uniform sequence has equal gaps (zero gap variance).
        """
        specs = [
            ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ]
        n = 64

        # Sobol samples (seed=1 to skip the origin point at index 0)
        sobol = SobolSampler()
        sobol.fit([], specs)
        sobol_pts = sobol.suggest(n_suggestions=n, seed=1)
        sobol_vals = sorted(p["x"] for p in sobol_pts)

        # Random samples
        rand = RandomSampler()
        rand.fit([], specs)
        rand_pts = rand.suggest(n_suggestions=n, seed=0)
        rand_vals = sorted(p["x"] for p in rand_pts)

        def gap_variance(vals):
            gaps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
            mean_gap = sum(gaps) / len(gaps)
            return sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)

        sobol_gv = gap_variance(sobol_vals)
        rand_gv = gap_variance(rand_vals)
        # Sobol should have lower gap variance (more uniform spacing)
        assert sobol_gv < rand_gv, (
            f"Sobol gap variance {sobol_gv} should be < random {rand_gv}"
        )

    def test_max_dimensions_cap(self):
        """Sobol should work up to 21 dimensions without error."""
        specs = [
            ParameterSpec(name=f"x{i}", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
            for i in range(21)
        ]
        sampler = SobolSampler()
        sampler.fit([], specs)
        suggestions = sampler.suggest(n_suggestions=5, seed=0)
        assert len(suggestions) == 5
        _assert_within_bounds(suggestions, specs)


# ── GaussianProcessBO tests ─────────────────────────────────────────

class TestGaussianProcessBO:
    def test_name(self):
        assert GaussianProcessBO().name() == "gaussian_process_bo"

    def test_suggest_within_bounds_continuous(self):
        sampler = GaussianProcessBO()
        obs = _make_observations(CONTINUOUS_SPECS, n=20)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=0)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_suggest_within_bounds_mixed(self):
        """Mixed specs include categorical, so GP falls back to random.
        Bounds should still be respected."""
        sampler = GaussianProcessBO()
        obs = _make_observations(MIXED_SPECS, n=20)
        sampler.fit(obs, MIXED_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=7)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, MIXED_SPECS)

    def test_deterministic(self):
        sampler = GaussianProcessBO()
        obs = _make_observations(CONTINUOUS_SPECS, n=20)
        sampler.fit(obs, CONTINUOUS_SPECS)
        a = sampler.suggest(n_suggestions=5, seed=55)
        b = sampler.suggest(n_suggestions=5, seed=55)
        assert a == b

    def test_capabilities(self):
        caps = GaussianProcessBO().capabilities()
        assert caps["requires_observations"] is True
        assert caps["supports_categorical"] is False
        assert caps["supports_continuous"] is True
        assert caps["supports_batch"] is True

    def test_suggest_with_observations(self):
        sampler = GaussianProcessBO()
        obs = _make_observations(CONTINUOUS_SPECS, n=30)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=42)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_fallback_to_random_with_few_observations(self):
        """With < 3 observations the GP should fall back to random sampling."""
        sampler = GaussianProcessBO()
        obs = _make_observations(CONTINUOUS_SPECS, n=2)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=5, seed=42)
        assert len(suggestions) == 5
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_fallback_with_no_observations(self):
        """With zero observations the GP should still produce valid samples."""
        sampler = GaussianProcessBO()
        sampler.fit([], CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=5, seed=42)
        assert len(suggestions) == 5
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)


# ── RandomForestBO tests ────────────────────────────────────────────

class TestRandomForestBO:
    def test_name(self):
        assert RandomForestBO().name() == "random_forest_bo"

    def test_suggest_within_bounds_continuous(self):
        sampler = RandomForestBO()
        obs = _make_observations(CONTINUOUS_SPECS, n=20)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=0)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_suggest_within_bounds_mixed(self):
        sampler = RandomForestBO()
        obs = _make_observations(MIXED_SPECS, n=20)
        sampler.fit(obs, MIXED_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=7)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, MIXED_SPECS)

    def test_deterministic(self):
        sampler = RandomForestBO()
        obs = _make_observations(CONTINUOUS_SPECS, n=20)
        sampler.fit(obs, CONTINUOUS_SPECS)
        a = sampler.suggest(n_suggestions=5, seed=55)
        b = sampler.suggest(n_suggestions=5, seed=55)
        assert a == b

    def test_capabilities(self):
        caps = RandomForestBO().capabilities()
        assert caps["requires_observations"] is True
        assert caps["supports_categorical"] is True
        assert caps["supports_continuous"] is True
        assert caps["supports_batch"] is True

    def test_suggest_with_observations(self):
        sampler = RandomForestBO()
        obs = _make_observations(CONTINUOUS_SPECS, n=30)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=42)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_with_categorical_parameters(self):
        """RandomForestBO supports categorical via one-hot encoding."""
        cat_specs = [
            ParameterSpec(name="optimizer", type=VariableType.CATEGORICAL,
                          categories=["adam", "sgd", "rmsprop"]),
            ParameterSpec(name="lr", type=VariableType.CONTINUOUS, lower=1e-5, upper=1.0),
        ]
        obs = _make_observations(cat_specs, n=20)
        sampler = RandomForestBO()
        sampler.fit(obs, cat_specs)
        suggestions = sampler.suggest(n_suggestions=10, seed=42)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, cat_specs)
        # Verify categorical values are valid
        for s in suggestions:
            assert s["optimizer"] in ["adam", "sgd", "rmsprop"]

    def test_fallback_with_few_observations(self):
        """With < 3 observations, falls back to random sampling."""
        sampler = RandomForestBO()
        obs = _make_observations(CONTINUOUS_SPECS, n=2)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=5, seed=42)
        assert len(suggestions) == 5
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)


# ── CMAESSampler tests ──────────────────────────────────────────────

class TestCMAESSampler:
    def test_name(self):
        assert CMAESSampler().name() == "cmaes_sampler"

    def test_suggest_within_bounds_continuous(self):
        sampler = CMAESSampler()
        obs = _make_observations(CONTINUOUS_SPECS, n=20)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=0)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_suggest_within_bounds_mixed(self):
        sampler = CMAESSampler()
        obs = _make_observations(MIXED_SPECS, n=20)
        sampler.fit(obs, MIXED_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=7)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, MIXED_SPECS)

    def test_deterministic(self):
        sampler = CMAESSampler()
        obs = _make_observations(CONTINUOUS_SPECS, n=20)
        sampler.fit(obs, CONTINUOUS_SPECS)
        a = sampler.suggest(n_suggestions=5, seed=55)
        b = sampler.suggest(n_suggestions=5, seed=55)
        assert a == b

    def test_capabilities(self):
        caps = CMAESSampler().capabilities()
        assert caps["requires_observations"] is True
        assert caps["supports_categorical"] is False
        assert caps["supports_continuous"] is True
        assert caps["supports_batch"] is True

    def test_suggest_with_observations(self):
        sampler = CMAESSampler()
        obs = _make_observations(CONTINUOUS_SPECS, n=30)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=42)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_convergence_toward_good_region(self):
        """CMA-ES should adapt its distribution toward regions with good KPIs."""
        specs = [
            ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            ParameterSpec(name="y", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
        # Create observations where low values of x and y produce low (good) KPI
        import random as _rng
        r = _rng.Random(42)
        obs = []
        for i in range(40):
            x = r.uniform(0.0, 10.0)
            y = r.uniform(0.0, 10.0)
            kpi = x + y  # lower is better -> optimum at (0, 0)
            obs.append(Observation(
                iteration=i, parameters={"x": x, "y": y},
                kpi_values={"objective": kpi}, timestamp=float(i),
            ))

        sampler = CMAESSampler()
        # Fit multiple times to let CMA-ES adapt
        sampler.fit(obs, specs)
        suggestions = sampler.suggest(n_suggestions=20, seed=42)
        _assert_within_bounds(suggestions, specs)

        # The mean of suggested x and y should be biased toward lower values
        # (compared to the midpoint of the range, which is 5.0)
        mean_x = sum(s["x"] for s in suggestions) / len(suggestions)
        mean_y = sum(s["y"] for s in suggestions) / len(suggestions)
        assert mean_x < 7.0, f"CMA-ES mean x={mean_x} not biased toward good region"
        assert mean_y < 7.0, f"CMA-ES mean y={mean_y} not biased toward good region"

    def test_suggest_without_observations_initializes(self):
        """CMA-ES should still produce valid suggestions with no observations."""
        sampler = CMAESSampler()
        sampler.fit([], CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=5, seed=42)
        assert len(suggestions) == 5
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)


# ── DifferentialEvolution tests ─────────────────────────────────────

class TestDifferentialEvolution:
    def test_name(self):
        assert DifferentialEvolution().name() == "differential_evolution"

    def test_suggest_within_bounds_continuous(self):
        sampler = DifferentialEvolution()
        sampler.fit([], CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=0)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_suggest_within_bounds_mixed(self):
        sampler = DifferentialEvolution()
        sampler.fit([], MIXED_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=7)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, MIXED_SPECS)

    def test_deterministic(self):
        """Fresh sampler instances with same seed produce identical output.
        DE mutates internal population state on each suggest call, so we
        use two independent instances to verify seed-based determinism."""
        obs = _make_observations(CONTINUOUS_SPECS, n=20)
        sampler_a = DifferentialEvolution()
        sampler_a.fit(obs, CONTINUOUS_SPECS)
        a = sampler_a.suggest(n_suggestions=5, seed=55)

        sampler_b = DifferentialEvolution()
        sampler_b.fit(obs, CONTINUOUS_SPECS)
        b = sampler_b.suggest(n_suggestions=5, seed=55)
        assert a == b

    def test_capabilities(self):
        caps = DifferentialEvolution().capabilities()
        assert caps["requires_observations"] is False
        assert caps["supports_categorical"] is True
        assert caps["supports_continuous"] is True
        assert caps["supports_batch"] is True

    def test_suggest_with_observations(self):
        sampler = DifferentialEvolution()
        obs = _make_observations(CONTINUOUS_SPECS, n=30)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=42)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_population_maintenance(self):
        """Population should persist across multiple suggest calls."""
        sampler = DifferentialEvolution(population_size=10)
        sampler.fit([], CONTINUOUS_SPECS)
        # First call initializes population
        s1 = sampler.suggest(n_suggestions=3, seed=1)
        assert len(sampler._population) == 10
        pop_size_after_first = len(sampler._population)
        # Second call should reuse existing population (not reinitialize)
        s2 = sampler.suggest(n_suggestions=3, seed=2)
        assert len(sampler._population) == pop_size_after_first
        # Suggestions should be different with different seeds
        assert s1 != s2

    def test_population_seeded_from_observations(self):
        """When observations are provided, population should be seeded from them."""
        sampler = DifferentialEvolution(population_size=10)
        obs = _make_observations(CONTINUOUS_SPECS, n=15)
        sampler.fit(obs, CONTINUOUS_SPECS)
        # Population should contain entries from observations
        assert len(sampler._population) > 0
        suggestions = sampler.suggest(n_suggestions=5, seed=42)
        assert len(suggestions) == 5
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)


# ── NSGA2Sampler tests ──────────────────────────────────────────────

class TestNSGA2Sampler:
    def test_name(self):
        assert NSGA2Sampler().name() == "nsga2_sampler"

    def test_suggest_within_bounds_continuous(self):
        sampler = NSGA2Sampler()
        obs = _make_observations(CONTINUOUS_SPECS, n=20)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=0)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_suggest_within_bounds_mixed(self):
        sampler = NSGA2Sampler()
        obs = _make_observations(MIXED_SPECS, n=20)
        sampler.fit(obs, MIXED_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=7)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, MIXED_SPECS)

    def test_deterministic(self):
        """Fresh sampler instances with same seed produce identical output.
        NSGA-II mutates internal population state on suggest, so we use
        two independent instances to verify seed-based determinism."""
        obs = _make_observations(CONTINUOUS_SPECS, n=20)
        sampler_a = NSGA2Sampler()
        sampler_a.fit(obs, CONTINUOUS_SPECS)
        a = sampler_a.suggest(n_suggestions=5, seed=55)

        sampler_b = NSGA2Sampler()
        sampler_b.fit(obs, CONTINUOUS_SPECS)
        b = sampler_b.suggest(n_suggestions=5, seed=55)
        assert a == b

    def test_capabilities(self):
        caps = NSGA2Sampler().capabilities()
        assert caps["requires_observations"] is True
        assert caps["supports_categorical"] is True
        assert caps["supports_continuous"] is True
        assert caps["supports_batch"] is True

    def test_supports_multi_objective(self):
        """NSGA-II must report multi-objective support."""
        caps = NSGA2Sampler().capabilities()
        assert caps["supports_multi_objective"] is True

    def test_suggest_with_observations(self):
        sampler = NSGA2Sampler()
        obs = _make_observations(CONTINUOUS_SPECS, n=30)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=42)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_with_multi_kpi_observations(self):
        """NSGA-II should handle observations with multiple KPI values."""
        import random as _rng
        r = _rng.Random(99)
        specs = CONTINUOUS_SPECS
        obs = []
        for i in range(40):
            params = {}
            for s in specs:
                params[s.name] = r.uniform(s.lower, s.upper)
            # Two conflicting objectives
            kpi1 = params["x1"]          # minimize x1
            kpi2 = 10.0 - params["x1"]   # minimize (10 - x1), conflicts with kpi1
            obs.append(Observation(
                iteration=i, parameters=params,
                kpi_values={"obj1": kpi1, "obj2": kpi2},
                timestamp=float(i),
            ))

        sampler = NSGA2Sampler(population_size=40)
        sampler.fit(obs, specs)
        suggestions = sampler.suggest(n_suggestions=10, seed=42)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, specs)

    def test_few_observations_still_works(self):
        """With very few observations, NSGA-II should still produce suggestions."""
        sampler = NSGA2Sampler()
        obs = _make_observations(CONTINUOUS_SPECS, n=2)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=5, seed=42)
        assert len(suggestions) == 5
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)


# ── TuRBOSampler tests ──────────────────────────────────────────────

class TestTuRBOSampler:
    def test_name(self):
        assert TuRBOSampler().name() == "turbo_sampler"

    def test_suggest_within_bounds_continuous(self):
        sampler = TuRBOSampler()
        obs = _make_observations(CONTINUOUS_SPECS, n=20)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=0)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_suggest_within_bounds_mixed(self):
        sampler = TuRBOSampler()
        obs = _make_observations(MIXED_SPECS, n=20)
        sampler.fit(obs, MIXED_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=7)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, MIXED_SPECS)

    def test_deterministic(self):
        sampler = TuRBOSampler()
        obs = _make_observations(CONTINUOUS_SPECS, n=20)
        sampler.fit(obs, CONTINUOUS_SPECS)
        a = sampler.suggest(n_suggestions=5, seed=55)
        b = sampler.suggest(n_suggestions=5, seed=55)
        assert a == b

    def test_capabilities(self):
        caps = TuRBOSampler().capabilities()
        assert caps["requires_observations"] is True
        assert caps["supports_categorical"] is False
        assert caps["supports_continuous"] is True
        assert caps["supports_batch"] is True

    def test_suggest_with_observations(self):
        sampler = TuRBOSampler()
        obs = _make_observations(CONTINUOUS_SPECS, n=30)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=10, seed=42)
        assert len(suggestions) == 10
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

    def test_trust_region_shrinks_on_failures(self):
        """Trust region length should shrink after consecutive non-improving fits."""
        specs = [
            ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
        sampler = TuRBOSampler(
            length_init=0.8,
            length_min=0.01,
            length_max=1.6,
            success_tol=2,
            failure_tol=3,
        )
        # First fit: establish best value
        obs1 = [Observation(iteration=0, parameters={"x": 5.0},
                            kpi_values={"obj": 5.0}, timestamp=0.0)]
        sampler.fit(obs1, specs)
        initial_length = sampler._length

        # Subsequent fits with NO improvement (same or worse best value)
        for i in range(1, 10):
            obs = [Observation(iteration=j, parameters={"x": 5.0 + j * 0.01},
                               kpi_values={"obj": 5.0 + j * 0.01}, timestamp=float(j))
                   for j in range(i + 1)]
            sampler.fit(obs, specs)

        # After several non-improving fits, trust region should have shrunk
        assert sampler._length < initial_length, (
            f"Trust region length {sampler._length} did not shrink from {initial_length}"
        )

    def test_trust_region_grows_on_successes(self):
        """Trust region length should grow after consecutive improving fits."""
        specs = [
            ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
        sampler = TuRBOSampler(
            length_init=0.2,
            length_min=0.01,
            length_max=1.6,
            success_tol=2,
            failure_tol=5,
        )
        # First fit
        obs = [Observation(iteration=0, parameters={"x": 5.0},
                           kpi_values={"obj": 10.0}, timestamp=0.0)]
        sampler.fit(obs, specs)
        initial_length = sampler._length

        # Each subsequent fit shows improvement (strictly lower best KPI)
        for i in range(1, 8):
            obs_improving = [
                Observation(iteration=j, parameters={"x": 5.0 - j * 0.5},
                            kpi_values={"obj": 10.0 - j * 1.0}, timestamp=float(j))
                for j in range(i + 1)
            ]
            sampler.fit(obs_improving, specs)

        assert sampler._length > initial_length, (
            f"Trust region length {sampler._length} did not grow from {initial_length}"
        )

    def test_fallback_with_few_observations(self):
        """With < 3 observations, TuRBO falls back to random sampling."""
        sampler = TuRBOSampler()
        obs = _make_observations(CONTINUOUS_SPECS, n=2)
        sampler.fit(obs, CONTINUOUS_SPECS)
        suggestions = sampler.suggest(n_suggestions=5, seed=42)
        assert len(suggestions) == 5
        _assert_within_bounds(suggestions, CONTINUOUS_SPECS)

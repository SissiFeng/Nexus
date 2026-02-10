"""Tests for the Optimizer Marketplace (Capability 13).

Covers module-level helpers, MarketplaceStatus enum, CullPolicy and
MarketplaceEntry dataclasses, and the full Marketplace class including
submission, querying, benchmarking, leaderboard, manual status management,
auto-cull state machine, and serialization round-trips.
"""

from __future__ import annotations

import random as _rng
from typing import Any

import pytest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.marketplace.marketplace import (
    CullPolicy,
    Marketplace,
    MarketplaceEntry,
    MarketplaceStatus,
    _compute_failure_rate_from_results,
    _incremental_mean,
)
from optimization_copilot.benchmark.runner import BenchmarkResult
from optimization_copilot.plugins.base import AlgorithmPlugin
from optimization_copilot.plugins.registry import PluginRegistry


# ── Helpers & Fixtures ───────────────────────────────────────────────


CONTINUOUS_SPECS = [
    ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
    ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
]


class _DummyPlugin(AlgorithmPlugin):
    """Deterministic plugin for marketplace testing."""

    def name(self) -> str:
        return "dummy_plugin"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = parameter_specs

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = _rng.Random(seed)
        results: list[dict[str, Any]] = []
        for _ in range(n_suggestions):
            point: dict[str, Any] = {}
            for spec in self._specs:
                if spec.type == VariableType.CATEGORICAL:
                    point[spec.name] = rng.choice(spec.categories or ["a"])
                elif spec.type == VariableType.DISCRETE:
                    lo = int(spec.lower) if spec.lower is not None else 0
                    hi = int(spec.upper) if spec.upper is not None else 10
                    point[spec.name] = rng.randint(lo, hi)
                else:
                    lo = spec.lower if spec.lower is not None else 0.0
                    hi = spec.upper if spec.upper is not None else 1.0
                    point[spec.name] = rng.uniform(lo, hi)
            results.append(point)
        return results

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": False,
            "max_dimensions": None,
        }


class _DummyPlugin2(AlgorithmPlugin):
    """Second deterministic plugin (different name) for multi-plugin tests."""

    def name(self) -> str:
        return "dummy_plugin_2"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = parameter_specs

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = _rng.Random(seed + 1000)
        results: list[dict[str, Any]] = []
        for _ in range(n_suggestions):
            point: dict[str, Any] = {}
            for spec in self._specs:
                if spec.type == VariableType.CATEGORICAL:
                    point[spec.name] = rng.choice(spec.categories or ["a"])
                elif spec.type == VariableType.DISCRETE:
                    lo = int(spec.lower) if spec.lower is not None else 0
                    hi = int(spec.upper) if spec.upper is not None else 10
                    point[spec.name] = rng.randint(lo, hi)
                else:
                    lo = spec.lower if spec.lower is not None else 0.0
                    hi = spec.upper if spec.upper is not None else 1.0
                    point[spec.name] = rng.uniform(lo, hi)
            results.append(point)
        return results

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": False,
            "max_dimensions": None,
        }


def _make_snapshot(n_obs: int = 20, seed: int = 42) -> CampaignSnapshot:
    """Build a minimal CampaignSnapshot with deterministic observations."""
    r = _rng.Random(seed)
    observations: list[Observation] = []
    for i in range(n_obs):
        x1 = r.uniform(0.0, 10.0)
        x2 = r.uniform(-5.0, 5.0)
        kpi = -(x1 - 5.0) ** 2 - (x2 - 0.0) ** 2 + r.gauss(0, 0.1)
        observations.append(
            Observation(
                iteration=i,
                parameters={"x1": x1, "x2": x2},
                kpi_values={"objective": kpi},
                timestamp=float(i),
            )
        )
    return CampaignSnapshot(
        campaign_id="test_campaign",
        parameter_specs=CONTINUOUS_SPECS,
        observations=observations,
        objective_names=["objective"],
        objective_directions=["maximize"],
        current_iteration=n_obs,
    )


def _make_scenarios(
    n_scenarios: int = 2,
    n_obs: int = 20,
    seed: int = 42,
) -> list[tuple[str, CampaignSnapshot]]:
    """Build a list of (name, snapshot) scenario tuples."""
    return [
        (f"scenario_{i}", _make_snapshot(n_obs=n_obs, seed=seed + i * 100))
        for i in range(n_scenarios)
    ]


# ── Module-level helpers ─────────────────────────────────────────────


class TestIncrementalMean:
    def test_first_value_returns_new_value(self):
        assert _incremental_mean(0.0, 0, 5.0) == 5.0

    def test_subsequent_values_compute_running_mean(self):
        # After first value of 4.0, mean is 4.0
        mean = _incremental_mean(0.0, 0, 4.0)
        assert mean == 4.0
        # After second value of 6.0, mean should be 5.0
        mean = _incremental_mean(mean, 1, 6.0)
        assert mean == pytest.approx(5.0)
        # After third value of 8.0, mean should be (4+6+8)/3 = 6.0
        mean = _incremental_mean(mean, 2, 8.0)
        assert mean == pytest.approx(6.0)

    def test_large_count(self):
        # After 99 values of 1.0, adding 100.0 should yield ~1.99
        mean = _incremental_mean(1.0, 99, 100.0)
        expected = (1.0 * 99 + 100.0) / 100
        assert mean == pytest.approx(expected)


class TestComputeFailureRateFromResults:
    def test_empty_results_returns_zero(self):
        assert _compute_failure_rate_from_results([]) == 0.0

    def test_zero_iterations_returns_zero(self):
        result = BenchmarkResult(
            backend_name="b",
            scenario_name="s",
            fingerprint_key="fp",
            final_best_kpi=0.0,
            convergence_iteration=0,
            total_iterations=0,
            failure_count=0,
            regret=0.0,
            sample_efficiency=0.0,
            wall_time_seconds=0.0,
        )
        assert _compute_failure_rate_from_results([result]) == 0.0

    def test_aggregate_across_results(self):
        r1 = BenchmarkResult(
            backend_name="b",
            scenario_name="s1",
            fingerprint_key="fp",
            final_best_kpi=0.0,
            convergence_iteration=0,
            total_iterations=10,
            failure_count=2,
            regret=0.0,
            sample_efficiency=0.0,
            wall_time_seconds=0.0,
        )
        r2 = BenchmarkResult(
            backend_name="b",
            scenario_name="s2",
            fingerprint_key="fp",
            final_best_kpi=0.0,
            convergence_iteration=0,
            total_iterations=20,
            failure_count=4,
            regret=0.0,
            sample_efficiency=0.0,
            wall_time_seconds=0.0,
        )
        # total failures = 6, total iterations = 30 -> 0.2
        rate = _compute_failure_rate_from_results([r1, r2])
        assert rate == pytest.approx(0.2)


# ── MarketplaceStatus enum ──────────────────────────────────────────


class TestMarketplaceStatus:
    def test_values(self):
        assert MarketplaceStatus.ACTIVE.value == "active"
        assert MarketplaceStatus.PROBATION.value == "probation"
        assert MarketplaceStatus.RETIRED.value == "retired"

    def test_string_construction(self):
        assert MarketplaceStatus("active") is MarketplaceStatus.ACTIVE
        assert MarketplaceStatus("probation") is MarketplaceStatus.PROBATION
        assert MarketplaceStatus("retired") is MarketplaceStatus.RETIRED


# ── CullPolicy dataclass ────────────────────────────────────────────


class TestCullPolicy:
    def test_defaults(self):
        policy = CullPolicy()
        assert policy.max_failure_rate == 0.3
        assert policy.min_avg_score == 0.2
        assert policy.min_benchmarks == 3
        assert policy.grace_benchmarks == 2

    def test_custom_values(self):
        policy = CullPolicy(
            max_failure_rate=0.5,
            min_avg_score=0.1,
            min_benchmarks=5,
            grace_benchmarks=3,
        )
        assert policy.max_failure_rate == 0.5
        assert policy.min_avg_score == 0.1
        assert policy.min_benchmarks == 5
        assert policy.grace_benchmarks == 3

    def test_to_dict_from_dict_round_trip(self):
        original = CullPolicy(
            max_failure_rate=0.4,
            min_avg_score=0.15,
            min_benchmarks=4,
            grace_benchmarks=1,
        )
        data = original.to_dict()
        restored = CullPolicy.from_dict(data)
        assert restored.max_failure_rate == original.max_failure_rate
        assert restored.min_avg_score == original.min_avg_score
        assert restored.min_benchmarks == original.min_benchmarks
        assert restored.grace_benchmarks == original.grace_benchmarks


# ── MarketplaceEntry dataclass ───────────────────────────────────────


class TestMarketplaceEntry:
    def test_default_construction(self):
        entry = MarketplaceEntry(plugin_name="test_plugin")
        assert entry.plugin_name == "test_plugin"
        assert entry.plugin_class_name == ""
        assert entry.n_benchmarks == 0
        assert entry.avg_score == 0.0
        assert entry.failure_rate == 0.0
        assert entry.last_benchmark_time == 0.0
        assert entry.status == MarketplaceStatus.ACTIVE
        assert entry.probation_benchmarks == 0
        assert entry.total_wins == 0
        assert entry.avg_regret == 0.0
        assert entry.avg_convergence_speed == 0.0

    def test_to_dict_from_dict_round_trip(self):
        original = MarketplaceEntry(
            plugin_name="my_plugin",
            plugin_class_name="some.module.MyPlugin",
            n_benchmarks=10,
            avg_score=0.85,
            failure_rate=0.05,
            last_benchmark_time=1234567890.0,
            status=MarketplaceStatus.PROBATION,
            probation_benchmarks=3,
            total_wins=7,
            avg_regret=0.12,
            avg_convergence_speed=0.45,
        )
        data = original.to_dict()
        restored = MarketplaceEntry.from_dict(data)
        assert restored.plugin_name == original.plugin_name
        assert restored.plugin_class_name == original.plugin_class_name
        assert restored.n_benchmarks == original.n_benchmarks
        assert restored.avg_score == original.avg_score
        assert restored.failure_rate == original.failure_rate
        assert restored.last_benchmark_time == original.last_benchmark_time
        assert restored.status == original.status
        assert restored.probation_benchmarks == original.probation_benchmarks
        assert restored.total_wins == original.total_wins
        assert restored.avg_regret == original.avg_regret
        assert restored.avg_convergence_speed == original.avg_convergence_speed

    def test_status_serialized_as_string(self):
        entry = MarketplaceEntry(
            plugin_name="x",
            status=MarketplaceStatus.RETIRED,
        )
        data = entry.to_dict()
        assert data["status"] == "retired"
        assert isinstance(data["status"], str)


# ── Marketplace class ────────────────────────────────────────────────


class TestMarketplaceConstruction:
    def test_defaults(self):
        mp = Marketplace()
        assert mp.list_all() == []
        assert mp.list_active() == []
        assert mp.get_leaderboard() is None

    def test_custom_registry_and_policy(self):
        registry = PluginRegistry()
        policy = CullPolicy(max_failure_rate=0.5)
        mp = Marketplace(registry=registry, cull_policy=policy)
        # Marketplace should use the provided registry and policy
        assert mp._registry is registry
        assert mp._cull_policy is policy


class TestMarketplaceSubmission:
    def test_submit_plugin(self):
        mp = Marketplace()
        entry = mp.submit_plugin(_DummyPlugin)
        assert entry.plugin_name == "dummy_plugin"
        assert "DummyPlugin" in entry.plugin_class_name or "_DummyPlugin" in entry.plugin_class_name
        assert entry.status == MarketplaceStatus.ACTIVE
        assert entry.n_benchmarks == 0

    def test_submit_duplicate_raises(self):
        mp = Marketplace()
        mp.submit_plugin(_DummyPlugin)
        with pytest.raises(ValueError, match="already registered"):
            mp.submit_plugin(_DummyPlugin)


class TestMarketplaceQueries:
    def test_get_entry(self):
        mp = Marketplace()
        mp.submit_plugin(_DummyPlugin)
        entry = mp.get_entry("dummy_plugin")
        assert entry.plugin_name == "dummy_plugin"

    def test_get_entry_unknown_raises(self):
        mp = Marketplace()
        with pytest.raises(KeyError, match="No marketplace entry"):
            mp.get_entry("nonexistent")

    def test_list_active_excludes_retired(self):
        mp = Marketplace()
        mp.submit_plugin(_DummyPlugin)
        mp.submit_plugin(_DummyPlugin2)
        mp.retire_plugin("dummy_plugin")
        active = mp.list_active()
        assert "dummy_plugin" not in active
        assert "dummy_plugin_2" in active

    def test_list_all_includes_everything(self):
        mp = Marketplace()
        mp.submit_plugin(_DummyPlugin)
        mp.submit_plugin(_DummyPlugin2)
        mp.retire_plugin("dummy_plugin")
        all_entries = mp.list_all()
        names = [e.plugin_name for e in all_entries]
        assert "dummy_plugin" in names
        assert "dummy_plugin_2" in names
        assert len(all_entries) == 2


class TestMarketplaceBenchmarking:
    def test_benchmark_plugin_returns_results(self):
        mp = Marketplace()
        mp.submit_plugin(_DummyPlugin)
        scenarios = _make_scenarios(n_scenarios=2, n_obs=20, seed=42)
        results = mp.benchmark_plugin(
            "dummy_plugin", scenarios, n_iterations=10, seed=42,
        )
        assert len(results) == 2
        for r in results:
            assert r.backend_name == "dummy_plugin"

    def test_benchmark_plugin_updates_health_metrics(self):
        mp = Marketplace()
        mp.submit_plugin(_DummyPlugin)
        scenarios = _make_scenarios(n_scenarios=1, n_obs=20, seed=42)
        mp.benchmark_plugin("dummy_plugin", scenarios, n_iterations=10, seed=42)
        entry = mp.get_entry("dummy_plugin")
        assert entry.n_benchmarks > 0
        assert entry.last_benchmark_time > 0.0

    def test_benchmark_all(self):
        mp = Marketplace()
        mp.submit_plugin(_DummyPlugin)
        mp.submit_plugin(_DummyPlugin2)
        scenarios = _make_scenarios(n_scenarios=1, n_obs=20, seed=42)
        results = mp.benchmark_all(scenarios, n_iterations=10, seed=42)
        backend_names = {r.backend_name for r in results}
        assert "dummy_plugin" in backend_names
        assert "dummy_plugin_2" in backend_names


class TestMarketplaceLeaderboard:
    def test_get_leaderboard_initially_none(self):
        mp = Marketplace()
        assert mp.get_leaderboard() is None

    def test_refresh_leaderboard(self):
        mp = Marketplace()
        mp.submit_plugin(_DummyPlugin)
        scenarios = _make_scenarios(n_scenarios=2, n_obs=20, seed=42)
        mp.benchmark_plugin("dummy_plugin", scenarios, n_iterations=10, seed=42)
        lb = mp.refresh_leaderboard()
        assert lb is not None
        assert lb.total_scenarios >= 1
        assert len(lb.entries) >= 1
        assert lb.entries[0].backend_name == "dummy_plugin"

    def test_refresh_leaderboard_filters_retired(self):
        mp = Marketplace()
        mp.submit_plugin(_DummyPlugin)
        mp.submit_plugin(_DummyPlugin2)
        scenarios = _make_scenarios(n_scenarios=1, n_obs=20, seed=42)
        mp.benchmark_all(scenarios, n_iterations=10, seed=42)
        mp.retire_plugin("dummy_plugin")
        lb = mp.refresh_leaderboard()
        backend_names = [e.backend_name for e in lb.entries]
        assert "dummy_plugin" not in backend_names
        assert "dummy_plugin_2" in backend_names

    def test_get_leaderboard_after_refresh(self):
        mp = Marketplace()
        mp.submit_plugin(_DummyPlugin)
        scenarios = _make_scenarios(n_scenarios=1, n_obs=20, seed=42)
        mp.benchmark_plugin("dummy_plugin", scenarios, n_iterations=10, seed=42)
        lb = mp.refresh_leaderboard()
        assert mp.get_leaderboard() is lb


class TestMarketplaceManualStatus:
    def test_retire_plugin(self):
        mp = Marketplace()
        mp.submit_plugin(_DummyPlugin)
        mp.retire_plugin("dummy_plugin")
        entry = mp.get_entry("dummy_plugin")
        assert entry.status == MarketplaceStatus.RETIRED

    def test_reinstate_plugin(self):
        mp = Marketplace()
        mp.submit_plugin(_DummyPlugin)
        mp.retire_plugin("dummy_plugin")
        mp.reinstate_plugin("dummy_plugin")
        entry = mp.get_entry("dummy_plugin")
        assert entry.status == MarketplaceStatus.ACTIVE
        assert entry.probation_benchmarks == 0


class TestMarketplaceAutoCull:
    def test_no_cull_when_below_min_benchmarks(self):
        """Cull logic should not trigger when n_benchmarks < min_benchmarks."""
        policy = CullPolicy(min_benchmarks=5, max_failure_rate=0.0, min_avg_score=1.0)
        mp = Marketplace(cull_policy=policy)
        mp.submit_plugin(_DummyPlugin)
        # Run just one scenario to get n_benchmarks=1 (below min_benchmarks=5)
        scenarios = _make_scenarios(n_scenarios=1, n_obs=20, seed=42)
        mp.benchmark_plugin("dummy_plugin", scenarios, n_iterations=10, seed=42)
        entry = mp.get_entry("dummy_plugin")
        # Even though metrics may violate thresholds, status stays ACTIVE
        assert entry.status == MarketplaceStatus.ACTIVE

    def test_active_to_probation_on_violation(self):
        """A plugin should transition ACTIVE -> PROBATION when in violation."""
        # Use a very strict policy that will be violated by any real benchmark
        policy = CullPolicy(
            max_failure_rate=0.0,  # any failure triggers
            min_avg_score=999.0,   # impossible to achieve
            min_benchmarks=1,
            grace_benchmarks=5,
        )
        mp = Marketplace(cull_policy=policy)
        mp.submit_plugin(_DummyPlugin)
        scenarios = _make_scenarios(n_scenarios=1, n_obs=20, seed=42)
        mp.benchmark_plugin("dummy_plugin", scenarios, n_iterations=10, seed=42)
        entry = mp.get_entry("dummy_plugin")
        assert entry.status == MarketplaceStatus.PROBATION

    def test_probation_to_retired_after_grace(self):
        """After grace_benchmarks on probation still in violation -> RETIRED."""
        policy = CullPolicy(
            max_failure_rate=0.0,
            min_avg_score=999.0,
            min_benchmarks=1,
            grace_benchmarks=1,  # only 1 extra needed
        )
        mp = Marketplace(cull_policy=policy)
        mp.submit_plugin(_DummyPlugin)
        scenarios = _make_scenarios(n_scenarios=1, n_obs=20, seed=42)
        # First benchmark: ACTIVE -> PROBATION
        mp.benchmark_plugin("dummy_plugin", scenarios, n_iterations=10, seed=42)
        assert mp.get_entry("dummy_plugin").status == MarketplaceStatus.PROBATION
        # Second benchmark: probation_benchmarks >= grace_benchmarks -> RETIRED
        mp.benchmark_plugin("dummy_plugin", scenarios, n_iterations=10, seed=43)
        assert mp.get_entry("dummy_plugin").status == MarketplaceStatus.RETIRED

    def test_probation_to_active_when_metrics_improve(self):
        """A plugin on PROBATION should return to ACTIVE if no longer in violation."""
        # Use a policy that triggers on avg_score but with a low threshold
        policy = CullPolicy(
            max_failure_rate=1.0,   # won't trigger on failure rate
            min_avg_score=0.0,      # effectively never triggers on score
            min_benchmarks=1,
            grace_benchmarks=10,
        )
        mp = Marketplace(cull_policy=policy)
        mp.submit_plugin(_DummyPlugin)

        # Manually put the entry on probation
        entry = mp.get_entry("dummy_plugin")
        entry.status = MarketplaceStatus.PROBATION
        entry.n_benchmarks = 5
        entry.avg_score = 0.5
        entry.failure_rate = 0.0

        # Run a benchmark -- since the policy thresholds are lenient,
        # the plugin should not be in violation and should recover to ACTIVE
        scenarios = _make_scenarios(n_scenarios=1, n_obs=20, seed=42)
        mp.benchmark_plugin("dummy_plugin", scenarios, n_iterations=10, seed=42)
        assert mp.get_entry("dummy_plugin").status == MarketplaceStatus.ACTIVE


class TestMarketplaceSerialization:
    def test_to_dict_from_dict_round_trip(self):
        registry = PluginRegistry()
        mp = Marketplace(registry=registry)
        mp.submit_plugin(_DummyPlugin)
        mp.submit_plugin(_DummyPlugin2)

        data = mp.to_dict()
        assert "entries" in data
        assert "cull_policy" in data
        assert "all_results_count" in data
        assert len(data["entries"]) == 2

        # Restore from dict (with a fresh registry)
        new_registry = PluginRegistry()
        restored = Marketplace.from_dict(data, registry=new_registry)
        restored_entries = restored.list_all()
        restored_names = {e.plugin_name for e in restored_entries}
        assert "dummy_plugin" in restored_names
        assert "dummy_plugin_2" in restored_names

    def test_serialized_entries_preserve_status(self):
        mp = Marketplace()
        mp.submit_plugin(_DummyPlugin)
        mp.retire_plugin("dummy_plugin")
        data = mp.to_dict()
        restored = Marketplace.from_dict(data)
        entry = restored.get_entry("dummy_plugin")
        assert entry.status == MarketplaceStatus.RETIRED

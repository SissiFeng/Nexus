"""Optimizer Marketplace — registry with health tracking, auto-culling, and leaderboard integration.

Provides a ``Marketplace`` that wraps the plugin registry with per-plugin health
metrics, automatic probation / retirement based on a configurable ``CullPolicy``,
and integrated benchmarking via ``BenchmarkRunner``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from optimization_copilot.benchmark.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    Leaderboard,
    LeaderboardEntry,
)
from optimization_copilot.core.models import CampaignSnapshot
from optimization_copilot.plugins.base import AlgorithmPlugin
from optimization_copilot.plugins.registry import PluginRegistry


# ── Helpers ──────────────────────────────────────────────────────────


def _incremental_mean(old_mean: float, old_count: int, new_value: float) -> float:
    """Compute an incremental running mean after one new observation."""
    if old_count == 0:
        return new_value
    return (old_mean * old_count + new_value) / (old_count + 1)


def _compute_failure_rate_from_results(results: list[BenchmarkResult]) -> float:
    """Aggregate failure rate across a batch of benchmark results."""
    total_failures = sum(r.failure_count for r in results)
    total_iterations = sum(r.total_iterations for r in results)
    if total_iterations == 0:
        return 0.0
    return total_failures / total_iterations


# ── Enums & dataclasses ──────────────────────────────────────────────


class MarketplaceStatus(str, Enum):
    """Health status of a marketplace entry."""

    ACTIVE = "active"
    PROBATION = "probation"
    RETIRED = "retired"


@dataclass
class CullPolicy:
    """Thresholds that govern automatic probation and retirement of plugins.

    Parameters
    ----------
    max_failure_rate:
        Plugins with a failure rate exceeding this are in violation.
    min_avg_score:
        Plugins scoring below this on the leaderboard are in violation.
    min_benchmarks:
        Minimum number of benchmark results before cull logic applies.
    grace_benchmarks:
        Number of additional benchmarks a plugin on probation gets before
        being retired (if still in violation).
    """

    max_failure_rate: float = 0.3
    min_avg_score: float = 0.2
    min_benchmarks: int = 3
    grace_benchmarks: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "max_failure_rate": self.max_failure_rate,
            "min_avg_score": self.min_avg_score,
            "min_benchmarks": self.min_benchmarks,
            "grace_benchmarks": self.grace_benchmarks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CullPolicy:
        """Deserialize from a plain dictionary."""
        return cls(
            max_failure_rate=data.get("max_failure_rate", 0.3),
            min_avg_score=data.get("min_avg_score", 0.2),
            min_benchmarks=data.get("min_benchmarks", 3),
            grace_benchmarks=data.get("grace_benchmarks", 2),
        )


@dataclass
class MarketplaceEntry:
    """Per-plugin health record tracked by the marketplace.

    Parameters
    ----------
    plugin_name:
        Unique name returned by the plugin's ``name()`` method.
    plugin_class_name:
        Fully-qualified class name (``module.qualname``) for serialization.
    n_benchmarks:
        Total number of individual benchmark results recorded.
    avg_score:
        Running average of the leaderboard score.
    failure_rate:
        Running average failure rate across all benchmarks.
    last_benchmark_time:
        ``time.time()`` of the most recent benchmark run.
    status:
        Current health status (active / probation / retired).
    probation_benchmarks:
        Number of benchmark results accumulated while on probation.
    total_wins:
        Cumulative win count across all leaderboard evaluations.
    avg_regret:
        Running average regret from leaderboard entries.
    avg_convergence_speed:
        Running average convergence speed from leaderboard entries.
    """

    plugin_name: str
    plugin_class_name: str = ""
    n_benchmarks: int = 0
    avg_score: float = 0.0
    failure_rate: float = 0.0
    last_benchmark_time: float = 0.0
    status: MarketplaceStatus = MarketplaceStatus.ACTIVE
    probation_benchmarks: int = 0
    total_wins: int = 0
    avg_regret: float = 0.0
    avg_convergence_speed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "plugin_name": self.plugin_name,
            "plugin_class_name": self.plugin_class_name,
            "n_benchmarks": self.n_benchmarks,
            "avg_score": self.avg_score,
            "failure_rate": self.failure_rate,
            "last_benchmark_time": self.last_benchmark_time,
            "status": self.status.value,
            "probation_benchmarks": self.probation_benchmarks,
            "total_wins": self.total_wins,
            "avg_regret": self.avg_regret,
            "avg_convergence_speed": self.avg_convergence_speed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarketplaceEntry:
        """Deserialize from a plain dictionary."""
        return cls(
            plugin_name=data["plugin_name"],
            plugin_class_name=data.get("plugin_class_name", ""),
            n_benchmarks=data.get("n_benchmarks", 0),
            avg_score=data.get("avg_score", 0.0),
            failure_rate=data.get("failure_rate", 0.0),
            last_benchmark_time=data.get("last_benchmark_time", 0.0),
            status=MarketplaceStatus(data.get("status", "active")),
            probation_benchmarks=data.get("probation_benchmarks", 0),
            total_wins=data.get("total_wins", 0),
            avg_regret=data.get("avg_regret", 0.0),
            avg_convergence_speed=data.get("avg_convergence_speed", 0.0),
        )


# ── Marketplace ──────────────────────────────────────────────────────


class Marketplace:
    """Optimizer Marketplace — plugin registry with health tracking and auto-culling.

    Wraps a :class:`PluginRegistry` to add:

    * Per-plugin health metrics (failure rate, score, regret, convergence).
    * Automatic probation and retirement via :class:`CullPolicy`.
    * Integrated benchmarking through :class:`BenchmarkRunner`.
    * A global leaderboard refreshed from accumulated results.

    Usage::

        mp = Marketplace()
        mp.submit_plugin(MyOptimiser)
        mp.benchmark_plugin("my_optimiser", scenarios)
        lb = mp.refresh_leaderboard()
    """

    def __init__(
        self,
        registry: PluginRegistry | None = None,
        cull_policy: CullPolicy | None = None,
    ) -> None:
        self._registry = registry or PluginRegistry()
        self._cull_policy = cull_policy or CullPolicy()
        self._entries: dict[str, MarketplaceEntry] = {}
        self._leaderboard: Leaderboard | None = None
        self._all_results: list[BenchmarkResult] = []

    # -- submission --------------------------------------------------------

    def submit_plugin(
        self,
        plugin_class: type[AlgorithmPlugin],
    ) -> MarketplaceEntry:
        """Register a new plugin and create its marketplace entry.

        Parameters
        ----------
        plugin_class:
            A subclass of :class:`AlgorithmPlugin` to register.

        Returns
        -------
        MarketplaceEntry
            The freshly created entry for the submitted plugin.

        Raises
        ------
        TypeError
            If *plugin_class* is not a valid ``AlgorithmPlugin`` subclass.
        ValueError
            If a plugin with the same name is already registered.
        """
        self._registry.register(plugin_class)
        instance = plugin_class()
        plugin_name = instance.name()
        class_name = f"{plugin_class.__module__}.{plugin_class.__qualname__}"
        entry = MarketplaceEntry(
            plugin_name=plugin_name,
            plugin_class_name=class_name,
        )
        self._entries[plugin_name] = entry
        return entry

    # -- queries -----------------------------------------------------------

    def get_entry(self, plugin_name: str) -> MarketplaceEntry:
        """Return the marketplace entry for *plugin_name*.

        Raises
        ------
        KeyError
            If no entry exists for the given name.
        """
        if plugin_name not in self._entries:
            raise KeyError(
                f"No marketplace entry for '{plugin_name}'. "
                f"Available: {list(self._entries.keys())}"
            )
        return self._entries[plugin_name]

    def list_active(self) -> list[str]:
        """Return names of all non-retired plugins."""
        return [
            name
            for name, entry in self._entries.items()
            if entry.status != MarketplaceStatus.RETIRED
        ]

    def list_all(self) -> list[MarketplaceEntry]:
        """Return all marketplace entries regardless of status."""
        return list(self._entries.values())

    # -- benchmarking ------------------------------------------------------

    def benchmark_plugin(
        self,
        plugin_name: str,
        scenarios: list[tuple[str, CampaignSnapshot]],
        n_iterations: int = 50,
        seed: int = 42,
    ) -> list[BenchmarkResult]:
        """Benchmark a single plugin on the provided scenarios.

        After running, the plugin's health metrics are updated and the
        cull policy is evaluated.

        Parameters
        ----------
        plugin_name:
            Name of the plugin to benchmark.
        scenarios:
            List of ``(scenario_name, snapshot)`` tuples.
        n_iterations:
            Maximum iterations per scenario.
        seed:
            Base random seed for reproducibility.

        Returns
        -------
        list[BenchmarkResult]
            One result per scenario.

        Raises
        ------
        KeyError
            If the plugin is not in the marketplace.
        """
        entry = self._entries[plugin_name]
        plugin = self._registry.get(plugin_name)
        runner = BenchmarkRunner({plugin_name: plugin})
        results = runner.run_all_scenarios(scenarios, n_iterations, seed)

        lb_entry: LeaderboardEntry | None = None
        if results:
            leaderboard = runner.build_leaderboard(results)
            if leaderboard.entries:
                lb_entry = leaderboard.entries[0]

        self._update_health_metrics(entry, results, lb_entry)
        self._check_cull(entry)
        self._all_results.extend(results)
        return results

    def benchmark_all(
        self,
        scenarios: list[tuple[str, CampaignSnapshot]],
        n_iterations: int = 50,
        seed: int = 42,
    ) -> list[BenchmarkResult]:
        """Benchmark every active plugin on the provided scenarios.

        Parameters
        ----------
        scenarios:
            List of ``(scenario_name, snapshot)`` tuples.
        n_iterations:
            Maximum iterations per scenario.
        seed:
            Base random seed for reproducibility.

        Returns
        -------
        list[BenchmarkResult]
            Combined results for all active plugins.
        """
        all_results: list[BenchmarkResult] = []
        for name in list(self.list_active()):
            results = self.benchmark_plugin(name, scenarios, n_iterations, seed)
            all_results.extend(results)
        return all_results

    # -- leaderboard -------------------------------------------------------

    def refresh_leaderboard(self) -> Leaderboard:
        """Rebuild the global leaderboard from all accumulated results.

        Only results belonging to currently active (non-retired) plugins
        are included.

        Returns
        -------
        Leaderboard
            The refreshed leaderboard.
        """
        active_names = set(self.list_active())
        filtered = [r for r in self._all_results if r.backend_name in active_names]
        runner = BenchmarkRunner({})
        self._leaderboard = runner.build_leaderboard(filtered)
        return self._leaderboard

    def get_leaderboard(self) -> Leaderboard | None:
        """Return the most recently computed leaderboard, or ``None``."""
        return self._leaderboard

    # -- manual status management ------------------------------------------

    def retire_plugin(self, plugin_name: str) -> None:
        """Manually retire a plugin, removing it from active benchmarking."""
        entry = self.get_entry(plugin_name)
        entry.status = MarketplaceStatus.RETIRED

    def reinstate_plugin(self, plugin_name: str) -> None:
        """Reinstate a previously retired or probated plugin to active status."""
        entry = self.get_entry(plugin_name)
        entry.status = MarketplaceStatus.ACTIVE
        entry.probation_benchmarks = 0

    # -- health & culling (private) ----------------------------------------

    def _check_cull(self, entry: MarketplaceEntry) -> None:
        """Evaluate the cull policy for *entry* and update status accordingly."""
        if entry.n_benchmarks < self._cull_policy.min_benchmarks:
            return  # not enough data yet

        in_violation = self._is_in_violation(entry)

        if entry.status == MarketplaceStatus.ACTIVE and in_violation:
            entry.status = MarketplaceStatus.PROBATION
            entry.probation_benchmarks = 0

        if entry.status == MarketplaceStatus.PROBATION:
            if not in_violation:
                entry.status = MarketplaceStatus.ACTIVE
                entry.probation_benchmarks = 0
            elif entry.probation_benchmarks >= self._cull_policy.grace_benchmarks:
                entry.status = MarketplaceStatus.RETIRED

    def _is_in_violation(self, entry: MarketplaceEntry) -> bool:
        """Return ``True`` if *entry* violates any cull-policy threshold."""
        return (
            entry.failure_rate > self._cull_policy.max_failure_rate
            or entry.avg_score < self._cull_policy.min_avg_score
        )

    def _update_health_metrics(
        self,
        entry: MarketplaceEntry,
        results: list[BenchmarkResult],
        leaderboard_entry: LeaderboardEntry | None,
    ) -> None:
        """Update *entry* health metrics from a fresh batch of results."""
        n_new = len(results)
        if n_new == 0:
            return

        new_failure_rate = _compute_failure_rate_from_results(results)
        entry.failure_rate = (
            (entry.failure_rate * entry.n_benchmarks + new_failure_rate * n_new)
            / (entry.n_benchmarks + n_new)
        )

        if leaderboard_entry is not None:
            entry.avg_score = _incremental_mean(
                entry.avg_score, entry.n_benchmarks, leaderboard_entry.score,
            )
            entry.avg_regret = _incremental_mean(
                entry.avg_regret, entry.n_benchmarks, leaderboard_entry.avg_regret,
            )
            entry.avg_convergence_speed = _incremental_mean(
                entry.avg_convergence_speed,
                entry.n_benchmarks,
                leaderboard_entry.avg_convergence_speed,
            )
            entry.total_wins += leaderboard_entry.win_count

        entry.n_benchmarks += n_new
        if entry.status == MarketplaceStatus.PROBATION:
            entry.probation_benchmarks += n_new
        entry.last_benchmark_time = time.time()

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize marketplace state to a plain dictionary."""
        return {
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
            "cull_policy": self._cull_policy.to_dict(),
            "all_results_count": len(self._all_results),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        registry: PluginRegistry | None = None,
    ) -> Marketplace:
        """Restore a marketplace from a serialized dictionary.

        Note: benchmark results are **not** restored (only the count is
        stored).  Re-run benchmarks to rebuild the leaderboard.
        """
        m = cls(
            registry=registry,
            cull_policy=CullPolicy.from_dict(data["cull_policy"]),
        )
        m._entries = {
            k: MarketplaceEntry.from_dict(v)
            for k, v in data["entries"].items()
        }
        return m

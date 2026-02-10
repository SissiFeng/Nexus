"""Benchmark & Leaderboard module for standardized comparison of optimization backends.

Provides ``BenchmarkRunner`` which simulates running each registered backend on
a set of scenarios (campaign snapshots), measures quality metrics, and produces
a ranked leaderboard.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.plugins.base import AlgorithmPlugin
from optimization_copilot.profiler.profiler import ProblemProfiler


# ── Result dataclasses ────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    """Outcome of running a single backend on a single scenario."""

    backend_name: str
    scenario_name: str
    fingerprint_key: str
    final_best_kpi: float
    convergence_iteration: int
    total_iterations: int
    failure_count: int
    regret: float
    sample_efficiency: float
    wall_time_seconds: float


@dataclass
class LeaderboardEntry:
    """Aggregated performance of one backend across scenarios."""

    backend_name: str
    n_scenarios: int
    avg_rank: float
    win_count: int
    avg_regret: float
    avg_convergence_speed: float
    avg_sample_efficiency: float
    score: float


@dataclass
class Leaderboard:
    """Full leaderboard with per-fingerprint breakdown."""

    entries: list[LeaderboardEntry]
    by_fingerprint: dict[str, list[LeaderboardEntry]]
    total_scenarios: int


# ── Helper utilities ──────────────────────────────────────────────────


def _fingerprint_key(snapshot: CampaignSnapshot) -> str:
    """Produce a human-readable string key from the problem fingerprint."""
    profiler = ProblemProfiler()
    fp = profiler.profile(snapshot)
    return "|".join(
        f"{k}={v}" for k, v in sorted(fp.to_dict().items())
    )


def _param_distance(
    suggested: dict[str, Any],
    observed: dict[str, Any],
    specs: list[ParameterSpec],
) -> float:
    """Normalised Euclidean distance between two parameter configurations.

    Continuous and discrete dimensions are normalised to [0, 1] using their
    bounds.  Categorical dimensions contribute 0 on match and 1 on mismatch.
    """
    total = 0.0
    for spec in specs:
        s_val = suggested.get(spec.name)
        o_val = observed.get(spec.name)
        if spec.type == VariableType.CATEGORICAL:
            total += 0.0 if s_val == o_val else 1.0
        else:
            lo = spec.lower if spec.lower is not None else 0.0
            hi = spec.upper if spec.upper is not None else 1.0
            span = hi - lo
            if span == 0:
                total += 0.0
            else:
                s_norm = (float(s_val) - lo) / span if s_val is not None else 0.5
                o_norm = (float(o_val) - lo) / span if o_val is not None else 0.5
                total += (s_norm - o_norm) ** 2
    return math.sqrt(total)


def _find_nearest(
    suggestion: dict[str, Any],
    pool: list[Observation],
    specs: list[ParameterSpec],
) -> int:
    """Return index into *pool* of the observation closest to *suggestion*."""
    best_idx = 0
    best_dist = float("inf")
    for idx, obs in enumerate(pool):
        d = _param_distance(suggestion, obs.parameters, specs)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def _best_kpi_in(
    observations: list[Observation],
    kpi_name: str,
    maximize: bool,
) -> float:
    """Return the best KPI value across *observations*."""
    values = [
        obs.kpi_values[kpi_name]
        for obs in observations
        if not obs.is_failure and kpi_name in obs.kpi_values
    ]
    if not values:
        return float("-inf") if maximize else float("inf")
    return max(values) if maximize else min(values)


def _is_better(
    candidate: float,
    reference: float,
    maximize: bool,
) -> bool:
    """Return True if *candidate* is better than *reference*."""
    if maximize:
        return candidate > reference
    return candidate < reference


# ── BenchmarkRunner ───────────────────────────────────────────────────


class BenchmarkRunner:
    """Run standardised benchmarks of optimisation backends on scenario suites.

    Parameters
    ----------
    backends:
        Mapping of ``backend_name -> AlgorithmPlugin`` instance.
    """

    def __init__(self, backends: dict[str, AlgorithmPlugin]) -> None:
        self._backends = dict(backends)

    # -- single scenario ---------------------------------------------------

    def run_scenario(
        self,
        scenario_name: str,
        snapshot: CampaignSnapshot,
        backend_name: str,
        plugin: AlgorithmPlugin,
        n_iterations: int = 50,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Simulate running *plugin* on *snapshot* for *n_iterations* steps.

        The simulation proceeds as follows:

        1. Seed the plugin with the first ``min(3, len(observations))``
           observations.
        2. At each iteration the plugin is fitted on current data, asked to
           suggest a point, and the nearest unused observation from the full
           dataset is "observed" and added to the training set.
        3. Metrics (convergence, regret, sample efficiency) are recorded.
        """
        rng = random.Random(seed)
        specs = snapshot.parameter_specs
        kpi_name = snapshot.objective_names[0]
        maximize = snapshot.objective_directions[0] == "maximize"

        fp_key = _fingerprint_key(snapshot)

        # Split into seed data and pool
        all_obs = list(snapshot.observations)
        n_seed = min(3, len(all_obs))
        seed_obs = all_obs[:n_seed]
        pool = list(all_obs[n_seed:])

        # Clamp iterations to available pool size
        effective_iters = min(n_iterations, len(pool))

        current_data: list[Observation] = list(seed_obs)
        best_so_far = _best_kpi_in(current_data, kpi_name, maximize)
        best_possible = _best_kpi_in(all_obs, kpi_name, maximize)

        best_trace: list[float] = [best_so_far]
        improvement_count = 0
        failure_count = sum(1 for o in seed_obs if o.is_failure)

        for i in range(effective_iters):
            # Fit and suggest
            plugin.fit(current_data, specs)
            suggestions = plugin.suggest(n_suggestions=1, seed=seed + i)
            suggestion = suggestions[0] if suggestions else {}

            # Find nearest unused observation
            if not pool:
                break
            nearest_idx = _find_nearest(suggestion, pool, specs)
            picked = pool.pop(nearest_idx)
            current_data.append(picked)

            if picked.is_failure:
                failure_count += 1
                best_trace.append(best_so_far)
                continue

            kpi_val = picked.kpi_values.get(kpi_name)
            if kpi_val is not None and _is_better(kpi_val, best_so_far, maximize):
                improvement_count += 1
                best_so_far = kpi_val

            best_trace.append(best_so_far)

        final_best = best_so_far

        # Convergence iteration: first iteration where best_so_far is within
        # 5% of the final best value.
        convergence_iter = effective_iters  # default: never converged
        if final_best != 0.0:
            threshold = abs(final_best) * 0.05
        else:
            threshold = 1e-9
        for idx, val in enumerate(best_trace):
            if abs(val - final_best) <= threshold:
                convergence_iter = idx
                break

        # Regret: gap between best possible and achieved
        if maximize:
            regret = max(0.0, best_possible - final_best)
        else:
            regret = max(0.0, final_best - best_possible)

        # Normalise regret by the range of KPIs in the dataset
        kpi_values = [
            o.kpi_values[kpi_name]
            for o in all_obs
            if not o.is_failure and kpi_name in o.kpi_values
        ]
        if kpi_values:
            kpi_range = max(kpi_values) - min(kpi_values)
            if kpi_range > 0:
                regret = regret / kpi_range

        # Sample efficiency: fraction of iterations that improved best
        sample_efficiency = (
            improvement_count / effective_iters if effective_iters > 0 else 0.0
        )

        return BenchmarkResult(
            backend_name=backend_name,
            scenario_name=scenario_name,
            fingerprint_key=fp_key,
            final_best_kpi=final_best,
            convergence_iteration=convergence_iter,
            total_iterations=effective_iters,
            failure_count=failure_count,
            regret=regret,
            sample_efficiency=sample_efficiency,
            wall_time_seconds=float(effective_iters),
        )

    # -- batch execution ---------------------------------------------------

    def run_all_scenarios(
        self,
        scenarios: list[tuple[str, CampaignSnapshot]],
        n_iterations: int = 50,
        seed: int = 42,
    ) -> list[BenchmarkResult]:
        """Run every registered backend on every scenario.

        Parameters
        ----------
        scenarios:
            List of ``(scenario_name, snapshot)`` tuples.
        n_iterations:
            Maximum number of optimisation iterations per scenario.
        seed:
            Base random seed (incremented per backend for independence).

        Returns
        -------
        list[BenchmarkResult]
            One result per ``(backend, scenario)`` pair.
        """
        results: list[BenchmarkResult] = []
        for s_idx, (scenario_name, snapshot) in enumerate(scenarios):
            for b_idx, (backend_name, plugin) in enumerate(self._backends.items()):
                run_seed = seed + s_idx * 1000 + b_idx
                result = self.run_scenario(
                    scenario_name=scenario_name,
                    snapshot=snapshot,
                    backend_name=backend_name,
                    plugin=plugin,
                    n_iterations=n_iterations,
                    seed=run_seed,
                )
                results.append(result)
        return results

    # -- leaderboard construction ------------------------------------------

    def build_leaderboard(
        self,
        results: list[BenchmarkResult],
    ) -> Leaderboard:
        """Aggregate benchmark results into a ranked leaderboard.

        Scoring formula::

            score = 0.4 * (1 / avg_rank)
                  + 0.3 * win_rate
                  + 0.2 * avg_sample_efficiency
                  + 0.1 * (1 - avg_regret_normalised)
        """
        # Group results by scenario
        scenarios: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            scenarios.setdefault(r.scenario_name, []).append(r)

        # Group results by fingerprint
        fp_scenarios: dict[str, dict[str, list[BenchmarkResult]]] = {}
        for r in results:
            fp_scenarios.setdefault(r.fingerprint_key, {}).setdefault(
                r.scenario_name, []
            ).append(r)

        # Collect all backend names
        all_backends = sorted({r.backend_name for r in results})

        # Per-scenario ranking (lower KPI is not always better -- we need
        # the direction, but BenchmarkResult stores final_best_kpi which is
        # already the best in the correct direction.  Higher final_best_kpi
        # is better for maximise, lower for minimise.  However we don't
        # store the direction.  Convention: rank by final_best_kpi descending
        # (higher is better).  The run_scenario already ensures final_best_kpi
        # tracks the best value.  For a fair ranking we assume higher is
        # better (if the user minimises, the "best" is the smallest, so
        # final_best_kpi is smallest, and ranking descending still gives rank 1
        # to the backend that achieved the lowest value... actually no.
        # We need to rank such that the best final_best_kpi gets rank 1.)
        #
        # Since run_scenario tracks best_so_far correctly per direction, the
        # backend with the best final_best_kpi has the value closest to
        # best_possible.  We can rank by regret ascending (lower regret = better).

        # Accumulate stats per backend
        backend_ranks: dict[str, list[float]] = {b: [] for b in all_backends}
        backend_wins: dict[str, int] = {b: 0 for b in all_backends}
        backend_regrets: dict[str, list[float]] = {b: [] for b in all_backends}
        backend_convergences: dict[str, list[float]] = {b: [] for b in all_backends}
        backend_efficiencies: dict[str, list[float]] = {b: [] for b in all_backends}

        for _scenario_name, scenario_results in scenarios.items():
            # Sort by regret ascending (best = lowest regret)
            sorted_results = sorted(scenario_results, key=lambda r: r.regret)
            # Assign ranks (1-based, tie-aware)
            rank = 1
            for i, r in enumerate(sorted_results):
                if i > 0 and r.regret != sorted_results[i - 1].regret:
                    rank = i + 1
                backend_ranks[r.backend_name].append(float(rank))

            # Winner = rank 1
            best_regret = sorted_results[0].regret
            for r in sorted_results:
                if r.regret == best_regret:
                    backend_wins[r.backend_name] += 1

            for r in scenario_results:
                backend_regrets[r.backend_name].append(r.regret)
                speed = (
                    r.convergence_iteration / r.total_iterations
                    if r.total_iterations > 0
                    else 1.0
                )
                backend_convergences[r.backend_name].append(speed)
                backend_efficiencies[r.backend_name].append(r.sample_efficiency)

        n_scenarios_total = len(scenarios)

        # Compute max regret for normalisation
        all_regrets = [r.regret for r in results]
        max_regret = max(all_regrets) if all_regrets else 1.0
        if max_regret == 0.0:
            max_regret = 1.0

        # Build global entries
        global_entries: list[LeaderboardEntry] = []
        for b in all_backends:
            n_scen = len(backend_ranks[b])
            avg_rank = (
                sum(backend_ranks[b]) / n_scen if n_scen > 0 else float("inf")
            )
            avg_regret = (
                sum(backend_regrets[b]) / n_scen if n_scen > 0 else 1.0
            )
            avg_convergence = (
                sum(backend_convergences[b]) / n_scen if n_scen > 0 else 1.0
            )
            avg_efficiency = (
                sum(backend_efficiencies[b]) / n_scen if n_scen > 0 else 0.0
            )
            win_rate = backend_wins[b] / n_scenarios_total if n_scenarios_total > 0 else 0.0
            avg_regret_norm = avg_regret / max_regret

            score = (
                0.4 * (1.0 / avg_rank if avg_rank > 0 else 0.0)
                + 0.3 * win_rate
                + 0.2 * avg_efficiency
                + 0.1 * (1.0 - avg_regret_norm)
            )
            global_entries.append(
                LeaderboardEntry(
                    backend_name=b,
                    n_scenarios=n_scen,
                    avg_rank=round(avg_rank, 3),
                    win_count=backend_wins[b],
                    avg_regret=round(avg_regret, 6),
                    avg_convergence_speed=round(avg_convergence, 3),
                    avg_sample_efficiency=round(avg_efficiency, 3),
                    score=round(score, 6),
                )
            )

        global_entries.sort(key=lambda e: e.score, reverse=True)

        # Build per-fingerprint entries
        by_fingerprint: dict[str, list[LeaderboardEntry]] = {}
        for fp_key, fp_scens in fp_scenarios.items():
            fp_results_flat = [
                r for scen_results in fp_scens.values() for r in scen_results
            ]
            fp_backends = sorted({r.backend_name for r in fp_results_flat})
            fp_ranks: dict[str, list[float]] = {b: [] for b in fp_backends}
            fp_wins_map: dict[str, int] = {b: 0 for b in fp_backends}
            fp_regrets: dict[str, list[float]] = {b: [] for b in fp_backends}
            fp_convergences: dict[str, list[float]] = {b: [] for b in fp_backends}
            fp_efficiencies: dict[str, list[float]] = {b: [] for b in fp_backends}

            for _sname, sresults in fp_scens.items():
                sorted_sr = sorted(sresults, key=lambda r: r.regret)
                rank = 1
                for i, r in enumerate(sorted_sr):
                    if i > 0 and r.regret != sorted_sr[i - 1].regret:
                        rank = i + 1
                    fp_ranks[r.backend_name].append(float(rank))
                best_reg = sorted_sr[0].regret
                for r in sorted_sr:
                    if r.regret == best_reg:
                        fp_wins_map[r.backend_name] += 1
                for r in sresults:
                    fp_regrets[r.backend_name].append(r.regret)
                    spd = (
                        r.convergence_iteration / r.total_iterations
                        if r.total_iterations > 0
                        else 1.0
                    )
                    fp_convergences[r.backend_name].append(spd)
                    fp_efficiencies[r.backend_name].append(r.sample_efficiency)

            fp_n_scen = len(fp_scens)
            fp_max_regret = max(
                (r.regret for r in fp_results_flat), default=1.0
            )
            if fp_max_regret == 0.0:
                fp_max_regret = 1.0

            fp_entries: list[LeaderboardEntry] = []
            for b in fp_backends:
                n_s = len(fp_ranks[b])
                a_rank = sum(fp_ranks[b]) / n_s if n_s > 0 else float("inf")
                a_regret = sum(fp_regrets[b]) / n_s if n_s > 0 else 1.0
                a_conv = sum(fp_convergences[b]) / n_s if n_s > 0 else 1.0
                a_eff = sum(fp_efficiencies[b]) / n_s if n_s > 0 else 0.0
                w_rate = fp_wins_map[b] / fp_n_scen if fp_n_scen > 0 else 0.0
                a_regret_n = a_regret / fp_max_regret

                sc = (
                    0.4 * (1.0 / a_rank if a_rank > 0 else 0.0)
                    + 0.3 * w_rate
                    + 0.2 * a_eff
                    + 0.1 * (1.0 - a_regret_n)
                )
                fp_entries.append(
                    LeaderboardEntry(
                        backend_name=b,
                        n_scenarios=n_s,
                        avg_rank=round(a_rank, 3),
                        win_count=fp_wins_map[b],
                        avg_regret=round(a_regret, 6),
                        avg_convergence_speed=round(a_conv, 3),
                        avg_sample_efficiency=round(a_eff, 3),
                        score=round(sc, 6),
                    )
                )
            fp_entries.sort(key=lambda e: e.score, reverse=True)
            by_fingerprint[fp_key] = fp_entries

        return Leaderboard(
            entries=global_entries,
            by_fingerprint=by_fingerprint,
            total_scenarios=n_scenarios_total,
        )

    # -- formatting --------------------------------------------------------

    @staticmethod
    def format_leaderboard(leaderboard: Leaderboard) -> str:
        """Return a human-readable table representation of the leaderboard."""
        lines: list[str] = []
        lines.append(
            f"=== Optimization Backend Leaderboard ({leaderboard.total_scenarios} scenarios) ==="
        )
        lines.append("")

        # Header
        header = (
            f"{'Rank':<6}"
            f"{'Backend':<25}"
            f"{'Score':>8}"
            f"{'AvgRank':>9}"
            f"{'Wins':>6}"
            f"{'AvgRegret':>11}"
            f"{'ConvSpeed':>11}"
            f"{'Efficiency':>12}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for rank_idx, entry in enumerate(leaderboard.entries, 1):
            row = (
                f"{rank_idx:<6}"
                f"{entry.backend_name:<25}"
                f"{entry.score:>8.4f}"
                f"{entry.avg_rank:>9.2f}"
                f"{entry.win_count:>6}"
                f"{entry.avg_regret:>11.6f}"
                f"{entry.avg_convergence_speed:>11.3f}"
                f"{entry.avg_sample_efficiency:>12.3f}"
            )
            lines.append(row)

        # Per-fingerprint breakdowns
        if leaderboard.by_fingerprint:
            lines.append("")
            lines.append("--- Breakdown by Problem Type ---")
            for fp_key, fp_entries in sorted(leaderboard.by_fingerprint.items()):
                lines.append("")
                lines.append(f"  [{fp_key}]")
                for rank_idx, entry in enumerate(fp_entries, 1):
                    lines.append(
                        f"    {rank_idx}. {entry.backend_name:<22}"
                        f" score={entry.score:.4f}"
                        f" regret={entry.avg_regret:.6f}"
                        f" wins={entry.win_count}"
                    )

        return "\n".join(lines)

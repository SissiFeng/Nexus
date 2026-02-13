"""Systematic multi-seed evaluation with statistical testing and ranking.

Evaluates all (backend x function x seed) combinations, computes aggregate
rankings, and uses the Wilcoxon signed-rank test for pairwise comparison.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.benchmark.direct_runner import (
    DirectBenchmarkResult,
    DirectBenchmarkRunner,
)
from optimization_copilot.benchmark.functions import BenchmarkFunction


# -- Configuration ------------------------------------------------------------


@dataclass
class EvaluationConfig:
    """Configuration for systematic evaluation."""

    seeds: list[int] = field(default_factory=lambda: list(range(42, 62)))  # 20 seeds
    n_init_fraction: float = 0.2
    budget_override: int | None = None  # None = auto from dimensionality


# -- Result data models -------------------------------------------------------


@dataclass
class BackendRanking:
    """Ranking for a single backend across all functions and seeds."""

    name: str
    avg_rank: float
    median_rank: float
    avg_log10_regret: float
    avg_auc: float
    win_count: int  # Number of (function, seed) pairs where this backend was best
    total_comparisons: int
    per_function_ranks: dict[str, float] = field(default_factory=dict)


@dataclass
class WilcoxonResult:
    """Result of a Wilcoxon signed-rank test between two backends."""

    backend_a: str
    backend_b: str
    statistic: float
    p_value: float
    n_pairs: int
    significant: bool  # p < 0.05


@dataclass
class EvaluationReport:
    """Full evaluation report with rankings and statistical tests."""

    rankings: list[BackendRanking]
    pairwise_tests: list[WilcoxonResult]
    n_functions: int
    n_seeds: int
    n_backends: int
    total_runs: int
    total_time_s: float
    raw_results: dict[str, dict[str, list[DirectBenchmarkResult]]] = field(
        default_factory=dict
    )  # {backend: {function: [results per seed]}}


# -- Pure-Python Wilcoxon signed-rank test ------------------------------------


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank(
    x: list[float], y: list[float]
) -> tuple[float, float, int]:
    """Wilcoxon signed-rank test (paired, two-sided).

    Parameters
    ----------
    x, y
        Paired samples of equal length.

    Returns
    -------
    statistic
        The W+ statistic (sum of positive ranks).
    p_value
        Two-sided p-value using normal approximation.
    n_effective
        Number of non-zero differences (ties with zero excluded).
    """
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")

    # Compute differences and exclude zeros
    diffs = [(xi - yi) for xi, yi in zip(x, y)]
    nonzero = [(abs(d), 1 if d > 0 else -1) for d in diffs if abs(d) > 1e-12]
    n = len(nonzero)

    if n == 0:
        return 0.0, 1.0, 0

    # Rank by absolute difference
    nonzero.sort(key=lambda t: t[0])

    # Assign ranks (average ties)
    ranks: list[tuple[float, int]] = []
    i = 0
    while i < n:
        j = i
        while j < n and abs(nonzero[j][0] - nonzero[i][0]) < 1e-12:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks.append((avg_rank, nonzero[k][1]))
        i = j

    # W+ = sum of ranks for positive differences
    w_plus = sum(r for r, sign in ranks if sign > 0)

    # Normal approximation for p-value
    mean = n * (n + 1) / 4.0
    # Tie correction
    std_dev = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)

    if std_dev < 1e-12:
        return w_plus, 1.0, n

    z = (w_plus - mean) / std_dev
    # Two-sided p-value
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))

    return w_plus, p_value, n


# -- Systematic Evaluator -----------------------------------------------------


class SystematicEvaluator:
    """Run all (backend x function x seed) combinations and compute rankings.

    Parameters
    ----------
    config
        Evaluation configuration (seeds, budget, etc.).
    """

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self._config = config or EvaluationConfig()
        self._runner = DirectBenchmarkRunner(
            n_init_fraction=self._config.n_init_fraction
        )

    def evaluate(
        self,
        backend_factories: dict[str, type],
        benchmarks: list[BenchmarkFunction],
    ) -> EvaluationReport:
        """Run the full systematic evaluation.

        Parameters
        ----------
        backend_factories
            Mapping of backend name -> AlgorithmPlugin class.
        benchmarks
            List of benchmark functions to evaluate on.

        Returns
        -------
        EvaluationReport
            Complete report with rankings, pairwise tests, and raw results.
        """
        t_start = time.monotonic()
        seeds = self._config.seeds
        raw: dict[str, dict[str, list[DirectBenchmarkResult]]] = {}
        total_runs = 0

        for bname, bfactory in backend_factories.items():
            raw[bname] = {}
            for benchmark in benchmarks:
                results = self._runner.run_multi_seed(
                    plugin_factory=bfactory,
                    benchmark=benchmark,
                    seeds=seeds,
                    budget=self._config.budget_override,
                )
                raw[bname][benchmark.name] = results
                total_runs += len(results)

        # Compute rankings
        rankings = self._compute_rankings(raw, benchmarks, seeds)

        # Pairwise Wilcoxon tests (top backend vs all others)
        pairwise = self._compute_pairwise_tests(raw, benchmarks, seeds)

        total_time = time.monotonic() - t_start

        return EvaluationReport(
            rankings=rankings,
            pairwise_tests=pairwise,
            n_functions=len(benchmarks),
            n_seeds=len(seeds),
            n_backends=len(backend_factories),
            total_runs=total_runs,
            total_time_s=total_time,
            raw_results=raw,
        )

    def _compute_rankings(
        self,
        raw: dict[str, dict[str, list[DirectBenchmarkResult]]],
        benchmarks: list[BenchmarkFunction],
        seeds: list[int],
    ) -> list[BackendRanking]:
        """Compute per-(function, seed) ranks and aggregate."""
        backend_names = list(raw.keys())
        n_backends = len(backend_names)

        # Collect per-(function, seed) log10-regrets
        all_ranks: dict[str, list[float]] = {b: [] for b in backend_names}
        all_regrets: dict[str, list[float]] = {b: [] for b in backend_names}
        all_aucs: dict[str, list[float]] = {b: [] for b in backend_names}
        per_func_ranks: dict[str, dict[str, list[float]]] = {
            b: {} for b in backend_names
        }
        win_counts: dict[str, int] = {b: 0 for b in backend_names}
        total_comparisons = 0

        for benchmark in benchmarks:
            fname = benchmark.name
            for b in backend_names:
                per_func_ranks[b][fname] = []

            for seed_idx in range(len(seeds)):
                # Get log10_regret for each backend on this (function, seed)
                regrets = {}
                for bname in backend_names:
                    results = raw[bname].get(fname, [])
                    if seed_idx < len(results):
                        regrets[bname] = results[seed_idx].log10_regret
                        all_regrets[bname].append(results[seed_idx].log10_regret)
                        all_aucs[bname].append(results[seed_idx].auc_normalized)
                    else:
                        regrets[bname] = float("inf")

                # Rank: sort by regret (lower = better = rank 1)
                sorted_backends = sorted(regrets, key=lambda b: regrets[b])

                # Assign ranks with ties
                i = 0
                while i < len(sorted_backends):
                    j = i
                    val = regrets[sorted_backends[i]]
                    while j < len(sorted_backends) and abs(regrets[sorted_backends[j]] - val) < 1e-12:
                        j += 1
                    avg_rank = (i + 1 + j) / 2.0
                    for k in range(i, j):
                        b = sorted_backends[k]
                        all_ranks[b].append(avg_rank)
                        per_func_ranks[b][fname].append(avg_rank)
                    i = j

                # Win count: rank 1 wins
                if sorted_backends:
                    best_val = regrets[sorted_backends[0]]
                    for b in sorted_backends:
                        if abs(regrets[b] - best_val) < 1e-12:
                            win_counts[b] += 1
                total_comparisons += 1

        # Build ranking objects
        rankings: list[BackendRanking] = []
        for bname in backend_names:
            ranks = all_ranks[bname]
            avg_rank = sum(ranks) / len(ranks) if ranks else float(n_backends)
            sorted_ranks = sorted(ranks)
            median_rank = sorted_ranks[len(sorted_ranks) // 2] if sorted_ranks else float(n_backends)
            avg_regret = sum(all_regrets[bname]) / len(all_regrets[bname]) if all_regrets[bname] else float("inf")
            avg_auc = sum(all_aucs[bname]) / len(all_aucs[bname]) if all_aucs[bname] else 1.0

            pf_ranks = {
                f: (sum(rs) / len(rs) if rs else float(n_backends))
                for f, rs in per_func_ranks[bname].items()
            }

            rankings.append(BackendRanking(
                name=bname,
                avg_rank=avg_rank,
                median_rank=median_rank,
                avg_log10_regret=avg_regret,
                avg_auc=avg_auc,
                win_count=win_counts[bname],
                total_comparisons=total_comparisons,
                per_function_ranks=pf_ranks,
            ))

        # Sort by avg_rank (best first)
        rankings.sort(key=lambda r: r.avg_rank)
        return rankings

    def _compute_pairwise_tests(
        self,
        raw: dict[str, dict[str, list[DirectBenchmarkResult]]],
        benchmarks: list[BenchmarkFunction],
        seeds: list[int],
    ) -> list[WilcoxonResult]:
        """Wilcoxon signed-rank test: best backend vs each other."""
        backend_names = list(raw.keys())
        if len(backend_names) < 2:
            return []

        # Collect per-(function, seed) log10_regrets for each backend
        regret_vectors: dict[str, list[float]] = {b: [] for b in backend_names}
        for benchmark in benchmarks:
            for seed_idx in range(len(seeds)):
                for bname in backend_names:
                    results = raw[bname].get(benchmark.name, [])
                    if seed_idx < len(results):
                        regret_vectors[bname].append(results[seed_idx].log10_regret)
                    else:
                        regret_vectors[bname].append(10.0)  # penalty

        # Find best backend (lowest average regret)
        avg_regrets = {
            b: (sum(v) / len(v) if v else float("inf"))
            for b, v in regret_vectors.items()
        }
        best = min(avg_regrets, key=lambda b: avg_regrets[b])

        # Test best vs each other
        results: list[WilcoxonResult] = []
        for bname in backend_names:
            if bname == best:
                continue
            x = regret_vectors[best]
            y = regret_vectors[bname]
            stat, p_val, n_eff = wilcoxon_signed_rank(x, y)
            results.append(WilcoxonResult(
                backend_a=best,
                backend_b=bname,
                statistic=stat,
                p_value=p_val,
                n_pairs=n_eff,
                significant=p_val < 0.05,
            ))

        results.sort(key=lambda r: r.p_value)
        return results


# -- Report generation --------------------------------------------------------


def generate_report(report: EvaluationReport) -> str:
    """Generate a Markdown report from evaluation results.

    Returns
    -------
    str
        Markdown-formatted evaluation report.
    """
    lines: list[str] = []
    lines.append("# Systematic Benchmark Evaluation Report\n")
    lines.append(f"**Backends**: {report.n_backends} | "
                 f"**Functions**: {report.n_functions} | "
                 f"**Seeds**: {report.n_seeds} | "
                 f"**Total runs**: {report.total_runs} | "
                 f"**Time**: {report.total_time_s:.1f}s\n")

    # Rankings table
    lines.append("## Overall Rankings\n")
    lines.append("| Rank | Backend | Avg Rank | Median Rank | Avg log10(regret) | Avg AUC | Wins | Win Rate |")
    lines.append("|------|---------|----------|-------------|-------------------|---------|------|----------|")
    for i, r in enumerate(report.rankings, 1):
        win_rate = r.win_count / max(r.total_comparisons, 1) * 100
        lines.append(
            f"| {i} | {r.name} | {r.avg_rank:.2f} | {r.median_rank:.1f} | "
            f"{r.avg_log10_regret:.3f} | {r.avg_auc:.3f} | "
            f"{r.win_count} | {win_rate:.1f}% |"
        )

    # Per-function ranks
    if report.rankings:
        lines.append("\n## Per-Function Average Ranks\n")
        func_names = sorted(report.rankings[0].per_function_ranks.keys())
        header = "| Backend | " + " | ".join(func_names) + " |"
        separator = "|---------|" + "|".join("-" * max(8, len(f) + 2) for f in func_names) + "|"
        lines.append(header)
        lines.append(separator)
        for r in report.rankings:
            row = f"| {r.name} |"
            for f in func_names:
                rank = r.per_function_ranks.get(f, float("nan"))
                row += f" {rank:.1f} |"
            lines.append(row)

    # Pairwise tests
    if report.pairwise_tests:
        lines.append("\n## Pairwise Wilcoxon Signed-Rank Tests\n")
        best = report.pairwise_tests[0].backend_a
        lines.append(f"Testing **{best}** (best overall) vs each competitor:\n")
        lines.append("| Competitor | W+ | p-value | n | Significant (p<0.05) |")
        lines.append("|------------|-----|---------|---|---------------------|")
        for t in report.pairwise_tests:
            sig = "Yes" if t.significant else "No"
            lines.append(
                f"| {t.backend_b} | {t.statistic:.1f} | {t.p_value:.4f} | "
                f"{t.n_pairs} | {sig} |"
            )

    lines.append("")
    return "\n".join(lines)

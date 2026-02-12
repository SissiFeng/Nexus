"""Ablation runner for heteroscedastic vs homoscedastic GP comparison.

Provides:
- AblationResult: structured result of a hetero-vs-homo comparison
- AblationRunner: runs the comparison on any ExperimentalBenchmark

This module wires together v4 (HeteroscedasticGP with per-point noise)
and v5 (offline replay case studies) to answer the key question:

    Does per-point noise modelling (heteroscedastic GP) improve
    optimisation performance over a standard homoscedastic GP?

Usage::

    from optimization_copilot.case_studies.zinc.benchmark import ZincBenchmark
    from optimization_copilot.case_studies.ablation import AblationRunner

    benchmark = ZincBenchmark(n_train=100, seed=42)
    runner = AblationRunner(benchmark)
    result = runner.run_hetero_vs_homo(budget=30, n_repeats=15)
    print(result.summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.backends.builtin import GaussianProcessBO
from optimization_copilot.backends.gp_heteroscedastic import HeteroscedasticGP
from optimization_copilot.case_studies.base import ExperimentalBenchmark
from optimization_copilot.case_studies.evaluator import (
    CaseStudyEvaluator,
    ComparisonResult,
    PerformanceMetrics,
)
from optimization_copilot.case_studies.statistics import (
    compute_effect_size,
    wilcoxon_signed_rank_test,
)
from optimization_copilot.plugins.base import AlgorithmPlugin


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    """Structured result of a heteroscedastic vs homoscedastic GP ablation.

    Attributes
    ----------
    comparison : ComparisonResult
        Full comparison result from the evaluator.
    statistical_test : dict
        Wilcoxon signed-rank test result (statistic, p_value, effect_size).
    cohens_d : float
        Cohen's d effect size for the paired comparison.
    hetero_wins : bool
        True if the heteroscedastic GP has a better median best_value.
    significant : bool
        True if p_value < alpha (default 0.05).
    hetero_median : float
        Median best_value across repeats for the heteroscedastic GP.
    homo_median : float
        Median best_value across repeats for the homoscedastic GP.
    noise_impact : list[dict] | None
        Per-point noise impact diagnostics from the last HeteroscedasticGP run.
    summary : str
        Human-readable summary of the ablation result.
    """

    comparison: ComparisonResult
    statistical_test: dict[str, float]
    cohens_d: float
    hetero_wins: bool
    significant: bool
    hetero_median: float
    homo_median: float
    noise_impact: list[dict[str, Any]] | None = None
    summary: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    """Return the median of a sorted list of floats."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# AblationRunner
# ---------------------------------------------------------------------------


class AblationRunner:
    """Runs a heteroscedastic vs homoscedastic GP ablation on a benchmark.

    Parameters
    ----------
    benchmark : ExperimentalBenchmark
        The benchmark to evaluate on.
    alpha : float
        Significance level for the statistical test (default 0.05).
    """

    def __init__(
        self,
        benchmark: ExperimentalBenchmark,
        alpha: float = 0.05,
    ) -> None:
        self.benchmark = benchmark
        self.alpha = alpha
        self._evaluator = CaseStudyEvaluator(benchmark)

    def run_hetero_vs_homo(
        self,
        budget: int = 30,
        n_repeats: int = 15,
        hetero_kwargs: dict[str, Any] | None = None,
        homo_kwargs: dict[str, Any] | None = None,
    ) -> AblationResult:
        """Compare HeteroscedasticGP vs standard GaussianProcessBO.

        Creates fresh strategy instances per repeat for clean comparisons.

        Parameters
        ----------
        budget : int
            Number of evaluation iterations per run.
        n_repeats : int
            Number of independent repetitions (paired by seed).
        hetero_kwargs : dict | None
            Keyword arguments for HeteroscedasticGP constructor.
        homo_kwargs : dict | None
            Keyword arguments for GaussianProcessBO constructor.

        Returns
        -------
        AblationResult
        """
        hetero_kwargs = hetero_kwargs or {}
        homo_kwargs = homo_kwargs or {}

        objectives = self.benchmark.get_objectives()
        obj_name = next(iter(objectives))
        direction = objectives[obj_name].get("direction", "minimize")

        all_metrics: dict[str, list[PerformanceMetrics]] = {
            "heteroscedastic_gp": [],
            "homoscedastic_gp": [],
        }
        all_curves: dict[str, list[list[float]]] = {
            "heteroscedastic_gp": [],
            "homoscedastic_gp": [],
        }

        last_hetero: HeteroscedasticGP | None = None

        for repeat in range(n_repeats):
            run_seed = 1000 * repeat + 42

            # Fresh strategy instances per repeat
            hetero = HeteroscedasticGP(**hetero_kwargs)
            homo = GaussianProcessBO(**homo_kwargs)

            for name, strategy in [
                ("heteroscedastic_gp", hetero),
                ("homoscedastic_gp", homo),
            ]:
                history, metrics = self._evaluator.run_single(
                    strategy, budget, seed=run_seed,
                )
                all_metrics[name].append(metrics)

                curve = self._extract_convergence(history, obj_name, direction)
                all_curves[name].append(curve)

            last_hetero = hetero

        comparison = ComparisonResult(
            strategy_names=["heteroscedastic_gp", "homoscedastic_gp"],
            metrics=all_metrics,
            convergence_curves=all_curves,
            budget=budget,
            n_repeats=n_repeats,
            benchmark_name=type(self.benchmark).__name__,
        )

        # Extract best_value per repeat for statistical testing
        hetero_vals = [
            m.best_value for m in all_metrics["heteroscedastic_gp"]
        ]
        homo_vals = [
            m.best_value for m in all_metrics["homoscedastic_gp"]
        ]

        # Statistical comparison
        test_result = wilcoxon_signed_rank_test(hetero_vals, homo_vals)
        cohens_d = compute_effect_size(hetero_vals, homo_vals)

        hetero_med = _median(hetero_vals)
        homo_med = _median(homo_vals)

        if direction == "maximize":
            hetero_wins = hetero_med > homo_med
        else:
            hetero_wins = hetero_med < homo_med

        significant = test_result["p_value"] < self.alpha

        # Noise impact diagnostics from the last run
        noise_impact: list[dict[str, Any]] | None = None
        if last_hetero is not None:
            try:
                noise_impact = last_hetero.compute_noise_impact()
            except Exception:
                pass

        # Build summary
        summary = self._build_summary(
            direction=direction,
            hetero_med=hetero_med,
            homo_med=homo_med,
            hetero_wins=hetero_wins,
            significant=significant,
            p_value=test_result["p_value"],
            cohens_d=cohens_d,
            n_repeats=n_repeats,
            budget=budget,
            obj_name=obj_name,
        )

        return AblationResult(
            comparison=comparison,
            statistical_test=test_result,
            cohens_d=cohens_d,
            hetero_wins=hetero_wins,
            significant=significant,
            hetero_median=hetero_med,
            homo_median=homo_med,
            noise_impact=noise_impact,
            summary=summary,
        )

    def run_noise_impact_analysis(
        self,
        budget: int = 30,
        seed: int = 42,
        **hetero_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run a single HeteroscedasticGP trial and return noise impact.

        This diagnostic shows how per-point noise variances change the
        effective weight of each observation vs a homoscedastic model.

        Parameters
        ----------
        budget : int
            Number of evaluation iterations.
        seed : int
            Random seed.
        **hetero_kwargs
            Passed to HeteroscedasticGP constructor.

        Returns
        -------
        list[dict]
            Per-point noise impact analysis from
            :meth:`HeteroscedasticGP.compute_noise_impact`.
        """
        strategy = HeteroscedasticGP(**hetero_kwargs)
        self._evaluator.run_single(strategy, budget, seed=seed)
        return strategy.compute_noise_impact()

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _extract_convergence(
        history: list[dict],
        obj_name: str,
        direction: str,
    ) -> list[float]:
        """Extract best-so-far convergence curve from history."""
        is_minimize = direction == "minimize"
        curve: list[float] = []
        for entry in history:
            result = entry["result"]
            if result is None:
                if curve:
                    curve.append(curve[-1])
                else:
                    curve.append(
                        float("inf") if is_minimize else float("-inf")
                    )
                continue
            val = result[obj_name]["value"]
            if not curve:
                curve.append(val)
            else:
                prev = curve[-1]
                if is_minimize:
                    curve.append(min(prev, val))
                else:
                    curve.append(max(prev, val))
        return curve

    @staticmethod
    def _build_summary(
        *,
        direction: str,
        hetero_med: float,
        homo_med: float,
        hetero_wins: bool,
        significant: bool,
        p_value: float,
        cohens_d: float,
        n_repeats: int,
        budget: int,
        obj_name: str,
    ) -> str:
        """Build a human-readable summary of the ablation result."""
        winner = "HeteroscedasticGP" if hetero_wins else "HomoscedasticGP"
        sig_str = "statistically significant" if significant else "not significant"
        diff = abs(hetero_med - homo_med)

        lines = [
            f"Ablation: HeteroscedasticGP vs HomoscedasticGP",
            f"  Objective: {obj_name} ({direction})",
            f"  Budget: {budget}, Repeats: {n_repeats}",
            f"  Hetero median: {hetero_med:.4f}",
            f"  Homo median:   {homo_med:.4f}",
            f"  Difference:    {diff:.4f} ({sig_str})",
            f"  Winner: {winner}",
            f"  p-value: {p_value:.4f}, Cohen's d: {cohens_d:.4f}",
        ]

        if cohens_d >= 0.8:
            lines.append("  Effect size: LARGE")
        elif cohens_d >= 0.5:
            lines.append("  Effect size: MEDIUM")
        elif cohens_d >= 0.2:
            lines.append("  Effect size: SMALL")
        else:
            lines.append("  Effect size: NEGLIGIBLE")

        return "\n".join(lines)

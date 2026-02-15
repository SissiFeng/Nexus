"""Conclusion robustness checking — tests whether scientific conclusions are stable."""

from __future__ import annotations

import math
import random

from optimization_copilot.robustness.models import ConclusionRobustness, RobustnessReport
from optimization_copilot.robustness.bootstrap import BootstrapAnalyzer


class ConclusionRobustnessChecker:
    """Test if scientific conclusions are stable under perturbation.

    Wraps :class:`BootstrapAnalyzer` to answer higher-level questions:
    is a ranking stable? Is feature importance ordering robust?
    Is Pareto front membership consistent?

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_bootstrap: int = 1000, seed: int = 42) -> None:
        self._bootstrap = BootstrapAnalyzer(n_bootstrap=n_bootstrap, seed=seed)
        self._n_bootstrap = n_bootstrap
        self._rng = random.Random(seed)

    def check_ranking_stability(
        self,
        values: list[float],
        names: list[str],
        k: int = 5,
    ) -> ConclusionRobustness:
        """Check whether the top-K ranking is stable under bootstrap resampling.

        The stability score is the fraction of bootstrap resamples where the
        top-1 item remains the same as in the original data.

        Parameters
        ----------
        values : list[float]
            Observed values (higher is better).
        names : list[str]
            Names corresponding to each value.
        k : int
            Size of the top-K set to evaluate.

        Returns
        -------
        ConclusionRobustness
        """
        result = self._bootstrap.bootstrap_top_k(values, names, k=k)

        return ConclusionRobustness(
            conclusion_type="ranking",
            stability_score=result["stability_score"],
            n_bootstrap=self._n_bootstrap,
            details={
                "top_k_frequency": result["top_k_frequency"],
                "original_top_k": result["original_top_k"],
                "k": k,
            },
        )

    def check_importance_stability(
        self,
        X: list[list[float]],
        y: list[float],
        var_names: list[str] | None = None,
    ) -> ConclusionRobustness:
        """Check if feature importance ordering is stable under resampling.

        Uses a simple variance-based importance measure: for each feature,
        importance is the correlation between feature values and the target.

        Parameters
        ----------
        X : list[list[float]]
            Feature matrix, shape ``(n_samples, n_features)``.
        y : list[float]
            Target values.
        var_names : list[str] or None
            Optional feature names.

        Returns
        -------
        ConclusionRobustness
        """
        n_features = len(X[0]) if X else 0
        if var_names is None:
            var_names = [f"x{i}" for i in range(n_features)]

        def importance_fn(X_: list[list[float]], y_: list[float]) -> dict[str, float]:
            """Variance-based feature importance: abs(correlation) with target."""
            result: dict[str, float] = {}
            n = len(y_)
            if n < 2:
                return {name: 0.0 for name in var_names}

            mean_y = sum(y_) / n
            var_y = sum((yi - mean_y) ** 2 for yi in y_)

            for j in range(n_features):
                col = [X_[i][j] for i in range(n)]
                mean_x = sum(col) / n
                var_x = sum((xi - mean_x) ** 2 for xi in col)
                if var_x < 1e-15 or var_y < 1e-15:
                    result[var_names[j]] = 0.0
                    continue
                cov = sum((col[i] - mean_x) * (y_[i] - mean_y) for i in range(n))
                r = abs(cov / math.sqrt(var_x * var_y))
                result[var_names[j]] = r
            return result

        boot_results = self._bootstrap.bootstrap_feature_importance(
            X, y, importance_fn
        )

        # Compute stability: fraction of bootstraps where the most important
        # feature stays the same
        observed_imp = importance_fn(X, y)
        original_ranking = sorted(observed_imp.keys(), key=lambda k: observed_imp[k], reverse=True)
        original_top1 = original_ranking[0] if original_ranking else ""

        # Check bootstrap distributions for ranking consistency
        n_boot = self._n_bootstrap
        top1_same_count = 0

        # Reconstruct per-bootstrap rankings from distributions
        feature_names = list(boot_results.keys())
        if feature_names:
            n_boot_actual = len(boot_results[feature_names[0]].bootstrap_distribution)
            for b in range(n_boot_actual):
                boot_imp = {
                    name: boot_results[name].bootstrap_distribution[b]
                    for name in feature_names
                }
                boot_top1 = max(boot_imp, key=lambda k: boot_imp[k])
                if boot_top1 == original_top1:
                    top1_same_count += 1
            stability = top1_same_count / max(n_boot_actual, 1)
        else:
            stability = 0.0

        return ConclusionRobustness(
            conclusion_type="importance",
            stability_score=stability,
            n_bootstrap=self._n_bootstrap,
            details={
                "original_ranking": original_ranking,
                "feature_cis": {
                    name: {
                        "observed": br.observed_value,
                        "ci_lower": br.ci_lower,
                        "ci_upper": br.ci_upper,
                    }
                    for name, br in boot_results.items()
                },
            },
        )

    def check_pareto_stability(
        self,
        objectives: list[list[float]],
        names: list[str],
        n_bootstrap: int | None = None,
    ) -> ConclusionRobustness:
        """Check if Pareto front membership is stable under bootstrap resampling.

        For each resample, Pareto optimal points are re-identified.
        Stability is the fraction of resamples where the original Pareto set
        membership is unchanged.

        Parameters
        ----------
        objectives : list[list[float]]
            Objective values, shape ``(n_points, n_objectives)``.
            Lower is better (minimization).
        names : list[str]
            Names for each point.
        n_bootstrap : int or None
            Override for bootstrap count.

        Returns
        -------
        ConclusionRobustness
        """
        n_boot = n_bootstrap if n_bootstrap is not None else self._n_bootstrap
        n = len(objectives)

        # Original Pareto set
        original_pareto = set()
        for i in range(n):
            if self._is_pareto_optimal(objectives[i], objectives, minimize=True):
                original_pareto.add(names[i])

        same_pareto_count = 0
        membership_counts: dict[str, int] = {name: 0 for name in names}

        for _ in range(n_boot):
            indices = [self._rng.randint(0, n - 1) for _ in range(n)]
            boot_objs = [objectives[i] for i in indices]
            boot_names = [names[i] for i in indices]

            # Deduplicate by name (keep first occurrence)
            seen: dict[str, list[float]] = {}
            for nm, obj in zip(boot_names, boot_objs):
                if nm not in seen:
                    seen[nm] = obj

            unique_names = list(seen.keys())
            unique_objs = [seen[nm] for nm in unique_names]

            boot_pareto = set()
            for i, nm in enumerate(unique_names):
                if self._is_pareto_optimal(unique_objs[i], unique_objs, minimize=True):
                    boot_pareto.add(nm)

            for nm in boot_pareto:
                membership_counts[nm] += 1

            if boot_pareto == original_pareto:
                same_pareto_count += 1

        stability = same_pareto_count / max(n_boot, 1)
        pareto_frequency = {name: count / n_boot for name, count in membership_counts.items()}

        return ConclusionRobustness(
            conclusion_type="pareto",
            stability_score=stability,
            n_bootstrap=n_boot,
            details={
                "original_pareto": sorted(original_pareto),
                "pareto_frequency": pareto_frequency,
            },
        )

    def comprehensive_robustness(
        self,
        values: list[float],
        names: list[str],
        X: list[list[float]] | None = None,
        y: list[float] | None = None,
    ) -> RobustnessReport:
        """Run all applicable stability checks and produce an aggregate report.

        Parameters
        ----------
        values : list[float]
            Observed values for ranking analysis.
        names : list[str]
            Names for each item.
        X : list[list[float]] or None
            Feature matrix for importance analysis.
        y : list[float] or None
            Target values for importance analysis.

        Returns
        -------
        RobustnessReport
        """
        analyses: list[ConclusionRobustness] = []
        warnings: list[str] = []

        # Ranking stability
        if len(values) >= 2:
            k = min(5, len(values))
            ranking_result = self.check_ranking_stability(values, names, k=k)
            analyses.append(ranking_result)
            if ranking_result.stability_score < 0.5:
                warnings.append(
                    f"Ranking is unstable (stability={ranking_result.stability_score:.2f}). "
                    "Top-1 item changes frequently under resampling."
                )
        else:
            warnings.append("Not enough values for ranking stability analysis.")

        # Importance stability
        if X is not None and y is not None and len(X) >= 3 and len(X[0]) >= 1:
            imp_result = self.check_importance_stability(X, y)
            analyses.append(imp_result)
            if imp_result.stability_score < 0.5:
                warnings.append(
                    f"Feature importance ordering is unstable "
                    f"(stability={imp_result.stability_score:.2f})."
                )

        # Overall robustness
        if analyses:
            overall = sum(a.stability_score for a in analyses) / len(analyses)
        else:
            overall = 0.0

        return RobustnessReport(
            analyses=analyses,
            overall_robustness=overall,
            warnings=warnings,
        )

    @staticmethod
    def _is_pareto_optimal(
        point: list[float],
        all_points: list[list[float]],
        minimize: bool = True,
    ) -> bool:
        """Check if *point* is Pareto optimal (non-dominated).

        A point is Pareto optimal if no other point is at least as good in
        all objectives and strictly better in at least one.

        Parameters
        ----------
        point : list[float]
            The objective values of the candidate point.
        all_points : list[list[float]]
            All objective values.
        minimize : bool
            If True, lower values are better.

        Returns
        -------
        bool
        """
        for other in all_points:
            if other is point:
                continue
            # Check if *other* dominates *point*
            all_leq = True
            any_lt = False
            for a, b in zip(other, point):
                if minimize:
                    if a > b:
                        all_leq = False
                        break
                    if a < b:
                        any_lt = True
                else:
                    if a < b:
                        all_leq = False
                        break
                    if a > b:
                        any_lt = True
            if all_leq and any_lt:
                return False
        return True

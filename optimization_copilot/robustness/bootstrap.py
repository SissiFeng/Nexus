"""Non-parametric bootstrap for confidence intervals and stability analysis."""

from __future__ import annotations

import math
import random
from typing import Callable

from optimization_copilot.robustness.models import BootstrapResult


class BootstrapAnalyzer:
    """Non-parametric bootstrap for confidence intervals.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap resamples to draw.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_bootstrap: int = 1000, seed: int = 42) -> None:
        self._n_bootstrap = n_bootstrap
        self._rng = random.Random(seed)

    def bootstrap_ci(
        self,
        data: list,
        statistic_fn: Callable,
        confidence: float = 0.95,
    ) -> BootstrapResult:
        """Compute a bootstrap confidence interval for a statistic.

        Parameters
        ----------
        data : list
            The original data sample.
        statistic_fn : Callable
            A function that takes a list and returns a float statistic.
        confidence : float
            Confidence level for the interval (e.g. 0.95).

        Returns
        -------
        BootstrapResult
            The observed statistic, CI bounds, and bootstrap distribution.
        """
        observed = statistic_fn(data)
        n = len(data)
        boot_stats: list[float] = []

        for _ in range(self._n_bootstrap):
            resample = [data[self._rng.randint(0, n - 1)] for _ in range(n)]
            boot_stats.append(statistic_fn(resample))

        boot_stats.sort()
        alpha = 1.0 - confidence
        ci_lower = self._percentile(boot_stats, alpha / 2.0)
        ci_upper = self._percentile(boot_stats, 1.0 - alpha / 2.0)

        # Standard error
        mean_boot = sum(boot_stats) / len(boot_stats)
        variance = sum((x - mean_boot) ** 2 for x in boot_stats) / max(len(boot_stats) - 1, 1)
        std_error = math.sqrt(variance)

        return BootstrapResult(
            statistic_name="custom",
            observed_value=observed,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence,
            n_bootstrap=self._n_bootstrap,
            bootstrap_distribution=boot_stats,
            std_error=std_error,
        )

    def bootstrap_top_k(
        self,
        values: list[float],
        names: list[str],
        k: int = 5,
        n_bootstrap: int | None = None,
    ) -> dict:
        """Assess how stable the top-K ranking is under resampling.

        For each bootstrap resample, the top-K items are re-identified.
        The result reports how frequently each item appears in the top-K.

        Parameters
        ----------
        values : list[float]
            Observed values (higher is better).
        names : list[str]
            Names corresponding to each value.
        k : int
            Size of the top-K set.
        n_bootstrap : int or None
            Override for the number of bootstrap resamples.

        Returns
        -------
        dict
            Keys: ``"top_k_frequency"`` (dict mapping names to frequency),
            ``"stability_score"`` (fraction where top-1 is unchanged),
            ``"original_top_k"`` (names of the original top-K).
        """
        n_boot = n_bootstrap if n_bootstrap is not None else self._n_bootstrap
        n = len(values)
        k = min(k, n)

        # Original top-K (by descending value)
        indexed = sorted(range(n), key=lambda i: values[i], reverse=True)
        original_top_k = [names[i] for i in indexed[:k]]
        original_top_1 = original_top_k[0]

        frequency: dict[str, int] = {name: 0 for name in names}
        top1_same_count = 0

        # Estimate noise scale from the data spread
        mean_val = sum(values) / n
        spread = math.sqrt(sum((v - mean_val) ** 2 for v in values) / max(n - 1, 1))
        # Use a fraction of the spread as noise; if values are very close
        # this makes rankings unstable (as expected)
        noise_scale = spread / math.sqrt(n) if spread > 1e-15 else 1e-6

        for _ in range(n_boot):
            # Bayesian bootstrap: perturb values with noise proportional to
            # estimation uncertainty
            perturbed = [
                values[i] + self._rng.gauss(0, noise_scale)
                for i in range(n)
            ]

            boot_sorted = sorted(range(n), key=lambda i: perturbed[i], reverse=True)
            boot_top_k = [names[i] for i in boot_sorted[:k]]

            for name in boot_top_k:
                frequency[name] += 1

            if boot_top_k[0] == original_top_1:
                top1_same_count += 1

        top_k_frequency = {name: count / n_boot for name, count in frequency.items()}
        stability_score = top1_same_count / n_boot

        return {
            "top_k_frequency": top_k_frequency,
            "stability_score": stability_score,
            "original_top_k": original_top_k,
        }

    def bootstrap_correlation(
        self,
        xs: list[float],
        ys: list[float],
        n_bootstrap: int | None = None,
    ) -> BootstrapResult:
        """Compute bootstrap CI for Pearson correlation coefficient.

        Parameters
        ----------
        xs : list[float]
            First variable.
        ys : list[float]
            Second variable.
        n_bootstrap : int or None
            Override for the number of bootstrap resamples.

        Returns
        -------
        BootstrapResult
            Bootstrap CI for the Pearson r value.
        """
        n_boot = n_bootstrap if n_bootstrap is not None else self._n_bootstrap
        observed_r = self._pearson_r(xs, ys)
        n = len(xs)
        boot_stats: list[float] = []

        for _ in range(n_boot):
            indices = [self._rng.randint(0, n - 1) for _ in range(n)]
            boot_xs = [xs[i] for i in indices]
            boot_ys = [ys[i] for i in indices]
            boot_stats.append(self._pearson_r(boot_xs, boot_ys))

        boot_stats.sort()
        ci_lower = self._percentile(boot_stats, 0.025)
        ci_upper = self._percentile(boot_stats, 0.975)

        mean_boot = sum(boot_stats) / len(boot_stats)
        variance = sum((x - mean_boot) ** 2 for x in boot_stats) / max(len(boot_stats) - 1, 1)
        std_error = math.sqrt(variance)

        return BootstrapResult(
            statistic_name="pearson_r",
            observed_value=observed_r,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=0.95,
            n_bootstrap=n_boot,
            bootstrap_distribution=boot_stats,
            std_error=std_error,
        )

    def bootstrap_feature_importance(
        self,
        X: list[list[float]],
        y: list[float],
        importance_fn: Callable,
        n_bootstrap: int | None = None,
    ) -> dict:
        """Compute bootstrap CIs for feature importance scores.

        Parameters
        ----------
        X : list[list[float]]
            Feature matrix.
        y : list[float]
            Target values.
        importance_fn : Callable
            Function ``(X, y) -> dict[str, float]`` returning importance
            scores for each feature.
        n_bootstrap : int or None
            Override for the number of bootstrap resamples.

        Returns
        -------
        dict
            Mapping from feature name to ``BootstrapResult``.
        """
        n_boot = n_bootstrap if n_bootstrap is not None else self._n_bootstrap
        n = len(X)

        # Observed importances
        observed = importance_fn(X, y)
        feature_names = list(observed.keys())
        boot_distributions: dict[str, list[float]] = {name: [] for name in feature_names}

        for _ in range(n_boot):
            indices = [self._rng.randint(0, n - 1) for _ in range(n)]
            boot_X = [X[i] for i in indices]
            boot_y = [y[i] for i in indices]
            boot_imp = importance_fn(boot_X, boot_y)
            for name in feature_names:
                boot_distributions[name].append(boot_imp.get(name, 0.0))

        results: dict[str, BootstrapResult] = {}
        for name in feature_names:
            dist = sorted(boot_distributions[name])
            ci_lower = self._percentile(dist, 0.025)
            ci_upper = self._percentile(dist, 0.975)
            mean_boot = sum(dist) / len(dist)
            variance = sum((x - mean_boot) ** 2 for x in dist) / max(len(dist) - 1, 1)
            std_error = math.sqrt(variance)

            results[name] = BootstrapResult(
                statistic_name=f"importance_{name}",
                observed_value=observed[name],
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                confidence_level=0.95,
                n_bootstrap=n_boot,
                bootstrap_distribution=dist,
                std_error=std_error,
            )

        return results

    @staticmethod
    def _percentile(sorted_values: list[float], p: float) -> float:
        """Compute the p-th percentile (0-1) of sorted values.

        Uses linear interpolation between adjacent data points.
        """
        if not sorted_values:
            return 0.0
        n = len(sorted_values)
        if n == 1:
            return sorted_values[0]

        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        lo = max(0, min(lo, n - 1))
        hi = max(0, min(hi, n - 1))

        if lo == hi:
            return sorted_values[lo]

        frac = idx - lo
        return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac

    @staticmethod
    def _pearson_r(xs: list[float], ys: list[float]) -> float:
        """Compute Pearson correlation coefficient (pure Python).

        Returns 0.0 if standard deviation of either variable is zero.
        """
        n = len(xs)
        if n < 2:
            return 0.0

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        var_x = sum((xs[i] - mean_x) ** 2 for i in range(n))
        var_y = sum((ys[i] - mean_y) ** 2 for i in range(n))

        denom = math.sqrt(var_x * var_y)
        if denom < 1e-15:
            return 0.0

        return cov / denom

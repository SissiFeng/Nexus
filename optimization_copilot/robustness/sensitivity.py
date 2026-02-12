"""Decision sensitivity analysis under noise perturbation."""

from __future__ import annotations

import math
import random


class DecisionSensitivityAnalyzer:
    """Analyse how much the optimal decision changes under uncertainty.

    Adds Gaussian noise to observed values and measures how frequently
    the recommended action (top-1 or top-K) changes.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def decision_sensitivity(
        self,
        values: list[float],
        names: list[str],
        noise_std: float = 0.1,
        n_perturbations: int = 100,
    ) -> dict:
        """Measure sensitivity of the top-1 decision to Gaussian noise.

        Parameters
        ----------
        values : list[float]
            Observed values (higher is better).
        names : list[str]
            Names corresponding to each value.
        noise_std : float
            Standard deviation of Gaussian noise added to each value.
        n_perturbations : int
            Number of noise perturbation trials.

        Returns
        -------
        dict
            ``"stability"`` (float): fraction of perturbations where top-1
            is unchanged.
            ``"top1_frequency"`` (dict): how often each item is top-1.
            ``"mean_rank_change"`` (float): average absolute change in rank
            of the original top-1 item.
        """
        n = len(values)
        if n == 0:
            return {"stability": 0.0, "top1_frequency": {}, "mean_rank_change": 0.0}

        # Original ranking (descending)
        original_sorted = sorted(range(n), key=lambda i: values[i], reverse=True)
        original_top1 = names[original_sorted[0]]
        original_top1_idx = original_sorted[0]

        top1_counts: dict[str, int] = {name: 0 for name in names}
        top1_same = 0
        rank_changes: list[float] = []

        for _ in range(n_perturbations):
            perturbed = [v + self._rng.gauss(0, noise_std) for v in values]
            pert_sorted = sorted(range(n), key=lambda i: perturbed[i], reverse=True)

            pert_top1 = names[pert_sorted[0]]
            top1_counts[pert_top1] += 1

            if pert_top1 == original_top1:
                top1_same += 1

            # Rank of original top-1 in perturbed ranking
            new_rank = pert_sorted.index(original_top1_idx)
            rank_changes.append(abs(new_rank - 0))

        stability = top1_same / n_perturbations
        top1_frequency = {name: count / n_perturbations for name, count in top1_counts.items()}
        mean_rank_change = sum(rank_changes) / len(rank_changes) if rank_changes else 0.0

        return {
            "stability": stability,
            "top1_frequency": top1_frequency,
            "mean_rank_change": mean_rank_change,
        }

    def recommendation_confidence(
        self,
        values: list[float],
        names: list[str],
        k: int = 5,
        noise_std: float = 0.1,
        n_perturbations: int = 500,
    ) -> dict:
        """For each item: probability it remains in top-K under noise.

        Parameters
        ----------
        values : list[float]
            Observed values (higher is better).
        names : list[str]
            Names corresponding to each value.
        k : int
            Size of the top-K set.
        noise_std : float
            Standard deviation of Gaussian noise.
        n_perturbations : int
            Number of perturbation trials.

        Returns
        -------
        dict
            Mapping from each name to its probability of appearing in top-K.
        """
        n = len(values)
        k = min(k, n)
        in_topk_counts: dict[str, int] = {name: 0 for name in names}

        for _ in range(n_perturbations):
            perturbed = [v + self._rng.gauss(0, noise_std) for v in values]
            pert_sorted = sorted(range(n), key=lambda i: perturbed[i], reverse=True)
            for idx in pert_sorted[:k]:
                in_topk_counts[names[idx]] += 1

        return {
            name: count / n_perturbations
            for name, count in in_topk_counts.items()
        }

    def value_at_risk(
        self,
        values: list[float],
        uncertainties: list[float],
        quantile: float = 0.05,
        n_samples: int = 1000,
    ) -> dict:
        """Compute worst-case expected value at a given confidence level.

        For each item, samples from ``N(value, uncertainty^2)`` and returns
        the ``quantile``-th percentile of the best item's value across samples.

        Parameters
        ----------
        values : list[float]
            Expected values.
        uncertainties : list[float]
            Standard deviations of uncertainty for each value.
        quantile : float
            Lower quantile for worst-case analysis (e.g. 0.05 for 5th percentile).
        n_samples : int
            Number of Monte Carlo samples.

        Returns
        -------
        dict
            ``"var"`` (float): value-at-risk (the quantile of max-value).
            ``"best_values"`` (list[float]): sorted distribution of best values.
            ``"expected_best"`` (float): mean of the best value distribution.
        """
        n = len(values)
        if n == 0:
            return {"var": 0.0, "best_values": [], "expected_best": 0.0}

        best_values: list[float] = []

        for _ in range(n_samples):
            sampled = [
                values[i] + self._rng.gauss(0, uncertainties[i])
                for i in range(n)
            ]
            best_values.append(max(sampled))

        best_values.sort()

        # Compute the quantile
        idx = quantile * (len(best_values) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        lo = max(0, min(lo, len(best_values) - 1))
        hi = max(0, min(hi, len(best_values) - 1))

        if lo == hi:
            var_value = best_values[lo]
        else:
            frac = idx - lo
            var_value = best_values[lo] * (1.0 - frac) + best_values[hi] * frac

        expected_best = sum(best_values) / len(best_values)

        return {
            "var": var_value,
            "best_values": best_values,
            "expected_best": expected_best,
        }

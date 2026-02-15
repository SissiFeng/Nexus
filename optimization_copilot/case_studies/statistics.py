"""Pure-Python statistical testing for case study comparisons.

Provides non-parametric hypothesis tests and effect size measures
for comparing optimization strategy performance across repeated runs.
No external dependencies -- uses only the stdlib and the project's
own ``norm_cdf`` helper.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Tuple

from optimization_copilot.backends._math.stats import norm_cdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median(values: List[float]) -> float:
    """Return the median of a list of floats."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _mean(values: List[float]) -> float:
    """Return the arithmetic mean of a list of floats."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: List[float], ddof: int = 1) -> float:
    """Return the sample standard deviation."""
    n = len(values)
    if n <= ddof:
        return 0.0
    m = _mean(values)
    ss = sum((v - m) ** 2 for v in values)
    return math.sqrt(ss / (n - ddof))


def _rank_with_ties(values: List[float]) -> List[float]:
    """Assign ranks to *values*, averaging ties.

    Returns a list of ranks (1-based) in the same order as *values*.
    """
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        # Find the run of ties.
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # average of 1-based ranks i+1..j
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j

    return ranks


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank_test(
    x: List[float],
    y: List[float],
) -> Dict[str, float]:
    """Wilcoxon signed-rank test for paired samples (pure Python).

    Non-parametric test comparing paired observations.
    Uses normal approximation for the *p*-value (valid for *n* >= 10).

    Parameters
    ----------
    x : list[float]
        First sample (e.g., strategy A results across repeats).
    y : list[float]
        Second sample (same length, paired by seed).

    Returns
    -------
    dict
        ``{"statistic": W, "p_value": p, "effect_size": r, "n_effective": n_eff}``

    Algorithm
    ---------
    1. Compute differences ``d_i = x_i - y_i``.
    2. Remove zero differences.
    3. Rank ``|d_i|`` (handle ties by averaging ranks).
    4. ``W+ = sum`` of ranks where ``d_i > 0``.
    5. ``W- = sum`` of ranks where ``d_i < 0``.
    6. ``W = min(W+, W-)``.
    7. Normal approximation: ``z = (W - mu_W) / sigma_W``
       where ``mu_W = n(n+1)/4``, ``sigma_W = sqrt(n(n+1)(2n+1)/24)``.
    8. ``p_value = 2 * (1 - Phi(|z|))`` using :func:`norm_cdf`.
    9. ``effect_size r = |z| / sqrt(n)``.
    """
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length, got {len(x)} and {len(y)}"
        )

    # Step 1: differences
    diffs = [xi - yi for xi, yi in zip(x, y)]

    # Step 2: remove zeros
    nonzero = [(d, abs(d)) for d in diffs if d != 0.0]
    n_eff = len(nonzero)

    # Edge case: all differences are zero
    if n_eff == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size": 0.0,
            "n_effective": 0,
        }

    if n_eff < 10:
        warnings.warn(
            f"Sample size after removing zeros is {n_eff} (< 10); "
            "normal approximation may be inaccurate.",
            stacklevel=2,
        )

    # Step 3: rank absolute differences (with tie-averaging)
    abs_diffs = [ad for _, ad in nonzero]
    ranks = _rank_with_ties(abs_diffs)

    # Steps 4-5: signed rank sums
    w_plus = 0.0
    w_minus = 0.0
    for (d, _), rank in zip(nonzero, ranks):
        if d > 0:
            w_plus += rank
        else:
            w_minus += rank

    # Step 6
    w = min(w_plus, w_minus)

    # Step 7: normal approximation
    mu_w = n_eff * (n_eff + 1) / 4.0
    sigma_w = math.sqrt(n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24.0)

    if sigma_w == 0.0:
        # Degenerate case (n_eff == 1 gives sigma > 0, but guard anyway)
        return {
            "statistic": w,
            "p_value": 1.0,
            "effect_size": 0.0,
            "n_effective": n_eff,
        }

    z = (w - mu_w) / sigma_w

    # Step 8: two-sided p-value
    p_value = 2.0 * (1.0 - norm_cdf(abs(z)))
    p_value = min(p_value, 1.0)  # clamp

    # Step 9: effect size
    effect_size = abs(z) / math.sqrt(n_eff)

    return {
        "statistic": w,
        "p_value": p_value,
        "effect_size": effect_size,
        "n_effective": n_eff,
    }


# ---------------------------------------------------------------------------
# Cohen's d for paired samples
# ---------------------------------------------------------------------------

def compute_effect_size(x: List[float], y: List[float]) -> float:
    """Return Cohen's d effect size for paired samples.

    ``d = mean(x - y) / std(x - y)``

    Returns 0.0 when the standard deviation is zero.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    diffs = [xi - yi for xi, yi in zip(x, y)]
    sd = _std(diffs, ddof=1)
    if sd == 0.0:
        return 0.0
    return abs(_mean(diffs)) / sd


# ---------------------------------------------------------------------------
# Pairwise comparison table
# ---------------------------------------------------------------------------

def paired_comparison_table(
    results: Dict[str, List[float]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute pairwise Wilcoxon tests between all strategy pairs.

    Parameters
    ----------
    results : dict[str, list[float]]
        ``{strategy_name: [repeat_1_value, repeat_2_value, ...]}``.

    Returns
    -------
    dict
        ``{strategy_a: {strategy_b: {"p_value": p, "effect_size": r, "winner": ...}}}``
        where ``winner`` is the name of the strategy with the lower median
        (better when minimising), ``"tie"`` if medians are equal.
    """
    names = sorted(results.keys())
    table: Dict[str, Dict[str, Dict[str, float]]] = {}

    for a in names:
        table[a] = {}
        for b in names:
            if a == b:
                table[a][b] = {
                    "p_value": 1.0,
                    "effect_size": 0.0,
                    "winner": "tie",
                }
                continue

            res = wilcoxon_signed_rank_test(results[a], results[b])
            med_a = _median(results[a])
            med_b = _median(results[b])
            if med_a < med_b:
                winner = a
            elif med_b < med_a:
                winner = b
            else:
                winner = "tie"

            table[a][b] = {
                "p_value": res["p_value"],
                "effect_size": res["effect_size"],
                "winner": winner,
            }

    return table


# ---------------------------------------------------------------------------
# Strategy ranking
# ---------------------------------------------------------------------------

def rank_strategies(
    results: Dict[str, List[float]],
    direction: str = "minimize",
) -> List[Tuple[str, float]]:
    """Rank strategies by median performance.

    Parameters
    ----------
    results : dict[str, list[float]]
        ``{strategy_name: [values...]}``.
    direction : str
        ``"minimize"`` (lower is better) or ``"maximize"`` (higher is better).

    Returns
    -------
    list[tuple[str, float]]
        Sorted list of ``(name, median)`` from best to worst.
    """
    if direction not in ("minimize", "maximize"):
        raise ValueError(f"direction must be 'minimize' or 'maximize', got {direction!r}")

    medians = [(name, _median(vals)) for name, vals in results.items()]
    reverse = direction == "maximize"
    medians.sort(key=lambda t: t[1], reverse=reverse)
    return medians

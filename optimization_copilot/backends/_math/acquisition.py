"""Pure-Python acquisition functions for Bayesian optimization.

These functions score candidate points to balance exploration and
exploitation during optimization.
"""

from __future__ import annotations

import math

from optimization_copilot.backends._math.stats import norm_cdf, norm_pdf


def expected_improvement(mu: float, sigma: float, best_y: float) -> float:
    """Standard Expected Improvement (EI) acquisition function.

    Computes EI assuming *minimization* of the objective.

    Parameters
    ----------
    mu : float
        Predicted mean at the candidate point.
    sigma : float
        Predicted standard deviation at the candidate point.
    best_y : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    float
        The expected improvement.  Returns 0 when *sigma* is negligible.
    """
    if sigma < 1e-12:
        return 0.0
    z = (best_y - mu) / sigma
    return (best_y - mu) * norm_cdf(z) + sigma * norm_pdf(z)


def upper_confidence_bound(mu: float, sigma: float, kappa: float = 2.0) -> float:
    """Upper Confidence Bound (UCB) acquisition function.

    For *minimization*, lower values of the returned score are better
    candidates: ``UCB = mu - kappa * sigma``.

    Parameters
    ----------
    mu : float
        Predicted mean at the candidate point.
    sigma : float
        Predicted standard deviation at the candidate point.
    kappa : float
        Exploration-exploitation trade-off parameter (default 2.0).
        Higher values encourage more exploration.

    Returns
    -------
    float
        The UCB score (lower is better for minimization).
    """
    return mu - kappa * sigma


def probability_of_improvement(mu: float, sigma: float, best_y: float) -> float:
    """Probability of Improvement (PI) acquisition function.

    Parameters
    ----------
    mu : float
        Predicted mean at the candidate point.
    sigma : float
        Predicted standard deviation at the candidate point.
    best_y : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    float
        The probability that this candidate improves on *best_y*.
        Returns 0 when *sigma* is negligible.
    """
    if sigma < 1e-12:
        return 0.0
    z = (best_y - mu) / sigma
    return norm_cdf(z)


def log_expected_improvement_per_cost(
    ei_values: list[float],
    costs: list[float],
) -> list[float]:
    """Log Expected Improvement per Cost (LogEIPC).

    Useful for cost-aware Bayesian optimization where evaluation costs
    vary across the search space.

    Parameters
    ----------
    ei_values : list[float]
        Expected improvement values for each candidate.
    costs : list[float]
        Associated costs for each candidate (must be positive).

    Returns
    -------
    list[float]
        Log(EI / cost) for each candidate.  Returns ``-inf`` when
        EI <= 0 or cost <= 0.
    """
    result: list[float] = []
    for ei, cost in zip(ei_values, costs):
        if ei <= 0.0 or cost <= 0.0:
            result.append(float("-inf"))
        else:
            result.append(math.log(ei) - math.log(cost))
    return result

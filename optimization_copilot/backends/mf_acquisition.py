"""Cost-aware acquisition functions for multi-fidelity Bayesian optimization.

All functions use pure-Python math -- zero external dependencies.
"""

from __future__ import annotations

import math

from optimization_copilot.backends._math import (
    expected_improvement,
    norm_cdf,
    norm_pdf,
    norm_ppf,
)


def cost_aware_ei(
    mean: float,
    variance: float,
    best_y: float,
    cost: float,
    xi: float = 0.01,
) -> float:
    """Cost-aware Expected Improvement.

    Computes ``EI(x) / cost(x)`` so that cheaper evaluations are
    favoured when their expected improvement is comparable.

    Parameters
    ----------
    mean : float
        Predicted posterior mean at the candidate.
    variance : float
        Predicted posterior variance at the candidate.
    best_y : float
        Best (lowest) observed objective value so far.
    cost : float
        Cost of evaluating at this candidate/fidelity.
    xi : float
        Exploration jitter added to the improvement threshold.

    Returns
    -------
    float
        EI / cost.  Returns 0 when cost <= 0 or sigma is negligible.
    """
    if cost <= 0.0:
        return 0.0
    sigma = math.sqrt(max(variance, 0.0))
    if sigma < 1e-12:
        return 0.0
    z = (best_y - mean - xi) / sigma
    ei = (best_y - mean - xi) * norm_cdf(z) + sigma * norm_pdf(z)
    return max(ei, 0.0) / cost


def fidelity_weighted_ei(
    mean: float,
    variance: float,
    best_y: float,
    fidelity: float,
    cost: float,
    xi: float = 0.01,
) -> float:
    """Fidelity-weighted Expected Improvement.

    Scales EI by the fidelity level (higher fidelity contributes more
    reliable information) and divides by cost.

    Parameters
    ----------
    mean : float
        Predicted posterior mean.
    variance : float
        Predicted posterior variance.
    best_y : float
        Best observed objective value.
    fidelity : float
        Fidelity level in [0, 1].
    cost : float
        Evaluation cost.
    xi : float
        Exploration jitter.

    Returns
    -------
    float
        ``fidelity * EI / cost``.
    """
    if cost <= 0.0 or fidelity <= 0.0:
        return 0.0
    sigma = math.sqrt(max(variance, 0.0))
    if sigma < 1e-12:
        return 0.0
    z = (best_y - mean - xi) / sigma
    ei = (best_y - mean - xi) * norm_cdf(z) + sigma * norm_pdf(z)
    return max(ei, 0.0) * fidelity / cost


def multi_fidelity_knowledge_gradient(
    means_by_fidelity: list[float],
    variances_by_fidelity: list[float],
    costs: list[float],
    current_best: float,
) -> tuple[float, int]:
    """Multi-fidelity Knowledge Gradient.

    For each fidelity level, estimates the expected value of information
    (improvement in the best predicted value) per unit cost and selects
    the fidelity level with the highest ratio.

    Parameters
    ----------
    means_by_fidelity : list[float]
        Predicted means at the candidate for each fidelity level.
    variances_by_fidelity : list[float]
        Predicted variances at the candidate for each fidelity level.
    costs : list[float]
        Cost of evaluation at each fidelity level.
    current_best : float
        Current best (lowest) observed value.

    Returns
    -------
    tuple[float, int]
        ``(best_kg_value, best_fidelity_index)`` -- the knowledge
        gradient value and the fidelity index that maximises value/cost.
    """
    n_fidelities = len(means_by_fidelity)
    if n_fidelities == 0:
        return (0.0, 0)

    best_kg = -math.inf
    best_idx = 0

    for i in range(n_fidelities):
        sigma = math.sqrt(max(variances_by_fidelity[i], 0.0))
        cost = costs[i] if i < len(costs) else 1.0
        if cost <= 0.0 or sigma < 1e-12:
            continue

        # Knowledge gradient approximation:
        # KG = sigma * phi(z) + (current_best - mu) * Phi(z)
        # where z = (current_best - mu) / sigma
        z = (current_best - means_by_fidelity[i]) / sigma
        kg = sigma * norm_pdf(z) + (current_best - means_by_fidelity[i]) * norm_cdf(z)
        kg_per_cost = max(kg, 0.0) / cost

        if kg_per_cost > best_kg:
            best_kg = kg_per_cost
            best_idx = i

    if best_kg == -math.inf:
        best_kg = 0.0

    return (best_kg, best_idx)


def entropy_search_multi_fidelity(
    means: list[float],
    variances: list[float],
    costs: list[float],
) -> tuple[float, int]:
    """Information-theoretic acquisition for multi-fidelity optimization.

    Approximates the expected information gain about the location of the
    global minimum at each fidelity level, divided by cost.  Uses a
    simplified predictive entropy reduction approach.

    Parameters
    ----------
    means : list[float]
        Predicted means at the candidate for each fidelity level.
    variances : list[float]
        Predicted variances at the candidate for each fidelity level.
    costs : list[float]
        Cost at each fidelity level.

    Returns
    -------
    tuple[float, int]
        ``(best_value, best_fidelity_index)`` -- the entropy search
        value and the fidelity index that maximises information/cost.
    """
    n_fidelities = len(means)
    if n_fidelities == 0:
        return (0.0, 0)

    best_val = -math.inf
    best_idx = 0

    for i in range(n_fidelities):
        sigma = math.sqrt(max(variances[i], 0.0))
        cost = costs[i] if i < len(costs) else 1.0
        if cost <= 0.0 or sigma < 1e-12:
            continue

        # Approximate information gain as 0.5 * log(1 + variance / noise)
        # Higher variance = more to learn; divide by cost for efficiency.
        noise_floor = 1e-4
        info_gain = 0.5 * math.log(1.0 + variances[i] / noise_floor)
        val = info_gain / cost

        if val > best_val:
            best_val = val
            best_idx = i

    if best_val == -math.inf:
        best_val = 0.0

    return (best_val, best_idx)


def _compute_ei_components(
    mean: float,
    variance: float,
    best_y: float,
    xi: float,
) -> tuple[float, float, float]:
    """Helper: compute z, EI, sigma for a single candidate.

    Returns (ei, z, sigma).
    """
    sigma = math.sqrt(max(variance, 0.0))
    if sigma < 1e-12:
        return (0.0, 0.0, 0.0)
    z = (best_y - mean - xi) / sigma
    ei = (best_y - mean - xi) * norm_cdf(z) + sigma * norm_pdf(z)
    return (max(ei, 0.0), z, sigma)

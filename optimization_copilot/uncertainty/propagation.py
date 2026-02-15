"""Uncertainty propagation engine for KPI-to-objective variance mapping.

Converts a set of ``MeasurementWithUncertainty`` values (one per KPI) into a
single ``PropagationResult`` that carries the propagated objective variance
together with a per-source uncertainty budget.

Three propagation methods are provided:

* **Linear** -- weighted sum objective: obj = sum(w_i * kpi_i).
* **Nonlinear (delta method)** -- first-order Taylor via Jacobian.
* **Monte Carlo** -- sampling-based (stdlib ``random`` only).

All methods are deterministic when given the same inputs (MC uses a fixed seed
by default).
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

from optimization_copilot.uncertainty.types import (
    MeasurementWithUncertainty,
    PropagationMethod,
    PropagationResult,
    UncertaintyBudget,
)


class UncertaintyPropagator:
    """Propagate KPI measurement uncertainties to a scalar objective.

    Usage
    -----
    >>> propagator = UncertaintyPropagator()
    >>> result = propagator.linear_propagation(measurements, weights)
    >>> obs = result.to_observation_with_noise()
    """

    # ── Linear propagation ────────────────────────────────────────────

    def linear_propagation(
        self,
        kpi_measurements: list[MeasurementWithUncertainty],
        weights: list[float],
    ) -> PropagationResult:
        """Propagate uncertainty for a weighted-sum objective.

        Objective: obj = sum(w_i * kpi_i)
        Variance:  var = sum(w_i**2 * var_i)   (independent KPIs)

        Parameters
        ----------
        kpi_measurements:
            One ``MeasurementWithUncertainty`` per KPI.
        weights:
            Corresponding scalar weight for each KPI.

        Returns
        -------
        PropagationResult
            Contains propagated value, variance, budget, and per-KPI details.
        """
        if len(kpi_measurements) != len(weights):
            raise ValueError(
                f"Length mismatch: {len(kpi_measurements)} measurements "
                f"vs {len(weights)} weights"
            )

        # Handle empty input gracefully.
        if not kpi_measurements:
            budget = UncertaintyBudget.from_contributions({})
            metadata = self._aggregate_metadata([], {})
            return PropagationResult(
                objective_value=0.0,
                objective_variance=0.0,
                method=PropagationMethod.LINEAR,
                budget=budget,
                kpi_details=[],
            )

        obj_value = sum(
            w * m.value for w, m in zip(weights, kpi_measurements)
        )

        # Per-KPI variance contributions: w_i**2 * sigma_i**2
        contributions: dict[str, float] = {}
        kpi_details: list[dict[str, Any]] = []

        for w, m in zip(weights, kpi_measurements):
            var_contribution = w ** 2 * m.variance
            contributions[m.source] = (
                contributions.get(m.source, 0.0) + var_contribution
            )
            kpi_details.append(
                {
                    "source": m.source,
                    "value": m.value,
                    "weight": w,
                    "variance": m.variance,
                    "var_contribution": var_contribution,
                }
            )

        obj_var = sum(contributions.values())

        # Compute fractional contributions for detail records.
        for detail in kpi_details:
            detail["var_fraction"] = (
                detail["var_contribution"] / obj_var if obj_var > 0 else 0.0
            )

        budget = UncertaintyBudget.from_contributions(contributions)
        metadata = self._aggregate_metadata(kpi_measurements, contributions)

        result = PropagationResult(
            objective_value=obj_value,
            objective_variance=obj_var,
            method=PropagationMethod.LINEAR,
            budget=budget,
            kpi_details=kpi_details,
        )
        return result

    # ── Nonlinear (delta method) propagation ──────────────────────────

    def nonlinear_propagation(
        self,
        kpi_measurements: list[MeasurementWithUncertainty],
        objective_func: Callable[..., float],
        jacobian_func: Callable[..., list[float]] | None = None,
    ) -> PropagationResult:
        """Propagate uncertainty via the delta method (first-order Taylor).

        Variance: var_obj ~ J^T Sigma J  where Sigma is diagonal.

        Parameters
        ----------
        kpi_measurements:
            One ``MeasurementWithUncertainty`` per KPI.
        objective_func:
            Callable that accepts ``len(kpi_measurements)`` floats and
            returns a scalar objective value.
        jacobian_func:
            Optional callable returning a list of partial derivatives.  When
            ``None``, a numerical Jacobian is computed via central differences.

        Returns
        -------
        PropagationResult
        """
        if not kpi_measurements:
            budget = UncertaintyBudget.from_contributions({})
            return PropagationResult(
                objective_value=0.0,
                objective_variance=0.0,
                method=PropagationMethod.DELTA,
                budget=budget,
                kpi_details=[],
            )

        values = [m.value for m in kpi_measurements]
        obj_value = objective_func(*values)

        # Compute Jacobian.
        if jacobian_func is not None:
            jacobian = jacobian_func(*values)
        else:
            jacobian = self._numerical_jacobian(objective_func, values)

        # Propagate: var_obj = sum( J_i**2 * sigma_i**2 )
        contributions: dict[str, float] = {}
        kpi_details: list[dict[str, Any]] = []

        for i, m in enumerate(kpi_measurements):
            var_contribution = jacobian[i] ** 2 * m.variance
            contributions[m.source] = (
                contributions.get(m.source, 0.0) + var_contribution
            )
            kpi_details.append(
                {
                    "source": m.source,
                    "value": m.value,
                    "jacobian": jacobian[i],
                    "variance": m.variance,
                    "var_contribution": var_contribution,
                }
            )

        obj_var = sum(contributions.values())

        for detail in kpi_details:
            detail["var_fraction"] = (
                detail["var_contribution"] / obj_var if obj_var > 0 else 0.0
            )

        budget = UncertaintyBudget.from_contributions(contributions)

        return PropagationResult(
            objective_value=obj_value,
            objective_variance=obj_var,
            method=PropagationMethod.DELTA,
            budget=budget,
            kpi_details=kpi_details,
        )

    # ── Monte Carlo propagation ───────────────────────────────────────

    def monte_carlo_propagation(
        self,
        kpi_measurements: list[MeasurementWithUncertainty],
        objective_func: Callable[..., float],
        n_samples: int = 10_000,
        seed: int = 42,
    ) -> PropagationResult:
        """Propagate uncertainty via Monte Carlo sampling.

        Each KPI is sampled from N(mu, sigma) and the objective is evaluated.
        Invalid samples (where the objective raises an exception) are silently
        discarded.

        Parameters
        ----------
        kpi_measurements:
            One ``MeasurementWithUncertainty`` per KPI.
        objective_func:
            Callable accepting ``len(kpi_measurements)`` floats.
        n_samples:
            Number of Monte Carlo draws (default 10 000).
        seed:
            RNG seed for reproducibility.

        Returns
        -------
        PropagationResult
        """
        if not kpi_measurements:
            budget = UncertaintyBudget.from_contributions({})
            return PropagationResult(
                objective_value=0.0,
                objective_variance=0.0,
                method=PropagationMethod.MONTE_CARLO,
                budget=budget,
                kpi_details=[],
            )

        rng = random.Random(seed)

        # Draw samples and evaluate objective.
        results: list[float] = []
        for _ in range(n_samples):
            sample = [rng.gauss(m.value, m.std) for m in kpi_measurements]
            try:
                val = objective_func(*sample)
                if math.isfinite(val):
                    results.append(val)
            except Exception:
                # Silently discard invalid samples.
                pass

        if not results:
            # All samples failed -- return the nominal evaluation.
            nominal = objective_func(*[m.value for m in kpi_measurements])
            budget = UncertaintyBudget.from_contributions(
                {m.source: 0.0 for m in kpi_measurements}
            )
            return PropagationResult(
                objective_value=nominal,
                objective_variance=0.0,
                method=PropagationMethod.MONTE_CARLO,
                budget=budget,
                kpi_details=[],
            )

        obj_mean = sum(results) / len(results)
        obj_var = (
            sum((r - obj_mean) ** 2 for r in results) / (len(results) - 1)
            if len(results) > 1
            else 0.0
        )

        # Build approximate per-KPI contributions via variance attribution.
        # We estimate each KPI's contribution by holding all others at their
        # mean and only varying one at a time.
        contributions: dict[str, float] = {}
        kpi_details: list[dict[str, Any]] = []

        for i, m in enumerate(kpi_measurements):
            partial_results: list[float] = []
            rng_partial = random.Random(seed + i + 1)
            for _ in range(min(n_samples, 2000)):
                sample = [mm.value for mm in kpi_measurements]
                sample[i] = rng_partial.gauss(m.value, m.std)
                try:
                    val = objective_func(*sample)
                    if math.isfinite(val):
                        partial_results.append(val)
                except Exception:
                    pass

            if len(partial_results) > 1:
                partial_mean = sum(partial_results) / len(partial_results)
                partial_var = sum(
                    (r - partial_mean) ** 2 for r in partial_results
                ) / (len(partial_results) - 1)
            else:
                partial_var = 0.0

            contributions[m.source] = (
                contributions.get(m.source, 0.0) + partial_var
            )
            kpi_details.append(
                {
                    "source": m.source,
                    "value": m.value,
                    "variance": m.variance,
                    "partial_var": partial_var,
                }
            )

        total_contrib = sum(contributions.values())
        for detail in kpi_details:
            detail["var_fraction"] = (
                detail["partial_var"] / total_contrib
                if total_contrib > 0
                else 0.0
            )

        budget = UncertaintyBudget.from_contributions(contributions)

        return PropagationResult(
            objective_value=obj_mean,
            objective_variance=obj_var,
            method=PropagationMethod.MONTE_CARLO,
            budget=budget,
            kpi_details=kpi_details,
        )

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _numerical_jacobian(
        func: Callable[..., float],
        values: list[float],
        eps: float = 1e-6,
    ) -> list[float]:
        """Compute numerical Jacobian via central differences.

        J[i] = (f(x + h*e_i) - f(x - h*e_i)) / (2h)

        Parameters
        ----------
        func:
            Scalar function of ``len(values)`` arguments.
        values:
            Point at which to evaluate the Jacobian.
        eps:
            Step size for finite differences.

        Returns
        -------
        list[float]
            Partial derivatives with respect to each variable.
        """
        jacobian: list[float] = []
        for i in range(len(values)):
            upper = list(values)
            lower = list(values)
            upper[i] += eps
            lower[i] -= eps
            df = (func(*upper) - func(*lower)) / (2.0 * eps)
            jacobian.append(df)
        return jacobian

    @staticmethod
    def _aggregate_metadata(
        measurements: list[MeasurementWithUncertainty],
        contributions: dict[str, float],
    ) -> dict[str, Any]:
        """Aggregate metadata across all measurements.

        Returns
        -------
        dict with keys:
            min_confidence, mean_confidence, unreliable_kpis,
            all_quality_flags, uncertainty_budget (fractional).
        """
        if not measurements:
            return {
                "min_confidence": 1.0,
                "mean_confidence": 1.0,
                "unreliable_kpis": [],
                "all_quality_flags": [],
                "uncertainty_budget": {},
            }

        confidences = [m.confidence for m in measurements]
        min_conf = min(confidences)
        mean_conf = sum(confidences) / len(confidences)

        unreliable = [m.source for m in measurements if not m.is_reliable]

        all_flags: list[str] = []
        for m in measurements:
            flags = m.metadata.get("quality_flags", [])
            for f in flags:
                if f not in all_flags:
                    all_flags.append(f)

        total_var = sum(contributions.values())
        budget_fractions: dict[str, float] = {}
        if total_var > 0:
            for src, var in contributions.items():
                budget_fractions[src] = var / total_var

        return {
            "min_confidence": min_conf,
            "mean_confidence": mean_conf,
            "unreliable_kpis": unreliable,
            "all_quality_flags": all_flags,
            "uncertainty_budget": budget_fractions,
        }

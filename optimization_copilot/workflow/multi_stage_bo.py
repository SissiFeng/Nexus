"""DAG-based multi-stage Bayesian optimizer.

Provides :class:`MultiStageBayesianOptimizer` which orchestrates
multi-stage experimental workflows, using per-stage proxy models
and continuation value analysis to optimize experiments while
minimizing cost through early termination.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from optimization_copilot.backends._math import (
    expected_improvement,
    norm_pdf,
    norm_cdf,
)
from optimization_copilot.workflow.stage import ExperimentStage, StageDAG
from optimization_copilot.workflow.proxy_model import ProxyModel
from optimization_copilot.workflow.continue_value import ContinueValue


@dataclass
class StageResult:
    """Result of running a single experiment stage.

    Parameters
    ----------
    stage_name : str
        Name of the stage that was executed.
    parameters : dict[str, float]
        Parameter values used for this stage.
    kpi_values : dict[str, float]
        Observed KPI values from this stage.
    cost : float
        Cost incurred by running this stage.
    continued : bool
        Whether the experiment continued to the next stage.
    """

    stage_name: str
    parameters: dict[str, float]
    kpi_values: dict[str, float]
    cost: float
    continued: bool = True


class MultiStageBayesianOptimizer:
    """DAG-based multi-stage Bayesian optimizer.

    Maintains per-stage proxy models and uses continuation value analysis
    to decide whether to proceed through remaining stages. Tracks cost
    savings from early termination decisions.

    Parameters
    ----------
    stage_dag : StageDAG
        The DAG defining the experiment workflow.
    seed : int
        Random seed for reproducibility (default 42).
    """

    def __init__(self, stage_dag: StageDAG, seed: int = 42) -> None:
        self._dag = stage_dag
        self._seed = seed
        self._rng = random.Random(seed)

        # Per-stage proxy models (KPI name -> ProxyModel)
        self._proxy_models: dict[str, ProxyModel] = {}

        # Per-stage observations: stage_name -> list of (x_dict, kpi_dict)
        self._observations: dict[str, list[tuple[dict[str, float], dict[str, float]]]] = {}

        # Cost tracking
        self._total_cost_spent: float = 0.0
        self._total_cost_saved: float = 0.0
        self._early_terminations: int = 0
        self._total_evaluations: int = 0

        # Initialize observations for each stage
        for stage in self._dag.stages():
            self._observations[stage.name] = []

    def _ensure_proxy_model(self, kpi_name: str) -> ProxyModel:
        """Get or create a proxy model for a KPI.

        Parameters
        ----------
        kpi_name : str
            Name of the KPI.

        Returns
        -------
        ProxyModel
            The proxy model for this KPI.
        """
        if kpi_name not in self._proxy_models:
            self._proxy_models[kpi_name] = ProxyModel()
        return self._proxy_models[kpi_name]

    def _refit_proxy_models(self) -> None:
        """Refit all proxy models with current observations."""
        # Collect all observations per KPI
        kpi_data: dict[str, tuple[list[list[float]], list[float]]] = {}

        for stage_name, obs_list in self._observations.items():
            stage = self._dag.get_stage(stage_name)
            param_names = stage.parameters

            for x_dict, kpi_dict in obs_list:
                x_vec = [x_dict.get(p, 0.0) for p in param_names]

                for kpi_name, kpi_val in kpi_dict.items():
                    if kpi_name not in kpi_data:
                        kpi_data[kpi_name] = ([], [])
                    kpi_data[kpi_name][0].append(x_vec)
                    kpi_data[kpi_name][1].append(kpi_val)

        for kpi_name, (X, y) in kpi_data.items():
            model = self._ensure_proxy_model(kpi_name)
            if len(X) >= 1:
                model.fit(X, y)

    def add_observation(
        self,
        stage_name: str,
        x: dict[str, float],
        kpis: dict[str, float],
    ) -> None:
        """Add an observation for a specific stage.

        Parameters
        ----------
        stage_name : str
            Name of the stage.
        x : dict[str, float]
            Parameter values used.
        kpis : dict[str, float]
            Observed KPI values.

        Raises
        ------
        KeyError
            If the stage does not exist in the DAG.
        """
        if stage_name not in self._observations:
            # Verify stage exists
            self._dag.get_stage(stage_name)
            self._observations[stage_name] = []

        self._observations[stage_name].append((dict(x), dict(kpis)))
        self._refit_proxy_models()

    def suggest_next(
        self,
        current_stage: str,
        n_suggestions: int = 1,
    ) -> list[dict[str, float]]:
        """Suggest next parameter configurations to try.

        Uses Expected Improvement from proxy model predictions to suggest
        promising parameter configurations. Falls back to random sampling
        when insufficient data is available.

        Parameters
        ----------
        current_stage : str
            Name of the current stage.
        n_suggestions : int
            Number of suggestions to generate (default 1).

        Returns
        -------
        list[dict[str, float]]
            List of parameter dictionaries for the stage.
        """
        stage = self._dag.get_stage(current_stage)
        param_names = stage.parameters

        # Generate candidate points
        n_candidates = max(50, n_suggestions * 20)
        candidates: list[dict[str, float]] = []
        for _ in range(n_candidates):
            point = {p: self._rng.uniform(0.0, 1.0) for p in param_names}
            candidates.append(point)

        # If we have observations and fitted models, use EI
        stage_obs = self._observations.get(current_stage, [])
        if stage_obs and stage.kpis:
            primary_kpi = stage.kpis[0]
            model = self._proxy_models.get(primary_kpi)
            if model is not None and model.is_fitted:
                # Find best observed value for primary KPI
                best_y = float("inf")
                for _, kpi_dict in stage_obs:
                    if primary_kpi in kpi_dict:
                        best_y = min(best_y, kpi_dict[primary_kpi])

                if best_y < float("inf"):
                    # Score candidates by EI
                    scored: list[tuple[float, dict[str, float]]] = []
                    for cand in candidates:
                        x_vec = [cand.get(p, 0.0) for p in param_names]
                        mu, var = model.predict_single(x_vec)
                        sigma = math.sqrt(max(var, 0.0))
                        ei = expected_improvement(mu, sigma, best_y)
                        scored.append((ei, cand))

                    scored.sort(key=lambda t: t[0], reverse=True)
                    return [s[1] for s in scored[:n_suggestions]]

        # Fallback: return random candidates
        self._rng = random.Random(self._seed + len(stage_obs))
        suggestions: list[dict[str, float]] = []
        for _ in range(n_suggestions):
            point = {p: self._rng.uniform(0.0, 1.0) for p in param_names}
            suggestions.append(point)
        return suggestions

    def should_continue_stage(
        self,
        stage_name: str,
        x: dict[str, float],
        kpis: dict[str, float],
    ) -> bool:
        """Decide whether to continue to the next stage after observing results.

        Parameters
        ----------
        stage_name : str
            Name of the current stage.
        x : dict[str, float]
            Parameter values used.
        kpis : dict[str, float]
            Observed KPI values from this stage.

        Returns
        -------
        bool
            True if the experiment should continue to the next stage.
        """
        # If this is the last stage, no continuation
        remaining = self._get_remaining_stages(stage_name)
        if not remaining:
            return False

        # Build ContinueValue
        cv = ContinueValue(self._proxy_models, self._dag)

        # Collect all observed KPIs so far
        all_observed: dict[str, float] = dict(kpis)

        # Convert parameters to list for the proxy model
        stage = self._dag.get_stage(stage_name)
        x_vec = [x.get(p, 0.0) for p in stage.parameters]

        return cv.should_continue(stage_name, x_vec, all_observed)

    def _get_remaining_stages(self, current_stage: str) -> list[str]:
        """Get stages after the current one in topological order."""
        topo = self._dag.topological_order()
        try:
            idx = topo.index(current_stage)
        except ValueError:
            return []
        return topo[idx + 1:]

    def run_campaign(
        self,
        evaluate_fn: Callable[[str, dict[str, float]], dict[str, float]],
        n_iterations: int = 10,
    ) -> list[StageResult]:
        """Run a multi-stage optimization campaign.

        For each iteration, starts from the first stage and progresses
        through the DAG, using continuation value to decide whether to
        proceed at each stage.

        Parameters
        ----------
        evaluate_fn : callable
            Function that takes (stage_name, parameters) and returns
            a dict of KPI values.
        n_iterations : int
            Number of experiment iterations to run (default 10).

        Returns
        -------
        list[StageResult]
            All stage results from the campaign.
        """
        results: list[StageResult] = []
        topo_order = self._dag.topological_order()

        for iteration in range(n_iterations):
            self._total_evaluations += 1
            iteration_kpis: dict[str, float] = {}
            stopped_early = False

            for stage_idx, stage_name in enumerate(topo_order):
                stage = self._dag.get_stage(stage_name)

                # Get suggestion for this stage
                suggestions = self.suggest_next(stage_name, n_suggestions=1)
                params = suggestions[0] if suggestions else {
                    p: self._rng.uniform(0.0, 1.0) for p in stage.parameters
                }

                # Evaluate the stage
                kpi_values = evaluate_fn(stage_name, params)

                # Record observation
                self.add_observation(stage_name, params, kpi_values)
                iteration_kpis.update(kpi_values)

                self._total_cost_spent += stage.cost

                # Decide whether to continue
                is_last = stage_idx == len(topo_order) - 1
                if not is_last:
                    should_continue = self.should_continue_stage(
                        stage_name, params, iteration_kpis
                    )
                else:
                    should_continue = False

                result = StageResult(
                    stage_name=stage_name,
                    parameters=params,
                    kpi_values=kpi_values,
                    cost=stage.cost,
                    continued=should_continue,
                )
                results.append(result)

                if not is_last and not should_continue:
                    # Early termination: track savings
                    remaining_cost = sum(
                        self._dag.get_stage(s).cost
                        for s in topo_order[stage_idx + 1:]
                    )
                    self._total_cost_saved += remaining_cost
                    self._early_terminations += 1
                    stopped_early = True
                    break

        return results

    def get_savings_report(self) -> dict[str, Any]:
        """Generate a report of cost savings from early termination.

        Returns
        -------
        dict[str, Any]
            Report containing:
            - total_cost_spent: Total cost of executed stages
            - total_cost_saved: Total cost avoided via early termination
            - total_evaluations: Number of experiment iterations
            - early_terminations: Number of times experiments were stopped early
            - savings_ratio: Fraction of potential cost that was saved
            - per_stage_counts: Dict of stage_name -> number of observations
        """
        max_possible_cost = self._dag.total_cost() * self._total_evaluations
        savings_ratio = 0.0
        if max_possible_cost > 0:
            savings_ratio = self._total_cost_saved / max_possible_cost

        per_stage_counts: dict[str, int] = {}
        for stage_name, obs_list in self._observations.items():
            per_stage_counts[stage_name] = len(obs_list)

        return {
            "total_cost_spent": self._total_cost_spent,
            "total_cost_saved": self._total_cost_saved,
            "total_evaluations": self._total_evaluations,
            "early_terminations": self._early_terminations,
            "savings_ratio": savings_ratio,
            "per_stage_counts": per_stage_counts,
        }

    def get_proxy_model(self, kpi_name: str) -> ProxyModel:
        """Get the proxy model for a specific KPI.

        Parameters
        ----------
        kpi_name : str
            Name of the KPI.

        Returns
        -------
        ProxyModel
            The proxy model. Creates a new unfitted model if one
            doesn't exist yet.
        """
        return self._ensure_proxy_model(kpi_name)

    def get_all_observations(
        self,
        stage_name: str,
    ) -> list[tuple[dict[str, float], dict[str, float]]]:
        """Get all observations for a stage.

        Parameters
        ----------
        stage_name : str
            Name of the stage.

        Returns
        -------
        list[tuple[dict[str, float], dict[str, float]]]
            List of (parameters, kpis) tuples.
        """
        return list(self._observations.get(stage_name, []))

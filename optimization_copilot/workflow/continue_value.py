"""Knowledge Gradient over remaining stages for early termination decisions.

Provides :class:`ContinueValue` which estimates the expected value of
continuing an experiment through remaining stages versus stopping at the
current stage, accounting for stage costs and proxy model predictions.
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.backends._math import norm_pdf, norm_cdf
from optimization_copilot.workflow.stage import StageDAG
from optimization_copilot.workflow.proxy_model import ProxyModel


class ContinueValue:
    """Knowledge Gradient for multi-stage experiment continuation decisions.

    Uses proxy models to estimate the value of proceeding through remaining
    stages in the DAG, versus stopping at the current stage.

    Parameters
    ----------
    proxy_models : dict[str, ProxyModel]
        Mapping from KPI name to fitted ProxyModel.
    stage_dag : StageDAG
        The stage DAG describing the workflow.
    """

    def __init__(
        self,
        proxy_models: dict[str, ProxyModel],
        stage_dag: StageDAG,
    ) -> None:
        self._proxy_models = proxy_models
        self._dag = stage_dag

    def _get_remaining_stages(self, current_stage: str) -> list[str]:
        """Get stages that come after the current stage in topological order.

        Parameters
        ----------
        current_stage : str
            Name of the current stage.

        Returns
        -------
        list[str]
            Remaining stage names in topological order.
        """
        topo = self._dag.topological_order()
        try:
            idx = topo.index(current_stage)
        except ValueError:
            return []
        return topo[idx + 1:]

    def _get_completed_stages(self, current_stage: str) -> list[str]:
        """Get stages up to and including the current stage.

        Parameters
        ----------
        current_stage : str
            Name of the current stage.

        Returns
        -------
        list[str]
            Completed stage names in topological order.
        """
        topo = self._dag.topological_order()
        try:
            idx = topo.index(current_stage)
        except ValueError:
            return []
        return topo[: idx + 1]

    def _predict_kpi(
        self,
        kpi_name: str,
        x: list[float],
    ) -> tuple[float, float]:
        """Predict a KPI value using the proxy model.

        Parameters
        ----------
        kpi_name : str
            Name of the KPI to predict.
        x : list[float]
            Input point.

        Returns
        -------
        tuple[float, float]
            (mean, variance) prediction. Returns (0.0, 1.0) if no
            proxy model exists for this KPI.
        """
        if kpi_name not in self._proxy_models:
            return 0.0, 1.0
        model = self._proxy_models[kpi_name]
        if not model.is_fitted:
            return 0.0, 1.0
        return model.predict_single(x)

    def compute(
        self,
        current_stage: str,
        x: list[float],
        y_observed: dict[str, float],
    ) -> float:
        """Estimate expected value of continuing vs stopping.

        Computes the Knowledge Gradient: the expected improvement in the
        final objective from continuing to execute remaining stages, minus
        the cost of those stages.

        Parameters
        ----------
        current_stage : str
            Name of the current stage.
        x : list[float]
            Current parameter values.
        y_observed : dict[str, float]
            Currently observed KPI values (KPI name -> value).

        Returns
        -------
        float
            The continuation value. Positive means continuing is expected
            to be beneficial; negative means stopping is preferable.
        """
        remaining = self._get_remaining_stages(current_stage)
        if not remaining:
            return 0.0

        # Current observed value: average of observed KPIs
        if y_observed:
            current_value = sum(y_observed.values()) / len(y_observed)
        else:
            current_value = 0.0

        # Estimate expected value from remaining stages
        expected_gain = 0.0
        total_remaining_cost = 0.0

        for stage_name in remaining:
            stage = self._dag.get_stage(stage_name)
            total_remaining_cost += stage.cost

            # For each KPI in this stage, estimate the expected improvement
            for kpi_name in stage.kpis:
                if kpi_name in y_observed:
                    # Already observed, no additional information
                    continue

                mu, var = self._predict_kpi(kpi_name, x)
                sigma = math.sqrt(max(var, 1e-12))

                # Expected improvement from learning this KPI
                # Uses the standard EI formula as a proxy for information gain
                if sigma > 1e-10:
                    z = (current_value - mu) / sigma
                    ei = (current_value - mu) * norm_cdf(z) + sigma * norm_pdf(z)
                    expected_gain += ei

        # Net value: expected information gain minus cost of remaining stages
        continuation_value = expected_gain - total_remaining_cost
        return continuation_value

    def should_continue(
        self,
        current_stage: str,
        x: list[float],
        y_observed: dict[str, float],
        threshold: float = 0.0,
    ) -> bool:
        """Decide whether to continue to the next stage.

        Parameters
        ----------
        current_stage : str
            Name of the current stage.
        x : list[float]
            Current parameter values.
        y_observed : dict[str, float]
            Currently observed KPI values.
        threshold : float
            Minimum continuation value required to proceed (default 0.0).

        Returns
        -------
        bool
            True if the expected value of continuing exceeds the threshold.
        """
        remaining = self._get_remaining_stages(current_stage)
        if not remaining:
            return False
        value = self.compute(current_stage, x, y_observed)
        return value > threshold

    def expected_final_value(
        self,
        x: list[float],
        y_observed: dict[str, float],
    ) -> float:
        """Estimate the expected final objective value.

        Combines observed KPIs with proxy model predictions for unobserved
        KPIs across all stages.

        Parameters
        ----------
        x : list[float]
            Parameter values.
        y_observed : dict[str, float]
            Currently observed KPI values.

        Returns
        -------
        float
            Estimated final objective value (sum of all KPI contributions).
        """
        total = 0.0
        count = 0
        all_kpis: set[str] = set()

        for stage in self._dag.stages():
            for kpi_name in stage.kpis:
                all_kpis.add(kpi_name)

        for kpi_name in all_kpis:
            if kpi_name in y_observed:
                total += y_observed[kpi_name]
            else:
                mu, _ = self._predict_kpi(kpi_name, x)
                total += mu
            count += 1

        if count == 0:
            return 0.0
        return total / count

    def stage_information_value(
        self,
        stage_name: str,
        x: list[float],
        y_observed: dict[str, float],
    ) -> float:
        """Estimate the information value of running a specific stage.

        Parameters
        ----------
        stage_name : str
            Name of the stage to evaluate.
        x : list[float]
            Parameter values.
        y_observed : dict[str, float]
            Currently observed KPI values.

        Returns
        -------
        float
            Information value of the stage minus its cost.
        """
        stage = self._dag.get_stage(stage_name)

        if y_observed:
            current_value = sum(y_observed.values()) / len(y_observed)
        else:
            current_value = 0.0

        info_value = 0.0
        for kpi_name in stage.kpis:
            if kpi_name in y_observed:
                continue
            mu, var = self._predict_kpi(kpi_name, x)
            sigma = math.sqrt(max(var, 1e-12))
            if sigma > 1e-10:
                z = (current_value - mu) / sigma
                ei = (current_value - mu) * norm_cdf(z) + sigma * norm_pdf(z)
                info_value += ei

        return info_value - stage.cost

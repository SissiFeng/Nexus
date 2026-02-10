"""High-dimensional variable screening to identify important parameters."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from optimization_copilot.core.models import CampaignSnapshot, ParameterSpec, VariableType


@dataclass
class ScreeningResult:
    """Output of variable screening."""
    ranked_parameters: list[str]  # Most important first
    importance_scores: dict[str, float]  # name -> score [0,1]
    suspected_interactions: list[tuple[str, str]]
    directionality: dict[str, float]  # name -> positive/negative correlation sign
    recommended_step_sizes: dict[str, float]


class VariableScreener:
    """Screen parameters for importance using Morris one-at-a-time and simple methods."""

    def screen(self, snapshot: CampaignSnapshot, seed: int = 42) -> ScreeningResult:
        obs = snapshot.successful_observations
        if len(obs) < 3 or not snapshot.parameter_specs:
            return self._empty_result(snapshot.parameter_names)

        obj_name = snapshot.objective_names[0] if snapshot.objective_names else None
        if obj_name is None:
            return self._empty_result(snapshot.parameter_names)

        # Compute importance via correlation-based method
        importance = self._correlation_importance(obs, snapshot.parameter_specs, obj_name)

        # Compute directionality
        directionality = self._compute_directionality(obs, snapshot.parameter_specs, obj_name)

        # Detect interactions (simple: pairs with high joint variance effect)
        interactions = self._detect_interactions(
            obs, snapshot.parameter_specs, obj_name, seed
        )

        # Step sizes: proportional to parameter range / importance
        step_sizes = self._compute_step_sizes(snapshot.parameter_specs, importance)

        # Rank
        ranked = sorted(importance, key=importance.get, reverse=True)

        return ScreeningResult(
            ranked_parameters=ranked,
            importance_scores=importance,
            suspected_interactions=interactions,
            directionality=directionality,
            recommended_step_sizes=step_sizes,
        )

    @staticmethod
    def _correlation_importance(
        obs: list, specs: list[ParameterSpec], obj_name: str
    ) -> dict[str, float]:
        """Simple absolute correlation between each param and KPI."""
        kpi_vals = [o.kpi_values.get(obj_name, 0.0) for o in obs]
        n = len(kpi_vals)
        if n < 2:
            return {s.name: 0.0 for s in specs}

        kpi_mean = sum(kpi_vals) / n
        kpi_std = (sum((v - kpi_mean) ** 2 for v in kpi_vals) / n) ** 0.5

        importance: dict[str, float] = {}
        for spec in specs:
            if spec.type == VariableType.CATEGORICAL:
                importance[spec.name] = 0.0
                continue
            param_vals = [o.parameters.get(spec.name, 0.0) for o in obs]
            p_mean = sum(param_vals) / n
            p_std = (sum((v - p_mean) ** 2 for v in param_vals) / n) ** 0.5

            if p_std < 1e-12 or kpi_std < 1e-12:
                importance[spec.name] = 0.0
                continue

            cov = sum(
                (param_vals[i] - p_mean) * (kpi_vals[i] - kpi_mean) for i in range(n)
            ) / n
            corr = abs(cov / (p_std * kpi_std))
            importance[spec.name] = min(corr, 1.0)

        return importance

    @staticmethod
    def _compute_directionality(
        obs: list, specs: list[ParameterSpec], obj_name: str
    ) -> dict[str, float]:
        kpi_vals = [o.kpi_values.get(obj_name, 0.0) for o in obs]
        n = len(kpi_vals)
        if n < 2:
            return {s.name: 0.0 for s in specs}

        kpi_mean = sum(kpi_vals) / n
        result: dict[str, float] = {}
        for spec in specs:
            if spec.type == VariableType.CATEGORICAL:
                result[spec.name] = 0.0
                continue
            param_vals = [o.parameters.get(spec.name, 0.0) for o in obs]
            p_mean = sum(param_vals) / n
            cov = sum(
                (param_vals[i] - p_mean) * (kpi_vals[i] - kpi_mean) for i in range(n)
            ) / n
            result[spec.name] = 1.0 if cov > 0 else (-1.0 if cov < 0 else 0.0)

        return result

    @staticmethod
    def _detect_interactions(
        obs: list, specs: list[ParameterSpec], obj_name: str, seed: int
    ) -> list[tuple[str, str]]:
        """Detect parameter interactions via simple product-term correlation."""
        if len(obs) < 5 or len(specs) < 2:
            return []

        kpi_vals = [o.kpi_values.get(obj_name, 0.0) for o in obs]
        n = len(kpi_vals)
        kpi_mean = sum(kpi_vals) / n
        kpi_std = (sum((v - kpi_mean) ** 2 for v in kpi_vals) / n) ** 0.5
        if kpi_std < 1e-12:
            return []

        continuous_specs = [s for s in specs if s.type != VariableType.CATEGORICAL]
        interactions = []

        for i, s1 in enumerate(continuous_specs):
            for s2 in continuous_specs[i + 1:]:
                product = [
                    o.parameters.get(s1.name, 0.0) * o.parameters.get(s2.name, 0.0)
                    for o in obs
                ]
                p_mean = sum(product) / n
                p_std = (sum((v - p_mean) ** 2 for v in product) / n) ** 0.5
                if p_std < 1e-12:
                    continue
                cov = sum(
                    (product[j] - p_mean) * (kpi_vals[j] - kpi_mean) for j in range(n)
                ) / n
                corr = abs(cov / (p_std * kpi_std))
                if corr > 0.5:
                    interactions.append((s1.name, s2.name))

        return interactions

    @staticmethod
    def _compute_step_sizes(
        specs: list[ParameterSpec], importance: dict[str, float]
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for spec in specs:
            if spec.lower is not None and spec.upper is not None:
                param_range = spec.upper - spec.lower
                imp = importance.get(spec.name, 0.0)
                # More important → smaller step (finer resolution)
                factor = 0.05 if imp > 0.5 else (0.1 if imp > 0.2 else 0.2)
                result[spec.name] = param_range * factor
            else:
                result[spec.name] = 0.1
        return result

    @staticmethod
    def _empty_result(param_names: list[str]) -> ScreeningResult:
        return ScreeningResult(
            ranked_parameters=param_names,
            importance_scores={n: 0.0 for n in param_names},
            suspected_interactions=[],
            directionality={n: 0.0 for n in param_names},
            recommended_step_sizes={n: 0.1 for n in param_names},
        )

"""Data conditioning and noise handling for optimization campaigns."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from optimization_copilot.core.models import CampaignSnapshot, Observation, StabilizeSpec


@dataclass
class StabilizedData:
    """Result of data stabilization."""
    observations: list[Observation]
    removed_indices: list[int] = field(default_factory=list)
    smoothed_kpis: dict[str, list[float]] = field(default_factory=dict)
    applied_policies: list[str] = field(default_factory=list)


class Stabilizer:
    """Apply data conditioning policies to campaign observations."""

    def stabilize(
        self, snapshot: CampaignSnapshot, spec: StabilizeSpec
    ) -> StabilizedData:
        obs = list(snapshot.observations)
        removed: list[int] = []
        policies: list[str] = []

        # 1. Failure handling
        obs, fail_removed = self._handle_failures(obs, spec.failure_handling)
        removed.extend(fail_removed)
        if fail_removed:
            policies.append(f"failure_handling:{spec.failure_handling}")

        # 2. Outlier rejection
        obs, outlier_removed = self._reject_outliers(
            obs, snapshot.objective_names, spec.outlier_rejection_sigma
        )
        removed.extend(outlier_removed)
        if outlier_removed:
            policies.append(f"outlier_rejection:sigma={spec.outlier_rejection_sigma}")

        # 3. Reweighting (mark in metadata, not actually remove)
        if spec.reweighting_strategy != "none":
            obs = self._apply_reweighting(obs, spec.reweighting_strategy)
            policies.append(f"reweighting:{spec.reweighting_strategy}")

        # 4. Noise smoothing
        smoothed = {}
        if spec.noise_smoothing_window > 1 and snapshot.objective_names:
            for obj_name in snapshot.objective_names:
                values = [
                    o.kpi_values.get(obj_name, 0.0) for o in obs if not o.is_failure
                ]
                smoothed[obj_name] = self._moving_average(
                    values, spec.noise_smoothing_window
                )
            policies.append(f"noise_smoothing:window={spec.noise_smoothing_window}")

        return StabilizedData(
            observations=obs,
            removed_indices=removed,
            smoothed_kpis=smoothed,
            applied_policies=policies,
        )

    @staticmethod
    def _handle_failures(
        obs: list[Observation], policy: str
    ) -> tuple[list[Observation], list[int]]:
        removed = []
        if policy == "exclude":
            result = []
            for i, o in enumerate(obs):
                if o.is_failure:
                    removed.append(i)
                else:
                    result.append(o)
            return result, removed
        elif policy == "penalize":
            # Keep failures but mark them — no removal
            return obs, []
        elif policy == "impute":
            # Replace failure KPIs with worst observed value
            valid_kpis: dict[str, list[float]] = {}
            for o in obs:
                if not o.is_failure:
                    for k, v in o.kpi_values.items():
                        valid_kpis.setdefault(k, []).append(v)
            worst: dict[str, float] = {}
            for k, vals in valid_kpis.items():
                worst[k] = min(vals) if vals else 0.0
            result = []
            for o in obs:
                if o.is_failure and worst:
                    new_obs = Observation(
                        iteration=o.iteration,
                        parameters=o.parameters,
                        kpi_values=dict(worst),
                        qc_passed=o.qc_passed,
                        is_failure=True,
                        failure_reason=o.failure_reason,
                        timestamp=o.timestamp,
                        metadata={**o.metadata, "imputed": True},
                    )
                    result.append(new_obs)
                else:
                    result.append(o)
            return result, []
        return obs, []

    @staticmethod
    def _reject_outliers(
        obs: list[Observation], objective_names: list[str], sigma: float
    ) -> tuple[list[Observation], list[int]]:
        if not objective_names or sigma <= 0:
            return obs, []

        removed = []
        for obj_name in objective_names:
            values = [
                o.kpi_values.get(obj_name, 0.0) for o in obs if not o.is_failure
            ]
            if len(values) < 3:
                continue
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 0.0
            if std == 0:
                continue
            for i, o in enumerate(obs):
                if o.is_failure:
                    continue
                val = o.kpi_values.get(obj_name, 0.0)
                if abs(val - mean) > sigma * std:
                    removed.append(i)

        removed_set = set(removed)
        result = [o for i, o in enumerate(obs) if i not in removed_set]
        return result, sorted(removed_set)

    @staticmethod
    def _apply_reweighting(
        obs: list[Observation], strategy: str
    ) -> list[Observation]:
        n = len(obs)
        if n == 0:
            return obs
        result = []
        for i, o in enumerate(obs):
            weight = 1.0
            if strategy == "recency":
                weight = (i + 1) / n
            elif strategy == "quality":
                weight = 1.0 if o.qc_passed and not o.is_failure else 0.5
            new_meta = {**o.metadata, "weight": weight}
            result.append(Observation(
                iteration=o.iteration,
                parameters=o.parameters,
                kpi_values=o.kpi_values,
                qc_passed=o.qc_passed,
                is_failure=o.is_failure,
                failure_reason=o.failure_reason,
                timestamp=o.timestamp,
                metadata=new_meta,
            ))
        return result

    @staticmethod
    def _moving_average(values: list[float], window: int) -> list[float]:
        if not values:
            return []
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_vals = values[start:i + 1]
            result.append(sum(window_vals) / len(window_vals))
        return result

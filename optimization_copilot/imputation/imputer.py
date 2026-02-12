"""Deterministic imputer with full traceability and audit trail."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from optimization_copilot.core.models import Observation, ParameterSpec, VariableType
from optimization_copilot.imputation.models import (
    ImputationConfig,
    ImputationRecord,
    ImputationResult,
    ImputationStrategy,
)


class DeterministicImputer:
    """Deterministic imputation engine for optimization observations.

    Replaces missing or failed KPI values using configurable strategies,
    producing a complete audit trail for every imputed value. Given
    identical inputs and configuration, always produces identical output
    (including the decision hash).

    Supported strategies
    --------------------
    - WORST_VALUE: Replace with the minimum observed valid value.
    - COLUMN_MEDIAN: Replace with the median of observed valid values.
    - COLUMN_MEAN: Replace with the mean of observed valid values.
    - KNN_PROXY: Replace with the mean KPI value of k nearest
      neighbors in normalized parameter space.
    """

    def __init__(self, config: ImputationConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def impute(
        self,
        observations: list[Observation],
        objective_names: list[str],
        parameter_specs: list[ParameterSpec],
    ) -> ImputationResult:
        """Run deterministic imputation on *observations*.

        Parameters
        ----------
        observations :
            All observations in the campaign (successes and failures).
        objective_names :
            KPI names that are optimization objectives.
        parameter_specs :
            Parameter specifications for distance calculations (KNN).

        Returns
        -------
        ImputationResult
            Imputed observations with full audit trail and decision hash.
        """
        if not observations:
            return ImputationResult(
                observations=[],
                records=[],
                decision_hash=self._compute_decision_hash([], self._config),
                config_used=self._config,
            )

        # Collect valid (non-missing) values per KPI from non-failure observations
        valid_kpis: dict[str, list[float]] = {}
        for obs in observations:
            if not obs.is_failure:
                for k, v in obs.kpi_values.items():
                    valid_kpis.setdefault(k, []).append(v)

        # Determine which KPIs we expect (union of objective_names and any
        # KPIs seen in valid observations)
        all_kpi_names: list[str] = list(
            dict.fromkeys(objective_names + list(valid_kpis.keys()))
        )

        result_obs: list[Observation] = []
        records: list[ImputationRecord] = []

        for idx, obs in enumerate(observations):
            needs_imputation = self._needs_imputation(obs, all_kpi_names)
            if not needs_imputation:
                result_obs.append(obs)
                continue

            new_kpi_values = dict(obs.kpi_values)
            obs_records: list[ImputationRecord] = []

            for kpi_name in all_kpi_names:
                if kpi_name in obs.kpi_values:
                    # KPI present -- no imputation needed for this one
                    continue

                strategy = self._get_strategy_for_kpi(kpi_name)
                valid_vals = valid_kpis.get(kpi_name, [])

                if strategy == ImputationStrategy.KNN_PROXY:
                    imputed_value, detail = self._impute_knn_proxy(
                        obs, observations, kpi_name, parameter_specs,
                        self._config.knn_k,
                    )
                elif strategy == ImputationStrategy.COLUMN_MEDIAN:
                    imputed_value, detail = self._impute_column_median(valid_vals)
                elif strategy == ImputationStrategy.COLUMN_MEAN:
                    imputed_value, detail = self._impute_column_mean(valid_vals)
                else:
                    # Default: WORST_VALUE
                    imputed_value, detail = self._impute_worst_value(valid_vals)

                new_kpi_values[kpi_name] = imputed_value

                record = ImputationRecord(
                    observation_index=idx,
                    kpi_name=kpi_name,
                    original_value=obs.kpi_values.get(kpi_name),
                    imputed_value=imputed_value,
                    strategy=strategy,
                    source_columns=list(valid_kpis.keys()) if strategy != ImputationStrategy.KNN_PROXY else [kpi_name],
                    k_neighbors=detail.get("k") if strategy == ImputationStrategy.KNN_PROXY else None,
                    neighbor_indices=detail.get("neighbor_indices") if strategy == ImputationStrategy.KNN_PROXY else None,
                )
                obs_records.append(record)

            records.extend(obs_records)

            if not obs_records:
                # Nothing was actually imputed (e.g. no expected KPIs)
                result_obs.append(obs)
                continue

            # Build enriched metadata
            new_metadata: dict[str, Any] = {**obs.metadata}
            new_metadata["imputed"] = True
            new_metadata["imputation_method"] = obs_records[-1].strategy.value
            new_metadata["source_columns"] = [r.kpi_name for r in obs_records]
            # Store per-kpi detail
            new_metadata["imputation_details"] = {
                r.kpi_name: {
                    "method": r.strategy.value,
                    "original_value": r.original_value,
                    "imputed_value": r.imputed_value,
                }
                for r in obs_records
            }

            new_obs = Observation(
                iteration=obs.iteration,
                parameters=obs.parameters,
                kpi_values=new_kpi_values,
                qc_passed=obs.qc_passed,
                is_failure=obs.is_failure,
                failure_reason=obs.failure_reason,
                timestamp=obs.timestamp,
                metadata=new_metadata,
            )
            result_obs.append(new_obs)

        decision_hash = self._compute_decision_hash(observations, self._config)

        return ImputationResult(
            observations=result_obs,
            records=records,
            decision_hash=decision_hash,
            config_used=self._config,
        )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _impute_worst_value(valid_values: list[float]) -> tuple[float, dict[str, Any]]:
        """Return the minimum of *valid_values* (worst-case imputation).

        Returns 0.0 when no valid values are available.
        """
        if not valid_values:
            return 0.0, {"method": "worst_value", "worst": 0.0}
        worst = min(valid_values)
        return worst, {"method": "worst_value", "worst": worst}

    @staticmethod
    def _impute_column_median(valid_values: list[float]) -> tuple[float, dict[str, Any]]:
        """Return the median of *valid_values*.

        For even-length lists, returns the average of the two middle values.
        Returns 0.0 when no valid values are available.
        """
        if not valid_values:
            return 0.0, {"method": "column_median", "median": 0.0}
        sorted_vals = sorted(valid_values)
        n = len(sorted_vals)
        if n % 2 == 1:
            median = sorted_vals[n // 2]
        else:
            median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
        return median, {"method": "column_median", "median": median}

    @staticmethod
    def _impute_column_mean(valid_values: list[float]) -> tuple[float, dict[str, Any]]:
        """Return the arithmetic mean of *valid_values*.

        Returns 0.0 when no valid values are available.
        """
        if not valid_values:
            return 0.0, {"method": "column_mean", "mean": 0.0}
        mean = sum(valid_values) / len(valid_values)
        return mean, {"method": "column_mean", "mean": mean}

    @staticmethod
    def _impute_knn_proxy(
        target_obs: Observation,
        all_obs: list[Observation],
        kpi_name: str,
        parameter_specs: list[ParameterSpec],
        k: int,
    ) -> tuple[float, dict[str, Any]]:
        """Impute using the mean KPI of *k* nearest non-missing neighbors.

        Distance is computed in normalized parameter space:
        - Continuous/discrete: (val - lower) / (upper - lower)
        - Categorical: 0 if same category, 1 otherwise

        If fewer than *k* valid neighbors exist, all available are used.
        Returns 0.0 if no valid neighbors exist.
        """
        # Collect candidate neighbors: non-failure observations with the KPI
        candidates: list[tuple[int, Observation, float]] = []  # (original_idx, obs, distance)

        for orig_idx, obs in enumerate(all_obs):
            if obs is target_obs:
                continue
            if obs.is_failure:
                continue
            if kpi_name not in obs.kpi_values:
                continue

            dist = _euclidean_distance(target_obs, obs, parameter_specs)
            candidates.append((orig_idx, obs, dist))

        if not candidates:
            return 0.0, {
                "method": "knn_proxy",
                "k": k,
                "neighbor_indices": [],
                "neighbor_distances": [],
            }

        # Sort by distance (deterministic: ties broken by original index)
        candidates.sort(key=lambda c: (c[2], c[0]))

        # Take at most k neighbors
        selected = candidates[:k]
        neighbor_indices = [c[0] for c in selected]
        neighbor_distances = [c[2] for c in selected]
        kpi_values = [c[1].kpi_values[kpi_name] for c in selected]

        imputed = sum(kpi_values) / len(kpi_values)

        return imputed, {
            "method": "knn_proxy",
            "k": len(selected),
            "neighbor_indices": neighbor_indices,
            "neighbor_distances": neighbor_distances,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_decision_hash(
        observations: list[Observation],
        config: ImputationConfig,
    ) -> str:
        """Compute a deterministic SHA-256 hash of inputs + config.

        The same (observations, config) always produces an identical hash,
        guaranteeing reproducibility verification.
        """
        # Build a canonical, sorted representation
        obs_data: list[tuple[Any, ...]] = []
        for obs in observations:
            sorted_params = tuple(sorted(obs.parameters.items()))
            sorted_kpis = tuple(sorted(obs.kpi_values.items()))
            obs_data.append((obs.iteration, sorted_params, sorted_kpis, obs.is_failure))

        config_data = (
            config.strategy.value,
            config.seed,
            config.knn_k,
            tuple(sorted(config.per_kpi_strategy.items())) if config.per_kpi_strategy else (),
        )

        payload = json.dumps(
            {"observations": obs_data, "config": config_data},
            sort_keys=True,
            default=str,
        )
        full_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return full_hash[:16]

    def _get_strategy_for_kpi(self, kpi_name: str) -> ImputationStrategy:
        """Return the strategy for *kpi_name*, respecting per-KPI overrides."""
        if self._config.per_kpi_strategy and kpi_name in self._config.per_kpi_strategy:
            return self._config.per_kpi_strategy[kpi_name]
        return self._config.strategy

    @staticmethod
    def _needs_imputation(obs: Observation, all_kpi_names: list[str]) -> bool:
        """Return True if *obs* needs imputation.

        An observation needs imputation when:
        - It is a failure (is_failure=True), OR
        - It is missing one or more expected KPI values.
        """
        if obs.is_failure:
            return True
        for kpi in all_kpi_names:
            if kpi not in obs.kpi_values:
                return True
        return False


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _euclidean_distance(
    a: Observation,
    b: Observation,
    specs: list[ParameterSpec],
) -> float:
    """Normalized Euclidean distance between two observations in parameter space.

    Continuous/discrete parameters are normalized to [0, 1] using their
    specification bounds. Categorical parameters contribute 0 (match) or 1
    (mismatch) per dimension.
    """
    total = 0.0
    for spec in specs:
        val_a = a.parameters.get(spec.name)
        val_b = b.parameters.get(spec.name)

        if spec.type == VariableType.CATEGORICAL:
            total += 0.0 if val_a == val_b else 1.0
        else:
            # Continuous or discrete: normalize
            lo = spec.lower if spec.lower is not None else 0.0
            hi = spec.upper if spec.upper is not None else 1.0
            rng = hi - lo
            if rng == 0:
                total += 0.0
                continue
            norm_a = (float(val_a) - lo) / rng if val_a is not None else 0.5
            norm_b = (float(val_b) - lo) / rng if val_b is not None else 0.5
            total += (norm_a - norm_b) ** 2

    return math.sqrt(total)

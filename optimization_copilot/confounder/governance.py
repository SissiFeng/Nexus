"""Confounder governor: applies correction policies to a CampaignSnapshot."""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.confounder.models import (
    ConfounderAuditTrail,
    ConfounderConfig,
    ConfounderCorrectionRecord,
    ConfounderPolicy,
    ConfounderSpec,
)


class ConfounderGovernor:
    """Applies confounder correction policies to a campaign snapshot.

    The governor iterates through the configured confounder specs and applies
    each policy in order.  The original snapshot is **never mutated**; a new
    snapshot is constructed with every correction step.

    Parameters
    ----------
    config : ConfounderConfig
        Configuration describing which confounders to correct and how.
    """

    def __init__(self, config: ConfounderConfig) -> None:
        self.config = config

    # -- Public API ---------------------------------------------------------

    def apply(
        self,
        snapshot: CampaignSnapshot,
        seed: int = 42,
    ) -> tuple[CampaignSnapshot, ConfounderAuditTrail]:
        """Apply all configured confounder corrections to *snapshot*.

        Parameters
        ----------
        snapshot : CampaignSnapshot
            The input campaign state.  Not mutated.
        seed : int
            Random seed reserved for future stochastic methods.

        Returns
        -------
        tuple[CampaignSnapshot, ConfounderAuditTrail]
            A new (corrected) snapshot and an audit trail documenting every
            correction step.
        """
        current = self._clone_snapshot(snapshot)
        corrections: list[ConfounderCorrectionRecord] = []

        for spec in self.config.confounders:
            handler = self._dispatch(spec.policy)
            current, record = handler(current, spec)
            corrections.append(record)

        summary_parts: list[str] = []
        for rec in corrections:
            summary_parts.append(
                f"{rec.column_name}: {rec.policy.value} "
                f"({rec.n_affected_rows} rows affected)"
            )
        summary = "; ".join(summary_parts) if summary_parts else "No corrections applied."

        audit = ConfounderAuditTrail(
            corrections=corrections,
            config_used=self.config,
            summary=summary,
        )
        return current, audit

    # -- Dispatch -----------------------------------------------------------

    def _dispatch(self, policy: ConfounderPolicy):  # noqa: ANN202
        """Return the handler method for *policy*."""
        mapping = {
            ConfounderPolicy.COVARIATE: self._apply_covariate,
            ConfounderPolicy.NORMALIZE: self._apply_normalize,
            ConfounderPolicy.HIGH_RISK_FLAG: self._apply_high_risk_flag,
            ConfounderPolicy.EXCLUDE: self._apply_exclude,
        }
        return mapping[policy]

    # -- COVARIATE ----------------------------------------------------------

    def _apply_covariate(
        self,
        snapshot: CampaignSnapshot,
        spec: ConfounderSpec,
    ) -> tuple[CampaignSnapshot, ConfounderCorrectionRecord]:
        """Promote the confounder to a formal continuous parameter.

        For every observation that carries ``spec.column_name`` in its metadata
        the value is copied into ``observation.parameters``.  A new
        ``ParameterSpec`` is added to the snapshot's parameter list.
        """
        original_stats = self._kpi_stats(snapshot.observations, snapshot.objective_names)

        # Gather confounder values from metadata to determine bounds.
        values: list[float] = []
        for obs in snapshot.observations:
            raw = obs.metadata.get(spec.column_name)
            if raw is not None:
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    pass

        lo = min(values) if values else 0.0
        hi = max(values) if values else 1.0
        if lo == hi:
            hi = lo + 1.0  # avoid degenerate range

        # Build new parameter spec.
        new_param = ParameterSpec(
            name=spec.column_name,
            type=VariableType.CONTINUOUS,
            lower=lo,
            upper=hi,
        )

        # Check if parameter spec already exists
        existing_names = {p.name for p in snapshot.parameter_specs}
        new_specs = list(snapshot.parameter_specs)
        if spec.column_name not in existing_names:
            new_specs.append(new_param)

        # Copy metadata value into each observation's parameters.
        new_obs: list[Observation] = []
        n_affected = 0
        for obs in snapshot.observations:
            raw = obs.metadata.get(spec.column_name)
            new_params = dict(obs.parameters)
            new_meta = dict(obs.metadata)
            if raw is not None:
                try:
                    new_params[spec.column_name] = float(raw)
                    n_affected += 1
                except (TypeError, ValueError):
                    pass
            new_obs.append(Observation(
                iteration=obs.iteration,
                parameters=new_params,
                kpi_values=dict(obs.kpi_values),
                qc_passed=obs.qc_passed,
                is_failure=obs.is_failure,
                failure_reason=obs.failure_reason,
                timestamp=obs.timestamp,
                metadata=new_meta,
            ))

        new_snapshot = CampaignSnapshot(
            campaign_id=snapshot.campaign_id,
            parameter_specs=new_specs,
            observations=new_obs,
            objective_names=list(snapshot.objective_names),
            objective_directions=list(snapshot.objective_directions),
            constraints=list(snapshot.constraints),
            current_iteration=snapshot.current_iteration,
            metadata=dict(snapshot.metadata),
        )

        corrected_stats = self._kpi_stats(new_snapshot.observations, new_snapshot.objective_names)

        record = ConfounderCorrectionRecord(
            column_name=spec.column_name,
            policy=ConfounderPolicy.COVARIATE,
            n_affected_rows=n_affected,
            correction_details={
                "param_lower": lo,
                "param_upper": hi,
            },
            original_kpi_stats=original_stats,
            corrected_kpi_stats=corrected_stats,
        )
        return new_snapshot, record

    # -- NORMALIZE ----------------------------------------------------------

    def _apply_normalize(
        self,
        snapshot: CampaignSnapshot,
        spec: ConfounderSpec,
    ) -> tuple[CampaignSnapshot, ConfounderCorrectionRecord]:
        """Remove the confounder's linear effect from each KPI.

        For each objective, the method fits ``KPI ~ confounder`` using
        ordinary least squares (pure Python) and replaces KPI values with
        ``residual + mean(KPI)`` so that the original mean is preserved.
        """
        original_stats = self._kpi_stats(snapshot.observations, snapshot.objective_names)

        # Extract confounder values.
        conf_values: list[float | None] = []
        for obs in snapshot.observations:
            raw = obs.metadata.get(spec.column_name)
            if raw is not None:
                try:
                    conf_values.append(float(raw))
                except (TypeError, ValueError):
                    conf_values.append(None)
            else:
                conf_values.append(None)

        # Deep-copy observations for mutation.
        new_obs: list[Observation] = []
        for obs in snapshot.observations:
            new_obs.append(Observation(
                iteration=obs.iteration,
                parameters=dict(obs.parameters),
                kpi_values=dict(obs.kpi_values),
                qc_passed=obs.qc_passed,
                is_failure=obs.is_failure,
                failure_reason=obs.failure_reason,
                timestamp=obs.timestamp,
                metadata=dict(obs.metadata),
            ))

        n_affected = 0
        details: dict[str, Any] = {}

        for obj_name in snapshot.objective_names:
            # Build paired (x, y) lists for observations that have both.
            xs: list[float] = []
            ys: list[float] = []
            indices: list[int] = []
            for i, obs in enumerate(snapshot.observations):
                if conf_values[i] is not None and obj_name in obs.kpi_values:
                    xs.append(conf_values[i])  # type: ignore[arg-type]
                    ys.append(obs.kpi_values[obj_name])
                    indices.append(i)

            if len(xs) < 2:
                continue

            residuals = self._linear_regression_residuals(xs, ys)
            y_mean = sum(ys) / len(ys)

            for j, idx in enumerate(indices):
                new_obs[idx].kpi_values[obj_name] = residuals[j] + y_mean

            n_affected = max(n_affected, len(indices))

            # Record regression details.
            x_mean = sum(xs) / len(xs)
            x_var = sum((xi - x_mean) ** 2 for xi in xs) / len(xs)
            if x_var > 1e-15:
                cov = sum((xs[k] - x_mean) * (ys[k] - y_mean) for k in range(len(xs))) / len(xs)
                slope = cov / x_var
            else:
                slope = 0.0
            intercept = y_mean - slope * x_mean
            details[obj_name] = {
                "slope": round(slope, 6),
                "intercept": round(intercept, 6),
                "n_points": len(xs),
            }

        new_snapshot = CampaignSnapshot(
            campaign_id=snapshot.campaign_id,
            parameter_specs=list(snapshot.parameter_specs),
            observations=new_obs,
            objective_names=list(snapshot.objective_names),
            objective_directions=list(snapshot.objective_directions),
            constraints=list(snapshot.constraints),
            current_iteration=snapshot.current_iteration,
            metadata=dict(snapshot.metadata),
        )

        corrected_stats = self._kpi_stats(new_snapshot.observations, new_snapshot.objective_names)

        record = ConfounderCorrectionRecord(
            column_name=spec.column_name,
            policy=ConfounderPolicy.NORMALIZE,
            n_affected_rows=n_affected,
            correction_details=details,
            original_kpi_stats=original_stats,
            corrected_kpi_stats=corrected_stats,
        )
        return new_snapshot, record

    # -- HIGH_RISK_FLAG -----------------------------------------------------

    def _apply_high_risk_flag(
        self,
        snapshot: CampaignSnapshot,
        spec: ConfounderSpec,
    ) -> tuple[CampaignSnapshot, ConfounderCorrectionRecord]:
        """Down-weight observations whose confounder falls outside thresholds.

        Observations that violate the thresholds have their
        ``metadata["weight"]`` halved and ``metadata["confounder_flagged"]``
        set to ``True``.  If no thresholds are configured, **all** observations
        with a numeric confounder value are flagged.
        """
        original_stats = self._kpi_stats(snapshot.observations, snapshot.objective_names)

        new_obs: list[Observation] = []
        n_affected = 0

        for obs in snapshot.observations:
            new_params = dict(obs.parameters)
            new_kpi = dict(obs.kpi_values)
            new_meta = dict(obs.metadata)

            raw = obs.metadata.get(spec.column_name)
            flagged = False
            if raw is not None:
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    val = None

                if val is not None:
                    if spec.threshold_low is None and spec.threshold_high is None:
                        # No thresholds set -- flag all with a numeric value.
                        flagged = True
                    else:
                        if spec.threshold_low is not None and val < spec.threshold_low:
                            flagged = True
                        if spec.threshold_high is not None and val > spec.threshold_high:
                            flagged = True

            if flagged:
                current_weight = new_meta.get("weight", 1.0)
                new_meta["weight"] = current_weight * 0.5
                new_meta["confounder_flagged"] = True
                n_affected += 1

            new_obs.append(Observation(
                iteration=obs.iteration,
                parameters=new_params,
                kpi_values=new_kpi,
                qc_passed=obs.qc_passed,
                is_failure=obs.is_failure,
                failure_reason=obs.failure_reason,
                timestamp=obs.timestamp,
                metadata=new_meta,
            ))

        new_snapshot = CampaignSnapshot(
            campaign_id=snapshot.campaign_id,
            parameter_specs=list(snapshot.parameter_specs),
            observations=new_obs,
            objective_names=list(snapshot.objective_names),
            objective_directions=list(snapshot.objective_directions),
            constraints=list(snapshot.constraints),
            current_iteration=snapshot.current_iteration,
            metadata=dict(snapshot.metadata),
        )

        corrected_stats = self._kpi_stats(new_snapshot.observations, new_snapshot.objective_names)

        record = ConfounderCorrectionRecord(
            column_name=spec.column_name,
            policy=ConfounderPolicy.HIGH_RISK_FLAG,
            n_affected_rows=n_affected,
            correction_details={
                "threshold_low": spec.threshold_low,
                "threshold_high": spec.threshold_high,
            },
            original_kpi_stats=original_stats,
            corrected_kpi_stats=corrected_stats,
        )
        return new_snapshot, record

    # -- EXCLUDE ------------------------------------------------------------

    def _apply_exclude(
        self,
        snapshot: CampaignSnapshot,
        spec: ConfounderSpec,
    ) -> tuple[CampaignSnapshot, ConfounderCorrectionRecord]:
        """Remove observations whose confounder value exceeds thresholds."""
        original_stats = self._kpi_stats(snapshot.observations, snapshot.objective_names)

        kept: list[Observation] = []
        n_removed = 0

        for obs in snapshot.observations:
            raw = obs.metadata.get(spec.column_name)
            exclude = False

            if raw is not None:
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    val = None

                if val is not None:
                    if spec.threshold_low is not None and val < spec.threshold_low:
                        exclude = True
                    if spec.threshold_high is not None and val > spec.threshold_high:
                        exclude = True

            if exclude:
                n_removed += 1
            else:
                # Deep copy the kept observation.
                kept.append(Observation(
                    iteration=obs.iteration,
                    parameters=dict(obs.parameters),
                    kpi_values=dict(obs.kpi_values),
                    qc_passed=obs.qc_passed,
                    is_failure=obs.is_failure,
                    failure_reason=obs.failure_reason,
                    timestamp=obs.timestamp,
                    metadata=dict(obs.metadata),
                ))

        new_snapshot = CampaignSnapshot(
            campaign_id=snapshot.campaign_id,
            parameter_specs=list(snapshot.parameter_specs),
            observations=kept,
            objective_names=list(snapshot.objective_names),
            objective_directions=list(snapshot.objective_directions),
            constraints=list(snapshot.constraints),
            current_iteration=snapshot.current_iteration,
            metadata=dict(snapshot.metadata),
        )

        corrected_stats = self._kpi_stats(new_snapshot.observations, new_snapshot.objective_names)

        record = ConfounderCorrectionRecord(
            column_name=spec.column_name,
            policy=ConfounderPolicy.EXCLUDE,
            n_affected_rows=n_removed,
            correction_details={
                "threshold_low": spec.threshold_low,
                "threshold_high": spec.threshold_high,
                "remaining_observations": len(kept),
            },
            original_kpi_stats=original_stats,
            corrected_kpi_stats=corrected_stats,
        )
        return new_snapshot, record

    # -- Static helpers -----------------------------------------------------

    @staticmethod
    def _linear_regression_residuals(x: list[float], y: list[float]) -> list[float]:
        """Compute OLS residuals for ``y ~ x``.

        Returns a list of residuals ``y_i - (slope * x_i + intercept)``.
        If *x* has zero variance the residuals equal ``y_i - mean(y)``.
        """
        n = len(x)
        if n == 0:
            return []
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        x_var = sum((xi - x_mean) ** 2 for xi in x) / n
        if x_var < 1e-15:
            # Constant confounder -- residuals are deviations from the mean.
            return [yi - y_mean for yi in y]

        cov = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / n
        slope = cov / x_var
        intercept = y_mean - slope * x_mean

        return [y[i] - (slope * x[i] + intercept) for i in range(n)]

    @staticmethod
    def _kpi_stats(
        observations: list[Observation],
        objective_names: list[str],
    ) -> dict[str, float]:
        """Compute mean and std for each KPI across observations."""
        stats: dict[str, float] = {}
        for name in objective_names:
            values = [
                obs.kpi_values[name]
                for obs in observations
                if name in obs.kpi_values
            ]
            if not values:
                stats[f"{name}_mean"] = 0.0
                stats[f"{name}_std"] = 0.0
                continue
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
            stats[f"{name}_mean"] = round(mean, 8)
            stats[f"{name}_std"] = round(std, 8)
        return stats

    # -- Snapshot cloning ---------------------------------------------------

    @staticmethod
    def _clone_snapshot(snapshot: CampaignSnapshot) -> CampaignSnapshot:
        """Return a deep copy of *snapshot* via serialization round-trip."""
        return CampaignSnapshot.from_dict(snapshot.to_dict())

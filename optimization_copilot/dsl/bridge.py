"""Bridge from DSL types to existing core models.

Translates the declarative OptimizationSpec representation into the runtime
data structures used by the optimization engine (CampaignSnapshot,
ProblemFingerprint, ParameterSpec, etc.).
"""

from __future__ import annotations

from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    CostProfile,
    DataScale,
    Dynamics,
    FailureInformativeness,
    FeasibleRegion,
    NoiseRegime,
    ObjectiveForm,
    Observation,
    ParameterSpec,
    ProblemFingerprint,
    RiskPosture,
    VariableType,
)
from optimization_copilot.dsl.spec import (
    Direction,
    OptimizationSpec,
    ParamType,
    ParameterDef,
    RiskPreference,
)


# ── Type mappings ─────────────────────────────────────

_PARAM_TYPE_MAP: dict[ParamType, VariableType] = {
    ParamType.CONTINUOUS: VariableType.CONTINUOUS,
    ParamType.DISCRETE: VariableType.DISCRETE,
    ParamType.CATEGORICAL: VariableType.CATEGORICAL,
}

_RISK_MAP: dict[RiskPreference, RiskPosture] = {
    RiskPreference.CONSERVATIVE: RiskPosture.CONSERVATIVE,
    RiskPreference.MODERATE: RiskPosture.MODERATE,
    RiskPreference.AGGRESSIVE: RiskPosture.AGGRESSIVE,
}

_DIRECTION_MAP: dict[Direction, str] = {
    Direction.MINIMIZE: "minimize",
    Direction.MAXIMIZE: "maximize",
}


# ── Bridge class ──────────────────────────────────────


class SpecBridge:
    """Translates DSL ``OptimizationSpec`` instances into core model objects.

    All methods are static/class-level -- no mutable state is held.
    """

    # ── Parameter conversion ──────────────────────────

    @staticmethod
    def to_parameter_specs(spec: OptimizationSpec) -> list[ParameterSpec]:
        """Convert active (non-frozen) DSL parameters to core ParameterSpec list.

        Frozen parameters are excluded because they are not subject to
        optimization.
        """
        result: list[ParameterSpec] = []
        for p in spec.parameters:
            if p.frozen:
                continue
            result.append(
                ParameterSpec(
                    name=p.name,
                    type=_PARAM_TYPE_MAP[p.type],
                    lower=p.lower,
                    upper=p.upper,
                    categories=list(p.categories) if p.categories else None,
                )
            )
        return result

    # ── Campaign snapshot ─────────────────────────────

    @staticmethod
    def to_campaign_snapshot(
        spec: OptimizationSpec,
        observations: list[Observation] | None = None,
    ) -> CampaignSnapshot:
        """Create a CampaignSnapshot from an OptimizationSpec.

        Parameters
        ----------
        spec:
            The DSL specification.
        observations:
            Optional list of observations collected so far.  Defaults to
            an empty list (cold-start campaign).
        """
        if observations is None:
            observations = []

        parameter_specs = SpecBridge.to_parameter_specs(spec)

        objective_names = [obj.name for obj in spec.objectives]
        objective_directions = [
            _DIRECTION_MAP[obj.direction] for obj in spec.objectives
        ]

        # Build constraints from objective bounds.
        constraints: list[dict[str, Any]] = []
        for obj in spec.objectives:
            if obj.constraint_lower is not None:
                constraints.append(
                    {
                        "type": "lower_bound",
                        "target": obj.name,
                        "value": obj.constraint_lower,
                    }
                )
            if obj.constraint_upper is not None:
                constraints.append(
                    {
                        "type": "upper_bound",
                        "target": obj.name,
                        "value": obj.constraint_upper,
                    }
                )

        current_iteration = (
            max((o.iteration for o in observations), default=0)
            if observations
            else 0
        )

        metadata: dict[str, Any] = {
            "dsl_campaign_id": spec.campaign_id,
            "seed": spec.seed,
            "risk_preference": spec.risk_preference.value,
            "batch_size": spec.parallel.batch_size,
        }
        if spec.name:
            metadata["name"] = spec.name
        if spec.description:
            metadata["description"] = spec.description
        metadata.update(spec.metadata)

        return CampaignSnapshot(
            campaign_id=spec.campaign_id,
            parameter_specs=parameter_specs,
            observations=observations,
            objective_names=objective_names,
            objective_directions=objective_directions,
            constraints=constraints,
            current_iteration=current_iteration,
            metadata=metadata,
        )

    # ── Problem fingerprint ───────────────────────────

    @staticmethod
    def to_initial_fingerprint(spec: OptimizationSpec) -> ProblemFingerprint:
        """Infer an initial ProblemFingerprint from the spec alone.

        Observation-dependent dimensions (noise_regime, data_scale,
        feasible_region, etc.) receive conservative defaults because no
        experimental data is available yet.
        """
        # Determine aggregate variable type.
        active_params = [p for p in spec.parameters if not p.frozen]
        if not active_params:
            variable_types = VariableType.CONTINUOUS
        else:
            types_seen = {p.type for p in active_params}
            if len(types_seen) > 1:
                variable_types = VariableType.MIXED
            else:
                sole_type = next(iter(types_seen))
                variable_types = _PARAM_TYPE_MAP[sole_type]

        # Determine objective form.
        has_constraints = any(
            obj.constraint_lower is not None or obj.constraint_upper is not None
            for obj in spec.objectives
        )
        if has_constraints:
            objective_form = ObjectiveForm.CONSTRAINED
        elif len(spec.objectives) > 1:
            objective_form = ObjectiveForm.MULTI_OBJECTIVE
        else:
            objective_form = ObjectiveForm.SINGLE

        # Conservative defaults for observation-dependent dimensions.
        return ProblemFingerprint(
            variable_types=variable_types,
            objective_form=objective_form,
            noise_regime=NoiseRegime.MEDIUM,
            cost_profile=CostProfile.UNIFORM,
            failure_informativeness=FailureInformativeness.WEAK,
            data_scale=DataScale.TINY,
            dynamics=Dynamics.STATIC,
            feasible_region=FeasibleRegion.WIDE,
        )

    # ── Risk posture ──────────────────────────────────

    @staticmethod
    def to_risk_posture(spec: OptimizationSpec) -> RiskPosture:
        """Map DSL RiskPreference to core RiskPosture."""
        return _RISK_MAP[spec.risk_preference]

    # ── Active parameters ─────────────────────────────

    @staticmethod
    def active_parameters(
        spec: OptimizationSpec,
        current_values: dict[str, Any] | None = None,
    ) -> list[ParameterDef]:
        """Return non-frozen parameters, evaluating conditions if values given.

        Parameters
        ----------
        spec:
            The DSL specification.
        current_values:
            Optional mapping of parameter names to their current values.
            When provided, conditional parameters whose parent value does
            not match are excluded.
        """
        result: list[ParameterDef] = []
        for p in spec.parameters:
            if p.frozen:
                continue
            if p.condition is not None and current_values is not None:
                parent_val = current_values.get(p.condition.parent_name)
                if parent_val != p.condition.parent_value:
                    continue
            result.append(p)
        return result

    # ── Frozen value injection ────────────────────────

    @staticmethod
    def apply_frozen_values(
        spec: OptimizationSpec,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Inject frozen parameter values into each candidate dict.

        Returns a new list of dicts with frozen values merged in.  Existing
        keys in the candidate dicts are **not** overwritten if they happen
        to share a name with a frozen parameter.
        """
        frozen_pairs: dict[str, Any] = {
            p.name: p.frozen_value
            for p in spec.parameters
            if p.frozen
        }
        if not frozen_pairs:
            return candidates

        result: list[dict[str, Any]] = []
        for candidate in candidates:
            merged = dict(frozen_pairs)
            merged.update(candidate)
            result.append(merged)
        return result

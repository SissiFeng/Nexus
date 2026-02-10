"""Spec validation with human-readable error messages.

Validates OptimizationSpec instances against structural and semantic
rules, producing clear error messages suitable for display to end users.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from optimization_copilot.dsl.spec import OptimizationSpec, ParamType


class DSLValidationError(Exception):
    """Raised when an OptimizationSpec fails validation.

    Attributes:
        errors: List of human-readable validation error strings.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        formatted = "\n".join(f"  - {e}" for e in errors)
        super().__init__(f"Optimization spec validation failed with {len(errors)} error(s):\n{formatted}")


@dataclass
class ValidationResult:
    """Outcome of spec validation.

    Attributes:
        valid: True if no errors were found.
        errors: List of error strings (empty when valid).
        warnings: List of warning strings (informational, non-blocking).
    """
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SpecValidator:
    """Validates OptimizationSpec instances against structural and semantic rules."""

    def validate(self, spec: OptimizationSpec) -> ValidationResult:
        """Validate a spec and return a result. Does NOT raise on errors."""
        errors: list[str] = []
        warnings: list[str] = []

        self._check_campaign_id(spec, errors)
        self._check_parameters_present(spec, errors)
        self._check_objectives_present(spec, errors)
        self._check_parameter_names_unique(spec, errors)
        self._check_objective_names_unique(spec, errors)
        self._check_parameter_bounds(spec, errors)
        self._check_categorical_categories(spec, errors)
        self._check_conditional_params(spec, errors)
        self._check_frozen_values(spec, errors)
        self._check_budget_values(spec, errors)

        self._check_budget_warnings(spec, warnings)
        self._check_all_frozen_warning(spec, warnings)
        self._check_single_param_warning(spec, warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_or_raise(self, spec: OptimizationSpec) -> None:
        """Validate a spec and raise DSLValidationError if invalid."""
        result = self.validate(spec)
        if not result.valid:
            raise DSLValidationError(result.errors)

    # ── Error checks ──────────────────────────────────

    @staticmethod
    def _check_campaign_id(spec: OptimizationSpec, errors: list[str]) -> None:
        if not spec.campaign_id or not spec.campaign_id.strip():
            errors.append("campaign_id must be a non-empty string.")

    @staticmethod
    def _check_parameters_present(spec: OptimizationSpec, errors: list[str]) -> None:
        if not spec.parameters:
            errors.append("At least one parameter is required.")

    @staticmethod
    def _check_objectives_present(spec: OptimizationSpec, errors: list[str]) -> None:
        if not spec.objectives:
            errors.append("At least one objective is required.")

    @staticmethod
    def _check_parameter_names_unique(spec: OptimizationSpec, errors: list[str]) -> None:
        seen: set[str] = set()
        for p in spec.parameters:
            if p.name in seen:
                errors.append(f"Duplicate parameter name: '{p.name}'.")
            seen.add(p.name)

    @staticmethod
    def _check_objective_names_unique(spec: OptimizationSpec, errors: list[str]) -> None:
        seen: set[str] = set()
        for o in spec.objectives:
            if o.name in seen:
                errors.append(f"Duplicate objective name: '{o.name}'.")
            seen.add(o.name)

    @staticmethod
    def _check_parameter_bounds(spec: OptimizationSpec, errors: list[str]) -> None:
        for p in spec.parameters:
            if p.type in (ParamType.CONTINUOUS, ParamType.DISCRETE):
                if p.lower is None or p.upper is None:
                    errors.append(
                        f"Parameter '{p.name}' ({p.type.value}): "
                        f"both lower and upper bounds are required."
                    )
                elif p.lower >= p.upper:
                    errors.append(
                        f"Parameter '{p.name}' ({p.type.value}): "
                        f"lower bound ({p.lower}) must be strictly less than "
                        f"upper bound ({p.upper})."
                    )

    @staticmethod
    def _check_categorical_categories(spec: OptimizationSpec, errors: list[str]) -> None:
        for p in spec.parameters:
            if p.type == ParamType.CATEGORICAL:
                if not p.categories:
                    errors.append(
                        f"Parameter '{p.name}' (categorical): "
                        f"categories list must be non-empty."
                    )

    @staticmethod
    def _check_conditional_params(spec: OptimizationSpec, errors: list[str]) -> None:
        param_map = {p.name: p for p in spec.parameters}
        for p in spec.parameters:
            if p.condition is None:
                continue
            parent_name = p.condition.parent_name
            if parent_name not in param_map:
                errors.append(
                    f"Parameter '{p.name}': conditional parent '{parent_name}' "
                    f"does not reference an existing parameter."
                )
                continue
            parent = param_map[parent_name]
            parent_value = p.condition.parent_value
            if parent.type == ParamType.CATEGORICAL:
                if parent.categories and parent_value not in parent.categories:
                    errors.append(
                        f"Parameter '{p.name}': conditional parent_value "
                        f"'{parent_value}' is not in parent '{parent_name}' "
                        f"categories {parent.categories}."
                    )
            elif parent.type in (ParamType.CONTINUOUS, ParamType.DISCRETE):
                if parent.lower is not None and parent.upper is not None:
                    try:
                        v = float(parent_value)
                    except (TypeError, ValueError):
                        errors.append(
                            f"Parameter '{p.name}': conditional parent_value "
                            f"'{parent_value}' is not numeric, but parent "
                            f"'{parent_name}' is {parent.type.value}."
                        )
                    else:
                        if v < parent.lower or v > parent.upper:
                            errors.append(
                                f"Parameter '{p.name}': conditional parent_value "
                                f"{v} is outside parent '{parent_name}' bounds "
                                f"[{parent.lower}, {parent.upper}]."
                            )

    @staticmethod
    def _check_frozen_values(spec: OptimizationSpec, errors: list[str]) -> None:
        for p in spec.parameters:
            if not p.frozen or p.frozen_value is None:
                continue
            if p.type == ParamType.CATEGORICAL:
                if p.categories and p.frozen_value not in p.categories:
                    errors.append(
                        f"Parameter '{p.name}': frozen_value '{p.frozen_value}' "
                        f"is not in categories {p.categories}."
                    )
            elif p.type in (ParamType.CONTINUOUS, ParamType.DISCRETE):
                if p.lower is not None and p.upper is not None:
                    try:
                        v = float(p.frozen_value)
                    except (TypeError, ValueError):
                        errors.append(
                            f"Parameter '{p.name}': frozen_value "
                            f"'{p.frozen_value}' is not numeric."
                        )
                    else:
                        if v < p.lower or v > p.upper:
                            errors.append(
                                f"Parameter '{p.name}': frozen_value {v} "
                                f"is outside bounds [{p.lower}, {p.upper}]."
                            )

    @staticmethod
    def _check_budget_values(spec: OptimizationSpec, errors: list[str]) -> None:
        if spec.budget.max_samples is not None and spec.budget.max_samples <= 0:
            errors.append(
                f"budget.max_samples must be positive, got {spec.budget.max_samples}."
            )
        if spec.budget.max_time_seconds is not None and spec.budget.max_time_seconds <= 0:
            errors.append(
                f"budget.max_time_seconds must be positive, got {spec.budget.max_time_seconds}."
            )
        if spec.budget.max_cost is not None and spec.budget.max_cost <= 0:
            errors.append(
                f"budget.max_cost must be positive, got {spec.budget.max_cost}."
            )
        if spec.budget.max_iterations is not None and spec.budget.max_iterations <= 0:
            errors.append(
                f"budget.max_iterations must be positive, got {spec.budget.max_iterations}."
            )

    # ── Warning checks ────────────────────────────────

    @staticmethod
    def _check_budget_warnings(spec: OptimizationSpec, warnings: list[str]) -> None:
        b = spec.budget
        if (
            b.max_samples is None
            and b.max_time_seconds is None
            and b.max_cost is None
            and b.max_iterations is None
        ):
            warnings.append(
                "No budget constraints set. The campaign may run indefinitely."
            )

    @staticmethod
    def _check_all_frozen_warning(spec: OptimizationSpec, warnings: list[str]) -> None:
        if spec.parameters and all(p.frozen for p in spec.parameters):
            warnings.append(
                "All parameters are frozen. No optimization will occur."
            )

    @staticmethod
    def _check_single_param_warning(spec: OptimizationSpec, warnings: list[str]) -> None:
        active_params = [p for p in spec.parameters if not p.frozen]
        if len(active_params) == 1:
            warnings.append(
                f"Only one active parameter ('{active_params[0].name}'). "
                f"Consider whether single-parameter optimization is intended."
            )

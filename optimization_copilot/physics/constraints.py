"""Physics-based constraint models for optimization.

Encodes physical laws (conservation, monotonicity, bounds) as
optimization constraints with feasibility checking and projection.
Pure Python stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConservationLaw:
    """A conservation constraint requiring a set of variables to sum to a target.

    Parameters
    ----------
    name : str
        Human-readable name of the conservation law.
    variables : list[str]
        Names of the variables that must sum to *target_sum*.
    target_sum : float
        The required sum.
    tolerance : float
        Absolute tolerance for feasibility checking (default 1e-6).
    """

    name: str
    variables: list[str]
    target_sum: float
    tolerance: float = 1e-6


@dataclass
class MonotonicityConstraint:
    """A constraint requiring a variable to be monotonic.

    Parameters
    ----------
    variable : str
        Name of the variable to check monotonicity for.
    direction : str
        ``"increasing"`` or ``"decreasing"``.
    condition : dict | None
        Optional range condition restricting when the constraint applies,
        e.g. ``{"min": 0.0, "max": 100.0}`` to apply only when the
        variable is in that range.
    """

    variable: str
    direction: str  # "increasing" or "decreasing"
    condition: dict[str, Any] | None = None


@dataclass
class PhysicsBound:
    """A bound on a single variable with an explanatory reason.

    Parameters
    ----------
    variable : str
        Name of the bounded variable.
    lower : float | None
        Lower bound (None if unbounded below).
    upper : float | None
        Upper bound (None if unbounded above).
    reason : str
        Human-readable reason for this bound.
    """

    variable: str
    lower: float | None = None
    upper: float | None = None
    reason: str = ""


class PhysicsConstraintModel:
    """Encode physical laws as optimization constraints.

    Manages three types of physics constraints:
    1. Conservation laws (sum of variables equals a target).
    2. Monotonicity constraints (a variable must be monotonic).
    3. Variable bounds (with physical justification).

    Provides feasibility checking, monotonicity validation,
    and projection to the nearest feasible point.
    """

    def __init__(self) -> None:
        self.conservation_laws: list[ConservationLaw] = []
        self.monotonicity: list[MonotonicityConstraint] = []
        self.bounds: list[PhysicsBound] = []

    def add_conservation_law(
        self,
        name: str,
        variables: list[str],
        target_sum: float,
        tolerance: float = 1e-6,
    ) -> None:
        """Add a conservation law constraint.

        Parameters
        ----------
        name : str
            Human-readable name.
        variables : list[str]
            Variable names that must sum to *target_sum*.
        target_sum : float
            Required sum.
        tolerance : float
            Absolute tolerance (default 1e-6).
        """
        self.conservation_laws.append(
            ConservationLaw(
                name=name,
                variables=variables,
                target_sum=target_sum,
                tolerance=tolerance,
            )
        )

    def add_monotonicity(
        self,
        variable: str,
        direction: str,
        condition: dict[str, Any] | None = None,
    ) -> None:
        """Add a monotonicity constraint.

        Parameters
        ----------
        variable : str
            Variable name.
        direction : str
            ``"increasing"`` or ``"decreasing"``.
        condition : dict | None
            Optional range condition.

        Raises
        ------
        ValueError
            If *direction* is not ``"increasing"`` or ``"decreasing"``.
        """
        if direction not in ("increasing", "decreasing"):
            raise ValueError(
                f"direction must be 'increasing' or 'decreasing', got {direction!r}"
            )
        self.monotonicity.append(
            MonotonicityConstraint(
                variable=variable,
                direction=direction,
                condition=condition,
            )
        )

    def add_bound(
        self,
        variable: str,
        lower: float | None = None,
        upper: float | None = None,
        reason: str = "",
    ) -> None:
        """Add a physics-based bound on a variable.

        Parameters
        ----------
        variable : str
            Variable name.
        lower : float | None
            Lower bound (None if unbounded below).
        upper : float | None
            Upper bound (None if unbounded above).
        reason : str
            Justification for the bound.
        """
        self.bounds.append(
            PhysicsBound(
                variable=variable,
                lower=lower,
                upper=upper,
                reason=reason,
            )
        )

    def check_feasibility(
        self, point: dict[str, float]
    ) -> tuple[bool, list[str]]:
        """Check if a point satisfies all bound and conservation constraints.

        Parameters
        ----------
        point : dict[str, float]
            Variable name to value mapping.

        Returns
        -------
        tuple[bool, list[str]]
            ``(feasible, violations)`` where *violations* is a list of
            human-readable descriptions of violated constraints.
        """
        violations: list[str] = []

        # Check bounds
        for b in self.bounds:
            if b.variable not in point:
                continue
            val = point[b.variable]
            if b.lower is not None and val < b.lower:
                violations.append(
                    f"Bound violated: {b.variable}={val} < lower={b.lower}"
                    + (f" ({b.reason})" if b.reason else "")
                )
            if b.upper is not None and val > b.upper:
                violations.append(
                    f"Bound violated: {b.variable}={val} > upper={b.upper}"
                    + (f" ({b.reason})" if b.reason else "")
                )

        # Check conservation laws
        for law in self.conservation_laws:
            missing = [v for v in law.variables if v not in point]
            if missing:
                violations.append(
                    f"Conservation '{law.name}': missing variables {missing}"
                )
                continue
            current_sum = sum(point[v] for v in law.variables)
            if abs(current_sum - law.target_sum) > law.tolerance:
                violations.append(
                    f"Conservation '{law.name}' violated: "
                    f"sum({law.variables})={current_sum} != {law.target_sum} "
                    f"(tol={law.tolerance})"
                )

        feasible = len(violations) == 0
        return feasible, violations

    def check_monotonicity(
        self,
        points: list[dict[str, float]],
        variable: str,
    ) -> tuple[bool, list[str]]:
        """Check if a sequence of points satisfies monotonicity constraints.

        Parameters
        ----------
        points : list[dict[str, float]]
            Ordered sequence of observation dicts.
        variable : str
            The variable to check monotonicity for.

        Returns
        -------
        tuple[bool, list[str]]
            ``(monotonic, violations)`` where *violations* describes
            each pair that violates the constraint.
        """
        violations: list[str] = []

        # Find applicable monotonicity constraints for this variable
        applicable = [
            mc for mc in self.monotonicity if mc.variable == variable
        ]
        if not applicable:
            return True, []

        for mc in applicable:
            values: list[float] = []
            indices: list[int] = []
            for idx, pt in enumerate(points):
                if variable not in pt:
                    continue
                val = pt[variable]

                # Check condition if present
                if mc.condition is not None:
                    cond_min = mc.condition.get("min", float("-inf"))
                    cond_max = mc.condition.get("max", float("inf"))
                    if val < cond_min or val > cond_max:
                        continue

                values.append(val)
                indices.append(idx)

            # Check monotonicity on filtered values
            for i in range(len(values) - 1):
                if mc.direction == "increasing" and values[i + 1] < values[i]:
                    violations.append(
                        f"Monotonicity '{mc.variable}' ({mc.direction}) "
                        f"violated: point[{indices[i]}]={values[i]} > "
                        f"point[{indices[i + 1]}]={values[i + 1]}"
                    )
                elif mc.direction == "decreasing" and values[i + 1] > values[i]:
                    violations.append(
                        f"Monotonicity '{mc.variable}' ({mc.direction}) "
                        f"violated: point[{indices[i]}]={values[i]} < "
                        f"point[{indices[i + 1]}]={values[i + 1]}"
                    )

        monotonic = len(violations) == 0
        return monotonic, violations

    def project_to_feasible(self, point: dict[str, float]) -> dict[str, float]:
        """Project an infeasible point to the nearest feasible point.

        Applies bound clamping first, then conservation law projection
        (uniform redistribution of excess/deficit).

        Parameters
        ----------
        point : dict[str, float]
            Variable name to value mapping (may be infeasible).

        Returns
        -------
        dict[str, float]
            A new dict projected to satisfy bounds and conservation laws.
        """
        projected = dict(point)

        # Step 1: Apply bounds
        for b in self.bounds:
            if b.variable not in projected:
                continue
            val = projected[b.variable]
            if b.lower is not None and val < b.lower:
                projected[b.variable] = b.lower
            if b.upper is not None and val > b.upper:
                projected[b.variable] = b.upper

        # Step 2: Project onto conservation law hyperplanes
        for law in self.conservation_laws:
            present_vars = [v for v in law.variables if v in projected]
            if not present_vars:
                continue
            current_sum = sum(projected[v] for v in present_vars)
            deficit = law.target_sum - current_sum
            if abs(deficit) > law.tolerance:
                # Distribute deficit uniformly among present variables
                adjustment = deficit / len(present_vars)
                for v in present_vars:
                    projected[v] += adjustment

        return projected

    def to_dict(self) -> dict[str, Any]:
        """Serialize the constraint model to a plain dict.

        Returns
        -------
        dict[str, Any]
            Serialized representation.
        """
        return {
            "conservation_laws": [
                {
                    "name": law.name,
                    "variables": law.variables,
                    "target_sum": law.target_sum,
                    "tolerance": law.tolerance,
                }
                for law in self.conservation_laws
            ],
            "monotonicity": [
                {
                    "variable": mc.variable,
                    "direction": mc.direction,
                    "condition": mc.condition,
                }
                for mc in self.monotonicity
            ],
            "bounds": [
                {
                    "variable": b.variable,
                    "lower": b.lower,
                    "upper": b.upper,
                    "reason": b.reason,
                }
                for b in self.bounds
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PhysicsConstraintModel:
        """Deserialize a constraint model from a plain dict.

        Parameters
        ----------
        d : dict[str, Any]
            Serialized representation (as produced by :meth:`to_dict`).

        Returns
        -------
        PhysicsConstraintModel
            Reconstructed constraint model.
        """
        model = cls()
        for law_d in d.get("conservation_laws", []):
            model.add_conservation_law(
                name=law_d["name"],
                variables=law_d["variables"],
                target_sum=law_d["target_sum"],
                tolerance=law_d.get("tolerance", 1e-6),
            )
        for mc_d in d.get("monotonicity", []):
            model.add_monotonicity(
                variable=mc_d["variable"],
                direction=mc_d["direction"],
                condition=mc_d.get("condition"),
            )
        for b_d in d.get("bounds", []):
            model.add_bound(
                variable=b_d["variable"],
                lower=b_d.get("lower"),
                upper=b_d.get("upper"),
                reason=b_d.get("reason", ""),
            )
        return model

    def __repr__(self) -> str:
        return (
            f"PhysicsConstraintModel("
            f"conservation_laws={len(self.conservation_laws)}, "
            f"monotonicity={len(self.monotonicity)}, "
            f"bounds={len(self.bounds)})"
        )

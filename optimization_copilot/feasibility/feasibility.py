"""Feasibility and failure learning for optimization campaigns."""

from __future__ import annotations

from dataclasses import dataclass, field

from optimization_copilot.core.models import CampaignSnapshot, Observation, ParameterSpec


@dataclass
class FeasibilityMap:
    """Representation of the feasible/infeasible region."""
    safe_bounds: dict[str, tuple[float, float]]  # param -> (safe_low, safe_high)
    infeasible_zones: list[dict[str, tuple[float, float]]]
    failure_density: dict[str, list[float]]  # param -> failure locations
    feasibility_score: float  # 0.0 = all infeasible, 1.0 = all feasible
    constraint_tightness: dict[str, float]  # constraint_name -> tightness [0,1]


class FeasibilityLearner:
    """Learn feasible regions from failure data."""

    def learn(self, snapshot: CampaignSnapshot) -> FeasibilityMap:
        specs = snapshot.parameter_specs
        obs = snapshot.observations

        safe_bounds = self._compute_safe_bounds(specs, obs)
        infeasible_zones = self._identify_infeasible_zones(specs, obs)
        failure_density = self._compute_failure_density(specs, obs)
        feasibility_score = 1.0 - snapshot.failure_rate
        constraint_tightness = self._assess_constraint_tightness(snapshot)

        return FeasibilityMap(
            safe_bounds=safe_bounds,
            infeasible_zones=infeasible_zones,
            failure_density=failure_density,
            feasibility_score=feasibility_score,
            constraint_tightness=constraint_tightness,
        )

    @staticmethod
    def _compute_safe_bounds(
        specs: list[ParameterSpec], obs: list[Observation]
    ) -> dict[str, tuple[float, float]]:
        """Compute parameter bounds where successes have been observed."""
        safe_bounds: dict[str, tuple[float, float]] = {}
        successful = [o for o in obs if not o.is_failure]
        if not successful:
            for s in specs:
                lo = s.lower if s.lower is not None else 0.0
                hi = s.upper if s.upper is not None else 1.0
                safe_bounds[s.name] = (lo, hi)
            return safe_bounds

        for spec in specs:
            if spec.lower is None or spec.upper is None:
                safe_bounds[spec.name] = (0.0, 1.0)
                continue
            vals = [o.parameters.get(spec.name, 0.0) for o in successful]
            safe_bounds[spec.name] = (min(vals), max(vals))

        return safe_bounds

    @staticmethod
    def _identify_infeasible_zones(
        specs: list[ParameterSpec], obs: list[Observation]
    ) -> list[dict[str, tuple[float, float]]]:
        """Identify regions where failures cluster."""
        failures = [o for o in obs if o.is_failure]
        if len(failures) < 2:
            return []

        # Simple: cluster failures by checking if they share similar param values
        zones: list[dict[str, tuple[float, float]]] = []
        for spec in specs:
            if spec.lower is None or spec.upper is None:
                continue
            fail_vals = [o.parameters.get(spec.name, 0.0) for o in failures]
            param_range = spec.upper - spec.lower
            if param_range == 0:
                continue

            # If failures are concentrated in a sub-region
            f_min, f_max = min(fail_vals), max(fail_vals)
            spread = (f_max - f_min) / param_range
            if spread < 0.5 and len(fail_vals) >= 2:
                margin = param_range * 0.05
                zones.append({
                    spec.name: (f_min - margin, f_max + margin)
                })

        return zones

    @staticmethod
    def _compute_failure_density(
        specs: list[ParameterSpec], obs: list[Observation]
    ) -> dict[str, list[float]]:
        failures = [o for o in obs if o.is_failure]
        density: dict[str, list[float]] = {}
        for spec in specs:
            density[spec.name] = [
                o.parameters.get(spec.name, 0.0) for o in failures
            ]
        return density

    @staticmethod
    def _assess_constraint_tightness(
        snapshot: CampaignSnapshot,
    ) -> dict[str, float]:
        """Assess how tight each constraint is (0=loose, 1=very tight)."""
        tightness: dict[str, float] = {}
        for i, constraint in enumerate(snapshot.constraints):
            name = constraint.get("name", f"constraint_{i}")
            # Simplified: estimate based on how many observations are near boundary
            tightness[name] = 0.5  # Default moderate tightness
        return tightness

    def is_feasible(
        self, parameters: dict, feasibility_map: FeasibilityMap
    ) -> bool:
        """Check if a parameter set is likely feasible."""
        for param, (lo, hi) in feasibility_map.safe_bounds.items():
            val = parameters.get(param)
            if val is not None and not (lo <= val <= hi):
                return False

        for zone in feasibility_map.infeasible_zones:
            in_zone = True
            for param, (lo, hi) in zone.items():
                val = parameters.get(param)
                if val is None or not (lo <= val <= hi):
                    in_zone = False
                    break
            if in_zone:
                return False

        return True

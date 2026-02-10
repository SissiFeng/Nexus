"""Synthetic benchmark generation covering all ProblemFingerprint dimensions.

Generates reproducible GoldenScenario instances from configurable landscape
functions, noise levels, failure modes, constraints, and multi-objective
setups.  Every run is deterministic given the same seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    Phase,
    RiskPosture,
    VariableType,
)
from optimization_copilot.validation.scenarios import (
    GoldenScenario,
    ScenarioExpectation,
)


# ── Landscape type enum ───────────────────────────────────


class LandscapeType(str, Enum):
    """Synthetic objective-function landscapes."""

    SPHERE = "sphere"
    ROSENBROCK = "rosenbrock"
    ACKLEY = "ackley"
    RASTRIGIN = "rastrigin"


# ── Module-local objective functions ──────────────────────


def _sphere(x: list[float], shift: float = 0.0) -> float:
    """Sum of squared deviations from *shift*."""
    return sum((xi - shift) ** 2 for xi in x)


def _rosenbrock(x: list[float]) -> float:
    """Classic Rosenbrock banana function."""
    total = 0.0
    for i in range(len(x) - 1):
        total += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return total


def _ackley(x: list[float]) -> float:
    """Ackley function with many local minima."""
    n = len(x)
    sum_sq = sum(xi ** 2 for xi in x) / n
    sum_cos = sum(math.cos(2.0 * math.pi * xi) for xi in x) / n
    return (
        -20.0 * math.exp(-0.2 * math.sqrt(sum_sq))
        - math.exp(sum_cos)
        + 20.0
        + math.e
    )


def _rastrigin(x: list[float]) -> float:
    """Rastrigin function -- highly multimodal."""
    n = len(x)
    return 10.0 * n + sum(
        xi ** 2 - 10.0 * math.cos(2.0 * math.pi * xi) for xi in x
    )


_LANDSCAPE_FN: dict[LandscapeType, Any] = {
    LandscapeType.SPHERE: _sphere,
    LandscapeType.ROSENBROCK: _rosenbrock,
    LandscapeType.ACKLEY: _ackley,
    LandscapeType.RASTRIGIN: _rastrigin,
}


# ── Helper utilities ──────────────────────────────────────


def _check_failure_zone(
    parameters: dict[str, Any],
    failure_zones: list[dict[str, tuple[float, float]]],
) -> bool:
    """Return ``True`` if *parameters* fall inside any failure zone.

    Each zone maps parameter names to ``(low, high)`` bounds.  A zone is
    triggered when **all** listed parameters sit within their respective
    bounds.
    """
    for zone in failure_zones:
        all_in = True
        for param_name, (low, high) in zone.items():
            val = parameters.get(param_name)
            if val is None or not (low <= float(val) <= high):
                all_in = False
                break
        if all_in:
            return True
    return False


def _check_constraints(
    parameters: dict[str, Any],
    constraints: list[dict[str, Any]],
) -> bool:
    """Return ``True`` when **all** constraints are satisfied."""
    for con in constraints:
        ctype = con["type"]
        param_names = con["parameters"]
        bound = con["bound"]
        if ctype == "sum_bound":
            total = sum(float(parameters.get(p, 0.0)) for p in param_names)
            if total > bound:
                return False
        elif ctype == "boundary":
            for p in param_names:
                val = float(parameters.get(p, 0.0))
                if not (0.0 <= val <= bound):
                    return False
    return True


def _apply_drift(value: float, drift_rate: float, iteration: int) -> float:
    """Shift the objective value linearly over iterations."""
    return value + drift_rate * iteration


# ── SyntheticObjective ────────────────────────────────────


@dataclass
class SyntheticObjective:
    """A configurable synthetic objective function.

    Wraps one of the standard landscape functions and layers on noise,
    failure modes, constraints, multi-objective conflicts, non-stationary
    drift, and categorical effects.
    """

    name: str
    landscape_type: LandscapeType
    n_dimensions: int
    noise_sigma: float = 0.0
    failure_rate: float = 0.0
    failure_zones: list[dict[str, tuple[float, float]]] = field(
        default_factory=list,
    )
    constraints: list[dict[str, Any]] = field(default_factory=list)
    n_objectives: int = 1
    drift_rate: float = 0.0
    has_categorical: bool = False
    categorical_effect: float = 0.0
    seed: int = 42

    # ── evaluation ────────────────────────────────────────

    def evaluate(
        self,
        parameters: dict[str, Any],
        iteration: int = 0,
    ) -> dict[str, Any]:
        """Evaluate the synthetic objective at *parameters*.

        Returns a dict with keys ``kpi_values``, ``is_failure``,
        ``failure_reason``, and ``constraint_violated``.
        """
        rng = random.Random(
            self.seed + hash(str(sorted(parameters.items()))) + iteration
        )

        # -- random failure --
        if rng.random() < self.failure_rate:
            return {
                "kpi_values": {},
                "is_failure": True,
                "failure_reason": "random_failure",
                "constraint_violated": False,
            }

        # -- zone-based failure --
        if _check_failure_zone(parameters, self.failure_zones):
            return {
                "kpi_values": {},
                "is_failure": True,
                "failure_reason": "failure_zone",
                "constraint_violated": False,
            }

        # -- extract continuous values --
        x = [
            float(parameters.get(f"x{i}", 0.0))
            for i in range(self.n_dimensions)
        ]

        # -- compute base objective --
        fn = _LANDSCAPE_FN[self.landscape_type]
        base_value: float = fn(x)

        # -- non-stationary drift --
        base_value = _apply_drift(base_value, self.drift_rate, iteration)

        # -- noise --
        if self.noise_sigma > 0:
            base_value += rng.gauss(0, self.noise_sigma)

        # -- categorical modifier --
        if self.has_categorical and parameters.get("category") == "best":
            base_value -= self.categorical_effect

        # -- build KPI dict (primary + conflicting objectives) --
        kpi_values: dict[str, float] = {"kpi_0": base_value}
        if self.n_objectives > 1:
            for i in range(1, self.n_objectives):
                # Conflicting objective: shift the landscape optimum
                kpi_values[f"kpi_{i}"] = _sphere(x, shift=1.0 * i)

        # -- constraint check --
        constraint_violated = not _check_constraints(
            parameters, self.constraints
        )

        return {
            "kpi_values": kpi_values,
            "is_failure": False,
            "failure_reason": None,
            "constraint_violated": constraint_violated,
        }

    # ── serialization ─────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "name": self.name,
            "landscape_type": self.landscape_type.value,
            "n_dimensions": self.n_dimensions,
            "noise_sigma": self.noise_sigma,
            "failure_rate": self.failure_rate,
            "failure_zones": list(self.failure_zones),
            "constraints": list(self.constraints),
            "n_objectives": self.n_objectives,
            "drift_rate": self.drift_rate,
            "has_categorical": self.has_categorical,
            "categorical_effect": self.categorical_effect,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyntheticObjective:
        """Deserialize from a plain dict."""
        data = data.copy()
        data["landscape_type"] = LandscapeType(data["landscape_type"])
        data.setdefault("failure_zones", [])
        data.setdefault("constraints", [])
        return cls(**data)


# ── BenchmarkSpec ─────────────────────────────────────────


@dataclass
class BenchmarkSpec:
    """Configuration for batch benchmark generation.

    Controls which ProblemFingerprint dimensions to sweep and which
    optional scenario families (failures, constraints, multi-objective,
    non-stationary) to include.
    """

    dimensionality_range: tuple[int, int] = (2, 10)
    landscape_types: list[LandscapeType] = field(
        default_factory=lambda: list(LandscapeType),
    )
    noise_levels: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.3],
    )
    variable_types: list[VariableType] = field(
        default_factory=lambda: [VariableType.CONTINUOUS],
    )
    include_constraints: bool = True
    include_failures: bool = True
    include_multi_objective: bool = True
    include_non_stationary: bool = True
    n_observations_per_scenario: int = 30
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "dimensionality_range": list(self.dimensionality_range),
            "landscape_types": [lt.value for lt in self.landscape_types],
            "noise_levels": list(self.noise_levels),
            "variable_types": [vt.value for vt in self.variable_types],
            "include_constraints": self.include_constraints,
            "include_failures": self.include_failures,
            "include_multi_objective": self.include_multi_objective,
            "include_non_stationary": self.include_non_stationary,
            "n_observations_per_scenario": self.n_observations_per_scenario,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkSpec:
        """Deserialize from a plain dict."""
        data = data.copy()
        data["dimensionality_range"] = tuple(data["dimensionality_range"])
        data["landscape_types"] = [
            LandscapeType(v) for v in data["landscape_types"]
        ]
        data["variable_types"] = [
            VariableType(v) for v in data["variable_types"]
        ]
        return cls(**data)


# ── BenchmarkGenerator ────────────────────────────────────


class BenchmarkGenerator:
    """Generate reproducible GoldenScenario suites from BenchmarkSpec.

    All randomness is derived from the provided seed, ensuring full
    determinism across runs.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    # ── parameter spec helpers ────────────────────────────

    def _make_parameter_specs(
        self,
        n_continuous: int,
        n_categorical: int = 0,
        seed: int = 42,
    ) -> list[ParameterSpec]:
        """Build a list of ``ParameterSpec`` for the given dimensionality."""
        specs: list[ParameterSpec] = []
        for i in range(n_continuous):
            specs.append(
                ParameterSpec(
                    name=f"x{i}",
                    type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                )
            )
        if n_categorical > 0:
            specs.append(
                ParameterSpec(
                    name="category",
                    type=VariableType.CATEGORICAL,
                    lower=None,
                    upper=None,
                    categories=["A", "B", "best"],
                )
            )
        return specs

    # ── observation generation ────────────────────────────

    def _generate_observations(
        self,
        objective: SyntheticObjective,
        specs: list[ParameterSpec],
        n_observations: int,
        seed: int,
    ) -> list[Observation]:
        """Sample *n_observations* from *objective* using *specs*."""
        rng = random.Random(seed)
        observations: list[Observation] = []
        for i in range(n_observations):
            params: dict[str, Any] = {}
            for spec in specs:
                if spec.type == VariableType.CATEGORICAL:
                    params[spec.name] = rng.choice(spec.categories)  # type: ignore[arg-type]
                else:
                    params[spec.name] = rng.uniform(
                        spec.lower,  # type: ignore[arg-type]
                        spec.upper,  # type: ignore[arg-type]
                    )

            result = objective.evaluate(params, iteration=i)
            obs = Observation(
                iteration=i,
                parameters=params,
                kpi_values=result["kpi_values"],
                is_failure=result["is_failure"],
                failure_reason=result.get("failure_reason"),
                qc_passed=not result["constraint_violated"],
            )
            observations.append(obs)
        return observations

    # ── expectation inference ─────────────────────────────

    @staticmethod
    def _infer_expectation(
        objective: SyntheticObjective,
        observations: list[Observation],
    ) -> ScenarioExpectation:
        """Heuristically infer a ``ScenarioExpectation`` from observations."""
        failure_count = sum(1 for o in observations if o.is_failure)
        failure_rate = (
            failure_count / len(observations) if observations else 0.0
        )

        if failure_rate > 0.5:
            expected_phase = Phase.STAGNATION
            expected_risk: RiskPosture | None = RiskPosture.CONSERVATIVE
        elif len(observations) < 10:
            expected_phase = Phase.COLD_START
            expected_risk = RiskPosture.MODERATE
        else:
            expected_phase = Phase.LEARNING
            expected_risk = RiskPosture.MODERATE

        return ScenarioExpectation(
            expected_phase=expected_phase,
            expected_risk=expected_risk,
        )

    # ── single-scenario generation ────────────────────────

    def generate_scenario(
        self,
        objective: SyntheticObjective,
        n_observations: int = 30,
        scenario_name: str | None = None,
        seed: int | None = None,
    ) -> GoldenScenario:
        """Create a single ``GoldenScenario`` from *objective*."""
        seed = seed if seed is not None else self._seed
        name = scenario_name or f"synthetic_{objective.name}"

        specs = self._make_parameter_specs(
            objective.n_dimensions,
            n_categorical=1 if objective.has_categorical else 0,
        )
        objective_names = [f"kpi_{i}" for i in range(objective.n_objectives)]
        objective_directions = ["minimize"] * objective.n_objectives

        observations = self._generate_observations(
            objective, specs, n_observations, seed
        )

        snapshot = CampaignSnapshot(
            campaign_id=f"bench_{name}",
            parameter_specs=specs,
            observations=observations,
            objective_names=objective_names,
            objective_directions=objective_directions,
            constraints=objective.constraints,
            current_iteration=n_observations,
        )

        expectation = self._infer_expectation(objective, observations)

        return GoldenScenario(
            name=name,
            description=f"Synthetic {objective.landscape_type.value} benchmark",
            snapshot=snapshot,
            expectation=expectation,
            seed=seed,
        )

    # ── objective factory ─────────────────────────────────

    def _make_objective(
        self,
        landscape: LandscapeType,
        n_dims: int,
        noise_sigma: float = 0.0,
        failure_rate: float = 0.0,
        n_objectives: int = 1,
        drift_rate: float = 0.0,
        has_categorical: bool = False,
        seed: int = 42,
    ) -> SyntheticObjective:
        """Build a ``SyntheticObjective`` with an auto-generated name."""
        name = f"{landscape.value}_d{n_dims}"
        if noise_sigma > 0:
            name += f"_n{noise_sigma}"
        if failure_rate > 0:
            name += f"_f{failure_rate}"
        if n_objectives > 1:
            name += f"_mo{n_objectives}"
        if drift_rate > 0:
            name += f"_drift{drift_rate}"
        if has_categorical:
            name += "_cat"

        return SyntheticObjective(
            name=name,
            landscape_type=landscape,
            n_dimensions=n_dims,
            noise_sigma=noise_sigma,
            failure_rate=failure_rate,
            n_objectives=n_objectives,
            drift_rate=drift_rate,
            has_categorical=has_categorical,
            seed=seed,
        )

    # ── spec-driven generation ────────────────────────────

    def generate_from_spec(
        self, spec: BenchmarkSpec
    ) -> list[GoldenScenario]:
        """Generate a list of ``GoldenScenario`` from *spec*.

        Iterates the Cartesian product of landscapes, noise levels, and
        variable types, then appends optional scenario families when
        enabled.
        """
        scenarios: list[GoldenScenario] = []
        rng = random.Random(spec.seed)
        counter = 0

        for landscape in spec.landscape_types:
            for noise in spec.noise_levels:
                for var_type in spec.variable_types:
                    n_dims = rng.randint(*spec.dimensionality_range)
                    has_cat = var_type in (
                        VariableType.CATEGORICAL,
                        VariableType.MIXED,
                    )
                    obj = self._make_objective(
                        landscape,
                        n_dims,
                        noise_sigma=noise,
                        has_categorical=has_cat,
                        seed=spec.seed + counter,
                    )
                    scenarios.append(
                        self.generate_scenario(
                            obj,
                            n_observations=spec.n_observations_per_scenario,
                            seed=spec.seed + counter,
                        )
                    )
                    counter += 1

        # -- optional scenario families --

        if spec.include_failures:
            obj = self._make_objective(
                LandscapeType.SPHERE,
                3,
                failure_rate=0.4,
                seed=spec.seed + counter,
            )
            scenarios.append(
                self.generate_scenario(
                    obj,
                    spec.n_observations_per_scenario,
                    seed=spec.seed + counter,
                )
            )
            counter += 1

        if spec.include_constraints:
            obj = SyntheticObjective(
                name="constrained_sphere",
                landscape_type=LandscapeType.SPHERE,
                n_dimensions=3,
                constraints=[
                    {
                        "type": "sum_bound",
                        "parameters": ["x0", "x1", "x2"],
                        "bound": 1.5,
                    }
                ],
                seed=spec.seed + counter,
            )
            scenarios.append(
                self.generate_scenario(
                    obj,
                    spec.n_observations_per_scenario,
                    seed=spec.seed + counter,
                )
            )
            counter += 1

        if spec.include_multi_objective:
            obj = self._make_objective(
                LandscapeType.SPHERE,
                3,
                n_objectives=2,
                seed=spec.seed + counter,
            )
            scenarios.append(
                self.generate_scenario(
                    obj,
                    spec.n_observations_per_scenario,
                    seed=spec.seed + counter,
                )
            )
            counter += 1

        if spec.include_non_stationary:
            obj = self._make_objective(
                LandscapeType.SPHERE,
                3,
                drift_rate=0.1,
                seed=spec.seed + counter,
            )
            scenarios.append(
                self.generate_scenario(
                    obj,
                    spec.n_observations_per_scenario,
                    seed=spec.seed + counter,
                )
            )
            counter += 1

        return scenarios

    # ── coverage-level suites ─────────────────────────────

    def generate_suite(
        self, fingerprint_coverage: str = "minimal"
    ) -> list[GoldenScenario]:
        """Generate a full benchmark suite at the requested coverage level.

        Parameters
        ----------
        fingerprint_coverage:
            ``"minimal"`` -- default spec (4 landscapes x 3 noise levels
            x 1 variable type + optional families).
            ``"pairwise"`` -- expanded variable types and wider
            dimensionality range.
            ``"full"`` -- exhaustive combination of all ProblemFingerprint
            dimensions.

        Raises
        ------
        ValueError
            If *fingerprint_coverage* is not one of the recognized levels.
        """
        if fingerprint_coverage == "minimal":
            return self.generate_from_spec(BenchmarkSpec(seed=self._seed))

        if fingerprint_coverage == "pairwise":
            spec = BenchmarkSpec(
                variable_types=[
                    VariableType.CONTINUOUS,
                    VariableType.CATEGORICAL,
                    VariableType.MIXED,
                ],
                dimensionality_range=(2, 15),
                seed=self._seed,
            )
            return self.generate_from_spec(spec)

        if fingerprint_coverage == "full":
            spec = BenchmarkSpec(
                landscape_types=list(LandscapeType),
                noise_levels=[0.0, 0.01, 0.05, 0.1, 0.3, 0.5],
                variable_types=[
                    VariableType.CONTINUOUS,
                    VariableType.DISCRETE,
                    VariableType.CATEGORICAL,
                    VariableType.MIXED,
                ],
                dimensionality_range=(2, 20),
                seed=self._seed,
            )
            return self.generate_from_spec(spec)

        raise ValueError(f"Unknown coverage level: {fingerprint_coverage}")

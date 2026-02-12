"""Base protocol class for SDL benchmarks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import time
import random
import math

from .schema import BenchmarkSchema, ParameterDefinition, ObjectiveDefinition


@dataclass
class BenchmarkResult:
    """Result of running a benchmark."""

    benchmark_name: str
    algorithm_name: str
    observations: list[dict[str, Any]]  # list of {parameters: ..., kpi_values: ...}
    best_value: float
    best_parameters: dict[str, float]
    total_cost: float
    wall_time_seconds: float
    n_evaluations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "algorithm_name": self.algorithm_name,
            "observations": list(self.observations),
            "best_value": self.best_value,
            "best_parameters": dict(self.best_parameters),
            "total_cost": self.total_cost,
            "wall_time_seconds": self.wall_time_seconds,
            "n_evaluations": self.n_evaluations,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Deserialize from dictionary."""
        return cls(
            benchmark_name=data["benchmark_name"],
            algorithm_name=data["algorithm_name"],
            observations=data["observations"],
            best_value=data["best_value"],
            best_parameters=data["best_parameters"],
            total_cost=data["total_cost"],
            wall_time_seconds=data["wall_time_seconds"],
            n_evaluations=data["n_evaluations"],
            metadata=data.get("metadata", {}),
        )


class SDLBenchmarkProtocol:
    """Base class for SDL benchmark protocols."""

    def __init__(self, schema: BenchmarkSchema, seed: int = 42) -> None:
        """Initialize the benchmark protocol.

        Args:
            schema: The benchmark schema defining parameters, objectives, etc.
            seed: Random seed for reproducibility.
        """
        self._schema = schema
        self._seed = seed
        self._rng = random.Random(seed)
        self._history: list[dict[str, Any]] = []
        self._n_evaluations = 0
        self._best_value: float | None = None
        self._best_parameters: dict[str, float] | None = None
        self._total_cost = 0.0

    def evaluate(self, parameters: dict[str, float]) -> dict[str, float]:
        """Evaluate the benchmark function at given parameters.

        Subclasses override _evaluate_impl. This method adds noise,
        tracks budget, and records history.

        Args:
            parameters: Dictionary mapping parameter names to values.

        Returns:
            Dictionary mapping objective names to values.

        Raises:
            RuntimeError: If evaluation budget is exhausted.
            ValueError: If parameters are invalid.
        """
        if self._n_evaluations >= self._schema.evaluation_budget:
            raise RuntimeError(
                f"Evaluation budget exhausted ({self._schema.evaluation_budget} evaluations)"
            )

        # Validate parameters
        self._validate_parameters(parameters)

        # Get raw result from implementation
        raw_result = self._evaluate_impl(parameters)

        # Add noise
        noisy_result: dict[str, float] = {}
        for key, value in raw_result.items():
            noise = self._rng.gauss(0, self._schema.noise_level)
            noisy_result[key] = value + noise

        # Track evaluation
        self._n_evaluations += 1
        self._total_cost += 1.0

        # Record observation
        observation = {
            "parameters": dict(parameters),
            "kpi_values": dict(noisy_result),
            "evaluation_index": self._n_evaluations,
        }
        self._history.append(observation)

        # Update best so far (use first objective for tracking)
        primary_objective = self._schema.objectives[0]
        primary_value = noisy_result[primary_objective.name]
        if self._best_value is None:
            self._best_value = primary_value
            self._best_parameters = dict(parameters)
        else:
            if primary_objective.direction == "minimize":
                if primary_value < self._best_value:
                    self._best_value = primary_value
                    self._best_parameters = dict(parameters)
            else:  # maximize
                if primary_value > self._best_value:
                    self._best_value = primary_value
                    self._best_parameters = dict(parameters)

        return noisy_result

    def _evaluate_impl(self, parameters: dict[str, float]) -> dict[str, float]:
        """Override in subclasses to provide the actual benchmark function.

        Args:
            parameters: Dictionary mapping parameter names to values.

        Returns:
            Dictionary mapping objective names to raw (noiseless) values.
        """
        raise NotImplementedError("Subclasses must implement _evaluate_impl")

    def _validate_parameters(self, parameters: dict[str, float]) -> None:
        """Validate that parameters match the schema.

        Args:
            parameters: Dictionary mapping parameter names to values.

        Raises:
            ValueError: If parameters are invalid.
        """
        schema_param_names = {p.name for p in self._schema.parameters}
        provided_names = set(parameters.keys())

        missing = schema_param_names - provided_names
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

        extra = provided_names - schema_param_names
        if extra:
            raise ValueError(f"Unexpected parameters: {extra}")

        for param_def in self._schema.parameters:
            value = parameters[param_def.name]
            if param_def.type in ("continuous", "discrete"):
                if param_def.lower is not None and param_def.upper is not None:
                    if value < param_def.lower or value > param_def.upper:
                        raise ValueError(
                            f"Parameter '{param_def.name}' value {value} is outside "
                            f"bounds [{param_def.lower}, {param_def.upper}]"
                        )

    def run(
        self, algorithm_fn: Callable, algorithm_name: str = "unknown"
    ) -> BenchmarkResult:
        """Run a complete benchmark evaluation.

        The algorithm_fn receives this protocol instance and should call
        protocol.evaluate() repeatedly within the evaluation budget.

        Args:
            algorithm_fn: Function that takes the protocol and calls evaluate().
            algorithm_name: Name of the algorithm being evaluated.

        Returns:
            BenchmarkResult with all evaluation data.
        """
        self.reset()
        start_time = time.monotonic()

        algorithm_fn(self)

        wall_time = time.monotonic() - start_time

        return BenchmarkResult(
            benchmark_name=self._schema.name,
            algorithm_name=algorithm_name,
            observations=list(self._history),
            best_value=self._best_value if self._best_value is not None else float("inf"),
            best_parameters=self._best_parameters if self._best_parameters is not None else {},
            total_cost=self._total_cost,
            wall_time_seconds=wall_time,
            n_evaluations=self._n_evaluations,
            metadata={
                "seed": self._seed,
                "evaluation_budget": self._schema.evaluation_budget,
                "noise_level": self._schema.noise_level,
            },
        )

    @property
    def schema(self) -> BenchmarkSchema:
        """Return the benchmark schema."""
        return self._schema

    @property
    def budget_remaining(self) -> int:
        """Return the number of evaluations remaining."""
        return self._schema.evaluation_budget - self._n_evaluations

    @property
    def best_so_far(self) -> tuple[dict[str, float], float] | None:
        """Return the best parameters and value seen so far, or None if no evaluations."""
        if self._best_value is None or self._best_parameters is None:
            return None
        return (self._best_parameters, self._best_value)

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return the evaluation history."""
        return list(self._history)

    def reset(self) -> None:
        """Reset the protocol for re-running."""
        self._rng = random.Random(self._seed)
        self._history = []
        self._n_evaluations = 0
        self._best_value = None
        self._best_parameters = None
        self._total_cost = 0.0


class SphereBenchmark(SDLBenchmarkProtocol):
    """Simple sphere function benchmark for testing.

    The sphere function is sum of squares, to be minimized.
    Global optimum is 0 at the origin.
    """

    def __init__(
        self,
        n_dims: int = 3,
        evaluation_budget: int = 100,
        noise_level: float = 0.01,
        seed: int = 42,
    ) -> None:
        """Initialize sphere benchmark.

        Args:
            n_dims: Number of dimensions.
            evaluation_budget: Maximum number of evaluations.
            noise_level: Standard deviation of Gaussian noise.
            seed: Random seed.
        """
        parameters = [
            ParameterDefinition(
                name=f"x{i}",
                type="continuous",
                lower=-5.0,
                upper=5.0,
            )
            for i in range(n_dims)
        ]
        objectives = [
            ObjectiveDefinition(
                name="objective",
                direction="minimize",
                target=0.0,
            )
        ]
        schema = BenchmarkSchema(
            name="sphere",
            version="1.0",
            description=f"Sphere function in {n_dims} dimensions",
            domain="synthetic",
            parameters=parameters,
            objectives=objectives,
            evaluation_budget=evaluation_budget,
            noise_level=noise_level,
        )
        super().__init__(schema, seed=seed)

    def _evaluate_impl(self, parameters: dict[str, float]) -> dict[str, float]:
        """Evaluate the sphere function: sum of squares."""
        val = sum(v ** 2 for v in parameters.values())
        return {"objective": val}

"""Standard benchmark test functions for optimization algorithm evaluation.

All functions accept a dict[str, float] of parameter values and return
a dict[str, float] of objective values. Parameters are defined with
standard bounds. Each function includes:
- The evaluation function
- Parameter specifications (name, bounds, type)
- Known global optimum (for regret calculation)
- Metadata (dimensionality, difficulty, characteristics)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BenchmarkFunction:
    """A standard benchmark test function."""

    name: str
    evaluate: Callable[[dict[str, float]], dict[str, float]]
    parameter_specs: list[dict[str, Any]]  # [{name, type, bounds, ...}]
    known_optimum: dict[str, float]  # {objective_name: optimal_value}
    optimal_params: dict[str, float] | None  # Known optimal x*
    metadata: dict[str, Any] = field(default_factory=dict)

    def __call__(self, params: dict[str, float]) -> dict[str, float]:
        return self.evaluate(params)


# ── Constants for Hartmann functions ─────────────────────────────────


_HARTMANN3_ALPHA = [1.0, 1.2, 3.0, 3.2]
_HARTMANN3_A = [
    [3.0, 10.0, 30.0],
    [0.1, 10.0, 35.0],
    [3.0, 10.0, 30.0],
    [0.1, 10.0, 35.0],
]
_HARTMANN3_P = [
    [0.3689, 0.1170, 0.2673],
    [0.4699, 0.4387, 0.7470],
    [0.1091, 0.8732, 0.5547],
    [0.0381, 0.5743, 0.8828],
]

_HARTMANN6_ALPHA = [1.0, 1.2, 3.0, 3.2]
_HARTMANN6_A = [
    [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
    [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
    [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
    [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
]
_HARTMANN6_P = [
    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
]


# ── Individual test functions ────────────────────────────────────────


def branin(params: dict[str, float]) -> dict[str, float]:
    """Branin-Hoo function. 2D with 3 global minima.

    Domain: x1 in [-5, 10], x2 in [0, 15]
    Global minimum: f* ~ 0.397887
    """
    x1 = params["x1"]
    x2 = params["x2"]
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    val = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1.0 - t) * math.cos(x1) + s
    return {"objective": val}


def hartmann3(params: dict[str, float]) -> dict[str, float]:
    """Hartmann 3D function.

    Domain: xi in [0, 1] for i = 1, 2, 3
    Global minimum: f* ~ -3.86278
    """
    x = [params["x1"], params["x2"], params["x3"]]
    outer = 0.0
    for i in range(4):
        inner = 0.0
        for j in range(3):
            inner += _HARTMANN3_A[i][j] * (x[j] - _HARTMANN3_P[i][j]) ** 2
        outer += _HARTMANN3_ALPHA[i] * math.exp(-inner)
    return {"objective": -outer}


def hartmann6(params: dict[str, float]) -> dict[str, float]:
    """Hartmann 6D function.

    Domain: xi in [0, 1] for i = 1..6
    Global minimum: f* ~ -3.32237
    """
    x = [params[f"x{i}"] for i in range(1, 7)]
    outer = 0.0
    for i in range(4):
        inner = 0.0
        for j in range(6):
            inner += _HARTMANN6_A[i][j] * (x[j] - _HARTMANN6_P[i][j]) ** 2
        outer += _HARTMANN6_ALPHA[i] * math.exp(-inner)
    return {"objective": -outer}


def levy(params: dict[str, float]) -> dict[str, float]:
    """Levy function in 10D.

    Domain: xi in [-10, 10]
    Global minimum: f* = 0 at xi = 1
    """
    d = 10
    x = [params[f"x{i}"] for i in range(1, d + 1)]
    w = [1.0 + (xi - 1.0) / 4.0 for xi in x]
    term1 = math.sin(math.pi * w[0]) ** 2
    term_sum = 0.0
    for i in range(d - 1):
        term_sum += (w[i] - 1.0) ** 2 * (1.0 + 10.0 * math.sin(math.pi * w[i] + 1.0) ** 2)
    term_last = (w[-1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * w[-1]) ** 2)
    return {"objective": term1 + term_sum + term_last}


def rosenbrock(params: dict[str, float]) -> dict[str, float]:
    """Rosenbrock function in configurable dimensions.

    Domain: xi in [-5, 10]
    Global minimum: f* = 0 at xi = 1
    """
    keys = sorted(k for k in params if k.startswith("x"))
    x = [params[k] for k in keys]
    total = 0.0
    for i in range(len(x) - 1):
        total += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return {"objective": total}


def constrained_branin(params: dict[str, float]) -> dict[str, float]:
    """Branin with feasibility constraint.

    Adds constraint: g(x) = x1 + x2 <= 14
    Returns: {"objective": value, "constraint_violation": 0.0 or positive}
    """
    result = branin(params)
    x1 = params["x1"]
    x2 = params["x2"]
    g = x1 + x2 - 14.0
    result["constraint_violation"] = max(0.0, g)
    return result


def noisy_hartmann6(params: dict[str, float], noise_std: float = 0.1) -> dict[str, float]:
    """Hartmann 6D with additive Gaussian noise."""
    result = hartmann6(params)
    seed_val = int(sum(params[f"x{i}"] * (1000 + i) for i in range(1, 7)))
    rng = random.Random(seed_val)
    result["objective"] += rng.gauss(0.0, noise_std)
    return result


def multifidelity_branin(params: dict[str, float], fidelity: int = 2) -> dict[str, float]:
    """Multi-fidelity Branin.

    fidelity=0: cheap approximation (with bias)
    fidelity=1: medium approximation
    fidelity=2: high fidelity (exact)
    """
    exact = branin(params)
    val = exact["objective"]
    x1 = params["x1"]
    x2 = params["x2"]
    if fidelity == 0:
        # Low fidelity: biased quadratic approximation
        bias = 0.5 * (x1 - 2.5) ** 2 + 0.5 * (x2 - 7.5) ** 2 - 20.0
        val = val + bias + 10.0
    elif fidelity == 1:
        # Medium fidelity: small systematic bias
        val = val + 2.0 * math.sin(x1) + 1.0
    # fidelity == 2: exact
    return {"objective": val, "fidelity": float(fidelity)}


def zdt1(params: dict[str, float]) -> dict[str, float]:
    """ZDT1 bi-objective test function.

    Domain: xi in [0, 1] for i = 1..30
    Returns: {"f1": value1, "f2": value2}
    Pareto front: f2 = 1 - sqrt(f1) when g = 1
    """
    n = 30
    x = [params[f"x{i}"] for i in range(1, n + 1)]
    f1 = x[0]
    g = 1.0 + 9.0 / (n - 1) * sum(x[1:])
    f2 = g * (1.0 - math.sqrt(f1 / g)) if g > 0 and f1 / g >= 0 else g
    return {"f1": f1, "f2": f2}


# ── Helper to build parameter spec lists ─────────────────────────────


def _make_specs(
    n_dims: int,
    bounds: tuple[float, float],
    prefix: str = "x",
) -> list[dict[str, Any]]:
    """Create a list of continuous parameter specs."""
    return [
        {"name": f"{prefix}{i}", "type": "continuous", "bounds": list(bounds)}
        for i in range(1, n_dims + 1)
    ]


# ── Wrapper factories for noisy / multi-fidelity callables ───────────


def _make_noisy_hartmann6(noise_std: float = 0.1) -> Callable[[dict[str, float]], dict[str, float]]:
    """Return a zero-argument-noise wrapper for noisy_hartmann6."""
    def _fn(params: dict[str, float]) -> dict[str, float]:
        return noisy_hartmann6(params, noise_std=noise_std)
    return _fn


def _make_multifidelity_branin(fidelity: int = 2) -> Callable[[dict[str, float]], dict[str, float]]:
    """Return a fixed-fidelity wrapper for multifidelity_branin."""
    def _fn(params: dict[str, float]) -> dict[str, float]:
        return multifidelity_branin(params, fidelity=fidelity)
    return _fn


# ── Registry ─────────────────────────────────────────────────────────


BENCHMARK_SUITE: dict[str, BenchmarkFunction] = {
    "branin": BenchmarkFunction(
        name="branin",
        evaluate=branin,
        parameter_specs=[
            {"name": "x1", "type": "continuous", "bounds": [-5.0, 10.0]},
            {"name": "x2", "type": "continuous", "bounds": [0.0, 15.0]},
        ],
        known_optimum={"objective": 0.397887},
        optimal_params={"x1": -math.pi, "x2": 12.275},
        metadata={
            "dimensionality": 2,
            "difficulty": "easy",
            "characteristics": ["multimodal", "continuous"],
            "n_local_minima": 3,
        },
    ),
    "hartmann3": BenchmarkFunction(
        name="hartmann3",
        evaluate=hartmann3,
        parameter_specs=_make_specs(3, (0.0, 1.0)),
        known_optimum={"objective": -3.86278},
        optimal_params={"x1": 0.114614, "x2": 0.555649, "x3": 0.852547},
        metadata={
            "dimensionality": 3,
            "difficulty": "easy",
            "characteristics": ["multimodal", "continuous"],
        },
    ),
    "hartmann6": BenchmarkFunction(
        name="hartmann6",
        evaluate=hartmann6,
        parameter_specs=_make_specs(6, (0.0, 1.0)),
        known_optimum={"objective": -3.32237},
        optimal_params={
            "x1": 0.20169,
            "x2": 0.15001,
            "x3": 0.47687,
            "x4": 0.27533,
            "x5": 0.31165,
            "x6": 0.65730,
        },
        metadata={
            "dimensionality": 6,
            "difficulty": "moderate",
            "characteristics": ["multimodal", "continuous"],
        },
    ),
    "levy10": BenchmarkFunction(
        name="levy10",
        evaluate=levy,
        parameter_specs=_make_specs(10, (-10.0, 10.0)),
        known_optimum={"objective": 0.0},
        optimal_params={f"x{i}": 1.0 for i in range(1, 11)},
        metadata={
            "dimensionality": 10,
            "difficulty": "hard",
            "characteristics": ["multimodal", "continuous", "high-dimensional"],
        },
    ),
    "rosenbrock5": BenchmarkFunction(
        name="rosenbrock5",
        evaluate=rosenbrock,
        parameter_specs=_make_specs(5, (-5.0, 10.0)),
        known_optimum={"objective": 0.0},
        optimal_params={f"x{i}": 1.0 for i in range(1, 6)},
        metadata={
            "dimensionality": 5,
            "difficulty": "moderate",
            "characteristics": ["unimodal", "continuous", "banana-valley"],
        },
    ),
    "rosenbrock20": BenchmarkFunction(
        name="rosenbrock20",
        evaluate=rosenbrock,
        parameter_specs=_make_specs(20, (-5.0, 10.0)),
        known_optimum={"objective": 0.0},
        optimal_params={f"x{i}": 1.0 for i in range(1, 21)},
        metadata={
            "dimensionality": 20,
            "difficulty": "hard",
            "characteristics": [
                "unimodal",
                "continuous",
                "banana-valley",
                "high-dimensional",
            ],
        },
    ),
    "constrained_branin": BenchmarkFunction(
        name="constrained_branin",
        evaluate=constrained_branin,
        parameter_specs=[
            {"name": "x1", "type": "continuous", "bounds": [-5.0, 10.0]},
            {"name": "x2", "type": "continuous", "bounds": [0.0, 15.0]},
        ],
        known_optimum={"objective": 0.397887, "constraint_violation": 0.0},
        optimal_params={"x1": -math.pi, "x2": 12.275},
        metadata={
            "dimensionality": 2,
            "difficulty": "moderate",
            "characteristics": ["multimodal", "continuous", "constrained"],
            "constraint": "x1 + x2 <= 14",
        },
    ),
    "noisy_hartmann6": BenchmarkFunction(
        name="noisy_hartmann6",
        evaluate=_make_noisy_hartmann6(noise_std=0.1),
        parameter_specs=_make_specs(6, (0.0, 1.0)),
        known_optimum={"objective": -3.32237},
        optimal_params={
            "x1": 0.20169,
            "x2": 0.15001,
            "x3": 0.47687,
            "x4": 0.27533,
            "x5": 0.31165,
            "x6": 0.65730,
        },
        metadata={
            "dimensionality": 6,
            "difficulty": "hard",
            "characteristics": ["multimodal", "continuous", "noisy"],
            "noise_std": 0.1,
        },
    ),
    "multifidelity_branin": BenchmarkFunction(
        name="multifidelity_branin",
        evaluate=_make_multifidelity_branin(fidelity=2),
        parameter_specs=[
            {"name": "x1", "type": "continuous", "bounds": [-5.0, 10.0]},
            {"name": "x2", "type": "continuous", "bounds": [0.0, 15.0]},
        ],
        known_optimum={"objective": 0.397887},
        optimal_params={"x1": -math.pi, "x2": 12.275},
        metadata={
            "dimensionality": 2,
            "difficulty": "moderate",
            "characteristics": ["multimodal", "continuous", "multi-fidelity"],
            "fidelity_levels": 3,
        },
    ),
    "zdt1": BenchmarkFunction(
        name="zdt1",
        evaluate=zdt1,
        parameter_specs=_make_specs(30, (0.0, 1.0)),
        known_optimum={"f1": 0.0, "f2": 0.0},
        optimal_params=None,  # Pareto front, no single optimum
        metadata={
            "dimensionality": 30,
            "difficulty": "hard",
            "characteristics": [
                "multi-objective",
                "continuous",
                "high-dimensional",
            ],
            "n_objectives": 2,
            "pareto_front": "f2 = 1 - sqrt(f1)",
        },
    ),
}


# ── Public API ───────────────────────────────────────────────────────


def get_benchmark(name: str) -> BenchmarkFunction:
    """Get a benchmark function by name.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name not in BENCHMARK_SUITE:
        available = ", ".join(sorted(BENCHMARK_SUITE))
        raise KeyError(f"Unknown benchmark {name!r}. Available: {available}")
    return BENCHMARK_SUITE[name]


def list_benchmarks() -> list[str]:
    """List all available benchmark function names."""
    return sorted(BENCHMARK_SUITE)


def make_spec(benchmark: BenchmarkFunction, budget_iterations: int = 50) -> dict:
    """Create an OptimizationSpec-compatible dict from a benchmark function.

    The returned dict contains the fields commonly needed to set up an
    optimization campaign from a benchmark definition.
    """
    objectives: list[dict[str, str]] = []
    for obj_name in benchmark.known_optimum:
        objectives.append({
            "name": obj_name,
            "direction": "minimize",
        })

    return {
        "name": benchmark.name,
        "parameters": benchmark.parameter_specs,
        "objectives": objectives,
        "budget_iterations": budget_iterations,
        "known_optimum": benchmark.known_optimum,
        "optimal_params": benchmark.optimal_params,
        "metadata": benchmark.metadata,
    }

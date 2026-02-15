"""Tabular benchmark with embedded real-world-inspired chemistry data.

Provides a ``TabularBenchmark`` that performs nearest-neighbor lookup on
an embedded 48-row Buchwald-Hartwig-inspired dataset — zero external
file dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.benchmark.functions import BenchmarkFunction


# ── Embedded dataset ────────────────────────────────────────────────

# 48-row Buchwald-Hartwig-inspired coupling reaction dataset.
# Columns: temperature (°C), catalyst_loading (mol%), concentration (M),
#          time (h), yield_pct (%).
_BUCHWALD_HARTWIG_DATA: list[dict[str, float]] = [
    # Row format: {"temperature": ..., "catalyst_loading": ..., "concentration": ..., "time": ..., "yield_pct": ...}
    # Low temperature regime (60-80°C) — generally lower yields
    {"temperature": 60.0, "catalyst_loading": 0.5, "concentration": 0.05, "time": 1.0, "yield_pct": 5.2},
    {"temperature": 60.0, "catalyst_loading": 1.0, "concentration": 0.10, "time": 4.0, "yield_pct": 12.8},
    {"temperature": 60.0, "catalyst_loading": 2.0, "concentration": 0.20, "time": 8.0, "yield_pct": 28.3},
    {"temperature": 60.0, "catalyst_loading": 3.0, "concentration": 0.30, "time": 12.0, "yield_pct": 35.1},
    {"temperature": 60.0, "catalyst_loading": 4.0, "concentration": 0.40, "time": 16.0, "yield_pct": 31.7},
    {"temperature": 60.0, "catalyst_loading": 5.0, "concentration": 0.50, "time": 24.0, "yield_pct": 22.4},
    {"temperature": 70.0, "catalyst_loading": 0.5, "concentration": 0.10, "time": 2.0, "yield_pct": 8.9},
    {"temperature": 70.0, "catalyst_loading": 1.5, "concentration": 0.15, "time": 6.0, "yield_pct": 25.6},
    {"temperature": 70.0, "catalyst_loading": 2.5, "concentration": 0.25, "time": 10.0, "yield_pct": 42.0},
    {"temperature": 70.0, "catalyst_loading": 3.5, "concentration": 0.35, "time": 14.0, "yield_pct": 48.3},
    {"temperature": 70.0, "catalyst_loading": 4.5, "concentration": 0.45, "time": 20.0, "yield_pct": 38.7},
    {"temperature": 70.0, "catalyst_loading": 5.0, "concentration": 0.50, "time": 24.0, "yield_pct": 30.1},
    # Medium temperature regime (80-100°C) — sweet spot
    {"temperature": 80.0, "catalyst_loading": 1.0, "concentration": 0.10, "time": 4.0, "yield_pct": 35.2},
    {"temperature": 80.0, "catalyst_loading": 2.0, "concentration": 0.20, "time": 8.0, "yield_pct": 58.6},
    {"temperature": 80.0, "catalyst_loading": 2.5, "concentration": 0.25, "time": 10.0, "yield_pct": 67.4},
    {"temperature": 80.0, "catalyst_loading": 3.0, "concentration": 0.30, "time": 12.0, "yield_pct": 72.1},
    {"temperature": 80.0, "catalyst_loading": 4.0, "concentration": 0.40, "time": 16.0, "yield_pct": 63.5},
    {"temperature": 80.0, "catalyst_loading": 5.0, "concentration": 0.50, "time": 24.0, "yield_pct": 51.8},
    {"temperature": 90.0, "catalyst_loading": 1.0, "concentration": 0.10, "time": 3.0, "yield_pct": 42.7},
    {"temperature": 90.0, "catalyst_loading": 2.0, "concentration": 0.15, "time": 6.0, "yield_pct": 68.9},
    {"temperature": 90.0, "catalyst_loading": 2.5, "concentration": 0.20, "time": 8.0, "yield_pct": 78.3},
    {"temperature": 90.0, "catalyst_loading": 3.0, "concentration": 0.25, "time": 10.0, "yield_pct": 85.2},
    {"temperature": 90.0, "catalyst_loading": 3.5, "concentration": 0.30, "time": 12.0, "yield_pct": 88.7},
    {"temperature": 90.0, "catalyst_loading": 4.0, "concentration": 0.35, "time": 14.0, "yield_pct": 82.1},
    {"temperature": 100.0, "catalyst_loading": 1.5, "concentration": 0.10, "time": 4.0, "yield_pct": 55.3},
    {"temperature": 100.0, "catalyst_loading": 2.0, "concentration": 0.15, "time": 6.0, "yield_pct": 73.8},
    {"temperature": 100.0, "catalyst_loading": 2.5, "concentration": 0.20, "time": 8.0, "yield_pct": 86.4},
    {"temperature": 100.0, "catalyst_loading": 3.0, "concentration": 0.25, "time": 10.0, "yield_pct": 93.1},
    {"temperature": 100.0, "catalyst_loading": 3.5, "concentration": 0.30, "time": 12.0, "yield_pct": 91.5},
    {"temperature": 100.0, "catalyst_loading": 4.5, "concentration": 0.40, "time": 18.0, "yield_pct": 78.9},
    # Higher temperature regime (110-120°C) — starts to drop off
    {"temperature": 110.0, "catalyst_loading": 1.0, "concentration": 0.10, "time": 3.0, "yield_pct": 48.6},
    {"temperature": 110.0, "catalyst_loading": 2.0, "concentration": 0.20, "time": 6.0, "yield_pct": 74.2},
    {"temperature": 110.0, "catalyst_loading": 2.5, "concentration": 0.25, "time": 8.0, "yield_pct": 82.7},
    {"temperature": 110.0, "catalyst_loading": 3.0, "concentration": 0.30, "time": 10.0, "yield_pct": 87.3},
    {"temperature": 110.0, "catalyst_loading": 3.5, "concentration": 0.35, "time": 14.0, "yield_pct": 79.6},
    {"temperature": 110.0, "catalyst_loading": 5.0, "concentration": 0.50, "time": 24.0, "yield_pct": 56.2},
    {"temperature": 120.0, "catalyst_loading": 1.5, "concentration": 0.15, "time": 4.0, "yield_pct": 52.1},
    {"temperature": 120.0, "catalyst_loading": 2.5, "concentration": 0.25, "time": 8.0, "yield_pct": 75.8},
    {"temperature": 120.0, "catalyst_loading": 3.0, "concentration": 0.30, "time": 10.0, "yield_pct": 80.4},
    {"temperature": 120.0, "catalyst_loading": 4.0, "concentration": 0.40, "time": 16.0, "yield_pct": 65.3},
    {"temperature": 120.0, "catalyst_loading": 5.0, "concentration": 0.50, "time": 20.0, "yield_pct": 47.9},
    # Extreme temperature (130-140°C) — degradation
    {"temperature": 130.0, "catalyst_loading": 2.0, "concentration": 0.20, "time": 6.0, "yield_pct": 62.4},
    {"temperature": 130.0, "catalyst_loading": 3.0, "concentration": 0.30, "time": 10.0, "yield_pct": 71.8},
    {"temperature": 130.0, "catalyst_loading": 4.0, "concentration": 0.40, "time": 16.0, "yield_pct": 55.3},
    {"temperature": 130.0, "catalyst_loading": 5.0, "concentration": 0.50, "time": 24.0, "yield_pct": 38.6},
    {"temperature": 140.0, "catalyst_loading": 2.0, "concentration": 0.20, "time": 4.0, "yield_pct": 45.2},
    {"temperature": 140.0, "catalyst_loading": 3.0, "concentration": 0.30, "time": 8.0, "yield_pct": 58.7},
    {"temperature": 140.0, "catalyst_loading": 5.0, "concentration": 0.50, "time": 24.0, "yield_pct": 28.3},
]

# Parameter bounds for normalization
_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature": (60.0, 140.0),
    "catalyst_loading": (0.5, 5.0),
    "concentration": (0.05, 0.5),
    "time": (1.0, 24.0),
}

_PARAM_NAMES = list(_PARAM_BOUNDS.keys())


@dataclass
class TabularBenchmark:
    """Tabular benchmark using nearest-neighbor lookup on embedded data.

    The dataset represents a Buchwald-Hartwig coupling reaction optimization
    landscape with 48 data points and 4 continuous parameters.

    The objective is negated yield (for minimization): lower = better yield.
    """

    data: list[dict[str, float]] = field(default_factory=lambda: list(_BUCHWALD_HARTWIG_DATA))
    name: str = "buchwald_hartwig"

    def evaluate(self, params: dict[str, float]) -> dict[str, float]:
        """Find nearest neighbor in embedded data and return its negated yield.

        Parameters are normalized to [0, 1] before distance computation.
        Returns {"objective": -yield_pct} (negated for minimization).
        """
        # Normalize query
        query_norm = self._normalize(params)

        # Find nearest neighbor by Euclidean distance in normalized space
        best_dist = float("inf")
        best_yield = 0.0
        for row in self.data:
            row_norm = self._normalize(row)
            dist = math.sqrt(sum((query_norm[k] - row_norm[k]) ** 2 for k in _PARAM_NAMES))
            if dist < best_dist:
                best_dist = dist
                best_yield = row["yield_pct"]

        # Negate yield for minimization (optimizer minimizes, we want max yield)
        return {"objective": -best_yield}

    def to_benchmark_function(self) -> BenchmarkFunction:
        """Convert to a BenchmarkFunction for use with the evaluation harness."""
        # Find the best row (highest yield)
        best_row = max(self.data, key=lambda r: r["yield_pct"])

        return BenchmarkFunction(
            name=self.name,
            evaluate=self.evaluate,
            parameter_specs=[
                {"name": name, "type": "continuous", "bounds": list(bounds)}
                for name, bounds in _PARAM_BOUNDS.items()
            ],
            known_optimum={"objective": -best_row["yield_pct"]},
            optimal_params={k: best_row[k] for k in _PARAM_NAMES},
            metadata={
                "dimensionality": len(_PARAM_NAMES),
                "difficulty": "moderate",
                "characteristics": ["tabular", "chemistry", "real-world-inspired"],
                "n_data_points": len(self.data),
                "best_yield_pct": best_row["yield_pct"],
                "source": "Buchwald-Hartwig coupling (synthetic, inspired by real data)",
            },
        )

    @staticmethod
    def _normalize(params: dict[str, float]) -> dict[str, float]:
        """Normalize parameters to [0, 1] range."""
        result = {}
        for name in _PARAM_NAMES:
            lo, hi = _PARAM_BOUNDS[name]
            val = params.get(name, (lo + hi) / 2.0)
            result[name] = (val - lo) / (hi - lo) if hi > lo else 0.5
        return result

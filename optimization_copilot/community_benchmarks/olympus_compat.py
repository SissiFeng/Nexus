"""Olympus benchmark compatibility layer.

Wraps Olympus-style tabular datasets as callable benchmark surfaces with
nearest-neighbor interpolation for off-grid queries.  Each surface can be
converted to the project's standard ``BenchmarkFunction`` dataclass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.benchmark.functions import BenchmarkFunction


# ---------------------------------------------------------------------------
# OlympusSurface
# ---------------------------------------------------------------------------

@dataclass
class OlympusSurface:
    """An Olympus-style tabular benchmark surface.

    Stores a lookup table of evaluated points and performs nearest-neighbor
    interpolation (Euclidean distance) for arbitrary query points.
    """

    name: str
    param_names: list[str]
    param_bounds: list[tuple[float, float]]
    lookup_table: list[dict[str, float]]
    objective_name: str = "objective"

    # -- evaluation --------------------------------------------------------

    def evaluate(self, params: dict[str, float]) -> dict[str, float]:
        """Return the objective value for *params* via nearest-neighbor lookup.

        Finds the entry in :pyattr:`lookup_table` that is closest (Euclidean)
        to the query point and returns its objective value.
        """
        best_dist = math.inf
        best_value = 0.0

        for entry in self.lookup_table:
            dist = 0.0
            for pname in self.param_names:
                diff = params.get(pname, 0.0) - entry[pname]
                dist += diff * diff
            dist = math.sqrt(dist)
            if dist < best_dist:
                best_dist = dist
                best_value = entry[self.objective_name]

        return {self.objective_name: best_value}

    # -- conversion --------------------------------------------------------

    def to_benchmark(self) -> BenchmarkFunction:
        """Convert this surface to a :class:`BenchmarkFunction`."""
        # Find the best (minimum) objective value in the lookup table
        best_entry: dict[str, float] | None = None
        best_obj = math.inf
        for entry in self.lookup_table:
            val = entry[self.objective_name]
            if val < best_obj:
                best_obj = val
                best_entry = entry

        known_optimum = {self.objective_name: best_obj}
        optimal_params: dict[str, float] | None = None
        if best_entry is not None:
            optimal_params = {
                p: best_entry[p] for p in self.param_names
            }

        parameter_specs: list[dict[str, Any]] = [
            {
                "name": pname,
                "type": "continuous",
                "bounds": list(bounds),
            }
            for pname, bounds in zip(self.param_names, self.param_bounds)
        ]

        return BenchmarkFunction(
            name=f"olympus_{self.name}",
            evaluate=self.evaluate,
            parameter_specs=parameter_specs,
            known_optimum=known_optimum,
            optimal_params=optimal_params,
            metadata={
                "source": "olympus",
                "surface_name": self.name,
                "objective_name": self.objective_name,
                "lookup_table_size": len(self.lookup_table),
            },
        )


# ---------------------------------------------------------------------------
# OlympusLoader
# ---------------------------------------------------------------------------

class OlympusLoader:
    """Factory methods for constructing :class:`OlympusSurface` instances."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> OlympusSurface:
        """Build a surface from a plain dictionary.

        Expected keys: ``name``, ``param_names``, ``param_bounds``,
        ``lookup_table``, and optionally ``objective_name``.
        """
        return OlympusSurface(
            name=data["name"],
            param_names=data["param_names"],
            param_bounds=[tuple(b) for b in data["param_bounds"]],
            lookup_table=data["lookup_table"],
            objective_name=data.get("objective_name", "objective"),
        )

    @staticmethod
    def from_csv_string(
        csv_text: str,
        param_names: list[str],
        objective_name: str,
        param_bounds: list[tuple[float, float]],
    ) -> OlympusSurface:
        """Parse a CSV string into an :class:`OlympusSurface`.

        The first row is treated as a header.  Each subsequent row becomes
        one entry in the lookup table.
        """
        lines = [ln.strip() for ln in csv_text.strip().splitlines() if ln.strip()]
        if not lines:
            raise ValueError("CSV text is empty")

        header = [h.strip() for h in lines[0].split(",")]
        lookup_table: list[dict[str, float]] = []

        for line in lines[1:]:
            values = [v.strip() for v in line.split(",")]
            entry: dict[str, float] = {}
            for col_name, val_str in zip(header, values):
                entry[col_name] = float(val_str)
            lookup_table.append(entry)

        # Derive name from the objective column
        name = objective_name.replace(" ", "_")

        return OlympusSurface(
            name=name,
            param_names=param_names,
            param_bounds=param_bounds,
            lookup_table=lookup_table,
            objective_name=objective_name,
        )


# ---------------------------------------------------------------------------
# Built-in registry  (small lookup tables for testing / demonstration)
# ---------------------------------------------------------------------------

OLYMPUS_REGISTRY: dict[str, OlympusSurface] = {
    # 1. photobleaching -- 2 params (power, duration), objective: bleach_rate
    "photobleaching": OlympusSurface(
        name="photobleaching",
        param_names=["power", "duration"],
        param_bounds=[(0.1, 10.0), (1.0, 60.0)],
        objective_name="bleach_rate",
        lookup_table=[
            {"power": 0.5, "duration": 5.0, "bleach_rate": 0.12},
            {"power": 1.0, "duration": 10.0, "bleach_rate": 0.25},
            {"power": 2.0, "duration": 15.0, "bleach_rate": 0.48},
            {"power": 5.0, "duration": 30.0, "bleach_rate": 0.78},
            {"power": 7.0, "duration": 45.0, "bleach_rate": 0.91},
            {"power": 10.0, "duration": 60.0, "bleach_rate": 0.99},
            {"power": 3.0, "duration": 20.0, "bleach_rate": 0.55},
            {"power": 0.1, "duration": 1.0, "bleach_rate": 0.02},
        ],
    ),

    # 2. crossed_barrel -- 2 params (angle, speed), objective: quality
    "crossed_barrel": OlympusSurface(
        name="crossed_barrel",
        param_names=["angle", "speed"],
        param_bounds=[(0.0, 90.0), (10.0, 500.0)],
        objective_name="quality",
        lookup_table=[
            {"angle": 10.0, "speed": 50.0, "quality": 0.65},
            {"angle": 25.0, "speed": 100.0, "quality": 0.42},
            {"angle": 45.0, "speed": 200.0, "quality": 0.28},
            {"angle": 60.0, "speed": 300.0, "quality": 0.35},
            {"angle": 75.0, "speed": 400.0, "quality": 0.55},
            {"angle": 90.0, "speed": 500.0, "quality": 0.72},
            {"angle": 30.0, "speed": 150.0, "quality": 0.31},
        ],
    ),

    # 3. colors_bob -- 3 params (red, green, blue), objective: match_score
    "colors_bob": OlympusSurface(
        name="colors_bob",
        param_names=["red", "green", "blue"],
        param_bounds=[(0.0, 255.0), (0.0, 255.0), (0.0, 255.0)],
        objective_name="match_score",
        lookup_table=[
            {"red": 128.0, "green": 128.0, "blue": 128.0, "match_score": 0.50},
            {"red": 200.0, "green": 50.0, "blue": 50.0, "match_score": 0.72},
            {"red": 50.0, "green": 200.0, "blue": 50.0, "match_score": 0.68},
            {"red": 50.0, "green": 50.0, "blue": 200.0, "match_score": 0.61},
            {"red": 255.0, "green": 255.0, "blue": 0.0, "match_score": 0.85},
            {"red": 0.0, "green": 255.0, "blue": 255.0, "match_score": 0.78},
            {"red": 255.0, "green": 0.0, "blue": 255.0, "match_score": 0.81},
            {"red": 0.0, "green": 0.0, "blue": 0.0, "match_score": 0.10},
            {"red": 255.0, "green": 255.0, "blue": 255.0, "match_score": 0.45},
        ],
    ),

    # 4. hplc -- 2 params (flow_rate, gradient), objective: resolution
    "hplc": OlympusSurface(
        name="hplc",
        param_names=["flow_rate", "gradient"],
        param_bounds=[(0.1, 5.0), (1.0, 100.0)],
        objective_name="resolution",
        lookup_table=[
            {"flow_rate": 0.5, "gradient": 10.0, "resolution": 1.2},
            {"flow_rate": 1.0, "gradient": 20.0, "resolution": 2.5},
            {"flow_rate": 1.5, "gradient": 40.0, "resolution": 3.8},
            {"flow_rate": 2.0, "gradient": 60.0, "resolution": 4.1},
            {"flow_rate": 3.0, "gradient": 80.0, "resolution": 3.2},
            {"flow_rate": 4.0, "gradient": 90.0, "resolution": 2.0},
            {"flow_rate": 5.0, "gradient": 100.0, "resolution": 1.5},
        ],
    ),

    # 5. benzylation -- 2 params (temperature, equiv), objective: yield
    "benzylation": OlympusSurface(
        name="benzylation",
        param_names=["temperature", "equiv"],
        param_bounds=[(20.0, 150.0), (0.5, 5.0)],
        objective_name="yield",
        lookup_table=[
            {"temperature": 25.0, "equiv": 1.0, "yield": 15.0},
            {"temperature": 50.0, "equiv": 1.5, "yield": 35.0},
            {"temperature": 75.0, "equiv": 2.0, "yield": 58.0},
            {"temperature": 100.0, "equiv": 2.5, "yield": 72.0},
            {"temperature": 120.0, "equiv": 3.0, "yield": 85.0},
            {"temperature": 140.0, "equiv": 4.0, "yield": 78.0},
            {"temperature": 150.0, "equiv": 5.0, "yield": 62.0},
            {"temperature": 85.0, "equiv": 2.8, "yield": 91.0},
        ],
    ),

    # 6. suzuki -- 3 params (temperature, catalyst_loading, base_equiv),
    #    objective: yield
    "suzuki": OlympusSurface(
        name="suzuki",
        param_names=["temperature", "catalyst_loading", "base_equiv"],
        param_bounds=[(25.0, 150.0), (0.01, 0.20), (1.0, 5.0)],
        objective_name="yield",
        lookup_table=[
            {"temperature": 40.0, "catalyst_loading": 0.02,
             "base_equiv": 1.5, "yield": 22.0},
            {"temperature": 60.0, "catalyst_loading": 0.05,
             "base_equiv": 2.0, "yield": 45.0},
            {"temperature": 80.0, "catalyst_loading": 0.08,
             "base_equiv": 2.5, "yield": 68.0},
            {"temperature": 100.0, "catalyst_loading": 0.10,
             "base_equiv": 3.0, "yield": 82.0},
            {"temperature": 120.0, "catalyst_loading": 0.12,
             "base_equiv": 3.5, "yield": 91.0},
            {"temperature": 130.0, "catalyst_loading": 0.15,
             "base_equiv": 4.0, "yield": 88.0},
            {"temperature": 150.0, "catalyst_loading": 0.20,
             "base_equiv": 5.0, "yield": 70.0},
            {"temperature": 110.0, "catalyst_loading": 0.11,
             "base_equiv": 3.2, "yield": 95.0},
        ],
    ),
}

"""Auto Benchmark Generator -- synthetic benchmark generation for optimization-copilot."""

from __future__ import annotations

from optimization_copilot.benchmark_generator.generator import (
    BenchmarkGenerator,
    BenchmarkSpec,
    LandscapeType,
    SyntheticObjective,
)

__all__ = [
    "LandscapeType",
    "SyntheticObjective",
    "BenchmarkSpec",
    "BenchmarkGenerator",
]

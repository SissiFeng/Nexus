"""Benchmark module for standardized comparison of optimization backends.

Provides ``BenchmarkRunner`` for backend comparison and ``BenchmarkFunction``
definitions for standard test problems.
"""

from optimization_copilot.benchmark.functions import (
    BENCHMARK_SUITE,
    BenchmarkFunction,
    get_benchmark,
    list_benchmarks,
    make_spec,
)
from optimization_copilot.benchmark.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    Leaderboard,
    LeaderboardEntry,
)

__all__ = [
    "BENCHMARK_SUITE",
    "BenchmarkFunction",
    "BenchmarkResult",
    "BenchmarkRunner",
    "Leaderboard",
    "LeaderboardEntry",
    "get_benchmark",
    "list_benchmarks",
    "make_spec",
]

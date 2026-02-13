"""Benchmark module for standardized comparison of optimization backends.

Provides ``BenchmarkRunner`` for backend comparison, ``BenchmarkFunction``
definitions for standard test problems, ``DirectBenchmarkRunner`` for
closed-loop evaluation, and ``SystematicEvaluator`` for multi-seed
statistical benchmarking.
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
from optimization_copilot.benchmark.direct_runner import (
    DirectBenchmarkResult,
    DirectBenchmarkRunner,
)
from optimization_copilot.benchmark.systematic_eval import (
    BackendRanking,
    EvaluationConfig,
    EvaluationReport,
    SystematicEvaluator,
    WilcoxonResult,
    generate_report,
    wilcoxon_signed_rank,
)
from optimization_copilot.benchmark.meta_adapter import MetaControllerAdapter
from optimization_copilot.benchmark.tabular_data import TabularBenchmark

__all__ = [
    "BENCHMARK_SUITE",
    "BackendRanking",
    "BenchmarkFunction",
    "BenchmarkResult",
    "BenchmarkRunner",
    "DirectBenchmarkResult",
    "DirectBenchmarkRunner",
    "EvaluationConfig",
    "EvaluationReport",
    "Leaderboard",
    "LeaderboardEntry",
    "MetaControllerAdapter",
    "SystematicEvaluator",
    "TabularBenchmark",
    "WilcoxonResult",
    "generate_report",
    "get_benchmark",
    "list_benchmarks",
    "make_spec",
    "wilcoxon_signed_rank",
]

"""Result aggregation and ranking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .protocol import BenchmarkResult


@dataclass
class LeaderboardEntry:
    """A single entry in the leaderboard ranking."""

    rank: int
    algorithm_name: str
    best_value: float
    n_evaluations: int
    total_cost: float
    wall_time_seconds: float


class Leaderboard:
    """Aggregates benchmark results and produces rankings."""

    def __init__(self, benchmark_name: str, direction: str = "minimize") -> None:
        """Initialize the leaderboard.

        Args:
            benchmark_name: Name of the benchmark being ranked.
            direction: Optimization direction, 'minimize' or 'maximize'.
        """
        if direction not in ("minimize", "maximize"):
            raise ValueError(
                f"Invalid direction '{direction}'. Must be 'minimize' or 'maximize'."
            )
        self._benchmark_name = benchmark_name
        self._direction = direction
        self._results: list[BenchmarkResult] = []

    @property
    def benchmark_name(self) -> str:
        """Return the benchmark name."""
        return self._benchmark_name

    @property
    def direction(self) -> str:
        """Return the optimization direction."""
        return self._direction

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the leaderboard.

        Args:
            result: The benchmark result to add.

        Raises:
            ValueError: If the result is for a different benchmark.
        """
        if result.benchmark_name != self._benchmark_name:
            raise ValueError(
                f"Result benchmark '{result.benchmark_name}' does not match "
                f"leaderboard benchmark '{self._benchmark_name}'"
            )
        self._results.append(result)

    def get_rankings(self) -> list[LeaderboardEntry]:
        """Get ranked leaderboard entries.

        Returns:
            List of LeaderboardEntry sorted by best_value
            (ascending for minimize, descending for maximize).
        """
        if not self._results:
            return []

        reverse = self._direction == "maximize"
        sorted_results = sorted(
            self._results,
            key=lambda r: r.best_value,
            reverse=reverse,
        )

        entries = []
        for i, result in enumerate(sorted_results):
            entries.append(
                LeaderboardEntry(
                    rank=i + 1,
                    algorithm_name=result.algorithm_name,
                    best_value=result.best_value,
                    n_evaluations=result.n_evaluations,
                    total_cost=result.total_cost,
                    wall_time_seconds=result.wall_time_seconds,
                )
            )
        return entries

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics about all results.

        Returns:
            Dictionary with summary statistics.
        """
        if not self._results:
            return {
                "benchmark_name": self._benchmark_name,
                "direction": self._direction,
                "n_entries": 0,
                "best_value": None,
                "worst_value": None,
                "mean_value": None,
                "mean_evaluations": None,
                "mean_cost": None,
                "mean_wall_time": None,
            }

        values = [r.best_value for r in self._results]
        n_evals = [r.n_evaluations for r in self._results]
        costs = [r.total_cost for r in self._results]
        wall_times = [r.wall_time_seconds for r in self._results]

        if self._direction == "minimize":
            best_val = min(values)
            worst_val = max(values)
        else:
            best_val = max(values)
            worst_val = min(values)

        return {
            "benchmark_name": self._benchmark_name,
            "direction": self._direction,
            "n_entries": len(self._results),
            "best_value": best_val,
            "worst_value": worst_val,
            "mean_value": sum(values) / len(values),
            "mean_evaluations": sum(n_evals) / len(n_evals),
            "mean_cost": sum(costs) / len(costs),
            "mean_wall_time": sum(wall_times) / len(wall_times),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize leaderboard to dictionary.

        Returns:
            Dictionary representation of the leaderboard.
        """
        rankings = self.get_rankings()
        return {
            "benchmark_name": self._benchmark_name,
            "direction": self._direction,
            "rankings": [
                {
                    "rank": e.rank,
                    "algorithm_name": e.algorithm_name,
                    "best_value": e.best_value,
                    "n_evaluations": e.n_evaluations,
                    "total_cost": e.total_cost,
                    "wall_time_seconds": e.wall_time_seconds,
                }
                for e in rankings
            ],
            "summary": self.get_summary(),
        }

    def render_text(self) -> str:
        """Render the leaderboard as a pretty-printed text table.

        Returns:
            Formatted string table of rankings.
        """
        rankings = self.get_rankings()
        if not rankings:
            return f"Leaderboard: {self._benchmark_name} (no entries)"

        # Header
        header = (
            f"Leaderboard: {self._benchmark_name} "
            f"(direction: {self._direction})\n"
        )
        sep = "-" * 80 + "\n"
        col_header = (
            f"{'Rank':>4}  {'Algorithm':<25}  {'Best Value':>12}  "
            f"{'Evals':>6}  {'Cost':>8}  {'Time (s)':>10}\n"
        )

        rows = []
        for e in rankings:
            row = (
                f"{e.rank:>4}  {e.algorithm_name:<25}  {e.best_value:>12.6f}  "
                f"{e.n_evaluations:>6}  {e.total_cost:>8.2f}  "
                f"{e.wall_time_seconds:>10.4f}\n"
            )
            rows.append(row)

        return header + sep + col_header + sep + "".join(rows) + sep

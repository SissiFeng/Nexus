"""Comparison and verification reports for deterministic replay.

Provides dataclasses for structured comparison between replay runs:
- IterationComparison: per-iteration diff between two runs
- ReplayVerification: result of verifying a log against a fresh replay
- ComparisonReport: side-by-side comparison of two recorded logs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# IterationComparison
# ---------------------------------------------------------------------------


@dataclass
class IterationComparison:
    """Per-iteration comparison between two runs."""

    iteration: int
    decisions_match: bool
    snapshot_hashes_match: bool
    diagnostics_hashes_match: bool
    decision_hashes_match: bool
    run_a: dict[str, Any]
    run_b: dict[str, Any]
    differences: list[str]


# ---------------------------------------------------------------------------
# ReplayVerification
# ---------------------------------------------------------------------------


@dataclass
class ReplayVerification:
    """Result of verifying a decision log via deterministic replay.

    ``verified`` is True only when every iteration produced identical
    hashes during re-execution.
    """

    verified: bool
    n_iterations: int
    n_mismatches: int
    first_mismatch_iteration: int | None
    mismatched_iterations: list[int]
    details: list[IterationComparison]

    def summary(self) -> str:
        """Return a human-readable verification summary."""
        lines: list[str] = []
        if self.verified:
            lines.append(
                f"VERIFIED: all {self.n_iterations} iterations reproduced exactly."
            )
        else:
            lines.append(
                f"MISMATCH: {self.n_mismatches} of {self.n_iterations} iterations "
                f"diverged."
            )
            if self.first_mismatch_iteration is not None:
                lines.append(
                    f"  First mismatch at iteration {self.first_mismatch_iteration}."
                )
            if self.mismatched_iterations:
                lines.append(
                    f"  Mismatched iterations: {self.mismatched_iterations}"
                )

        # Detailed differences for mismatched iterations
        for detail in self.details:
            if detail.differences:
                lines.append(f"  Iteration {detail.iteration}:")
                for diff in detail.differences:
                    lines.append(f"    - {diff}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ComparisonReport
# ---------------------------------------------------------------------------


@dataclass
class ComparisonReport:
    """Side-by-side comparison of two recorded decision logs.

    Useful for comparing two different runs (different seeds, backends,
    or what-if branches) without re-executing the pipeline.
    """

    run_a_id: str
    run_b_id: str
    n_iterations: int
    divergence_iteration: int | None
    iteration_comparisons: list[IterationComparison]
    run_a_best_kpi_curve: list[float]
    run_b_best_kpi_curve: list[float]
    run_a_phases: list[tuple[int, str]]
    run_b_phases: list[tuple[int, str]]
    run_a_final_kpi: dict[str, float]
    run_b_final_kpi: dict[str, float]
    n_decision_differences: int
    n_backend_differences: int
    n_phase_differences: int

    def summary(self) -> str:
        """Return a human-readable comparison summary."""
        lines: list[str] = [
            f"Comparison: {self.run_a_id} vs {self.run_b_id}",
            f"  Iterations compared: {self.n_iterations}",
        ]

        if self.divergence_iteration is not None:
            lines.append(
                f"  First divergence at iteration: {self.divergence_iteration}"
            )
        else:
            lines.append("  No divergence detected (runs are identical).")

        lines.append(f"  Decision differences: {self.n_decision_differences}")
        lines.append(f"  Backend differences: {self.n_backend_differences}")
        lines.append(f"  Phase differences: {self.n_phase_differences}")

        # Final KPI comparison
        if self.run_a_final_kpi or self.run_b_final_kpi:
            lines.append("  Final KPIs:")
            all_keys = sorted(
                set(self.run_a_final_kpi) | set(self.run_b_final_kpi)
            )
            for key in all_keys:
                val_a = self.run_a_final_kpi.get(key)
                val_b = self.run_b_final_kpi.get(key)
                lines.append(f"    {key}: run_a={val_a}, run_b={val_b}")

        # Convergence curve summary
        if self.run_a_best_kpi_curve and self.run_b_best_kpi_curve:
            lines.append(
                f"  Final best KPI: run_a={self.run_a_best_kpi_curve[-1]:.6f}, "
                f"run_b={self.run_b_best_kpi_curve[-1]:.6f}"
            )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "run_a_id": self.run_a_id,
            "run_b_id": self.run_b_id,
            "n_iterations": self.n_iterations,
            "divergence_iteration": self.divergence_iteration,
            "iteration_comparisons": [
                {
                    "iteration": ic.iteration,
                    "decisions_match": ic.decisions_match,
                    "snapshot_hashes_match": ic.snapshot_hashes_match,
                    "diagnostics_hashes_match": ic.diagnostics_hashes_match,
                    "decision_hashes_match": ic.decision_hashes_match,
                    "run_a": ic.run_a,
                    "run_b": ic.run_b,
                    "differences": ic.differences,
                }
                for ic in self.iteration_comparisons
            ],
            "run_a_best_kpi_curve": self.run_a_best_kpi_curve,
            "run_b_best_kpi_curve": self.run_b_best_kpi_curve,
            "run_a_phases": self.run_a_phases,
            "run_b_phases": self.run_b_phases,
            "run_a_final_kpi": self.run_a_final_kpi,
            "run_b_final_kpi": self.run_b_final_kpi,
            "n_decision_differences": self.n_decision_differences,
            "n_backend_differences": self.n_backend_differences,
            "n_phase_differences": self.n_phase_differences,
        }

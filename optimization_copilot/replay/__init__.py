"""Deterministic replay engine for optimization campaign audit and analysis.

Provides three core capabilities:

* **Decision logging** -- Record every iteration's pipeline state for
  reproducibility audits.
* **Replay verification** -- Re-run a recorded campaign and verify that
  every decision is hash-identical.
* **What-if analysis** -- Branch from a recorded campaign with different
  seeds, backends, or evaluation functions to explore counterfactual
  scenarios.
"""

from optimization_copilot.replay.engine import ReplayEngine, ReplayMode
from optimization_copilot.replay.log import DecisionLog, DecisionLogEntry
from optimization_copilot.replay.report import (
    ComparisonReport,
    IterationComparison,
    ReplayVerification,
)

__all__ = [
    "ComparisonReport",
    "DecisionLog",
    "DecisionLogEntry",
    "IterationComparison",
    "ReplayEngine",
    "ReplayMode",
    "ReplayVerification",
]

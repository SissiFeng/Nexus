"""Hypothesis Engine (Layer 3).

Generates, tests, and tracks competing scientific hypotheses from
multiple analysis sources including symbolic regression, causal graphs,
fANOVA importance decompositions, and correlation analysis.
"""

from __future__ import annotations

from optimization_copilot.hypothesis.generator import HypothesisGenerator
from optimization_copilot.hypothesis.models import (
    Evidence,
    Hypothesis,
    HypothesisStatus,
    Prediction,
)
from optimization_copilot.hypothesis.testing import HypothesisTester
from optimization_copilot.hypothesis.tracker import HypothesisTracker

__all__ = [
    "Hypothesis",
    "HypothesisStatus",
    "Prediction",
    "Evidence",
    "HypothesisGenerator",
    "HypothesisTester",
    "HypothesisTracker",
]

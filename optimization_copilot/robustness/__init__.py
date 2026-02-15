"""Layer 4: Decision Robustness Analysis.

Bootstrap confidence intervals, conclusion stability checking,
decision sensitivity analysis, and cross-model consistency.
"""

from optimization_copilot.robustness.models import (
    BootstrapResult,
    ConclusionRobustness,
    RobustnessReport,
)
from optimization_copilot.robustness.bootstrap import BootstrapAnalyzer
from optimization_copilot.robustness.conclusion import ConclusionRobustnessChecker
from optimization_copilot.robustness.sensitivity import DecisionSensitivityAnalyzer
from optimization_copilot.robustness.consistency import CrossModelConsistency

__all__ = [
    "BootstrapResult",
    "ConclusionRobustness",
    "RobustnessReport",
    "BootstrapAnalyzer",
    "ConclusionRobustnessChecker",
    "DecisionSensitivityAnalyzer",
    "CrossModelConsistency",
]

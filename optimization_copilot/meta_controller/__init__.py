"""Meta-Controller: intelligent strategy selection and phase orchestration.

Provides algorithm selection, phase management, and intelligent switching
based on data characteristics.
"""

from __future__ import annotations

from optimization_copilot.meta_controller.controller import (
    MetaController,
    SwitchingThresholds,
)
from optimization_copilot.meta_controller.intelligent_selector import (
    IntelligentAlgorithmSelector,
    AlgorithmRecommendation,
    AlgorithmSwitchExplanation,
)

__all__ = [
    "MetaController",
    "SwitchingThresholds",
    "IntelligentAlgorithmSelector",
    "AlgorithmRecommendation",
    "AlgorithmSwitchExplanation",
]

"""Multi-fidelity optimization: cheap simulation → expensive experiments.

Provides planning and execution of multi-stage evaluation with
adaptive promotion strategies.
"""

from __future__ import annotations

from optimization_copilot.multi_fidelity.planner import (
    FidelityLevel,
    FidelityPlan,
    MultiFidelityPlan,
    MultiFidelityPlanner,
)
from optimization_copilot.multi_fidelity.enhanced_planner import (
    AdaptiveFidelityState,
    EnhancedMultiFidelityOptimizer,
    FidelityGateResult,
)

__all__ = [
    "FidelityLevel",
    "FidelityPlan",
    "MultiFidelityPlan",
    "MultiFidelityPlanner",
    "EnhancedMultiFidelityOptimizer",
    "AdaptiveFidelityState",
    "FidelityGateResult",
]

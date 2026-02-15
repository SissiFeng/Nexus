"""Human-in-the-loop (HITL) optimization support.

Provides expert prior injection, progressive autonomy control,
and interactive steering for optimization campaigns.
"""

from optimization_copilot.hitl.autonomy import (
    AutonomyLevel,
    AutonomyPolicy,
    TrustTracker,
)
from optimization_copilot.hitl.priors import (
    ExpertPrior,
    PriorRegistry,
    PriorType,
)
from optimization_copilot.hitl.steering import (
    SteeringAction,
    SteeringDirective,
    SteeringEngine,
)

__all__ = [
    "AutonomyLevel",
    "AutonomyPolicy",
    "ExpertPrior",
    "PriorRegistry",
    "PriorType",
    "SteeringAction",
    "SteeringDirective",
    "SteeringEngine",
    "TrustTracker",
]

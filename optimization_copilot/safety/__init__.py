"""Safety monitoring and emergency protocols for optimization experiments.

Provides hazard classification, real-time safety monitoring, and emergency
stop protocols to ensure self-driving lab experiments operate within safe
boundaries.

Three-layer safety architecture:
1. **Hazard classification** — Define safe operating ranges per parameter
2. **Real-time monitoring** — Evaluate points and generate safety events
3. **Emergency protocols** — Decide on CONTINUE / PAUSE / FALLBACK / STOP
"""

from optimization_copilot.safety.emergency import (
    EmergencyAction,
    EmergencyEvaluation,
    EmergencyLog,
    EmergencyProtocol,
)
from optimization_copilot.safety.hazards import (
    HazardCategory,
    HazardLevel,
    HazardRegistry,
    HazardSpec,
)
from optimization_copilot.safety.monitor import (
    SafetyEvent,
    SafetyMonitor,
    SafetyStatus,
)

__all__ = [
    # hazards
    "HazardCategory",
    "HazardLevel",
    "HazardRegistry",
    "HazardSpec",
    # monitor
    "SafetyEvent",
    "SafetyMonitor",
    "SafetyStatus",
    # emergency
    "EmergencyAction",
    "EmergencyEvaluation",
    "EmergencyLog",
    "EmergencyProtocol",
]

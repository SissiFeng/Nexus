"""Closed-loop campaign engine for actionable optimization.

Provides the three engineering components for iterative experimental
optimization:

1. **Surrogate model** — GP fitted on SMILES fingerprints
2. **Acquisition + ranking** — candidate prioritization via EI/UCB/PI
3. **Data return interface** — ingest results, update model, report learnings

Output is structured as three layers:

- **Layer 1 (Dashboard)**: ranked candidate table + stage gate protocol
- **Layer 2 (Intelligence)**: Pareto front, model metrics, learning report
- **Layer 3 (Reasoning)**: diagnostics, fANOVA, execution traces
"""

from optimization_copilot.campaign.surrogate import (
    FingerprintSurrogate,
    PredictionResult,
    SurrogateFitResult,
)
from optimization_copilot.campaign.ranker import (
    CandidateRanker,
    RankedCandidate,
    RankedTable,
)
from optimization_copilot.campaign.stage_gate import (
    ProtocolStep,
    ScreeningProtocol,
    StageGateProtocol,
)
from optimization_copilot.campaign.output import (
    CampaignDeliverable,
    Layer1Dashboard,
    Layer2Intelligence,
    Layer3Reasoning,
    LearningReport,
    ModelMetrics,
)
from optimization_copilot.campaign.loop import CampaignLoop

__all__ = [
    # surrogate
    "FingerprintSurrogate",
    "PredictionResult",
    "SurrogateFitResult",
    # ranker
    "CandidateRanker",
    "RankedCandidate",
    "RankedTable",
    # stage_gate
    "ProtocolStep",
    "ScreeningProtocol",
    "StageGateProtocol",
    # output
    "CampaignDeliverable",
    "Layer1Dashboard",
    "Layer2Intelligence",
    "Layer3Reasoning",
    "LearningReport",
    "ModelMetrics",
    # loop
    "CampaignLoop",
]

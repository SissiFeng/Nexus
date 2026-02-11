"""Infrastructure modules for optimization platform v2."""

from optimization_copilot.infrastructure.auto_sampler import AutoSampler, SelectionResult
from optimization_copilot.infrastructure.batch_scheduler import (
    AsyncTrial,
    BatchScheduler,
    TrialStatus,
)
from optimization_copilot.infrastructure.constraint_engine import (
    Constraint,
    ConstraintEngine,
    ConstraintEvaluation,
    ConstraintStatus,
    ConstraintType,
)
from optimization_copilot.infrastructure.cost_tracker import CostTracker, TrialCost
from optimization_copilot.infrastructure.domain_encoding import (
    CustomDescriptorEncoding,
    Encoding,
    EncodingPipeline,
    OneHotEncoding,
    OrdinalEncoding,
    SpatialEncoding,
)
from optimization_copilot.infrastructure.multi_fidelity import (
    FidelityLevel,
    MultiFidelityManager,
)
from optimization_copilot.infrastructure.parameter_importance import (
    ImportanceResult,
    ParameterImportanceAnalyzer,
)
from optimization_copilot.infrastructure.robust_optimizer import RobustOptimizer
from optimization_copilot.infrastructure.stopping_rule import StoppingDecision, StoppingRule
from optimization_copilot.infrastructure.transfer_learning import (
    CampaignData,
    TransferLearningEngine,
)

__all__ = [
    "AsyncTrial",
    "AutoSampler",
    "BatchScheduler",
    "CampaignData",
    "Constraint",
    "ConstraintEngine",
    "ConstraintEvaluation",
    "ConstraintStatus",
    "ConstraintType",
    "CostTracker",
    "CustomDescriptorEncoding",
    "Encoding",
    "EncodingPipeline",
    "FidelityLevel",
    "ImportanceResult",
    "MultiFidelityManager",
    "OneHotEncoding",
    "OrdinalEncoding",
    "ParameterImportanceAnalyzer",
    "RobustOptimizer",
    "SelectionResult",
    "SpatialEncoding",
    "StoppingDecision",
    "StoppingRule",
    "TransferLearningEngine",
    "TrialCost",
    "TrialStatus",
]

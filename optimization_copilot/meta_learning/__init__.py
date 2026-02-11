"""Meta-learning optimizer selector — cross-campaign learning for strategy selection."""

from optimization_copilot.meta_learning.models import (
    BackendPerformance,
    CampaignOutcome,
    DriftRobustness,
    ExperienceRecord,
    FailureStrategy,
    LearnedThresholds,
    LearnedWeights,
    MetaAdvice,
    MetaLearningConfig,
)
from optimization_copilot.meta_learning.experience_store import ExperienceStore
from optimization_copilot.meta_learning.strategy_learner import StrategyLearner
from optimization_copilot.meta_learning.weight_tuner import WeightTuner
from optimization_copilot.meta_learning.threshold_learner import ThresholdLearner
from optimization_copilot.meta_learning.failure_learner import FailureStrategyLearner
from optimization_copilot.meta_learning.drift_learner import DriftRobustnessTracker
from optimization_copilot.meta_learning.advisor import MetaLearningAdvisor

__all__ = [
    # Models
    "BackendPerformance",
    "CampaignOutcome",
    "DriftRobustness",
    "ExperienceRecord",
    "FailureStrategy",
    "LearnedThresholds",
    "LearnedWeights",
    "MetaAdvice",
    "MetaLearningConfig",
    # Components
    "ExperienceStore",
    "StrategyLearner",
    "WeightTuner",
    "ThresholdLearner",
    "FailureStrategyLearner",
    "DriftRobustnessTracker",
    "MetaLearningAdvisor",
]

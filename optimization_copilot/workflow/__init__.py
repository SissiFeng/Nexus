"""Multi-stage experimental workflow package for optimization."""

from optimization_copilot.workflow.stage import ExperimentStage, StageDAG
from optimization_copilot.workflow.proxy_model import ProxyModel
from optimization_copilot.workflow.continue_value import ContinueValue
from optimization_copilot.workflow.multi_stage_bo import MultiStageBayesianOptimizer
from optimization_copilot.workflow.fidelity_graph import (
    CandidateTrajectory,
    FidelityGraph,
    FidelityStage,
    GateCondition,
    GateDecision,
    StageGate,
)
from optimization_copilot.workflow.simulator import (
    FidelitySimulator,
    PerGateStats,
    SimulationResult,
)

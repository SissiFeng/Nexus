"""Causal Discovery Engine (Layer 1).

Provides causal graph construction, structure learning (PC algorithm),
interventional reasoning (do-calculus), causal effect estimation, and
counterfactual reasoning via structural causal models.
"""

from __future__ import annotations

from optimization_copilot.causal.counterfactual import CounterfactualReasoner
from optimization_copilot.causal.effects import CausalEffectEstimator
from optimization_copilot.causal.interventional import InterventionalEngine
from optimization_copilot.causal.metrics import (
    StructureRecoveryMetrics,
    edge_precision_recall,
    evaluate_structure_recovery,
    orientation_accuracy,
    structural_hamming_distance,
)
from optimization_copilot.causal.models import CausalEdge, CausalGraph, CausalNode
from optimization_copilot.causal.optimizer_integration import (
    CausalOptimizationAnalyzer,
    CausalOptimizationInsight,
    VariableCausalImpact,
)
from optimization_copilot.causal.structure import CausalStructureLearner

__all__ = [
    "CausalGraph",
    "CausalNode",
    "CausalEdge",
    "CausalStructureLearner",
    "InterventionalEngine",
    "CausalEffectEstimator",
    "CounterfactualReasoner",
    "StructureRecoveryMetrics",
    "structural_hamming_distance",
    "edge_precision_recall",
    "orientation_accuracy",
    "evaluate_structure_recovery",
    "CausalOptimizationAnalyzer",
    "CausalOptimizationInsight",
    "VariableCausalImpact",
]

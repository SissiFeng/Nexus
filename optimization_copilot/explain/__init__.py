"""v7b Explainable Optimization — interaction maps, physics kernels, symbolic regression."""

from optimization_copilot.explain.interaction_map import InteractionMap
from optimization_copilot.explain.physics_kernels import (
    SaturationKernel,
    InteractionKernel,
    PhysicsKernelFactory,
)
from optimization_copilot.explain.equation_discovery import EquationDiscovery
from optimization_copilot.explain.report_generator import InsightReportGenerator

__all__ = [
    "InteractionMap",
    "SaturationKernel",
    "InteractionKernel",
    "PhysicsKernelFactory",
    "EquationDiscovery",
    "InsightReportGenerator",
]

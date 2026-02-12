"""Theory-Data Hybrid Engine (Layer 5).

Combines deterministic theory models with data-driven Gaussian
Process residual correction for physics-informed Bayesian
optimization.

Modules
-------
theory
    Abstract base and concrete theory models (Arrhenius,
    Michaelis-Menten, power-law, ODE).
residual
    Residual GP fitted on theory-data discrepancy.
composite
    HybridModel combining theory + residual GP.
discrepancy
    DiscrepancyAnalyzer for failure region detection and
    theory revision suggestions.
"""

from __future__ import annotations

from optimization_copilot.hybrid.theory import (
    ArrheniusModel,
    MichaelisMentenModel,
    ODEModel,
    PowerLawModel,
    TheoryModel,
)
from optimization_copilot.hybrid.residual import ResidualGP
from optimization_copilot.hybrid.composite import HybridModel
from optimization_copilot.hybrid.discrepancy import DiscrepancyAnalyzer

__all__ = [
    # theory
    "TheoryModel",
    "ArrheniusModel",
    "MichaelisMentenModel",
    "PowerLawModel",
    "ODEModel",
    # residual
    "ResidualGP",
    # composite
    "HybridModel",
    # discrepancy
    "DiscrepancyAnalyzer",
]

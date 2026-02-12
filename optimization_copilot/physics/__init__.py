"""Physics-informed modeling layer for optimization.

Provides physics-based kernels, prior mean functions, ODE solvers,
and constraint models that encode domain knowledge into the
optimization loop.
"""

from __future__ import annotations

from optimization_copilot.physics.constraints import (
    ConservationLaw,
    MonotonicityConstraint,
    PhysicsBound,
    PhysicsConstraintModel,
)
from optimization_copilot.physics.kernels import (
    CompositeKernel,
    LinearKernel,
    PeriodicKernel,
    linear_kernel,
    linear_kernel_matrix,
    periodic_kernel,
    periodic_kernel_matrix,
    symmetry_kernel,
)
from optimization_copilot.physics.ode_solver import RK4Solver
from optimization_copilot.physics.priors import (
    ArrheniusPrior,
    MichaelisMentenPrior,
    PowerLawPrior,
)

__all__ = [
    # kernels
    "PeriodicKernel",
    "LinearKernel",
    "CompositeKernel",
    "periodic_kernel",
    "linear_kernel",
    "symmetry_kernel",
    "periodic_kernel_matrix",
    "linear_kernel_matrix",
    # priors
    "ArrheniusPrior",
    "MichaelisMentenPrior",
    "PowerLawPrior",
    # ode
    "RK4Solver",
    # constraints
    "ConservationLaw",
    "MonotonicityConstraint",
    "PhysicsBound",
    "PhysicsConstraintModel",
]

"""Pure-Python math utilities for optimization backends.

Re-exports all public functions from sub-modules for convenience::

    from optimization_copilot.backends._math import vec_dot, cholesky, norm_cdf
"""

from optimization_copilot.backends._math.linalg import (
    cholesky,
    determinant,
    eigen_symmetric,
    identity,
    mat_add,
    mat_inv,
    mat_mul,
    mat_scale,
    mat_vec,
    outer_product,
    solve_cholesky,
    solve_lower,
    solve_upper,
    transpose,
    vec_dot,
)
from optimization_copilot.backends._math.stats import (
    binary_entropy,
    norm_cdf,
    norm_logpdf,
    norm_pdf,
    norm_ppf,
)
from optimization_copilot.backends._math.sobol import (
    SOBOL_DIRECTION_NUMBERS,
    sobol_sequence,
)
from optimization_copilot.backends._math.kernels import (
    distance_matrix,
    kernel_matrix,
    matern52_kernel,
    rbf_kernel,
)
from optimization_copilot.backends._math.acquisition import (
    expected_improvement,
    log_expected_improvement_per_cost,
    probability_of_improvement,
    upper_confidence_bound,
)

__all__ = [
    # linalg
    "vec_dot",
    "mat_mul",
    "mat_vec",
    "cholesky",
    "solve_lower",
    "solve_upper",
    "solve_cholesky",
    "transpose",
    "identity",
    "mat_add",
    "mat_scale",
    "outer_product",
    "mat_inv",
    "determinant",
    "eigen_symmetric",
    # stats
    "norm_pdf",
    "norm_cdf",
    "norm_ppf",
    "norm_logpdf",
    "binary_entropy",
    # sobol
    "SOBOL_DIRECTION_NUMBERS",
    "sobol_sequence",
    # kernels
    "rbf_kernel",
    "matern52_kernel",
    "distance_matrix",
    "kernel_matrix",
    # acquisition
    "expected_improvement",
    "upper_confidence_bound",
    "probability_of_improvement",
    "log_expected_improvement_per_cost",
]

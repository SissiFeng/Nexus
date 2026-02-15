"""Domain-specific Gaussian process kernels for scientific optimisation.

Provides kernels that encode physical priors such as saturation behaviour
(e.g. coulombic efficiency near 100 %) and parameter interactions
(e.g. catalyst composition effects).
"""

from __future__ import annotations

import math

from optimization_copilot.backends._math.kernels import kernel_matrix


# ---------------------------------------------------------------------------
# SaturationKernel
# ---------------------------------------------------------------------------

class SaturationKernel:
    """Matern 1/2 kernel with input saturation warping.

    For parameters that asymptotically approach a ceiling (e.g. CE near
    100 %), the warping ``w(x) = s * tanh(steepness * x / s)`` compresses
    the input space near the saturation point so the GP naturally slows
    its predictions near the limit.

    Parameters
    ----------
    length_scale : float
        Length-scale of the Matern 1/2 base kernel.
    saturation_point : float
        Asymptotic ceiling in the warped space.
    steepness : float
        Controls how quickly the warp approaches saturation.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        saturation_point: float = 1.0,
        steepness: float = 5.0,
    ) -> None:
        self.length_scale = length_scale
        self.saturation_point = saturation_point
        self.steepness = steepness

    def _warp(self, x: list[float]) -> list[float]:
        """Apply saturation warping to an input vector."""
        s = max(self.saturation_point, 1e-12)
        return [s * math.tanh(self.steepness * xi / s) for xi in x]

    def __call__(self, x1: list[float], x2: list[float]) -> float:
        """Evaluate the warped Matern 1/2 kernel.

        Parameters
        ----------
        x1, x2 : list[float]
            Input vectors of equal length.

        Returns
        -------
        float
            Kernel value ``exp(-r / length_scale)`` in warped space.
        """
        x1_w = self._warp(x1)
        x2_w = self._warp(x2)
        r = math.sqrt(sum((a - b) ** 2 for a, b in zip(x1_w, x2_w)))
        return math.exp(-r / max(self.length_scale, 1e-12))

    def matrix(self, X: list[list[float]], noise: float = 1e-4) -> list[list[float]]:
        """Build the full kernel matrix with optional diagonal noise.

        Parameters
        ----------
        X : list[list[float]]
            Input data, shape ``(n, d)``.
        noise : float
            Diagonal noise variance.

        Returns
        -------
        list[list[float]]
            Positive-definite kernel matrix.
        """
        return kernel_matrix(X, self.__call__, noise)

    def __repr__(self) -> str:
        return (
            f"SaturationKernel(length_scale={self.length_scale}, "
            f"saturation_point={self.saturation_point}, "
            f"steepness={self.steepness})"
        )


# ---------------------------------------------------------------------------
# InteractionKernel
# ---------------------------------------------------------------------------

class InteractionKernel:
    """Additive RBF kernel with pairwise interaction terms.

    The kernel decomposes as:

    .. math::

        k(x_1, x_2) = \\sum_i k_i + \\gamma \\sum_{i<j} k_i k_j

    where each ``k_i`` is a 1-D RBF kernel on dimension *i* and
    ``gamma`` controls the interaction strength.

    Parameters
    ----------
    base_length_scales : list[float] | None
        Per-dimension length-scales.  If *None*, all default to 1.0.
    interaction_strength : float
        Multiplicative weight ``gamma`` for the interaction terms.
    """

    def __init__(
        self,
        base_length_scales: list[float] | None = None,
        interaction_strength: float = 0.1,
    ) -> None:
        self.base_length_scales = base_length_scales
        self.interaction_strength = interaction_strength

    def __call__(self, x1: list[float], x2: list[float]) -> float:
        """Evaluate the additive+interaction kernel.

        Parameters
        ----------
        x1, x2 : list[float]
            Input vectors of equal length.

        Returns
        -------
        float
            Kernel value.
        """
        d = len(x1)
        ls = self.base_length_scales if self.base_length_scales else [1.0] * d

        # Per-dimension RBF values
        per_dim: list[float] = []
        for i in range(d):
            diff_sq = (x1[i] - x2[i]) ** 2
            k_i = math.exp(-0.5 * diff_sq / max(ls[i] ** 2, 1e-12))
            per_dim.append(k_i)

        # Main effects (additive)
        main = sum(per_dim)

        # Pairwise interaction terms
        interaction = 0.0
        for i in range(d):
            for j in range(i + 1, d):
                interaction += per_dim[i] * per_dim[j]

        return main + self.interaction_strength * interaction

    def matrix(self, X: list[list[float]], noise: float = 1e-4) -> list[list[float]]:
        """Build the full kernel matrix with optional diagonal noise.

        Parameters
        ----------
        X : list[list[float]]
            Input data, shape ``(n, d)``.
        noise : float
            Diagonal noise variance.

        Returns
        -------
        list[list[float]]
            Positive-definite kernel matrix.
        """
        return kernel_matrix(X, self.__call__, noise)

    def __repr__(self) -> str:
        return (
            f"InteractionKernel(base_length_scales={self.base_length_scales}, "
            f"interaction_strength={self.interaction_strength})"
        )


# ---------------------------------------------------------------------------
# PhysicsKernelFactory
# ---------------------------------------------------------------------------

class PhysicsKernelFactory:
    """Factory for creating domain-specific GP kernels.

    Supported domains:

    - ``"electrochemistry"`` — saturation near CE = 100 %
    - ``"catalysis"`` — strong compositional interactions
    - ``"perovskite"`` — saturation near PCE ~ 35 %
    - anything else — default ``SaturationKernel``
    """

    @staticmethod
    def for_domain(domain: str) -> SaturationKernel | InteractionKernel:
        """Return an appropriate kernel for the given scientific domain.

        Parameters
        ----------
        domain : str
            One of ``"electrochemistry"``, ``"catalysis"``,
            ``"perovskite"``, or any other string (falls back to default).

        Returns
        -------
        SaturationKernel | InteractionKernel
            A callable kernel instance.
        """
        if domain == "electrochemistry":
            return SaturationKernel(saturation_point=100.0, steepness=3.0)
        elif domain == "catalysis":
            return InteractionKernel(interaction_strength=0.2)
        elif domain == "perovskite":
            return SaturationKernel(saturation_point=35.0, steepness=2.0)
        else:
            return SaturationKernel()

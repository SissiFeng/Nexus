"""High-level controller for latent-space optimisation decisions."""

from __future__ import annotations

from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    ParameterSpec,
    ProblemFingerprint,
    VariableType,
)
from optimization_copilot.latent.models import LatentSpace
from optimization_copilot.latent.transform import LatentTransform
from optimization_copilot.plugins.base import AlgorithmPlugin


class LatentOptimizer:
    """Decides when and how to apply latent-space optimisation.

    This controller inspects the campaign snapshot and fingerprint to
    determine whether the problem is high-dimensional enough to benefit
    from PCA-based dimensionality reduction, fits the latent space, and
    wraps an existing algorithm plugin so it operates in the reduced
    space.

    Parameters
    ----------
    min_dim_for_latent:
        Minimum number of numeric parameters before latent optimisation
        is considered.
    min_observations_for_fit:
        Minimum number of successful observations required to fit PCA.
    min_variance_explained:
        Minimum cumulative explained variance for the latent space
        (passed through to :class:`LatentTransform`).
    stagnation_dim_increase:
        Number of extra latent dimensions to add when stagnation is
        detected.
    """

    def __init__(
        self,
        min_dim_for_latent: int = 5,
        min_observations_for_fit: int = 15,
        min_variance_explained: float = 0.8,
        stagnation_dim_increase: int = 1,
    ) -> None:
        self.min_dim_for_latent = min_dim_for_latent
        self.min_observations_for_fit = min_observations_for_fit
        self.min_variance_explained = min_variance_explained
        self.stagnation_dim_increase = stagnation_dim_increase
        self._transform = LatentTransform(
            min_variance_explained=min_variance_explained,
        )

    # -- public API ---------------------------------------------------------

    def should_use_latent(
        self,
        snapshot: CampaignSnapshot,
        fingerprint: ProblemFingerprint,
    ) -> bool:
        """Determine whether the campaign would benefit from latent optimisation.

        Conditions (all must be true):

        1. At least ``min_dim_for_latent`` numeric (non-CATEGORICAL) parameters.
        2. At least ``min_observations_for_fit`` successful observations.
        3. The problem is not purely categorical.
        """
        numeric_count = sum(
            1
            for spec in snapshot.parameter_specs
            if spec.type != VariableType.CATEGORICAL
        )

        if numeric_count < self.min_dim_for_latent:
            return False

        successful_count = len(snapshot.successful_observations)
        if successful_count < self.min_observations_for_fit:
            return False

        # Purely categorical problems cannot use PCA.
        if numeric_count == 0:
            return False

        return True

    def fit(
        self,
        snapshot: CampaignSnapshot,
        seed: int = 42,
    ) -> LatentSpace:
        """Fit a latent space from the campaign snapshot.

        Returns
        -------
        LatentSpace
            The fitted PCA latent space.
        """
        return self._transform.fit(snapshot, seed)

    def adjust_dimensions(
        self,
        latent_space: LatentSpace,
        diagnostics: Any,
        snapshot: CampaignSnapshot,
        seed: int = 42,
    ) -> LatentSpace:
        """Optionally increase latent dimensions when stagnation is detected.

        If ``diagnostics.kpi_plateau_length > 5`` and the current number of
        latent dimensions is below the original dimensionality, the latent
        space is re-fitted with ``max_components`` increased by
        ``stagnation_dim_increase``.

        Parameters
        ----------
        latent_space:
            The current latent space.
        diagnostics:
            A diagnostics vector (or any object with a ``kpi_plateau_length``
            attribute).
        snapshot:
            The current campaign snapshot used for re-fitting.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        LatentSpace
            Either the original latent space (if no adjustment is needed) or
            a newly fitted one with more components.
        """
        plateau_length = getattr(diagnostics, "kpi_plateau_length", 0)

        if (
            plateau_length > 5
            and latent_space.n_components < latent_space.original_dim
        ):
            new_max = latent_space.n_components + self.stagnation_dim_increase
            adjusted_transform = LatentTransform(
                min_variance_explained=self.min_variance_explained,
                max_components=new_max,
            )
            return adjusted_transform.fit(snapshot, seed)

        return latent_space

    def wrap_plugin(
        self,
        plugin: AlgorithmPlugin,
        latent_space: LatentSpace,
        specs: list[ParameterSpec] | None = None,
    ) -> AlgorithmPlugin:
        """Wrap *plugin* so that it operates in the given latent space.

        Parameters
        ----------
        plugin:
            The inner algorithm plugin to wrap.
        latent_space:
            A fitted latent space.
        specs:
            Original parameter specifications for bound clamping and
            discrete rounding during reconstruction.

        Returns
        -------
        AlgorithmPlugin
            A :class:`LatentPlugin` that delegates to *plugin* in latent
            space.
        """
        from optimization_copilot.latent.plugin import LatentPlugin

        return LatentPlugin(plugin, latent_space, self._transform, specs)

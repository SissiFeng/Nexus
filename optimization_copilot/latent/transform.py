"""Pure-stdlib PCA via power iteration for latent space dimensionality reduction."""

from __future__ import annotations

import math
import random
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.latent.models import LatentSpace


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_data_matrix(
    observations: list[Observation],
    param_names: list[str],
) -> list[list[float]]:
    """Build a data matrix from observations.

    Each row is an observation, each column is a parameter value.
    Parameters are looked up by *param_names* ordering.

    Returns
    -------
    list[list[float]]
        Matrix with shape ``(n_observations, n_params)``.
    """
    matrix: list[list[float]] = []
    for obs in observations:
        row: list[float] = []
        for name in param_names:
            val = obs.parameters.get(name, 0.0)
            row.append(float(val))
        matrix.append(row)
    return matrix


def _covariance_matrix(
    data: list[list[float]],
    n_samples: int,
    n_features: int,
) -> list[list[float]]:
    """Compute the sample covariance matrix ``C[i][j] = sum(x_i * x_j) / (n-1)``.

    *data* is assumed to be already centred (zero-mean).

    Parameters
    ----------
    data:
        Centred data matrix of shape ``(n_samples, n_features)``.
    n_samples:
        Number of rows.
    n_features:
        Number of columns.

    Returns
    -------
    list[list[float]]
        Covariance matrix of shape ``(n_features, n_features)``.
    """
    cov: list[list[float]] = [
        [0.0] * n_features for _ in range(n_features)
    ]
    divisor = max(n_samples - 1, 1)
    for i in range(n_features):
        for j in range(i, n_features):
            s = 0.0
            for k in range(n_samples):
                s += data[k][i] * data[k][j]
            val = s / divisor
            cov[i][j] = val
            cov[j][i] = val
    return cov


def _power_iteration_eigendecomp(
    matrix: list[list[float]],
    n_components: int,
    seed: int,
    max_iterations: int,
    tol: float,
) -> tuple[list[float], list[list[float]]]:
    """Compute top-k eigenvalues and eigenvectors via power iteration with deflation.

    Parameters
    ----------
    matrix:
        Symmetric square matrix of shape ``(d, d)``.
    n_components:
        Number of eigenvalue/vector pairs to extract.
    seed:
        Seed for the deterministic random initialisation.
    max_iterations:
        Maximum number of power iteration steps per component.
    tol:
        Convergence tolerance on the eigenvector change.

    Returns
    -------
    tuple[list[float], list[list[float]]]
        ``(eigenvalues, eigenvectors)`` where each eigenvector is a list of
        length ``d``.  Eigenvalues are guaranteed non-negative.
    """
    d = len(matrix)
    rng = random.Random(seed)

    # Deep copy the matrix so deflation does not mutate the caller's data.
    mat: list[list[float]] = [list(row) for row in matrix]

    eigenvalues: list[float] = []
    eigenvectors: list[list[float]] = []

    for _ in range(n_components):
        # Initialise a random unit vector.
        v = [rng.gauss(0.0, 1.0) for _ in range(d)]
        norm_v = math.sqrt(sum(x * x for x in v))
        if norm_v == 0.0:
            norm_v = 1.0
        v = [x / norm_v for x in v]

        eigenvalue = 0.0

        for _it in range(max_iterations):
            # w = mat @ v
            w: list[float] = [0.0] * d
            for i in range(d):
                s = 0.0
                for j in range(d):
                    s += mat[i][j] * v[j]
                w[i] = s

            # eigenvalue = dot(w, v)
            eigenvalue = sum(w_i * v_i for w_i, v_i in zip(w, v))

            # Normalise w to get new v.
            norm_w = math.sqrt(sum(x * x for x in w))
            if norm_w == 0.0:
                break
            v_new = [x / norm_w for x in w]

            # Check convergence: ||v_new - v_old||
            diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(v_new, v)))
            v = v_new

            if diff < tol:
                break

        # Ensure non-negative eigenvalue.
        eigenvalue = max(eigenvalue, 0.0)

        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)

        # Deflate: mat = mat - eigenvalue * v * v^T
        for i in range(d):
            for j in range(d):
                mat[i][j] -= eigenvalue * v[i] * v[j]

    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# LatentTransform
# ---------------------------------------------------------------------------

class LatentTransform:
    """Fits a PCA latent space and transforms parameters to/from it.

    All computation is pure Python stdlib (no numpy/scipy).  The PCA is
    deterministic given the same seed.

    Parameters
    ----------
    min_variance_explained:
        Minimum cumulative explained variance ratio to retain (0-1).
    max_components:
        Hard upper limit on the number of retained components.
        ``None`` means no explicit cap beyond the data-driven limit.
    max_power_iterations:
        Maximum number of iterations in the power iteration loop.
    convergence_tol:
        Convergence tolerance for eigenvector stability.
    """

    def __init__(
        self,
        min_variance_explained: float = 0.8,
        max_components: int | None = None,
        max_power_iterations: int = 200,
        convergence_tol: float = 1e-8,
    ) -> None:
        self.min_variance_explained = min_variance_explained
        self.max_components = max_components
        self.max_power_iterations = max_power_iterations
        self.convergence_tol = convergence_tol

    # -- public API ---------------------------------------------------------

    def fit(self, snapshot: CampaignSnapshot, seed: int = 42) -> LatentSpace:
        """Fit a PCA latent space from the successful observations in *snapshot*.

        Steps
        -----
        1. Extract successful observations and filter to numeric parameters.
        2. Build a data matrix (rows = observations, cols = parameters).
        3. Standardise (centre and scale) each column.
        4. Compute the sample covariance matrix.
        5. Power iteration with deflation for top-k eigenvectors.
        6. Select the number of components that explain at least
           ``min_variance_explained`` of the total variance.

        Raises
        ------
        ValueError
            If fewer than 2 successful observations exist or there are no
            numeric parameters.
        """
        # 1. Get successful observations and numeric parameter names.
        successful = snapshot.successful_observations
        if len(successful) < 2:
            raise ValueError(
                f"Need at least 2 successful observations for PCA, "
                f"got {len(successful)}."
            )

        numeric_params: list[str] = [
            spec.name
            for spec in snapshot.parameter_specs
            if spec.type != VariableType.CATEGORICAL
        ]
        if not numeric_params:
            raise ValueError(
                "No numeric (non-CATEGORICAL) parameters found in the snapshot."
            )

        n_samples = len(successful)
        n_features = len(numeric_params)

        # 2. Build data matrix.
        data = _build_data_matrix(successful, numeric_params)

        # 3. Standardise: compute per-column mean and std, then centre & scale.
        col_mean: list[float] = [0.0] * n_features
        for j in range(n_features):
            col_mean[j] = sum(data[i][j] for i in range(n_samples)) / n_samples

        col_std: list[float] = [0.0] * n_features
        for j in range(n_features):
            variance = sum(
                (data[i][j] - col_mean[j]) ** 2 for i in range(n_samples)
            ) / max(n_samples - 1, 1)
            col_std[j] = math.sqrt(variance) if variance > 0.0 else 1.0

        # Centre and scale in-place.
        for i in range(n_samples):
            for j in range(n_features):
                data[i][j] = (data[i][j] - col_mean[j]) / col_std[j]

        # 4. Compute covariance matrix of the standardised data.
        cov = _covariance_matrix(data, n_samples, n_features)

        # 5. Power iteration for top-k eigenvectors.
        max_k = min(n_features, n_samples - 1)
        if self.max_components is not None:
            max_k = min(max_k, self.max_components)
        max_k = max(max_k, 1)

        eigenvalues, eigenvectors = _power_iteration_eigendecomp(
            cov, max_k, seed, self.max_power_iterations, self.convergence_tol
        )

        # 6. Compute explained variance ratios and select n_components.
        total_variance = sum(eigenvalues)
        if total_variance <= 0.0:
            total_variance = 1.0  # Safeguard: avoid division by zero.

        explained_ratios: list[float] = [
            ev / total_variance for ev in eigenvalues
        ]

        # Determine how many components to keep.
        cumulative = 0.0
        n_components = max_k  # Default: keep all.
        for idx, ratio in enumerate(explained_ratios):
            cumulative += ratio
            if cumulative >= self.min_variance_explained:
                n_components = idx + 1
                break

        # Trim to selected number of components.
        components = eigenvectors[:n_components]
        eigenvalues = eigenvalues[:n_components]
        explained_ratios = explained_ratios[:n_components]
        total_explained = sum(explained_ratios)

        return LatentSpace(
            components=components,
            eigenvalues=eigenvalues,
            mean=col_mean,
            std=col_std,
            n_components=n_components,
            original_dim=n_features,
            explained_variance_ratio=explained_ratios,
            total_explained_variance=total_explained,
            param_names=numeric_params,
        )

    def to_latent(
        self,
        params: dict[str, Any],
        latent_space: LatentSpace,
    ) -> list[float]:
        """Project a parameter dictionary into latent coordinates.

        Parameters
        ----------
        params:
            Mapping of parameter name to value (must contain all numeric
            parameters used when the latent space was fitted).
        latent_space:
            The fitted latent space.

        Returns
        -------
        list[float]
            Latent coordinates of length ``latent_space.n_components``.
        """
        # Standardise.
        x: list[float] = []
        for j, name in enumerate(latent_space.param_names):
            val = float(params.get(name, 0.0))
            x.append((val - latent_space.mean[j]) / latent_space.std[j])

        # Project: z[i] = dot(component[i], x).
        z: list[float] = []
        for i in range(latent_space.n_components):
            comp = latent_space.components[i]
            dot = sum(c * xi for c, xi in zip(comp, x))
            z.append(dot)

        return z

    def from_latent(
        self,
        latent_coords: list[float],
        latent_space: LatentSpace,
        specs: list[ParameterSpec] | None = None,
    ) -> dict[str, Any]:
        """Reconstruct original-space parameters from latent coordinates.

        Parameters
        ----------
        latent_coords:
            Latent-space coordinates (length ``n_components``).
        latent_space:
            The fitted latent space.
        specs:
            Optional parameter specifications used to clamp values to bounds
            and round discrete parameters.

        Returns
        -------
        dict[str, Any]
            Reconstructed parameter dictionary.
        """
        d = latent_space.original_dim

        # Reconstruct in standardised space: x[j] = sum(z[i] * component[i][j]).
        x: list[float] = [0.0] * d
        for i, z_i in enumerate(latent_coords):
            comp = latent_space.components[i]
            for j in range(d):
                x[j] += z_i * comp[j]

        # Unstandardise: val = x[j] * std[j] + mean[j].
        result: dict[str, Any] = {}
        spec_map: dict[str, ParameterSpec] = {}
        if specs is not None:
            spec_map = {s.name: s for s in specs}

        for j, name in enumerate(latent_space.param_names):
            val = x[j] * latent_space.std[j] + latent_space.mean[j]

            spec = spec_map.get(name)
            if spec is not None:
                # Clamp to bounds.
                if spec.lower is not None:
                    val = max(val, spec.lower)
                if spec.upper is not None:
                    val = min(val, spec.upper)

                # Round discrete parameters.
                if spec.type == VariableType.DISCRETE:
                    val = round(val)

            result[name] = val

        return result

    def explained_variance(self, latent_space: LatentSpace) -> float:
        """Return the total explained variance ratio of *latent_space*."""
        return latent_space.total_explained_variance

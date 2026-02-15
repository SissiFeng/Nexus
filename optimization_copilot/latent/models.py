"""Data models for latent space optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatentSpace:
    """Represents a learned latent space from PCA decomposition.

    Attributes
    ----------
    components:
        Eigenvectors of the covariance matrix (each is a list of floats
        with length ``original_dim``).  ``components[i]`` is the *i*-th
        principal component.
    eigenvalues:
        Eigenvalue corresponding to each component, sorted descending.
    mean:
        Per-parameter mean used for standardisation (length ``original_dim``).
    std:
        Per-parameter standard deviation used for standardisation
        (length ``original_dim``).
    n_components:
        Number of retained principal components.
    original_dim:
        Dimensionality of the original parameter space.
    explained_variance_ratio:
        Fraction of total variance explained by each component.
    total_explained_variance:
        Cumulative fraction of variance explained by all retained components.
    param_names:
        Names of the original parameters in column order.
    """

    components: list[list[float]]
    eigenvalues: list[float]
    mean: list[float]
    std: list[float]
    n_components: int
    original_dim: int
    explained_variance_ratio: list[float]
    total_explained_variance: float
    param_names: list[str] = field(default_factory=list)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the latent space to a plain dictionary."""
        return {
            "components": [list(c) for c in self.components],
            "eigenvalues": list(self.eigenvalues),
            "mean": list(self.mean),
            "std": list(self.std),
            "n_components": self.n_components,
            "original_dim": self.original_dim,
            "explained_variance_ratio": list(self.explained_variance_ratio),
            "total_explained_variance": self.total_explained_variance,
            "param_names": list(self.param_names),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatentSpace:
        """Reconstruct a ``LatentSpace`` from its dictionary representation."""
        return cls(
            components=[list(c) for c in data["components"]],
            eigenvalues=list(data["eigenvalues"]),
            mean=list(data["mean"]),
            std=list(data["std"]),
            n_components=int(data["n_components"]),
            original_dim=int(data["original_dim"]),
            explained_variance_ratio=list(data["explained_variance_ratio"]),
            total_explained_variance=float(data["total_explained_variance"]),
            param_names=list(data.get("param_names", [])),
        )

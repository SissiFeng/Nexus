"""Abstract base class for representation providers.

Representation providers map raw parameter values to vector representations
and compute similarity between values. This abstraction allows the optimization
engine to work with different encoding strategies (fingerprints, descriptors,
embeddings) through a uniform interface.

Design rationale:
- Separates encoding logic from optimization logic
- Allows swapping representations to study their effect on recommendations
- Provides a standard interface for both built-in and custom encodings
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RepresentationProvider(ABC):
    """Maps raw parameter values to vector representations with similarity computation.

    A RepresentationProvider is responsible for:
    1. Encoding raw values (strings, numbers, etc.) into fixed-size float vectors
    2. Computing pairwise similarity between raw values
    3. Providing metadata about its encoding scheme

    Subclasses must implement all abstract methods. The encode/similarity
    methods must be deterministic: given the same inputs, they must always
    produce the same outputs.

    Example usage::

        provider = NGramTanimoto(n=3, fingerprint_size=128)
        vectors = provider.encode(["CCO", "CC(=O)O", "c1ccccc1"])
        sim = provider.similarity("CCO", "CC(=O)O")
    """

    @abstractmethod
    def encode(self, raw_values: list[Any]) -> list[list[float]]:
        """Encode a list of raw values into feature vectors.

        Each raw value is mapped to a fixed-size float vector. All returned
        vectors have the same dimensionality.

        Args:
            raw_values: List of raw parameter values (strings, numbers, etc.).

        Returns:
            List of feature vectors, one per input value. Each vector is a
            list of floats with length equal to the provider's feature count.
        """

    @abstractmethod
    def similarity(self, a: Any, b: Any) -> float:
        """Compute similarity between two raw values.

        The similarity metric is provider-specific (e.g., Tanimoto for
        fingerprints, cosine for real-valued vectors).

        Args:
            a: First raw parameter value.
            b: Second raw parameter value.

        Returns:
            Float in [0, 1] where 1.0 means identical and 0.0 means
            maximally dissimilar.
        """

    @abstractmethod
    def encoding_metadata(self) -> dict[str, Any]:
        """Return metadata about this encoding.

        Metadata should include at minimum:
        - name: unique identifier for the provider
        - n_features: dimensionality of encoded vectors
        - description: human-readable description

        Returns:
            Dictionary containing encoding metadata.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this provider.

        Used as the key in RepresentationRegistry and for serialization.

        Returns:
            String identifier (e.g., "ngram_tanimoto", "encoding_adapter").
        """

    def __repr__(self) -> str:
        """Return string representation of this provider."""
        meta = self.encoding_metadata()
        n_features = meta.get("n_features", "?")
        return f"{self.__class__.__name__}(name={self.name!r}, n_features={n_features})"

    def batch_similarity(self, values: list[Any]) -> list[list[float]]:
        """Compute pairwise similarity matrix for a list of values.

        This is a convenience method that computes all pairwise similarities.
        Subclasses may override for efficiency.

        Args:
            values: List of raw parameter values.

        Returns:
            Square matrix (list of lists) where entry [i][j] is the
            similarity between values[i] and values[j].
        """
        n = len(values)
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                sim = self.similarity(values[i], values[j])
                matrix[i][j] = sim
                matrix[j][i] = sim
        return matrix

    def rank_by_similarity(
        self, query: Any, candidates: list[Any]
    ) -> list[tuple[int, float]]:
        """Rank candidates by similarity to a query value.

        Args:
            query: Reference value to compare against.
            candidates: List of candidate values to rank.

        Returns:
            List of (index, similarity) tuples sorted by descending similarity.
        """
        scored = [
            (i, self.similarity(query, candidate))
            for i, candidate in enumerate(candidates)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

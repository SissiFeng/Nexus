"""Adapter wrapping existing Encoding instances as RepresentationProviders.

Bridges the existing domain_encoding.Encoding ABC (which encodes single
values) with the RepresentationProvider interface (which encodes batches
and computes similarity).

Similarity is computed using cosine similarity between encoded vectors,
which is appropriate for the real-valued descriptor vectors produced
by the existing encodings.
"""

from __future__ import annotations

from typing import Any

from optimization_copilot.infrastructure.domain_encoding import Encoding
from optimization_copilot.representation.provider import RepresentationProvider


class EncodingAdapter(RepresentationProvider):
    """Wraps an existing Encoding instance as a RepresentationProvider.

    This adapter enables all existing Encoding subclasses (OneHotEncoding,
    OrdinalEncoding, CustomDescriptorEncoding, SpatialEncoding) to be used
    wherever a RepresentationProvider is expected.

    Similarity is computed via cosine similarity between the encoded
    feature vectors.

    Args:
        encoding: An existing Encoding instance to wrap.
        provider_name: Unique name for this provider instance.

    Example::

        from optimization_copilot.infrastructure.domain_encoding import OneHotEncoding
        enc = OneHotEncoding(["red", "green", "blue"])
        provider = EncodingAdapter(enc, provider_name="color_onehot")
        vectors = provider.encode(["red", "blue"])
        sim = provider.similarity("red", "green")
    """

    def __init__(
        self, encoding: Encoding, provider_name: str = "encoding_adapter"
    ) -> None:
        """Initialize the encoding adapter.

        Args:
            encoding: An Encoding instance to wrap.
            provider_name: Unique identifier for this provider.
        """
        self._encoding = encoding
        self._name = provider_name

    @property
    def name(self) -> str:
        """Unique identifier for this provider."""
        return self._name

    def encode(self, raw_values: list[Any]) -> list[list[float]]:
        """Encode a list of raw values using the wrapped Encoding.

        Calls the underlying Encoding.encode() for each value.

        Args:
            raw_values: List of raw parameter values.

        Returns:
            List of feature vectors, one per input value.
        """
        return [self._encoding.encode(v) for v in raw_values]

    def similarity(self, a: Any, b: Any) -> float:
        """Compute cosine similarity between two encoded values.

        Cosine similarity = dot(a, b) / (||a|| * ||b||)

        Returns 0.0 if either vector has near-zero norm. Result is
        clamped to [0, 1].

        Args:
            a: First raw parameter value.
            b: Second raw parameter value.

        Returns:
            Float in [0, 1].
        """
        vec_a = self._encoding.encode(a)
        vec_b = self._encoding.encode(b)

        dot = sum(x * y for x, y in zip(vec_a, vec_b))
        norm_a = sum(x ** 2 for x in vec_a) ** 0.5
        norm_b = sum(x ** 2 for x in vec_b) ** 0.5

        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0

        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    def encoding_metadata(self) -> dict[str, Any]:
        """Return metadata about this encoding.

        Returns:
            Dictionary with name, n_features, encoding_type, and description.
        """
        return {
            "name": self._name,
            "n_features": self._encoding.n_features,
            "encoding_type": self._encoding.__class__.__name__,
            "description": f"Adapter wrapping {self._encoding.__class__.__name__}",
        }

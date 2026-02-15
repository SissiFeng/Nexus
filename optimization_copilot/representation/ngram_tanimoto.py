"""N-gram Tanimoto fingerprint representation provider.

Default provider for SMILES strings and other text-based parameters.
Computes character n-grams, hashes them to fixed-size fingerprints,
and uses Tanimoto coefficient for similarity.

The Tanimoto coefficient (a.k.a. Jaccard index for binary vectors)
is the standard similarity metric for molecular fingerprints:
    T(A, B) = |A intersection B| / |A union B|

This approach is widely used in cheminformatics for comparing
molecular structures via their SMILES representations.
"""

from __future__ import annotations

from typing import Any

from optimization_copilot.representation.provider import RepresentationProvider


class NGramTanimoto(RepresentationProvider):
    """Character n-gram fingerprint provider with Tanimoto similarity.

    Encodes strings by:
    1. Extracting all character n-grams (substrings of length n)
    2. Hashing each n-gram to a bit position in a fixed-size fingerprint
    3. Setting the corresponding bit to 1.0

    Similarity is computed as the Tanimoto coefficient between two
    fingerprints, which equals |A intersection B| / |A union B| for
    binary vectors.

    Args:
        n: Length of character n-grams. Default 3 (trigrams).
        fingerprint_size: Number of bits in the fingerprint. Default 128.

    Example::

        provider = NGramTanimoto(n=3, fingerprint_size=128)
        fps = provider.encode(["CCO", "CC(=O)O"])
        sim = provider.similarity("CCO", "CC(=O)O")
    """

    def __init__(self, n: int = 3, fingerprint_size: int = 128) -> None:
        """Initialize n-gram Tanimoto provider.

        Args:
            n: Length of character n-grams. Must be >= 1.
            fingerprint_size: Number of bits in the fingerprint. Must be >= 1.

        Raises:
            ValueError: If n or fingerprint_size is less than 1.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if fingerprint_size < 1:
            raise ValueError(f"fingerprint_size must be >= 1, got {fingerprint_size}")
        self._n = n
        self._fingerprint_size = fingerprint_size

    @property
    def name(self) -> str:
        """Unique identifier for this provider."""
        return "ngram_tanimoto"

    def _extract_ngrams(self, text: str) -> set[str]:
        """Extract character n-grams from a string.

        For strings shorter than n, the entire string is used as a single
        n-gram to ensure non-empty fingerprints for non-empty inputs.

        Args:
            text: Input string.

        Returns:
            Set of n-gram strings.
        """
        if not text:
            return set()
        if len(text) < self._n:
            # For short strings, use the whole string as a single gram
            return {text}
        return {text[i : i + self._n] for i in range(len(text) - self._n + 1)}

    def _ngrams_to_fingerprint(self, ngrams: set[str]) -> list[float]:
        """Convert a set of n-grams to a binary fingerprint.

        Each n-gram is hashed to a bit position using Python's built-in
        hash function modulo fingerprint_size.

        Args:
            ngrams: Set of n-gram strings.

        Returns:
            Binary fingerprint as a list of 0.0/1.0 floats.
        """
        fp = [0.0] * self._fingerprint_size
        for gram in ngrams:
            # Use abs to handle negative hash values
            bit_pos = abs(hash(gram)) % self._fingerprint_size
            fp[bit_pos] = 1.0
        return fp

    def _ngram_set(self, value: Any) -> set[str]:
        """Extract n-gram set from a raw value.

        Converts the value to string first.

        Args:
            value: Raw parameter value.

        Returns:
            Set of n-gram strings.
        """
        text = str(value)
        return self._extract_ngrams(text)

    def encode(self, raw_values: list[Any]) -> list[list[float]]:
        """Encode a list of raw values into fingerprint vectors.

        Each value is converted to a string, n-grams are extracted,
        and each n-gram is hashed to a bit position in the fingerprint.

        Args:
            raw_values: List of raw parameter values (typically strings).

        Returns:
            List of binary fingerprint vectors, each of length fingerprint_size.
        """
        result: list[list[float]] = []
        for value in raw_values:
            ngrams = self._ngram_set(value)
            fp = self._ngrams_to_fingerprint(ngrams)
            result.append(fp)
        return result

    def similarity(self, a: Any, b: Any) -> float:
        """Compute Tanimoto coefficient between two values.

        Tanimoto coefficient = |A intersection B| / |A union B|
        where A and B are the sets of n-gram hashes.

        For identical inputs, returns 1.0. For inputs with no shared
        n-grams, returns 0.0. For two empty strings, returns 1.0
        (by convention, as they are identical).

        Args:
            a: First raw parameter value.
            b: Second raw parameter value.

        Returns:
            Float in [0, 1].
        """
        ngrams_a = self._ngram_set(a)
        ngrams_b = self._ngram_set(b)

        # Both empty -> identical
        if not ngrams_a and not ngrams_b:
            return 1.0
        # One empty, one not -> completely dissimilar
        if not ngrams_a or not ngrams_b:
            return 0.0

        intersection = ngrams_a & ngrams_b
        union = ngrams_a | ngrams_b
        return len(intersection) / len(union)

    def encoding_metadata(self) -> dict[str, Any]:
        """Return metadata about this encoding.

        Returns:
            Dictionary with name, n_features, n_gram_size, fingerprint_size,
            and description.
        """
        return {
            "name": self.name,
            "n_features": self._fingerprint_size,
            "n_gram_size": self._n,
            "fingerprint_size": self._fingerprint_size,
            "description": (
                f"Character {self._n}-gram fingerprint with Tanimoto similarity "
                f"({self._fingerprint_size}-bit)"
            ),
        }

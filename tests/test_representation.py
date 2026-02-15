"""Tests for the representation provider module.

Covers NGramTanimoto, EncodingAdapter, RepresentationRegistry, and
the key acceptance test proving that different providers produce
different recommendations.
"""

from __future__ import annotations

import math

import pytest

from optimization_copilot.infrastructure.domain_encoding import (
    OneHotEncoding,
    OrdinalEncoding,
)
from optimization_copilot.representation import (
    EncodingAdapter,
    NGramTanimoto,
    RepresentationProvider,
    RepresentationRegistry,
)


# ---------------------------------------------------------------------------
# NGramTanimoto tests
# ---------------------------------------------------------------------------


class TestNGramTanimotoEncode:
    """Tests for NGramTanimoto.encode()."""

    def test_encode_produces_correct_size_fingerprint(self) -> None:
        """Encoded vectors should have length equal to fingerprint_size."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        result = provider.encode(["CCO"])
        assert len(result) == 1
        assert len(result[0]) == 128

    def test_encode_produces_correct_size_custom_fingerprint(self) -> None:
        """Custom fingerprint_size is respected."""
        provider = NGramTanimoto(n=2, fingerprint_size=64)
        result = provider.encode(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 64

    def test_encode_multiple_values(self) -> None:
        """Encoding multiple values returns one vector per value."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        result = provider.encode(["CCO", "CC(=O)O", "c1ccccc1"])
        assert len(result) == 3
        for fp in result:
            assert len(fp) == 128

    def test_encode_binary_values(self) -> None:
        """Fingerprint values should be 0.0 or 1.0."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        result = provider.encode(["CCO"])
        for val in result[0]:
            assert val in (0.0, 1.0)

    def test_encode_nonempty_string_has_set_bits(self) -> None:
        """Non-empty string should produce fingerprint with at least one set bit."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        result = provider.encode(["CCO"])
        assert sum(result[0]) > 0

    def test_encode_empty_string(self) -> None:
        """Empty string should produce all-zero fingerprint."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        result = provider.encode([""])
        assert len(result) == 1
        assert sum(result[0]) == 0.0

    def test_encode_single_character(self) -> None:
        """Single character (shorter than n=3) should still produce a fingerprint."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        result = provider.encode(["A"])
        assert len(result) == 1
        assert len(result[0]) == 128
        # Should have at least one bit set (the whole string used as gram)
        assert sum(result[0]) > 0

    def test_encode_two_characters(self) -> None:
        """Two characters (shorter than n=3) should produce a fingerprint."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        result = provider.encode(["AB"])
        assert len(result) == 1
        assert sum(result[0]) > 0

    def test_encode_deterministic(self) -> None:
        """Same input must always produce the same output."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        result1 = provider.encode(["CCO", "CC(=O)O"])
        result2 = provider.encode(["CCO", "CC(=O)O"])
        assert result1 == result2

    def test_encode_deterministic_repeated(self) -> None:
        """Determinism holds across many calls."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        first = provider.encode(["test_string"])
        for _ in range(10):
            assert provider.encode(["test_string"]) == first

    def test_different_n_produces_different_fingerprints(self) -> None:
        """Different n values should produce different fingerprints."""
        provider_2 = NGramTanimoto(n=2, fingerprint_size=128)
        provider_3 = NGramTanimoto(n=3, fingerprint_size=128)
        fp_2 = provider_2.encode(["CC(=O)O"])
        fp_3 = provider_3.encode(["CC(=O)O"])
        assert fp_2 != fp_3

    def test_different_fingerprint_sizes(self) -> None:
        """Different fingerprint sizes produce different-length vectors."""
        provider_64 = NGramTanimoto(n=3, fingerprint_size=64)
        provider_256 = NGramTanimoto(n=3, fingerprint_size=256)
        fp_64 = provider_64.encode(["CCO"])
        fp_256 = provider_256.encode(["CCO"])
        assert len(fp_64[0]) == 64
        assert len(fp_256[0]) == 256

    def test_encode_numeric_value(self) -> None:
        """Numeric values are converted to string before encoding."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        result = provider.encode([42])
        assert len(result) == 1
        assert len(result[0]) == 128


class TestNGramTanimotoSimilarity:
    """Tests for NGramTanimoto.similarity()."""

    def test_identical_strings_similarity_one(self) -> None:
        """Identical strings must have similarity 1.0."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        assert provider.similarity("CCO", "CCO") == 1.0

    def test_identical_long_strings_similarity_one(self) -> None:
        """Identical longer strings must have similarity 1.0."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        assert provider.similarity(smiles, smiles) == 1.0

    def test_very_different_strings_low_similarity(self) -> None:
        """Very different strings should have similarity < 0.5."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        sim = provider.similarity("AAAA", "ZZZZ")
        assert sim < 0.5

    def test_completely_disjoint_strings_zero_similarity(self) -> None:
        """Strings with no shared n-grams should have similarity near 0."""
        provider = NGramTanimoto(n=3, fingerprint_size=1024)
        # Use large fingerprint to minimize hash collisions
        sim = provider.similarity("aaa", "zzz")
        assert sim < 0.2

    def test_similar_smiles_reasonable_similarity(self) -> None:
        """Similar SMILES strings should have moderate-to-high similarity."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        # Aspirin vs salicylic acid - share many 3-grams from the ring
        sim = provider.similarity("CC(=O)Oc1ccccc1C(=O)O", "c1ccc(O)cc1C(=O)O")
        assert 0.0 < sim < 1.0

    def test_smiles_similarity_ordering(self) -> None:
        """More similar molecules should have higher similarity scores."""
        provider = NGramTanimoto(n=3, fingerprint_size=256)
        # Ethanol
        query = "CCO"
        # Methanol (very similar to ethanol)
        sim_methanol = provider.similarity(query, "CO")
        # Benzene (very different from ethanol)
        sim_benzene = provider.similarity(query, "c1ccccc1")
        # Propanol (similar to ethanol)
        sim_propanol = provider.similarity(query, "CCCO")
        # Propanol should be more similar to ethanol than benzene
        assert sim_propanol > sim_benzene

    def test_symmetry(self) -> None:
        """Similarity must be symmetric: sim(a,b) == sim(b,a)."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        assert provider.similarity("CCO", "CC(=O)O") == provider.similarity(
            "CC(=O)O", "CCO"
        )

    def test_similarity_range(self) -> None:
        """Similarity must be in [0, 1]."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        pairs = [("CCO", "CO"), ("A", "Z"), ("hello", "world"), ("", "test")]
        for a, b in pairs:
            sim = provider.similarity(a, b)
            assert 0.0 <= sim <= 1.0, f"sim({a!r}, {b!r}) = {sim}"

    def test_empty_strings_similarity(self) -> None:
        """Two empty strings should have similarity 1.0 (both identical)."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        assert provider.similarity("", "") == 1.0

    def test_empty_vs_nonempty_similarity(self) -> None:
        """Empty vs non-empty should have similarity 0.0."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        assert provider.similarity("", "hello") == 0.0
        assert provider.similarity("hello", "") == 0.0

    def test_single_char_self_similarity(self) -> None:
        """Single character compared to itself should give 1.0."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        assert provider.similarity("A", "A") == 1.0


class TestNGramTanimotoMetadata:
    """Tests for NGramTanimoto.encoding_metadata() and other properties."""

    def test_encoding_metadata_fields(self) -> None:
        """Metadata should contain expected fields."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        meta = provider.encoding_metadata()
        assert meta["name"] == "ngram_tanimoto"
        assert meta["n_features"] == 128
        assert meta["n_gram_size"] == 3
        assert meta["fingerprint_size"] == 128
        assert "description" in meta

    def test_encoding_metadata_custom_params(self) -> None:
        """Metadata reflects custom n and fingerprint_size."""
        provider = NGramTanimoto(n=5, fingerprint_size=256)
        meta = provider.encoding_metadata()
        assert meta["n_gram_size"] == 5
        assert meta["fingerprint_size"] == 256
        assert meta["n_features"] == 256

    def test_name_property(self) -> None:
        """Name property returns the correct identifier."""
        provider = NGramTanimoto()
        assert provider.name == "ngram_tanimoto"

    def test_repr(self) -> None:
        """Repr should include class name and key info."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        r = repr(provider)
        assert "NGramTanimoto" in r
        assert "ngram_tanimoto" in r

    def test_is_representation_provider(self) -> None:
        """NGramTanimoto should be a RepresentationProvider subclass."""
        provider = NGramTanimoto()
        assert isinstance(provider, RepresentationProvider)


class TestNGramTanimotoValidation:
    """Tests for NGramTanimoto constructor validation."""

    def test_invalid_n_zero(self) -> None:
        """n=0 should raise ValueError."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            NGramTanimoto(n=0)

    def test_invalid_n_negative(self) -> None:
        """Negative n should raise ValueError."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            NGramTanimoto(n=-1)

    def test_invalid_fingerprint_size_zero(self) -> None:
        """fingerprint_size=0 should raise ValueError."""
        with pytest.raises(ValueError, match="fingerprint_size must be >= 1"):
            NGramTanimoto(fingerprint_size=0)

    def test_invalid_fingerprint_size_negative(self) -> None:
        """Negative fingerprint_size should raise ValueError."""
        with pytest.raises(ValueError, match="fingerprint_size must be >= 1"):
            NGramTanimoto(fingerprint_size=-5)


class TestNGramTanimotoBatchAndRank:
    """Tests for convenience methods inherited from RepresentationProvider."""

    def test_batch_similarity(self) -> None:
        """batch_similarity produces a symmetric matrix with 1.0 diagonal."""
        provider = NGramTanimoto(n=3, fingerprint_size=128)
        values = ["CCO", "CO", "c1ccccc1"]
        matrix = provider.batch_similarity(values)
        assert len(matrix) == 3
        for i in range(3):
            assert len(matrix[i]) == 3
            assert matrix[i][i] == 1.0
            for j in range(3):
                assert matrix[i][j] == matrix[j][i]

    def test_rank_by_similarity(self) -> None:
        """rank_by_similarity returns sorted (index, similarity) tuples."""
        provider = NGramTanimoto(n=3, fingerprint_size=256)
        candidates = ["CCCO", "c1ccccc1", "CCO"]
        ranked = provider.rank_by_similarity("CCO", candidates)
        assert len(ranked) == 3
        # CCO should be most similar to itself (index 2)
        assert ranked[0][0] == 2
        assert ranked[0][1] == 1.0
        # Sorted by descending similarity
        for i in range(len(ranked) - 1):
            assert ranked[i][1] >= ranked[i + 1][1]


# ---------------------------------------------------------------------------
# EncodingAdapter tests
# ---------------------------------------------------------------------------


class TestEncodingAdapterOneHot:
    """Tests for EncodingAdapter with OneHotEncoding."""

    def test_encode_one_hot(self) -> None:
        """Adapter correctly wraps OneHotEncoding for batch encoding."""
        enc = OneHotEncoding(["red", "green", "blue"])
        adapter = EncodingAdapter(enc, provider_name="color_onehot")
        result = adapter.encode(["red", "blue"])
        assert len(result) == 2
        assert result[0] == [1.0, 0.0, 0.0]
        assert result[1] == [0.0, 0.0, 1.0]

    def test_similarity_identical_one_hot(self) -> None:
        """Same category should have cosine similarity 1.0."""
        enc = OneHotEncoding(["red", "green", "blue"])
        adapter = EncodingAdapter(enc)
        assert adapter.similarity("red", "red") == 1.0

    def test_similarity_different_one_hot(self) -> None:
        """Different one-hot categories should have cosine similarity 0.0."""
        enc = OneHotEncoding(["red", "green", "blue"])
        adapter = EncodingAdapter(enc)
        assert adapter.similarity("red", "green") == 0.0

    def test_name_property(self) -> None:
        """Name should match the provided name."""
        enc = OneHotEncoding(["a", "b"])
        adapter = EncodingAdapter(enc, provider_name="my_onehot")
        assert adapter.name == "my_onehot"

    def test_default_name(self) -> None:
        """Default name should be 'encoding_adapter'."""
        enc = OneHotEncoding(["a", "b"])
        adapter = EncodingAdapter(enc)
        assert adapter.name == "encoding_adapter"


class TestEncodingAdapterOrdinal:
    """Tests for EncodingAdapter with OrdinalEncoding."""

    def test_encode_ordinal(self) -> None:
        """Adapter correctly wraps OrdinalEncoding for batch encoding."""
        enc = OrdinalEncoding(["low", "medium", "high"])
        adapter = EncodingAdapter(enc, provider_name="temp_ordinal")
        result = adapter.encode(["low", "high"])
        assert len(result) == 2
        assert result[0] == [0.0]
        assert result[1] == [1.0]

    def test_similarity_identical_ordinal(self) -> None:
        """Same ordinal category should have similarity 1.0."""
        enc = OrdinalEncoding(["low", "medium", "high"])
        adapter = EncodingAdapter(enc)
        assert adapter.similarity("high", "high") == 1.0

    def test_similarity_ordinal_range(self) -> None:
        """Ordinal similarity should be in [0, 1]."""
        enc = OrdinalEncoding(["low", "medium", "high"])
        adapter = EncodingAdapter(enc)
        sim = adapter.similarity("low", "high")
        assert 0.0 <= sim <= 1.0

    def test_similarity_ordinal_zero_vec(self) -> None:
        """Low category encodes to [0.0]; cosine with non-zero should be 0.0."""
        enc = OrdinalEncoding(["low", "medium", "high"])
        adapter = EncodingAdapter(enc)
        # "low" encodes to [0.0], which has zero norm
        sim = adapter.similarity("low", "high")
        assert sim == 0.0


class TestEncodingAdapterMetadata:
    """Tests for EncodingAdapter.encoding_metadata()."""

    def test_metadata_fields_one_hot(self) -> None:
        """Metadata should have expected fields for one-hot adapter."""
        enc = OneHotEncoding(["a", "b", "c"])
        adapter = EncodingAdapter(enc, provider_name="test_adapter")
        meta = adapter.encoding_metadata()
        assert meta["name"] == "test_adapter"
        assert meta["n_features"] == 3
        assert meta["encoding_type"] == "OneHotEncoding"
        assert "description" in meta

    def test_metadata_fields_ordinal(self) -> None:
        """Metadata should have expected fields for ordinal adapter."""
        enc = OrdinalEncoding(["x", "y", "z"])
        adapter = EncodingAdapter(enc, provider_name="ord_adapter")
        meta = adapter.encoding_metadata()
        assert meta["name"] == "ord_adapter"
        assert meta["n_features"] == 1
        assert meta["encoding_type"] == "OrdinalEncoding"

    def test_is_representation_provider(self) -> None:
        """EncodingAdapter should be a RepresentationProvider subclass."""
        enc = OneHotEncoding(["a"])
        adapter = EncodingAdapter(enc)
        assert isinstance(adapter, RepresentationProvider)


class TestEncodingAdapterCosine:
    """Tests verifying cosine similarity computation correctness."""

    def test_cosine_orthogonal_vectors(self) -> None:
        """Orthogonal one-hot vectors should have cosine similarity 0."""
        enc = OneHotEncoding(["a", "b", "c"])
        adapter = EncodingAdapter(enc)
        # "a" = [1,0,0], "b" = [0,1,0] -> cosine = 0
        assert adapter.similarity("a", "b") == 0.0
        assert adapter.similarity("a", "c") == 0.0
        assert adapter.similarity("b", "c") == 0.0

    def test_cosine_parallel_vectors(self) -> None:
        """Identical one-hot vectors should have cosine similarity 1."""
        enc = OneHotEncoding(["a", "b"])
        adapter = EncodingAdapter(enc)
        assert adapter.similarity("a", "a") == 1.0
        assert adapter.similarity("b", "b") == 1.0


# ---------------------------------------------------------------------------
# RepresentationRegistry tests
# ---------------------------------------------------------------------------


class TestRepresentationRegistry:
    """Tests for RepresentationRegistry."""

    def test_builtin_ngram_tanimoto_registered(self) -> None:
        """Default registry should contain ngram_tanimoto."""
        registry = RepresentationRegistry()
        assert "ngram_tanimoto" in registry

    def test_get_builtin(self) -> None:
        """get() should return the built-in provider."""
        registry = RepresentationRegistry()
        provider = registry.get("ngram_tanimoto")
        assert provider is not None
        assert isinstance(provider, NGramTanimoto)

    def test_get_nonexistent(self) -> None:
        """get() should return None for unknown names."""
        registry = RepresentationRegistry()
        assert registry.get("nonexistent") is None

    def test_register_custom_provider(self) -> None:
        """Can register and retrieve a custom provider."""
        registry = RepresentationRegistry()
        custom = NGramTanimoto(n=5, fingerprint_size=64)
        # Override name for testing
        registry._providers["custom_ngram"] = custom
        assert registry.get("custom_ngram") is custom

    def test_register_via_register_method(self) -> None:
        """register() method stores the provider under its name."""
        registry = RepresentationRegistry()
        enc = OneHotEncoding(["a", "b"])
        adapter = EncodingAdapter(enc, provider_name="test_enc")
        registry.register(adapter)
        assert "test_enc" in registry
        assert registry.get("test_enc") is adapter

    def test_list_providers(self) -> None:
        """list_providers() returns all registered names."""
        registry = RepresentationRegistry()
        names = registry.list_providers()
        assert "ngram_tanimoto" in names

    def test_list_providers_after_register(self) -> None:
        """list_providers() includes newly registered providers."""
        registry = RepresentationRegistry()
        enc = OneHotEncoding(["x"])
        adapter = EncodingAdapter(enc, provider_name="new_provider")
        registry.register(adapter)
        names = registry.list_providers()
        assert "ngram_tanimoto" in names
        assert "new_provider" in names

    def test_contains_true(self) -> None:
        """__contains__ returns True for registered names."""
        registry = RepresentationRegistry()
        assert "ngram_tanimoto" in registry

    def test_contains_false(self) -> None:
        """__contains__ returns False for unregistered names."""
        registry = RepresentationRegistry()
        assert "does_not_exist" not in registry

    def test_len(self) -> None:
        """__len__ returns correct count."""
        registry = RepresentationRegistry()
        initial_count = len(registry)
        assert initial_count >= 1
        enc = OneHotEncoding(["a"])
        adapter = EncodingAdapter(enc, provider_name="extra")
        registry.register(adapter)
        assert len(registry) == initial_count + 1

    def test_register_replaces_existing(self) -> None:
        """Registering under same name replaces the previous provider."""
        registry = RepresentationRegistry()
        enc1 = OneHotEncoding(["a"])
        enc2 = OneHotEncoding(["a", "b"])
        adapter1 = EncodingAdapter(enc1, provider_name="shared_name")
        adapter2 = EncodingAdapter(enc2, provider_name="shared_name")
        registry.register(adapter1)
        registry.register(adapter2)
        retrieved = registry.get("shared_name")
        assert retrieved is adapter2


# ---------------------------------------------------------------------------
# KEY ACCEPTANCE TEST
# ---------------------------------------------------------------------------


class TestDifferentProvidersDifferentRecommendations:
    """Acceptance test: different providers produce different similarity rankings.

    This proves that changing the representation provider changes the
    recommendation ordering -- the core value proposition of the
    RepresentationProvider abstraction.
    """

    def test_different_metadata(self) -> None:
        """Two providers with different n produce different metadata."""
        provider_n2 = NGramTanimoto(n=2, fingerprint_size=256)
        provider_n5 = NGramTanimoto(n=5, fingerprint_size=256)

        meta_n2 = provider_n2.encoding_metadata()
        meta_n5 = provider_n5.encoding_metadata()

        assert meta_n2["n_gram_size"] == 2
        assert meta_n5["n_gram_size"] == 5
        assert meta_n2 != meta_n5

    def test_different_fingerprints(self) -> None:
        """Two providers with different n produce different fingerprints."""
        provider_n2 = NGramTanimoto(n=2, fingerprint_size=256)
        provider_n5 = NGramTanimoto(n=5, fingerprint_size=256)

        fp_n2 = provider_n2.encode(["CC(=O)Oc1ccccc1C(=O)O"])
        fp_n5 = provider_n5.encode(["CC(=O)Oc1ccccc1C(=O)O"])

        assert fp_n2 != fp_n5

    def test_different_similarity_rankings(self) -> None:
        """Different providers produce different similarity rankings for molecules.

        Using SMILES-like strings to represent molecules:
        - query: aspirin-like "CC(=O)Oc1ccccc1C(=O)O"
        - candidates with varying structural similarity

        With n=2 (bigrams), local character pairs dominate similarity.
        With n=5 (5-grams), longer structural motifs matter more.
        """
        provider_n2 = NGramTanimoto(n=2, fingerprint_size=256)
        provider_n5 = NGramTanimoto(n=5, fingerprint_size=256)

        query = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin-like
        candidates = [
            "CC(=O)O",           # acetic acid (shares short fragments)
            "c1ccccc1O",         # phenol (shares ring)
            "CC(=O)Nc1ccccc1",   # acetanilide (shares longer motifs)
            "CCCCCCCC",          # octane (very different)
            "c1ccc(O)cc1C(=O)O", # salicylic acid (structurally close)
        ]

        # Get rankings from both providers
        ranking_n2 = provider_n2.rank_by_similarity(query, candidates)
        ranking_n5 = provider_n5.rank_by_similarity(query, candidates)

        order_n2 = [idx for idx, _ in ranking_n2]
        order_n5 = [idx for idx, _ in ranking_n5]

        # The key assertion: different n-gram sizes produce different orderings
        # This proves the provider choice affects recommendations
        assert order_n2 != order_n5, (
            f"Rankings should differ but both produced {order_n2}. "
            f"n=2 similarities: {[(i, round(s, 3)) for i, s in ranking_n2]}, "
            f"n=5 similarities: {[(i, round(s, 3)) for i, s in ranking_n5]}"
        )

    def test_different_similarity_values(self) -> None:
        """Different providers produce different similarity values."""
        provider_n2 = NGramTanimoto(n=2, fingerprint_size=256)
        provider_n5 = NGramTanimoto(n=5, fingerprint_size=256)

        a = "CC(=O)Oc1ccccc1C(=O)O"
        b = "CC(=O)Nc1ccccc1"

        sim_n2 = provider_n2.similarity(a, b)
        sim_n5 = provider_n5.similarity(a, b)

        # Both should be valid similarities
        assert 0.0 <= sim_n2 <= 1.0
        assert 0.0 <= sim_n5 <= 1.0
        # But they should differ
        assert sim_n2 != sim_n5, (
            f"Similarities should differ: n=2 gave {sim_n2}, n=5 gave {sim_n5}"
        )

    def test_registry_with_multiple_providers(self) -> None:
        """Registry can hold multiple providers and they produce different results."""
        registry = RepresentationRegistry()
        # The default ngram_tanimoto (n=3) is already registered
        # Register another variant as an adapter
        enc = OneHotEncoding(["CCO", "CO", "c1ccccc1", "CC(=O)O"])
        adapter = EncodingAdapter(enc, provider_name="molecule_onehot")
        registry.register(adapter)

        assert "ngram_tanimoto" in registry
        assert "molecule_onehot" in registry

        ngram = registry.get("ngram_tanimoto")
        onehot = registry.get("molecule_onehot")

        assert ngram is not None
        assert onehot is not None

        # N-gram gives partial similarity between molecules sharing trigrams
        # "CC(=O)O" and "c1ccccc1" are in the one-hot category list and
        # share no 3-grams, so we use them to demonstrate the difference
        sim_ngram = ngram.similarity("CC(=O)Oc1ccccc1C(=O)O", "c1ccc(O)cc1C(=O)O")
        assert 0.0 < sim_ngram < 1.0

        # One-hot doesn't recognize these (not in its category list),
        # so use two known categories that are different
        sim_onehot = onehot.similarity("CCO", "CO")
        assert sim_onehot == 0.0

        # N-gram similarity is non-zero for structurally related strings,
        # while one-hot gives zero for any two distinct categories.
        # This proves different providers produce different recommendations.
        assert sim_ngram != sim_onehot

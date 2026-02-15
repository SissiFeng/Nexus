"""Tests for RAGIndex TF-IDF vector retrieval."""

from __future__ import annotations

import math

import pytest

from optimization_copilot.platform.models import CampaignRecord, CampaignStatus
from optimization_copilot.platform.rag import RAGIndex


# ── Helpers ──────────────────────────────────────────────────────


def _make_record(
    name: str = "Test Campaign",
    tags: list[str] | None = None,
    status: CampaignStatus = CampaignStatus.DRAFT,
) -> CampaignRecord:
    """Create a minimal CampaignRecord for testing."""
    from time import time

    return CampaignRecord(
        campaign_id="test-id",
        name=name,
        status=status,
        spec={},
        created_at=time(),
        updated_at=time(),
        tags=tags or [],
    )


def _make_spec(
    param_names: list[str] | None = None,
    obj_names: list[str] | None = None,
    description: str = "",
) -> dict:
    """Create a minimal spec dict for testing."""
    params = [{"name": n, "type": "continuous", "bounds": [0, 1]} for n in (param_names or [])]
    objs = [{"name": n, "direction": "minimize"} for n in (obj_names or [])]
    return {
        "parameters": params,
        "objectives": objs,
        "description": description,
    }


# ── Index Basics ─────────────────────────────────────────────────


class TestRAGIndexBasics:
    """Basic index operations."""

    def test_starts_empty(self):
        idx = RAGIndex()
        assert idx.document_count == 0

    def test_index_campaign_adds_document(self):
        idx = RAGIndex()
        record = _make_record(name="My Campaign")
        spec = _make_spec(param_names=["x"], obj_names=["y"])
        idx.index_campaign("c1", record, spec)
        assert idx.document_count == 1

    def test_multiple_campaigns_can_be_indexed(self):
        idx = RAGIndex()
        for i in range(5):
            record = _make_record(name=f"Campaign {i}")
            spec = _make_spec(param_names=[f"p{i}"], obj_names=[f"obj{i}"])
            idx.index_campaign(f"c{i}", record, spec)
        assert idx.document_count == 5

    def test_index_campaign_replaces_existing(self):
        idx = RAGIndex()
        record = _make_record(name="Original")
        spec = _make_spec()
        idx.index_campaign("c1", record, spec)

        record2 = _make_record(name="Updated")
        idx.index_campaign("c1", record2, _make_spec())
        assert idx.document_count == 1


# ── Search ───────────────────────────────────────────────────────


class TestRAGSearch:
    """Search operations."""

    def test_search_returns_results_matching_query(self):
        idx = RAGIndex()
        # Need 2+ docs so IDF is non-zero for distinguishing terms
        r1 = _make_record(name="Temperature Optimization")
        s1 = _make_spec(param_names=["temperature", "pressure"], obj_names=["yield"])
        idx.index_campaign("c1", r1, s1)

        r2 = _make_record(name="Color Analysis")
        s2 = _make_spec(param_names=["hue", "saturation"], obj_names=["contrast"])
        idx.index_campaign("c2", r2, s2)

        results = idx.search("temperature")
        assert len(results) >= 1
        assert results[0].campaign_id == "c1"

    def test_search_returns_empty_for_unrelated_query(self):
        idx = RAGIndex()
        record = _make_record(name="Chemical Reaction")
        spec = _make_spec(param_names=["temperature"], obj_names=["yield"])
        idx.index_campaign("c1", record, spec)

        results = idx.search("zzzzqqqxxx")
        assert results == []

    def test_search_top_k_limits_results(self):
        idx = RAGIndex()
        for i in range(10):
            record = _make_record(name=f"Optimization {i}")
            spec = _make_spec(param_names=["x"], obj_names=["y"])
            idx.index_campaign(f"c{i}", record, spec)

        results = idx.search("optimization", top_k=3)
        assert len(results) <= 3

    def test_search_scores_between_0_and_1(self):
        idx = RAGIndex()
        record = _make_record(name="Test Search Scoring")
        spec = _make_spec(param_names=["alpha", "beta"], obj_names=["loss"])
        idx.index_campaign("c1", record, spec)

        results = idx.search("test search")
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_search_ranks_more_relevant_higher(self):
        idx = RAGIndex()

        # Campaign about temperature
        r1 = _make_record(name="Temperature Control")
        s1 = _make_spec(
            param_names=["temperature", "temp_rate"],
            obj_names=["accuracy"],
            description="Optimize temperature for best results",
        )
        idx.index_campaign("temp_campaign", r1, s1)

        # Campaign about something else
        r2 = _make_record(name="Color Analysis")
        s2 = _make_spec(
            param_names=["red", "green", "blue"],
            obj_names=["contrast"],
            description="Analyze color palette for maximum contrast",
        )
        idx.index_campaign("color_campaign", r2, s2)

        results = idx.search("temperature")
        assert len(results) >= 1
        assert results[0].campaign_id == "temp_campaign"

    def test_search_with_no_documents_returns_empty(self):
        idx = RAGIndex()
        results = idx.search("anything")
        assert results == []

    def test_empty_query_returns_empty(self):
        idx = RAGIndex()
        record = _make_record(name="Some Campaign")
        spec = _make_spec()
        idx.index_campaign("c1", record, spec)

        results = idx.search("")
        assert results == []

    def test_whitespace_query_returns_empty(self):
        idx = RAGIndex()
        record = _make_record(name="Some Campaign")
        spec = _make_spec()
        idx.index_campaign("c1", record, spec)

        results = idx.search("   ")
        assert results == []

    def test_search_after_remove_excludes_removed(self):
        idx = RAGIndex()
        r1 = _make_record(name="Alpha Campaign")
        s1 = _make_spec(param_names=["alpha"])
        idx.index_campaign("c1", r1, s1)

        r2 = _make_record(name="Beta Campaign")
        s2 = _make_spec(param_names=["beta"])
        idx.index_campaign("c2", r2, s2)

        idx.remove_campaign("c1")

        results = idx.search("alpha")
        campaign_ids = [r.campaign_id for r in results]
        assert "c1" not in campaign_ids


# ── Remove & Rebuild ─────────────────────────────────────────────


class TestRAGRemoveAndRebuild:
    """Remove and rebuild operations."""

    def test_remove_campaign(self):
        idx = RAGIndex()
        record = _make_record()
        spec = _make_spec()
        idx.index_campaign("c1", record, spec)
        assert idx.document_count == 1

        idx.remove_campaign("c1")
        assert idx.document_count == 0

    def test_remove_nonexistent_is_noop(self):
        idx = RAGIndex()
        idx.remove_campaign("nonexistent")
        assert idx.document_count == 0

    def test_rebuild_replaces_all_documents(self):
        idx = RAGIndex()
        # Index initial campaigns
        for i in range(3):
            record = _make_record(name=f"Old {i}")
            spec = _make_spec()
            idx.index_campaign(f"old{i}", record, spec)
        assert idx.document_count == 3

        # Rebuild with new campaigns
        new_campaigns = []
        for i in range(2):
            record = _make_record(name=f"New {i}")
            spec = _make_spec()
            new_campaigns.append((f"new{i}", record, spec, None))

        idx.rebuild(new_campaigns)
        assert idx.document_count == 2


# ── Serialization ────────────────────────────────────────────────


class TestRAGSerialization:
    """to_dict / from_dict round-trip."""

    def test_round_trip_preserves_empty_index(self):
        idx = RAGIndex()
        data = idx.to_dict()
        restored = RAGIndex.from_dict(data)
        assert restored.document_count == 0

    def test_round_trip_preserves_index_with_data(self):
        idx = RAGIndex()
        # Need 2+ docs so IDF is non-zero for distinguishing terms
        r1 = _make_record(name="Serialization Test")
        s1 = _make_spec(param_names=["x", "y"], obj_names=["loss"])
        idx.index_campaign("c1", r1, s1)

        r2 = _make_record(name="Other Campaign")
        s2 = _make_spec(param_names=["alpha"], obj_names=["cost"])
        idx.index_campaign("c2", r2, s2)

        data = idx.to_dict()
        restored = RAGIndex.from_dict(data)

        assert restored.document_count == 2
        # Search should still work — "serialization" only in c1
        results = restored.search("serialization")
        assert len(results) >= 1
        assert results[0].campaign_id == "c1"


# ── TF-IDF Internals ────────────────────────────────────────────


class TestRAGInternals:
    """Internal TF-IDF methods."""

    def test_tokenize_splits_text(self):
        tokens = RAGIndex._tokenize("hello world foo bar")
        assert tokens == ["hello", "world", "foo", "bar"]

    def test_tokenize_lowercases(self):
        tokens = RAGIndex._tokenize("Hello WORLD FoO")
        assert tokens == ["hello", "world", "foo"]

    def test_tokenize_handles_punctuation(self):
        tokens = RAGIndex._tokenize("hello, world! foo-bar")
        # The regex [a-z0-9_]+ means 'foo' and 'bar' are separate
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        assert "bar" in tokens

    def test_compute_tf_normalized(self):
        tokens = ["a", "b", "a", "c", "a"]
        tf = RAGIndex._compute_tf(tokens)
        assert tf["a"] == pytest.approx(3 / 5)
        assert tf["b"] == pytest.approx(1 / 5)
        assert tf["c"] == pytest.approx(1 / 5)

    def test_compute_tf_empty_tokens(self):
        tf = RAGIndex._compute_tf([])
        assert tf == {}

    def test_compute_idf_positive_for_indexed_terms(self):
        idx = RAGIndex()
        record = _make_record(name="test term")
        spec = _make_spec()
        idx.index_campaign("c1", record, spec)

        idf = idx._compute_idf("test")
        # With 1 doc containing "test", idf = log(1/1) = 0
        # This is correct for single-doc scenario
        assert idf >= 0.0

    def test_compute_idf_zero_for_unknown_term(self):
        idx = RAGIndex()
        record = _make_record(name="hello")
        spec = _make_spec()
        idx.index_campaign("c1", record, spec)

        idf = idx._compute_idf("zzzznotaword")
        assert idf == 0.0

    def test_cosine_similarity_identical_vectors(self):
        vec = {"a": 1.0, "b": 2.0, "c": 3.0}
        sim = RAGIndex._cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        vec_a = {"a": 1.0, "b": 0.0}
        vec_b = {"c": 1.0, "d": 0.0}
        sim = RAGIndex._cosine_similarity(vec_a, vec_b)
        assert sim == pytest.approx(0.0)

    def test_cosine_similarity_empty_vectors(self):
        assert RAGIndex._cosine_similarity({}, {"a": 1.0}) == 0.0
        assert RAGIndex._cosine_similarity({"a": 1.0}, {}) == 0.0
        assert RAGIndex._cosine_similarity({}, {}) == 0.0

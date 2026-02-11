"""Pure Python TF-IDF vector retrieval for campaign search."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from optimization_copilot.platform.models import CampaignRecord, SearchResult


class RAGIndex:
    """TF-IDF based search index for campaigns.

    Uses pure Python: collections.Counter for TF, math.log for IDF,
    cosine similarity via dot product / (norm_a * norm_b).
    """

    def __init__(self) -> None:
        # campaign_id -> {field_name: text}
        self._documents: dict[str, dict[str, str]] = {}
        # campaign_id -> {field_name: tf_vector}
        self._tf_vectors: dict[str, dict[str, dict[str, float]]] = {}
        # term -> document_frequency (number of documents containing term)
        self._df: Counter[str] = Counter()
        self._total_docs: int = 0

    # ── Indexing ──────────────────────────────────────────────

    def index_campaign(
        self,
        campaign_id: str,
        record: CampaignRecord,
        spec_dict: dict[str, Any],
        result_dict: dict[str, Any] | None = None,
    ) -> None:
        """Index a campaign for search."""
        # Remove old entry first
        if campaign_id in self._documents:
            self.remove_campaign(campaign_id)

        # Build document fields
        fields: dict[str, str] = {}
        fields["name"] = record.name
        fields["tags"] = " ".join(record.tags)
        fields["status"] = record.status.value

        # Extract parameter names from spec
        params = spec_dict.get("parameters", [])
        fields["parameters"] = " ".join(p.get("name", "") for p in params)

        # Extract objective names
        objectives = spec_dict.get("objectives", [])
        fields["objectives"] = " ".join(o.get("name", "") for o in objectives)

        # Spec description and metadata
        fields["description"] = spec_dict.get("description", "")
        fields["campaign_id"] = campaign_id

        # Result data if available
        if result_dict:
            fields["termination"] = result_dict.get("termination_reason", "")
            best_kpi = result_dict.get("best_kpi_values", {})
            fields["best_kpi"] = " ".join(
                f"{k}={v}" for k, v in best_kpi.items()
            )

        # Compute TF vectors for each field
        self._documents[campaign_id] = fields
        self._tf_vectors[campaign_id] = {}

        all_terms: set[str] = set()
        for field_name, text in fields.items():
            tokens = self._tokenize(text)
            tf = self._compute_tf(tokens)
            self._tf_vectors[campaign_id][field_name] = tf
            all_terms.update(tf.keys())

        # Update document frequency
        for term in all_terms:
            self._df[term] += 1
        self._total_docs += 1

    def remove_campaign(self, campaign_id: str) -> None:
        """Remove a campaign from the index."""
        if campaign_id not in self._documents:
            return

        # Collect all unique terms from this campaign
        all_terms: set[str] = set()
        for tf in self._tf_vectors.get(campaign_id, {}).values():
            all_terms.update(tf.keys())

        # Decrement document frequencies
        for term in all_terms:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]

        del self._documents[campaign_id]
        del self._tf_vectors[campaign_id]
        self._total_docs -= 1

    def rebuild(self, campaigns: list[tuple[str, CampaignRecord, dict[str, Any], dict[str, Any] | None]]) -> None:
        """Rebuild index from scratch."""
        self._documents.clear()
        self._tf_vectors.clear()
        self._df.clear()
        self._total_docs = 0

        for campaign_id, record, spec_dict, result_dict in campaigns:
            self.index_campaign(campaign_id, record, spec_dict, result_dict)

    # ── Search ────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search campaigns by query string."""
        if not self._documents or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        query_tf = self._compute_tf(query_tokens)
        query_tfidf = self._to_tfidf(query_tf)

        results: list[SearchResult] = []

        for campaign_id, field_tfs in self._tf_vectors.items():
            best_score = 0.0
            best_field = ""
            best_snippet = ""

            for field_name, tf in field_tfs.items():
                doc_tfidf = self._to_tfidf(tf)
                score = self._cosine_similarity(query_tfidf, doc_tfidf)

                if score > best_score:
                    best_score = score
                    best_field = field_name
                    best_snippet = self._documents[campaign_id].get(field_name, "")

            if best_score > 0:
                results.append(
                    SearchResult(
                        campaign_id=campaign_id,
                        field=best_field,
                        snippet=best_snippet[:200],
                        score=round(best_score, 6),
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ── TF-IDF Internals ──────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase words."""
        return re.findall(r"[a-z0-9_]+", text.lower())

    @staticmethod
    def _compute_tf(tokens: list[str]) -> dict[str, float]:
        """Compute term frequency (normalized by document length)."""
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = len(tokens)
        return {term: count / total for term, count in counts.items()}

    def _compute_idf(self, term: str) -> float:
        """Compute inverse document frequency for a term."""
        df = self._df.get(term, 0)
        if df == 0 or self._total_docs == 0:
            return 0.0
        return math.log(self._total_docs / df)

    def _to_tfidf(self, tf: dict[str, float]) -> dict[str, float]:
        """Convert TF vector to TF-IDF vector."""
        return {term: freq * self._compute_idf(term) for term, freq in tf.items()}

    @staticmethod
    def _cosine_similarity(
        vec_a: dict[str, float], vec_b: dict[str, float]
    ) -> float:
        """Compute cosine similarity between two sparse vectors."""
        if not vec_a or not vec_b:
            return 0.0

        # Dot product
        dot = sum(vec_a.get(k, 0.0) * vec_b.get(k, 0.0) for k in vec_a if k in vec_b)
        if dot == 0.0:
            return 0.0

        # Norms
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self._documents,
            "tf_vectors": self._tf_vectors,
            "df": dict(self._df),
            "total_docs": self._total_docs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGIndex:
        index = cls()
        index._documents = data.get("documents", {})
        index._tf_vectors = data.get("tf_vectors", {})
        index._df = Counter(data.get("df", {}))
        index._total_docs = data.get("total_docs", 0)
        return index

    @property
    def document_count(self) -> int:
        return self._total_docs

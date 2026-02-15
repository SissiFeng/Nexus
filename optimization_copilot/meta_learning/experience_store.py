"""Persistent cross-campaign outcome storage.

Stores ``CampaignOutcome`` records and supports fingerprint-based
similarity retrieval for cross-campaign transfer learning.
"""

from __future__ import annotations

import json
import math
from typing import Any

from optimization_copilot.core.models import ProblemFingerprint
from optimization_copilot.meta_learning.models import (
    CampaignOutcome,
    ExperienceRecord,
    MetaLearningConfig,
)


class ExperienceStore:
    """Persistent cross-campaign outcome storage with similarity search."""

    def __init__(self, config: MetaLearningConfig | None = None) -> None:
        self._config = config or MetaLearningConfig()
        self._records: list[ExperienceRecord] = []
        self._by_campaign: dict[str, ExperienceRecord] = {}
        self._by_fingerprint: dict[str, list[ExperienceRecord]] = {}

    # ── Mutation ──────────────────────────────────────────

    def record_outcome(self, outcome: CampaignOutcome) -> ExperienceRecord:
        """Store a campaign outcome and return the experience record."""
        fp_key = str(outcome.fingerprint.to_tuple())
        record = ExperienceRecord(outcome=outcome, fingerprint_key=fp_key)
        self._records.append(record)
        self._by_campaign[outcome.campaign_id] = record
        self._by_fingerprint.setdefault(fp_key, []).append(record)
        return record

    def record_outcomes(
        self, outcomes: list[CampaignOutcome]
    ) -> list[ExperienceRecord]:
        """Store multiple campaign outcomes."""
        return [self.record_outcome(o) for o in outcomes]

    # ── Query ─────────────────────────────────────────────

    def get_by_campaign(self, campaign_id: str) -> ExperienceRecord | None:
        """Retrieve experience by campaign ID."""
        return self._by_campaign.get(campaign_id)

    def get_by_fingerprint(self, fingerprint_key: str) -> list[ExperienceRecord]:
        """Retrieve all experiences for an exact fingerprint key."""
        return list(self._by_fingerprint.get(fingerprint_key, []))

    def get_similar(
        self, fingerprint: ProblemFingerprint, max_results: int = 10
    ) -> list[tuple[ExperienceRecord, float]]:
        """Find experiences with similar fingerprints, sorted by similarity desc.

        Returns (record, similarity) tuples where similarity is 0.0-1.0.
        """
        scored: list[tuple[ExperienceRecord, float]] = []
        for record in self._records:
            sim = self._fingerprint_similarity(fingerprint, record.outcome.fingerprint)
            scored.append((record, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_results]

    def get_all(self) -> list[ExperienceRecord]:
        """Return all stored experience records."""
        return list(self._records)

    def count(self) -> int:
        """Return the number of stored records."""
        return len(self._records)

    # ── Similarity ────────────────────────────────────────

    def _fingerprint_similarity(
        self, fp1: ProblemFingerprint, fp2: ProblemFingerprint
    ) -> float:
        """Compute continuous similarity between two fingerprints using RBF kernel.

        Uses domain-aware ordinal encoding for enum fields, producing smooth
        similarity gradients instead of binary match/no-match.

        Kernel: k(x,y) = exp(-||x-y||^2 / 2)  (RBF with length_scale=1.0)

        Returns similarity score in [0.0, 1.0] where 1.0 is identical.
        """
        v1 = fp1.to_continuous_vector()
        v2 = fp2.to_continuous_vector()
        sq_dist = sum((a - b) ** 2 for a, b in zip(v1, v2))
        return math.exp(-sq_dist / 2.0)

    # ── Recency weighting ─────────────────────────────────

    def recency_weight(self, record: ExperienceRecord, latest_ts: float) -> float:
        """Compute recency weight for a record using exponential decay.

        Weight = 2^(-(latest_ts - record_ts) / halflife) when timestamps
        represent campaign indices.  Falls back to 1.0 if halflife <= 0.
        """
        halflife = self._config.recency_halflife
        if halflife <= 0:
            return 1.0
        age = latest_ts - record.outcome.timestamp
        if age <= 0:
            return 1.0
        return math.pow(2.0, -age / halflife)

    # ── Serialization ─────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "min_experiences_for_learning": self._config.min_experiences_for_learning,
                "similarity_decay": self._config.similarity_decay,
                "weight_learning_rate": self._config.weight_learning_rate,
                "threshold_learning_rate": self._config.threshold_learning_rate,
                "recency_halflife": self._config.recency_halflife,
            },
            "records": [r.to_dict() for r in self._records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperienceStore:
        config = MetaLearningConfig(**data.get("config", {}))
        store = cls(config=config)
        for rd in data.get("records", []):
            record = ExperienceRecord.from_dict(rd)
            store._records.append(record)
            store._by_campaign[record.outcome.campaign_id] = record
            store._by_fingerprint.setdefault(record.fingerprint_key, []).append(record)
        return store

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> ExperienceStore:
        return cls.from_dict(json.loads(json_str))

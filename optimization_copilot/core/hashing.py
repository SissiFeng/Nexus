"""Deterministic hashing for audit trail and reproducibility."""

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .models import CampaignSnapshot, StrategyDecision


def _stable_serialize(obj: Any) -> str:
    """JSON serialize with sorted keys for deterministic output."""
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=True)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def snapshot_hash(snapshot: CampaignSnapshot) -> str:
    """Deterministic hash of a CampaignSnapshot."""
    return _sha256(_stable_serialize(asdict(snapshot)))


def decision_hash(decision: StrategyDecision) -> str:
    """Deterministic hash of a StrategyDecision."""
    return _sha256(_stable_serialize(decision.to_dict()))


def diagnostics_hash(vector: dict) -> str:
    """Deterministic hash of a diagnostics vector."""
    return _sha256(_stable_serialize(vector))

"""Failure taxonomy and conditional failure modeling.

Extends the FeasibilityLearner with failure classification, providing
structured analysis of *why* experiments fail and what strategy
adjustments are appropriate for each failure type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
)


# ── Enums ─────────────────────────────────────────────────


class FailureType(str, Enum):
    """Taxonomy of experiment failure modes."""

    HARDWARE = "hardware"        # equipment malfunction
    CHEMISTRY = "chemistry"      # reaction failure, precipitation, etc.
    DATA = "data"                # data corruption, measurement error
    PROTOCOL = "protocol"        # procedure violation
    UNKNOWN = "unknown"


# ── Data classes ──────────────────────────────────────────


@dataclass
class ClassifiedFailure:
    """A single failure observation with its classification."""

    observation_index: int
    failure_type: FailureType
    confidence: float  # 0-1
    evidence: list[str] = field(default_factory=list)


@dataclass
class FailureTaxonomy:
    """Aggregate taxonomy of all failures in a campaign snapshot."""

    classified_failures: list[ClassifiedFailure]
    type_counts: dict[str, int]
    dominant_type: FailureType
    type_rates: dict[str, float]           # each type's fraction of total failures
    strategy_adjustments: dict[str, str]    # failure_type -> recommended change


# ── Default keyword lists ─────────────────────────────────

_DEFAULT_HARDWARE_KEYWORDS: list[str] = [
    "timeout",
    "connection",
    "instrument",
    "sensor",
    "hardware",
    "mechanical",
    "power",
]

_DEFAULT_CHEMISTRY_KEYWORDS: list[str] = [
    "precipitate",
    "reaction",
    "yield",
    "solubility",
    "pH",
    "temperature",
    "concentration",
    "crystallization",
]

_DEFAULT_PROTOCOL_KEYWORDS: list[str] = [
    "protocol",
    "procedure",
    "step",
    "sequence",
    "manual",
    "operator",
    "compliance",
]


# ── Classifier ────────────────────────────────────────────


class FailureClassifier:
    """Classify experiment failures into a structured taxonomy.

    Parameters
    ----------
    hardware_keywords:
        Words in ``failure_reason`` that indicate hardware problems.
    chemistry_keywords:
        Words in ``failure_reason`` that indicate chemistry problems.
    protocol_keywords:
        Words in ``failure_reason`` that indicate protocol problems.
    """

    def __init__(
        self,
        hardware_keywords: list[str] | None = None,
        chemistry_keywords: list[str] | None = None,
        protocol_keywords: list[str] | None = None,
    ) -> None:
        self.hardware_keywords = hardware_keywords or list(_DEFAULT_HARDWARE_KEYWORDS)
        self.chemistry_keywords = chemistry_keywords or list(_DEFAULT_CHEMISTRY_KEYWORDS)
        self.protocol_keywords = protocol_keywords or list(_DEFAULT_PROTOCOL_KEYWORDS)

    # ── public API ────────────────────────────────────────

    def classify(self, snapshot: CampaignSnapshot) -> FailureTaxonomy:
        """Classify every failure in *snapshot* and return a taxonomy."""
        all_failures = [o for o in snapshot.observations if o.is_failure]

        # Also gather data-quality failures (qc_passed=False, not is_failure)
        data_quality_issues = [
            o for o in snapshot.observations
            if not o.qc_passed and not o.is_failure
        ]

        classified: list[ClassifiedFailure] = []

        # Classify hard failures
        for obs in all_failures:
            idx = snapshot.observations.index(obs)
            cf = self._classify_single(
                obs, idx, all_failures, snapshot.parameter_specs,
            )
            classified.append(cf)

        # Classify data-quality observations
        for obs in data_quality_issues:
            idx = snapshot.observations.index(obs)
            classified.append(
                ClassifiedFailure(
                    observation_index=idx,
                    failure_type=FailureType.DATA,
                    confidence=0.85,
                    evidence=["qc_passed=False without is_failure flag"],
                )
            )

        # Aggregate counts
        type_counts: dict[str, int] = {ft.value: 0 for ft in FailureType}
        for cf in classified:
            type_counts[cf.failure_type.value] += 1

        total = len(classified) if classified else 1  # avoid division by zero
        type_rates: dict[str, float] = {
            k: v / total for k, v in type_counts.items()
        }

        # Dominant type
        if classified:
            dominant_type = FailureType(
                max(type_counts, key=lambda k: type_counts[k])
            )
        else:
            dominant_type = FailureType.UNKNOWN

        # Strategy adjustments
        strategy_adjustments = self.recommend_adjustments(
            type_counts, dominant_type,
        )

        return FailureTaxonomy(
            classified_failures=classified,
            type_counts=type_counts,
            dominant_type=dominant_type,
            type_rates=type_rates,
            strategy_adjustments=strategy_adjustments,
        )

    def recommend_adjustments(
        self,
        type_counts: dict[str, int],
        dominant_type: FailureType,
    ) -> dict[str, str]:
        """Return per-failure-type strategy recommendations.

        The mapping is ``{failure_type_value: recommendation_string}``.
        """
        adjustments: dict[str, str] = {}

        # Always provide a recommendation for the dominant type
        _dominant_map = {
            FailureType.HARDWARE: "reduce_exploration",
            FailureType.CHEMISTRY: "adjust_bounds",
            FailureType.DATA: "increase_replicates",
            FailureType.PROTOCOL: "enforce_protocol_checks",
            FailureType.UNKNOWN: "conservative_exploration",
        }

        # Check if we have a truly mixed situation (2+ types with counts > 0,
        # and no single type accounts for more than 60 %).
        nonzero = {k: v for k, v in type_counts.items() if v > 0}
        total = sum(type_counts.values()) or 1
        is_mixed = (
            len(nonzero) >= 2
            and (type_counts.get(dominant_type.value, 0) / total) <= 0.6
        )

        if is_mixed:
            adjustments["mixed"] = "conservative_exploration"

        # Per-type recommendations for any type that actually occurred
        for ft in FailureType:
            if type_counts.get(ft.value, 0) > 0:
                adjustments[ft.value] = _dominant_map[ft]

        # Always include the dominant recommendation at key "dominant"
        adjustments["dominant"] = _dominant_map[dominant_type]

        return adjustments

    # ── internal helpers ──────────────────────────────────

    def _classify_single(
        self,
        obs: Observation,
        obs_index: int,
        all_failures: list[Observation],
        specs: list[ParameterSpec],
    ) -> ClassifiedFailure:
        """Classify a single failed observation."""
        evidence: list[str] = []
        scores: dict[FailureType, float] = {ft: 0.0 for ft in FailureType}

        reason = (obs.failure_reason or "").lower()

        # ---- Rule 1: keyword matching on failure_reason ----
        hw_hits = [kw for kw in self.hardware_keywords if kw.lower() in reason]
        ch_hits = [kw for kw in self.chemistry_keywords if kw.lower() in reason]
        pr_hits = [kw for kw in self.protocol_keywords if kw.lower() in reason]

        if hw_hits:
            scores[FailureType.HARDWARE] += 0.6
            evidence.append(f"hardware keywords matched: {hw_hits}")
        if ch_hits:
            scores[FailureType.CHEMISTRY] += 0.6
            evidence.append(f"chemistry keywords matched: {ch_hits}")
        if pr_hits:
            scores[FailureType.PROTOCOL] += 0.6
            evidence.append(f"protocol keywords matched: {pr_hits}")

        # ---- Rule 2: data quality issue ----
        if not obs.qc_passed and not obs.is_failure:
            scores[FailureType.DATA] += 0.7
            evidence.append("qc_passed=False without is_failure flag")

        # ---- Rule 3: spatial clustering (systematic -> chemistry) ----
        if len(all_failures) >= 3 and specs:
            is_systematic = self._is_spatially_clustered(obs, all_failures, specs)
            if is_systematic:
                scores[FailureType.CHEMISTRY] += 0.3
                evidence.append("failure clusters in parameter space (systematic)")
            else:
                scores[FailureType.HARDWARE] += 0.15
                evidence.append("failure scattered in parameter space (sporadic)")

        # ---- Rule 4: has kpi_values but qc_passed=False (data issue) ----
        if obs.kpi_values and not obs.qc_passed:
            scores[FailureType.DATA] += 0.4
            evidence.append("has kpi_values but qc_passed=False")

        # Pick the best-scoring type
        best_type = max(scores, key=lambda ft: scores[ft])
        best_score = scores[best_type]

        # If nothing matched at all, label UNKNOWN
        if best_score == 0.0:
            best_type = FailureType.UNKNOWN
            evidence.append("no classification signals detected")

        confidence = min(best_score, 1.0)

        return ClassifiedFailure(
            observation_index=obs_index,
            failure_type=best_type,
            confidence=confidence,
            evidence=evidence,
        )

    @staticmethod
    def _is_spatially_clustered(
        obs: Observation,
        all_failures: list[Observation],
        specs: list[ParameterSpec],
    ) -> bool:
        """Return True if failures cluster tightly in parameter space.

        Uses a simple normalised spread heuristic: if the standard
        deviation of failure positions (normalised to [0,1] per parameter)
        is below a threshold for at least half the parameters, the failures
        are considered *systematic* (chemistry-like).
        """
        if len(all_failures) < 3:
            return False

        clustered_dims = 0
        evaluated_dims = 0

        for spec in specs:
            if spec.lower is None or spec.upper is None:
                continue
            param_range = spec.upper - spec.lower
            if param_range == 0:
                continue

            evaluated_dims += 1
            vals = [
                (o.parameters.get(spec.name, 0.0) - spec.lower) / param_range
                for o in all_failures
            ]
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(variance)

            # Threshold: if std < 0.25 of the normalised range, consider clustered
            if std < 0.25:
                clustered_dims += 1

        if evaluated_dims == 0:
            return False

        return clustered_dims / evaluated_dims >= 0.5

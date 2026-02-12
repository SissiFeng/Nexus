"""CandidatePool: external candidate pool with acquisition-style ranking."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.candidate_pool.models import PoolCandidate, PoolVersion


class CandidatePool:
    """External candidate pool with acquisition-style ranking.

    Maintains a pool of candidate parameter sets from external sources
    (e.g., a molecular library). Scores candidates against the current
    campaign state and provides ranked suggestions.
    """

    def __init__(self) -> None:
        self._candidates: list[PoolCandidate] = []
        self._versions: list[PoolVersion] = []
        self._version_counter: int = 0

    # ── Properties ────────────────────────────────────────

    @property
    def n_candidates(self) -> int:
        return len(self._candidates)

    @property
    def versions(self) -> list[PoolVersion]:
        return list(self._versions)

    @property
    def latest_version(self) -> PoolVersion | None:
        return self._versions[-1] if self._versions else None

    # ── Loading ───────────────────────────────────────────

    def load(self, candidates: list[dict[str, Any]]) -> None:
        """Load candidates from list of parameter dicts."""
        self._candidates = [
            PoolCandidate(parameters=dict(c)) for c in candidates
        ]

    def load_from_rows(
        self,
        rows: list[dict[str, Any]],
        param_names: list[str],
    ) -> None:
        """Load candidates from tabular data rows.

        Each row is a dict; extract only *param_names* as parameters.
        Remaining keys go into metadata.
        """
        self._candidates = []
        for row in rows:
            params = {k: row[k] for k in param_names if k in row}
            meta = {k: v for k, v in row.items() if k not in param_names}
            self._candidates.append(
                PoolCandidate(parameters=params, metadata=meta, source="tabular")
            )

    # ── Scoring ───────────────────────────────────────────

    def score(
        self,
        snapshot: CampaignSnapshot,
        parameter_specs: list[ParameterSpec],
        seed: int = 42,
    ) -> PoolVersion:
        """Score and rank all candidates against current campaign state.

        Scoring combines:
        1. Distance to nearest observation (exploration bonus: farther = better)
        2. Nearest-neighbor improvement prediction (exploitation: near good obs = better)

        Both are deterministic given the same snapshot and seed.
        """
        if not self._candidates:
            raise ValueError("No candidates loaded")

        # Compute snapshot hash for versioning
        snapshot_hash = self._compute_snapshot_hash(snapshot, seed)

        # Build observation vectors and aggregate KPI scores
        obs_param_vectors: list[list[float]] = []
        obs_kpi_scores: list[float] = []
        for obs in snapshot.observations:
            vec = self._paramdict_to_vector(obs.parameters, parameter_specs)
            obs_param_vectors.append(vec)
            if obs.kpi_values and not obs.is_failure:
                score_val = sum(obs.kpi_values.values()) / len(obs.kpi_values)
                obs_kpi_scores.append(score_val)
            else:
                obs_kpi_scores.append(float("-inf"))

        # Score each candidate
        scored_candidates: list[PoolCandidate] = []
        for cand in self._candidates:
            cand_vec = self._paramdict_to_vector(cand.parameters, parameter_specs)

            if obs_param_vectors:
                # Distance to nearest observation
                min_dist = float("inf")
                nearest_score = float("-inf")
                for obs_vec, obs_score in zip(obs_param_vectors, obs_kpi_scores):
                    dist = self._euclidean_distance(cand_vec, obs_vec)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_score = obs_score

                # Combine exploration (distance) and exploitation (nearest score)
                exploration_score = min_dist
                exploitation_score = (
                    nearest_score if nearest_score > float("-inf") else 0.0
                )
                total_score = 0.5 * exploration_score + 0.5 * exploitation_score
            else:
                # No observations yet: use hash-based deterministic score
                hash_val = hashlib.sha256(
                    json.dumps(
                        cand.parameters, sort_keys=True, default=str
                    ).encode()
                ).hexdigest()
                total_score = int(hash_val[:8], 16) / (16**8)

            scored_candidates.append(
                PoolCandidate(
                    parameters=dict(cand.parameters),
                    score=total_score,
                    rank=0,  # assigned after sorting
                    metadata=dict(cand.metadata),
                    source=cand.source,
                )
            )

        # Rank by score descending (rank 1 = best)
        scored_candidates.sort(key=lambda c: c.score, reverse=True)
        for i, cand in enumerate(scored_candidates):
            cand.rank = i + 1

        # Create and store version
        self._version_counter += 1
        version = PoolVersion(
            version=self._version_counter,
            candidates=scored_candidates,
            scoring_seed=seed,
            scoring_snapshot_hash=snapshot_hash,
            timestamp=time.time(),
        )
        self._versions.append(version)

        return version

    # ── Suggestion ────────────────────────────────────────

    def suggest(self, n: int, seed: int = 42) -> list[dict[str, Any]]:
        """Get top-n candidates as parameter dicts from the latest version."""
        if not self._versions:
            raise ValueError("No scored versions available. Call score() first.")
        latest = self._versions[-1]
        top = latest.top_n(n)
        return [dict(c.parameters) for c in top]

    # ── Internal helpers ──────────────────────────────────

    @staticmethod
    def _paramdict_to_vector(
        params: dict[str, Any],
        specs: list[ParameterSpec],
    ) -> list[float]:
        """Convert parameter dict to normalised numeric vector."""
        vec: list[float] = []
        for spec in specs:
            val = params.get(spec.name)
            if val is None:
                vec.append(0.0)
            elif spec.type == VariableType.CATEGORICAL:
                # Hash categorical value to [0, 1]
                h = hashlib.sha256(str(val).encode()).hexdigest()
                vec.append(int(h[:8], 16) / (16**8))
            elif spec.type in (VariableType.CONTINUOUS, VariableType.DISCRETE):
                fval = float(val)
                lb = spec.lower if spec.lower is not None else 0.0
                ub = spec.upper if spec.upper is not None else 1.0
                if ub > lb:
                    vec.append((fval - lb) / (ub - lb))
                else:
                    vec.append(fval)
            else:
                vec.append(float(val) if isinstance(val, (int, float)) else 0.0)
        return vec

    @staticmethod
    def _euclidean_distance(a: list[float], b: list[float]) -> float:
        """Compute Euclidean distance between two vectors."""
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    @staticmethod
    def _compute_snapshot_hash(snapshot: CampaignSnapshot, seed: int) -> str:
        """Compute deterministic hash of snapshot + seed for versioning."""
        data = {
            "n_observations": snapshot.n_observations,
            "parameter_names": snapshot.parameter_names,
            "objective_names": snapshot.objective_names,
            "seed": seed,
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

"""Cross-campaign knowledge transfer engine.

Enables warm-starting new optimization campaigns from historical data,
computing campaign similarity, projecting observations across parameter
spaces, and pooling compatible data with similarity-based weights.

Features:
1. Campaign similarity (parameter overlap, range overlap, metadata)
2. Warm start: initial points from similar campaigns' best regions
3. Data pooling: merge compatible data with similarity-based weights
4. Transfer history tracking and serialization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CampaignData:
    """Registered campaign for knowledge transfer.

    Attributes:
        campaign_id: Unique identifier for the campaign.
        parameter_specs: Parameter definitions, each a dict with keys
            ``name``, ``type`` (``"continuous"`` or ``"categorical"``),
            and for continuous params ``lower`` / ``upper``, for
            categorical params ``categories``.
        observations: List of observation dicts.  Each dict maps
            parameter names to values and must include an ``"objective"``
            key with the observed objective value.
        metadata: Arbitrary metadata (domain, objective name, etc.)
            used for metadata similarity scoring.
    """

    campaign_id: str
    parameter_specs: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------

    @property
    def param_names(self) -> set[str]:
        """Return the set of parameter names."""
        return {s["name"] for s in self.parameter_specs}

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    def _spec_by_name(self) -> dict[str, dict[str, Any]]:
        """Return parameter specs keyed by name."""
        return {s["name"]: s for s in self.parameter_specs}

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "campaign_id": self.campaign_id,
            "parameter_specs": [dict(s) for s in self.parameter_specs],
            "observations": [dict(o) for o in self.observations],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignData:
        """Deserialize from a dict."""
        return cls(
            campaign_id=data["campaign_id"],
            parameter_specs=data.get("parameter_specs", []),
            observations=data.get("observations", []),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Transfer learning engine
# ---------------------------------------------------------------------------

class TransferLearningEngine:
    """Cross-campaign knowledge transfer.

    The engine maintains a registry of historical campaigns and provides
    methods to:

    * Compute similarity between a new problem specification and
      previously registered campaigns.
    * Produce warm-start points drawn from the top-performing regions
      of similar campaigns.
    * Pool compatible historical data with similarity-based weights
      for surrogate model training.

    Similarity is a weighted combination of three factors::

        similarity = 0.4 * parameter_overlap
                   + 0.4 * range_overlap
                   + 0.2 * metadata_similarity
    """

    # Similarity component weights
    _W_PARAM_OVERLAP: float = 0.4
    _W_RANGE_OVERLAP: float = 0.4
    _W_METADATA: float = 0.2

    def __init__(self) -> None:
        self._history: list[CampaignData] = []
        self._transfer_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_campaigns(self) -> int:
        """Number of registered campaigns."""
        return len(self._history)

    @property
    def campaign_ids(self) -> list[str]:
        """Return registered campaign IDs in registration order."""
        return [c.campaign_id for c in self._history]

    @property
    def transfer_log(self) -> list[dict[str, Any]]:
        """Return a copy of the transfer event log."""
        return list(self._transfer_log)

    # -- registration -----------------------------------------------------

    def register_campaign(
        self,
        campaign_id: str,
        parameter_specs: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a completed or running campaign for future transfer.

        Duplicate *campaign_id* values are replaced (most recent wins).

        Args:
            campaign_id: Unique campaign identifier.
            parameter_specs: List of parameter specification dicts.
            observations: List of observation dicts (must contain
                ``"objective"`` key).
            metadata: Optional metadata dict for similarity scoring.
        """
        campaign = CampaignData(
            campaign_id=campaign_id,
            parameter_specs=[dict(s) for s in parameter_specs],
            observations=[dict(o) for o in observations],
            metadata=dict(metadata) if metadata else {},
        )
        # Replace existing campaign with same id
        self._history = [
            c for c in self._history if c.campaign_id != campaign_id
        ]
        self._history.append(campaign)

    def unregister_campaign(self, campaign_id: str) -> bool:
        """Remove a campaign from the registry.

        Returns:
            ``True`` if the campaign was found and removed.
        """
        before = len(self._history)
        self._history = [
            c for c in self._history if c.campaign_id != campaign_id
        ]
        return len(self._history) < before

    # -- similarity -------------------------------------------------------

    def compute_similarity(
        self,
        current_specs: list[dict[str, Any]],
        target: CampaignData,
    ) -> float:
        """Compute similarity in ``[0, 1]`` between *current_specs* and *target*.

        The similarity score is a weighted combination:

        * **Parameter overlap** (Jaccard on names): weight 0.4
        * **Range overlap** (average intersection/union of continuous
          parameter ranges on shared params): weight 0.4
        * **Metadata similarity** (key overlap ratio): weight 0.2

        Args:
            current_specs: Parameter specifications for the new campaign.
            target: A registered :class:`CampaignData` to compare against.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        p_overlap = self._parameter_overlap(current_specs, target.parameter_specs)
        r_overlap = self._range_overlap(current_specs, target.parameter_specs)
        m_sim = self._metadata_similarity({}, target.metadata)
        return (
            self._W_PARAM_OVERLAP * p_overlap
            + self._W_RANGE_OVERLAP * r_overlap
            + self._W_METADATA * m_sim
        )

    def compute_similarity_with_meta(
        self,
        current_specs: list[dict[str, Any]],
        target: CampaignData,
        current_metadata: dict[str, Any] | None = None,
    ) -> float:
        """Like :meth:`compute_similarity` but accepts current metadata.

        Args:
            current_specs: Parameter specifications for the new campaign.
            target: A registered :class:`CampaignData` to compare against.
            current_metadata: Metadata for the current campaign.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        p_overlap = self._parameter_overlap(current_specs, target.parameter_specs)
        r_overlap = self._range_overlap(current_specs, target.parameter_specs)
        m_sim = self._metadata_similarity(
            current_metadata if current_metadata else {},
            target.metadata,
        )
        return (
            self._W_PARAM_OVERLAP * p_overlap
            + self._W_RANGE_OVERLAP * r_overlap
            + self._W_METADATA * m_sim
        )

    # -- warm start -------------------------------------------------------

    def warm_start_points(
        self,
        current_specs: list[dict[str, Any]],
        n_points: int = 5,
        min_similarity: float = 0.5,
        current_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract initial points from similar campaigns' top observations.

        For each sufficiently similar campaign the top 20 % of
        observations (by objective value, maximized) are collected,
        projected into the current parameter space, and returned.

        Projection handles dimension mismatches by:

        * Dropping parameters that do not exist in the current space.
        * Clipping continuous values to the current bounds.
        * Dropping categorical values not present in the current categories.

        Args:
            current_specs: Parameter specs for the new campaign.
            n_points: Maximum number of warm-start points to return.
            min_similarity: Minimum similarity threshold (0-1).
            current_metadata: Optional metadata for the current campaign.

        Returns:
            List of parameter dicts projected into the current space.
        """
        similar = self._find_similar_sorted(
            current_specs, min_similarity, current_metadata
        )

        if not similar:
            return []

        candidates: list[tuple[float, float, dict[str, Any]]] = []
        for campaign, sim in similar:
            if not campaign.observations:
                continue
            top_obs = self._get_top_observations(campaign.observations, fraction=0.2)
            projected = self._project_to_current_space(
                top_obs, campaign.parameter_specs, current_specs,
            )
            for obs in projected:
                obj = obs.get("objective", 0.0)
                candidates.append((sim, obj, obs))

        # Sort by similarity first, then by objective (descending)
        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)

        # Deduplicate (exact param dict match)
        seen: list[dict[str, Any]] = []
        result: list[dict[str, Any]] = []
        for _, _, obs in candidates:
            params_only = {
                k: v for k, v in obs.items() if k != "objective"
            }
            if params_only not in seen:
                seen.append(params_only)
                result.append(obs)
            if len(result) >= n_points:
                break

        self._log_transfer(
            event="warm_start",
            n_sources=len(similar),
            n_points=len(result),
            min_similarity=min_similarity,
        )

        return result

    # -- data pooling -----------------------------------------------------

    def transfer_data(
        self,
        current_specs: list[dict[str, Any]],
        min_similarity: float = 0.6,
        current_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Get compatible historical data with similarity-based weights.

        Only returns observations from campaigns whose parameter spaces
        are **fully compatible** with the current specification (all
        current parameter names exist in the source campaign, though
        the source may have extra parameters that are dropped).

        Each returned observation dict receives an additional key
        ``_transfer_weight`` proportional to the source campaign's
        similarity score, in ``(0, 1]``.

        Args:
            current_specs: Parameter specs for the current campaign.
            min_similarity: Minimum similarity threshold.
            current_metadata: Optional metadata for similarity scoring.

        Returns:
            List of observation dicts with ``_transfer_weight`` field.
        """
        current_names = {s["name"] for s in current_specs}
        similar = self._find_similar_sorted(
            current_specs, min_similarity, current_metadata,
        )

        result: list[dict[str, Any]] = []
        for campaign, sim in similar:
            source_names = campaign.param_names
            # Compatibility: all *current* params must exist in source
            if not current_names.issubset(source_names):
                continue

            projected = self._project_to_current_space(
                campaign.observations,
                campaign.parameter_specs,
                current_specs,
            )
            for obs in projected:
                obs["_transfer_weight"] = sim
                result.append(obs)

        self._log_transfer(
            event="data_pool",
            n_sources=len(similar),
            n_observations=len(result),
            min_similarity=min_similarity,
        )

        return result

    # -- search -----------------------------------------------------------

    def find_similar_campaigns(
        self,
        current_specs: list[dict[str, Any]],
        min_similarity: float = 0.5,
        current_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(campaign_id, similarity)`` pairs sorted by similarity.

        Args:
            current_specs: Parameter specs for the current campaign.
            min_similarity: Minimum similarity threshold.
            current_metadata: Optional metadata for similarity scoring.

        Returns:
            List of ``(campaign_id, similarity)`` tuples in descending
            similarity order.
        """
        pairs = self._find_similar_sorted(
            current_specs, min_similarity, current_metadata,
        )
        return [(c.campaign_id, s) for c, s in pairs]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_similar_sorted(
        self,
        current_specs: list[dict[str, Any]],
        min_similarity: float,
        current_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[CampaignData, float]]:
        """Find and sort campaigns by similarity above *min_similarity*."""
        results: list[tuple[CampaignData, float]] = []
        for campaign in self._history:
            sim = self.compute_similarity_with_meta(
                current_specs, campaign, current_metadata,
            )
            if sim >= min_similarity:
                results.append((campaign, sim))
        results.sort(key=lambda t: t[1], reverse=True)
        return results

    # -- similarity components -------------------------------------------

    @staticmethod
    def _parameter_overlap(
        specs_a: list[dict[str, Any]],
        specs_b: list[dict[str, Any]],
    ) -> float:
        """Jaccard similarity of parameter name sets.

        Returns:
            ``|A & B| / |A | B|``, or 0.0 when both sets are empty.
        """
        names_a = {s["name"] for s in specs_a}
        names_b = {s["name"] for s in specs_b}
        if not names_a and not names_b:
            return 0.0
        intersection = names_a & names_b
        union = names_a | names_b
        return len(intersection) / len(union)

    @staticmethod
    def _range_overlap(
        specs_a: list[dict[str, Any]],
        specs_b: list[dict[str, Any]],
    ) -> float:
        """Average range intersection-over-union for shared continuous params.

        For each parameter name present in both spec lists:

        * If continuous (has ``lower`` and ``upper``): compute
          ``intersection_length / union_length``.
        * If categorical (has ``categories``): compute Jaccard on
          the category sets.

        Returns the mean overlap across shared parameters, or 0.0
        when there are no shared parameters.
        """
        by_name_a = {s["name"]: s for s in specs_a}
        by_name_b = {s["name"]: s for s in specs_b}
        shared = set(by_name_a) & set(by_name_b)

        if not shared:
            return 0.0

        overlaps: list[float] = []
        for name in shared:
            sa = by_name_a[name]
            sb = by_name_b[name]

            # Continuous parameters
            if "lower" in sa and "upper" in sa and "lower" in sb and "upper" in sb:
                lo_a, hi_a = float(sa["lower"]), float(sa["upper"])
                lo_b, hi_b = float(sb["lower"]), float(sb["upper"])
                inter_lo = max(lo_a, lo_b)
                inter_hi = min(hi_a, hi_b)
                intersection = max(0.0, inter_hi - inter_lo)
                union_lo = min(lo_a, lo_b)
                union_hi = max(hi_a, hi_b)
                union = max(union_hi - union_lo, 1e-12)
                overlaps.append(intersection / union)

            # Categorical parameters
            elif "categories" in sa and "categories" in sb:
                cats_a = set(sa["categories"])
                cats_b = set(sb["categories"])
                if not cats_a and not cats_b:
                    overlaps.append(0.0)
                else:
                    overlaps.append(
                        len(cats_a & cats_b) / len(cats_a | cats_b)
                    )
            else:
                # Mixed or unknown types: partial credit
                overlaps.append(0.5)

        return sum(overlaps) / len(overlaps)

    @staticmethod
    def _metadata_similarity(
        meta_a: dict[str, Any],
        meta_b: dict[str, Any],
    ) -> float:
        """Key-and-value overlap ratio for metadata dicts.

        Computes the fraction of keys present in both dicts whose
        values are equal.  Falls back to key-only Jaccard when no
        keys are shared.

        Returns:
            Similarity score in ``[0, 1]``.  Returns 0.0 when both
            dicts are empty.
        """
        if not meta_a and not meta_b:
            return 0.0
        if not meta_a or not meta_b:
            return 0.0

        keys_a = set(meta_a)
        keys_b = set(meta_b)
        union_keys = keys_a | keys_b
        if not union_keys:
            return 0.0

        shared_keys = keys_a & keys_b
        if not shared_keys:
            # No overlapping keys at all
            return 0.0

        # Count matching key-value pairs
        matching = sum(1 for k in shared_keys if meta_a[k] == meta_b[k])
        return matching / len(union_keys)

    # -- projection ------------------------------------------------------

    @staticmethod
    def _project_to_current_space(
        observations: list[dict[str, Any]],
        source_specs: list[dict[str, Any]],
        target_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Project observations from *source* space into *target* space.

        * Parameters in the target but not in the source are skipped
          (the observation is dropped if any *required* target param
          is missing).
        * Continuous values are clipped to the target bounds.
        * Categorical values not in the target categories are dropped.
        * Extra source parameters are removed.
        * The ``"objective"`` key is preserved when present.

        Returns:
            List of projected observation dicts.  Observations that
            cannot be fully projected (missing required parameters)
            are omitted.
        """
        target_by_name = {s["name"]: s for s in target_specs}
        target_names = set(target_by_name)
        source_names = {s["name"] for s in source_specs}

        projected: list[dict[str, Any]] = []
        for obs in observations:
            new_obs: dict[str, Any] = {}
            valid = True

            for name, spec in target_by_name.items():
                if name not in obs:
                    # Parameter not in observation -- cannot project
                    valid = False
                    break

                value = obs[name]

                # Clip continuous to target bounds
                if "lower" in spec and "upper" in spec:
                    try:
                        value = float(value)
                        value = max(float(spec["lower"]), min(float(spec["upper"]), value))
                    except (TypeError, ValueError):
                        valid = False
                        break

                # Validate categorical against target categories
                elif "categories" in spec:
                    if value not in spec["categories"]:
                        valid = False
                        break

                new_obs[name] = value

            if not valid:
                continue

            # Preserve objective
            if "objective" in obs:
                new_obs["objective"] = obs["objective"]

            projected.append(new_obs)

        return projected

    @staticmethod
    def _get_top_observations(
        observations: list[dict[str, Any]],
        fraction: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Return the top *fraction* of observations by objective value.

        Observations are assumed to be *maximized* (higher is better).
        At least one observation is always returned if the list is
        non-empty.

        Args:
            observations: Observation dicts with ``"objective"`` key.
            fraction: Fraction of top observations to return (0, 1].

        Returns:
            Sorted list of top observations (best first).
        """
        if not observations:
            return []

        # Filter to observations that have an objective value
        with_obj = [o for o in observations if "objective" in o]
        if not with_obj:
            return []

        sorted_obs = sorted(
            with_obj,
            key=lambda o: float(o["objective"]),
            reverse=True,
        )
        n = max(1, int(len(sorted_obs) * fraction))
        return sorted_obs[:n]

    # -- logging ----------------------------------------------------------

    def _log_transfer(self, **kwargs: Any) -> None:
        """Append an entry to the transfer log."""
        self._transfer_log.append(dict(kwargs))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the engine state to a plain dict."""
        return {
            "history": [c.to_dict() for c in self._history],
            "transfer_log": [dict(e) for e in self._transfer_log],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransferLearningEngine:
        """Restore engine state from a dict produced by :meth:`to_dict`."""
        engine = cls()
        for c_data in data.get("history", []):
            engine._history.append(CampaignData.from_dict(c_data))
        engine._transfer_log = list(data.get("transfer_log", []))
        return engine

    def __repr__(self) -> str:
        return (
            f"TransferLearningEngine(n_campaigns={self.n_campaigns}, "
            f"transfer_events={len(self._transfer_log)})"
        )

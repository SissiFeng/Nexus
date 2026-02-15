"""Strategy learner for cross-campaign backend selection.

Analyzes historical campaign outcomes to learn which optimization backends
work best for which problem fingerprints.  Uses exact and similarity-based
matching with recency weighting to rank backends for new problems.
"""

from __future__ import annotations

from optimization_copilot.core.models import ProblemFingerprint
from optimization_copilot.meta_learning.models import MetaLearningConfig
from optimization_copilot.meta_learning.experience_store import ExperienceStore


class StrategyLearner:
    """Learns backend rankings from cross-campaign experience records."""

    def __init__(
        self,
        experience_store: ExperienceStore,
        config: MetaLearningConfig | None = None,
    ) -> None:
        self._store = experience_store
        self._config = config or MetaLearningConfig()

    # ── Public API ─────────────────────────────────────────

    def rank_backends(
        self, fingerprint: ProblemFingerprint
    ) -> list[tuple[str, float]]:
        """Return (backend_name, score) pairs sorted by score descending.

        Algorithm
        ---------
        1. Collect exact-match records (weight=1.0).
        2. If not enough, supplement with similar records whose weight is
           ``similarity * (1 - similarity_decay)``.
        3. Apply recency weighting via the experience store.
        4. For each backend, compute a composite score:
           ``weighted_avg(sample_efficiency) - weighted_avg(regret) - weighted_avg(failure_rate)``
        5. If total record count < ``min_experiences_for_learning``, return
           an empty list (cold-start guard).
        """
        weighted_records = self._collect_weighted_records(fingerprint)

        if len(weighted_records) < self._config.min_experiences_for_learning:
            return []

        return self._aggregate_scores(weighted_records)

    def has_enough_data(self, fingerprint: ProblemFingerprint) -> bool:
        """Check whether enough experience exists for learning.

        Returns ``True`` when the number of exact-match *plus* similar
        records (similarity > 0.5) meets the configured threshold.
        """
        fp_key = str(fingerprint.to_tuple())
        exact = self._store.get_by_fingerprint(fp_key)
        count = len(exact)

        if count >= self._config.min_experiences_for_learning:
            return True

        similar = self._store.get_similar(fingerprint, max_results=50)
        for _record, sim in similar:
            if sim > 0.5:
                count += 1

        # Subtract exact matches that also appear in similar results
        # (get_similar returns *all* records including exact matches).
        # Exact matches have similarity=1.0 which is > 0.5, so they are
        # already counted in the loop above.  We only need to count once.
        # Reset and recount properly:
        exact_keys = {r.outcome.campaign_id for r in exact}
        count = len(exact)
        for record, sim in similar:
            if sim > 0.5 and record.outcome.campaign_id not in exact_keys:
                count += 1

        return count >= self._config.min_experiences_for_learning

    def get_backend_stats(
        self, fingerprint: ProblemFingerprint, backend_name: str
    ) -> dict:
        """Return aggregated stats for *backend_name* under *fingerprint*.

        Keys
        ----
        n_campaigns : int
        avg_sample_efficiency : float
        avg_regret : float
        avg_failure_rate : float
        avg_convergence_iteration : float | None
            Average across campaigns where convergence was reached.
        """
        weighted_records = self._collect_weighted_records(fingerprint)

        n_campaigns = 0
        total_se = 0.0
        total_regret = 0.0
        total_fr = 0.0
        convergence_vals: list[float] = []

        for record, _weight in weighted_records:
            for bp in record.outcome.backend_performances:
                if bp.backend_name != backend_name:
                    continue
                n_campaigns += 1
                total_se += bp.sample_efficiency
                total_regret += bp.regret
                total_fr += bp.failure_rate
                if bp.convergence_iteration is not None:
                    convergence_vals.append(float(bp.convergence_iteration))

        if n_campaigns == 0:
            return {
                "n_campaigns": 0,
                "avg_sample_efficiency": 0.0,
                "avg_regret": 0.0,
                "avg_failure_rate": 0.0,
                "avg_convergence_iteration": None,
            }

        return {
            "n_campaigns": n_campaigns,
            "avg_sample_efficiency": total_se / n_campaigns,
            "avg_regret": total_regret / n_campaigns,
            "avg_failure_rate": total_fr / n_campaigns,
            "avg_convergence_iteration": (
                sum(convergence_vals) / len(convergence_vals)
                if convergence_vals
                else None
            ),
        }

    # ── Internals ──────────────────────────────────────────

    def _collect_weighted_records(
        self, fingerprint: ProblemFingerprint
    ) -> list[tuple["_ExperienceRecord", float]]:
        """Gather exact and similar records with combined weights.

        Exact matches receive ``weight = 1.0``.
        Similar (non-exact) matches receive
        ``weight = similarity * (1 - similarity_decay)``.
        Both are further multiplied by the recency weight.

        Returns a list of ``(ExperienceRecord, combined_weight)`` tuples.
        """
        from optimization_copilot.meta_learning.models import ExperienceRecord as _ExperienceRecord  # noqa: F811

        fp_key = str(fingerprint.to_tuple())
        exact_records = self._store.get_by_fingerprint(fp_key)
        exact_ids = {r.outcome.campaign_id for r in exact_records}

        # Determine latest timestamp across all records for recency calc.
        all_records = self._store.get_all()
        if all_records:
            latest_ts = max(r.outcome.timestamp for r in all_records)
        else:
            latest_ts = 0.0

        weighted: list[tuple[_ExperienceRecord, float]] = []

        # Exact matches: base weight = 1.0
        for record in exact_records:
            recency = self._store.recency_weight(record, latest_ts)
            weighted.append((record, 1.0 * recency))

        # Check if we need similar records to supplement.
        if len(exact_records) < self._config.min_experiences_for_learning:
            similar = self._store.get_similar(fingerprint, max_results=20)
            for record, similarity in similar:
                if record.outcome.campaign_id in exact_ids:
                    continue  # already counted as exact
                base_weight = similarity * (1.0 - self._config.similarity_decay)
                recency = self._store.recency_weight(record, latest_ts)
                weighted.append((record, base_weight * recency))

        return weighted

    def _aggregate_scores(
        self,
        weighted_records: list[tuple["_ExperienceRecord", float]],
    ) -> list[tuple[str, float]]:
        """Compute per-backend composite scores from weighted records.

        For each backend found across all ``BackendPerformance`` entries,
        the score is::

            weighted_avg(sample_efficiency)
            - weighted_avg(regret)
            - weighted_avg(failure_rate)

        Backends are sorted by score descending.
        """
        # Accumulate per-backend weighted sums.
        # Keys: backend_name -> {sum_se, sum_regret, sum_fr, sum_weight}
        accum: dict[str, dict[str, float]] = {}

        for record, weight in weighted_records:
            for bp in record.outcome.backend_performances:
                name = bp.backend_name
                if name not in accum:
                    accum[name] = {
                        "sum_se": 0.0,
                        "sum_regret": 0.0,
                        "sum_fr": 0.0,
                        "sum_weight": 0.0,
                    }
                accum[name]["sum_se"] += bp.sample_efficiency * weight
                accum[name]["sum_regret"] += bp.regret * weight
                accum[name]["sum_fr"] += bp.failure_rate * weight
                accum[name]["sum_weight"] += weight

        results: list[tuple[str, float]] = []
        for name, vals in accum.items():
            w = vals["sum_weight"]
            if w <= 0.0:
                continue
            avg_se = vals["sum_se"] / w
            avg_regret = vals["sum_regret"] / w
            avg_fr = vals["sum_fr"] / w
            score = avg_se - avg_regret - avg_fr
            results.append((name, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

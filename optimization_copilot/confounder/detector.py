"""Automatic confounder detection via metadata-KPI correlation scanning."""

from __future__ import annotations

import math

from optimization_copilot.core.models import CampaignSnapshot, Observation
from optimization_copilot.confounder.models import (
    ConfounderPolicy,
    ConfounderSpec,
)


class ConfounderDetector:
    """Scan observation metadata for numeric columns that correlate with KPIs.

    Columns whose absolute Pearson correlation with any objective exceeds the
    given threshold are flagged as candidate confounders with a default
    ``HIGH_RISK_FLAG`` policy.
    """

    def detect(
        self,
        snapshot: CampaignSnapshot,
        threshold: float = 0.3,
    ) -> list[ConfounderSpec]:
        """Detect candidate confounders from observation metadata.

        Parameters
        ----------
        snapshot : CampaignSnapshot
            Campaign state whose observations will be scanned.
        threshold : float
            Absolute Pearson |r| above which a metadata column is considered
            a confounder candidate.

        Returns
        -------
        list[ConfounderSpec]
            One spec per detected confounder column, using
            ``ConfounderPolicy.HIGH_RISK_FLAG`` as the default policy.
        """
        if not snapshot.observations:
            return []

        # Collect all metadata keys and attempt to extract numeric values.
        meta_keys: set[str] = set()
        for obs in snapshot.observations:
            meta_keys.update(obs.metadata.keys())

        # Exclude keys that are already formal parameters.
        param_names = {p.name for p in snapshot.parameter_specs}
        candidate_keys = sorted(meta_keys - param_names)

        results: list[ConfounderSpec] = []

        for key in candidate_keys:
            # Extract numeric pairs (confounder, kpi) for each objective.
            is_confounder = False
            max_corr = 0.0

            for obj_name in snapshot.objective_names:
                xs: list[float] = []
                ys: list[float] = []
                for obs in snapshot.observations:
                    raw = obs.metadata.get(key)
                    kpi = obs.kpi_values.get(obj_name)
                    if raw is not None and kpi is not None:
                        try:
                            xs.append(float(raw))
                            ys.append(float(kpi))
                        except (TypeError, ValueError):
                            continue

                if len(xs) < 3:
                    continue

                corr = abs(self._pearson_correlation(xs, ys))
                if corr > max_corr:
                    max_corr = corr
                if corr > threshold:
                    is_confounder = True

            if is_confounder:
                results.append(ConfounderSpec(
                    column_name=key,
                    policy=ConfounderPolicy.HIGH_RISK_FLAG,
                    metadata={"max_abs_correlation": round(max_corr, 6)},
                ))

        return results

    # -- Static helpers -----------------------------------------------------

    @staticmethod
    def _pearson_correlation(x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient between two lists.

        Returns 0.0 if either series has zero variance or the lists are
        shorter than 2 elements.
        """
        n = len(x)
        if n < 2 or n != len(y):
            return 0.0

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        x_std = math.sqrt(sum((xi - x_mean) ** 2 for xi in x) / n)
        y_std = math.sqrt(sum((yi - y_mean) ** 2 for yi in y) / n)

        if x_std < 1e-12 or y_std < 1e-12:
            return 0.0

        cov = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / n
        return cov / (x_std * y_std)

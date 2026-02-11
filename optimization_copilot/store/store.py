"""In-memory Canonical Experiment Store with JSON serialization.

Provides unified storage for experimental data including parameters, KPIs,
metadata, and artifacts.  Designed to bridge to CampaignSnapshot for engine
consumption via StoreBridge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.store.models import Artifact, ArtifactType, Experiment


# ── Query / Summary dataclasses ───────────────────────────────


@dataclass
class StoreQuery:
    """Filter criteria for querying experiments.

    All fields are optional.  When multiple fields are set they are
    combined with AND logic.
    """

    campaign_id: str | None = None
    iteration: int | None = None
    iteration_range: tuple[int, int] | None = None  # (start, end) inclusive
    has_artifact_type: ArtifactType | None = None
    only_successful: bool = False


@dataclass
class StoreSummary:
    """Summary statistics for the store or a campaign within it."""

    n_experiments: int
    n_campaigns: int
    campaign_ids: list[str]
    n_artifacts: int
    parameter_names: list[str]
    kpi_names: list[str]
    iteration_range: tuple[int, int] | None  # (min, max) or None if empty


# ── Store ──────────────────────────────────────────────────────


class ExperimentStore:
    """In-memory canonical experiment store with JSON serialization."""

    def __init__(self) -> None:
        self._experiments: dict[str, Experiment] = {}  # experiment_id → Experiment

    # ── mutation ───────────────────────────────────────────

    def add_experiment(self, experiment: Experiment) -> None:
        """Add a single experiment.  Raises ValueError on duplicate ID."""
        if experiment.experiment_id in self._experiments:
            raise ValueError(
                f"Duplicate experiment_id: '{experiment.experiment_id}'"
            )
        self._experiments[experiment.experiment_id] = experiment

    def add_experiments(self, experiments: list[Experiment]) -> None:
        """Add multiple experiments."""
        for exp in experiments:
            self.add_experiment(exp)

    def attach_artifact(self, experiment_id: str, artifact: Artifact) -> None:
        """Attach an artifact to an existing experiment.

        Raises KeyError if experiment_id not found.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise KeyError(f"Experiment not found: '{experiment_id}'")
        exp.artifacts.append(artifact)

    # ── queries ────────────────────────────────────────────

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Get a single experiment by ID.  Raises KeyError if not found."""
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise KeyError(f"Experiment not found: '{experiment_id}'")
        return exp

    def query(self, q: StoreQuery) -> list[Experiment]:
        """Query experiments with filters.  Returns matches sorted by iteration."""
        results: list[Experiment] = []
        for exp in self._experiments.values():
            if q.campaign_id is not None and exp.campaign_id != q.campaign_id:
                continue
            if q.iteration is not None and exp.iteration != q.iteration:
                continue
            if q.iteration_range is not None:
                lo, hi = q.iteration_range
                if exp.iteration < lo or exp.iteration > hi:
                    continue
            if q.has_artifact_type is not None:
                if not any(
                    a.artifact_type == q.has_artifact_type for a in exp.artifacts
                ):
                    continue
            if q.only_successful and exp.is_failure:
                continue
            results.append(exp)
        results.sort(key=lambda e: e.iteration)
        return results

    def get_by_campaign(self, campaign_id: str) -> list[Experiment]:
        """Get all experiments for a campaign, sorted by iteration."""
        return self.query(StoreQuery(campaign_id=campaign_id))

    def get_by_iteration(
        self, campaign_id: str, iteration: int
    ) -> list[Experiment]:
        """Get experiments for a specific campaign + iteration."""
        return self.query(
            StoreQuery(campaign_id=campaign_id, iteration=iteration)
        )

    def list_campaigns(self) -> list[str]:
        """Return sorted list of unique campaign IDs."""
        return sorted({exp.campaign_id for exp in self._experiments.values()})

    # ── introspection ──────────────────────────────────────

    def summary(self, campaign_id: str | None = None) -> StoreSummary:
        """Compute summary statistics, optionally filtered by campaign."""
        if campaign_id is not None:
            experiments = self.get_by_campaign(campaign_id)
            campaigns = [campaign_id] if experiments else []
        else:
            experiments = list(self._experiments.values())
            campaigns = self.list_campaigns()

        n_artifacts = sum(len(exp.artifacts) for exp in experiments)

        param_names: set[str] = set()
        kpi_names: set[str] = set()
        iterations: list[int] = []

        for exp in experiments:
            param_names.update(exp.parameters.keys())
            kpi_names.update(exp.kpi_values.keys())
            iterations.append(exp.iteration)

        iter_range: tuple[int, int] | None = None
        if iterations:
            iter_range = (min(iterations), max(iterations))

        return StoreSummary(
            n_experiments=len(experiments),
            n_campaigns=len(campaigns),
            campaign_ids=campaigns,
            n_artifacts=n_artifacts,
            parameter_names=sorted(param_names),
            kpi_names=sorted(kpi_names),
            iteration_range=iter_range,
        )

    def column_names(self, campaign_id: str) -> dict[str, list[str]]:
        """Return parameter, KPI, and metadata column names for a campaign."""
        experiments = self.get_by_campaign(campaign_id)
        param_names: set[str] = set()
        kpi_names: set[str] = set()
        meta_names: set[str] = set()

        for exp in experiments:
            param_names.update(exp.parameters.keys())
            kpi_names.update(exp.kpi_values.keys())
            meta_names.update(exp.metadata.keys())

        return {
            "parameters": sorted(param_names),
            "kpis": sorted(kpi_names),
            "metadata": sorted(meta_names),
        }

    def column_values(self, campaign_id: str, column_name: str) -> list[Any]:
        """Return all values for a named column across experiments in a campaign.

        Searches parameters, kpi_values, and metadata in that order.
        """
        experiments = self.get_by_campaign(campaign_id)
        values: list[Any] = []
        for exp in experiments:
            if column_name in exp.parameters:
                values.append(exp.parameters[column_name])
            elif column_name in exp.kpi_values:
                values.append(exp.kpi_values[column_name])
            elif column_name in exp.metadata:
                values.append(exp.metadata[column_name])
        return values

    # ── serialization ──────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiments": [
                exp.to_dict()
                for exp in sorted(
                    self._experiments.values(),
                    key=lambda e: (e.campaign_id, e.iteration),
                )
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentStore:
        store = cls()
        for exp_data in data.get("experiments", []):
            store.add_experiment(Experiment.from_dict(exp_data))
        return store

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ExperimentStore:
        return cls.from_dict(json.loads(json_str))

    def __len__(self) -> int:
        return len(self._experiments)

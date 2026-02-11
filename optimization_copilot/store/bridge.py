"""Bridge from Canonical Experiment Store to core model objects.

Translates ExperimentStore data into runtime data structures used by the
optimization engine (CampaignSnapshot, Observation, etc.).  Analogous to
dsl.bridge.SpecBridge but operates on store data rather than DSL spec.
"""

from __future__ import annotations

from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
)
from optimization_copilot.dsl.bridge import SpecBridge
from optimization_copilot.dsl.spec import OptimizationSpec
from optimization_copilot.feature_extraction.extractors import CurveData
from optimization_copilot.store.models import ArtifactType, Experiment
from optimization_copilot.store.store import ExperimentStore


class StoreBridge:
    """Converts ExperimentStore data into core model objects.

    All methods are static — no mutable state is held.
    """

    @staticmethod
    def to_observations(
        store: ExperimentStore,
        campaign_id: str,
    ) -> list[Observation]:
        """Convert store experiments to core Observation objects.

        Strips artifacts (the engine does not consume them).
        """
        experiments = store.get_by_campaign(campaign_id)
        observations: list[Observation] = []

        for exp in experiments:
            obs = Observation(
                iteration=exp.iteration,
                parameters=dict(exp.parameters),
                kpi_values=dict(exp.kpi_values),
                qc_passed=exp.qc_passed,
                is_failure=exp.is_failure,
                failure_reason=exp.failure_reason,
                timestamp=exp.timestamp,
                metadata=dict(exp.metadata),
            )
            observations.append(obs)

        return observations

    @staticmethod
    def to_campaign_snapshot(
        store: ExperimentStore,
        spec: OptimizationSpec,
        campaign_id: str | None = None,
    ) -> CampaignSnapshot:
        """Create a CampaignSnapshot from store data + spec.

        Uses SpecBridge for parameter_specs and objective info, then
        populates observations from the store.

        Parameters
        ----------
        store:
            The experiment store.
        spec:
            The optimization spec (defines param types, objectives).
        campaign_id:
            Campaign to pull observations from.  Defaults to
            spec.campaign_id.
        """
        cid = campaign_id or spec.campaign_id
        observations = StoreBridge.to_observations(store, cid)

        # Reuse SpecBridge for parameter specs and objective info.
        snapshot = SpecBridge.to_campaign_snapshot(spec, observations)

        return snapshot

    @staticmethod
    def extract_curves(
        store: ExperimentStore,
        campaign_id: str,
        artifact_name: str | None = None,
    ) -> list[CurveData]:
        """Extract CurveData artifacts from the store.

        Useful for feeding into FeatureExtractorRegistry.

        Parameters
        ----------
        store:
            The experiment store.
        campaign_id:
            Campaign to pull curves from.
        artifact_name:
            Optional filter: only return artifacts with this name.
        """
        experiments = store.get_by_campaign(campaign_id)
        curves: list[CurveData] = []

        for exp in experiments:
            for artifact in exp.artifacts:
                if artifact.artifact_type != ArtifactType.CURVE:
                    continue
                if artifact_name is not None and artifact.name != artifact_name:
                    continue
                # Reconstruct CurveData from the stored dict.
                data = artifact.data
                if isinstance(data, dict):
                    x_values = data.get("x_values", [])
                    y_values = data.get("y_values", [])
                    metadata = data.get("metadata", {})
                    curves.append(
                        CurveData(
                            x_values=list(x_values),
                            y_values=list(y_values),
                            metadata=dict(metadata),
                        )
                    )

        return curves

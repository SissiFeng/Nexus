"""Standardized data import/export for optimization campaigns.

Supports CSV, JSON, JSON-LD, and SDF formats for interoperability
with external lab systems and data pipelines.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)


# ── Enums ──────────────────────────────────────────────


class DataFormat(str, Enum):
    """Supported data interchange formats."""

    CSV = "csv"
    JSON = "json"
    JSONLD = "jsonld"
    SDF = "sdf"


# ── Dataclasses ────────────────────────────────────────


@dataclass
class ColumnMapping:
    """Describes how CSV columns map to campaign data fields.

    Parameters
    ----------
    parameter_columns:
        Column names that correspond to optimization parameters.
    objective_columns:
        Column names that correspond to objective/KPI values.
    metadata_columns:
        Column names that correspond to observation metadata.
    iteration_column:
        Column name for the iteration index. If ``None``, iterations
        are assigned sequentially starting from 0.
    """

    parameter_columns: list[str]
    objective_columns: list[str]
    metadata_columns: list[str] = field(default_factory=list)
    iteration_column: str | None = None


# ── Exporter ───────────────────────────────────────────


class CampaignExporter:
    """Export a CampaignSnapshot to various file formats."""

    def export(
        self,
        snapshot: CampaignSnapshot,
        format: DataFormat,
        path: str,
    ) -> None:
        """Write a campaign snapshot to a file.

        Parameters
        ----------
        snapshot:
            The campaign snapshot to export.
        format:
            Target data format.
        path:
            File path to write to.
        """
        content = self.export_string(snapshot, format)
        with open(path, "w", newline="") as f:
            f.write(content)

    def export_string(
        self,
        snapshot: CampaignSnapshot,
        format: DataFormat,
    ) -> str:
        """Serialize a campaign snapshot to a string.

        Parameters
        ----------
        snapshot:
            The campaign snapshot to export.
        format:
            Target data format.

        Returns
        -------
        str
            The serialized campaign data.

        Raises
        ------
        ValueError
            If the format is not supported for string export.
        """
        if format == DataFormat.CSV:
            return self._export_csv(snapshot)
        elif format == DataFormat.JSON:
            return self._export_json(snapshot)
        elif format == DataFormat.JSONLD:
            return self._export_jsonld(snapshot)
        else:
            raise ValueError(f"Unsupported export format: {format.value}")

    def _export_csv(self, snapshot: CampaignSnapshot) -> str:
        """Export snapshot as CSV with one row per observation."""
        param_names = snapshot.parameter_names
        kpi_names = snapshot.objective_names

        # Collect all metadata keys across observations.
        meta_keys: list[str] = []
        seen_meta: set[str] = set()
        for obs in snapshot.observations:
            for key in obs.metadata:
                if key not in seen_meta:
                    meta_keys.append(key)
                    seen_meta.add(key)

        header = (
            ["iteration"]
            + param_names
            + kpi_names
            + meta_keys
            + ["qc_passed", "is_failure", "failure_reason", "timestamp"]
        )

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(header)

        for obs in snapshot.observations:
            row: list[Any] = [obs.iteration]
            for pname in param_names:
                row.append(obs.parameters.get(pname, ""))
            for kname in kpi_names:
                row.append(obs.kpi_values.get(kname, ""))
            for mkey in meta_keys:
                row.append(obs.metadata.get(mkey, ""))
            row.extend([obs.qc_passed, obs.is_failure, obs.failure_reason or "", obs.timestamp])
            writer.writerow(row)

        return output.getvalue()

    def _export_json(self, snapshot: CampaignSnapshot) -> str:
        """Export snapshot as JSON with schema version."""
        payload = {
            "schema_version": "1.0",
            "campaign": snapshot.to_dict(),
        }
        return json.dumps(payload, indent=2, default=str)

    def _export_jsonld(self, snapshot: CampaignSnapshot) -> str:
        """Export snapshot as JSON-LD with semantic context."""
        payload = {
            "@context": {
                "@vocab": "https://schema.org/",
                "campaign": "https://w3id.org/sdl#Campaign",
                "observation": "https://w3id.org/sdl#Observation",
            },
            "@type": "campaign",
            "schema_version": "1.0",
            "campaign": snapshot.to_dict(),
        }
        return json.dumps(payload, indent=2, default=str)


# ── Importer ───────────────────────────────────────────


class CampaignImporter:
    """Import campaign data from various file formats."""

    def import_csv(
        self,
        path: str,
        mapping: ColumnMapping,
        campaign_id: str = "imported",
    ) -> CampaignSnapshot:
        """Import a CSV file into a CampaignSnapshot.

        Parameters
        ----------
        path:
            Path to the CSV file.
        mapping:
            Column mapping describing how to interpret CSV columns.
        campaign_id:
            Identifier for the created campaign.

        Returns
        -------
        CampaignSnapshot
            The reconstructed campaign snapshot.
        """
        with open(path, "r", newline="") as f:
            return self._parse_csv(f, mapping, campaign_id)

    def import_csv_string(
        self,
        data: str,
        mapping: ColumnMapping,
        campaign_id: str = "imported",
    ) -> CampaignSnapshot:
        """Import a CSV string into a CampaignSnapshot.

        Parameters
        ----------
        data:
            The CSV content as a string.
        mapping:
            Column mapping describing how to interpret CSV columns.
        campaign_id:
            Identifier for the created campaign.

        Returns
        -------
        CampaignSnapshot
            The reconstructed campaign snapshot.
        """
        return self._parse_csv(io.StringIO(data), mapping, campaign_id)

    def _parse_csv(
        self,
        source: io.TextIOBase | io.StringIO,
        mapping: ColumnMapping,
        campaign_id: str,
    ) -> CampaignSnapshot:
        """Parse CSV data from a file-like object."""
        reader = csv.DictReader(source)
        observations: list[Observation] = []
        param_min: dict[str, float] = {}
        param_max: dict[str, float] = {}

        for row_idx, row in enumerate(reader):
            # Determine iteration.
            if mapping.iteration_column and mapping.iteration_column in row:
                iteration = int(row[mapping.iteration_column])
            else:
                iteration = row_idx

            # Extract parameters.
            parameters: dict[str, Any] = {}
            for col in mapping.parameter_columns:
                raw = row.get(col, "")
                try:
                    val = float(raw)
                    parameters[col] = val
                    # Track min/max for ParameterSpec inference.
                    if col not in param_min or val < param_min[col]:
                        param_min[col] = val
                    if col not in param_max or val > param_max[col]:
                        param_max[col] = val
                except (ValueError, TypeError):
                    parameters[col] = raw

            # Extract KPI values.
            kpi_values: dict[str, float] = {}
            for col in mapping.objective_columns:
                raw = row.get(col, "")
                try:
                    kpi_values[col] = float(raw)
                except (ValueError, TypeError):
                    pass

            # Extract metadata.
            metadata: dict[str, Any] = {}
            for col in mapping.metadata_columns:
                if col in row:
                    metadata[col] = row[col]

            observations.append(
                Observation(
                    iteration=iteration,
                    parameters=parameters,
                    kpi_values=kpi_values,
                    metadata=metadata,
                )
            )

        # Infer ParameterSpecs from data.
        parameter_specs = [
            ParameterSpec(
                name=col,
                type=VariableType.CONTINUOUS,
                lower=param_min.get(col),
                upper=param_max.get(col),
            )
            for col in mapping.parameter_columns
        ]

        # Objective directions default to "minimize".
        objective_directions = ["minimize"] * len(mapping.objective_columns)

        return CampaignSnapshot(
            campaign_id=campaign_id,
            parameter_specs=parameter_specs,
            observations=observations,
            objective_names=list(mapping.objective_columns),
            objective_directions=objective_directions,
            current_iteration=len(observations),
        )

    def import_json(self, path: str) -> CampaignSnapshot:
        """Import a JSON file into a CampaignSnapshot.

        Expects the JSON structure produced by ``CampaignExporter.export``
        with a top-level ``campaign`` key.

        Parameters
        ----------
        path:
            Path to the JSON file.

        Returns
        -------
        CampaignSnapshot
            The reconstructed campaign snapshot.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return CampaignSnapshot.from_dict(data["campaign"])

    def import_json_string(self, data: str) -> CampaignSnapshot:
        """Import a JSON string into a CampaignSnapshot.

        Parameters
        ----------
        data:
            The JSON content as a string.

        Returns
        -------
        CampaignSnapshot
            The reconstructed campaign snapshot.
        """
        parsed = json.loads(data)
        return CampaignSnapshot.from_dict(parsed["campaign"])

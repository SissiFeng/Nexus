"""Data Ingestion Agent — orchestrates parsing, profiling, and storage.

Parses CSV/JSON data, infers column roles, detects anomalies, and
converts rows into Experiment objects stored in an ExperimentStore.
"""

from __future__ import annotations

import time
from typing import Any

from optimization_copilot.ingestion.models import (
    ColumnProfile,
    ColumnRole,
    IngestionReport,
)
from optimization_copilot.ingestion.parsers import CSVParser, JSONParser
from optimization_copilot.ingestion.profiler import (
    AnomalyDetector,
    ColumnProfiler,
    MissingDataAnalyzer,
    RoleInferenceEngine,
    _coerce_value,
    _try_float,
    _try_int,
)
from optimization_copilot.store.models import Experiment
from optimization_copilot.store.store import ExperimentStore


class DataIngestionAgent:
    """Orchestrate data ingestion: parse → profile → store.

    Usage::

        agent = DataIngestionAgent()
        store = ExperimentStore()
        report = agent.ingest_csv("data.csv", store, campaign_id="exp1")
    """

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        null_threshold: float = 0.3,
    ) -> None:
        self._profiler = ColumnProfiler()
        self._role_engine = RoleInferenceEngine()
        self._anomaly_detector = AnomalyDetector(
            iqr_multiplier=iqr_multiplier,
            null_threshold=null_threshold,
        )
        self._missing_analyzer = MissingDataAnalyzer()

    # ── Public API ────────────────────────────────────────

    def ingest_csv(
        self,
        path: str,
        store: ExperimentStore,
        campaign_id: str | None = None,
        role_overrides: dict[str, ColumnRole] | None = None,
    ) -> IngestionReport:
        """Ingest a CSV file into the store."""
        rows = CSVParser.parse_file(path)
        if campaign_id is None:
            campaign_id = _campaign_from_path(path)
        return self._ingest_rows(
            rows, store, campaign_id,
            source_format="csv", source_path=path,
            role_overrides=role_overrides,
        )

    def ingest_csv_string(
        self,
        csv_string: str,
        store: ExperimentStore,
        campaign_id: str = "imported",
        role_overrides: dict[str, ColumnRole] | None = None,
    ) -> IngestionReport:
        """Ingest a CSV string into the store."""
        rows = CSVParser.parse_string(csv_string)
        return self._ingest_rows(
            rows, store, campaign_id,
            source_format="csv", source_path="<string>",
            role_overrides=role_overrides,
        )

    def ingest_json(
        self,
        path: str,
        store: ExperimentStore,
        campaign_id: str | None = None,
        role_overrides: dict[str, ColumnRole] | None = None,
    ) -> IngestionReport:
        """Ingest a JSON file into the store."""
        rows = JSONParser.parse_file(path)
        if campaign_id is None:
            campaign_id = _campaign_from_path(path)
        return self._ingest_rows(
            rows, store, campaign_id,
            source_format="json", source_path=path,
            role_overrides=role_overrides,
        )

    def ingest_json_string(
        self,
        json_string: str,
        store: ExperimentStore,
        campaign_id: str = "imported",
        role_overrides: dict[str, ColumnRole] | None = None,
    ) -> IngestionReport:
        """Ingest a JSON string into the store."""
        rows = JSONParser.parse_string(json_string)
        return self._ingest_rows(
            rows, store, campaign_id,
            source_format="json", source_path="<string>",
            role_overrides=role_overrides,
        )

    # ── Internal ──────────────────────────────────────────

    def _ingest_rows(
        self,
        rows: list[dict[str, Any]],
        store: ExperimentStore,
        campaign_id: str,
        source_format: str,
        source_path: str,
        role_overrides: dict[str, ColumnRole] | None = None,
    ) -> IngestionReport:
        """Core ingestion pipeline: profile → infer → detect → convert → store."""
        warnings: list[str] = []

        if not rows:
            missing_report = self._missing_analyzer.analyze([], 0)
            return IngestionReport(
                source_format=source_format,
                source_path=source_path,
                n_rows=0,
                n_columns=0,
                column_profiles=[],
                anomalies=[],
                missing_data=missing_report,
                warnings=["No data rows found"],
                experiments_created=0,
                campaign_id=campaign_id,
            )

        # 1. Profile columns.
        profiles = self._profiler.profile_columns(rows)

        # 2. Infer roles.
        profiles = self._role_engine.infer_roles(profiles)

        # 3. Apply user overrides.
        if role_overrides:
            for profile in profiles:
                if profile.name in role_overrides:
                    profile.inferred_role = role_overrides[profile.name]
                    profile.role_confidence = 1.0

        # 4. Detect anomalies.
        anomalies = self._anomaly_detector.detect(rows, profiles)
        if anomalies:
            warnings.append(f"Detected {len(anomalies)} anomalies in data")

        # 5. Analyze missing data.
        missing_report = self._missing_analyzer.analyze(profiles, len(rows))
        if missing_report.total_missing > 0:
            warnings.append(
                f"Missing data: {missing_report.total_missing} cells "
                f"({missing_report.missing_rate:.1%})"
            )

        # 6. Validate roles — warn if no KPI or no PARAMETER found.
        roles_found = {p.inferred_role for p in profiles}
        if ColumnRole.KPI not in roles_found:
            warnings.append(
                "No KPI column detected; consider specifying role_overrides"
            )
        if ColumnRole.PARAMETER not in roles_found:
            warnings.append(
                "No PARAMETER column detected; consider specifying role_overrides"
            )

        # 7. Convert rows to experiments and store.
        experiments = self._rows_to_experiments(rows, profiles, campaign_id)
        n_created = 0
        for exp in experiments:
            try:
                store.add_experiment(exp)
                n_created += 1
            except ValueError:
                warnings.append(
                    f"Duplicate experiment ID '{exp.experiment_id}'; skipped"
                )

        return IngestionReport(
            source_format=source_format,
            source_path=source_path,
            n_rows=len(rows),
            n_columns=len(profiles),
            column_profiles=profiles,
            anomalies=anomalies,
            missing_data=missing_report,
            warnings=warnings,
            experiments_created=n_created,
            campaign_id=campaign_id,
        )

    @staticmethod
    def _rows_to_experiments(
        rows: list[dict[str, Any]],
        profiles: list[ColumnProfile],
        campaign_id: str,
    ) -> list[Experiment]:
        """Convert raw rows into Experiment objects using inferred roles."""
        # Build role lookup.
        role_map: dict[str, ColumnRole] = {
            p.name: p.inferred_role for p in profiles
        }

        # Find the iteration column (if any).
        iteration_col: str | None = None
        for p in profiles:
            if p.inferred_role == ColumnRole.ITERATION:
                iteration_col = p.name
                break

        # Find the timestamp column (if any).
        timestamp_col: str | None = None
        for p in profiles:
            if p.inferred_role == ColumnRole.TIMESTAMP:
                timestamp_col = p.name
                break

        experiments: list[Experiment] = []
        now = time.time()

        for i, row in enumerate(rows):
            parameters: dict[str, Any] = {}
            kpi_values: dict[str, float] = {}
            metadata: dict[str, Any] = {}
            iteration = i
            timestamp = now

            for col_name, raw_value in row.items():
                role = role_map.get(col_name, ColumnRole.UNKNOWN)
                coerced = _coerce_value(raw_value)

                if role == ColumnRole.PARAMETER:
                    parameters[col_name] = coerced if coerced is not None else raw_value
                elif role == ColumnRole.KPI:
                    f = _try_float(coerced if coerced is not None else raw_value)
                    if f is not None:
                        kpi_values[col_name] = f
                    else:
                        # KPI must be numeric; fall back to metadata.
                        metadata[col_name] = coerced
                elif role == ColumnRole.ITERATION:
                    val = _try_int(coerced if coerced is not None else raw_value)
                    if val is not None:
                        iteration = val
                elif role == ColumnRole.TIMESTAMP:
                    f = _try_float(coerced if coerced is not None else raw_value)
                    if f is not None:
                        timestamp = f
                    else:
                        metadata[f"{col_name}_raw"] = raw_value
                elif role == ColumnRole.IDENTIFIER:
                    metadata[col_name] = coerced if coerced is not None else raw_value
                else:
                    # METADATA, UNKNOWN, or anything else.
                    if coerced is not None:
                        metadata[col_name] = coerced
                    elif raw_value is not None:
                        metadata[col_name] = raw_value

            experiment_id = f"{campaign_id}-{iteration:04d}"
            experiments.append(
                Experiment(
                    experiment_id=experiment_id,
                    campaign_id=campaign_id,
                    iteration=iteration,
                    parameters=parameters,
                    kpi_values=kpi_values,
                    metadata=metadata,
                    timestamp=timestamp,
                )
            )

        return experiments


def _campaign_from_path(path: str) -> str:
    """Derive a campaign_id from a file path (stem without extension)."""
    # Use basic string ops to avoid pathlib import.
    name = path.rsplit("/", 1)[-1] if "/" in path else path
    name = name.rsplit("\\", 1)[-1] if "\\" in name else name
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name or "imported"

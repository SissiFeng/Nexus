"""Data models for the ingestion module.

Defines column roles, profiles, anomaly flags, and ingestion reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums ──────────────────────────────────────────────


class ColumnRole(str, Enum):
    """Inferred role of a data column."""

    PARAMETER = "parameter"
    KPI = "kpi"
    METADATA = "metadata"
    TIMESTAMP = "timestamp"
    ITERATION = "iteration"
    IDENTIFIER = "identifier"
    UNKNOWN = "unknown"


class DataType(str, Enum):
    """Inferred data type of column values."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    MIXED = "mixed"


class MissingDataStrategy(str, Enum):
    """Strategy for handling missing data."""

    SKIP_ROW = "skip_row"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_ZERO = "fill_zero"
    FLAG_ONLY = "flag_only"


class AnomalyType(str, Enum):
    """Classification of detected anomaly."""

    OUTLIER_IQR = "outlier_iqr"
    CONSTANT_COLUMN = "constant_column"
    HIGH_NULL_RATE = "high_null_rate"
    TYPE_MISMATCH = "type_mismatch"


# ── Dataclasses ────────────────────────────────────────


@dataclass
class ColumnProfile:
    """Statistical profile of a single data column."""

    name: str
    data_type: DataType
    inferred_role: ColumnRole = ColumnRole.UNKNOWN
    role_confidence: float = 0.0
    n_values: int = 0
    n_nulls: int = 0
    n_unique: int = 0
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    std_value: float | None = None
    sample_values: list[Any] = field(default_factory=list)
    unit_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type.value,
            "inferred_role": self.inferred_role.value,
            "role_confidence": self.role_confidence,
            "n_values": self.n_values,
            "n_nulls": self.n_nulls,
            "n_unique": self.n_unique,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "sample_values": list(self.sample_values),
            "unit_hint": self.unit_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnProfile:
        data = data.copy()
        data["data_type"] = DataType(data["data_type"])
        data["inferred_role"] = ColumnRole(data["inferred_role"])
        return cls(**data)


@dataclass
class AnomalyFlag:
    """A detected anomaly in the raw data."""

    column_name: str
    row_index: int
    value: Any
    anomaly_type: AnomalyType
    severity: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "row_index": self.row_index,
            "value": self.value,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnomalyFlag:
        data = data.copy()
        data["anomaly_type"] = AnomalyType(data["anomaly_type"])
        return cls(**data)


@dataclass
class MissingDataReport:
    """Summary of missing data in a dataset."""

    columns_with_missing: dict[str, int]
    total_missing: int
    total_cells: int
    missing_rate: float
    suggested_strategy: dict[str, MissingDataStrategy]

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns_with_missing": dict(self.columns_with_missing),
            "total_missing": self.total_missing,
            "total_cells": self.total_cells,
            "missing_rate": self.missing_rate,
            "suggested_strategy": {
                k: v.value for k, v in self.suggested_strategy.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissingDataReport:
        data = data.copy()
        data["suggested_strategy"] = {
            k: MissingDataStrategy(v)
            for k, v in data.get("suggested_strategy", {}).items()
        }
        return cls(**data)


@dataclass
class IngestionReport:
    """Complete report from a data ingestion operation."""

    source_format: str
    source_path: str
    n_rows: int
    n_columns: int
    column_profiles: list[ColumnProfile]
    anomalies: list[AnomalyFlag]
    missing_data: MissingDataReport
    warnings: list[str]
    experiments_created: int
    campaign_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "source_path": self.source_path,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "column_profiles": [p.to_dict() for p in self.column_profiles],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "missing_data": self.missing_data.to_dict(),
            "warnings": list(self.warnings),
            "experiments_created": self.experiments_created,
            "campaign_id": self.campaign_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestionReport:
        data = data.copy()
        data["column_profiles"] = [
            ColumnProfile.from_dict(p) for p in data["column_profiles"]
        ]
        data["anomalies"] = [AnomalyFlag.from_dict(a) for a in data["anomalies"]]
        data["missing_data"] = MissingDataReport.from_dict(data["missing_data"])
        return cls(**data)

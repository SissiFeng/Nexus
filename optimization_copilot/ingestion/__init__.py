"""Data Ingestion Agent — auto-parse, profile, and store experimental data."""

from optimization_copilot.ingestion.agent import DataIngestionAgent
from optimization_copilot.ingestion.models import (
    AnomalyFlag,
    AnomalyType,
    ColumnProfile,
    ColumnRole,
    DataType,
    IngestionReport,
    MissingDataReport,
    MissingDataStrategy,
)
from optimization_copilot.ingestion.parsers import CSVParser, JSONParser
from optimization_copilot.ingestion.profiler import (
    AnomalyDetector,
    ColumnProfiler,
    MissingDataAnalyzer,
    RoleInferenceEngine,
)

__all__ = [
    "AnomalyDetector",
    "AnomalyFlag",
    "AnomalyType",
    "CSVParser",
    "ColumnProfile",
    "ColumnProfiler",
    "ColumnRole",
    "DataIngestionAgent",
    "DataType",
    "IngestionReport",
    "JSONParser",
    "MissingDataAnalyzer",
    "MissingDataReport",
    "MissingDataStrategy",
    "RoleInferenceEngine",
]

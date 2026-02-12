"""Data models for deterministic imputation with full traceability."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from optimization_copilot.core.models import Observation


class ImputationStrategy(str, Enum):
    """Available imputation strategies."""

    WORST_VALUE = "worst_value"
    COLUMN_MEDIAN = "column_median"
    COLUMN_MEAN = "column_mean"
    KNN_PROXY = "knn_proxy"


@dataclass
class ImputationRecord:
    """Audit record for a single imputed value.

    Captures exactly what was imputed, why, and from what source data,
    enabling full traceability and reproducibility.
    """

    observation_index: int
    kpi_name: str
    original_value: float | None
    imputed_value: float
    strategy: ImputationStrategy
    source_columns: list[str]
    k_neighbors: int | None = None
    neighbor_indices: list[int] | None = None


@dataclass
class ImputationConfig:
    """Configuration for deterministic imputation.

    Parameters
    ----------
    strategy :
        Default imputation strategy for all KPIs.
    seed :
        Random seed for reproducibility (used by KNN tie-breaking).
    knn_k :
        Number of neighbors for KNN_PROXY strategy.
    per_kpi_strategy :
        Optional per-KPI strategy overrides. Keys are KPI names.
    """

    strategy: ImputationStrategy = ImputationStrategy.WORST_VALUE
    seed: int = 42
    knn_k: int = 3
    per_kpi_strategy: dict[str, ImputationStrategy] | None = None


@dataclass
class ImputationResult:
    """Complete result of an imputation run.

    Parameters
    ----------
    observations :
        The full list of observations after imputation.
    records :
        Audit trail of every imputed value.
    decision_hash :
        Deterministic hash guaranteeing reproducibility.
        Same (observations, config) always produces the same hash.
    config_used :
        The configuration that produced this result.
    """

    observations: list[Observation]
    records: list[ImputationRecord]
    decision_hash: str
    config_used: ImputationConfig

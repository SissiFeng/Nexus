"""Data models for confounder governance: policies, specs, configs, and audit trails."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# -- Enums ------------------------------------------------------------------


class ConfounderPolicy(str, Enum):
    """Available confounder correction policies.

    COVARIATE      -- Promote the confounder to a formal parameter so the
                      surrogate model can account for it.
    NORMALIZE      -- Remove the confounder's linear effect from each KPI via
                      ordinary-least-squares residuals.
    HIGH_RISK_FLAG -- Down-weight observations whose confounder value falls
                      outside acceptable thresholds.
    EXCLUDE        -- Remove observations whose confounder value falls outside
                      acceptable thresholds entirely.
    """
    COVARIATE = "covariate"
    NORMALIZE = "normalize"
    HIGH_RISK_FLAG = "high_risk_flag"
    EXCLUDE = "exclude"


# -- Dataclasses ------------------------------------------------------------


@dataclass
class ConfounderSpec:
    """Specification for a single confounder column and its correction policy.

    Parameters
    ----------
    column_name : str
        The metadata key that holds the confounder values.
    policy : ConfounderPolicy
        Which correction strategy to apply.
    threshold_low : float | None
        Lower acceptable bound (used by HIGH_RISK_FLAG and EXCLUDE).
    threshold_high : float | None
        Upper acceptable bound (used by HIGH_RISK_FLAG and EXCLUDE).
    metadata : dict
        Arbitrary extra information (e.g. unit, source).
    """
    column_name: str
    policy: ConfounderPolicy
    threshold_low: float | None = None
    threshold_high: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfounderConfig:
    """Configuration for the confounder governance pipeline.

    Parameters
    ----------
    confounders : list[ConfounderSpec]
        Ordered list of confounder specifications to apply.
    auto_detect : bool
        Whether to run automatic confounder detection before applying
        user-specified specs.
    correlation_threshold : float
        Pearson |r| threshold used by auto-detection (default 0.3).
    """
    confounders: list[ConfounderSpec] = field(default_factory=list)
    auto_detect: bool = False
    correlation_threshold: float = 0.3


@dataclass
class ConfounderCorrectionRecord:
    """Record of a single correction step applied by the governor.

    Parameters
    ----------
    column_name : str
        Confounder column that was corrected.
    policy : ConfounderPolicy
        Policy that was applied.
    n_affected_rows : int
        Number of observations modified (or removed).
    correction_details : dict
        Policy-specific details (e.g. regression coefficients, thresholds used).
    original_kpi_stats : dict[str, float]
        Mean and std of each KPI *before* this correction.
    corrected_kpi_stats : dict[str, float]
        Mean and std of each KPI *after* this correction.
    """
    column_name: str
    policy: ConfounderPolicy
    n_affected_rows: int
    correction_details: dict[str, Any] = field(default_factory=dict)
    original_kpi_stats: dict[str, float] = field(default_factory=dict)
    corrected_kpi_stats: dict[str, float] = field(default_factory=dict)


@dataclass
class ConfounderAuditTrail:
    """Complete audit trail for a confounder governance run.

    Parameters
    ----------
    corrections : list[ConfounderCorrectionRecord]
        Ordered list of correction records (one per applied spec).
    config_used : ConfounderConfig
        The configuration that produced these corrections.
    summary : str
        Human-readable summary of all corrections applied.
    """
    corrections: list[ConfounderCorrectionRecord] = field(default_factory=list)
    config_used: ConfounderConfig = field(default_factory=ConfounderConfig)
    summary: str = ""

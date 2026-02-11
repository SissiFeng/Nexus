"""Column profiling, role inference, and anomaly detection.

All algorithms are pure Python (math module only).
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.ingestion.models import (
    AnomalyFlag,
    AnomalyType,
    ColumnProfile,
    ColumnRole,
    DataType,
    MissingDataReport,
    MissingDataStrategy,
)


# ── Helpers ────────────────────────────────────────────────


def _is_null(value: Any) -> bool:
    """Check if a value is null / missing / empty string."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _try_int(value: Any) -> int | None:
    """Try to convert a value to int."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value == int(value) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        try:
            f = float(value)
            if f == int(f) and math.isfinite(f):
                return int(f)
        except (ValueError, OverflowError):
            pass
    return None


def _try_float(value: Any) -> float | None:
    """Try to convert a value to float."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        f = float(value)
        if math.isfinite(f):
            return f
        return None
    if isinstance(value, str):
        try:
            f = float(value)
            if math.isfinite(f):
                return f
        except (ValueError, OverflowError):
            pass
    return None


def _coerce_value(value: Any) -> Any:
    """Coerce a string value to int, float, or leave as str."""
    if _is_null(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        # Try int first.
        i = _try_int(value)
        if i is not None:
            return i
        # Try float.
        f = _try_float(value)
        if f is not None:
            return f
        # Boolean strings.
        low = value.strip().lower()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
    return value


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def _std(values: list[float]) -> float:
    return math.sqrt(_variance(values))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    k = (p / 100.0) * (n - 1)
    lo = int(math.floor(k))
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


# ── Column Profiler ────────────────────────────────────────


class ColumnProfiler:
    """Compute statistical profiles for each column in tabular data."""

    def profile_columns(
        self, rows: list[dict[str, Any]]
    ) -> list[ColumnProfile]:
        """Profile all columns in the dataset."""
        if not rows:
            return []

        # Collect all column names (preserving order from first row).
        all_cols: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for k in row:
                if k not in seen:
                    all_cols.append(k)
                    seen.add(k)

        profiles: list[ColumnProfile] = []
        for col in all_cols:
            raw_values = [row.get(col) for row in rows]
            profiles.append(self._profile_single(col, raw_values))

        return profiles

    def _profile_single(
        self, name: str, raw_values: list[Any]
    ) -> ColumnProfile:
        """Profile a single column."""
        n_values = len(raw_values)
        n_nulls = sum(1 for v in raw_values if _is_null(v))

        # Coerce values.
        coerced = [_coerce_value(v) for v in raw_values]
        non_null = [v for v in coerced if v is not None]

        n_unique = len(set(str(v) for v in non_null))
        data_type = self._infer_data_type(non_null)

        # Numeric stats.
        numeric_vals: list[float] = []
        for v in non_null:
            f = _try_float(v)
            if f is not None:
                numeric_vals.append(f)

        min_val: float | None = None
        max_val: float | None = None
        mean_val: float | None = None
        std_val: float | None = None
        if numeric_vals:
            min_val = min(numeric_vals)
            max_val = max(numeric_vals)
            mean_val = _mean(numeric_vals)
            std_val = _std(numeric_vals)

        sample = non_null[:5]

        return ColumnProfile(
            name=name,
            data_type=data_type,
            n_values=n_values,
            n_nulls=n_nulls,
            n_unique=n_unique,
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            std_value=std_val,
            sample_values=sample,
        )

    @staticmethod
    def _infer_data_type(values: list[Any]) -> DataType:
        """Infer the data type from a list of non-null values."""
        if not values:
            return DataType.STRING

        types_seen: set[str] = set()
        for v in values:
            if isinstance(v, bool):
                types_seen.add("bool")
            elif isinstance(v, int):
                types_seen.add("int")
            elif isinstance(v, float):
                types_seen.add("float")
            else:
                types_seen.add("str")

        if types_seen == {"bool"}:
            return DataType.BOOLEAN
        if types_seen <= {"int"}:
            return DataType.INTEGER
        if types_seen <= {"int", "float"}:
            return DataType.FLOAT
        if types_seen == {"str"}:
            return DataType.STRING
        return DataType.MIXED


# ── Role Inference Engine ──────────────────────────────────


class RoleInferenceEngine:
    """Heuristic column-role inference engine."""

    _ITERATION_PATTERNS = [
        "iteration", "iter", "run", "step", "cycle", "trial", "experiment_num",
    ]
    _TIMESTAMP_PATTERNS = [
        "time", "timestamp", "date", "datetime", "created_at", "recorded_at",
    ]
    _ID_PATTERNS = [
        "id", "sample_id", "experiment_id", "run_id", "trial_id", "uid",
    ]
    _KPI_PATTERNS = [
        "yield", "purity", "error", "loss", "score", "efficiency",
        "conversion", "selectivity", "performance", "cost", "output",
        "result", "response", "target", "objective", "kpi", "metric",
    ]

    def infer_roles(
        self, profiles: list[ColumnProfile]
    ) -> list[ColumnProfile]:
        """Update profiles with inferred roles and confidence scores."""
        for profile in profiles:
            scores = self._score_role(profile)
            # Pick the role with the highest score.
            best_role = ColumnRole.UNKNOWN
            best_score = 0.0
            for role, score in scores.items():
                if score > best_score:
                    best_role = role
                    best_score = score
            profile.inferred_role = best_role
            profile.role_confidence = best_score

        return profiles

    def _score_role(
        self, profile: ColumnProfile
    ) -> dict[ColumnRole, float]:
        """Score each possible role for a column."""
        scores: dict[ColumnRole, float] = {role: 0.0 for role in ColumnRole}
        name_lower = profile.name.lower().strip()

        # Name-based heuristics.
        for pattern in self._ITERATION_PATTERNS:
            if pattern in name_lower:
                scores[ColumnRole.ITERATION] += 0.7
                break

        for pattern in self._TIMESTAMP_PATTERNS:
            if pattern in name_lower:
                scores[ColumnRole.TIMESTAMP] += 0.7
                break

        for pattern in self._ID_PATTERNS:
            if name_lower == pattern or name_lower.endswith("_id"):
                scores[ColumnRole.IDENTIFIER] += 0.7
                break

        for pattern in self._KPI_PATTERNS:
            if pattern in name_lower:
                scores[ColumnRole.KPI] += 0.6
                break

        # Statistical heuristics.
        if profile.data_type in (DataType.INTEGER, DataType.FLOAT):
            # Numeric columns are likely parameters or KPIs.
            if scores[ColumnRole.KPI] < 0.3:
                scores[ColumnRole.PARAMETER] += 0.4
            # If it looks like a monotonic index, boost iteration.
            if (
                profile.data_type == DataType.INTEGER
                and profile.min_value is not None
                and profile.max_value is not None
                and profile.n_unique == profile.n_values - profile.n_nulls
                and profile.min_value >= 0
            ):
                scores[ColumnRole.ITERATION] += 0.3

        elif profile.data_type == DataType.STRING:
            # String columns with many unique values → metadata.
            if profile.n_unique > 0.8 * (profile.n_values - profile.n_nulls):
                scores[ColumnRole.METADATA] += 0.5
            else:
                # Few unique strings could be categorical parameter.
                scores[ColumnRole.PARAMETER] += 0.3
                scores[ColumnRole.METADATA] += 0.2

        elif profile.data_type == DataType.BOOLEAN:
            scores[ColumnRole.METADATA] += 0.4

        # Unit hints.
        unit_suffixes = {
            "_mm": "mm", "_cm": "cm", "_m": "m", "_kg": "kg",
            "_g": "g", "_ml": "mL", "_l": "L", "_c": "°C",
            "_pct": "%", "_percent": "%",
        }
        for suffix, unit in unit_suffixes.items():
            if name_lower.endswith(suffix):
                profile.unit_hint = unit
                scores[ColumnRole.PARAMETER] += 0.1
                break

        return scores


# ── Anomaly Detector ───────────────────────────────────────


class AnomalyDetector:
    """Detect anomalies in raw tabular data."""

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        null_threshold: float = 0.3,
    ) -> None:
        self.iqr_multiplier = iqr_multiplier
        self.null_threshold = null_threshold

    def detect(
        self,
        rows: list[dict[str, Any]],
        profiles: list[ColumnProfile],
    ) -> list[AnomalyFlag]:
        """Detect anomalies across all columns."""
        anomalies: list[AnomalyFlag] = []

        for profile in profiles:
            # High null rate.
            if profile.n_values > 0:
                null_rate = profile.n_nulls / profile.n_values
                if null_rate > self.null_threshold:
                    anomalies.append(
                        AnomalyFlag(
                            column_name=profile.name,
                            row_index=-1,
                            value=None,
                            anomaly_type=AnomalyType.HIGH_NULL_RATE,
                            severity=min(1.0, null_rate),
                            message=(
                                f"Column '{profile.name}' has {null_rate:.0%} "
                                f"missing values ({profile.n_nulls}/{profile.n_values})"
                            ),
                        )
                    )

            # Constant column.
            if profile.n_unique <= 1 and (profile.n_values - profile.n_nulls) > 1:
                anomalies.append(
                    AnomalyFlag(
                        column_name=profile.name,
                        row_index=-1,
                        value=profile.sample_values[0] if profile.sample_values else None,
                        anomaly_type=AnomalyType.CONSTANT_COLUMN,
                        severity=0.5,
                        message=f"Column '{profile.name}' has only 1 unique value",
                    )
                )

            # IQR outliers for numeric columns.
            if profile.data_type in (DataType.INTEGER, DataType.FLOAT):
                numeric_vals: list[float] = []
                for row in rows:
                    v = row.get(profile.name)
                    f = _try_float(_coerce_value(v))
                    if f is not None:
                        numeric_vals.append(f)

                if len(numeric_vals) >= 4:
                    q1 = _percentile(numeric_vals, 25.0)
                    q3 = _percentile(numeric_vals, 75.0)
                    iqr = q3 - q1
                    lower_fence = q1 - self.iqr_multiplier * iqr
                    upper_fence = q3 + self.iqr_multiplier * iqr

                    for i, row in enumerate(rows):
                        v = row.get(profile.name)
                        f = _try_float(_coerce_value(v))
                        if f is not None and (f < lower_fence or f > upper_fence):
                            distance = max(
                                abs(f - lower_fence) if f < lower_fence else 0,
                                abs(f - upper_fence) if f > upper_fence else 0,
                            )
                            severity = min(1.0, distance / (iqr + 1e-10))
                            anomalies.append(
                                AnomalyFlag(
                                    column_name=profile.name,
                                    row_index=i,
                                    value=f,
                                    anomaly_type=AnomalyType.OUTLIER_IQR,
                                    severity=severity,
                                    message=(
                                        f"Value {f} in column '{profile.name}' "
                                        f"is outside IQR fences "
                                        f"[{lower_fence:.4g}, {upper_fence:.4g}]"
                                    ),
                                )
                            )

        return anomalies


# ── Missing Data Analyzer ──────────────────────────────────


class MissingDataAnalyzer:
    """Analyze missing data patterns and suggest strategies."""

    def analyze(
        self, profiles: list[ColumnProfile], n_rows: int
    ) -> MissingDataReport:
        """Produce a missing data report."""
        cols_missing: dict[str, int] = {}
        total_missing = 0
        strategies: dict[str, MissingDataStrategy] = {}

        for p in profiles:
            if p.n_nulls > 0:
                cols_missing[p.name] = p.n_nulls
                total_missing += p.n_nulls

                # Suggest strategy based on data type and rate.
                null_rate = p.n_nulls / p.n_values if p.n_values > 0 else 0
                if null_rate > 0.5:
                    strategies[p.name] = MissingDataStrategy.SKIP_ROW
                elif p.data_type in (DataType.INTEGER, DataType.FLOAT):
                    strategies[p.name] = MissingDataStrategy.FILL_MEAN
                else:
                    strategies[p.name] = MissingDataStrategy.FLAG_ONLY

        n_columns = len(profiles)
        total_cells = n_rows * n_columns if n_columns > 0 else 0
        missing_rate = total_missing / total_cells if total_cells > 0 else 0.0

        return MissingDataReport(
            columns_with_missing=cols_missing,
            total_missing=total_missing,
            total_cells=total_cells,
            missing_rate=missing_rate,
            suggested_strategy=strategies,
        )

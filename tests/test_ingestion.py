"""Comprehensive tests for the optimization_copilot.ingestion module.

Covers models, parsers, profiler helpers, column profiling, role inference,
anomaly detection, missing data analysis, and the DataIngestionAgent.
"""

from __future__ import annotations

import json
import math

import pytest

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
    _coerce_value,
    _is_null,
    _mean,
    _median,
    _percentile,
    _std,
    _try_float,
    _try_int,
    _variance,
)
from optimization_copilot.ingestion.agent import DataIngestionAgent, _campaign_from_path
from optimization_copilot.store.store import ExperimentStore


# ===================================================================
# TestModels (~10 tests)
# ===================================================================


class TestModels:
    """Tests for ingestion dataclasses and enums."""

    def test_column_role_values(self):
        assert ColumnRole.PARAMETER.value == "parameter"
        assert ColumnRole.KPI.value == "kpi"
        assert ColumnRole.METADATA.value == "metadata"
        assert ColumnRole.TIMESTAMP.value == "timestamp"
        assert ColumnRole.ITERATION.value == "iteration"
        assert ColumnRole.IDENTIFIER.value == "identifier"
        assert ColumnRole.UNKNOWN.value == "unknown"

    def test_data_type_values(self):
        assert DataType.INTEGER.value == "integer"
        assert DataType.FLOAT.value == "float"
        assert DataType.STRING.value == "string"
        assert DataType.BOOLEAN.value == "boolean"
        assert DataType.MIXED.value == "mixed"

    def test_missing_data_strategy_values(self):
        assert MissingDataStrategy.SKIP_ROW.value == "skip_row"
        assert MissingDataStrategy.FILL_MEAN.value == "fill_mean"
        assert MissingDataStrategy.FILL_MEDIAN.value == "fill_median"
        assert MissingDataStrategy.FILL_ZERO.value == "fill_zero"
        assert MissingDataStrategy.FLAG_ONLY.value == "flag_only"

    def test_anomaly_type_values(self):
        assert AnomalyType.OUTLIER_IQR.value == "outlier_iqr"
        assert AnomalyType.CONSTANT_COLUMN.value == "constant_column"
        assert AnomalyType.HIGH_NULL_RATE.value == "high_null_rate"
        assert AnomalyType.TYPE_MISMATCH.value == "type_mismatch"

    def test_column_profile_to_dict_from_dict_round_trip(self):
        profile = ColumnProfile(
            name="temperature",
            data_type=DataType.FLOAT,
            inferred_role=ColumnRole.PARAMETER,
            role_confidence=0.85,
            n_values=100,
            n_nulls=3,
            n_unique=97,
            min_value=20.0,
            max_value=80.0,
            mean_value=50.0,
            std_value=15.0,
            sample_values=[20.0, 30.0, 40.0],
            unit_hint="C",
        )
        d = profile.to_dict()
        restored = ColumnProfile.from_dict(d)
        assert restored.name == profile.name
        assert restored.data_type == profile.data_type
        assert restored.inferred_role == profile.inferred_role
        assert restored.role_confidence == profile.role_confidence
        assert restored.n_values == profile.n_values
        assert restored.n_nulls == profile.n_nulls
        assert restored.n_unique == profile.n_unique
        assert restored.min_value == profile.min_value
        assert restored.max_value == profile.max_value
        assert restored.mean_value == profile.mean_value
        assert restored.std_value == profile.std_value
        assert restored.sample_values == profile.sample_values
        assert restored.unit_hint == profile.unit_hint

    def test_anomaly_flag_to_dict_from_dict_round_trip(self):
        flag = AnomalyFlag(
            column_name="pressure",
            row_index=42,
            value=999.9,
            anomaly_type=AnomalyType.OUTLIER_IQR,
            severity=0.9,
            message="Outlier detected",
        )
        d = flag.to_dict()
        restored = AnomalyFlag.from_dict(d)
        assert restored.column_name == flag.column_name
        assert restored.row_index == flag.row_index
        assert restored.value == flag.value
        assert restored.anomaly_type == flag.anomaly_type
        assert restored.severity == flag.severity
        assert restored.message == flag.message

    def test_missing_data_report_to_dict_from_dict_round_trip(self):
        report = MissingDataReport(
            columns_with_missing={"col_a": 5, "col_b": 10},
            total_missing=15,
            total_cells=200,
            missing_rate=0.075,
            suggested_strategy={
                "col_a": MissingDataStrategy.FILL_MEAN,
                "col_b": MissingDataStrategy.SKIP_ROW,
            },
        )
        d = report.to_dict()
        restored = MissingDataReport.from_dict(d)
        assert restored.columns_with_missing == report.columns_with_missing
        assert restored.total_missing == report.total_missing
        assert restored.total_cells == report.total_cells
        assert restored.missing_rate == report.missing_rate
        assert restored.suggested_strategy == report.suggested_strategy

    def test_ingestion_report_to_dict_from_dict_round_trip(self):
        profile = ColumnProfile(
            name="x",
            data_type=DataType.INTEGER,
            inferred_role=ColumnRole.PARAMETER,
            role_confidence=0.7,
            n_values=10,
            n_nulls=0,
            n_unique=10,
        )
        anomaly = AnomalyFlag(
            column_name="x",
            row_index=0,
            value=100,
            anomaly_type=AnomalyType.OUTLIER_IQR,
            severity=0.8,
            message="test",
        )
        missing = MissingDataReport(
            columns_with_missing={},
            total_missing=0,
            total_cells=10,
            missing_rate=0.0,
            suggested_strategy={},
        )
        report = IngestionReport(
            source_format="csv",
            source_path="/tmp/data.csv",
            n_rows=10,
            n_columns=1,
            column_profiles=[profile],
            anomalies=[anomaly],
            missing_data=missing,
            warnings=["test warning"],
            experiments_created=10,
            campaign_id="test_campaign",
        )
        d = report.to_dict()
        restored = IngestionReport.from_dict(d)
        assert restored.source_format == report.source_format
        assert restored.source_path == report.source_path
        assert restored.n_rows == report.n_rows
        assert restored.n_columns == report.n_columns
        assert len(restored.column_profiles) == 1
        assert restored.column_profiles[0].name == "x"
        assert len(restored.anomalies) == 1
        assert restored.anomalies[0].column_name == "x"
        assert restored.missing_data.total_missing == 0
        assert restored.warnings == ["test warning"]
        assert restored.experiments_created == 10
        assert restored.campaign_id == "test_campaign"

    def test_column_profile_defaults(self):
        profile = ColumnProfile(name="col", data_type=DataType.STRING)
        assert profile.inferred_role == ColumnRole.UNKNOWN
        assert profile.role_confidence == 0.0
        assert profile.n_values == 0
        assert profile.n_nulls == 0
        assert profile.n_unique == 0
        assert profile.min_value is None
        assert profile.max_value is None
        assert profile.mean_value is None
        assert profile.std_value is None
        assert profile.sample_values == []
        assert profile.unit_hint is None

    def test_enum_str_subclass(self):
        """All enums are str subclasses for easy JSON serialization."""
        assert isinstance(ColumnRole.KPI, str)
        assert isinstance(DataType.FLOAT, str)
        assert isinstance(MissingDataStrategy.FILL_MEAN, str)
        assert isinstance(AnomalyType.OUTLIER_IQR, str)


# ===================================================================
# TestCSVParser (~5 tests)
# ===================================================================


class TestCSVParser:
    """Tests for CSVParser."""

    def test_parse_string_simple(self):
        csv_data = "a,b,c\n1,2,3\n4,5,6\n"
        rows = CSVParser.parse_string(csv_data)
        assert len(rows) == 2
        assert rows[0] == {"a": "1", "b": "2", "c": "3"}
        assert rows[1] == {"a": "4", "b": "5", "c": "6"}

    def test_parse_string_header_only(self):
        csv_data = "a,b,c\n"
        rows = CSVParser.parse_string(csv_data)
        assert len(rows) == 0

    def test_parse_string_missing_values(self):
        csv_data = "a,b,c\n1,,3\n,5,\n"
        rows = CSVParser.parse_string(csv_data)
        assert len(rows) == 2
        assert rows[0]["b"] == ""
        assert rows[1]["a"] == ""
        assert rows[1]["c"] == ""

    def test_parse_file_with_tmp_path(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y\n10,20\n30,40\n", encoding="utf-8")
        rows = CSVParser.parse_file(str(csv_file))
        assert len(rows) == 2
        assert rows[0] == {"x": "10", "y": "20"}
        assert rows[1] == {"x": "30", "y": "40"}

    def test_parse_string_with_quotes(self):
        csv_data = 'name,desc\n"Alice","has, comma"\n"Bob","no comma"\n'
        rows = CSVParser.parse_string(csv_data)
        assert len(rows) == 2
        assert rows[0]["desc"] == "has, comma"


# ===================================================================
# TestJSONParser (~5 tests)
# ===================================================================


class TestJSONParser:
    """Tests for JSONParser."""

    def test_parse_string_list_of_dicts(self):
        data = json.dumps([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        rows = JSONParser.parse_string(data)
        assert len(rows) == 2
        assert rows[0] == {"a": 1, "b": 2}
        assert rows[1] == {"a": 3, "b": 4}

    def test_parse_string_dict_of_lists(self):
        data = json.dumps({"a": [1, 2, 3], "b": [4, 5, 6]})
        rows = JSONParser.parse_string(data)
        assert len(rows) == 3
        assert rows[0] == {"a": 1, "b": 4}
        assert rows[1] == {"a": 2, "b": 5}
        assert rows[2] == {"a": 3, "b": 6}

    def test_parse_string_single_value_dict(self):
        """A dict where values are not lists gets wrapped into single-row output."""
        data = json.dumps({"a": 10, "b": 20})
        rows = JSONParser.parse_string(data)
        assert len(rows) == 1
        assert rows[0] == {"a": 10, "b": 20}

    def test_parse_string_invalid_type_raises_value_error(self):
        with pytest.raises(ValueError, match="must be a list of dicts or a dict of lists"):
            JSONParser.parse_string('"just a string"')

    def test_parse_file_with_tmp_path(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text(
            json.dumps([{"x": 1, "y": 2}, {"x": 3, "y": 4}]),
            encoding="utf-8",
        )
        rows = JSONParser.parse_file(str(json_file))
        assert len(rows) == 2
        assert rows[0] == {"x": 1, "y": 2}


# ===================================================================
# TestHelpers (~8 tests)
# ===================================================================


class TestHelpers:
    """Tests for profiler helper functions."""

    def test_is_null_none(self):
        assert _is_null(None) is True

    def test_is_null_empty_string(self):
        assert _is_null("") is True

    def test_is_null_whitespace(self):
        assert _is_null("   ") is True

    def test_is_null_non_null_values(self):
        assert _is_null(0) is False
        assert _is_null("hello") is False
        assert _is_null(0.0) is False
        assert _is_null(False) is False

    def test_try_int(self):
        assert _try_int(42) == 42
        assert _try_int(3.0) == 3
        assert _try_int("7") == 7
        assert _try_int("3.0") == 3
        assert _try_int("hello") is None
        assert _try_int(3.5) is None
        assert _try_int(True) is None

    def test_try_float(self):
        assert _try_float(42) == 42.0
        assert _try_float(3.14) == 3.14
        assert _try_float("2.718") == pytest.approx(2.718)
        assert _try_float("hello") is None
        assert _try_float(True) is None

    def test_coerce_value(self):
        assert _coerce_value(None) is None
        assert _coerce_value("") is None
        assert _coerce_value("  ") is None
        assert _coerce_value("42") == 42
        assert _coerce_value("3.14") == pytest.approx(3.14)
        assert _coerce_value("true") is True
        assert _coerce_value("False") is False
        assert _coerce_value("yes") is True
        assert _coerce_value("no") is False
        assert _coerce_value("hello") == "hello"
        assert _coerce_value(True) is True
        assert _coerce_value(99) == 99

    def test_mean_variance_std(self):
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert _mean(values) == 5.0
        assert _variance(values) == pytest.approx(4.0)
        assert _std(values) == pytest.approx(2.0)
        assert _mean([]) == 0.0
        assert _variance([]) == 0.0
        assert _variance([5.0]) == 0.0

    def test_median(self):
        assert _median([1.0, 3.0, 5.0]) == 3.0
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5
        assert _median([]) == 0.0

    def test_percentile(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(values, 0) == pytest.approx(1.0)
        assert _percentile(values, 50) == pytest.approx(3.0)
        assert _percentile(values, 100) == pytest.approx(5.0)
        assert _percentile([], 50) == 0.0
        assert _percentile([7.0], 50) == 7.0


# ===================================================================
# TestColumnProfiler (~8 tests)
# ===================================================================


class TestColumnProfiler:
    """Tests for ColumnProfiler."""

    def setup_method(self):
        self.profiler = ColumnProfiler()

    def test_profile_empty_returns_empty(self):
        assert self.profiler.profile_columns([]) == []

    def test_profile_numeric_data(self):
        rows = [{"val": "10"}, {"val": "20"}, {"val": "30"}, {"val": "40"}]
        profiles = self.profiler.profile_columns(rows)
        assert len(profiles) == 1
        p = profiles[0]
        assert p.name == "val"
        assert p.data_type == DataType.INTEGER
        assert p.min_value == 10.0
        assert p.max_value == 40.0
        assert p.mean_value == pytest.approx(25.0)
        assert p.n_values == 4
        assert p.n_nulls == 0

    def test_profile_mixed_types(self):
        rows = [{"col": "1"}, {"col": "hello"}, {"col": "3.5"}, {"col": "true"}]
        profiles = self.profiler.profile_columns(rows)
        p = profiles[0]
        assert p.data_type == DataType.MIXED

    def test_infer_data_type_integer(self):
        assert ColumnProfiler._infer_data_type([1, 2, 3]) == DataType.INTEGER

    def test_infer_data_type_float(self):
        assert ColumnProfiler._infer_data_type([1, 2.5, 3]) == DataType.FLOAT

    def test_infer_data_type_string(self):
        assert ColumnProfiler._infer_data_type(["a", "b"]) == DataType.STRING

    def test_infer_data_type_boolean(self):
        assert ColumnProfiler._infer_data_type([True, False, True]) == DataType.BOOLEAN

    def test_profile_counts_nulls_and_uniques(self):
        rows = [
            {"col": "a"},
            {"col": "b"},
            {"col": ""},
            {"col": "a"},
            {"col": None},
        ]
        profiles = self.profiler.profile_columns(rows)
        p = profiles[0]
        assert p.n_values == 5
        assert p.n_nulls == 2
        assert p.n_unique == 2


# ===================================================================
# TestRoleInferenceEngine (~10 tests)
# ===================================================================


class TestRoleInferenceEngine:
    """Tests for RoleInferenceEngine."""

    def setup_method(self):
        self.engine = RoleInferenceEngine()

    def _make_profile(
        self,
        name,
        data_type=DataType.FLOAT,
        n_values=10,
        n_nulls=0,
        n_unique=10,
        min_value=0.0,
        max_value=9.0,
    ):
        return ColumnProfile(
            name=name,
            data_type=data_type,
            n_values=n_values,
            n_nulls=n_nulls,
            n_unique=n_unique,
            min_value=min_value,
            max_value=max_value,
        )

    def test_iteration_column_by_name(self):
        p = self._make_profile("iteration", DataType.INTEGER)
        [result] = self.engine.infer_roles([p])
        assert result.inferred_role == ColumnRole.ITERATION

    def test_step_column_detected_as_iteration(self):
        p = self._make_profile("step_number", DataType.INTEGER)
        [result] = self.engine.infer_roles([p])
        assert result.inferred_role == ColumnRole.ITERATION

    def test_kpi_column_by_name_yield(self):
        p = self._make_profile("yield_pct", DataType.FLOAT)
        [result] = self.engine.infer_roles([p])
        assert result.inferred_role == ColumnRole.KPI

    def test_kpi_column_by_name_purity(self):
        p = self._make_profile("purity", DataType.FLOAT)
        [result] = self.engine.infer_roles([p])
        assert result.inferred_role == ColumnRole.KPI

    def test_parameter_column_numeric_no_name_match(self):
        p = self._make_profile("temperature", DataType.FLOAT)
        [result] = self.engine.infer_roles([p])
        assert result.inferred_role == ColumnRole.PARAMETER

    def test_timestamp_column_by_name(self):
        p = self._make_profile("timestamp", DataType.FLOAT)
        [result] = self.engine.infer_roles([p])
        assert result.inferred_role == ColumnRole.TIMESTAMP

    def test_identifier_column(self):
        p = self._make_profile("sample_id", DataType.STRING, n_unique=10)
        [result] = self.engine.infer_roles([p])
        assert result.inferred_role == ColumnRole.IDENTIFIER

    def test_metadata_column_string_high_cardinality(self):
        p = self._make_profile(
            "notes",
            DataType.STRING,
            n_values=10,
            n_nulls=0,
            n_unique=9,
        )
        [result] = self.engine.infer_roles([p])
        assert result.inferred_role == ColumnRole.METADATA

    def test_unit_hint_mm(self):
        p = self._make_profile("length_mm", DataType.FLOAT)
        self.engine.infer_roles([p])
        assert p.unit_hint == "mm"

    def test_unit_hint_pct(self):
        p = self._make_profile("concentration_pct", DataType.FLOAT)
        self.engine.infer_roles([p])
        assert p.unit_hint == "%"


# ===================================================================
# TestAnomalyDetector (~6 tests)
# ===================================================================


class TestAnomalyDetector:
    """Tests for AnomalyDetector."""

    def setup_method(self):
        self.profiler = ColumnProfiler()
        self.detector = AnomalyDetector(iqr_multiplier=1.5, null_threshold=0.3)

    def test_no_anomalies_on_clean_data(self):
        rows = [{"x": str(i)} for i in range(10)]
        profiles = self.profiler.profile_columns(rows)
        anomalies = self.detector.detect(rows, profiles)
        assert all(a.anomaly_type != AnomalyType.OUTLIER_IQR for a in anomalies)

    def test_iqr_outlier_detection(self):
        rows = [{"x": str(i)} for i in range(1, 11)] + [{"x": "1000"}]
        profiles = self.profiler.profile_columns(rows)
        anomalies = self.detector.detect(rows, profiles)
        iqr_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.OUTLIER_IQR]
        assert len(iqr_anomalies) >= 1
        assert any(a.value == 1000.0 for a in iqr_anomalies)

    def test_constant_column_detection(self):
        rows = [{"x": "5"} for _ in range(10)]
        profiles = self.profiler.profile_columns(rows)
        anomalies = self.detector.detect(rows, profiles)
        constant_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.CONSTANT_COLUMN]
        assert len(constant_anomalies) == 1
        assert constant_anomalies[0].column_name == "x"

    def test_high_null_rate_detection(self):
        rows = [{"x": ""}] * 8 + [{"x": "1"}, {"x": "2"}]
        profiles = self.profiler.profile_columns(rows)
        anomalies = self.detector.detect(rows, profiles)
        null_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.HIGH_NULL_RATE]
        assert len(null_anomalies) == 1

    def test_min_data_requirement_no_iqr_check(self):
        """With < 4 numeric values, IQR outlier check should not fire."""
        rows = [{"x": "1"}, {"x": "2"}, {"x": "1000"}]
        profiles = self.profiler.profile_columns(rows)
        anomalies = self.detector.detect(rows, profiles)
        iqr_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.OUTLIER_IQR]
        assert len(iqr_anomalies) == 0

    def test_no_anomalies_below_null_threshold(self):
        """2 out of 10 null (20%) should not trigger high_null_rate at 30% threshold."""
        rows = [{"x": ""}] * 2 + [{"x": str(i)} for i in range(8)]
        profiles = self.profiler.profile_columns(rows)
        anomalies = self.detector.detect(rows, profiles)
        null_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.HIGH_NULL_RATE]
        assert len(null_anomalies) == 0


# ===================================================================
# TestMissingDataAnalyzer (~5 tests)
# ===================================================================


class TestMissingDataAnalyzer:
    """Tests for MissingDataAnalyzer."""

    def setup_method(self):
        self.analyzer = MissingDataAnalyzer()

    def test_no_missing_data(self):
        profiles = [
            ColumnProfile(name="a", data_type=DataType.FLOAT, n_values=10, n_nulls=0),
            ColumnProfile(name="b", data_type=DataType.STRING, n_values=10, n_nulls=0),
        ]
        report = self.analyzer.analyze(profiles, n_rows=10)
        assert report.total_missing == 0
        assert report.missing_rate == 0.0
        assert report.columns_with_missing == {}
        assert report.suggested_strategy == {}

    def test_some_missing_data(self):
        profiles = [
            ColumnProfile(name="a", data_type=DataType.FLOAT, n_values=10, n_nulls=2),
            ColumnProfile(name="b", data_type=DataType.STRING, n_values=10, n_nulls=1),
        ]
        report = self.analyzer.analyze(profiles, n_rows=10)
        assert report.total_missing == 3
        assert report.total_cells == 20
        assert report.missing_rate == pytest.approx(3 / 20)
        assert "a" in report.columns_with_missing
        assert "b" in report.columns_with_missing

    def test_high_missing_rate_skip_row_strategy(self):
        """Column with >50% missing should suggest SKIP_ROW."""
        profiles = [
            ColumnProfile(name="col", data_type=DataType.FLOAT, n_values=10, n_nulls=6),
        ]
        report = self.analyzer.analyze(profiles, n_rows=10)
        assert report.suggested_strategy["col"] == MissingDataStrategy.SKIP_ROW

    def test_numeric_missing_fill_mean_strategy(self):
        """Numeric column with moderate missing should suggest FILL_MEAN."""
        profiles = [
            ColumnProfile(name="col", data_type=DataType.FLOAT, n_values=10, n_nulls=3),
        ]
        report = self.analyzer.analyze(profiles, n_rows=10)
        assert report.suggested_strategy["col"] == MissingDataStrategy.FILL_MEAN

    def test_string_missing_flag_only_strategy(self):
        """String column with moderate missing should suggest FLAG_ONLY."""
        profiles = [
            ColumnProfile(name="col", data_type=DataType.STRING, n_values=10, n_nulls=3),
        ]
        report = self.analyzer.analyze(profiles, n_rows=10)
        assert report.suggested_strategy["col"] == MissingDataStrategy.FLAG_ONLY


# ===================================================================
# TestDataIngestionAgent (~12 tests)
# ===================================================================


class TestDataIngestionAgent:
    """Tests for DataIngestionAgent."""

    def setup_method(self):
        self.agent = DataIngestionAgent()
        self.store = ExperimentStore()

    def _basic_csv(self):
        return (
            "iteration,temperature,pressure,yield\n"
            "0,100,1.0,85.5\n"
            "1,110,1.5,87.2\n"
            "2,120,2.0,90.1\n"
            "3,130,2.5,92.8\n"
            "4,140,3.0,95.0\n"
        )

    def test_ingest_csv_string_happy_path(self):
        report = self.agent.ingest_csv_string(self._basic_csv(), self.store, campaign_id="test")
        assert report.source_format == "csv"
        assert report.n_rows == 5
        assert report.n_columns == 4
        assert report.experiments_created == 5
        assert report.campaign_id == "test"

    def test_ingest_json_string_happy_path(self):
        data = json.dumps([
            {"iteration": 0, "temperature": 100, "yield": 85.5},
            {"iteration": 1, "temperature": 110, "yield": 87.2},
        ])
        report = self.agent.ingest_json_string(data, self.store, campaign_id="json_test")
        assert report.source_format == "json"
        assert report.n_rows == 2
        assert report.experiments_created == 2

    def test_ingest_csv_string_with_role_overrides(self):
        csv_data = "idx,temp,press,output\n0,100,1.0,85.5\n1,110,1.5,87.2\n"
        report = self.agent.ingest_csv_string(
            csv_data,
            self.store,
            campaign_id="overrides",
            role_overrides={
                "output": ColumnRole.KPI,
                "temp": ColumnRole.PARAMETER,
            },
        )
        profile_map = {p.name: p for p in report.column_profiles}
        assert profile_map["output"].inferred_role == ColumnRole.KPI
        assert profile_map["output"].role_confidence == 1.0
        assert profile_map["temp"].inferred_role == ColumnRole.PARAMETER

    def test_ingest_csv_string_empty_data(self):
        csv_data = "a,b,c\n"
        report = self.agent.ingest_csv_string(csv_data, self.store, campaign_id="empty")
        assert report.n_rows == 0
        assert report.n_columns == 0
        assert report.experiments_created == 0
        assert "No data rows found" in report.warnings

    def test_report_row_and_column_counts(self):
        csv_data = "a,b\n1,2\n3,4\n5,6\n"
        report = self.agent.ingest_csv_string(csv_data, self.store, campaign_id="cnt")
        assert report.n_rows == 3
        assert report.n_columns == 2

    def test_experiments_created_in_store(self):
        self.agent.ingest_csv_string(self._basic_csv(), self.store, campaign_id="store_test")
        assert len(self.store) == 5
        experiments = self.store.get_by_campaign("store_test")
        assert len(experiments) == 5

    def test_parameters_and_kpis_mapped(self):
        csv_data = "iteration,temperature,yield\n0,100,85.5\n1,110,87.2\n"
        self.agent.ingest_csv_string(csv_data, self.store, campaign_id="mapping")
        experiments = self.store.get_by_campaign("mapping")
        for exp in experiments:
            assert exp.parameters or exp.kpi_values

    def test_iteration_column_used_for_experiment_iteration(self):
        csv_data = "iteration,value\n10,1.0\n20,2.0\n30,3.0\n"
        self.agent.ingest_csv_string(csv_data, self.store, campaign_id="iter_test")
        experiments = self.store.get_by_campaign("iter_test")
        iterations = sorted([e.iteration for e in experiments])
        assert iterations == [10, 20, 30]

    def test_campaign_id_auto_derived_from_file_path(self):
        assert _campaign_from_path("/data/experiments/my_data.csv") == "my_data"
        assert _campaign_from_path("results.json") == "results"
        assert _campaign_from_path("/some/path/campaign_42.csv") == "campaign_42"

    def test_duplicate_experiment_id_warning_not_crash(self):
        csv_data = "iteration,value\n0,1.0\n0,2.0\n"
        report = self.agent.ingest_csv_string(csv_data, self.store, campaign_id="dup")
        assert report.experiments_created == 1
        assert any("Duplicate" in w for w in report.warnings)

    def test_ingest_json_string_dict_of_lists(self):
        data = json.dumps({
            "iteration": [0, 1, 2],
            "temperature": [100, 110, 120],
            "yield": [85.0, 87.0, 90.0],
        })
        report = self.agent.ingest_json_string(data, self.store, campaign_id="dol")
        assert report.n_rows == 3
        assert report.experiments_created == 3

    def test_ingest_csv_file_with_tmp_path(self, tmp_path):
        csv_file = tmp_path / "experiment_data.csv"
        csv_file.write_text(self._basic_csv(), encoding="utf-8")
        report = self.agent.ingest_csv(str(csv_file), self.store)
        assert report.n_rows == 5
        assert report.experiments_created == 5
        assert report.campaign_id == "experiment_data"

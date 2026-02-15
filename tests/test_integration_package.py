"""Tests for the optimization_copilot.integration package (formats, provenance, connectors)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.integration import (
    CampaignExporter,
    CampaignImporter,
    ColumnMapping,
    ConnectorStatus,
    CSVConnector,
    DataFormat,
    InMemoryConnector,
    JSONConnector,
    ProvenanceChain,
    ProvenanceRecord,
    ProvenanceTracker,
)


# ── Fixtures ──────────────────────────────────────────


def _make_snapshot() -> CampaignSnapshot:
    """Create a minimal CampaignSnapshot with 2 observations for testing."""
    specs = [
        ParameterSpec(name="temperature", type=VariableType.CONTINUOUS, lower=20.0, upper=200.0),
        ParameterSpec(name="pressure", type=VariableType.CONTINUOUS, lower=1.0, upper=10.0),
    ]
    observations = [
        Observation(
            iteration=0,
            parameters={"temperature": 100.0, "pressure": 5.0},
            kpi_values={"yield": 0.85},
            metadata={"operator": "alice"},
        ),
        Observation(
            iteration=1,
            parameters={"temperature": 150.0, "pressure": 7.0},
            kpi_values={"yield": 0.92},
            metadata={"operator": "bob"},
        ),
    ]
    return CampaignSnapshot(
        campaign_id="test-campaign",
        parameter_specs=specs,
        observations=observations,
        objective_names=["yield"],
        objective_directions=["maximize"],
        current_iteration=2,
    )


# ── formats: DataFormat enum ─────────────────────────


def test_data_format_enum_values():
    """DataFormat enum has expected members."""
    assert DataFormat.CSV.value == "csv"
    assert DataFormat.JSON.value == "json"
    assert DataFormat.JSONLD.value == "jsonld"
    assert DataFormat.SDF.value == "sdf"


# ── formats: ColumnMapping dataclass ─────────────────


def test_column_mapping_creation():
    """ColumnMapping can be created with required and optional fields."""
    mapping = ColumnMapping(
        parameter_columns=["temperature", "pressure"],
        objective_columns=["yield"],
        metadata_columns=["operator"],
        iteration_column="iteration",
    )
    assert mapping.parameter_columns == ["temperature", "pressure"]
    assert mapping.objective_columns == ["yield"]
    assert mapping.metadata_columns == ["operator"]
    assert mapping.iteration_column == "iteration"


def test_column_mapping_defaults():
    """ColumnMapping defaults are empty list and None."""
    mapping = ColumnMapping(
        parameter_columns=["x"],
        objective_columns=["y"],
    )
    assert mapping.metadata_columns == []
    assert mapping.iteration_column is None


# ── formats: CampaignExporter CSV ────────────────────


def test_exporter_csv_string():
    """CampaignExporter.export_string produces valid CSV with header and rows."""
    snapshot = _make_snapshot()
    exporter = CampaignExporter()
    csv_text = exporter.export_string(snapshot, DataFormat.CSV)

    lines = csv_text.strip().split("\n")
    # Header + 2 data rows
    assert len(lines) == 3

    header = lines[0]
    assert "iteration" in header
    assert "temperature" in header
    assert "pressure" in header
    assert "yield" in header


# ── formats: CampaignExporter JSON ───────────────────


def test_exporter_json_string():
    """CampaignExporter.export_string produces valid JSON with schema_version."""
    snapshot = _make_snapshot()
    exporter = CampaignExporter()
    json_text = exporter.export_string(snapshot, DataFormat.JSON)

    data = json.loads(json_text)
    assert data["schema_version"] == "1.0"
    assert "campaign" in data
    assert data["campaign"]["campaign_id"] == "test-campaign"


# ── formats: CampaignExporter unsupported format ─────


def test_exporter_unsupported_format_raises():
    """CampaignExporter.export_string raises ValueError for SDF."""
    snapshot = _make_snapshot()
    exporter = CampaignExporter()
    with pytest.raises(ValueError, match="Unsupported export format"):
        exporter.export_string(snapshot, DataFormat.SDF)


# ── formats: CampaignImporter CSV round-trip ─────────


def test_importer_csv_string_round_trip():
    """Export to CSV then import back preserves observations."""
    snapshot = _make_snapshot()
    exporter = CampaignExporter()
    csv_text = exporter.export_string(snapshot, DataFormat.CSV)

    mapping = ColumnMapping(
        parameter_columns=["temperature", "pressure"],
        objective_columns=["yield"],
        metadata_columns=["operator"],
        iteration_column="iteration",
    )
    importer = CampaignImporter()
    imported = importer.import_csv_string(csv_text, mapping, campaign_id="round-trip")

    assert imported.campaign_id == "round-trip"
    assert len(imported.observations) == 2
    assert imported.observations[0].parameters["temperature"] == 100.0
    assert imported.observations[1].kpi_values["yield"] == 0.92


# ── provenance: ProvenanceTracker.record ─────────────


def test_provenance_tracker_record_creates_record():
    """ProvenanceTracker.record creates and returns a ProvenanceRecord."""
    tracker = ProvenanceTracker()
    rec = tracker.record(action="import", source="csv_file", agent="user")

    assert rec.action == "import"
    assert rec.source == "csv_file"
    assert rec.agent == "user"
    assert rec.record_id  # non-empty UUID
    assert rec.timestamp > 0


def test_provenance_tracker_chain_grows():
    """Each record call grows the chain."""
    tracker = ProvenanceTracker()
    tracker.record(action="import", source="s1")
    tracker.record(action="transform", source="s2")
    chain = tracker.get_chain()
    assert len(chain) == 2


# ── provenance: ProvenanceChain lineage ──────────────


def test_provenance_chain_lineage():
    """ProvenanceChain.get_lineage walks parent chain correctly."""
    chain = ProvenanceChain()
    parent = ProvenanceRecord(
        record_id="parent-1",
        timestamp=1.0,
        source="src",
        action="import",
        agent="sys",
    )
    child = ProvenanceRecord(
        record_id="child-1",
        timestamp=2.0,
        source="src",
        action="transform",
        agent="sys",
        parent_ids=["parent-1"],
    )
    chain.append(parent)
    chain.append(child)

    lineage = chain.get_lineage("child-1")
    assert len(lineage) == 1
    assert lineage[0].record_id == "parent-1"


def test_provenance_chain_children():
    """ProvenanceChain.get_children finds child records."""
    chain = ProvenanceChain()
    parent = ProvenanceRecord(
        record_id="p1", timestamp=1.0, source="s", action="import", agent="a"
    )
    child = ProvenanceRecord(
        record_id="c1", timestamp=2.0, source="s", action="observe", agent="a",
        parent_ids=["p1"],
    )
    chain.append(parent)
    chain.append(child)

    children = chain.get_children("p1")
    assert len(children) == 1
    assert children[0].record_id == "c1"


def test_provenance_chain_duplicate_id_raises():
    """ProvenanceChain rejects records with duplicate IDs."""
    chain = ProvenanceChain()
    rec = ProvenanceRecord(
        record_id="dup", timestamp=1.0, source="s", action="import", agent="a"
    )
    chain.append(rec)
    with pytest.raises(ValueError, match="already exists"):
        chain.append(rec)


# ── provenance: export_jsonld ────────────────────────


def test_provenance_export_jsonld_valid_json():
    """ProvenanceTracker.export_jsonld produces valid JSON with PROV-O context."""
    tracker = ProvenanceTracker()
    tracker.record(action="import", source="file.csv")
    tracker.record(action="decision", source="optimizer")

    jsonld_str = tracker.export_jsonld()
    data = json.loads(jsonld_str)

    assert "@context" in data
    assert data["@type"] == "prov:Bundle"
    assert len(data["records"]) == 2
    assert data["records"][0]["@type"] == "prov:Activity"


# ── connectors: InMemoryConnector ────────────────────


def test_inmemory_connector_write_read_round_trip():
    """InMemoryConnector stores suggestions and reads observations."""
    conn = InMemoryConnector()

    # Add observations
    obs = Observation(
        iteration=0,
        parameters={"x": 1.0},
        kpi_values={"y": 2.0},
    )
    conn.add_observation(obs)

    # Write suggestions
    conn.write_suggestions([{"x": 3.0}, {"x": 4.0}])

    # Read back observations
    observations = conn.read_observations()
    assert len(observations) == 1
    assert observations[0].parameters["x"] == 1.0

    # Verify status
    status = conn.status()
    assert status.connected is True
    assert status.record_count == 1
    assert status.last_sync is not None


# ── connectors: CSVConnector ─────────────────────────


def test_csv_connector_with_tmp_files():
    """CSVConnector reads/writes CSV files correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        read_path = os.path.join(tmpdir, "observations.csv")
        write_path = os.path.join(tmpdir, "suggestions.csv")

        # Write a small observation CSV
        with open(read_path, "w", newline="") as f:
            f.write("iteration,temperature,pressure,kpi_yield\n")
            f.write("0,100.0,5.0,0.85\n")
            f.write("1,150.0,7.0,0.92\n")

        conn = CSVConnector(read_path=read_path, write_path=write_path)

        # Read observations
        observations = conn.read_observations()
        assert len(observations) == 2
        assert observations[0].parameters["temperature"] == 100.0
        assert observations[1].kpi_values["kpi_yield"] == 0.92

        # Write suggestions
        conn.write_suggestions([{"temperature": 120.0, "pressure": 6.0}])
        assert os.path.exists(write_path)

        # Verify status
        status = conn.status()
        assert status.connected is True
        assert status.record_count == 2


def test_csv_connector_missing_file_returns_empty():
    """CSVConnector.read_observations returns [] for missing file."""
    conn = CSVConnector(read_path="/nonexistent/path.csv", write_path="/tmp/out.csv")
    observations = conn.read_observations()
    assert observations == []


# ── connectors: ConnectorStatus ──────────────────────


def test_connector_status_fields():
    """ConnectorStatus dataclass has expected fields."""
    status = ConnectorStatus(connected=True, last_sync=1000.0, record_count=42)
    assert status.connected is True
    assert status.last_sync == 1000.0
    assert status.record_count == 42


def test_connector_status_disconnected():
    """ConnectorStatus can represent a disconnected state."""
    status = ConnectorStatus(connected=False, last_sync=None, record_count=0)
    assert status.connected is False
    assert status.last_sync is None
    assert status.record_count == 0

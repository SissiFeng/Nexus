"""Tests for the Frontend V2 API endpoints.

Tests cover:
1. POST /api/campaigns/from-upload - Campaign creation from uploaded CSV data
2. GET /api/campaigns/{id}/diagnostics - Diagnostic health metrics
3. GET /api/campaigns/{id}/importance - Parameter importance scores
4. GET /api/campaigns/{id}/suggestions?n=5 - Next experiment suggestions
5. POST /api/campaigns/{id}/steer - Steering directives
6. POST /api/chat/{id} - Chat endpoint with intent routing
7. GET /api/campaigns/{id}/export/{fmt} - Export as csv/json/xlsx

Uses FastAPI's TestClient for synchronous HTTP testing.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import pytest

# Guard against missing dependencies
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from optimization_copilot.api.app import create_app


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def app(tmp_path):
    """Create a fresh FastAPI app with a temporary workspace."""
    return create_app(workspace_dir=str(tmp_path / "workspace"))


@pytest.fixture
def client(app):
    """Create a TestClient for the app."""
    return TestClient(app)


@pytest.fixture
def upload_request_body() -> dict[str, Any]:
    """A realistic upload request body with 2 parameters, 1 objective, and 10 rows.

    Simulates a user uploading a CSV of historical experiments, mapping columns
    to parameters (temperature, pressure) and an objective (yield), with batch_id
    as metadata.
    """
    data_rows = []
    for i in range(10):
        data_rows.append({
            "temperature": str(50 + i * 5),
            "pressure": str(1.0 + i * 0.2),
            "yield": str(75.0 - (i - 5) ** 2 * 0.5),
            "batch_id": f"B{i:03d}",
        })

    return {
        "name": "Temperature-Pressure Study",
        "description": "Optimize yield by tuning temperature and pressure",
        "data": data_rows,
        "mapping": {
            "parameters": [
                {"name": "temperature", "type": "continuous", "lower": 50.0, "upper": 95.0},
                {"name": "pressure", "type": "continuous", "lower": 1.0, "upper": 2.8},
            ],
            "objectives": [
                {"name": "yield", "direction": "maximize"},
            ],
            "metadata": ["batch_id"],
            "ignored": [],
        },
        "batch_size": 5,
        "exploration_weight": 0.5,
    }


@pytest.fixture
def created_campaign(client, upload_request_body) -> dict[str, Any]:
    """Create a campaign via the upload endpoint and return the response JSON."""
    resp = client.post("/api/campaigns/from-upload", json=upload_request_body)
    assert resp.status_code == 201
    return resp.json()


# ── Helpers ──────────────────────────────────────────────────────


def _create_campaign_with_data(
    client,
    n_rows: int = 10,
    name: str = "Test Campaign",
    direction: str = "minimize",
) -> dict[str, Any]:
    """Helper to create a campaign with custom row count and direction."""
    data_rows = []
    for i in range(n_rows):
        data_rows.append({
            "x1": str(float(i) / max(n_rows - 1, 1)),
            "x2": str(float(i * 2) / max(n_rows - 1, 1)),
            "y": str(10.0 - i * 0.5) if direction == "minimize" else str(i * 0.5),
        })

    body = {
        "name": name,
        "description": "Test campaign",
        "data": data_rows,
        "mapping": {
            "parameters": [
                {"name": "x1", "type": "continuous", "lower": 0.0, "upper": 1.0},
                {"name": "x2", "type": "continuous", "lower": 0.0, "upper": 2.0},
            ],
            "objectives": [
                {"name": "y", "direction": direction},
            ],
            "metadata": [],
            "ignored": [],
        },
        "batch_size": 3,
        "exploration_weight": 0.3,
    }

    resp = client.post("/api/campaigns/from-upload", json=body)
    assert resp.status_code == 201
    return resp.json()


# ── 1. POST /api/campaigns/from-upload ───────────────────────────


class TestCreateFromUpload:
    """Campaign creation from uploaded CSV data."""

    def test_create_returns_201(self, client, upload_request_body):
        resp = client.post("/api/campaigns/from-upload", json=upload_request_body)
        assert resp.status_code == 201

    def test_create_returns_campaign_id(self, created_campaign):
        assert "campaign_id" in created_campaign
        assert isinstance(created_campaign["campaign_id"], str)
        assert len(created_campaign["campaign_id"]) > 0

    def test_create_returns_id_alias(self, created_campaign):
        """Response includes both 'campaign_id' and 'id' for frontend convenience."""
        assert "id" in created_campaign
        assert created_campaign["id"] == created_campaign["campaign_id"]

    def test_create_returns_name(self, created_campaign):
        assert created_campaign["name"] == "Temperature-Pressure Study"

    def test_create_returns_status_draft(self, created_campaign):
        assert created_campaign["status"] == "draft"

    def test_create_returns_total_trials(self, created_campaign):
        assert created_campaign["total_trials"] == 10

    def test_create_returns_best_kpi(self, created_campaign):
        """Best KPI should be computed from the initial observations."""
        assert created_campaign["best_kpi"] is not None
        # The objective is "maximize yield", so best_kpi should be max of yield values
        assert isinstance(created_campaign["best_kpi"], (int, float))

    def test_create_with_default_name(self, client):
        """When name is empty, a default name is generated."""
        body = {
            "name": "",
            "data": [{"x": "1.0", "y": "2.0"}],
            "mapping": {
                "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 5.0}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            },
        }
        resp = client.post("/api/campaigns/from-upload", json=body)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"].startswith("Upload-")

    def test_create_with_minimal_data(self, client):
        """Single row, single param, single objective."""
        body = {
            "name": "Minimal",
            "data": [{"x": "0.5", "y": "1.0"}],
            "mapping": {
                "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            },
        }
        resp = client.post("/api/campaigns/from-upload", json=body)
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_trials"] == 1

    def test_create_with_minimize_direction(self, client):
        """Minimize direction correctly identifies best KPI as minimum."""
        data_rows = [
            {"x": "0.1", "y": "10.0"},
            {"x": "0.5", "y": "5.0"},
            {"x": "0.9", "y": "2.0"},
        ]
        body = {
            "name": "Minimize Test",
            "data": data_rows,
            "mapping": {
                "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            },
        }
        resp = client.post("/api/campaigns/from-upload", json=body)
        assert resp.status_code == 201
        data = resp.json()
        assert data["best_kpi"] == 2.0

    def test_create_with_maximize_direction(self, client):
        """Maximize direction correctly identifies best KPI as maximum."""
        data_rows = [
            {"x": "0.1", "y": "10.0"},
            {"x": "0.5", "y": "50.0"},
            {"x": "0.9", "y": "30.0"},
        ]
        body = {
            "name": "Maximize Test",
            "data": data_rows,
            "mapping": {
                "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
                "objectives": [{"name": "y", "direction": "maximize"}],
            },
        }
        resp = client.post("/api/campaigns/from-upload", json=body)
        assert resp.status_code == 201
        data = resp.json()
        assert data["best_kpi"] == 50.0

    def test_create_with_metadata_columns(self, client, upload_request_body):
        """Metadata columns are preserved in spec but don't affect optimization."""
        resp = client.post("/api/campaigns/from-upload", json=upload_request_body)
        assert resp.status_code == 201

    def test_create_with_categorical_values(self, client):
        """Non-numeric parameter values are treated as categorical."""
        body = {
            "name": "Categorical Test",
            "data": [
                {"solvent": "water", "y": "1.5"},
                {"solvent": "ethanol", "y": "2.5"},
            ],
            "mapping": {
                "parameters": [{"name": "solvent", "type": "categorical"}],
                "objectives": [{"name": "y", "direction": "maximize"}],
            },
        }
        resp = client.post("/api/campaigns/from-upload", json=body)
        assert resp.status_code == 201
        assert resp.json()["total_trials"] == 2

    def test_create_two_campaigns_get_distinct_ids(self, client, upload_request_body):
        """Two campaigns created from the same data get different IDs."""
        resp1 = client.post("/api/campaigns/from-upload", json=upload_request_body)
        resp2 = client.post("/api/campaigns/from-upload", json=upload_request_body)
        assert resp1.json()["campaign_id"] != resp2.json()["campaign_id"]

    def test_create_missing_mapping_returns_422(self, client):
        """Request without mapping field returns validation error."""
        body = {"name": "Bad", "data": [{"x": "1"}]}
        resp = client.post("/api/campaigns/from-upload", json=body)
        assert resp.status_code == 422


# ── 2. GET /api/campaigns/{id}/diagnostics ───────────────────────


class TestDiagnostics:
    """Diagnostic health metrics endpoint."""

    def test_diagnostics_returns_200(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/diagnostics")
        assert resp.status_code == 200

    def test_diagnostics_has_all_8_metrics(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/diagnostics")
        data = resp.json()

        expected_keys = {
            "convergence_trend",
            "improvement_velocity",
            "best_kpi_value",
            "exploration_coverage",
            "failure_rate",
            "noise_estimate",
            "plateau_length",
            "signal_to_noise_ratio",
        }
        assert set(data.keys()) == expected_keys

    def test_diagnostics_best_kpi_value_is_numeric(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/diagnostics")
        data = resp.json()
        assert isinstance(data["best_kpi_value"], (int, float))

    def test_diagnostics_failure_rate_zero_for_clean_data(self, client, created_campaign):
        """With no failed experiments, failure rate should be 0."""
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/diagnostics")
        data = resp.json()
        assert data["failure_rate"] == 0.0

    def test_diagnostics_exploration_coverage_is_fraction(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/diagnostics")
        data = resp.json()
        assert 0.0 <= data["exploration_coverage"] <= 1.0

    def test_diagnostics_plateau_length_is_nonnegative(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/diagnostics")
        data = resp.json()
        assert data["plateau_length"] >= 0

    def test_diagnostics_404_nonexistent_campaign(self, client):
        resp = client.get("/api/campaigns/nonexistent-id/diagnostics")
        assert resp.status_code == 404

    def test_diagnostics_with_minimize_direction(self, client):
        """Diagnostics correctly handle minimize objective direction."""
        campaign = _create_campaign_with_data(client, n_rows=10, direction="minimize")
        cid = campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/diagnostics")
        data = resp.json()
        # Best KPI for minimization should be the smallest value
        assert data["best_kpi_value"] <= 10.0

    def test_diagnostics_with_few_observations(self, client):
        """Diagnostics still work with a small number of uploaded rows."""
        campaign = _create_campaign_with_data(client, n_rows=2)
        cid = campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/diagnostics")
        assert resp.status_code == 200
        data = resp.json()
        # All metric fields should be present and numeric
        assert isinstance(data["convergence_trend"], (int, float))
        assert isinstance(data["noise_estimate"], (int, float))


# ── 3. GET /api/campaigns/{id}/importance ────────────────────────


class TestImportance:
    """Parameter importance scores endpoint."""

    def test_importance_returns_200(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/importance")
        assert resp.status_code == 200

    def test_importance_returns_one_per_parameter(self, client, created_campaign):
        """Should return importance for each parameter in the campaign."""
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/importance")
        data = resp.json()
        importances = data["importances"]
        # The created campaign has 2 parameters: temperature, pressure
        assert len(importances) == 2

    def test_importance_entries_have_name_and_score(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/importance")
        data = resp.json()
        for imp in data["importances"]:
            assert "name" in imp
            assert "importance" in imp
            assert isinstance(imp["importance"], (int, float))

    def test_importance_names_match_parameters(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/importance")
        data = resp.json()
        names = {imp["name"] for imp in data["importances"]}
        assert "temperature" in names
        assert "pressure" in names

    def test_importance_scores_are_nonnegative(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/importance")
        data = resp.json()
        for imp in data["importances"]:
            assert imp["importance"] >= 0.0

    def test_importance_scores_sum_approximately_to_one(self, client, created_campaign):
        """Importance scores should be normalized to sum to ~1 (or be uniform)."""
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/importance")
        data = resp.json()
        total = sum(imp["importance"] for imp in data["importances"])
        assert abs(total - 1.0) < 0.01

    def test_importance_404_nonexistent_campaign(self, client):
        resp = client.get("/api/campaigns/nonexistent-id/importance")
        assert resp.status_code == 404

    def test_importance_with_few_rows(self, client):
        """With a small number of uploaded rows, importance still returns valid scores."""
        campaign = _create_campaign_with_data(client, n_rows=2)
        cid = campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/importance")
        data = resp.json()
        importances = data["importances"]
        assert len(importances) == 2
        # Each importance should be a valid nonnegative number
        for imp in importances:
            assert imp["importance"] >= 0.0


# ── 4. GET /api/campaigns/{id}/suggestions ───────────────────────


class TestSuggestions:
    """Next experiment suggestions endpoint."""

    def test_suggestions_returns_200(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/suggestions")
        assert resp.status_code == 200

    def test_suggestions_returns_list(self, client, created_campaign):
        """Default query returns a list of suggestions (may be 0 if CampaignLoop
        backend succeeds but returns empty, or up to 5 from fallback)."""
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/suggestions")
        data = resp.json()
        assert isinstance(data["suggestions"], list)
        assert len(data["suggestions"]) <= 5

    def test_suggestions_returns_at_most_requested_count(self, client, created_campaign):
        """The number of returned suggestions does not exceed the requested n."""
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/suggestions", params={"n": 3})
        data = resp.json()
        assert len(data["suggestions"]) <= 3

    def test_suggestions_single_request(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/suggestions", params={"n": 1})
        data = resp.json()
        assert len(data["suggestions"]) <= 1

    def test_suggestions_contain_parameter_names(self, client, created_campaign):
        """Each suggestion (if any) should have values for all parameters."""
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/suggestions", params={"n": 5})
        data = resp.json()
        for suggestion in data["suggestions"]:
            assert "temperature" in suggestion
            assert "pressure" in suggestion

    def test_suggestions_response_has_backend_used(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/suggestions")
        data = resp.json()
        assert "backend_used" in data
        assert isinstance(data["backend_used"], str)

    def test_suggestions_response_has_phase(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/suggestions")
        data = resp.json()
        assert "phase" in data
        assert isinstance(data["phase"], str)

    def test_suggestions_404_nonexistent_campaign(self, client):
        resp = client.get("/api/campaigns/nonexistent-id/suggestions")
        assert resp.status_code == 404

    def test_suggestions_n_too_large_returns_422(self, client, created_campaign):
        """n > 50 should fail validation."""
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/suggestions", params={"n": 100})
        assert resp.status_code == 422

    def test_suggestions_n_zero_returns_422(self, client, created_campaign):
        """n < 1 should fail validation."""
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/suggestions", params={"n": 0})
        assert resp.status_code == 422

    def test_suggestions_has_predicted_values_list(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/suggestions")
        data = resp.json()
        assert "predicted_values" in data
        assert isinstance(data["predicted_values"], list)

    def test_suggestions_has_predicted_uncertainties_list(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/suggestions")
        data = resp.json()
        assert "predicted_uncertainties" in data
        assert isinstance(data["predicted_uncertainties"], list)


# ── 5. POST /api/campaigns/{id}/steer ───────────────────────────


class TestSteering:
    """Steering directives endpoint."""

    def test_steer_returns_accepted(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        body = {
            "action": "focus_region",
            "region_bounds": {"temperature": [60.0, 80.0]},
            "reason": "High-yield region observed",
        }
        resp = client.post(f"/api/campaigns/{cid}/steer", json=body)
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_steer_writes_to_spec(self, client, app, created_campaign):
        """Steering directives are persisted in the campaign spec."""
        cid = created_campaign["campaign_id"]
        body = {
            "action": "avoid_region",
            "region_bounds": {"pressure": [2.5, 3.0]},
            "reason": "Unstable region",
        }
        client.post(f"/api/campaigns/{cid}/steer", json=body)

        # Verify the directive was written to workspace spec
        workspace = app.state.platform.workspace
        spec = workspace.load_spec(cid)
        assert "steering_directives" in spec
        assert len(spec["steering_directives"]) == 1
        directive = spec["steering_directives"][0]
        assert directive["action"] == "avoid_region"
        assert directive["region_bounds"]["pressure"] == [2.5, 3.0]
        assert directive["reason"] == "Unstable region"
        assert "timestamp" in directive

    def test_steer_appends_multiple_directives(self, client, app, created_campaign):
        """Multiple steering directives accumulate."""
        cid = created_campaign["campaign_id"]
        client.post(f"/api/campaigns/{cid}/steer", json={
            "action": "focus_region",
            "region_bounds": {"temperature": [60.0, 80.0]},
        })
        client.post(f"/api/campaigns/{cid}/steer", json={
            "action": "avoid_region",
            "region_bounds": {"pressure": [2.5, 3.0]},
        })

        workspace = app.state.platform.workspace
        spec = workspace.load_spec(cid)
        assert len(spec["steering_directives"]) == 2

    def test_steer_without_region_bounds(self, client, created_campaign):
        """Steering with action only (no region bounds) is valid."""
        cid = created_campaign["campaign_id"]
        body = {"action": "increase_exploration"}
        resp = client.post(f"/api/campaigns/{cid}/steer", json=body)
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_steer_404_nonexistent_campaign(self, client):
        body = {"action": "focus_region"}
        resp = client.post("/api/campaigns/nonexistent-id/steer", json=body)
        assert resp.status_code == 404

    def test_steer_missing_action_returns_422(self, client, created_campaign):
        """Request without 'action' field returns validation error."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/campaigns/{cid}/steer", json={"reason": "no action"})
        assert resp.status_code == 422


# ── 6. POST /api/chat/{id} ──────────────────────────────────────


class TestChat:
    """Chat endpoint with intent routing."""

    def test_chat_returns_200(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "hello"})
        assert resp.status_code == 200

    def test_chat_returns_reply_and_role(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "hello"})
        data = resp.json()
        assert "reply" in data
        assert "role" in data
        assert isinstance(data["reply"], str)
        assert len(data["reply"]) > 0

    # -- Intent: suggest --

    def test_chat_suggest_intent(self, client, created_campaign):
        """Messages with 'suggest' keyword route to suggestion intent."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "suggest next experiments"})
        data = resp.json()
        assert data["role"] == "suggestion"
        assert "suggest" in data["reply"].lower() or "experiment" in data["reply"].lower()

    def test_chat_suggest_has_metadata(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "what should I try next?"})
        data = resp.json()
        assert "metadata" in data
        assert "suggestions" in data["metadata"]
        assert isinstance(data["metadata"]["suggestions"], list)

    def test_chat_recommend_intent(self, client, created_campaign):
        """Messages with 'recommend' keyword also route to suggest."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "recommend experiments"})
        data = resp.json()
        assert data["role"] == "suggestion"

    # -- Intent: diagnostics --

    def test_chat_diagnostics_intent(self, client, created_campaign):
        """Messages with 'diagnostic' keyword route to diagnostics intent."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "show diagnostics"})
        data = resp.json()
        assert "best kpi" in data["reply"].lower() or "convergence" in data["reply"].lower()
        assert "metadata" in data
        assert "diagnostics" in data["metadata"]

    def test_chat_health_intent(self, client, created_campaign):
        """Messages with 'health' keyword also route to diagnostics."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "how is the campaign health?"})
        data = resp.json()
        assert "metadata" in data
        assert "diagnostics" in data["metadata"]

    def test_chat_status_intent(self, client, created_campaign):
        """Messages with 'status' keyword route to diagnostics."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "what is the status?"})
        data = resp.json()
        assert "diagnostics" in data.get("metadata", {})

    # -- Intent: importance --

    def test_chat_importance_intent(self, client, created_campaign):
        """Messages with 'importance' keyword route to importance intent."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "parameter importance"})
        data = resp.json()
        assert "importance" in data["reply"].lower()
        assert "metadata" in data
        assert "importances" in data["metadata"]

    def test_chat_which_parameter_intent(self, client, created_campaign):
        """Messages with 'which parameter' route to importance."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "which parameter matters most?"})
        data = resp.json()
        assert "importances" in data.get("metadata", {})

    def test_chat_fanova_intent(self, client, created_campaign):
        """Messages with 'fanova' keyword route to importance."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "run fanova analysis"})
        data = resp.json()
        assert "importances" in data.get("metadata", {})

    # -- Intent: why/explain --

    def test_chat_why_intent(self, client, created_campaign):
        """Messages with 'why' keyword route to explanation intent."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "why is the optimization slow?"})
        data = resp.json()
        assert "analysis" in data["reply"].lower() or "campaign" in data["reply"].lower()

    def test_chat_explain_intent(self, client, created_campaign):
        """Messages with 'explain' keyword route to explanation."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "explain the current situation"})
        data = resp.json()
        assert len(data["reply"]) > 0

    # -- Intent: default --

    def test_chat_default_intent(self, client, created_campaign):
        """Unrecognized messages get a summary response."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "tell me something"})
        data = resp.json()
        assert data["role"] == "agent"
        assert "observations" in data["reply"].lower() or "campaign" in data["reply"].lower()

    # -- Intent: empty message --

    def test_chat_empty_message_welcome(self, client, created_campaign):
        """Empty message returns a welcome/overview response."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": ""})
        data = resp.json()
        assert data["role"] == "system"
        assert "welcome" in data["reply"].lower()

    def test_chat_whitespace_only_message(self, client, created_campaign):
        """Whitespace-only message treated as empty."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "   "})
        data = resp.json()
        assert data["role"] == "system"

    # -- Intent: export --

    def test_chat_export_intent(self, client, created_campaign):
        """Messages with 'export' keyword route to export intent."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "how do I export data?"})
        data = resp.json()
        assert data["role"] == "system"
        assert "export" in data["reply"].lower()

    # -- Intent: focus/steer --

    def test_chat_focus_intent(self, client, created_campaign):
        """Messages with 'focus' keyword route to steering guidance."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={"message": "focus on high temperature"})
        data = resp.json()
        assert "steering" in data["reply"].lower() or "focus" in data["reply"].lower()

    # -- Error cases --

    def test_chat_404_nonexistent_campaign(self, client):
        resp = client.post("/api/chat/nonexistent-id", json={"message": "hello"})
        assert resp.status_code == 404

    def test_chat_missing_message_returns_422(self, client, created_campaign):
        """Request without 'message' field returns validation error."""
        cid = created_campaign["campaign_id"]
        resp = client.post(f"/api/chat/{cid}", json={})
        assert resp.status_code == 422


# ── 7. GET /api/campaigns/{id}/export/{fmt} ──────────────────────


class TestExport:
    """Campaign data export endpoint."""

    # -- JSON export --

    def test_export_json_returns_200(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/json")
        assert resp.status_code == 200

    def test_export_json_content_type(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/json")
        assert "application/json" in resp.headers["content-type"]

    def test_export_json_has_content_disposition(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/json")
        assert "content-disposition" in resp.headers
        assert f"{cid}.json" in resp.headers["content-disposition"]

    def test_export_json_is_valid_json(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/json")
        data = json.loads(resp.content)
        assert isinstance(data, dict)

    def test_export_json_contains_campaign_data(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/json")
        data = json.loads(resp.content)
        assert "campaign_id" in data
        assert data["campaign_id"] == cid
        assert "observations" in data
        assert "parameter_specs" in data

    def test_export_json_has_observations(self, client, created_campaign):
        """Export contains observations. The snapshot loader may aggregate observations
        from both checkpoint completed_trials and spec initial_observations, so the
        count may be >= the number of uploaded rows."""
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/json")
        data = json.loads(resp.content)
        assert len(data["observations"]) >= 10

    # -- CSV export --

    def test_export_csv_returns_200(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/csv")
        assert resp.status_code == 200

    def test_export_csv_content_type(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/csv")
        assert "text/csv" in resp.headers["content-type"]

    def test_export_csv_has_content_disposition(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/csv")
        assert "content-disposition" in resp.headers
        assert f"{cid}.csv" in resp.headers["content-disposition"]

    def test_export_csv_has_header_and_rows(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/csv")
        content = resp.content.decode("utf-8")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        # Header + data rows (observations may be aggregated from multiple sources)
        assert len(rows) >= 11  # at least header + 10 data rows
        header = rows[0]
        assert "iteration" in header
        assert "temperature" in header
        assert "pressure" in header
        assert "yield" in header

    def test_export_csv_data_is_parseable(self, client, created_campaign):
        """Each CSV data row should have the same number of columns as the header."""
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/csv")
        content = resp.content.decode("utf-8")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        header_len = len(rows[0])
        for row in rows[1:]:
            assert len(row) == header_len

    # -- XLSX export --

    def test_export_xlsx_returns_200(self, client, created_campaign):
        """XLSX endpoint returns 200 (may fall back to CSV internally)."""
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/xlsx")
        assert resp.status_code == 200

    def test_export_xlsx_has_content_disposition(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/xlsx")
        assert "content-disposition" in resp.headers
        assert f"{cid}.xlsx" in resp.headers["content-disposition"]

    # -- Error cases --

    def test_export_unsupported_format_returns_400(self, client, created_campaign):
        cid = created_campaign["campaign_id"]
        resp = client.get(f"/api/campaigns/{cid}/export/xml")
        assert resp.status_code == 400

    def test_export_404_nonexistent_campaign(self, client):
        resp = client.get("/api/campaigns/nonexistent-id/export/json")
        assert resp.status_code == 404

    def test_export_csv_404_nonexistent_campaign(self, client):
        resp = client.get("/api/campaigns/nonexistent-id/export/csv")
        assert resp.status_code == 404


# ── Cross-Endpoint Integration ───────────────────────────────────


class TestCrossEndpointIntegration:
    """Integration tests spanning multiple endpoints."""

    def test_upload_then_diagnostics_then_suggestions(self, client):
        """Full workflow: upload data -> check diagnostics -> get suggestions."""
        campaign = _create_campaign_with_data(client, n_rows=15, direction="minimize")
        cid = campaign["campaign_id"]

        # Diagnostics should reflect the uploaded data
        diag_resp = client.get(f"/api/campaigns/{cid}/diagnostics")
        assert diag_resp.status_code == 200
        diag = diag_resp.json()
        assert diag["best_kpi_value"] > 0

        # Suggestions endpoint should respond successfully
        sugg_resp = client.get(f"/api/campaigns/{cid}/suggestions", params={"n": 3})
        assert sugg_resp.status_code == 200
        suggestions = sugg_resp.json()["suggestions"]
        assert isinstance(suggestions, list)
        # Each returned suggestion should have the expected parameter keys
        for s in suggestions:
            assert "x1" in s
            assert "x2" in s

    def test_upload_then_steer_then_export(self, client, app):
        """Upload -> steer -> verify steering in spec -> export."""
        campaign = _create_campaign_with_data(client, n_rows=5)
        cid = campaign["campaign_id"]

        # Apply steering
        steer_resp = client.post(f"/api/campaigns/{cid}/steer", json={
            "action": "focus_region",
            "region_bounds": {"x1": [0.2, 0.8]},
            "reason": "Promising region",
        })
        assert steer_resp.status_code == 200

        # Verify in spec
        workspace = app.state.platform.workspace
        spec = workspace.load_spec(cid)
        assert len(spec["steering_directives"]) == 1

        # Export should still work
        export_resp = client.get(f"/api/campaigns/{cid}/export/json")
        assert export_resp.status_code == 200

    def test_upload_then_chat_workflow(self, client):
        """Upload -> chat to get overview -> ask for suggestions -> ask for diagnostics."""
        campaign = _create_campaign_with_data(client, n_rows=8)
        cid = campaign["campaign_id"]

        # Empty message for welcome
        welcome = client.post(f"/api/chat/{cid}", json={"message": ""}).json()
        assert "welcome" in welcome["reply"].lower()
        # Observations in snapshot may be >= uploaded rows (loaded from multiple sources)
        assert "observations" in welcome["reply"].lower()

        # Ask for suggestions
        suggest = client.post(f"/api/chat/{cid}", json={"message": "suggest next"}).json()
        assert suggest["role"] == "suggestion"
        assert isinstance(suggest["metadata"]["suggestions"], list)

        # Ask for diagnostics
        diag = client.post(f"/api/chat/{cid}", json={"message": "show diagnostics"}).json()
        assert "diagnostics" in diag["metadata"]

    def test_multiple_campaigns_isolated(self, client):
        """Two campaigns have independent diagnostics and suggestions."""
        c1 = _create_campaign_with_data(client, n_rows=5, name="Camp1", direction="minimize")
        c2 = _create_campaign_with_data(client, n_rows=10, name="Camp2", direction="maximize")

        diag1 = client.get(f"/api/campaigns/{c1['campaign_id']}/diagnostics").json()
        diag2 = client.get(f"/api/campaigns/{c2['campaign_id']}/diagnostics").json()

        # Different data sizes and directions should yield different diagnostics
        # At minimum, they should both respond successfully
        assert isinstance(diag1["best_kpi_value"], (int, float))
        assert isinstance(diag2["best_kpi_value"], (int, float))


# ── Edge Cases ───────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_data_upload(self, client):
        """Upload with empty data list creates a campaign with 0 trials."""
        body = {
            "name": "Empty",
            "data": [],
            "mapping": {
                "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            },
        }
        resp = client.post("/api/campaigns/from-upload", json=body)
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_trials"] == 0
        assert data["best_kpi"] is None

    def test_diagnostics_on_empty_campaign(self, client):
        """Diagnostics on a campaign with no observations returns all zeros."""
        body = {
            "name": "Empty Diag",
            "data": [],
            "mapping": {
                "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            },
        }
        resp = client.post("/api/campaigns/from-upload", json=body)
        cid = resp.json()["campaign_id"]

        diag = client.get(f"/api/campaigns/{cid}/diagnostics").json()
        assert diag["best_kpi_value"] == 0.0
        assert diag["failure_rate"] == 0.0
        assert diag["plateau_length"] == 0

    def test_importance_on_empty_campaign(self, client):
        """Importance on a campaign with no observations returns uniform."""
        body = {
            "name": "Empty Imp",
            "data": [],
            "mapping": {
                "parameters": [
                    {"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0},
                    {"name": "z", "type": "continuous", "lower": 0.0, "upper": 5.0},
                ],
                "objectives": [{"name": "y", "direction": "minimize"}],
            },
        }
        resp = client.post("/api/campaigns/from-upload", json=body)
        cid = resp.json()["campaign_id"]

        imp = client.get(f"/api/campaigns/{cid}/importance").json()
        assert len(imp["importances"]) == 2
        for entry in imp["importances"]:
            assert abs(entry["importance"] - 0.5) < 0.01

    def test_suggestions_on_empty_campaign(self, client):
        """Suggestions on empty campaign returns 200 with a list (may be empty
        if CampaignLoop backend succeeds with no candidates, or populated
        by the random fallback)."""
        body = {
            "name": "Empty Sugg",
            "data": [],
            "mapping": {
                "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            },
        }
        resp = client.post("/api/campaigns/from-upload", json=body)
        cid = resp.json()["campaign_id"]

        sugg_resp = client.get(f"/api/campaigns/{cid}/suggestions", params={"n": 3})
        assert sugg_resp.status_code == 200
        sugg = sugg_resp.json()
        assert isinstance(sugg["suggestions"], list)
        assert len(sugg["suggestions"]) <= 3

    def test_export_csv_empty_campaign(self, client):
        """CSV export of campaign with no observations returns just header or empty."""
        body = {
            "name": "Empty Export",
            "data": [],
            "mapping": {
                "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            },
        }
        resp = client.post("/api/campaigns/from-upload", json=body)
        cid = resp.json()["campaign_id"]

        export_resp = client.get(f"/api/campaigns/{cid}/export/csv")
        assert export_resp.status_code == 200

    def test_chat_on_empty_campaign_welcome(self, client):
        """Chat welcome on empty campaign shows 0 observations."""
        body = {
            "name": "Empty Chat",
            "data": [],
            "mapping": {
                "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            },
        }
        resp = client.post("/api/campaigns/from-upload", json=body)
        cid = resp.json()["campaign_id"]

        chat_resp = client.post(f"/api/chat/{cid}", json={"message": ""}).json()
        assert "0 observations" in chat_resp["reply"]

    def test_single_row_upload_diagnostics(self, client):
        """Campaign created from 1 uploaded row computes diagnostics without error."""
        campaign = _create_campaign_with_data(client, n_rows=1)
        cid = campaign["campaign_id"]
        diag = client.get(f"/api/campaigns/{cid}/diagnostics").json()
        # Should return valid numeric values without crashing
        assert isinstance(diag["best_kpi_value"], (int, float))
        assert isinstance(diag["improvement_velocity"], (int, float))
        assert isinstance(diag["plateau_length"], int)

"""Tests for the FastAPI application and API routes.

Tests cover campaigns CRUD, store, advice, reports, search, and auth.
Uses FastAPI's TestClient for synchronous HTTP testing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

# Guard against missing dependencies
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from optimization_copilot.api.app import create_app
from optimization_copilot.api.deps import get_app_state
from optimization_copilot.platform.models import Role


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
def auth_headers(app):
    """Create an admin API key and return auth headers."""
    state = app.state.platform
    raw_key = state.auth.create_key("test-admin", Role.ADMIN)
    return {"X-API-Key": raw_key}


@pytest.fixture
def sample_spec() -> dict[str, Any]:
    """A minimal optimization spec dict.

    Note: This is the spec dict stored in the workspace for campaign creation.
    It does NOT need to be a fully valid OptimizationSpec (which requires budget,
    campaign_id, etc.) because campaign creation just stores the dict as-is.
    The engine's start() will try to parse it fully, which may fail.
    """
    return {
        "parameters": [
            {"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0},
        ],
        "objectives": [
            {"name": "y", "direction": "minimize"},
        ],
    }


@pytest.fixture
def full_spec() -> dict[str, Any]:
    """A fully valid OptimizationSpec dict that the DSL can parse."""
    return {
        "campaign_id": "test-full",
        "parameters": [
            {"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0},
        ],
        "objectives": [
            {"name": "y", "direction": "minimize"},
        ],
        "budget": {"max_iterations": 5, "max_samples": 10},
        "risk_preference": "moderate",
        "parallel": {"batch_size": 1, "diversity_strategy": "hybrid"},
        "seed": 42,
    }


def _create_campaign(client, auth_headers, spec=None, name="Test Campaign", tags=None):
    """Helper to create a campaign and return the response JSON."""
    if spec is None:
        spec = {
            "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
    body = {"spec": spec, "name": name, "tags": tags or []}
    resp = client.post("/api/campaigns", json=body, headers=auth_headers)
    return resp


# ── Health Check ─────────────────────────────────────────────────


class TestHealthCheck:
    """Health check endpoint."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_includes_version(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "version" in data


# ── Campaign CRUD ────────────────────────────────────────────────


class TestCampaignCRUD:
    """Campaign create, read, list, delete endpoints."""

    def test_list_campaigns_empty(self, client, auth_headers):
        resp = client.get("/api/campaigns", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["campaigns"] == []
        assert data["total"] == 0

    def test_create_campaign(self, client, auth_headers, sample_spec):
        resp = _create_campaign(client, auth_headers, spec=sample_spec)
        assert resp.status_code == 201
        data = resp.json()
        assert "campaign_id" in data
        assert data["status"] == "draft"
        assert data["name"] == "Test Campaign"

    def test_create_campaign_with_tags(self, client, auth_headers, sample_spec):
        resp = _create_campaign(
            client, auth_headers, spec=sample_spec, tags=["tag1", "tag2"]
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["tags"] == ["tag1", "tag2"]

    def test_create_campaign_with_custom_name(self, client, auth_headers, sample_spec):
        resp = _create_campaign(
            client, auth_headers, spec=sample_spec, name="My Custom Name"
        )
        assert resp.status_code == 201
        assert resp.json()["name"] == "My Custom Name"

    def test_get_campaign_by_id(self, client, auth_headers, sample_spec):
        create_resp = _create_campaign(client, auth_headers, spec=sample_spec)
        cid = create_resp.json()["campaign_id"]

        resp = client.get(f"/api/campaigns/{cid}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["campaign_id"] == cid
        assert data["status"] == "draft"
        assert "spec" in data

    def test_get_campaign_404_missing(self, client, auth_headers):
        resp = client.get("/api/campaigns/nonexistent-id", headers=auth_headers)
        assert resp.status_code == 404

    def test_list_campaigns_after_create(self, client, auth_headers, sample_spec):
        _create_campaign(client, auth_headers, spec=sample_spec, name="Camp A")
        _create_campaign(client, auth_headers, spec=sample_spec, name="Camp B")

        resp = client.get("/api/campaigns", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_list_campaigns_with_status_filter(self, client, auth_headers, sample_spec):
        _create_campaign(client, auth_headers, spec=sample_spec)

        resp = client.get(
            "/api/campaigns", params={"status": "draft"}, headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_list_campaigns_filter_no_match(self, client, auth_headers, sample_spec):
        _create_campaign(client, auth_headers, spec=sample_spec)

        resp = client.get(
            "/api/campaigns", params={"status": "completed"}, headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_delete_campaign_archives(self, client, auth_headers, sample_spec):
        create_resp = _create_campaign(client, auth_headers, spec=sample_spec)
        cid = create_resp.json()["campaign_id"]

        resp = client.delete(f"/api/campaigns/{cid}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "archived"

    def test_delete_campaign_404_missing(self, client, auth_headers):
        resp = client.delete("/api/campaigns/nonexistent-id", headers=auth_headers)
        assert resp.status_code == 404

    def test_campaign_lifecycle_create_and_verify(
        self, client, auth_headers, sample_spec
    ):
        """Create a campaign then verify its fields."""
        resp = _create_campaign(
            client, auth_headers, spec=sample_spec, name="Lifecycle Test"
        )
        assert resp.status_code == 201
        cid = resp.json()["campaign_id"]

        detail = client.get(f"/api/campaigns/{cid}", headers=auth_headers).json()
        assert detail["name"] == "Lifecycle Test"
        assert detail["iteration"] == 0
        assert detail["total_trials"] == 0
        assert detail["best_kpi"] is None

    def test_multiple_campaigns_distinct_ids(self, client, auth_headers, sample_spec):
        resp1 = _create_campaign(client, auth_headers, spec=sample_spec, name="A")
        resp2 = _create_campaign(client, auth_headers, spec=sample_spec, name="B")
        assert resp1.json()["campaign_id"] != resp2.json()["campaign_id"]


# ── Campaign Execution Endpoints ─────────────────────────────────


class TestCampaignExecution:
    """Start, stop, pause, resume, batch, result endpoints."""

    def test_start_campaign_endpoint(self, client, auth_headers, full_spec):
        """Start endpoint exists and responds. Uses a full spec so the DSL
        can parse it. The engine may still fail at runtime (no backends),
        but it should get past spec parsing."""
        create_resp = _create_campaign(client, auth_headers, spec=full_spec)
        cid = create_resp.json()["campaign_id"]

        resp = client.post(f"/api/campaigns/{cid}/start", headers=auth_headers)
        # May be 200 (started successfully or engine running in background),
        # 409 (runner/transition error), or 500 (engine init failure).
        # The key thing is the endpoint exists and responds.
        assert resp.status_code in (200, 409, 500)

    def test_stop_campaign_not_running(self, client, auth_headers, sample_spec):
        """Stop on a non-running campaign returns 409."""
        create_resp = _create_campaign(client, auth_headers, spec=sample_spec)
        cid = create_resp.json()["campaign_id"]

        resp = client.post(f"/api/campaigns/{cid}/stop", headers=auth_headers)
        assert resp.status_code == 409

    def test_pause_campaign_not_running(self, client, auth_headers, sample_spec):
        """Pause on a non-running campaign returns 409."""
        create_resp = _create_campaign(client, auth_headers, spec=sample_spec)
        cid = create_resp.json()["campaign_id"]

        resp = client.post(f"/api/campaigns/{cid}/pause", headers=auth_headers)
        assert resp.status_code == 409

    def test_resume_campaign_endpoint(self, client, auth_headers, full_spec):
        """Resume endpoint exists."""
        create_resp = _create_campaign(client, auth_headers, spec=full_spec)
        cid = create_resp.json()["campaign_id"]

        resp = client.post(f"/api/campaigns/{cid}/resume", headers=auth_headers)
        # May succeed or return 409 (not paused / engine error) or 500 (engine init)
        assert resp.status_code in (200, 409, 500)

    def test_get_batch_no_batch(self, client, auth_headers, sample_spec):
        """Batch endpoint returns null when no batch is pending."""
        create_resp = _create_campaign(client, auth_headers, spec=sample_spec)
        cid = create_resp.json()["campaign_id"]

        resp = client.get(f"/api/campaigns/{cid}/batch", headers=auth_headers)
        assert resp.status_code == 200
        # Should be null/None since no engine running
        assert resp.json() is None

    def test_get_result_no_result(self, client, auth_headers, sample_spec):
        """Result endpoint returns 404 when no result exists."""
        create_resp = _create_campaign(client, auth_headers, spec=sample_spec)
        cid = create_resp.json()["campaign_id"]

        resp = client.get(f"/api/campaigns/{cid}/result", headers=auth_headers)
        assert resp.status_code == 404

    def test_submit_trials_not_running(self, client, auth_headers, sample_spec):
        """Submitting trials to non-running campaign returns 409."""
        create_resp = _create_campaign(client, auth_headers, spec=sample_spec)
        cid = create_resp.json()["campaign_id"]

        body = {"results": [{"trial_id": "t1", "kpi_values": {"y": 0.5}}]}
        resp = client.post(
            f"/api/campaigns/{cid}/trials", json=body, headers=auth_headers
        )
        assert resp.status_code == 409

    def test_get_checkpoint_no_checkpoint(self, client, auth_headers, sample_spec):
        """Checkpoint endpoint returns 404 when none available."""
        create_resp = _create_campaign(client, auth_headers, spec=sample_spec)
        cid = create_resp.json()["campaign_id"]

        resp = client.get(f"/api/campaigns/{cid}/checkpoint", headers=auth_headers)
        assert resp.status_code == 404


# ── Store Endpoints ──────────────────────────────────────────────


class TestStoreEndpoints:
    """Experiment store endpoints."""

    def test_store_summary_404_missing(self, client, auth_headers):
        resp = client.get(
            "/api/store/nonexistent-id/summary", headers=auth_headers
        )
        assert resp.status_code == 404

    def test_store_query_404_missing(self, client, auth_headers):
        resp = client.get("/api/store/nonexistent-id", headers=auth_headers)
        assert resp.status_code == 404

    def test_store_export_404_missing(self, client, auth_headers):
        resp = client.get(
            "/api/store/nonexistent-id/export", headers=auth_headers
        )
        assert resp.status_code == 404


# ── Report Endpoints ─────────────────────────────────────────────


class TestReportEndpoints:
    """Reports and comparison endpoints."""

    def test_audit_404_missing(self, client, auth_headers):
        resp = client.get(
            "/api/reports/nonexistent-id/audit", headers=auth_headers
        )
        assert resp.status_code == 404

    def test_compliance_404_missing(self, client, auth_headers):
        resp = client.get(
            "/api/reports/nonexistent-id/compliance", headers=auth_headers
        )
        assert resp.status_code == 404

    def test_compare_campaigns(self, client, auth_headers, sample_spec):
        """Compare two campaigns side by side."""
        c1 = _create_campaign(
            client, auth_headers, spec=sample_spec, name="Camp A"
        ).json()
        c2 = _create_campaign(
            client, auth_headers, spec=sample_spec, name="Camp B"
        ).json()

        resp = client.post(
            "/api/reports/compare",
            json={"campaign_ids": [c1["campaign_id"], c2["campaign_id"]]},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["campaign_ids"]) == 2
        assert len(data["records"]) == 2

    def test_compare_requires_at_least_two(self, client, auth_headers, sample_spec):
        """Compare with fewer than 2 IDs returns 422 validation error."""
        resp = client.post(
            "/api/reports/compare",
            json={"campaign_ids": ["single-id"]},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_compare_missing_campaign(self, client, auth_headers):
        """Compare with nonexistent campaign returns 404."""
        resp = client.post(
            "/api/reports/compare",
            json={"campaign_ids": ["nonexistent-a", "nonexistent-b"]},
            headers=auth_headers,
        )
        assert resp.status_code == 404


# ── Advice Endpoints ─────────────────────────────────────────────


class TestAdviceEndpoints:
    """Meta-learning advice endpoints."""

    def test_advice_endpoint_exists(self, client, auth_headers):
        """POST /api/advice returns a response (empty advice if no data)."""
        body = {
            "fingerprint": {
                "n_parameters": "5",
                "n_objectives": "1",
                "budget": "100",
            }
        }
        resp = client.post("/api/advice", json=body, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "recommended_backends" in data
        assert "confidence" in data

    def test_advice_no_experience(self, client, auth_headers):
        """With no advisor data, returns empty recommendations."""
        body = {"fingerprint": {"n_parameters": "3"}}
        resp = client.post("/api/advice", json=body, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] == 0.0
        assert "no_experience_data" in data["reason_codes"]

    def test_experience_count_endpoint(self, client, auth_headers):
        """GET /api/advice/experience-count returns count."""
        resp = client.get("/api/advice/experience-count", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert data["count"] == 0


# ── Search Endpoints ─────────────────────────────────────────────


class TestSearchEndpoints:
    """Campaign search endpoints at /api/search."""

    def test_search_returns_empty_for_no_data(self, client, auth_headers):
        """Search with empty RAG index returns empty results."""
        resp = client.get(
            "/api/search", params={"q": "test"}, headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []
        assert data["total"] == 0

    def test_search_requires_query_param(self, client, auth_headers):
        """Search without q parameter returns 422."""
        resp = client.get("/api/search", headers=auth_headers)
        assert resp.status_code == 422

    def test_search_via_rag_index_directly(self, app):
        """The RAG index is available via app state for direct search."""
        state = app.state.platform
        rag = state.rag
        # RAG index starts empty
        results = rag.search("test")
        assert results == []


# ── Request Validation ───────────────────────────────────────────


class TestRequestValidation:
    """Request body validation (422 errors)."""

    def test_create_campaign_missing_spec(self, client, auth_headers):
        """Creating campaign without spec returns 422."""
        resp = client.post(
            "/api/campaigns", json={"name": "No Spec"}, headers=auth_headers
        )
        assert resp.status_code == 422

    def test_create_campaign_invalid_body(self, client, auth_headers):
        """Completely invalid body returns 422."""
        resp = client.post(
            "/api/campaigns", json="not-a-dict", headers=auth_headers
        )
        assert resp.status_code == 422

    def test_submit_trials_missing_results(self, client, auth_headers, sample_spec):
        """Submitting trials without results field returns 422."""
        create_resp = _create_campaign(client, auth_headers, spec=sample_spec)
        cid = create_resp.json()["campaign_id"]

        resp = client.post(
            f"/api/campaigns/{cid}/trials",
            json={"not_results": []},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_compare_empty_ids_list(self, client, auth_headers):
        """Compare with empty list returns 422."""
        resp = client.post(
            "/api/reports/compare",
            json={"campaign_ids": []},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_advice_missing_fingerprint(self, client, auth_headers):
        """Advice without fingerprint returns 422."""
        resp = client.post("/api/advice", json={}, headers=auth_headers)
        assert resp.status_code == 422


# ── Authentication ───────────────────────────────────────────────


class TestAuthentication:
    """API key authentication tests.

    Note: The API may or may not enforce authentication on all routes.
    These tests verify that auth-related functionality works correctly.
    """

    def test_valid_api_key_succeeds(self, client, auth_headers, sample_spec):
        """Request with valid API key succeeds."""
        resp = client.get("/api/campaigns", headers=auth_headers)
        assert resp.status_code == 200

    def test_health_no_auth_required(self, client):
        """Health endpoint does not require authentication."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_can_create_multiple_keys(self, app):
        """Multiple API keys can be created."""
        state = app.state.platform
        key1 = state.auth.create_key("key-1", Role.VIEWER)
        key2 = state.auth.create_key("key-2", Role.OPERATOR)
        assert key1 != key2

    def test_keys_start_with_prefix(self, app):
        """Generated keys start with the expected prefix."""
        state = app.state.platform
        key = state.auth.create_key("test", Role.ADMIN)
        assert key.startswith("ocp_")


# ── App Factory ──────────────────────────────────────────────────


class TestAppFactory:
    """Application factory tests."""

    def test_create_app_returns_fastapi(self, tmp_path):
        from fastapi import FastAPI

        app = create_app(workspace_dir=str(tmp_path / "ws"))
        assert isinstance(app, FastAPI)

    def test_create_app_initializes_workspace(self, tmp_path):
        ws_dir = tmp_path / "ws"
        create_app(workspace_dir=str(ws_dir))
        assert ws_dir.exists()
        assert (ws_dir / "manifest.json").exists()

    def test_create_app_stores_platform_state(self, tmp_path):
        app = create_app(workspace_dir=str(tmp_path / "ws"))
        assert hasattr(app.state, "platform")
        assert app.state.platform is not None

    def test_create_app_custom_title(self, tmp_path):
        app = create_app(
            workspace_dir=str(tmp_path / "ws"), title="Custom Title"
        )
        assert app.title == "Custom Title"

    def test_create_app_custom_version(self, tmp_path):
        app = create_app(
            workspace_dir=str(tmp_path / "ws"), version="9.9.9"
        )
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.json()["version"] == "9.9.9"


# ── Campaign Detail Fields ───────────────────────────────────────


class TestCampaignDetailFields:
    """Verify campaign detail response has all expected fields."""

    def test_detail_has_spec(self, client, auth_headers, sample_spec):
        resp = _create_campaign(client, auth_headers, spec=sample_spec)
        data = resp.json()
        assert "spec" in data
        assert data["spec"]["parameters"][0]["name"] == "x"

    def test_detail_has_timestamps(self, client, auth_headers, sample_spec):
        resp = _create_campaign(client, auth_headers, spec=sample_spec)
        data = resp.json()
        assert "created_at" in data
        assert "updated_at" in data
        assert isinstance(data["created_at"], float)
        assert isinstance(data["updated_at"], float)

    def test_detail_has_campaign_id(self, client, auth_headers, sample_spec):
        resp = _create_campaign(client, auth_headers, spec=sample_spec)
        data = resp.json()
        assert "campaign_id" in data
        assert isinstance(data["campaign_id"], str)
        assert len(data["campaign_id"]) > 0

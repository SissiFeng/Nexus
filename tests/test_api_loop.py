"""Tests for the CampaignLoop API endpoints.

Tests the full loop lifecycle: create → iterate → ingest → status → delete.
Uses FastAPI's TestClient for synchronous HTTP testing.
"""

from __future__ import annotations

import pytest

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
def sample_loop_request():
    """A minimal loop creation request with synthetic data."""
    return {
        "campaign_id": "test-loop",
        "observations": [
            {
                "parameters": {"smiles": "CCO", "temp": 100},
                "kpi_values": {"yield": 0.8},
                "iteration": 0,
            },
            {
                "parameters": {"smiles": "CCCO", "temp": 120},
                "kpi_values": {"yield": 0.6},
                "iteration": 0,
            },
            {
                "parameters": {"smiles": "CCCCO", "temp": 90},
                "kpi_values": {"yield": 0.9},
                "iteration": 0,
            },
            {
                "parameters": {"smiles": "CCCCCO", "temp": 110},
                "kpi_values": {"yield": 0.7},
                "iteration": 0,
            },
            {
                "parameters": {"smiles": "CC(=O)O", "temp": 105},
                "kpi_values": {"yield": 0.85},
                "iteration": 0,
            },
        ],
        "candidates": [
            {"smiles": "C1CCCCC1", "temp": 100, "name": "Candidate-1"},
            {"smiles": "c1ccccc1", "temp": 110, "name": "Candidate-2"},
            {"smiles": "CC(C)O", "temp": 95, "name": "Candidate-3"},
            {"smiles": "CCOC", "temp": 115, "name": "Candidate-4"},
            {"smiles": "CC=CC", "temp": 100, "name": "Candidate-5"},
            {"smiles": "CCOCC", "temp": 90, "name": "Candidate-6"},
            {"smiles": "CCC(=O)C", "temp": 105, "name": "Candidate-7"},
            {"smiles": "CCCCCC", "temp": 100, "name": "Candidate-8"},
            {"smiles": "CC(O)CC", "temp": 110, "name": "Candidate-9"},
            {"smiles": "CCCOC", "temp": 95, "name": "Candidate-10"},
        ],
        "parameter_specs": [
            {"name": "smiles", "type": "categorical"},
            {"name": "temp", "type": "continuous", "lower": 80.0, "upper": 130.0},
        ],
        "objectives": ["yield"],
        "objective_directions": {"yield": "maximize"},
        "smiles_param": "smiles",
        "batch_size": 3,
        "acquisition_strategy": "ucb",
    }


# ── Test: Create Loop ─────────────────────────────────────────────


class TestCreateLoop:
    def test_create_success(self, client, sample_loop_request):
        resp = client.post("/api/loop", json=sample_loop_request)
        assert resp.status_code == 201
        data = resp.json()
        assert "loop_id" in data
        assert data["n_observations"] == 5
        assert data["n_candidates"] == 10

    def test_create_invalid_param_type(self, client, sample_loop_request):
        sample_loop_request["parameter_specs"][0]["type"] = "invalid_type"
        resp = client.post("/api/loop", json=sample_loop_request)
        assert resp.status_code == 400

    def test_create_missing_observations(self, client, sample_loop_request):
        del sample_loop_request["observations"]
        resp = client.post("/api/loop", json=sample_loop_request)
        assert resp.status_code == 422  # Pydantic validation


# ── Test: Iterate Loop ────────────────────────────────────────────


class TestIterateLoop:
    def test_iterate_success(self, client, sample_loop_request):
        # Create loop
        create_resp = client.post("/api/loop", json=sample_loop_request)
        loop_id = create_resp.json()["loop_id"]

        # Iterate
        resp = client.post(f"/api/loop/{loop_id}/iterate")
        assert resp.status_code == 200
        data = resp.json()

        # Check dashboard layer
        assert "dashboard" in data
        db = data["dashboard"]
        assert "ranked_candidates" in db
        assert "batch" in db
        assert len(db["batch"]) <= 3  # batch_size=3
        assert len(db["ranked_candidates"]) == 10  # all candidates ranked

        # Check each candidate has required fields
        for c in db["ranked_candidates"]:
            assert "rank" in c
            assert "name" in c
            assert "acquisition_score" in c
            assert "predicted_mean" in c
            assert "predicted_std" in c
            assert "parameters" in c

        # Check intelligence layer
        assert "intelligence" in data
        intel = data["intelligence"]
        assert "model_metrics" in intel
        assert len(intel["model_metrics"]) >= 1
        mm = intel["model_metrics"][0]
        assert mm["n_training_points"] == 5
        assert isinstance(mm["y_mean"], float)
        assert isinstance(mm["fit_duration_ms"], float)

        # Check reasoning layer
        assert "reasoning" in data

    def test_iterate_not_found(self, client):
        resp = client.post("/api/loop/nonexistent-id/iterate")
        assert resp.status_code == 404

    def test_iterate_twice(self, client, sample_loop_request):
        """Two iterations should work (idempotent model re-fit)."""
        create_resp = client.post("/api/loop", json=sample_loop_request)
        loop_id = create_resp.json()["loop_id"]

        resp1 = client.post(f"/api/loop/{loop_id}/iterate")
        assert resp1.status_code == 200

        resp2 = client.post(f"/api/loop/{loop_id}/iterate")
        assert resp2.status_code == 200
        assert resp2.json()["iteration"] >= resp1.json()["iteration"]


# ── Test: Ingest Results ──────────────────────────────────────────


class TestIngestResults:
    def test_ingest_success(self, client, sample_loop_request):
        # Create and iterate first
        create_resp = client.post("/api/loop", json=sample_loop_request)
        loop_id = create_resp.json()["loop_id"]
        client.post(f"/api/loop/{loop_id}/iterate")

        # Ingest new results
        ingest_req = {
            "results": [
                {
                    "parameters": {"smiles": "C1CCCCC1", "temp": 100},
                    "kpi_values": {"yield": 0.92},
                    "iteration": 1,
                },
                {
                    "parameters": {"smiles": "c1ccccc1", "temp": 110},
                    "kpi_values": {"yield": 0.78},
                    "iteration": 1,
                },
            ]
        }
        resp = client.post(f"/api/loop/{loop_id}/ingest", json=ingest_req)
        assert resp.status_code == 200
        data = resp.json()

        # Should have learning report
        intel = data["intelligence"]
        assert "learning_report" in intel
        lr = intel["learning_report"]
        assert lr["model_updated"] is True
        assert isinstance(lr["mean_absolute_error"], float)

    def test_ingest_empty_results(self, client, sample_loop_request):
        create_resp = client.post("/api/loop", json=sample_loop_request)
        loop_id = create_resp.json()["loop_id"]
        client.post(f"/api/loop/{loop_id}/iterate")

        resp = client.post(f"/api/loop/{loop_id}/ingest", json={"results": []})
        assert resp.status_code == 400

    def test_ingest_not_found(self, client):
        resp = client.post(
            "/api/loop/fake-id/ingest",
            json={
                "results": [
                    {
                        "parameters": {},
                        "kpi_values": {"y": 1},
                        "iteration": 0,
                    }
                ]
            },
        )
        assert resp.status_code == 404


# ── Test: Get Status ──────────────────────────────────────────────


class TestGetLoopStatus:
    def test_status_success(self, client, sample_loop_request):
        create_resp = client.post("/api/loop", json=sample_loop_request)
        loop_id = create_resp.json()["loop_id"]

        resp = client.get(f"/api/loop/{loop_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["loop_id"] == loop_id
        assert data["n_observations"] == 5
        assert data["n_candidates"] == 10
        assert data["objectives"] == ["yield"]
        assert data["iterations_run"] == 0

    def test_status_after_iterate(self, client, sample_loop_request):
        create_resp = client.post("/api/loop", json=sample_loop_request)
        loop_id = create_resp.json()["loop_id"]
        client.post(f"/api/loop/{loop_id}/iterate")

        resp = client.get(f"/api/loop/{loop_id}")
        data = resp.json()
        assert data["iterations_run"] == 1

    def test_status_not_found(self, client):
        resp = client.get("/api/loop/fake-id")
        assert resp.status_code == 404


# ── Test: Delete Loop ─────────────────────────────────────────────


class TestDeleteLoop:
    def test_delete_success(self, client, sample_loop_request):
        create_resp = client.post("/api/loop", json=sample_loop_request)
        loop_id = create_resp.json()["loop_id"]

        resp = client.delete(f"/api/loop/{loop_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify it's gone
        resp2 = client.get(f"/api/loop/{loop_id}")
        assert resp2.status_code == 404

    def test_delete_not_found(self, client):
        resp = client.delete("/api/loop/fake-id")
        assert resp.status_code == 404


# ── Test: Full Lifecycle ──────────────────────────────────────────


class TestLoopFullLifecycle:
    def test_create_iterate_ingest_iterate_delete(self, client, sample_loop_request):
        """Full lifecycle: create → iterate → ingest → iterate → status → delete."""
        # 1. Create
        resp = client.post("/api/loop", json=sample_loop_request)
        assert resp.status_code == 201
        loop_id = resp.json()["loop_id"]

        # 2. First iteration
        resp = client.post(f"/api/loop/{loop_id}/iterate")
        assert resp.status_code == 200
        deliverable = resp.json()
        batch = deliverable["dashboard"]["batch"]
        assert len(batch) > 0

        # 3. Ingest results for top candidate
        top = batch[0]
        ingest_req = {
            "results": [
                {
                    "parameters": top["parameters"],
                    "kpi_values": {"yield": 0.95},
                    "iteration": 1,
                }
            ]
        }
        resp = client.post(f"/api/loop/{loop_id}/ingest", json=ingest_req)
        assert resp.status_code == 200
        assert resp.json()["intelligence"]["learning_report"] is not None

        # 4. Second iteration (model updated with new data)
        resp = client.post(f"/api/loop/{loop_id}/iterate")
        assert resp.status_code == 200

        # 5. Check status
        resp = client.get(f"/api/loop/{loop_id}")
        assert resp.status_code == 200
        status = resp.json()
        assert status["iterations_run"] >= 2

        # 6. Delete
        resp = client.delete(f"/api/loop/{loop_id}")
        assert resp.status_code == 200

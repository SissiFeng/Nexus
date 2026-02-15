"""Tests for the DataAnalysisPipeline API endpoints.

Tests all 9 analysis endpoints with realistic data, verifying traced results
are returned with correct structure and execution metadata.
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


def _assert_traced_response(data: dict, expected_computed: bool = True):
    """Shared assertion helper for traced result responses."""
    assert "value" in data
    assert "tag" in data
    assert "traces" in data
    assert "is_computed" in data
    if expected_computed:
        assert data["tag"] == "computed"
        assert data["is_computed"] is True
    assert isinstance(data["traces"], list)
    if data["traces"]:
        trace = data["traces"][0]
        assert "module" in trace
        assert "method" in trace
        assert "duration_ms" in trace


# ── Test: Top-K ──────────────────────────────────────────────────


class TestTopK:
    def test_top_k_basic(self, client):
        resp = client.post("/api/analysis/top-k", json={
            "values": [0.8, 0.3, 0.95, 0.6, 0.72],
            "names": ["A", "B", "C", "D", "E"],
            "k": 3,
            "descending": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        _assert_traced_response(data)
        # Top 3 descending: C(0.95), A(0.8), E(0.72)
        # value is a list of {"name", "value", "rank"} dicts
        top_k = data["value"]
        assert len(top_k) == 3
        assert top_k[0]["name"] == "C"

    def test_top_k_ascending(self, client):
        resp = client.post("/api/analysis/top-k", json={
            "values": [0.8, 0.3, 0.95],
            "names": ["A", "B", "C"],
            "k": 2,
            "descending": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        _assert_traced_response(data)
        top_k = data["value"]
        assert top_k[0]["name"] == "B"  # Smallest


# ── Test: Ranking ────────────────────────────────────────────────


class TestRanking:
    def test_ranking_basic(self, client):
        resp = client.post("/api/analysis/ranking", json={
            "values": [0.5, 0.9, 0.1, 0.7],
            "names": ["W", "X", "Y", "Z"],
        })
        assert resp.status_code == 200
        data = resp.json()
        _assert_traced_response(data)
        # value is a list of {"name", "value", "rank"} dicts
        ranking = data["value"]
        assert len(ranking) == 4
        assert ranking[0]["name"] == "X"  # Highest


# ── Test: Outliers ───────────────────────────────────────────────


class TestOutliers:
    def test_outlier_detection(self, client):
        # Include one extreme value
        values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 5.0]
        names = ["A", "B", "C", "D", "E", "F", "Outlier"]
        resp = client.post("/api/analysis/outliers", json={
            "values": values,
            "names": names,
            "n_sigma": 2.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        _assert_traced_response(data)
        report = data["value"]
        assert "outliers" in report
        assert len(report["outliers"]) >= 1


# ── Test: Correlation ────────────────────────────────────────────


class TestCorrelation:
    def test_positive_correlation(self, client):
        resp = client.post("/api/analysis/correlation", json={
            "xs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "ys": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        })
        assert resp.status_code == 200
        data = resp.json()
        _assert_traced_response(data)
        # value is dict with "r", "n", "mean_x", "mean_y"
        r = data["value"]["r"]
        assert abs(r - 1.0) < 0.01  # Perfect positive

    def test_no_correlation(self, client):
        resp = client.post("/api/analysis/correlation", json={
            "xs": [1, 2, 3, 4, 5],
            "ys": [5, 1, 4, 2, 3],
        })
        assert resp.status_code == 200
        data = resp.json()
        _assert_traced_response(data)


# ── Test: fANOVA ─────────────────────────────────────────────────


class TestFanova:
    def test_fanova_basic(self, client):
        # 20 samples, 3 features
        import random
        random.seed(42)
        X = [[random.random() for _ in range(3)] for _ in range(20)]
        y = [sum(row) + random.gauss(0, 0.1) for row in X]

        resp = client.post("/api/analysis/fanova", json={
            "X": X,
            "y": y,
            "var_names": ["x1", "x2", "x3"],
            "n_trees": 20,
            "seed": 42,
        })
        assert resp.status_code == 200
        data = resp.json()
        _assert_traced_response(data)
        assert "main_effects" in data["value"]

    def test_fanova_too_few_samples(self, client):
        resp = client.post("/api/analysis/fanova", json={
            "X": [[1.0], [2.0]],
            "y": [1.0, 2.0],
            "n_trees": 5,
        })
        assert resp.status_code == 200
        # Should still return (possibly with FAILED tag if too few samples)
        data = resp.json()
        assert "tag" in data


# ── Test: Symbolic Regression ────────────────────────────────────


class TestSymreg:
    def test_symreg_basic(self, client):
        # Simple linear relationship y = 2*x1 + 3*x2
        import random
        random.seed(42)
        X = [[random.random(), random.random()] for _ in range(30)]
        y = [2.0 * r[0] + 3.0 * r[1] + random.gauss(0, 0.01) for r in X]

        resp = client.post("/api/analysis/symreg", json={
            "X": X,
            "y": y,
            "var_names": ["x1", "x2"],
            "pop_size": 50,
            "n_gen": 10,
            "seed": 42,
        })
        assert resp.status_code == 200
        data = resp.json()
        _assert_traced_response(data)
        assert "best_equation" in data["value"]


# ── Test: Pareto ─────────────────────────────────────────────────


class TestPareto:
    @pytest.fixture
    def pareto_request(self):
        return {
            "observations": [
                {"parameters": {"x": 1}, "kpi_values": {"f1": 0.1, "f2": 0.9}, "iteration": 0},
                {"parameters": {"x": 2}, "kpi_values": {"f1": 0.5, "f2": 0.5}, "iteration": 0},
                {"parameters": {"x": 3}, "kpi_values": {"f1": 0.9, "f2": 0.1}, "iteration": 0},
                {"parameters": {"x": 4}, "kpi_values": {"f1": 0.3, "f2": 0.3}, "iteration": 0},
                {"parameters": {"x": 5}, "kpi_values": {"f1": 0.7, "f2": 0.7}, "iteration": 0},
            ],
            "parameter_specs": [
                {"name": "x", "type": "continuous", "lower": 0.0, "upper": 10.0},
            ],
            "objectives": ["f1", "f2"],
            "objective_directions": ["minimize", "minimize"],
            "campaign_id": "pareto-test",
        }

    def test_pareto_basic(self, client, pareto_request):
        resp = client.post("/api/analysis/pareto", json=pareto_request)
        assert resp.status_code == 200
        data = resp.json()
        _assert_traced_response(data)
        assert "pareto_front" in data["value"] or "n_pareto" in data["value"]


# ── Test: Diagnostics ────────────────────────────────────────────


class TestDiagnostics:
    def test_diagnostics_basic(self, client):
        resp = client.post("/api/analysis/diagnostics", json={
            "observations": [
                {"parameters": {"x": float(i)}, "kpi_values": {"y": float(i) * 0.5}, "iteration": i}
                for i in range(10)
            ],
            "parameter_specs": [
                {"name": "x", "type": "continuous", "lower": 0.0, "upper": 10.0},
            ],
            "objectives": ["y"],
            "objective_directions": ["minimize"],
        })
        assert resp.status_code == 200
        data = resp.json()
        _assert_traced_response(data)


# ── Test: Molecular Pipeline ─────────────────────────────────────


class TestMolecularPipeline:
    def test_molecular_basic(self, client):
        resp = client.post("/api/analysis/molecular", json={
            "smiles_list": ["CCO", "CCCO", "CCCCO", "CC(=O)O", "CC(C)O"],
            "observations": [
                {"parameters": {"smiles": "CCO"}, "kpi_values": {"yield": 0.8}, "iteration": 0},
                {"parameters": {"smiles": "CCCO"}, "kpi_values": {"yield": 0.6}, "iteration": 0},
                {"parameters": {"smiles": "CCCCO"}, "kpi_values": {"yield": 0.9}, "iteration": 0},
                {"parameters": {"smiles": "CC(=O)O"}, "kpi_values": {"yield": 0.7}, "iteration": 0},
                {"parameters": {"smiles": "CC(C)O"}, "kpi_values": {"yield": 0.85}, "iteration": 0},
            ],
            "parameter_specs": [
                {"name": "smiles", "type": "categorical"},
            ],
            "objective_name": "yield",
            "n_suggestions": 3,
            "seed": 42,
        })
        assert resp.status_code == 200
        data = resp.json()
        # Molecular pipeline may fail if SMILES can't be encoded with
        # basic n-gram approach (no rdkit), so accept both computed and failed
        _assert_traced_response(data, expected_computed=False)
        assert data["tag"] in ("computed", "failed")


# ── Test: Error Handling ─────────────────────────────────────────


class TestAnalysisErrors:
    def test_empty_values(self, client):
        resp = client.post("/api/analysis/top-k", json={
            "values": [],
            "names": [],
            "k": 3,
        })
        # Should return 200 with empty result or 400
        assert resp.status_code in (200, 400)

    def test_mismatched_lengths(self, client):
        resp = client.post("/api/analysis/correlation", json={
            "xs": [1, 2, 3],
            "ys": [1, 2],
        })
        # Pipeline may handle this gracefully or return 400
        assert resp.status_code in (200, 400)

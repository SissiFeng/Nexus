"""Platform integration tests: campaign lifecycle, workspace+RAG, auth+API,
multi-campaign concurrency, WebSocket routes, and checkpoint/resume flows.

These tests exercise the platform layer (CampaignManager, Workspace, RAGIndex,
AsyncEventBus, AuthManager) and verify cross-component interactions through
realistic multi-step scenarios.
"""

from __future__ import annotations

import asyncio
import json
import random
from time import time
from typing import Any

import pytest

from optimization_copilot.platform.campaign_manager import (
    CampaignManager,
    InvalidTransitionError,
)
from optimization_copilot.platform.events import AsyncEventBus
from optimization_copilot.platform.models import (
    ApiKey,
    CampaignRecord,
    CampaignStatus,
    CompareReport,
    Role,
    SearchResult,
    VALID_TRANSITIONS,
)
from optimization_copilot.platform.rag import RAGIndex
from optimization_copilot.platform.workspace import (
    CampaignNotFoundError,
    Workspace,
)
from optimization_copilot.platform.auth import AuthManager


# ── Helpers ─────────────────────────────────────────────────────


def _make_spec(
    param_name: str = "x",
    obj_name: str = "y",
    direction: str = "minimize",
    description: str = "",
) -> dict[str, Any]:
    """Build a minimal spec dict for campaign creation."""
    spec: dict[str, Any] = {
        "parameters": [
            {"name": param_name, "type": "continuous", "lower": 0.0, "upper": 1.0},
        ],
        "objectives": [
            {"name": obj_name, "direction": direction},
        ],
    }
    if description:
        spec["description"] = description
    return spec


def _make_spec_multi_kpi(kpi_names: list[str]) -> dict[str, Any]:
    """Build a spec with multiple objectives."""
    return {
        "parameters": [
            {"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0},
        ],
        "objectives": [
            {"name": name, "direction": "minimize", "is_primary": i == 0}
            for i, name in enumerate(kpi_names)
        ],
    }


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    """Create and initialize a fresh workspace."""
    ws = Workspace(tmp_path / "workspace")
    ws.init()
    return ws


@pytest.fixture
def manager(workspace) -> CampaignManager:
    """Create a CampaignManager backed by the workspace."""
    return CampaignManager(workspace)


@pytest.fixture
def rag() -> RAGIndex:
    """Create an empty RAG index."""
    return RAGIndex()


@pytest.fixture
def event_bus() -> AsyncEventBus:
    """Create an event bus without an event loop."""
    return AsyncEventBus()


@pytest.fixture
def auth(workspace) -> AuthManager:
    """Create an AuthManager backed by the workspace."""
    return AuthManager(workspace)


# ====================================================================
# 1. Campaign Lifecycle Integration (~10 tests)
# ====================================================================


class TestCampaignLifecycle:
    """Test the full campaign lifecycle through the platform layer."""

    def test_create_running_progress_completed(self, manager: CampaignManager):
        """Create -> running -> update progress -> completed."""
        record = manager.create(spec_dict=_make_spec(), name="lifecycle-basic")
        assert record.status == CampaignStatus.DRAFT

        manager.mark_running(record.campaign_id)
        record = manager.get(record.campaign_id)
        assert record.status == CampaignStatus.RUNNING

        manager.update_progress(record.campaign_id, iteration=3, total_trials=9, best_kpi=0.42)
        record = manager.get(record.campaign_id)
        assert record.iteration == 3
        assert record.total_trials == 9
        assert record.best_kpi == 0.42

        manager.mark_completed(record.campaign_id, best_kpi=0.35)
        record = manager.get(record.campaign_id)
        assert record.status == CampaignStatus.COMPLETED
        assert record.best_kpi == 0.35

    def test_create_running_paused_resume_completed(self, manager: CampaignManager):
        """Create -> running -> paused -> running again -> completed."""
        record = manager.create(spec_dict=_make_spec(), name="pause-resume")
        manager.mark_running(record.campaign_id)
        manager.mark_paused(record.campaign_id)
        record = manager.get(record.campaign_id)
        assert record.status == CampaignStatus.PAUSED

        # Resume = mark running from paused
        manager.mark_running(record.campaign_id)
        record = manager.get(record.campaign_id)
        assert record.status == CampaignStatus.RUNNING

        manager.mark_completed(record.campaign_id)
        record = manager.get(record.campaign_id)
        assert record.status == CampaignStatus.COMPLETED

    def test_create_running_failed(self, manager: CampaignManager):
        """Create -> running -> failed with error message."""
        record = manager.create(spec_dict=_make_spec(), name="fail-test")
        manager.mark_running(record.campaign_id)
        manager.mark_failed(record.campaign_id, error="Engine exploded")
        record = manager.get(record.campaign_id)
        assert record.status == CampaignStatus.FAILED
        assert record.error_message == "Engine exploded"

    def test_create_delete_archived(self, manager: CampaignManager):
        """Create -> delete (archives) -> verify status."""
        record = manager.create(spec_dict=_make_spec(), name="delete-test")
        cid = record.campaign_id
        manager.delete(cid)
        record = manager.get(cid)
        assert record.status == CampaignStatus.ARCHIVED

    def test_list_with_status_filter(self, manager: CampaignManager):
        """Create 3 campaigns -> filter by status."""
        c1 = manager.create(spec_dict=_make_spec(), name="c1")
        c2 = manager.create(spec_dict=_make_spec(), name="c2")
        c3 = manager.create(spec_dict=_make_spec(), name="c3")

        manager.mark_running(c1.campaign_id)
        manager.mark_running(c2.campaign_id)
        manager.mark_completed(c2.campaign_id)

        drafts = manager.list_all(status=CampaignStatus.DRAFT)
        running = manager.list_all(status=CampaignStatus.RUNNING)
        completed = manager.list_all(status=CampaignStatus.COMPLETED)

        assert len(drafts) == 1
        assert drafts[0].campaign_id == c3.campaign_id
        assert len(running) == 1
        assert running[0].campaign_id == c1.campaign_id
        assert len(completed) == 1
        assert completed[0].campaign_id == c2.campaign_id

    def test_progress_accumulates(self, manager: CampaignManager):
        """Multiple progress updates accumulate correctly."""
        record = manager.create(spec_dict=_make_spec(), name="progress-test")
        manager.mark_running(record.campaign_id)

        manager.update_progress(record.campaign_id, iteration=1, total_trials=3, best_kpi=1.0)
        manager.update_progress(record.campaign_id, iteration=2, total_trials=6, best_kpi=0.8)
        manager.update_progress(record.campaign_id, iteration=3, total_trials=9, best_kpi=0.5)

        record = manager.get(record.campaign_id)
        assert record.iteration == 3
        assert record.total_trials == 9
        assert record.best_kpi == 0.5

    def test_compare_two_campaigns(self, manager: CampaignManager):
        """Compare 2 campaigns with different KPIs."""
        spec_a = _make_spec_multi_kpi(["loss"])
        spec_b = _make_spec_multi_kpi(["loss"])

        c1 = manager.create(spec_dict=spec_a, name="compare-a")
        c2 = manager.create(spec_dict=spec_b, name="compare-b")

        manager.mark_running(c1.campaign_id)
        manager.mark_completed(c1.campaign_id, best_kpi=0.3)
        manager.mark_running(c2.campaign_id)
        manager.mark_completed(c2.campaign_id, best_kpi=0.6)

        report = manager.compare([c1.campaign_id, c2.campaign_id])
        assert isinstance(report, CompareReport)
        assert len(report.campaign_ids) == 2
        assert len(report.records) == 2
        # c1 has lower best_kpi (0.3) so it should be winner
        assert report.winner == c1.campaign_id

    def test_invalid_transition_raises(self, manager: CampaignManager):
        """Invalid state transition raises InvalidTransitionError."""
        record = manager.create(spec_dict=_make_spec(), name="invalid-transition")
        # Draft -> Completed is not valid
        with pytest.raises(InvalidTransitionError):
            manager.mark_completed(record.campaign_id)

    def test_create_running_stopped(self, manager: CampaignManager):
        """Create -> running -> stopped."""
        record = manager.create(spec_dict=_make_spec(), name="stop-test")
        manager.mark_running(record.campaign_id)
        manager.mark_stopped(record.campaign_id)
        record = manager.get(record.campaign_id)
        assert record.status == CampaignStatus.STOPPED

    def test_update_tags(self, manager: CampaignManager):
        """Create campaign, update tags, verify persisted."""
        record = manager.create(spec_dict=_make_spec(), name="tag-test", tags=["initial"])
        assert record.tags == ["initial"]

        record = manager.update_tags(record.campaign_id, ["updated", "tags"])
        assert record.tags == ["updated", "tags"]

        reloaded = manager.get(record.campaign_id)
        assert reloaded.tags == ["updated", "tags"]


# ====================================================================
# 2. Workspace + RAG Integration (~8 tests)
# ====================================================================


class TestWorkspaceRAGIntegration:
    """Test RAG indexing with real workspace data."""

    def _add_background_doc(self, manager: CampaignManager, rag: RAGIndex) -> None:
        """Add a background document so TF-IDF has >1 doc for meaningful IDF scores."""
        bg = manager.create(
            spec_dict=_make_spec(description="background filler document for indexing"),
            name="background-doc",
        )
        rag.index_campaign(bg.campaign_id, bg, bg.spec)

    def test_create_index_search(self, manager: CampaignManager, rag: RAGIndex):
        """Create campaign -> index in RAG -> search finds it."""
        self._add_background_doc(manager, rag)
        record = manager.create(
            spec_dict=_make_spec(description="optimize neural network learning rate"),
            name="neural-lr",
        )
        spec = record.spec
        rag.index_campaign(record.campaign_id, record, spec)

        results = rag.search("neural network")
        assert len(results) >= 1
        assert any(r.campaign_id == record.campaign_id for r in results)

    def test_search_by_name_matches_correct(self, manager: CampaignManager, rag: RAGIndex):
        """Create 3 campaigns with different names -> search by name matches correct one."""
        names = ["alpha-model", "beta-optimizer", "gamma-search"]
        records = []
        for name in names:
            r = manager.create(spec_dict=_make_spec(), name=name)
            rag.index_campaign(r.campaign_id, r, r.spec)
            records.append(r)

        results = rag.search("alpha")
        assert len(results) >= 1
        assert results[0].campaign_id == records[0].campaign_id

        results = rag.search("gamma")
        assert len(results) >= 1
        assert results[0].campaign_id == records[2].campaign_id

    def test_delete_from_rag_not_searchable(
        self, manager: CampaignManager, rag: RAGIndex
    ):
        """Delete campaign from workspace -> remove from RAG -> search doesn't find it."""
        self._add_background_doc(manager, rag)
        record = manager.create(
            spec_dict=_make_spec(description="unique quantum experiment"),
            name="quantum-exp",
        )
        rag.index_campaign(record.campaign_id, record, record.spec)

        # Verify it's findable
        results = rag.search("quantum")
        assert len(results) >= 1

        # Remove and verify gone
        rag.remove_campaign(record.campaign_id)
        results = rag.search("quantum")
        assert not any(r.campaign_id == record.campaign_id for r in results)

    def test_to_dict_from_dict_roundtrip(self, manager: CampaignManager, rag: RAGIndex):
        """RAG to_dict/from_dict round-trip preserves search quality."""
        self._add_background_doc(manager, rag)
        record = manager.create(
            spec_dict=_make_spec(description="roundtrip test experiment"),
            name="roundtrip-exp",
        )
        rag.index_campaign(record.campaign_id, record, record.spec)

        # Serialize and deserialize
        data = rag.to_dict()
        rag2 = RAGIndex.from_dict(data)

        results_original = rag.search("roundtrip")
        results_restored = rag2.search("roundtrip")

        assert len(results_original) == len(results_restored)
        assert len(results_original) >= 1
        assert results_original[0].campaign_id == results_restored[0].campaign_id
        assert abs(results_original[0].score - results_restored[0].score) < 1e-6

    def test_rag_rebuild_from_workspace(
        self, manager: CampaignManager, workspace: Workspace, rag: RAGIndex
    ):
        """Rebuild RAG index from workspace data."""
        r1 = manager.create(
            spec_dict=_make_spec(description="rebuild alpha unique"),
            name="rebuild-a",
        )
        r2 = manager.create(
            spec_dict=_make_spec(description="rebuild beta unique"),
            name="rebuild-b",
        )
        # Add a third campaign without "rebuild" in name/desc to give IDF contrast
        r3 = manager.create(
            spec_dict=_make_spec(description="unrelated filler document"),
            name="filler-doc",
        )

        campaigns = [
            (r1.campaign_id, r1, r1.spec, None),
            (r2.campaign_id, r2, r2.spec, None),
            (r3.campaign_id, r3, r3.spec, None),
        ]
        rag.rebuild(campaigns)

        assert rag.document_count == 3
        results = rag.search("rebuild")
        assert len(results) >= 2
        result_ids = {r.campaign_id for r in results[:2]}
        assert r1.campaign_id in result_ids
        assert r2.campaign_id in result_ids

    def test_search_by_tags(self, manager: CampaignManager, rag: RAGIndex):
        """Search after indexing campaigns with various tags."""
        r1 = manager.create(spec_dict=_make_spec(), name="tagged-1", tags=["production", "ml"])
        r2 = manager.create(spec_dict=_make_spec(), name="tagged-2", tags=["staging", "nlp"])

        rag.index_campaign(r1.campaign_id, r1, r1.spec)
        rag.index_campaign(r2.campaign_id, r2, r2.spec)

        results = rag.search("production")
        assert len(results) >= 1
        assert results[0].campaign_id == r1.campaign_id

        results = rag.search("nlp")
        assert len(results) >= 1
        assert results[0].campaign_id == r2.campaign_id

    def test_empty_workspace_empty_search(self, rag: RAGIndex):
        """Empty workspace -> RAG search returns empty."""
        results = rag.search("anything")
        assert results == []

    def test_index_update_reindex_reflects_change(
        self, manager: CampaignManager, rag: RAGIndex
    ):
        """Index campaign, update it, re-index -> search reflects updated data."""
        self._add_background_doc(manager, rag)
        record = manager.create(
            spec_dict=_make_spec(description="original description"),
            name="updatable-exp",
        )
        rag.index_campaign(record.campaign_id, record, record.spec)

        # Update the campaign name and tags
        record = manager.update_tags(record.campaign_id, ["revised", "updated"])
        updated_record = manager.get(record.campaign_id)

        # Re-index with updated data
        new_spec = dict(record.spec)
        new_spec["description"] = "completely revised description about robotics"
        rag.index_campaign(updated_record.campaign_id, updated_record, new_spec)

        results = rag.search("robotics")
        assert len(results) >= 1
        assert results[0].campaign_id == updated_record.campaign_id


# ====================================================================
# 3. Auth + API Integration (~8 tests)
# ====================================================================


class TestAuthAPIIntegration:
    """Test auth with the API (uses TestClient from FastAPI)."""

    @pytest.fixture(autouse=True)
    def _check_fastapi(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

    @pytest.fixture
    def app(self, tmp_path):
        from optimization_copilot.api.app import create_app

        return create_app(workspace_dir=str(tmp_path / "workspace"))

    @pytest.fixture
    def client(self, app):
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_create_admin_key_access_api(self, app, client):
        """Create app -> create admin key -> use it to access API."""
        state = app.state.platform
        raw_key = state.auth.create_key("test-admin", Role.ADMIN)
        assert raw_key.startswith("ocp_")

        headers = {"X-API-Key": raw_key}
        resp = client.get("/api/campaigns", headers=headers)
        assert resp.status_code == 200

    def test_viewer_key_can_list_campaigns(self, app, client):
        """Create viewer key -> can list campaigns (API doesn't enforce roles on routes currently)."""
        state = app.state.platform
        raw_key = state.auth.create_key("test-viewer", Role.VIEWER)
        headers = {"X-API-Key": raw_key}

        resp = client.get("/api/campaigns", headers=headers)
        assert resp.status_code == 200

    def test_viewer_delete_campaign_note(self, app, client):
        """Create viewer key -> attempt delete -> verify behavior.

        The current API does not enforce RBAC on individual routes, so delete
        will succeed. We document this as expected behavior.
        """
        state = app.state.platform
        admin_key = state.auth.create_key("admin", Role.ADMIN)
        viewer_key = state.auth.create_key("viewer", Role.VIEWER)

        # Create a campaign with admin key
        body = {
            "spec": _make_spec(),
            "name": "viewer-delete-test",
            "tags": [],
        }
        resp = client.post(
            "/api/campaigns", json=body, headers={"X-API-Key": admin_key}
        )
        assert resp.status_code == 201
        cid = resp.json()["campaign_id"]

        # Delete with viewer key - should succeed since RBAC is not enforced on routes
        resp = client.delete(
            f"/api/campaigns/{cid}", headers={"X-API-Key": viewer_key}
        )
        # API does not currently enforce per-route roles, so this succeeds
        assert resp.status_code == 200

    def test_multiple_keys_authenticate_independently(self, app):
        """Multiple keys can authenticate independently."""
        state = app.state.platform
        key_a = state.auth.create_key("key-a", Role.ADMIN)
        key_b = state.auth.create_key("key-b", Role.OPERATOR)

        result_a = state.auth.authenticate(key_a)
        result_b = state.auth.authenticate(key_b)

        assert result_a is not None
        assert result_b is not None
        assert result_a.name == "key-a"
        assert result_b.name == "key-b"
        assert result_a.role == Role.ADMIN
        assert result_b.role == Role.OPERATOR

    def test_revoked_key_cannot_authenticate(self, app):
        """Revoked key cannot authenticate."""
        state = app.state.platform
        raw_key = state.auth.create_key("revocable", Role.ADMIN)

        # Authenticate first
        api_key = state.auth.authenticate(raw_key)
        assert api_key is not None

        # Revoke
        state.auth.revoke_key(api_key.key_hash)

        # Cannot authenticate anymore
        result = state.auth.authenticate(raw_key)
        assert result is None

    def test_key_list_shows_all_created(self, app):
        """Key list shows all created keys."""
        state = app.state.platform
        state.auth.create_key("key-1", Role.ADMIN)
        state.auth.create_key("key-2", Role.VIEWER)
        state.auth.create_key("key-3", Role.OPERATOR)

        keys = state.auth.list_keys()
        assert len(keys) == 3
        names = {k.name for k in keys}
        assert names == {"key-1", "key-2", "key-3"}

    def test_different_roles_authorization(self, app):
        """Different roles for different operations via authorize()."""
        state = app.state.platform
        viewer_key_raw = state.auth.create_key("viewer", Role.VIEWER)
        admin_key_raw = state.auth.create_key("admin", Role.ADMIN)

        viewer = state.auth.authenticate(viewer_key_raw)
        admin = state.auth.authenticate(admin_key_raw)

        assert viewer is not None
        assert admin is not None

        # Viewer can do viewer-level ops
        assert AuthManager.authorize(viewer, Role.VIEWER) is True
        # Viewer cannot do admin-level ops
        assert AuthManager.authorize(viewer, Role.ADMIN) is False
        # Admin can do everything
        assert AuthManager.authorize(admin, Role.VIEWER) is True
        assert AuthManager.authorize(admin, Role.ADMIN) is True

    def test_api_key_header_format(self, app):
        """API key header format starts with ocp_."""
        state = app.state.platform
        raw_key = state.auth.create_key("format-test", Role.ADMIN)
        assert raw_key.startswith("ocp_")
        # Key should be long enough to be secure
        assert len(raw_key) > 20


# ====================================================================
# 4. Multi-Campaign Concurrency (~5 tests)
# ====================================================================


class TestMultiCampaignConcurrency:
    """Test that multiple campaigns don't interfere with each other."""

    def test_update_a_does_not_affect_b(self, manager: CampaignManager):
        """Update campaign A progress -> B is unaffected."""
        a = manager.create(spec_dict=_make_spec(), name="camp-a")
        b = manager.create(spec_dict=_make_spec(), name="camp-b")

        manager.mark_running(a.campaign_id)
        manager.mark_running(b.campaign_id)

        manager.update_progress(a.campaign_id, iteration=5, total_trials=15, best_kpi=0.1)

        b_record = manager.get(b.campaign_id)
        assert b_record.iteration == 0
        assert b_record.total_trials == 0
        assert b_record.best_kpi is None

    def test_delete_a_b_still_accessible(self, manager: CampaignManager):
        """Delete A -> B still accessible."""
        a = manager.create(spec_dict=_make_spec(), name="delete-a")
        b = manager.create(spec_dict=_make_spec(), name="keep-b")

        manager.delete(a.campaign_id)

        b_record = manager.get(b.campaign_id)
        assert b_record.name == "keep-b"
        assert b_record.status == CampaignStatus.DRAFT

    def test_compare_a_and_b(self, manager: CampaignManager):
        """Compare A and B -> report includes both."""
        a = manager.create(spec_dict=_make_spec_multi_kpi(["metric"]), name="cmp-a")
        b = manager.create(spec_dict=_make_spec_multi_kpi(["metric"]), name="cmp-b")

        manager.mark_running(a.campaign_id)
        manager.mark_completed(a.campaign_id, best_kpi=0.2)
        manager.mark_running(b.campaign_id)
        manager.mark_completed(b.campaign_id, best_kpi=0.4)

        report = manager.compare([a.campaign_id, b.campaign_id])
        assert len(report.records) == 2
        assert a.campaign_id in report.campaign_ids
        assert b.campaign_id in report.campaign_ids

    def test_rag_search_returns_both(self, manager: CampaignManager, rag: RAGIndex):
        """RAG search returns both when indexed."""
        a = manager.create(
            spec_dict=_make_spec(description="concurrent experiment alpha"),
            name="concurrent-a",
        )
        b = manager.create(
            spec_dict=_make_spec(description="concurrent experiment beta"),
            name="concurrent-b",
        )
        # Add a background doc without "concurrent" to give IDF contrast
        bg = manager.create(
            spec_dict=_make_spec(description="unrelated filler document"),
            name="filler-doc",
        )

        rag.index_campaign(a.campaign_id, a, a.spec)
        rag.index_campaign(b.campaign_id, b, b.spec)
        rag.index_campaign(bg.campaign_id, bg, bg.spec)

        results = rag.search("concurrent experiment")
        assert len(results) >= 2
        result_ids = {r.campaign_id for r in results[:2]}
        assert a.campaign_id in result_ids
        assert b.campaign_id in result_ids

    def test_workspace_lists_both(self, manager: CampaignManager):
        """Workspace lists both campaigns."""
        a = manager.create(spec_dict=_make_spec(), name="ws-a")
        b = manager.create(spec_dict=_make_spec(), name="ws-b")

        all_campaigns = manager.list_all()
        assert len(all_campaigns) == 2
        ids = {c.campaign_id for c in all_campaigns}
        assert a.campaign_id in ids
        assert b.campaign_id in ids


# ====================================================================
# 5. WebSocket Route Existence (~5 tests)
# ====================================================================


class TestWebSocketRoutes:
    """Test WebSocket routes exist and event bus works end-to-end."""

    @pytest.fixture(autouse=True)
    def _check_fastapi(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

    @pytest.fixture
    def app(self, tmp_path):
        from optimization_copilot.api.app import create_app

        return create_app(workspace_dir=str(tmp_path / "workspace"))

    def test_ws_campaign_route_exists(self, app):
        """WS endpoint /api/ws/{campaign_id} exists in app routes."""
        routes = []
        for route in app.routes:
            if hasattr(route, "path"):
                routes.append(route.path)
            # Check sub-routers
            if hasattr(route, "routes"):
                for sub in route.routes:
                    if hasattr(sub, "path"):
                        routes.append(sub.path)
                    if hasattr(sub, "routes"):
                        for sub2 in sub.routes:
                            if hasattr(sub2, "path"):
                                routes.append(sub2.path)

        # The WS route is at /api/ws/{campaign_id}
        assert any("ws" in r and "{campaign_id}" in r for r in routes), (
            f"WS campaign route not found in: {routes}"
        )

    def test_ws_all_events_route_exists(self, app):
        """WS endpoint /api/ws exists in app routes."""
        routes = []
        for route in app.routes:
            if hasattr(route, "path"):
                routes.append(route.path)
            if hasattr(route, "routes"):
                for sub in route.routes:
                    if hasattr(sub, "path"):
                        routes.append(sub.path)
                    if hasattr(sub, "routes"):
                        for sub2 in sub.routes:
                            if hasattr(sub2, "path"):
                                routes.append(sub2.path)

        # Look for /api/ws (no trailing path segment)
        assert any(r.rstrip("/").endswith("/ws") for r in routes), (
            f"WS all-events route not found in: {routes}"
        )

    def test_event_bus_subscribe_publish_cycle(self):
        """Event bus subscribe/publish cycle works end-to-end (no API)."""
        loop = asyncio.new_event_loop()
        try:
            bus = AsyncEventBus(loop=None)

            async def _run():
                queue = bus.subscribe(campaign_id="test-campaign")
                bus.publish("test-campaign", "iteration_done", {"iteration": 1})

                msg = queue.get_nowait()
                assert msg["campaign_id"] == "test-campaign"
                assert msg["event"] == "iteration_done"
                assert msg["data"]["iteration"] == 1
                assert "timestamp" in msg

            loop.run_until_complete(_run())
        finally:
            loop.close()

    def test_event_bus_no_crosstalk(self):
        """Event bus handles multiple campaigns without cross-talk."""
        loop = asyncio.new_event_loop()
        try:
            bus = AsyncEventBus(loop=None)

            async def _run():
                q_a = bus.subscribe(campaign_id="campaign-a")
                q_b = bus.subscribe(campaign_id="campaign-b")

                bus.publish("campaign-a", "event-a", {"value": "a"})
                bus.publish("campaign-b", "event-b", {"value": "b"})

                msg_a = q_a.get_nowait()
                assert msg_a["data"]["value"] == "a"

                msg_b = q_b.get_nowait()
                assert msg_b["data"]["value"] == "b"

                # Verify no crosstalk: q_a should not have campaign-b events
                assert q_a.empty()
                assert q_b.empty()

            loop.run_until_complete(_run())
        finally:
            loop.close()

    def test_event_bus_format_sse(self):
        """Event bus format_sse produces valid SSE format."""
        bus = AsyncEventBus()
        message = {
            "campaign_id": "test",
            "event": "progress",
            "data": {"iteration": 5},
            "timestamp": 1234567890.0,
        }
        sse = bus.format_sse(message)
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")

        # Parse the JSON payload
        json_str = sse[len("data: "):].strip()
        parsed = json.loads(json_str)
        assert parsed["campaign_id"] == "test"
        assert parsed["event"] == "progress"


# ====================================================================
# 6. Checkpoint/Resume Flow (~5 tests)
# ====================================================================


class TestCheckpointResumeFlow:
    """Test checkpoint persistence through workspace."""

    def test_save_load_checkpoint_roundtrip(self, workspace: Workspace, manager: CampaignManager):
        """Save checkpoint dict -> load it back -> verify identical."""
        record = manager.create(spec_dict=_make_spec(), name="checkpoint-test")
        cid = record.campaign_id

        checkpoint = {
            "iteration": 5,
            "best_params": {"x": 0.42},
            "state": "exploration",
            "rng_state": [1, 2, 3, 4, 5],
        }
        workspace.save_checkpoint(cid, checkpoint)
        loaded = workspace.load_checkpoint(cid)

        assert loaded is not None
        assert loaded == checkpoint

    def test_save_twice_load_returns_latest(self, workspace: Workspace, manager: CampaignManager):
        """Save checkpoint -> save again with updated state -> load returns latest."""
        record = manager.create(spec_dict=_make_spec(), name="checkpoint-update")
        cid = record.campaign_id

        checkpoint_v1 = {"iteration": 3, "version": 1}
        checkpoint_v2 = {"iteration": 7, "version": 2}

        workspace.save_checkpoint(cid, checkpoint_v1)
        workspace.save_checkpoint(cid, checkpoint_v2)

        loaded = workspace.load_checkpoint(cid)
        assert loaded is not None
        assert loaded["version"] == 2
        assert loaded["iteration"] == 7

    def test_no_checkpoint_returns_none(self, workspace: Workspace, manager: CampaignManager):
        """No checkpoint -> load returns None."""
        record = manager.create(spec_dict=_make_spec(), name="no-checkpoint")
        loaded = workspace.load_checkpoint(record.campaign_id)
        assert loaded is None

    def test_checkpoint_and_result_independent(
        self, workspace: Workspace, manager: CampaignManager
    ):
        """Campaign with checkpoint + result -> both accessible independently."""
        record = manager.create(spec_dict=_make_spec(), name="both-artifacts")
        cid = record.campaign_id

        checkpoint = {"iteration": 10, "state": "converged"}
        result = {"best_kpi_values": {"y": 0.01}, "total_iterations": 10}

        workspace.save_checkpoint(cid, checkpoint)
        workspace.save_result(cid, result)

        loaded_checkpoint = workspace.load_checkpoint(cid)
        loaded_result = workspace.load_result(cid)

        assert loaded_checkpoint is not None
        assert loaded_result is not None
        assert loaded_checkpoint["iteration"] == 10
        assert loaded_result["best_kpi_values"]["y"] == 0.01

    def test_delete_campaign_removes_checkpoint(
        self, workspace: Workspace, manager: CampaignManager
    ):
        """Workspace delete campaign -> checkpoint removed too."""
        record = manager.create(spec_dict=_make_spec(), name="delete-with-checkpoint")
        cid = record.campaign_id

        workspace.save_checkpoint(cid, {"iteration": 5})
        # Verify it's there
        assert workspace.load_checkpoint(cid) is not None

        # Delete the campaign directory entirely
        workspace.delete_campaign(cid)

        # Checkpoint should be gone
        assert workspace.load_checkpoint(cid) is None


# ====================================================================
# 7. InfrastructureStack Integration Tests (~80 tests)
# ====================================================================

from optimization_copilot.infrastructure.integration import (
    InfrastructureConfig,
    InfrastructureStack,
)
from optimization_copilot.infrastructure.cost_tracker import CostTracker, TrialCost
from optimization_copilot.infrastructure.stopping_rule import StoppingDecision, StoppingRule
from optimization_copilot.infrastructure.constraint_engine import (
    Constraint,
    ConstraintEngine,
    ConstraintType,
)
from optimization_copilot.infrastructure.auto_sampler import AutoSampler
from optimization_copilot.infrastructure.batch_scheduler import BatchScheduler
from optimization_copilot.infrastructure.multi_fidelity import FidelityLevel, MultiFidelityManager
from optimization_copilot.infrastructure.parameter_importance import (
    ImportanceResult,
    ParameterImportanceAnalyzer,
)
from optimization_copilot.infrastructure.domain_encoding import (
    EncodingPipeline,
    OneHotEncoding,
    OrdinalEncoding,
)
from optimization_copilot.infrastructure.robust_optimizer import RobustOptimizer
from optimization_copilot.infrastructure.transfer_learning import TransferLearningEngine


# ── Helpers for InfrastructureStack tests ─────────────────────────


def _make_constraint_defs(
    include_hard: bool = True,
    include_soft: bool = False,
    include_unknown: bool = False,
) -> list[dict]:
    """Build constraint definition dicts for InfrastructureConfig."""
    defs = []
    if include_hard:
        defs.append({
            "name": "x_positive",
            "constraint_type": "known_hard",
            "evaluate": lambda params: params.get("x", 0) > 0,
            "tolerance": 0.0,
        })
    if include_soft:
        defs.append({
            "name": "x_small",
            "constraint_type": "known_soft",
            "evaluate": lambda params: params.get("x", 0) < 10,
            "tolerance": 0.3,
        })
    if include_unknown:
        defs.append({
            "name": "feasibility_region",
            "constraint_type": "unknown",
            "tolerance": 0.0,
            "safety_probability": 0.9,
        })
    return defs


def _make_fidelity_level_dicts() -> list[dict]:
    """Build fidelity level dicts for config."""
    return [
        {"level": 0, "name": "coarse", "cost_multiplier": 1.0, "correlation": 0.6},
        {"level": 1, "name": "fine", "cost_multiplier": 5.0, "correlation": 0.9},
        {"level": 2, "name": "experimental", "cost_multiplier": 50.0, "correlation": 1.0},
    ]


def _make_encoding_config() -> dict:
    """Build encoding config dict."""
    return {
        "color": {
            "type": "one_hot",
            "categories": ["red", "green", "blue"],
        },
        "size": {
            "type": "ordinal",
            "ordered_categories": ["small", "medium", "large"],
        },
    }


def _make_historical_campaigns() -> list[dict]:
    """Build historical campaign dicts for transfer learning."""
    return [
        {
            "campaign_id": "hist_001",
            "parameter_specs": [
                {"name": "x", "type": "continuous", "lower": 0.0, "upper": 10.0},
                {"name": "y", "type": "continuous", "lower": 0.0, "upper": 10.0},
            ],
            "observations": [
                {"x": 1.0, "y": 2.0, "objective": 0.5},
                {"x": 3.0, "y": 4.0, "objective": 0.8},
                {"x": 5.0, "y": 6.0, "objective": 0.9},
                {"x": 7.0, "y": 8.0, "objective": 0.7},
                {"x": 2.0, "y": 3.0, "objective": 0.6},
            ],
            "metadata": {"domain": "chemistry"},
        }
    ]


def _make_full_config() -> InfrastructureConfig:
    """Build a config that activates all modules."""
    return InfrastructureConfig(
        budget=1000.0,
        cost_field="total_cost",
        max_trials=50,
        max_cost=900.0,
        improvement_patience=10,
        improvement_threshold=0.001,
        min_uncertainty=0.01,
        constraints=_make_constraint_defs(
            include_hard=True, include_soft=True, include_unknown=True,
        ),
        available_backends=["tpe_sampler", "gaussian_process_bo", "random_sampler"],
        sampler_weights=None,
        historical_campaigns=_make_historical_campaigns(),
        n_workers=4,
        batch_strategy="simple",
        fidelity_levels=_make_fidelity_level_dicts(),
        importance_method="auto",
        encodings=_make_encoding_config(),
        input_noise={"x": 0.1, "y": 0.2},
        n_perturbations=10,
    )


class _FakeSnapshot:
    """Fake snapshot for pre_decide_signals tests."""

    def __init__(
        self,
        n_observations: int = 10,
        constraints: list | None = None,
        objective_names: list[str] | None = None,
        parameter_specs: list | None = None,
    ):
        self.n_observations = n_observations
        self.constraints = constraints or []
        self.objective_names = objective_names or ["objective"]
        self.parameter_specs = parameter_specs or [{"name": "x"}]


class _FakeFingerprint:
    """Fake fingerprint for pre_decide_signals tests."""

    def __init__(self, noise_regime: str = "low"):
        self.noise_regime = noise_regime


# ── InfrastructureConfig Tests ────────────────────────────────────


class TestInfrastructureConfig:
    """Test InfrastructureConfig construction, defaults, and serialization."""

    def test_default_config_all_none_or_empty(self):
        """Default config has budget=None, empty lists, etc."""
        cfg = InfrastructureConfig()
        assert cfg.budget is None
        assert cfg.cost_field == "total_cost"
        assert cfg.max_trials is None
        assert cfg.max_cost is None
        assert cfg.improvement_patience == 15
        assert cfg.improvement_threshold == 0.01
        assert cfg.min_uncertainty is None
        assert cfg.constraints == []
        assert cfg.available_backends == []
        assert cfg.sampler_weights is None
        assert cfg.historical_campaigns == []
        assert cfg.n_workers == 1
        assert cfg.batch_strategy == "simple"
        assert cfg.fidelity_levels == []
        assert cfg.importance_method == "auto"
        assert cfg.encodings == {}
        assert cfg.input_noise == {}
        assert cfg.n_perturbations == 20

    def test_config_with_budget(self):
        """Config with budget set."""
        cfg = InfrastructureConfig(budget=500.0)
        assert cfg.budget == 500.0

    def test_config_with_constraints(self):
        """Config with constraints list."""
        defs = _make_constraint_defs()
        cfg = InfrastructureConfig(constraints=defs)
        assert len(cfg.constraints) == 1
        assert cfg.constraints[0]["name"] == "x_positive"

    def test_config_with_all_fields(self):
        """Config with all fields populated."""
        cfg = _make_full_config()
        assert cfg.budget == 1000.0
        assert cfg.max_trials == 50
        assert cfg.max_cost == 900.0
        assert len(cfg.constraints) == 3
        assert len(cfg.available_backends) == 3
        assert len(cfg.historical_campaigns) == 1
        assert cfg.n_workers == 4
        assert len(cfg.fidelity_levels) == 3
        assert len(cfg.encodings) == 2
        assert len(cfg.input_noise) == 2

    def test_to_dict_returns_all_keys(self):
        """to_dict returns dict with all config keys."""
        cfg = InfrastructureConfig(budget=100.0, max_trials=20)
        d = cfg.to_dict()
        assert "budget" in d
        assert "cost_field" in d
        assert "max_trials" in d
        assert "max_cost" in d
        assert "improvement_patience" in d
        assert "improvement_threshold" in d
        assert "constraints" in d
        assert "available_backends" in d
        assert "fidelity_levels" in d
        assert "encodings" in d
        assert "input_noise" in d
        assert d["budget"] == 100.0
        assert d["max_trials"] == 20

    def test_from_dict_restores_defaults(self):
        """from_dict with empty dict restores default values."""
        cfg = InfrastructureConfig.from_dict({})
        assert cfg.budget is None
        assert cfg.cost_field == "total_cost"
        assert cfg.improvement_patience == 15
        assert cfg.n_workers == 1

    def test_to_dict_from_dict_roundtrip(self):
        """Round-trip serialization preserves all scalar fields."""
        cfg = InfrastructureConfig(
            budget=200.0,
            cost_field="resource_cost",
            max_trials=30,
            max_cost=180.0,
            improvement_patience=20,
            improvement_threshold=0.005,
            min_uncertainty=0.02,
            available_backends=["tpe_sampler", "random_sampler"],
            n_workers=3,
            batch_strategy="round_robin",
            importance_method="variance",
            input_noise={"x": 0.5},
            n_perturbations=30,
        )
        d = cfg.to_dict()
        cfg2 = InfrastructureConfig.from_dict(d)
        assert cfg2.budget == cfg.budget
        assert cfg2.cost_field == cfg.cost_field
        assert cfg2.max_trials == cfg.max_trials
        assert cfg2.max_cost == cfg.max_cost
        assert cfg2.improvement_patience == cfg.improvement_patience
        assert cfg2.improvement_threshold == cfg.improvement_threshold
        assert cfg2.min_uncertainty == cfg.min_uncertainty
        assert cfg2.available_backends == cfg.available_backends
        assert cfg2.n_workers == cfg.n_workers
        assert cfg2.batch_strategy == cfg.batch_strategy
        assert cfg2.importance_method == cfg.importance_method
        assert cfg2.input_noise == cfg.input_noise
        assert cfg2.n_perturbations == cfg.n_perturbations

    def test_to_dict_from_dict_roundtrip_with_fidelity_levels(self):
        """Round-trip preserves fidelity levels."""
        cfg = InfrastructureConfig(fidelity_levels=_make_fidelity_level_dicts())
        d = cfg.to_dict()
        cfg2 = InfrastructureConfig.from_dict(d)
        assert len(cfg2.fidelity_levels) == 3
        assert cfg2.fidelity_levels[0]["level"] == 0
        assert cfg2.fidelity_levels[2]["cost_multiplier"] == 50.0

    def test_to_dict_from_dict_roundtrip_with_encodings(self):
        """Round-trip preserves encoding config."""
        cfg = InfrastructureConfig(encodings=_make_encoding_config())
        d = cfg.to_dict()
        cfg2 = InfrastructureConfig.from_dict(d)
        assert "color" in cfg2.encodings
        assert cfg2.encodings["color"]["type"] == "one_hot"
        assert "size" in cfg2.encodings

    def test_config_sampler_weights_none_serializes(self):
        """sampler_weights=None serializes to None."""
        cfg = InfrastructureConfig(sampler_weights=None)
        d = cfg.to_dict()
        assert d["sampler_weights"] is None
        cfg2 = InfrastructureConfig.from_dict(d)
        assert cfg2.sampler_weights is None


# ── InfrastructureStack Initialization Tests ──────────────────────


class TestInfrastructureStackInit:
    """Test that InfrastructureStack initializes modules based on config."""

    def test_empty_config_no_optional_modules(self):
        """Empty config creates stack with no optional modules (except importance)."""
        stack = InfrastructureStack()
        assert stack.cost_tracker is None
        assert stack.stopping_rule is None
        assert stack.constraint_engine is None
        assert stack.auto_sampler is None
        assert stack.transfer_engine is None
        assert stack.batch_scheduler is None
        assert stack.multi_fidelity is None
        assert stack.encoding_pipeline is None
        assert stack.robust_optimizer is None
        # importance analyzer always present
        assert stack.importance_analyzer is not None

    def test_none_config_same_as_empty(self):
        """Passing None config is same as default InfrastructureConfig."""
        stack = InfrastructureStack(None)
        assert stack.cost_tracker is None
        assert stack.importance_analyzer is not None

    def test_budget_creates_cost_tracker(self):
        """Config with budget creates CostTracker."""
        cfg = InfrastructureConfig(budget=500.0)
        stack = InfrastructureStack(cfg)
        assert stack.cost_tracker is not None
        assert stack.cost_tracker.budget == 500.0

    def test_no_budget_no_cost_tracker(self):
        """Config without budget does not create CostTracker."""
        cfg = InfrastructureConfig(budget=None)
        stack = InfrastructureStack(cfg)
        assert stack.cost_tracker is None

    def test_max_trials_creates_stopping_rule(self):
        """Config with max_trials creates StoppingRule."""
        cfg = InfrastructureConfig(max_trials=100)
        stack = InfrastructureStack(cfg)
        assert stack.stopping_rule is not None

    def test_max_cost_creates_stopping_rule(self):
        """Config with max_cost creates StoppingRule."""
        cfg = InfrastructureConfig(max_cost=500.0)
        stack = InfrastructureStack(cfg)
        assert stack.stopping_rule is not None

    def test_min_uncertainty_creates_stopping_rule(self):
        """Config with min_uncertainty creates StoppingRule."""
        cfg = InfrastructureConfig(min_uncertainty=0.01)
        stack = InfrastructureStack(cfg)
        assert stack.stopping_rule is not None

    def test_no_stopping_criteria_no_rule(self):
        """Config with no stopping criteria does not create StoppingRule."""
        cfg = InfrastructureConfig()
        stack = InfrastructureStack(cfg)
        assert stack.stopping_rule is None

    def test_constraints_create_engine(self):
        """Config with constraints creates ConstraintEngine."""
        cfg = InfrastructureConfig(
            constraints=_make_constraint_defs(include_hard=True),
        )
        stack = InfrastructureStack(cfg)
        assert stack.constraint_engine is not None
        assert stack.constraint_engine.n_constraints == 1

    def test_no_constraints_no_engine(self):
        """Config with empty constraints does not create ConstraintEngine."""
        cfg = InfrastructureConfig(constraints=[])
        stack = InfrastructureStack(cfg)
        assert stack.constraint_engine is None

    def test_backends_create_auto_sampler(self):
        """Config with available_backends creates AutoSampler."""
        cfg = InfrastructureConfig(
            available_backends=["tpe_sampler", "random_sampler"],
        )
        stack = InfrastructureStack(cfg)
        assert stack.auto_sampler is not None
        assert len(stack.auto_sampler.available_backends) == 2

    def test_no_backends_no_sampler(self):
        """Config with empty backends does not create AutoSampler."""
        cfg = InfrastructureConfig(available_backends=[])
        stack = InfrastructureStack(cfg)
        assert stack.auto_sampler is None

    def test_n_workers_gt_1_creates_batch_scheduler(self):
        """Config with n_workers > 1 creates BatchScheduler."""
        cfg = InfrastructureConfig(n_workers=4, batch_strategy="simple")
        stack = InfrastructureStack(cfg)
        assert stack.batch_scheduler is not None
        assert stack.batch_scheduler.n_workers == 4

    def test_n_workers_1_no_batch_scheduler(self):
        """Config with n_workers=1 does not create BatchScheduler."""
        cfg = InfrastructureConfig(n_workers=1)
        stack = InfrastructureStack(cfg)
        assert stack.batch_scheduler is None

    def test_fidelity_levels_create_multi_fidelity(self):
        """Config with fidelity_levels creates MultiFidelityManager."""
        cfg = InfrastructureConfig(fidelity_levels=_make_fidelity_level_dicts())
        stack = InfrastructureStack(cfg)
        assert stack.multi_fidelity is not None
        assert stack.multi_fidelity.n_levels == 3

    def test_no_fidelity_levels_no_multi_fidelity(self):
        """Config with empty fidelity_levels does not create MultiFidelityManager."""
        cfg = InfrastructureConfig(fidelity_levels=[])
        stack = InfrastructureStack(cfg)
        assert stack.multi_fidelity is None

    def test_encodings_create_pipeline(self):
        """Config with encodings creates EncodingPipeline."""
        cfg = InfrastructureConfig(encodings=_make_encoding_config())
        stack = InfrastructureStack(cfg)
        assert stack.encoding_pipeline is not None
        assert "color" in stack.encoding_pipeline.param_names
        assert "size" in stack.encoding_pipeline.param_names

    def test_no_encodings_no_pipeline(self):
        """Config with empty encodings does not create EncodingPipeline."""
        cfg = InfrastructureConfig(encodings={})
        stack = InfrastructureStack(cfg)
        assert stack.encoding_pipeline is None

    def test_input_noise_creates_robust_optimizer(self):
        """Config with input_noise creates RobustOptimizer."""
        cfg = InfrastructureConfig(
            input_noise={"x": 0.1},
            n_perturbations=15,
        )
        stack = InfrastructureStack(cfg)
        assert stack.robust_optimizer is not None
        assert stack.robust_optimizer.n_perturbations == 15

    def test_no_input_noise_no_robust_optimizer(self):
        """Config with empty input_noise does not create RobustOptimizer."""
        cfg = InfrastructureConfig(input_noise={})
        stack = InfrastructureStack(cfg)
        assert stack.robust_optimizer is None

    def test_historical_campaigns_create_transfer_engine(self):
        """Config with historical_campaigns creates TransferLearningEngine."""
        cfg = InfrastructureConfig(
            historical_campaigns=_make_historical_campaigns(),
        )
        stack = InfrastructureStack(cfg)
        assert stack.transfer_engine is not None
        assert stack.transfer_engine.n_campaigns == 1
        assert "hist_001" in stack.transfer_engine.campaign_ids

    def test_no_campaigns_no_transfer_engine(self):
        """Config with empty historical_campaigns does not create engine."""
        cfg = InfrastructureConfig(historical_campaigns=[])
        stack = InfrastructureStack(cfg)
        assert stack.transfer_engine is None

    def test_full_config_creates_all_modules(self):
        """Full config creates all modules."""
        cfg = _make_full_config()
        stack = InfrastructureStack(cfg)
        assert stack.cost_tracker is not None
        assert stack.stopping_rule is not None
        assert stack.constraint_engine is not None
        assert stack.auto_sampler is not None
        assert stack.transfer_engine is not None
        assert stack.batch_scheduler is not None
        assert stack.multi_fidelity is not None
        assert stack.importance_analyzer is not None
        assert stack.encoding_pipeline is not None
        assert stack.robust_optimizer is not None

    def test_importance_method_passed_through(self):
        """importance_method config is passed to ParameterImportanceAnalyzer."""
        cfg = InfrastructureConfig(importance_method="variance")
        stack = InfrastructureStack(cfg)
        assert stack.importance_analyzer.method == "variance"


# ── pre_decide_signals Tests ─────────────────────────────────────


class TestPreDecideSignals:
    """Test pre_decide_signals method."""

    def test_empty_stack_returns_empty_dict(self):
        """No modules -> empty signals dict."""
        stack = InfrastructureStack()
        signals = stack.pre_decide_signals(
            snapshot=_FakeSnapshot(),
            diagnostics={},
            fingerprint=_FakeFingerprint(),
        )
        assert signals == {}

    def test_cost_tracker_adds_cost_signals(self):
        """CostTracker active -> signals contain cost_signals."""
        cfg = InfrastructureConfig(budget=1000.0)
        stack = InfrastructureStack(cfg)
        # Record a trial so there is cost data
        stack.record_trial(
            trial_params={"x": 1.0},
            kpi_values={"obj": 0.5},
            wall_time=1.0,
            resource_cost=10.0,
        )
        signals = stack.pre_decide_signals(
            snapshot=_FakeSnapshot(),
            diagnostics={},
            fingerprint=_FakeFingerprint(),
        )
        assert "cost_signals" in signals
        cs = signals["cost_signals"]
        assert "total_spent" in cs
        assert "remaining_budget" in cs
        assert "average_cost_per_trial" in cs
        assert "estimated_remaining_trials" in cs
        assert "n_trials_recorded" in cs
        assert cs["n_trials_recorded"] == 1

    def test_auto_sampler_adds_backend_policy(self):
        """AutoSampler active -> signals contain backend_policy."""
        cfg = InfrastructureConfig(
            available_backends=["tpe_sampler", "random_sampler"],
        )
        stack = InfrastructureStack(cfg)
        signals = stack.pre_decide_signals(
            snapshot=_FakeSnapshot(n_observations=10),
            diagnostics={"phase": "learning"},
            fingerprint=_FakeFingerprint(noise_regime="low"),
        )
        assert "backend_policy" in signals
        assert isinstance(signals["backend_policy"], str)

    def test_both_cost_and_sampler_present(self):
        """Both CostTracker and AutoSampler active -> both signals present."""
        cfg = InfrastructureConfig(
            budget=500.0,
            available_backends=["tpe_sampler", "gaussian_process_bo"],
        )
        stack = InfrastructureStack(cfg)
        signals = stack.pre_decide_signals(
            snapshot=_FakeSnapshot(),
            diagnostics={},
            fingerprint=_FakeFingerprint(),
        )
        assert "cost_signals" in signals
        assert "backend_policy" in signals

    def test_cost_signals_reflect_recorded_trials(self):
        """Cost signals reflect actual trial history."""
        cfg = InfrastructureConfig(budget=100.0)
        stack = InfrastructureStack(cfg)
        for i in range(5):
            stack.record_trial(
                trial_params={"x": float(i)},
                kpi_values={"obj": float(i)},
                resource_cost=10.0,
            )
        signals = stack.pre_decide_signals(
            snapshot=_FakeSnapshot(),
            diagnostics={},
            fingerprint=_FakeFingerprint(),
        )
        cs = signals["cost_signals"]
        assert cs["n_trials_recorded"] == 5
        assert cs["total_spent"] > 0

    def test_snapshot_attributes_used_for_sampler(self):
        """Snapshot attributes are extracted for AutoSampler selection."""
        cfg = InfrastructureConfig(
            available_backends=["tpe_sampler", "gaussian_process_bo"],
        )
        stack = InfrastructureStack(cfg)
        snapshot = _FakeSnapshot(
            n_observations=25,
            constraints=[{"name": "c1"}],
            objective_names=["obj1", "obj2"],
            parameter_specs=[{"name": "x"}, {"name": "y"}, {"name": "z"}],
        )
        signals = stack.pre_decide_signals(
            snapshot=snapshot,
            diagnostics={"phase": "exploitation"},
            fingerprint=_FakeFingerprint(noise_regime="high"),
        )
        assert "backend_policy" in signals

    def test_no_snapshot_attributes_still_works(self):
        """Object without snapshot attributes does not crash."""
        cfg = InfrastructureConfig(
            available_backends=["tpe_sampler"],
        )
        stack = InfrastructureStack(cfg)
        # Pass a plain object with no attributes
        signals = stack.pre_decide_signals(
            snapshot=object(),
            diagnostics={},
            fingerprint=object(),
        )
        assert "backend_policy" in signals

    def test_budget_remaining_passed_to_sampler(self):
        """When CostTracker is active with trials, budget_remaining reaches sampler."""
        cfg = InfrastructureConfig(
            budget=100.0,
            available_backends=["tpe_sampler", "random_sampler"],
        )
        stack = InfrastructureStack(cfg)
        # Record many trials to make budget tight
        for i in range(9):
            stack.record_trial(
                trial_params={"x": float(i)},
                kpi_values={"obj": float(i)},
                resource_cost=10.0,
            )
        signals = stack.pre_decide_signals(
            snapshot=_FakeSnapshot(),
            diagnostics={},
            fingerprint=_FakeFingerprint(),
        )
        assert "backend_policy" in signals
        assert "cost_signals" in signals

    def test_diagnostics_phase_used_for_sampler(self):
        """Phase from diagnostics is forwarded to AutoSampler."""
        cfg = InfrastructureConfig(
            available_backends=["tpe_sampler", "gaussian_process_bo", "random_sampler"],
        )
        stack = InfrastructureStack(cfg)
        signals = stack.pre_decide_signals(
            snapshot=_FakeSnapshot(),
            diagnostics={"phase": "cold_start"},
            fingerprint=_FakeFingerprint(),
        )
        assert "backend_policy" in signals

    def test_empty_diagnostics_uses_learning_phase(self):
        """Empty diagnostics defaults to 'learning' phase."""
        cfg = InfrastructureConfig(
            available_backends=["tpe_sampler"],
        )
        stack = InfrastructureStack(cfg)
        # Should not raise
        signals = stack.pre_decide_signals(
            snapshot=_FakeSnapshot(),
            diagnostics={},
            fingerprint=_FakeFingerprint(),
        )
        assert "backend_policy" in signals


# ── check_stopping Tests ─────────────────────────────────────────


class TestCheckStopping:
    """Test check_stopping method."""

    def test_no_stopping_rule_returns_none(self):
        """No StoppingRule -> returns None."""
        stack = InfrastructureStack()
        result = stack.check_stopping(n_trials=100)
        assert result is None

    def test_max_trials_triggers_stop(self):
        """Reaching max_trials triggers stopping."""
        cfg = InfrastructureConfig(max_trials=10)
        stack = InfrastructureStack(cfg)
        result = stack.check_stopping(n_trials=10)
        assert result is not None
        assert result.should_stop is True
        assert result.criterion == "max_trials"

    def test_below_max_trials_continues(self):
        """Below max_trials does not trigger stopping."""
        cfg = InfrastructureConfig(max_trials=10)
        stack = InfrastructureStack(cfg)
        result = stack.check_stopping(n_trials=5)
        assert result is not None
        assert result.should_stop is False

    def test_budget_exceeded_triggers_stop(self):
        """Exceeding max_cost with CostTracker triggers stopping."""
        cfg = InfrastructureConfig(budget=100.0, max_cost=50.0)
        stack = InfrastructureStack(cfg)
        # Record trials that exceed max_cost
        for i in range(6):
            stack.record_trial(
                trial_params={"x": float(i)},
                kpi_values={"obj": float(i)},
                resource_cost=10.0,
            )
        result = stack.check_stopping(n_trials=6)
        assert result is not None
        assert result.should_stop is True
        assert result.criterion == "budget_exhausted"

    def test_budget_not_exceeded_continues(self):
        """Cost below max_cost does not trigger stopping."""
        cfg = InfrastructureConfig(budget=100.0, max_cost=50.0)
        stack = InfrastructureStack(cfg)
        # Record a few cheap trials
        for i in range(2):
            stack.record_trial(
                trial_params={"x": float(i)},
                kpi_values={"obj": float(i)},
                resource_cost=5.0,
            )
        result = stack.check_stopping(n_trials=2)
        assert result is not None
        assert result.should_stop is False

    def test_improvement_stagnation_triggers_stop(self):
        """Stagnant best_values triggers stopping."""
        cfg = InfrastructureConfig(
            max_trials=100,
            improvement_patience=5,
            improvement_threshold=0.01,
        )
        stack = InfrastructureStack(cfg)
        # 5 identical best values = stagnation
        best_values = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = stack.check_stopping(n_trials=5, best_values=best_values)
        assert result is not None
        assert result.should_stop is True
        assert result.criterion == "stagnation"

    def test_improving_values_continues(self):
        """Improving best_values does not trigger stagnation."""
        cfg = InfrastructureConfig(
            max_trials=100,
            improvement_patience=5,
            improvement_threshold=0.01,
        )
        stack = InfrastructureStack(cfg)
        best_values = [1.0, 0.9, 0.8, 0.7, 0.6]
        result = stack.check_stopping(n_trials=5, best_values=best_values)
        assert result is not None
        assert result.should_stop is False

    def test_convergence_triggers_stop(self):
        """Low uncertainty triggers convergence stopping."""
        cfg = InfrastructureConfig(min_uncertainty=0.05)
        stack = InfrastructureStack(cfg)
        result = stack.check_stopping(n_trials=10, current_uncertainty=0.01)
        assert result is not None
        assert result.should_stop is True
        assert result.criterion == "convergence"

    def test_high_uncertainty_continues(self):
        """High uncertainty does not trigger convergence."""
        cfg = InfrastructureConfig(min_uncertainty=0.05)
        stack = InfrastructureStack(cfg)
        result = stack.check_stopping(n_trials=10, current_uncertainty=0.1)
        assert result is not None
        assert result.should_stop is False

    def test_stopping_decision_is_dataclass(self):
        """check_stopping returns StoppingDecision with expected fields."""
        cfg = InfrastructureConfig(max_trials=5)
        stack = InfrastructureStack(cfg)
        result = stack.check_stopping(n_trials=5)
        assert hasattr(result, "should_stop")
        assert hasattr(result, "reason")
        assert hasattr(result, "criterion")
        assert hasattr(result, "details")


# ── filter_candidates Tests ──────────────────────────────────────


class TestFilterCandidates:
    """Test filter_candidates method."""

    def test_no_constraint_engine_returns_all(self):
        """No ConstraintEngine -> returns all candidates unchanged."""
        stack = InfrastructureStack()
        candidates = [{"x": 1}, {"x": 2}, {"x": 3}]
        result = stack.filter_candidates(candidates)
        assert result == candidates

    def test_returns_copy_not_same_list(self):
        """Returns a new list, not the same object."""
        stack = InfrastructureStack()
        candidates = [{"x": 1}]
        result = stack.filter_candidates(candidates)
        assert result is not candidates

    def test_hard_constraint_filters_violators(self):
        """Hard constraint removes violating candidates."""
        cfg = InfrastructureConfig(
            constraints=_make_constraint_defs(include_hard=True),
        )
        stack = InfrastructureStack(cfg)
        candidates = [
            {"x": 5.0},   # passes: x > 0
            {"x": -1.0},  # fails: x <= 0
            {"x": 0.0},   # fails: x == 0, not > 0
            {"x": 3.0},   # passes: x > 0
        ]
        result = stack.filter_candidates(candidates)
        assert len(result) == 2
        assert all(c["x"] > 0 for c in result)

    def test_all_violate_returns_empty(self):
        """All candidates violate -> returns empty list."""
        cfg = InfrastructureConfig(
            constraints=_make_constraint_defs(include_hard=True),
        )
        stack = InfrastructureStack(cfg)
        candidates = [{"x": -1.0}, {"x": 0.0}, {"x": -5.0}]
        result = stack.filter_candidates(candidates)
        assert result == []

    def test_all_pass_returns_all(self):
        """All candidates pass -> returns all."""
        cfg = InfrastructureConfig(
            constraints=_make_constraint_defs(include_hard=True),
        )
        stack = InfrastructureStack(cfg)
        candidates = [{"x": 1.0}, {"x": 5.0}, {"x": 10.0}]
        result = stack.filter_candidates(candidates)
        assert len(result) == 3

    def test_empty_candidates_returns_empty(self):
        """Empty candidate list -> returns empty."""
        cfg = InfrastructureConfig(
            constraints=_make_constraint_defs(include_hard=True),
        )
        stack = InfrastructureStack(cfg)
        result = stack.filter_candidates([])
        assert result == []

    def test_soft_constraints_do_not_filter(self):
        """Soft constraints do not filter candidates (only hard do)."""
        cfg = InfrastructureConfig(
            constraints=_make_constraint_defs(include_hard=False, include_soft=True),
        )
        stack = InfrastructureStack(cfg)
        candidates = [{"x": 100.0}, {"x": 200.0}]  # violate x < 10
        result = stack.filter_candidates(candidates)
        assert len(result) == 2

    def test_parameter_specs_passed_through(self):
        """parameter_specs argument is accepted without error."""
        cfg = InfrastructureConfig(
            constraints=_make_constraint_defs(include_hard=True),
        )
        stack = InfrastructureStack(cfg)
        candidates = [{"x": 1.0}]
        specs = [{"name": "x", "type": "continuous"}]
        result = stack.filter_candidates(candidates, parameter_specs=specs)
        assert len(result) == 1


# ── record_trial Tests ───────────────────────────────────────────


class TestRecordTrial:
    """Test record_trial method."""

    def test_no_modules_no_op(self):
        """No modules configured -> record_trial is a no-op (no error)."""
        stack = InfrastructureStack()
        stack.record_trial(
            trial_params={"x": 1.0},
            kpi_values={"obj": 0.5},
            wall_time=1.0,
        )
        # No assertion needed; just verify no exception

    def test_cost_tracker_updated(self):
        """CostTracker is updated on record_trial."""
        cfg = InfrastructureConfig(budget=1000.0)
        stack = InfrastructureStack(cfg)
        assert stack.cost_tracker.n_trials == 0
        stack.record_trial(
            trial_params={"x": 1.0},
            kpi_values={"obj": 0.5},
            wall_time=2.0,
            resource_cost=10.0,
            compute_cost=5.0,
        )
        assert stack.cost_tracker.n_trials == 1
        assert stack.cost_tracker.total_spent > 0

    def test_multiple_trials_accumulate(self):
        """Multiple record_trial calls accumulate in CostTracker."""
        cfg = InfrastructureConfig(budget=1000.0)
        stack = InfrastructureStack(cfg)
        for i in range(5):
            stack.record_trial(
                trial_params={"x": float(i)},
                kpi_values={"obj": float(i)},
                resource_cost=10.0,
            )
        assert stack.cost_tracker.n_trials == 5

    def test_constraint_engine_updated_with_results(self):
        """ConstraintEngine unknown constraints are updated."""
        cfg = InfrastructureConfig(
            constraints=_make_constraint_defs(include_hard=False, include_unknown=True),
        )
        stack = InfrastructureStack(cfg)
        stack.record_trial(
            trial_params={"x": 1.0},
            kpi_values={"obj": 0.5},
            constraint_results={"feasibility_region": True},
        )
        # Check that the unknown constraint got an observation
        for c in stack.constraint_engine.constraints:
            if c.name == "feasibility_region":
                assert len(c.observations) == 1

    def test_no_constraint_results_skips_update(self):
        """constraint_results=None skips constraint update."""
        cfg = InfrastructureConfig(
            constraints=_make_constraint_defs(include_hard=False, include_unknown=True),
        )
        stack = InfrastructureStack(cfg)
        stack.record_trial(
            trial_params={"x": 1.0},
            kpi_values={"obj": 0.5},
            constraint_results=None,
        )
        for c in stack.constraint_engine.constraints:
            if c.name == "feasibility_region":
                assert len(c.observations) == 0

    def test_multi_fidelity_updated(self):
        """MultiFidelityManager is updated with fidelity_level."""
        cfg = InfrastructureConfig(fidelity_levels=_make_fidelity_level_dicts())
        stack = InfrastructureStack(cfg)
        stack.record_trial(
            trial_params={"x": 1.0},
            kpi_values={"obj": 0.8},
            fidelity_level=0,
        )
        obs = stack.multi_fidelity.get_observations(fidelity=0)
        assert len(obs) == 1
        assert obs[0]["objective"] == 0.8

    def test_fidelity_none_skips_multi_fidelity(self):
        """fidelity_level=None skips multi-fidelity update."""
        cfg = InfrastructureConfig(fidelity_levels=_make_fidelity_level_dicts())
        stack = InfrastructureStack(cfg)
        stack.record_trial(
            trial_params={"x": 1.0},
            kpi_values={"obj": 0.8},
            fidelity_level=None,
        )
        all_obs = stack.multi_fidelity.get_observations()
        assert len(all_obs) == 0

    def test_trial_id_auto_generated(self):
        """trial_id is auto-generated when empty."""
        cfg = InfrastructureConfig(budget=1000.0)
        stack = InfrastructureStack(cfg)
        stack.record_trial(
            trial_params={"x": 1.0},
            kpi_values={"obj": 0.5},
            trial_id="",
        )
        assert stack.cost_tracker.n_trials == 1


# ── warm_start_points Tests ──────────────────────────────────────


class TestWarmStartPoints:
    """Test warm_start_points method."""

    def test_no_transfer_engine_returns_empty(self):
        """No TransferLearningEngine -> returns empty list."""
        stack = InfrastructureStack()
        result = stack.warm_start_points(
            parameter_specs=[{"name": "x", "type": "continuous", "lower": 0, "upper": 10}],
        )
        assert result == []

    def test_with_similar_campaigns_returns_points(self):
        """Similar campaign -> returns warm-start points."""
        cfg = InfrastructureConfig(
            historical_campaigns=_make_historical_campaigns(),
        )
        stack = InfrastructureStack(cfg)
        # Use same parameter specs as historical campaign
        current_specs = [
            {"name": "x", "type": "continuous", "lower": 0.0, "upper": 10.0},
            {"name": "y", "type": "continuous", "lower": 0.0, "upper": 10.0},
        ]
        result = stack.warm_start_points(
            parameter_specs=current_specs,
            n_points=3,
            min_similarity=0.3,
        )
        # Should get some points from hist_001 (exact parameter overlap)
        assert len(result) >= 1

    def test_no_similar_campaigns_returns_empty(self):
        """High similarity threshold with dissimilar campaigns -> empty."""
        cfg = InfrastructureConfig(
            historical_campaigns=_make_historical_campaigns(),
        )
        stack = InfrastructureStack(cfg)
        # Use completely different parameter specs
        current_specs = [
            {"name": "alpha", "type": "continuous", "lower": 0, "upper": 100},
            {"name": "beta", "type": "continuous", "lower": 0, "upper": 100},
        ]
        result = stack.warm_start_points(
            parameter_specs=current_specs,
            n_points=5,
            min_similarity=0.99,
        )
        assert result == []

    def test_n_points_limits_output(self):
        """n_points limits the number of returned points."""
        cfg = InfrastructureConfig(
            historical_campaigns=_make_historical_campaigns(),
        )
        stack = InfrastructureStack(cfg)
        current_specs = [
            {"name": "x", "type": "continuous", "lower": 0.0, "upper": 10.0},
            {"name": "y", "type": "continuous", "lower": 0.0, "upper": 10.0},
        ]
        result = stack.warm_start_points(
            parameter_specs=current_specs,
            n_points=1,
            min_similarity=0.3,
        )
        assert len(result) <= 1

    def test_current_metadata_used(self):
        """current_metadata is passed to transfer engine."""
        cfg = InfrastructureConfig(
            historical_campaigns=_make_historical_campaigns(),
        )
        stack = InfrastructureStack(cfg)
        current_specs = [
            {"name": "x", "type": "continuous", "lower": 0.0, "upper": 10.0},
            {"name": "y", "type": "continuous", "lower": 0.0, "upper": 10.0},
        ]
        # Should not raise
        result = stack.warm_start_points(
            parameter_specs=current_specs,
            n_points=3,
            min_similarity=0.3,
            current_metadata={"domain": "chemistry"},
        )
        # With matching domain metadata, similarity should be higher
        assert isinstance(result, list)


# ── Serialization Tests ──────────────────────────────────────────


class TestStackSerialization:
    """Test to_dict / from_dict round-trip."""

    def test_empty_stack_roundtrip(self):
        """Empty stack serializes and deserializes correctly."""
        stack = InfrastructureStack()
        data = stack.to_dict()
        assert "config" in data
        assert "importance_analyzer" in data

        restored = InfrastructureStack.from_dict(data)
        assert restored.cost_tracker is None
        assert restored.stopping_rule is None
        assert restored.importance_analyzer is not None

    def test_budget_stack_roundtrip(self):
        """Stack with budget preserves cost tracker state."""
        cfg = InfrastructureConfig(budget=500.0)
        stack = InfrastructureStack(cfg)
        stack.record_trial(
            trial_params={"x": 1.0},
            kpi_values={"obj": 0.5},
            resource_cost=25.0,
        )

        data = stack.to_dict()
        restored = InfrastructureStack.from_dict(data)

        assert restored.cost_tracker is not None
        assert restored.cost_tracker.budget == 500.0
        assert restored.cost_tracker.n_trials == 1
        assert restored.cost_tracker.total_spent > 0

    def test_stopping_rule_roundtrip(self):
        """Stack with stopping rule preserves config."""
        cfg = InfrastructureConfig(max_trials=50, max_cost=200.0)
        stack = InfrastructureStack(cfg)

        data = stack.to_dict()
        restored = InfrastructureStack.from_dict(data)

        assert restored.stopping_rule is not None
        # Test that the stopping rule works the same
        result = restored.check_stopping(n_trials=50)
        assert result is not None
        assert result.should_stop is True

    def test_auto_sampler_roundtrip(self):
        """Stack with auto sampler preserves backends."""
        cfg = InfrastructureConfig(
            available_backends=["tpe_sampler", "random_sampler"],
        )
        stack = InfrastructureStack(cfg)

        data = stack.to_dict()
        restored = InfrastructureStack.from_dict(data)

        assert restored.auto_sampler is not None
        assert set(restored.auto_sampler.available_backends) == {"tpe_sampler", "random_sampler"}

    def test_batch_scheduler_roundtrip(self):
        """Stack with batch scheduler preserves config."""
        cfg = InfrastructureConfig(n_workers=4, batch_strategy="simple")
        stack = InfrastructureStack(cfg)

        data = stack.to_dict()
        restored = InfrastructureStack.from_dict(data)

        assert restored.batch_scheduler is not None
        assert restored.batch_scheduler.n_workers == 4

    def test_multi_fidelity_roundtrip(self):
        """Stack with multi-fidelity preserves levels and observations."""
        cfg = InfrastructureConfig(fidelity_levels=_make_fidelity_level_dicts())
        stack = InfrastructureStack(cfg)
        stack.record_trial(
            trial_params={"x": 1.0},
            kpi_values={"obj": 0.8},
            fidelity_level=0,
        )

        data = stack.to_dict()
        restored = InfrastructureStack.from_dict(data)

        assert restored.multi_fidelity is not None
        assert restored.multi_fidelity.n_levels == 3
        obs = restored.multi_fidelity.get_observations(fidelity=0)
        assert len(obs) == 1

    def test_robust_optimizer_roundtrip(self):
        """Stack with robust optimizer preserves config."""
        cfg = InfrastructureConfig(
            input_noise={"x": 0.1, "y": 0.2},
            n_perturbations=25,
        )
        stack = InfrastructureStack(cfg)

        data = stack.to_dict()
        restored = InfrastructureStack.from_dict(data)

        assert restored.robust_optimizer is not None
        assert restored.robust_optimizer.noise_config == {"x": 0.1, "y": 0.2}
        assert restored.robust_optimizer.n_perturbations == 25

    def test_transfer_engine_roundtrip(self):
        """Stack with transfer engine preserves campaigns."""
        cfg = InfrastructureConfig(
            historical_campaigns=_make_historical_campaigns(),
        )
        stack = InfrastructureStack(cfg)

        data = stack.to_dict()
        restored = InfrastructureStack.from_dict(data)

        assert restored.transfer_engine is not None
        assert restored.transfer_engine.n_campaigns == 1

    def test_full_stack_roundtrip(self):
        """Full stack with all modules round-trips."""
        cfg = _make_full_config()
        stack = InfrastructureStack(cfg)

        # Record some data
        stack.record_trial(
            trial_params={"x": 1.0, "y": 2.0},
            kpi_values={"obj": 0.5},
            wall_time=1.0,
            resource_cost=10.0,
            compute_cost=5.0,
            fidelity_level=0,
        )

        data = stack.to_dict()
        restored = InfrastructureStack.from_dict(data)

        assert restored.cost_tracker is not None
        assert restored.stopping_rule is not None
        assert restored.constraint_engine is not None
        assert restored.auto_sampler is not None
        assert restored.transfer_engine is not None
        assert restored.batch_scheduler is not None
        assert restored.multi_fidelity is not None
        assert restored.encoding_pipeline is not None
        assert restored.robust_optimizer is not None
        assert restored.importance_analyzer is not None

    def test_to_dict_contains_config_key(self):
        """to_dict output always contains 'config' key."""
        stack = InfrastructureStack()
        data = stack.to_dict()
        assert "config" in data
        assert isinstance(data["config"], dict)


# ── Summary Tests ────────────────────────────────────────────────


class TestStackSummary:
    """Test summary method."""

    def test_empty_stack_summary(self):
        """Empty stack summary has only importance_analyzer."""
        stack = InfrastructureStack()
        s = stack.summary()
        assert "active_modules" in s
        assert "importance_analyzer" in s["active_modules"]
        # No other modules
        for mod in [
            "cost_tracker", "stopping_rule", "constraint_engine",
            "auto_sampler", "transfer_engine", "batch_scheduler",
            "multi_fidelity", "encoding_pipeline", "robust_optimizer",
        ]:
            assert mod not in s["active_modules"]

    def test_full_stack_summary_all_modules(self):
        """Full stack summary lists all active modules."""
        cfg = _make_full_config()
        stack = InfrastructureStack(cfg)
        s = stack.summary()
        expected_modules = [
            "cost_tracker", "stopping_rule", "constraint_engine",
            "auto_sampler", "transfer_engine", "batch_scheduler",
            "multi_fidelity", "importance_analyzer", "encoding_pipeline",
            "robust_optimizer",
        ]
        for mod in expected_modules:
            assert mod in s["active_modules"], f"{mod} missing from active_modules"

    def test_summary_cost_tracker_details(self):
        """Summary includes cost tracker details."""
        cfg = InfrastructureConfig(budget=500.0)
        stack = InfrastructureStack(cfg)
        s = stack.summary()
        assert "cost_tracker" in s
        assert s["cost_tracker"]["budget"] == 500.0
        assert s["cost_tracker"]["total_spent"] == 0.0
        assert s["cost_tracker"]["n_trials"] == 0

    def test_summary_importance_analyzer_details(self):
        """Summary includes importance analyzer method."""
        cfg = InfrastructureConfig(importance_method="variance")
        stack = InfrastructureStack(cfg)
        s = stack.summary()
        assert "importance_analyzer" in s
        assert s["importance_analyzer"]["method"] == "variance"

    def test_repr_contains_active_modules(self):
        """repr includes active module names."""
        cfg = InfrastructureConfig(budget=100.0)
        stack = InfrastructureStack(cfg)
        r = repr(stack)
        assert "InfrastructureStack" in r
        assert "cost_tracker" in r


# ── Additional Method Tests ──────────────────────────────────────


class TestAdditionalMethods:
    """Test encoding, robustify, batch, fidelity, and analysis methods."""

    def test_encode_params_without_pipeline(self):
        """encode_params without pipeline extracts numeric values."""
        stack = InfrastructureStack()
        result = stack.encode_params({"x": 1.0, "y": 2.5, "name": "foo"})
        assert result == [1.0, 2.5]

    def test_encode_params_with_pipeline(self):
        """encode_params with pipeline uses registered encodings."""
        cfg = InfrastructureConfig(encodings=_make_encoding_config())
        stack = InfrastructureStack(cfg)
        result = stack.encode_params({
            "color": "red",
            "size": "medium",
            "pressure": 1.5,
        })
        # one_hot(red) = [1,0,0], ordinal(medium) = [0.5], pass-through = [1.5]
        assert len(result) == 5
        assert result[0] == 1.0  # red
        assert result[1] == 0.0  # green
        assert result[2] == 0.0  # blue
        assert result[3] == 0.5  # medium
        assert result[4] == 1.5  # pressure

    def test_decode_features_without_pipeline_raises(self):
        """decode_features without pipeline raises RuntimeError."""
        stack = InfrastructureStack()
        with pytest.raises(RuntimeError, match="Cannot decode"):
            stack.decode_features([1.0, 2.0], ["x", "y"])

    def test_decode_features_with_pipeline(self):
        """decode_features with pipeline decodes correctly."""
        cfg = InfrastructureConfig(encodings=_make_encoding_config())
        stack = InfrastructureStack(cfg)
        # Encode first
        params = {"color": "green", "size": "large", "pressure": 2.0}
        features = stack.encode_params(params)
        # Decode with param names in encoding order, then pass-through
        decoded = stack.decode_features(features, ["color", "size", "pressure"])
        assert decoded["color"] == "green"
        assert decoded["size"] == "large"
        assert decoded["pressure"] == 2.0

    def test_robustify_acquisition_without_optimizer(self):
        """robustify_acquisition without optimizer returns original values."""
        stack = InfrastructureStack()
        acq_values = [1.0, 2.0, 3.0]
        result = stack.robustify_acquisition(
            candidates=[{"x": 1}, {"x": 2}, {"x": 3}],
            acquisition_values=acq_values,
            parameter_specs=[{"name": "x", "min": 0, "max": 10}],
        )
        assert result == acq_values

    def test_robustify_acquisition_with_optimizer(self):
        """robustify_acquisition with optimizer penalizes candidates."""
        cfg = InfrastructureConfig(
            input_noise={"x": 0.5},
            n_perturbations=10,
        )
        stack = InfrastructureStack(cfg)
        result = stack.robustify_acquisition(
            candidates=[{"x": 0.1}, {"x": 5.0}],
            acquisition_values=[10.0, 10.0],
            parameter_specs=[{"name": "x", "min": 0, "max": 10}],
        )
        assert len(result) == 2
        # Near boundary (x=0.1) should be penalized more
        assert result[0] <= result[1]

    def test_weight_by_constraints_without_engine(self):
        """weight_by_constraints without engine returns original values."""
        stack = InfrastructureStack()
        acq_values = [1.0, 2.0]
        result = stack.weight_by_constraints(
            acquisition_values=acq_values,
            candidates=[{"x": 1}, {"x": 2}],
        )
        assert result == acq_values

    def test_schedule_batch_without_scheduler(self):
        """schedule_batch without scheduler returns suggestions as-is."""
        stack = InfrastructureStack()
        suggestions = [{"x": 1.0}, {"x": 2.0}]
        result = stack.schedule_batch(suggestions)
        assert result == suggestions

    def test_schedule_batch_with_scheduler(self):
        """schedule_batch with scheduler creates AsyncTrial objects."""
        cfg = InfrastructureConfig(n_workers=2)
        stack = InfrastructureStack(cfg)
        result = stack.schedule_batch([{"x": 1.0}, {"x": 2.0}])
        assert len(result) == 2
        # Should be AsyncTrial instances, not dicts
        assert hasattr(result[0], "trial_id")

    def test_needs_backfill_without_scheduler(self):
        """needs_backfill without scheduler returns False."""
        stack = InfrastructureStack()
        assert stack.needs_backfill() is False

    def test_backfill_count_without_scheduler(self):
        """backfill_count without scheduler returns 0."""
        stack = InfrastructureStack()
        assert stack.backfill_count() == 0

    def test_suggest_fidelity_without_manager(self):
        """suggest_fidelity without manager returns None."""
        stack = InfrastructureStack()
        assert stack.suggest_fidelity({"x": 1.0}) is None

    def test_suggest_fidelity_with_manager(self):
        """suggest_fidelity with manager returns fidelity dict."""
        cfg = InfrastructureConfig(fidelity_levels=_make_fidelity_level_dicts())
        stack = InfrastructureStack(cfg)
        result = stack.suggest_fidelity({"x": 1.0})
        assert result is not None
        assert "level" in result
        assert "name" in result
        assert "cost_multiplier" in result

    def test_analyze_importance_empty_returns_none(self):
        """analyze_importance with empty observations returns None."""
        stack = InfrastructureStack()
        result = stack.analyze_importance(
            observations=[],
            parameter_specs=[{"name": "x", "type": "continuous"}],
        )
        assert result is None

    def test_analyze_importance_with_data(self):
        """analyze_importance with valid data returns ImportanceResult."""
        stack = InfrastructureStack()
        observations = [
            {"x": 1.0, "y": 2.0, "objective": 0.5},
            {"x": 3.0, "y": 1.0, "objective": 0.8},
            {"x": 5.0, "y": 0.5, "objective": 0.9},
        ]
        specs = [
            {"name": "x", "type": "continuous"},
            {"name": "y", "type": "continuous"},
        ]
        result = stack.analyze_importance(observations, specs)
        assert result is not None
        assert "x" in result.scores
        assert "y" in result.scores

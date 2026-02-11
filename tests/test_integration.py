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

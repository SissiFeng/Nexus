"""Tests for JSON-file workspace persistence with atomic writes."""

import json
import threading
from pathlib import Path

import pytest

from optimization_copilot.platform.models import (
    ApiKey,
    CampaignRecord,
    CampaignStatus,
    Role,
    WorkspaceManifest,
)
from optimization_copilot.platform.workspace import (
    CampaignNotFoundError,
    Workspace,
    WorkspaceError,
)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    """Create and initialise a workspace in a temporary directory."""
    workspace = Workspace(tmp_path / "workspace")
    workspace.init()
    return workspace


def _make_spec() -> dict:
    return {
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }


def _make_record(cid: str = "c-001", name: str = "Test") -> CampaignRecord:
    return CampaignRecord(
        campaign_id=cid,
        name=name,
        status=CampaignStatus.DRAFT,
        spec=_make_spec(),
        created_at=1700000000.0,
        updated_at=1700000000.0,
    )


# ── Workspace Init ─────────────────────────────────────────────────


class TestWorkspaceInit:
    def test_init_creates_directories(self, tmp_path: Path):
        ws = Workspace(tmp_path / "new_ws")
        ws.init()
        assert (ws.root / "auth").is_dir()
        assert (ws.root / "campaigns").is_dir()
        assert (ws.root / "meta_learning").is_dir()
        assert (ws.root / "rag").is_dir()
        assert (ws.root / "manifest.json").is_file()

    def test_init_is_idempotent(self, tmp_path: Path):
        ws = Workspace(tmp_path / "idempotent_ws")
        m1 = ws.init()
        m2 = ws.init()
        # workspace_id should not change on second init
        assert m1.workspace_id == m2.workspace_id

    def test_manifest_returns_workspace_manifest(self, ws: Workspace):
        m = ws.manifest()
        assert isinstance(m, WorkspaceManifest)
        assert m.version == "1.0.0"

    def test_manifest_raises_when_uninitialised(self, tmp_path: Path):
        ws_uninit = Workspace(tmp_path / "empty_ws")
        with pytest.raises(WorkspaceError, match="not initialized"):
            ws_uninit.manifest()


# ── Campaign CRUD ──────────────────────────────────────────────────


class TestCampaignCRUD:
    def test_save_and_load_campaign_round_trip(self, ws: Workspace):
        record = _make_record()
        ws.save_campaign(record)
        loaded = ws.load_campaign("c-001")
        assert loaded.campaign_id == record.campaign_id
        assert loaded.name == record.name
        assert loaded.status is CampaignStatus.DRAFT

    def test_list_campaigns_returns_all(self, ws: Workspace):
        ws.save_campaign(_make_record("c-001", "First"))
        ws.save_campaign(_make_record("c-002", "Second"))
        records = ws.list_campaigns()
        ids = {r.campaign_id for r in records}
        assert ids == {"c-001", "c-002"}

    def test_delete_campaign_removes_directory(self, ws: Workspace):
        ws.save_campaign(_make_record("c-del"))
        ws.delete_campaign("c-del")
        assert not ws.campaign_exists("c-del")

    def test_campaign_exists_true(self, ws: Workspace):
        ws.save_campaign(_make_record("c-exist"))
        assert ws.campaign_exists("c-exist") is True

    def test_campaign_exists_false(self, ws: Workspace):
        assert ws.campaign_exists("nonexistent") is False

    def test_load_missing_campaign_raises(self, ws: Workspace):
        with pytest.raises(CampaignNotFoundError):
            ws.load_campaign("no-such-campaign")

    def test_delete_removes_from_manifest(self, ws: Workspace):
        ws.save_campaign(_make_record("c-m"))
        ws.delete_campaign("c-m")
        m = ws.manifest()
        assert "c-m" not in m.campaigns

    def test_multiple_campaigns_coexist(self, ws: Workspace):
        for i in range(5):
            ws.save_campaign(_make_record(f"c-{i}", f"Campaign {i}"))
        records = ws.list_campaigns()
        assert len(records) == 5


# ── Campaign Artifacts ─────────────────────────────────────────────


class TestCampaignArtifacts:
    def test_save_and_load_spec(self, ws: Workspace):
        ws.save_campaign(_make_record("c-spec"))
        spec = _make_spec()
        ws.save_spec("c-spec", spec)
        loaded = ws.load_spec("c-spec")
        assert loaded == spec

    def test_load_spec_raises_when_missing(self, ws: Workspace):
        with pytest.raises(CampaignNotFoundError):
            ws.load_spec("no-spec-campaign")

    def test_save_and_load_checkpoint(self, ws: Workspace):
        ws.save_campaign(_make_record("c-ckpt"))
        checkpoint = {"iteration": 5, "state": [1, 2, 3]}
        ws.save_checkpoint("c-ckpt", checkpoint)
        loaded = ws.load_checkpoint("c-ckpt")
        assert loaded == checkpoint

    def test_load_checkpoint_returns_none_when_missing(self, ws: Workspace):
        ws.save_campaign(_make_record("c-no-ckpt"))
        assert ws.load_checkpoint("c-no-ckpt") is None

    def test_save_and_load_result(self, ws: Workspace):
        ws.save_campaign(_make_record("c-res"))
        result = {"best_kpi_values": {"y": 0.42}, "trials": 100}
        ws.save_result("c-res", result)
        loaded = ws.load_result("c-res")
        assert loaded == result

    def test_load_result_returns_none_when_missing(self, ws: Workspace):
        ws.save_campaign(_make_record("c-no-res"))
        assert ws.load_result("c-no-res") is None

    def test_save_and_load_store(self, ws: Workspace):
        ws.save_campaign(_make_record("c-store"))
        store = {"observations": [{"x": 1}]}
        ws.save_store("c-store", store)
        loaded = ws.load_store("c-store")
        assert loaded == store

    def test_save_and_load_audit(self, ws: Workspace):
        ws.save_campaign(_make_record("c-audit"))
        audit = {"decisions": ["chose strategy A"]}
        ws.save_audit("c-audit", audit)
        loaded = ws.load_audit("c-audit")
        assert loaded == audit


# ── Meta-learning ──────────────────────────────────────────────────


class TestMetaLearning:
    def test_save_and_load_advisor(self, ws: Workspace):
        advisor = {"model_version": "v1", "weights": [0.1, 0.2]}
        ws.save_advisor(advisor)
        loaded = ws.load_advisor()
        assert loaded == advisor

    def test_load_advisor_returns_none_when_missing(self, tmp_path: Path):
        ws_new = Workspace(tmp_path / "fresh_ws")
        ws_new.init()
        assert ws_new.load_advisor() is None


# ── Auth Keys ──────────────────────────────────────────────────────


class TestAuthKeys:
    def test_save_and_load_keys_round_trip(self, ws: Workspace):
        keys = [
            ApiKey(key_hash="hash1", name="key1", role=Role.ADMIN, created_at=1.0),
            ApiKey(key_hash="hash2", name="key2", role=Role.VIEWER, created_at=2.0),
        ]
        ws.save_keys(keys)
        loaded = ws.load_keys()
        assert len(loaded) == 2
        assert loaded[0].key_hash == "hash1"
        assert loaded[1].role is Role.VIEWER

    def test_load_keys_returns_empty_when_missing(self, tmp_path: Path):
        ws_new = Workspace(tmp_path / "no_keys_ws")
        ws_new.init()
        assert ws_new.load_keys() == []


# ── RAG Index ──────────────────────────────────────────────────────


class TestRagIndex:
    def test_save_and_load_rag_index(self, ws: Workspace):
        index = {"entries": [{"id": "doc1", "vector": [0.1, 0.2]}]}
        ws.save_rag_index(index)
        loaded = ws.load_rag_index()
        assert loaded == index

    def test_load_rag_index_returns_none_when_missing(self, tmp_path: Path):
        ws_new = Workspace(tmp_path / "no_rag_ws")
        ws_new.init()
        assert ws_new.load_rag_index() is None


# ── Atomic Write ───────────────────────────────────────────────────


class TestAtomicWrite:
    def test_atomic_write_creates_file(self, ws: Workspace):
        target = ws.root / "test_atomic.json"
        ws._atomic_write(target, {"hello": "world"})
        assert target.is_file()
        with open(target) as f:
            data = json.load(f)
        assert data == {"hello": "world"}

    def test_atomic_write_no_tmp_left(self, ws: Workspace):
        target = ws.root / "no_tmp.json"
        ws._atomic_write(target, {"ok": True})
        tmp = target.with_suffix(".tmp")
        assert not tmp.exists()


# ── Thread Safety ──────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_saves_do_not_corrupt(self, ws: Workspace):
        """Basic thread safety: save many campaigns from multiple threads."""
        errors = []

        def save_campaign(index: int):
            try:
                record = _make_record(f"c-thread-{index}", f"Thread {index}")
                ws.save_campaign(record)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=save_campaign, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        records = ws.list_campaigns()
        ids = {r.campaign_id for r in records}
        expected = {f"c-thread-{i}" for i in range(10)}
        assert ids == expected

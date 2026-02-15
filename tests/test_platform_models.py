"""Tests for platform data models: CampaignRecord, ApiKey, WorkspaceManifest, SearchResult, CompareReport, enums."""

import pytest

from optimization_copilot.platform.models import (
    CampaignRecord,
    CampaignStatus,
    ApiKey,
    CompareReport,
    Role,
    ROLE_HIERARCHY,
    SearchResult,
    VALID_TRANSITIONS,
    WorkspaceManifest,
)


# ── CampaignStatus enum ───────────────────────────────────────────


class TestCampaignStatus:
    def test_all_values_present(self):
        expected = {"draft", "running", "paused", "completed", "stopped", "failed", "archived"}
        actual = {s.value for s in CampaignStatus}
        assert actual == expected

    def test_string_mixin(self):
        assert str(CampaignStatus.DRAFT) == "CampaignStatus.DRAFT"
        assert CampaignStatus.DRAFT.value == "draft"

    def test_from_string(self):
        assert CampaignStatus("running") is CampaignStatus.RUNNING


# ── Role enum ──────────────────────────────────────────────────────


class TestRole:
    def test_all_values_present(self):
        expected = {"viewer", "operator", "admin"}
        actual = {r.value for r in Role}
        assert actual == expected

    def test_from_string(self):
        assert Role("admin") is Role.ADMIN


# ── ROLE_HIERARCHY ─────────────────────────────────────────────────


class TestRoleHierarchy:
    def test_admin_gt_operator(self):
        assert ROLE_HIERARCHY[Role.ADMIN] > ROLE_HIERARCHY[Role.OPERATOR]

    def test_operator_gt_viewer(self):
        assert ROLE_HIERARCHY[Role.OPERATOR] > ROLE_HIERARCHY[Role.VIEWER]

    def test_admin_gt_viewer(self):
        assert ROLE_HIERARCHY[Role.ADMIN] > ROLE_HIERARCHY[Role.VIEWER]

    def test_all_roles_in_hierarchy(self):
        for role in Role:
            assert role in ROLE_HIERARCHY


# ── VALID_TRANSITIONS ──────────────────────────────────────────────


class TestValidTransitions:
    def test_draft_can_go_to_running(self):
        assert CampaignStatus.RUNNING in VALID_TRANSITIONS[CampaignStatus.DRAFT]

    def test_draft_can_go_to_archived(self):
        assert CampaignStatus.ARCHIVED in VALID_TRANSITIONS[CampaignStatus.DRAFT]

    def test_running_has_multiple_targets(self):
        targets = VALID_TRANSITIONS[CampaignStatus.RUNNING]
        assert CampaignStatus.PAUSED in targets
        assert CampaignStatus.COMPLETED in targets
        assert CampaignStatus.STOPPED in targets
        assert CampaignStatus.FAILED in targets
        assert CampaignStatus.ARCHIVED in targets

    def test_paused_can_resume(self):
        assert CampaignStatus.RUNNING in VALID_TRANSITIONS[CampaignStatus.PAUSED]

    def test_archived_can_unarchive(self):
        assert VALID_TRANSITIONS[CampaignStatus.ARCHIVED] == {CampaignStatus.DRAFT}

    def test_all_statuses_have_transitions(self):
        for status in CampaignStatus:
            assert status in VALID_TRANSITIONS


# ── CampaignRecord ─────────────────────────────────────────────────


class TestCampaignRecord:
    def _make_record(self, **overrides):
        defaults = dict(
            campaign_id="test-001",
            name="Test Campaign",
            status=CampaignStatus.DRAFT,
            spec={"parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}]},
            created_at=1700000000.0,
            updated_at=1700000000.0,
        )
        defaults.update(overrides)
        return CampaignRecord(**defaults)

    def test_creation_with_required_fields(self):
        record = self._make_record()
        assert record.campaign_id == "test-001"
        assert record.name == "Test Campaign"
        assert record.status is CampaignStatus.DRAFT

    def test_default_values(self):
        record = self._make_record()
        assert record.iteration == 0
        assert record.best_kpi is None
        assert record.total_trials == 0
        assert record.error_message is None
        assert record.tags == []
        assert record.metadata == {}

    def test_all_fields(self):
        record = self._make_record(
            iteration=5,
            best_kpi=0.42,
            total_trials=50,
            error_message="boom",
            tags=["tag1", "tag2"],
            metadata={"key": "val"},
        )
        assert record.iteration == 5
        assert record.best_kpi == 0.42
        assert record.total_trials == 50
        assert record.error_message == "boom"
        assert record.tags == ["tag1", "tag2"]
        assert record.metadata == {"key": "val"}

    def test_to_dict(self):
        record = self._make_record(tags=["a"])
        d = record.to_dict()
        assert d["campaign_id"] == "test-001"
        assert d["status"] == "draft"
        assert d["tags"] == ["a"]
        assert isinstance(d["spec"], dict)

    def test_from_dict_round_trip(self):
        original = self._make_record(
            iteration=3,
            best_kpi=1.5,
            total_trials=30,
            tags=["prod"],
            metadata={"run": 1},
        )
        d = original.to_dict()
        restored = CampaignRecord.from_dict(d)
        assert restored.campaign_id == original.campaign_id
        assert restored.status is original.status
        assert restored.iteration == original.iteration
        assert restored.best_kpi == original.best_kpi
        assert restored.total_trials == original.total_trials
        assert restored.tags == original.tags
        assert restored.metadata == original.metadata

    def test_from_dict_with_missing_optional_fields(self):
        minimal = {
            "campaign_id": "c1",
            "name": "Minimal",
            "status": "running",
            "spec": {},
            "created_at": 1.0,
            "updated_at": 2.0,
        }
        record = CampaignRecord.from_dict(minimal)
        assert record.iteration == 0
        assert record.best_kpi is None
        assert record.tags == []
        assert record.metadata == {}


# ── ApiKey ─────────────────────────────────────────────────────────


class TestApiKey:
    def _make_key(self, **overrides):
        defaults = dict(
            key_hash="abc123hash",
            name="test-key",
            role=Role.OPERATOR,
            created_at=1700000000.0,
        )
        defaults.update(overrides)
        return ApiKey(**defaults)

    def test_creation(self):
        key = self._make_key()
        assert key.key_hash == "abc123hash"
        assert key.name == "test-key"
        assert key.role is Role.OPERATOR

    def test_default_active_flag(self):
        key = self._make_key()
        assert key.active is True

    def test_inactive_key(self):
        key = self._make_key(active=False)
        assert key.active is False

    def test_to_dict(self):
        key = self._make_key()
        d = key.to_dict()
        assert d["role"] == "operator"
        assert d["active"] is True
        assert d["last_used"] is None

    def test_from_dict_round_trip(self):
        original = self._make_key(last_used=1700001000.0)
        d = original.to_dict()
        restored = ApiKey.from_dict(d)
        assert restored.key_hash == original.key_hash
        assert restored.role is original.role
        assert restored.last_used == original.last_used
        assert restored.active == original.active


# ── WorkspaceManifest ──────────────────────────────────────────────


class TestWorkspaceManifest:
    def test_creation(self):
        m = WorkspaceManifest(workspace_id="ws-1", created_at=1.0)
        assert m.workspace_id == "ws-1"
        assert m.version == "1.0.0"

    def test_default_campaigns_dict(self):
        m = WorkspaceManifest(workspace_id="ws-1", created_at=1.0)
        assert m.campaigns == {}

    def test_to_dict(self):
        m = WorkspaceManifest(workspace_id="ws-1", created_at=1.0, campaigns={"c1": "name1"})
        d = m.to_dict()
        assert d["workspace_id"] == "ws-1"
        assert d["campaigns"] == {"c1": "name1"}

    def test_from_dict_round_trip(self):
        original = WorkspaceManifest(
            workspace_id="ws-2",
            created_at=2.0,
            version="2.0.0",
            campaigns={"c1": "Campaign One"},
        )
        d = original.to_dict()
        restored = WorkspaceManifest.from_dict(d)
        assert restored.workspace_id == original.workspace_id
        assert restored.version == original.version
        assert restored.campaigns == original.campaigns


# ── SearchResult ───────────────────────────────────────────────────


class TestSearchResult:
    def test_creation(self):
        sr = SearchResult(campaign_id="c1", field="name", snippet="hello", score=0.95)
        assert sr.campaign_id == "c1"
        assert sr.score == 0.95

    def test_to_dict(self):
        sr = SearchResult(campaign_id="c1", field="spec", snippet="data", score=0.5)
        d = sr.to_dict()
        assert d == {"campaign_id": "c1", "field": "spec", "snippet": "data", "score": 0.5}


# ── CompareReport ──────────────────────────────────────────────────


class TestCompareReport:
    def _make_record(self, cid, kpi=None):
        return CampaignRecord(
            campaign_id=cid,
            name=f"Campaign {cid}",
            status=CampaignStatus.COMPLETED,
            spec={"objectives": [{"name": "y", "direction": "minimize"}]},
            created_at=1.0,
            updated_at=2.0,
            best_kpi=kpi,
        )

    def test_creation(self):
        r1 = self._make_record("c1", kpi=0.5)
        r2 = self._make_record("c2", kpi=0.3)
        report = CompareReport(
            campaign_ids=["c1", "c2"],
            records=[r1, r2],
            kpi_comparison={"y": [0.5, 0.3]},
            iteration_comparison=[10, 20],
            winner="c2",
        )
        assert report.winner == "c2"
        assert len(report.records) == 2

    def test_to_dict(self):
        r1 = self._make_record("c1")
        report = CompareReport(
            campaign_ids=["c1"],
            records=[r1],
            kpi_comparison={"y": [None]},
            iteration_comparison=[0],
        )
        d = report.to_dict()
        assert d["campaign_ids"] == ["c1"]
        assert len(d["records"]) == 1
        assert d["winner"] is None

    def test_from_dict_with_nested_campaign_records(self):
        r1 = self._make_record("c1", kpi=1.0)
        r2 = self._make_record("c2", kpi=2.0)
        original = CompareReport(
            campaign_ids=["c1", "c2"],
            records=[r1, r2],
            kpi_comparison={"y": [1.0, 2.0]},
            iteration_comparison=[5, 10],
            winner="c1",
        )
        d = original.to_dict()
        restored = CompareReport.from_dict(d)
        assert restored.campaign_ids == original.campaign_ids
        assert len(restored.records) == 2
        assert restored.records[0].campaign_id == "c1"
        assert restored.records[1].campaign_id == "c2"
        assert restored.kpi_comparison == original.kpi_comparison
        assert restored.winner == "c1"

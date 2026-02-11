"""Tests for campaign lifecycle management with state machine transitions."""

from pathlib import Path

import pytest

from optimization_copilot.platform.campaign_manager import (
    CampaignManager,
    InvalidTransitionError,
)
from optimization_copilot.platform.models import CampaignRecord, CampaignStatus
from optimization_copilot.platform.workspace import CampaignNotFoundError, Workspace


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    workspace = Workspace(tmp_path / "cm_ws")
    workspace.init()
    return workspace


@pytest.fixture
def cm(ws: Workspace) -> CampaignManager:
    return CampaignManager(ws)


def _simple_spec() -> dict:
    return {
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }


# ── Create ─────────────────────────────────────────────────────────


class TestCreate:
    def test_create_returns_draft_record(self, cm: CampaignManager):
        record = cm.create(_simple_spec())
        assert isinstance(record, CampaignRecord)
        assert record.status is CampaignStatus.DRAFT

    def test_create_persists_to_workspace(self, cm: CampaignManager, ws: Workspace):
        record = cm.create(_simple_spec())
        loaded = ws.load_campaign(record.campaign_id)
        assert loaded.campaign_id == record.campaign_id

    def test_create_with_custom_name_and_tags(self, cm: CampaignManager):
        record = cm.create(_simple_spec(), name="Custom Name", tags=["prod", "v2"])
        assert record.name == "Custom Name"
        assert record.tags == ["prod", "v2"]

    def test_create_with_explicit_campaign_id(self, cm: CampaignManager):
        record = cm.create(_simple_spec(), campaign_id="explicit-id")
        assert record.campaign_id == "explicit-id"

    def test_create_auto_generates_name_from_id(self, cm: CampaignManager):
        record = cm.create(_simple_spec(), campaign_id="abcd1234-rest")
        assert record.name.startswith("Campaign abcd1234")

    def test_create_saves_spec(self, cm: CampaignManager, ws: Workspace):
        spec = _simple_spec()
        record = cm.create(spec)
        loaded_spec = ws.load_spec(record.campaign_id)
        assert loaded_spec == spec


# ── Get ────────────────────────────────────────────────────────────


class TestGet:
    def test_get_returns_existing_campaign(self, cm: CampaignManager):
        created = cm.create(_simple_spec())
        fetched = cm.get(created.campaign_id)
        assert fetched.campaign_id == created.campaign_id

    def test_get_raises_for_missing_campaign(self, cm: CampaignManager):
        with pytest.raises(CampaignNotFoundError):
            cm.get("nonexistent-campaign-id")


# ── List ───────────────────────────────────────────────────────────


class TestListAll:
    def test_list_all_returns_all_campaigns(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="c1")
        cm.create(_simple_spec(), campaign_id="c2")
        cm.create(_simple_spec(), campaign_id="c3")
        result = cm.list_all()
        assert len(result) == 3

    def test_list_all_filters_by_status(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="draft-1")
        cm.create(_simple_spec(), campaign_id="draft-2")

        # Transition one to RUNNING
        cm.mark_running("draft-1")

        drafts = cm.list_all(status=CampaignStatus.DRAFT)
        running = cm.list_all(status=CampaignStatus.RUNNING)
        assert len(drafts) == 1
        assert drafts[0].campaign_id == "draft-2"
        assert len(running) == 1
        assert running[0].campaign_id == "draft-1"


# ── Delete ─────────────────────────────────────────────────────────


class TestDelete:
    def test_delete_sets_status_to_archived(self, cm: CampaignManager):
        record = cm.create(_simple_spec(), campaign_id="to-delete")
        cm.delete("to-delete")
        fetched = cm.get("to-delete")
        assert fetched.status is CampaignStatus.ARCHIVED

    def test_delete_raises_for_missing_campaign(self, cm: CampaignManager):
        with pytest.raises(CampaignNotFoundError):
            cm.delete("ghost-campaign")


# ── Update Tags ────────────────────────────────────────────────────


class TestUpdateTags:
    def test_update_tags_changes_tags(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="tagged", tags=["old"])
        updated = cm.update_tags("tagged", ["new", "fresh"])
        assert updated.tags == ["new", "fresh"]

        # Verify persistence
        fetched = cm.get("tagged")
        assert fetched.tags == ["new", "fresh"]


# ── State Transitions ──────────────────────────────────────────────


class TestStateTransitions:
    def test_mark_running_from_draft(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="st-run")
        record = cm.mark_running("st-run")
        assert record.status is CampaignStatus.RUNNING

    def test_mark_paused_from_running(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="st-pause")
        cm.mark_running("st-pause")
        record = cm.mark_paused("st-pause")
        assert record.status is CampaignStatus.PAUSED

    def test_mark_completed_from_running(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="st-complete")
        cm.mark_running("st-complete")
        record = cm.mark_completed("st-complete", best_kpi=0.42)
        assert record.status is CampaignStatus.COMPLETED
        assert record.best_kpi == 0.42

    def test_mark_stopped_from_running(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="st-stop")
        cm.mark_running("st-stop")
        record = cm.mark_stopped("st-stop")
        assert record.status is CampaignStatus.STOPPED

    def test_mark_failed_from_running(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="st-fail")
        cm.mark_running("st-fail")
        record = cm.mark_failed("st-fail", error="OOM")
        assert record.status is CampaignStatus.FAILED
        assert record.error_message == "OOM"

    def test_invalid_transition_raises(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="st-invalid")
        # DRAFT -> COMPLETED is not a valid transition
        with pytest.raises(InvalidTransitionError):
            cm.mark_completed("st-invalid")

    def test_invalid_transition_paused_to_completed(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="st-p2c")
        cm.mark_running("st-p2c")
        cm.mark_paused("st-p2c")
        # PAUSED -> COMPLETED is not valid
        with pytest.raises(InvalidTransitionError):
            cm.mark_completed("st-p2c")

    def test_archived_is_terminal(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="st-arch")
        cm.delete("st-arch")  # transitions to ARCHIVED
        with pytest.raises(InvalidTransitionError):
            cm.mark_running("st-arch")

    def test_transition_updates_updated_at(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="st-ts")
        before = cm.get("st-ts")
        original_updated = before.updated_at

        cm.mark_running("st-ts")
        after = cm.get("st-ts")
        assert after.updated_at > original_updated


# ── Progress Updates ───────────────────────────────────────────────


class TestUpdateProgress:
    def test_update_progress_sets_fields(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="prog")
        cm.mark_running("prog")
        record = cm.update_progress("prog", iteration=10, total_trials=100, best_kpi=0.05)
        assert record.iteration == 10
        assert record.total_trials == 100
        assert record.best_kpi == 0.05

    def test_update_progress_without_kpi(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="prog-no-kpi")
        cm.mark_running("prog-no-kpi")
        record = cm.update_progress("prog-no-kpi", iteration=3, total_trials=30)
        assert record.iteration == 3
        assert record.best_kpi is None


# ── Compare ────────────────────────────────────────────────────────


class TestCompare:
    def test_compare_returns_compare_report(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="cmp-a")
        cm.create(_simple_spec(), campaign_id="cmp-b")
        report = cm.compare(["cmp-a", "cmp-b"])
        assert report.campaign_ids == ["cmp-a", "cmp-b"]
        assert len(report.records) == 2

    def test_compare_determines_winner_by_best_kpi(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="w-a")
        cm.create(_simple_spec(), campaign_id="w-b")
        cm.mark_running("w-a")
        cm.mark_running("w-b")
        cm.update_progress("w-a", iteration=10, total_trials=100, best_kpi=0.5)
        cm.update_progress("w-b", iteration=10, total_trials=100, best_kpi=0.1)

        report = cm.compare(["w-a", "w-b"])
        # _determine_winner picks the campaign with the lowest best_kpi
        assert report.winner == "w-b"

    def test_compare_winner_is_none_when_no_kpi(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="nk-a")
        cm.create(_simple_spec(), campaign_id="nk-b")
        report = cm.compare(["nk-a", "nk-b"])
        assert report.winner is None

    def test_compare_iteration_comparison(self, cm: CampaignManager):
        cm.create(_simple_spec(), campaign_id="it-a")
        cm.create(_simple_spec(), campaign_id="it-b")
        cm.mark_running("it-a")
        cm.mark_running("it-b")
        cm.update_progress("it-a", iteration=5, total_trials=50)
        cm.update_progress("it-b", iteration=15, total_trials=150)

        report = cm.compare(["it-a", "it-b"])
        assert report.iteration_comparison == [5, 15]

    def test_compare_kpi_comparison_populated(self, cm: CampaignManager, ws: Workspace):
        spec = _simple_spec()
        cm.create(spec, campaign_id="kpi-a")
        cm.create(spec, campaign_id="kpi-b")
        cm.mark_running("kpi-a")
        cm.mark_running("kpi-b")
        cm.update_progress("kpi-a", iteration=10, total_trials=100, best_kpi=0.3)
        cm.update_progress("kpi-b", iteration=10, total_trials=100, best_kpi=0.7)

        # Save result data for kpi-a so compare can use best_kpi_values
        ws.save_result("kpi-a", {"best_kpi_values": {"y": 0.3}})
        ws.save_result("kpi-b", {"best_kpi_values": {"y": 0.7}})

        report = cm.compare(["kpi-a", "kpi-b"])
        assert "y" in report.kpi_comparison
        assert report.kpi_comparison["y"] == [0.3, 0.7]

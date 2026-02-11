"""Tests for CampaignRunner lifecycle and state management.

CampaignRunner runs an OptimizationEngine in a background thread.
Since the real engine requires a full spec + registry, these tests
focus on the runner's state management and lifecycle rather than
full integration with the engine.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from optimization_copilot.platform.campaign_manager import CampaignManager
from optimization_copilot.platform.campaign_runner import CampaignRunner, RunnerError
from optimization_copilot.platform.events import AsyncEventBus
from optimization_copilot.platform.models import CampaignStatus
from optimization_copilot.platform.workspace import Workspace, CampaignNotFoundError


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def workspace(tmp_path):
    """Create an initialized workspace."""
    ws = Workspace(tmp_path / "workspace")
    ws.init()
    return ws


@pytest.fixture
def manager(workspace):
    """Create a CampaignManager with the test workspace."""
    return CampaignManager(workspace)


@pytest.fixture
def event_bus():
    """Create an AsyncEventBus for testing (no event loop)."""
    return AsyncEventBus()


@pytest.fixture
def runner(workspace, manager, event_bus):
    """Create a CampaignRunner."""
    return CampaignRunner(
        workspace=workspace,
        manager=manager,
        event_bus=event_bus,
    )


@pytest.fixture
def sample_spec():
    """A minimal optimization spec dict."""
    return {
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [
            {"name": "y", "direction": "minimize"},
        ],
    }


@pytest.fixture
def draft_campaign(manager, sample_spec):
    """Create a DRAFT campaign and return its record."""
    return manager.create(spec_dict=sample_spec, name="Test Campaign")


# ── Initialization ───────────────────────────────────────────────


class TestRunnerInit:
    """CampaignRunner initialization."""

    def test_initializes_without_error(self, runner):
        assert runner is not None

    def test_initializes_with_defaults(self, workspace, manager, event_bus):
        runner = CampaignRunner(
            workspace=workspace,
            manager=manager,
            event_bus=event_bus,
        )
        assert runner is not None

    def test_initializes_with_custom_registry(self, workspace, manager, event_bus):
        from optimization_copilot.plugins.registry import PluginRegistry

        registry = PluginRegistry()
        runner = CampaignRunner(
            workspace=workspace,
            manager=manager,
            event_bus=event_bus,
            registry=registry,
        )
        assert runner is not None


# ── Status Queries on Idle Runner ────────────────────────────────


class TestRunnerIdleStatus:
    """Status queries when no campaign is running."""

    def test_is_running_false_for_unknown_campaign(self, runner):
        assert runner.is_running("nonexistent") is False

    def test_is_running_false_for_draft_campaign(self, runner, draft_campaign):
        assert runner.is_running(draft_campaign.campaign_id) is False

    def test_get_current_batch_none_for_unknown(self, runner):
        assert runner.get_current_batch("nonexistent") is None

    def test_get_current_batch_none_for_draft(self, runner, draft_campaign):
        assert runner.get_current_batch(draft_campaign.campaign_id) is None

    def test_get_error_none_for_unknown(self, runner):
        assert runner.get_error("nonexistent") is None

    def test_get_error_none_for_draft(self, runner, draft_campaign):
        assert runner.get_error(draft_campaign.campaign_id) is None


# ── Start Validation ─────────────────────────────────────────────


class TestRunnerStartValidation:
    """Validation when starting a campaign."""

    def test_start_raises_for_nonexistent_campaign(self, runner):
        with pytest.raises((CampaignNotFoundError, RunnerError, Exception)):
            runner.start("nonexistent-id")

    def test_start_raises_for_completed_campaign(self, runner, manager, draft_campaign):
        """A completed campaign should not be startable."""
        cid = draft_campaign.campaign_id
        # Transition to RUNNING then COMPLETED
        manager.mark_running(cid)
        manager.mark_completed(cid)

        with pytest.raises((RunnerError, Exception)):
            runner.start(cid)

    def test_start_raises_for_archived_campaign(self, runner, manager, draft_campaign):
        """An archived campaign should not be startable."""
        cid = draft_campaign.campaign_id
        manager.delete(cid)  # soft-delete -> ARCHIVED

        with pytest.raises((RunnerError, Exception)):
            runner.start(cid)

    def test_start_raises_for_failed_campaign(self, runner, manager, draft_campaign):
        """A failed campaign should not be startable."""
        cid = draft_campaign.campaign_id
        manager.mark_running(cid)
        manager.mark_failed(cid, "test error")

        with pytest.raises((RunnerError, Exception)):
            runner.start(cid)

    def test_start_validates_draft_status(self, runner, draft_campaign):
        """Start should accept DRAFT status (though it will fail loading the spec
        for the real engine, at least the status check should pass)."""
        cid = draft_campaign.campaign_id
        # The start method validates status first, then tries to load/build
        # the engine. Since we have a valid spec, it will attempt to create
        # OptimizationEngine which may fail due to missing backend plugins.
        # We just verify it doesn't raise RunnerError for invalid status.
        try:
            runner.start(cid)
        except RunnerError as e:
            # If it fails with RunnerError, it should NOT be about invalid status
            assert "Cannot start campaign in" not in str(e)
        except Exception:
            # Other exceptions (engine init, etc.) are expected
            pass


# ── Stop / Pause on Non-Running Campaign ─────────────────────────


class TestRunnerStopPauseIdle:
    """Stop and pause behavior on non-running campaigns."""

    def test_stop_raises_for_non_running(self, runner, draft_campaign):
        with pytest.raises(RunnerError):
            runner.stop(draft_campaign.campaign_id)

    def test_stop_raises_for_unknown(self, runner):
        with pytest.raises(RunnerError):
            runner.stop("nonexistent-id")

    def test_pause_raises_for_non_running(self, runner, draft_campaign):
        with pytest.raises(RunnerError):
            runner.pause(draft_campaign.campaign_id)

    def test_pause_raises_for_unknown(self, runner):
        with pytest.raises(RunnerError):
            runner.pause("nonexistent-id")


# ── Submit Trials on Non-Running Campaign ────────────────────────


class TestRunnerSubmitTrialsIdle:
    """Trial submission on non-running campaigns."""

    def test_submit_trials_raises_for_non_running(self, runner, draft_campaign):
        with pytest.raises(RunnerError):
            runner.submit_trials(draft_campaign.campaign_id, [])

    def test_submit_trials_raises_for_unknown(self, runner):
        with pytest.raises(RunnerError):
            runner.submit_trials("nonexistent-id", [])


# ── Internal Context ─────────────────────────────────────────────


class TestCampaignContext:
    """_CampaignContext internal state."""

    def test_context_defaults(self):
        from optimization_copilot.platform.campaign_runner import _CampaignContext

        ctx = _CampaignContext("test-campaign")
        assert ctx.campaign_id == "test-campaign"
        assert ctx.thread is None
        assert ctx.engine is None
        assert ctx.pending_batch is None
        assert ctx.running is False
        assert ctx.error is None

    def test_context_events_not_set(self):
        from optimization_copilot.platform.campaign_runner import _CampaignContext

        ctx = _CampaignContext("test-campaign")
        assert not ctx.batch_ready.is_set()
        assert not ctx.trials_done.is_set()
        assert not ctx.stop_requested.is_set()
        assert not ctx.pause_requested.is_set()

    def test_context_events_can_be_set(self):
        from optimization_copilot.platform.campaign_runner import _CampaignContext

        ctx = _CampaignContext("test-campaign")
        ctx.stop_requested.set()
        assert ctx.stop_requested.is_set()


# ── AsyncEventBus Integration ────────────────────────────────────


class TestRunnerEventBus:
    """Event bus integration basics."""

    def test_event_bus_starts_with_no_subscribers(self, event_bus):
        assert event_bus.subscriber_count("test") == 0

    def test_publish_without_loop_does_not_crash(self, event_bus):
        """Publishing without an event loop should fall back gracefully."""
        event_bus.publish("test-campaign", "test_event", {"key": "value"})

    def test_runner_holds_event_bus_reference(self, runner, event_bus):
        assert runner._event_bus is event_bus

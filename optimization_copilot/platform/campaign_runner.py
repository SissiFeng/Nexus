"""Generator-to-async bridge for running OptimizationEngine campaigns.

Runs the synchronous engine.run() generator in a background thread,
using threading.Event pairs to synchronize with async FastAPI handlers.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from optimization_copilot.engine.engine import EngineConfig, OptimizationEngine
from optimization_copilot.engine.events import EngineEvent, EventPayload
from optimization_copilot.engine.trial import Trial, TrialBatch
from optimization_copilot.dsl.spec import OptimizationSpec
from optimization_copilot.plugins.registry import PluginRegistry
from optimization_copilot.platform.campaign_manager import CampaignManager
from optimization_copilot.platform.events import AsyncEventBus
from optimization_copilot.platform.models import CampaignStatus
from optimization_copilot.platform.workspace import Workspace

logger = logging.getLogger(__name__)


class RunnerError(Exception):
    """Campaign runner error."""


class _CampaignContext:
    """Internal state for a running campaign."""

    def __init__(self, campaign_id: str) -> None:
        self.campaign_id = campaign_id
        self.thread: threading.Thread | None = None
        self.engine: OptimizationEngine | None = None
        self.pending_batch: TrialBatch | None = None
        self.batch_ready = threading.Event()
        self.trials_done = threading.Event()
        self.stop_requested = threading.Event()
        self.pause_requested = threading.Event()
        self.running = False
        self.error: str | None = None


class CampaignRunner:
    """Bridges synchronous engine generator to async handlers."""

    def __init__(
        self,
        workspace: Workspace,
        manager: CampaignManager,
        event_bus: AsyncEventBus,
        registry: PluginRegistry | None = None,
        engine_config: EngineConfig | None = None,
    ) -> None:
        self._workspace = workspace
        self._manager = manager
        self._event_bus = event_bus
        self._registry = registry or PluginRegistry()
        self._engine_config = engine_config
        self._contexts: dict[str, _CampaignContext] = {}

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self, campaign_id: str) -> None:
        """Start a campaign (launches engine in background thread)."""
        if campaign_id in self._contexts and self._contexts[campaign_id].running:
            raise RunnerError(f"Campaign already running: {campaign_id}")

        # Validate campaign exists and is in valid state
        record = self._manager.get(campaign_id)
        if record.status not in (CampaignStatus.DRAFT, CampaignStatus.PAUSED):
            raise RunnerError(
                f"Cannot start campaign in {record.status.value} state"
            )

        # Load spec
        spec_dict = self._workspace.load_spec(campaign_id)
        spec = OptimizationSpec.from_dict(spec_dict)

        # Load checkpoint if resuming
        state = None
        checkpoint = self._workspace.load_checkpoint(campaign_id)
        if checkpoint is not None:
            from optimization_copilot.engine.state import CampaignState
            state = CampaignState.from_dict(checkpoint)

        # Create engine
        config = self._engine_config or EngineConfig()
        engine = OptimizationEngine(
            spec=spec,
            registry=self._registry,
            config=config,
            state=state,
        )

        # Create context
        ctx = _CampaignContext(campaign_id)
        ctx.engine = engine
        ctx.running = True
        self._contexts[campaign_id] = ctx

        # Mark running
        self._manager.mark_running(campaign_id)

        # Start engine thread
        ctx.thread = threading.Thread(
            target=self._run_engine_thread,
            args=(ctx,),
            daemon=True,
            name=f"engine-{campaign_id[:8]}",
        )
        ctx.thread.start()

        self._event_bus.publish(campaign_id, "campaign_started", {
            "campaign_id": campaign_id,
        })

    def stop(self, campaign_id: str) -> None:
        """Request graceful stop of a running campaign."""
        ctx = self._contexts.get(campaign_id)
        if ctx is None or not ctx.running:
            raise RunnerError(f"Campaign not running: {campaign_id}")

        ctx.stop_requested.set()
        # Unblock if waiting for trial results
        ctx.trials_done.set()

        if ctx.engine is not None:
            ctx.engine.stop(reason="api_stop_request")

    def pause(self, campaign_id: str) -> None:
        """Pause a running campaign (checkpoint and stop)."""
        ctx = self._contexts.get(campaign_id)
        if ctx is None or not ctx.running:
            raise RunnerError(f"Campaign not running: {campaign_id}")

        ctx.pause_requested.set()
        ctx.stop_requested.set()
        ctx.trials_done.set()

        if ctx.engine is not None:
            ctx.engine.stop(reason="api_pause_request")

    # ── Trial Submission ──────────────────────────────────────

    def submit_trials(
        self, campaign_id: str, results: list[dict[str, Any]]
    ) -> None:
        """Submit trial results for a pending batch."""
        ctx = self._contexts.get(campaign_id)
        if ctx is None or not ctx.running:
            raise RunnerError(f"Campaign not running: {campaign_id}")

        batch = ctx.pending_batch
        if batch is None:
            raise RunnerError(f"No pending batch for campaign: {campaign_id}")

        # Update trial states from results
        result_map = {r["trial_id"]: r for r in results}
        for trial in batch.trials:
            if trial.trial_id in result_map:
                r = result_map[trial.trial_id]
                if r.get("is_failure", False):
                    trial.fail(r.get("failure_reason", "external_failure"))
                else:
                    trial.complete(
                        kpi_values=r.get("kpi_values", {}),
                        metadata=r.get("metadata"),
                    )

        # Signal engine thread to continue
        ctx.trials_done.set()

    # ── Status ────────────────────────────────────────────────

    def is_running(self, campaign_id: str) -> bool:
        ctx = self._contexts.get(campaign_id)
        return ctx is not None and ctx.running

    def get_current_batch(self, campaign_id: str) -> TrialBatch | None:
        ctx = self._contexts.get(campaign_id)
        if ctx is None:
            return None
        return ctx.pending_batch

    def get_error(self, campaign_id: str) -> str | None:
        ctx = self._contexts.get(campaign_id)
        if ctx is None:
            return None
        return ctx.error

    # ── Engine Thread ─────────────────────────────────────────

    def _run_engine_thread(self, ctx: _CampaignContext) -> None:
        """Runs in background thread — executes engine generator."""
        campaign_id = ctx.campaign_id
        engine = ctx.engine
        assert engine is not None

        # Register event callbacks
        for event_type in EngineEvent:
            engine.on(
                event_type,
                lambda payload, cid=campaign_id: self._on_engine_event(cid, payload),
            )

        try:
            for batch in engine.run():
                if ctx.stop_requested.is_set():
                    break

                # Store batch for API access
                ctx.pending_batch = batch
                ctx.batch_ready.set()

                # Publish batch ready event
                self._event_bus.publish(campaign_id, "batch_ready", {
                    "batch_id": batch.batch_id,
                    "iteration": batch.iteration,
                    "n_trials": len(batch.trials),
                    "trials": [t.to_dict() for t in batch.trials],
                })

                # Wait for trial results
                ctx.trials_done.wait()
                ctx.trials_done.clear()

                if ctx.stop_requested.is_set():
                    break

                # Update progress
                record = self._manager.get(campaign_id)
                self._manager.update_progress(
                    campaign_id,
                    iteration=batch.iteration,
                    total_trials=record.total_trials + len(batch.trials),
                    best_kpi=record.best_kpi,
                )

            # Engine finished
            ctx.pending_batch = None

            if ctx.pause_requested.is_set():
                # Save checkpoint and mark paused
                state = engine.checkpoint()
                self._workspace.save_checkpoint(campaign_id, state.to_dict())
                self._manager.mark_paused(campaign_id)
                self._event_bus.publish(campaign_id, "campaign_paused", {
                    "campaign_id": campaign_id,
                })
            elif ctx.stop_requested.is_set():
                # Save result and mark stopped
                result = engine.result()
                self._workspace.save_result(campaign_id, result.to_dict())
                self._manager.mark_stopped(campaign_id)
                self._event_bus.publish(campaign_id, "campaign_stopped", {
                    "campaign_id": campaign_id,
                })
            else:
                # Natural completion
                result = engine.result()
                self._workspace.save_result(campaign_id, result.to_dict())
                best_kpi = None
                if result.best_kpi_values:
                    best_kpi = next(iter(result.best_kpi_values.values()), None)
                self._manager.mark_completed(campaign_id, best_kpi)
                self._event_bus.publish(campaign_id, "campaign_completed", {
                    "campaign_id": campaign_id,
                    "best_kpi_values": result.best_kpi_values,
                    "total_iterations": result.total_iterations,
                })

        except Exception as e:
            logger.exception(f"Engine error for campaign {campaign_id}")
            ctx.error = str(e)
            try:
                self._manager.mark_failed(campaign_id, str(e))
            except Exception:
                logger.exception(f"Failed to mark campaign as failed: {campaign_id}")
            self._event_bus.publish(campaign_id, "campaign_failed", {
                "campaign_id": campaign_id,
                "error": str(e),
            })
        finally:
            ctx.running = False
            ctx.batch_ready.clear()

    def _on_engine_event(
        self, campaign_id: str, payload: EventPayload
    ) -> None:
        """Forward engine events to the async event bus."""
        self._event_bus.publish(
            campaign_id,
            payload.event.value,
            {
                "iteration": payload.iteration,
                **payload.data,
            },
        )

"""Campaign lifecycle management with state machine transitions."""

from __future__ import annotations

import uuid
from time import time
from typing import Any

from optimization_copilot.platform.models import (
    CampaignRecord,
    CampaignStatus,
    CompareReport,
    VALID_TRANSITIONS,
)
from optimization_copilot.platform.workspace import (
    CampaignNotFoundError,
    Workspace,
)


class InvalidTransitionError(Exception):
    """Invalid campaign state transition."""


class CampaignManager:
    """Campaign lifecycle state machine."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    # ── CRUD ──────────────────────────────────────────────────

    def create(
        self,
        spec_dict: dict[str, Any],
        name: str = "",
        tags: list[str] | None = None,
        campaign_id: str | None = None,
    ) -> CampaignRecord:
        """Create a new campaign in DRAFT status."""
        cid = campaign_id or str(uuid.uuid4())
        now = time()

        record = CampaignRecord(
            campaign_id=cid,
            name=name or f"Campaign {cid[:8]}",
            status=CampaignStatus.DRAFT,
            spec=spec_dict,
            created_at=now,
            updated_at=now,
            tags=tags or [],
        )

        self._workspace.save_campaign(record)
        self._workspace.save_spec(cid, spec_dict)
        return record

    def get(self, campaign_id: str) -> CampaignRecord:
        """Get a campaign by ID."""
        return self._workspace.load_campaign(campaign_id)

    def list_all(
        self, status: CampaignStatus | None = None
    ) -> list[CampaignRecord]:
        """List campaigns, optionally filtered by status."""
        records = self._workspace.list_campaigns()
        if status is not None:
            records = [r for r in records if r.status == status]
        return records

    def delete(self, campaign_id: str) -> None:
        """Soft-delete: transition to ARCHIVED."""
        self._transition(campaign_id, CampaignStatus.ARCHIVED)

    def update_tags(
        self, campaign_id: str, tags: list[str]
    ) -> CampaignRecord:
        """Update campaign tags."""
        record = self._workspace.load_campaign(campaign_id)
        record.tags = tags
        record.updated_at = time()
        self._workspace.save_campaign(record)
        return record

    # ── State Transitions ─────────────────────────────────────

    def mark_running(self, campaign_id: str) -> CampaignRecord:
        return self._transition(campaign_id, CampaignStatus.RUNNING)

    def mark_paused(self, campaign_id: str) -> CampaignRecord:
        return self._transition(campaign_id, CampaignStatus.PAUSED)

    def mark_completed(
        self, campaign_id: str, best_kpi: float | None = None
    ) -> CampaignRecord:
        record = self._transition(campaign_id, CampaignStatus.COMPLETED)
        if best_kpi is not None:
            record.best_kpi = best_kpi
            record.updated_at = time()
            self._workspace.save_campaign(record)
        return record

    def mark_stopped(self, campaign_id: str) -> CampaignRecord:
        return self._transition(campaign_id, CampaignStatus.STOPPED)

    def mark_failed(self, campaign_id: str, error: str) -> CampaignRecord:
        record = self._transition(campaign_id, CampaignStatus.FAILED)
        record.error_message = error
        record.updated_at = time()
        self._workspace.save_campaign(record)
        return record

    # ── Progress Updates ──────────────────────────────────────

    def update_progress(
        self,
        campaign_id: str,
        iteration: int,
        total_trials: int,
        best_kpi: float | None = None,
    ) -> CampaignRecord:
        """Update campaign progress (called each iteration)."""
        record = self._workspace.load_campaign(campaign_id)
        record.iteration = iteration
        record.total_trials = total_trials
        if best_kpi is not None:
            record.best_kpi = best_kpi
        record.updated_at = time()
        self._workspace.save_campaign(record)
        return record

    # ── Comparison ────────────────────────────────────────────

    def compare(self, campaign_ids: list[str]) -> CompareReport:
        """Build a side-by-side comparison of multiple campaigns."""
        records = [self._workspace.load_campaign(cid) for cid in campaign_ids]

        # Collect KPI names from specs
        all_kpi_names: set[str] = set()
        for r in records:
            objectives = r.spec.get("objectives", [])
            for obj in objectives:
                name = obj.get("name", "")
                if name:
                    all_kpi_names.add(name)

        # Build KPI comparison
        kpi_comparison: dict[str, list[float | None]] = {}
        for kpi_name in sorted(all_kpi_names):
            values: list[float | None] = []
            for r in records:
                # Try to load result to get best KPI
                result = self._workspace.load_result(r.campaign_id)
                if result and kpi_name in result.get("best_kpi_values", {}):
                    values.append(result["best_kpi_values"][kpi_name])
                else:
                    values.append(r.best_kpi if kpi_name == self._primary_kpi(r) else None)
            kpi_comparison[kpi_name] = values

        # Determine winner by best primary KPI
        iteration_comparison = [r.iteration for r in records]
        winner = self._determine_winner(records)

        return CompareReport(
            campaign_ids=campaign_ids,
            records=records,
            kpi_comparison=kpi_comparison,
            iteration_comparison=iteration_comparison,
            winner=winner,
        )

    # ── Internal ──────────────────────────────────────────────

    def _transition(
        self, campaign_id: str, to_status: CampaignStatus
    ) -> CampaignRecord:
        """Execute a state transition with validation."""
        record = self._workspace.load_campaign(campaign_id)
        valid = VALID_TRANSITIONS.get(record.status, set())

        if to_status not in valid:
            raise InvalidTransitionError(
                f"Cannot transition from {record.status.value} to {to_status.value}"
            )

        record.status = to_status
        record.updated_at = time()
        self._workspace.save_campaign(record)
        return record

    @staticmethod
    def _primary_kpi(record: CampaignRecord) -> str | None:
        """Get the primary KPI name from a campaign's spec."""
        objectives = record.spec.get("objectives", [])
        for obj in objectives:
            if obj.get("is_primary", True):
                return obj.get("name")
        return objectives[0]["name"] if objectives else None

    @staticmethod
    def _determine_winner(records: list[CampaignRecord]) -> str | None:
        """Determine which campaign has the best primary KPI."""
        best_id = None
        best_value = None

        for r in records:
            if r.best_kpi is not None:
                if best_value is None or r.best_kpi < best_value:
                    best_value = r.best_kpi
                    best_id = r.campaign_id

        return best_id

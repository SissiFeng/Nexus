"""Compliance reports and campaign comparison routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from optimization_copilot.api.deps import get_manager, get_workspace
from optimization_copilot.api.schemas import (
    CampaignResponse,
    CompareRequest,
    CompareResponse,
)
from optimization_copilot.platform.workspace import CampaignNotFoundError

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{campaign_id}/audit")
def get_audit_log(campaign_id: str) -> dict:
    """Get audit log for a campaign."""
    workspace = get_workspace()
    audit_data = workspace.load_audit(campaign_id)
    if audit_data is None:
        raise HTTPException(status_code=404, detail="Audit log not found")
    return audit_data


@router.get("/{campaign_id}/compliance")
def get_compliance_report(campaign_id: str) -> dict:
    """Get compliance report for a campaign."""
    workspace = get_workspace()
    audit_data = workspace.load_audit(campaign_id)
    if audit_data is None:
        raise HTTPException(status_code=404, detail="Audit log not found")

    from optimization_copilot.compliance.audit import AuditLog, verify_chain

    audit_log = AuditLog.from_dict(audit_data)
    verification = verify_chain(audit_log)

    return {
        "campaign_id": campaign_id,
        "chain_verification": verification.to_dict(),
        "n_entries": audit_log.n_entries,
        "chain_intact": audit_log.chain_intact,
    }


@router.post("/compare", response_model=CompareResponse)
def compare_campaigns(req: CompareRequest) -> CompareResponse:
    """Compare multiple campaigns side by side."""
    try:
        manager = get_manager()
        report = manager.compare(req.campaign_ids)
    except CampaignNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return CompareResponse(
        campaign_ids=report.campaign_ids,
        records=[
            CampaignResponse(
                campaign_id=r.campaign_id,
                name=r.name,
                status=r.status.value,
                created_at=r.created_at,
                updated_at=r.updated_at,
                iteration=r.iteration,
                best_kpi=r.best_kpi,
                total_trials=r.total_trials,
                error_message=r.error_message,
                tags=r.tags,
            )
            for r in report.records
        ],
        kpi_comparison=report.kpi_comparison,
        iteration_comparison=report.iteration_comparison,
        winner=report.winner,
    )

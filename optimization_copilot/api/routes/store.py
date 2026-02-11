"""Experiment store query routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from optimization_copilot.api.deps import get_workspace
from optimization_copilot.api.schemas import StoreSummaryResponse

router = APIRouter(prefix="/store", tags=["store"])


@router.get("/{campaign_id}")
def query_store(
    campaign_id: str,
    iteration: int | None = Query(None),
    only_successful: bool = Query(False),
) -> dict:
    """Query experiment store for a campaign."""
    workspace = get_workspace()
    store_data = workspace.load_store(campaign_id)
    if store_data is None:
        raise HTTPException(status_code=404, detail="Store not found for campaign")

    # If ExperimentStore is loaded, we can filter
    from optimization_copilot.store.store import ExperimentStore, StoreQuery

    store = ExperimentStore.from_dict(store_data)

    query = StoreQuery(
        campaign_id=campaign_id,
        iteration=iteration,
        only_successful=only_successful,
    )
    experiments = store.query(query)
    return {
        "experiments": [e.to_dict() for e in experiments],
        "count": len(experiments),
    }


@router.get("/{campaign_id}/summary", response_model=StoreSummaryResponse)
def store_summary(campaign_id: str) -> StoreSummaryResponse:
    """Get store summary for a campaign."""
    workspace = get_workspace()
    store_data = workspace.load_store(campaign_id)
    if store_data is None:
        raise HTTPException(status_code=404, detail="Store not found for campaign")

    from optimization_copilot.store.store import ExperimentStore

    store = ExperimentStore.from_dict(store_data)
    summary = store.summary(campaign_id=campaign_id)

    return StoreSummaryResponse(
        n_experiments=summary.n_experiments,
        n_campaigns=summary.n_campaigns,
        campaign_ids=summary.campaign_ids,
        n_artifacts=summary.n_artifacts,
        parameter_names=summary.parameter_names,
        kpi_names=summary.kpi_names,
    )


@router.get("/{campaign_id}/export")
def export_store(campaign_id: str) -> dict:
    """Export full store as JSON."""
    workspace = get_workspace()
    store_data = workspace.load_store(campaign_id)
    if store_data is None:
        raise HTTPException(status_code=404, detail="Store not found for campaign")
    return store_data

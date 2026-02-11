"""Campaign CRUD and execution routes."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query

from optimization_copilot.api.deps import (
    get_manager,
    get_rag,
    get_runner,
    get_workspace,
)
from optimization_copilot.api.schemas import (
    BatchResponse,
    CampaignDetailResponse,
    CampaignListResponse,
    CampaignResponse,
    CreateCampaignRequest,
    SearchResponse,
    SearchResultItem,
    StatusResponse,
    SubmitTrialsRequest,
    TrialResponse,
)
from optimization_copilot.platform.campaign_manager import InvalidTransitionError
from optimization_copilot.platform.campaign_runner import RunnerError
from optimization_copilot.platform.models import CampaignRecord, CampaignStatus
from optimization_copilot.platform.workspace import CampaignNotFoundError

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


def _to_response(record: CampaignRecord) -> CampaignResponse:
    return CampaignResponse(
        campaign_id=record.campaign_id,
        name=record.name,
        status=record.status.value,
        created_at=record.created_at,
        updated_at=record.updated_at,
        iteration=record.iteration,
        best_kpi=record.best_kpi,
        total_trials=record.total_trials,
        error_message=record.error_message,
        tags=record.tags,
    )


@router.post("", response_model=CampaignDetailResponse, status_code=201)
def create_campaign(req: CreateCampaignRequest) -> CampaignDetailResponse:
    manager = get_manager()
    record = manager.create(spec_dict=req.spec, name=req.name, tags=req.tags)
    return CampaignDetailResponse(
        campaign_id=record.campaign_id,
        name=record.name,
        status=record.status.value,
        spec=record.spec,
        created_at=record.created_at,
        updated_at=record.updated_at,
        tags=record.tags,
        metadata=record.metadata,
    )


@router.get("", response_model=CampaignListResponse)
def list_campaigns(status: str | None = Query(None)) -> CampaignListResponse:
    manager = get_manager()
    filter_status = CampaignStatus(status) if status else None
    records = manager.list_all(status=filter_status)
    return CampaignListResponse(
        campaigns=[_to_response(r) for r in records],
        total=len(records),
    )


@router.get("/{campaign_id}", response_model=CampaignDetailResponse)
def get_campaign(campaign_id: str) -> CampaignDetailResponse:
    try:
        manager = get_manager()
        record = manager.get(campaign_id)
    except CampaignNotFoundError:
        raise HTTPException(status_code=404, detail=f"Campaign not found: {campaign_id}")
    return CampaignDetailResponse(
        campaign_id=record.campaign_id,
        name=record.name,
        status=record.status.value,
        spec=record.spec,
        created_at=record.created_at,
        updated_at=record.updated_at,
        iteration=record.iteration,
        best_kpi=record.best_kpi,
        total_trials=record.total_trials,
        error_message=record.error_message,
        tags=record.tags,
        metadata=record.metadata,
    )


@router.delete("/{campaign_id}", response_model=StatusResponse)
def delete_campaign(campaign_id: str) -> StatusResponse:
    try:
        manager = get_manager()
        manager.delete(campaign_id)
    except CampaignNotFoundError:
        raise HTTPException(status_code=404, detail=f"Campaign not found: {campaign_id}")
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return StatusResponse(status="archived", message=f"Campaign {campaign_id} archived")


@router.post("/{campaign_id}/start", response_model=StatusResponse)
def start_campaign(campaign_id: str) -> StatusResponse:
    try:
        runner = get_runner()
        runner.start(campaign_id)
    except CampaignNotFoundError:
        raise HTTPException(status_code=404, detail=f"Campaign not found: {campaign_id}")
    except (RunnerError, InvalidTransitionError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    return StatusResponse(status="running", message=f"Campaign {campaign_id} started")


@router.post("/{campaign_id}/stop", response_model=StatusResponse)
def stop_campaign(campaign_id: str) -> StatusResponse:
    try:
        runner = get_runner()
        runner.stop(campaign_id)
    except RunnerError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return StatusResponse(status="stopping", message=f"Campaign {campaign_id} stop requested")


@router.post("/{campaign_id}/pause", response_model=StatusResponse)
def pause_campaign(campaign_id: str) -> StatusResponse:
    try:
        runner = get_runner()
        runner.pause(campaign_id)
    except RunnerError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return StatusResponse(status="pausing", message=f"Campaign {campaign_id} pause requested")


@router.post("/{campaign_id}/resume", response_model=StatusResponse)
def resume_campaign(campaign_id: str) -> StatusResponse:
    try:
        runner = get_runner()
        runner.start(campaign_id)  # start handles resume from checkpoint
    except (RunnerError, InvalidTransitionError, CampaignNotFoundError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    return StatusResponse(status="running", message=f"Campaign {campaign_id} resumed")


@router.get("/{campaign_id}/batch", response_model=BatchResponse | None)
def get_current_batch(campaign_id: str) -> BatchResponse | None:
    runner = get_runner()
    batch = runner.get_current_batch(campaign_id)
    if batch is None:
        return None
    return BatchResponse(
        batch_id=batch.batch_id,
        iteration=batch.iteration,
        trials=[
            TrialResponse(
                trial_id=t.trial_id,
                iteration=t.iteration,
                parameters=t.parameters,
                state=t.state.value,
                kpi_values=t.kpi_values,
            )
            for t in batch.trials
        ],
    )


@router.post("/{campaign_id}/trials", response_model=StatusResponse)
def submit_trials(campaign_id: str, req: SubmitTrialsRequest) -> StatusResponse:
    try:
        runner = get_runner()
        runner.submit_trials(
            campaign_id,
            [r.model_dump() for r in req.results],
        )
    except RunnerError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return StatusResponse(status="accepted", message=f"Submitted {len(req.results)} trial results")


@router.get("/{campaign_id}/result")
def get_result(campaign_id: str) -> dict:
    workspace = get_workspace()
    result = workspace.load_result(campaign_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not available yet")
    return result


@router.get("/{campaign_id}/checkpoint")
def get_checkpoint(campaign_id: str) -> dict:
    workspace = get_workspace()
    checkpoint = workspace.load_checkpoint(campaign_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="No checkpoint available")
    return checkpoint


# ── Search ────────────────────────────────────────────────────

search_router = APIRouter(prefix="/search", tags=["search"])


@search_router.get("", response_model=SearchResponse)
def search_campaigns(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=100),
) -> SearchResponse:
    rag = get_rag()
    results = rag.search(q, top_k=top_k)
    return SearchResponse(
        results=[
            SearchResultItem(
                campaign_id=r.campaign_id,
                field=r.field,
                snippet=r.snippet,
                score=r.score,
            )
            for r in results
        ],
        total=len(results),
    )


# Add search router to campaigns router parent
router.include_router(search_router)

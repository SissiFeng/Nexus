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
    UpdateCampaignStatusRequest,
)
from optimization_copilot.platform.campaign_manager import InvalidTransitionError
from optimization_copilot.platform.campaign_runner import RunnerError
from optimization_copilot.platform.models import CampaignRecord, CampaignStatus
from optimization_copilot.platform.workspace import CampaignNotFoundError

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


def _to_response(record: CampaignRecord) -> CampaignResponse:
    objective_names = [
        obj.get("name", "")
        for obj in record.spec.get("objectives", [])
        if obj.get("name")
    ]
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
        objective_names=objective_names,
    )


@router.post("", response_model=CampaignDetailResponse, status_code=201)
def create_campaign(req: CreateCampaignRequest) -> CampaignDetailResponse:
    manager = get_manager()
    record = manager.create(spec_dict=req.spec, name=req.name, tags=req.tags)
    obj_names = [
        obj.get("name", "")
        for obj in record.spec.get("objectives", [])
        if obj.get("name")
    ]
    return CampaignDetailResponse(
        campaign_id=record.campaign_id,
        name=record.name,
        status=record.status.value,
        spec=record.spec,
        created_at=record.created_at,
        updated_at=record.updated_at,
        tags=record.tags,
        metadata=record.metadata,
        objective_names=obj_names,
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

    # Extract computed fields from workspace artifacts
    workspace = get_workspace()
    best_parameters = None
    phases: list[dict] = []
    kpi_history: dict = {"iterations": [], "values": []}
    observations: list[dict] = []

    # Try to extract best_parameters from result
    result = workspace.load_result(campaign_id)
    if result is not None:
        best_trial = result.get("best_trial")
        if best_trial is not None:
            best_parameters = best_trial.get("parameters")

    # Try to extract phases and kpi_history from checkpoint
    checkpoint = workspace.load_checkpoint(campaign_id)
    if checkpoint is not None:
        # Build phases from phase_history
        raw_phases = checkpoint.get("phase_history", [])
        for i, entry in enumerate(raw_phases):
            start_iter = entry.get("iteration", 0)
            # End is the start of the next phase, or the current iteration
            if i + 1 < len(raw_phases):
                end_iter = raw_phases[i + 1].get("iteration", start_iter)
            else:
                end_iter = checkpoint.get("iteration", start_iter)
            phases.append({
                "name": entry.get("to_phase", "unknown"),
                "start": start_iter,
                "end": end_iter,
            })

        # Build kpi_history and observations from completed_trials
        completed = checkpoint.get("completed_trials", [])
        iter_kpi_pairs: list[tuple[int, float]] = []
        for trial in completed:
            trial_iter = trial.get("iteration", 0)
            kpi_values = trial.get("kpi_values", {})
            trial_params = trial.get("parameters", {})
            if kpi_values:
                # Use the first KPI value as the primary metric
                primary_value = next(iter(kpi_values.values()), None)
                if primary_value is not None:
                    iter_kpi_pairs.append((trial_iter, primary_value))
                # Build full observation for multi-objective support
                observations.append({
                    "iteration": trial_iter,
                    "parameters": trial_params,
                    "kpi_values": kpi_values,
                })

        # Sort by iteration and build the history arrays
        iter_kpi_pairs.sort(key=lambda p: p[0])
        if iter_kpi_pairs:
            kpi_history = {
                "iterations": [p[0] for p in iter_kpi_pairs],
                "values": [p[1] for p in iter_kpi_pairs],
            }

    # Extract objective directions from spec
    objective_directions: dict[str, str] = {}
    spec_objectives = record.spec.get("objectives", [])
    for obj in spec_objectives:
        obj_name = obj.get("name", "")
        obj_dir = obj.get("direction", "minimize")
        if obj_name:
            objective_directions[obj_name] = obj_dir

    objective_names = [
        obj.get("name", "")
        for obj in spec_objectives
        if obj.get("name")
    ]

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
        best_parameters=best_parameters,
        phases=phases,
        kpi_history=kpi_history,
        observations=observations,
        objective_directions=objective_directions,
        objective_names=objective_names,
    )


@router.patch("/{campaign_id}", response_model=CampaignResponse)
def update_campaign_status(
    campaign_id: str, req: UpdateCampaignStatusRequest
) -> CampaignResponse:
    """Update a campaign's status (archive / unarchive)."""
    try:
        manager = get_manager()
        target_status = CampaignStatus(req.status)
        record = manager.get(campaign_id)

        if target_status == CampaignStatus.ARCHIVED:
            manager.delete(campaign_id)  # delete() does soft-archive
        elif target_status == CampaignStatus.DRAFT and record.status == CampaignStatus.ARCHIVED:
            manager._transition(campaign_id, CampaignStatus.DRAFT)
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot update status to '{req.status}' from '{record.status.value}'",
            )

        updated = manager.get(campaign_id)
        return _to_response(updated)
    except CampaignNotFoundError:
        raise HTTPException(status_code=404, detail=f"Campaign not found: {campaign_id}")
    except InvalidTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))


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


# ── Search (separate router, registered at /api/search) ──────

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

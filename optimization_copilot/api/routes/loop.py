"""CampaignLoop API endpoints for iterative optimization loops.

Provides stateful in-memory management of CampaignLoop instances.
Each loop maintains its own surrogate models, candidate pool, and
iteration history across multiple API calls.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from optimization_copilot.api.schemas import StatusResponse
from optimization_copilot.campaign.loop import CampaignLoop
from optimization_copilot.campaign.output import CampaignDeliverable
from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)

router = APIRouter(prefix="/loop", tags=["loop"])

# In-memory store for active loops, keyed by loop_id.
_loops: dict[str, CampaignLoop] = {}


# ── Request / Response Models ────────────────────────────────────


class CreateLoopRequest(BaseModel):
    """Request body for creating a new CampaignLoop."""

    campaign_id: str = "default"
    observations: list[dict[str, Any]] = Field(
        ...,
        description='Each has "parameters", "kpi_values", "iteration"',
    )
    candidates: list[dict[str, Any]] = Field(
        ...,
        description="Untested candidate parameter dicts",
    )
    parameter_specs: list[dict[str, Any]] = Field(
        ...,
        description='Each has "name", "type", optional "categories"',
    )
    objectives: list[str] = Field(
        ...,
        description="KPI names to optimize",
    )
    objective_directions: dict[str, str] = Field(
        ...,
        description='e.g. {"HER": "minimize"}',
    )
    smiles_param: str = "smiles"
    batch_size: int = 5
    acquisition_strategy: str = "ucb"


class IngestRequest(BaseModel):
    """Request body for ingesting new experimental results."""

    results: list[dict[str, Any]] = Field(
        ...,
        description='Each has "parameters", "kpi_values", "iteration"',
    )


class CreateLoopResponse(BaseModel):
    """Response after successfully creating a loop."""

    loop_id: str
    n_observations: int
    n_candidates: int


class LoopStatusResponse(BaseModel):
    """Current state of an active loop."""

    loop_id: str
    n_observations: int
    n_candidates: int
    objectives: list[str]
    iterations_run: int


# ── Helpers ──────────────────────────────────────────────────────


def _build_parameter_specs(raw_specs: list[dict[str, Any]]) -> list[ParameterSpec]:
    """Convert raw parameter spec dicts to ParameterSpec dataclasses."""
    specs: list[ParameterSpec] = []
    for raw in raw_specs:
        var_type = VariableType(raw["type"])
        specs.append(ParameterSpec(
            name=raw["name"],
            type=var_type,
            lower=raw.get("lower"),
            upper=raw.get("upper"),
            categories=raw.get("categories"),
        ))
    return specs


def _build_observations(raw_observations: list[dict[str, Any]]) -> list[Observation]:
    """Convert raw observation dicts to Observation dataclasses."""
    observations: list[Observation] = []
    for raw in raw_observations:
        observations.append(Observation(
            iteration=raw.get("iteration", 0),
            parameters=raw.get("parameters", {}),
            kpi_values=raw.get("kpi_values", {}),
            is_failure=raw.get("is_failure", False),
            failure_reason=raw.get("failure_reason"),
            metadata=raw.get("metadata", {}),
        ))
    return observations


def _get_loop(loop_id: str) -> CampaignLoop:
    """Retrieve a loop by ID or raise 404."""
    loop = _loops.get(loop_id)
    if loop is None:
        raise HTTPException(status_code=404, detail=f"Loop not found: {loop_id}")
    return loop


def _serialize_deliverable(deliverable: CampaignDeliverable) -> dict[str, Any]:
    """Serialize a CampaignDeliverable to a JSON-friendly dict.

    Uses the built-in ``to_dict()`` methods on each layer, then reshapes
    the output into a flatter structure suitable for API consumers.
    """
    result: dict[str, Any] = {
        "iteration": deliverable.iteration,
        "timestamp": deliverable.timestamp,
    }

    # Dashboard layer
    db = deliverable.dashboard
    ranked = []
    for candidate in db.ranked_table.candidates:
        ranked.append({
            "rank": candidate.rank,
            "name": candidate.name,
            "acquisition_score": candidate.acquisition_score,
            "predicted_mean": candidate.predicted_mean,
            "predicted_std": candidate.predicted_std,
            "parameters": candidate.parameters,
        })
    result["dashboard"] = {
        "ranked_candidates": ranked,
        "batch": ranked[: db.batch_size],
        "batch_size": db.batch_size,
        "objective_name": db.ranked_table.objective_name,
        "direction": db.ranked_table.direction,
        "acquisition_strategy": db.ranked_table.acquisition_strategy,
    }

    # Intelligence layer
    intel = deliverable.intelligence
    metrics_list = []
    for mm in intel.model_metrics:
        metrics_list.append({
            "objective_name": mm.objective_name,
            "n_training_points": mm.n_training_points,
            "y_mean": mm.y_mean,
            "y_std": mm.y_std,
            "fit_duration_ms": mm.fit_duration_ms,
        })
    intelligence: dict[str, Any] = {
        "model_metrics": metrics_list,
        "iteration_count": intel.iteration_count,
    }
    if intel.learning_report is not None:
        lr = intel.learning_report
        intelligence["learning_report"] = {
            "new_observations": lr.new_observations,
            "prediction_errors": lr.prediction_errors,
            "mean_absolute_error": lr.mean_absolute_error,
            "model_updated": lr.model_updated,
            "summary": lr.summary,
        }
    if intel.pareto_summary is not None:
        intelligence["pareto_summary"] = intel.pareto_summary
    result["intelligence"] = intelligence

    # Reasoning layer
    reasoning = deliverable.reasoning
    result["reasoning"] = {
        "diagnostic_summary": reasoning.diagnostic_summary,
        "fanova_result": reasoning.fanova_result,
        "execution_traces": reasoning.execution_traces,
    }

    return result


# ── Endpoints ────────────────────────────────────────────────────


@router.post("", response_model=CreateLoopResponse, status_code=201)
def create_loop(req: CreateLoopRequest) -> CreateLoopResponse:
    """Create a new CampaignLoop and store it in memory.

    Builds the snapshot from the provided observations and parameter specs,
    then initialises the loop with the candidate pool and surrogate settings.
    """
    try:
        parameter_specs = _build_parameter_specs(req.parameter_specs)
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameter_specs: {exc}",
        )

    observations = _build_observations(req.observations)

    # Convert objective_directions dict to ordered list matching objectives
    direction_list = [
        req.objective_directions.get(obj, "maximize")
        for obj in req.objectives
    ]

    snapshot = CampaignSnapshot(
        campaign_id=req.campaign_id,
        parameter_specs=parameter_specs,
        observations=observations,
        objective_names=list(req.objectives),
        objective_directions=direction_list,
    )

    try:
        loop = CampaignLoop(
            snapshot=snapshot,
            candidates=list(req.candidates),
            smiles_param=req.smiles_param,
            objectives=list(req.objectives),
            objective_directions=dict(req.objective_directions),
            batch_size=req.batch_size,
            acquisition_strategy=req.acquisition_strategy,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to create CampaignLoop: {exc}",
        )

    loop_id = str(uuid.uuid4())
    _loops[loop_id] = loop

    return CreateLoopResponse(
        loop_id=loop_id,
        n_observations=len(observations),
        n_candidates=len(req.candidates),
    )


@router.post("/{loop_id}/iterate")
def iterate_loop(loop_id: str) -> dict[str, Any]:
    """Run one iteration of the campaign loop.

    Fits surrogates on current observations, ranks candidates by acquisition
    score, and returns the full three-layer deliverable.
    """
    loop = _get_loop(loop_id)
    try:
        deliverable = loop.run_iteration()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Iteration failed: {exc}",
        )
    return _serialize_deliverable(deliverable)


@router.post("/{loop_id}/ingest")
def ingest_results(loop_id: str, req: IngestRequest) -> dict[str, Any]:
    """Ingest new experimental results and produce the next deliverable.

    Compares model predictions against actuals, updates the snapshot,
    removes tested candidates from the pool, and re-runs the loop.
    """
    loop = _get_loop(loop_id)

    new_observations = _build_observations(req.results)
    if not new_observations:
        raise HTTPException(status_code=400, detail="No results provided")

    try:
        deliverable = loop.ingest_results(new_observations)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ingest failed: {exc}",
        )
    return _serialize_deliverable(deliverable)


@router.get("/{loop_id}", response_model=LoopStatusResponse)
def get_loop_status(loop_id: str) -> LoopStatusResponse:
    """Return the current state of an active loop."""
    loop = _get_loop(loop_id)
    return LoopStatusResponse(
        loop_id=loop_id,
        n_observations=loop._snapshot.n_observations,
        n_candidates=loop.n_candidates_remaining,
        objectives=list(loop._objectives),
        iterations_run=loop.iteration,
    )


@router.delete("/{loop_id}", response_model=StatusResponse)
def delete_loop(loop_id: str) -> StatusResponse:
    """Remove a loop from the in-memory store."""
    if loop_id not in _loops:
        raise HTTPException(status_code=404, detail=f"Loop not found: {loop_id}")
    del _loops[loop_id]
    return StatusResponse(status="deleted", message=f"Loop {loop_id} deleted")

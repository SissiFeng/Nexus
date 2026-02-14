"""Frontend v2 API endpoints.

Provides campaign-level convenience endpoints for the scientist-facing
web UI: create from upload, diagnostics, parameter importance, suggestions,
chat, export, and steering directives.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from optimization_copilot.api.deps import get_manager, get_workspace
from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.platform.workspace import CampaignNotFoundError

router = APIRouter(tags=["frontend-v2"])
logger = logging.getLogger(__name__)


# ── Schemas ──────────────────────────────────────────────────────────


class ColumnMappingParam(BaseModel):
    name: str
    type: str  # "continuous" or "categorical"
    lower: float | None = None
    upper: float | None = None


class ColumnMappingObjective(BaseModel):
    name: str
    direction: str  # "minimize" or "maximize"


class ColumnMapping(BaseModel):
    parameters: list[ColumnMappingParam]
    objectives: list[ColumnMappingObjective]
    metadata: list[str] = Field(default_factory=list)
    ignored: list[str] = Field(default_factory=list)


class CreateFromUploadRequest(BaseModel):
    name: str = ""
    description: str = ""
    data: list[dict[str, str]]
    mapping: ColumnMapping
    batch_size: int = 5
    exploration_weight: float = 0.5


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    role: str = "agent"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiagnosticsResponse(BaseModel):
    convergence_trend: float = 0.0
    improvement_velocity: float = 0.0
    best_kpi_value: float = 0.0
    exploration_coverage: float = 0.0
    failure_rate: float = 0.0
    noise_estimate: float = 0.0
    plateau_length: int = 0
    signal_to_noise_ratio: float = 0.0


class ImportanceResponse(BaseModel):
    importances: list[dict[str, Any]]


class SuggestionResponse(BaseModel):
    suggestions: list[dict[str, Any]]
    predicted_values: list[float] = Field(default_factory=list)
    predicted_uncertainties: list[float] = Field(default_factory=list)
    backend_used: str = "builtin"
    phase: str = "exploitation"


class AppendRequest(BaseModel):
    data: list[dict[str, str]]


class SteerRequest(BaseModel):
    action: str
    region_bounds: dict[str, list[float]] | None = None
    reason: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────


def _load_snapshot(campaign_id: str) -> CampaignSnapshot:
    """Reconstruct a CampaignSnapshot from workspace artifacts."""
    workspace = get_workspace()

    if not workspace.campaign_exists(campaign_id):
        raise HTTPException(status_code=404, detail=f"Campaign not found: {campaign_id}")

    spec = workspace.load_spec(campaign_id)
    checkpoint = workspace.load_checkpoint(campaign_id)

    # Build parameter specs
    param_specs = []
    for p in spec.get("parameters", []):
        var_type = VariableType(p.get("type", "continuous"))
        param_specs.append(ParameterSpec(
            name=p["name"],
            type=var_type,
            lower=p.get("lower"),
            upper=p.get("upper"),
            categories=p.get("categories"),
        ))

    # Build observations: prefer checkpoint (includes initial + new trials),
    # fall back to spec.initial_observations if no checkpoint exists.
    observations = []
    if checkpoint and checkpoint.get("completed_trials"):
        for trial in checkpoint["completed_trials"]:
            observations.append(Observation(
                iteration=trial.get("iteration", 0),
                parameters=trial.get("parameters", {}),
                kpi_values=trial.get("kpi_values", {}),
                is_failure=trial.get("is_failure", False),
                metadata=trial.get("metadata", {}),
            ))
    else:
        # No checkpoint yet — use initial observations from spec
        for obs_raw in spec.get("initial_observations", []):
            observations.append(Observation(
                iteration=obs_raw.get("iteration", 0),
                parameters=obs_raw.get("parameters", {}),
                kpi_values=obs_raw.get("kpi_values", {}),
                is_failure=obs_raw.get("is_failure", False),
                metadata=obs_raw.get("metadata", {}),
            ))

    objective_names = [o.get("name", "") for o in spec.get("objectives", [])]
    objective_directions = [o.get("direction", "minimize") for o in spec.get("objectives", [])]

    return CampaignSnapshot(
        campaign_id=campaign_id,
        parameter_specs=param_specs,
        observations=observations,
        objective_names=objective_names,
        objective_directions=objective_directions,
        current_iteration=checkpoint.get("iteration", len(observations)) if checkpoint else len(observations),
    )


def _compute_diagnostics(snapshot: CampaignSnapshot) -> DiagnosticsResponse:
    """Compute diagnostic metrics from snapshot data.

    Uses the DataAnalysisPipeline when available, falls back to direct
    computation for robustness.
    """
    obs = snapshot.successful_observations
    if not obs or not snapshot.objective_names:
        return DiagnosticsResponse()

    primary_obj = snapshot.objective_names[0]
    values = [o.kpi_values.get(primary_obj, 0.0) for o in obs if primary_obj in o.kpi_values]

    if not values:
        return DiagnosticsResponse()

    n = len(values)

    # Best KPI
    direction = snapshot.objective_directions[0] if snapshot.objective_directions else "minimize"
    best_kpi = min(values) if direction == "minimize" else max(values)

    # Convergence trend: slope of recent values normalized
    if n >= 3:
        recent = values[-min(n, 10):]
        x_mean = (len(recent) - 1) / 2.0
        y_mean = sum(recent) / len(recent)
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(len(recent)))
        slope = numerator / denominator if denominator > 0 else 0.0
        # Normalize by value range
        val_range = max(values) - min(values)
        convergence_trend = slope / val_range if val_range > 0 else 0.0
    else:
        convergence_trend = 0.0

    # Improvement velocity: relative improvement per iteration
    if n >= 2:
        if direction == "minimize":
            running_best = [min(values[: i + 1]) for i in range(n)]
        else:
            running_best = [max(values[: i + 1]) for i in range(n)]
        improvements = [abs(running_best[i] - running_best[i - 1]) for i in range(1, len(running_best))]
        improvement_velocity = sum(improvements) / len(improvements) if improvements else 0.0
    else:
        improvement_velocity = 0.0

    # Exploration coverage: fraction of parameter space sampled
    param_coverage = []
    for spec in snapshot.parameter_specs:
        if spec.lower is not None and spec.upper is not None and spec.upper > spec.lower:
            param_vals = [o.parameters.get(spec.name, 0.0) for o in obs if spec.name in o.parameters]
            if param_vals:
                covered = (max(param_vals) - min(param_vals)) / (spec.upper - spec.lower)
                param_coverage.append(min(covered, 1.0))
    exploration_coverage = sum(param_coverage) / len(param_coverage) if param_coverage else 0.0

    # Failure rate
    failure_rate = snapshot.failure_rate

    # Noise estimate: coefficient of variation of recent values
    if n >= 3:
        recent = values[-min(n, 10):]
        mean_val = sum(recent) / len(recent)
        if mean_val != 0:
            variance = sum((v - mean_val) ** 2 for v in recent) / len(recent)
            noise_estimate = (variance ** 0.5) / abs(mean_val)
        else:
            noise_estimate = 0.0
    else:
        noise_estimate = 0.0

    # Plateau length: consecutive iterations without improvement
    plateau_length = 0
    if n >= 2:
        if direction == "minimize":
            running_best = [min(values[: i + 1]) for i in range(n)]
        else:
            running_best = [max(values[: i + 1]) for i in range(n)]
        for i in range(len(running_best) - 1, 0, -1):
            if abs(running_best[i] - running_best[i - 1]) < 1e-10:
                plateau_length += 1
            else:
                break

    # Signal to noise ratio
    val_range = max(values) - min(values)
    if noise_estimate > 0 and val_range > 0:
        mean_val = sum(values) / len(values)
        signal_to_noise_ratio = val_range / (noise_estimate * abs(mean_val)) if mean_val != 0 else 0.0
    else:
        signal_to_noise_ratio = float("inf") if val_range > 0 else 0.0

    return DiagnosticsResponse(
        convergence_trend=round(convergence_trend, 6),
        improvement_velocity=round(improvement_velocity, 6),
        best_kpi_value=round(best_kpi, 6),
        exploration_coverage=round(exploration_coverage, 4),
        failure_rate=round(failure_rate, 4),
        noise_estimate=round(noise_estimate, 6),
        plateau_length=plateau_length,
        signal_to_noise_ratio=round(signal_to_noise_ratio, 4) if signal_to_noise_ratio != float("inf") else 999.0,
    )


def _compute_importance(snapshot: CampaignSnapshot) -> ImportanceResponse:
    """Compute parameter importance via variance-based analysis."""
    obs = snapshot.successful_observations
    if len(obs) < 3 or not snapshot.objective_names:
        return ImportanceResponse(importances=[
            {"name": p.name, "importance": 1.0 / len(snapshot.parameter_specs)}
            for p in snapshot.parameter_specs
        ])

    primary_obj = snapshot.objective_names[0]
    y_values = [o.kpi_values.get(primary_obj, 0.0) for o in obs if primary_obj in o.kpi_values]

    if len(y_values) < 3:
        return ImportanceResponse(importances=[
            {"name": p.name, "importance": 1.0 / max(len(snapshot.parameter_specs), 1)}
            for p in snapshot.parameter_specs
        ])

    # Try using fANOVA via DataAnalysisPipeline
    try:
        from optimization_copilot.agents.data_pipeline import DataAnalysisPipeline

        param_names = [p.name for p in snapshot.parameter_specs]
        X = []
        y = []
        for o in obs:
            if primary_obj in o.kpi_values:
                row = [float(o.parameters.get(pn, 0.0)) for pn in param_names]
                X.append(row)
                y.append(o.kpi_values[primary_obj])

        if len(X) >= 3 and len(X[0]) >= 1:
            pipeline = DataAnalysisPipeline()
            result = pipeline.run_fanova(X, y, param_names)
            if result.is_computed and isinstance(result.value, dict):
                importances_raw = result.value.get("importances", {})
                importances = [
                    {"name": name, "importance": round(score, 6)}
                    for name, score in importances_raw.items()
                ]
                if importances:
                    return ImportanceResponse(importances=importances)
    except Exception:
        pass

    # Fallback: correlation-based importance
    importances = []
    y_mean = sum(y_values) / len(y_values)
    y_var = sum((v - y_mean) ** 2 for v in y_values)

    for spec in snapshot.parameter_specs:
        x_vals = [float(o.parameters.get(spec.name, 0.0)) for o in obs if primary_obj in o.kpi_values]
        if len(x_vals) != len(y_values):
            importances.append({"name": spec.name, "importance": 0.0})
            continue

        x_mean = sum(x_vals) / len(x_vals)
        x_var = sum((v - x_mean) ** 2 for v in x_vals)

        if x_var > 0 and y_var > 0:
            cov = sum((x_vals[i] - x_mean) * (y_values[i] - y_mean) for i in range(len(x_vals)))
            correlation = cov / (x_var * y_var) ** 0.5
            importances.append({"name": spec.name, "importance": round(abs(correlation), 6)})
        else:
            importances.append({"name": spec.name, "importance": 0.0})

    # Normalize to sum to 1
    total = sum(imp["importance"] for imp in importances)
    if total > 0:
        for imp in importances:
            imp["importance"] = round(imp["importance"] / total, 6)

    return ImportanceResponse(importances=importances)


def _generate_suggestions(
    snapshot: CampaignSnapshot, n: int = 5
) -> SuggestionResponse:
    """Generate next experiment suggestions."""
    if not snapshot.parameter_specs:
        return SuggestionResponse(suggestions=[])

    # Try using CampaignLoop for intelligent suggestions
    try:
        from optimization_copilot.campaign.loop import CampaignLoop

        loop = CampaignLoop(
            snapshot=snapshot,
            candidates=[],  # No pre-defined candidates
            smiles_param="",
            objectives=list(snapshot.objective_names),
            objective_directions={
                name: direction
                for name, direction in zip(snapshot.objective_names, snapshot.objective_directions)
            },
            batch_size=n,
        )
        deliverable = loop.run_iteration()
        suggestions = []
        predicted_values = []
        predicted_uncertainties = []
        for c in deliverable.dashboard.ranked_table.candidates[:n]:
            suggestions.append(c.parameters)
            predicted_values.append(c.predicted_mean)
            predicted_uncertainties.append(c.predicted_std)

        return SuggestionResponse(
            suggestions=suggestions,
            predicted_values=predicted_values,
            predicted_uncertainties=predicted_uncertainties,
            backend_used="campaign_loop",
            phase=deliverable.reasoning.diagnostic_summary.get("phase", "exploitation")
            if deliverable.reasoning.diagnostic_summary else "exploitation",
        )
    except Exception:
        pass

    # Fallback: generate suggestions via Latin Hypercube sampling within bounds
    import random
    suggestions = []
    for _ in range(n):
        params: dict[str, Any] = {}
        for spec in snapshot.parameter_specs:
            if spec.type == VariableType.CATEGORICAL and spec.categories:
                params[spec.name] = random.choice(spec.categories)
            elif spec.lower is not None and spec.upper is not None:
                params[spec.name] = round(
                    spec.lower + random.random() * (spec.upper - spec.lower), 6
                )
            else:
                params[spec.name] = round(random.gauss(0, 1), 6)
        suggestions.append(params)

    return SuggestionResponse(
        suggestions=suggestions,
        backend_used="random_fallback",
        phase="exploration",
    )


# ── Endpoints ────────────────────────────────────────────────────────


@router.post("/campaigns/from-upload", status_code=201)
def create_campaign_from_upload(req: CreateFromUploadRequest) -> dict[str, Any]:
    """Create a new campaign from uploaded and mapped data.

    The frontend wizard calls this after the user:
    1. Uploads a CSV/Excel file
    2. Maps columns to parameters/objectives/metadata
    3. Configures campaign settings (batch size, exploration weight)
    """
    manager = get_manager()
    workspace = get_workspace()

    # ── Input validation ────────────────────────────────────────────
    if not req.mapping.parameters:
        raise HTTPException(
            status_code=422,
            detail="At least one parameter and one objective are required",
        )
    if not req.mapping.objectives:
        raise HTTPException(
            status_code=422,
            detail="At least one parameter and one objective are required",
        )

    if not (1 <= req.batch_size <= 100):
        raise HTTPException(
            status_code=422,
            detail=f"batch_size must be between 1 and 100, got {req.batch_size}",
        )

    for p in req.mapping.parameters:
        if p.type == "continuous" and p.lower is not None and p.upper is not None:
            if p.lower > p.upper:
                raise HTTPException(
                    status_code=422,
                    detail=f"Parameter '{p.name}' has invalid bounds: lower ({p.lower}) must be less than upper ({p.upper})",
                )

    warnings: list[str] = []

    if len(req.data) < 3 and len(req.data) > 0:
        warnings.append(
            f"Only {len(req.data)} observations uploaded — at least 3 recommended for meaningful analysis."
        )

    # Check for missing objective values
    obj_names = [o.name for o in req.mapping.objectives]
    total_obj_cells = len(req.data) * len(obj_names)
    missing_count = 0
    for row in req.data:
        for oname in obj_names:
            raw = row.get(oname, "")
            if not raw:
                missing_count += 1
            else:
                try:
                    float(raw)
                except ValueError:
                    missing_count += 1
    missing_pct = missing_count / total_obj_cells if total_obj_cells > 0 else 0.0

    if missing_pct > 0.5:
        raise HTTPException(
            status_code=422,
            detail=f"More than 50% of objective values are missing or non-numeric ({missing_count}/{total_obj_cells}). Please provide more complete data.",
        )

    if 0.1 <= missing_pct <= 0.5:
        warnings.append(
            f"{missing_count}/{total_obj_cells} objective values ({missing_pct:.0%}) are missing or non-numeric. Results may be affected."
        )
    # ── End validation ──────────────────────────────────────────────

    # Build spec from mapping
    parameters = []
    for p in req.mapping.parameters:
        param_dict: dict[str, Any] = {"name": p.name, "type": p.type}
        if p.lower is not None:
            param_dict["lower"] = p.lower
        if p.upper is not None:
            param_dict["upper"] = p.upper
        parameters.append(param_dict)

    objectives = [
        {"name": o.name, "direction": o.direction}
        for o in req.mapping.objectives
    ]

    # Parse initial observations from uploaded data rows
    initial_observations = []
    for i, row in enumerate(req.data):
        params: dict[str, Any] = {}
        kpi_values: dict[str, float] = {}
        metadata_vals: dict[str, str] = {}

        for p in req.mapping.parameters:
            raw = row.get(p.name, "")
            try:
                params[p.name] = float(raw) if raw else 0.0
            except ValueError:
                params[p.name] = raw  # categorical

        for o in req.mapping.objectives:
            raw = row.get(o.name, "")
            try:
                kpi_values[o.name] = float(raw) if raw else 0.0
            except ValueError:
                pass

        for m in req.mapping.metadata:
            metadata_vals[m] = row.get(m, "")

        initial_observations.append({
            "iteration": i,
            "parameters": params,
            "kpi_values": kpi_values,
            "metadata": metadata_vals,
        })

    # Build full spec
    spec_dict: dict[str, Any] = {
        "parameters": parameters,
        "objectives": objectives,
        "metadata_columns": req.mapping.metadata,
        "batch_size": req.batch_size,
        "exploration_weight": req.exploration_weight,
        "description": req.description,
        "initial_observations": initial_observations,
    }

    # Create campaign via manager
    campaign_name = req.name or f"Upload-{time.strftime('%Y%m%d-%H%M%S')}"
    record = manager.create(spec_dict=spec_dict, name=campaign_name)

    # Save initial checkpoint with observations
    checkpoint = {
        "iteration": len(initial_observations),
        "completed_trials": initial_observations,
        "phase_history": [{"iteration": 0, "to_phase": "learning"}],
    }
    workspace.save_checkpoint(record.campaign_id, checkpoint)

    # Update record with progress
    best_kpi = None
    if initial_observations and objectives:
        primary = objectives[0]["name"]
        direction = objectives[0].get("direction", "minimize")
        kpi_vals = [obs["kpi_values"].get(primary) for obs in initial_observations if obs["kpi_values"].get(primary) is not None]
        if kpi_vals:
            best_kpi = min(kpi_vals) if direction == "minimize" else max(kpi_vals)
            manager.update_progress(
                record.campaign_id,
                iteration=len(initial_observations),
                total_trials=len(initial_observations),
                best_kpi=best_kpi,
            )

    result: dict[str, Any] = {
        "campaign_id": record.campaign_id,
        "id": record.campaign_id,
        "name": record.name,
        "status": record.status.value,
        "total_trials": len(initial_observations),
        "best_kpi": best_kpi,
    }
    if warnings:
        result["warnings"] = warnings
    return result


@router.post("/campaigns/{campaign_id}/append")
def append_observations(campaign_id: str, req: AppendRequest) -> dict[str, Any]:
    """Append new observations to an existing campaign.

    Validates that new data columns match the existing parameter and
    objective names, parses rows into observations, and updates the
    campaign checkpoint.
    """
    workspace = get_workspace()
    manager = get_manager()

    if not workspace.campaign_exists(campaign_id):
        raise HTTPException(status_code=404, detail=f"Campaign not found: {campaign_id}")

    if not req.data:
        raise HTTPException(status_code=422, detail="No data rows provided")

    spec = workspace.load_spec(campaign_id)
    checkpoint = workspace.load_checkpoint(campaign_id) or {
        "iteration": 0,
        "completed_trials": [],
        "phase_history": [],
    }

    param_defs = spec.get("parameters", [])
    obj_defs = spec.get("objectives", [])
    metadata_cols = spec.get("metadata_columns", [])
    param_names = {p["name"] for p in param_defs}
    obj_names = {o["name"] for o in obj_defs}

    # Validate column presence in new data
    all_expected = param_names | obj_names
    first_row_keys = set(req.data[0].keys()) if req.data else set()
    missing_cols = all_expected - first_row_keys
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"New data is missing required columns: {sorted(missing_cols)}",
        )

    existing_trials = checkpoint.get("completed_trials", [])
    start_iteration = checkpoint.get("iteration", len(existing_trials))

    new_observations = []
    for i, row in enumerate(req.data):
        params: dict[str, Any] = {}
        kpi_values: dict[str, float] = {}
        metadata_vals: dict[str, str] = {}

        for p in param_defs:
            raw = row.get(p["name"], "")
            try:
                params[p["name"]] = float(raw) if raw else 0.0
            except ValueError:
                params[p["name"]] = raw  # categorical

        for o in obj_defs:
            raw = row.get(o["name"], "")
            try:
                kpi_values[o["name"]] = float(raw) if raw else 0.0
            except ValueError:
                pass

        for m in metadata_cols:
            metadata_vals[m] = row.get(m, "")

        new_observations.append({
            "iteration": start_iteration + i,
            "parameters": params,
            "kpi_values": kpi_values,
            "metadata": metadata_vals,
        })

    # Append to existing checkpoint
    all_trials = existing_trials + new_observations
    new_iteration = start_iteration + len(new_observations)

    checkpoint["completed_trials"] = all_trials
    checkpoint["iteration"] = new_iteration

    # Recalculate best_kpi
    best_kpi = None
    if obj_defs:
        primary = obj_defs[0]["name"]
        direction = obj_defs[0].get("direction", "minimize")
        kpi_vals = [
            t["kpi_values"].get(primary)
            for t in all_trials
            if t["kpi_values"].get(primary) is not None
        ]
        if kpi_vals:
            best_kpi = min(kpi_vals) if direction == "minimize" else max(kpi_vals)

    workspace.save_checkpoint(campaign_id, checkpoint)

    # Update campaign record progress
    manager.update_progress(
        campaign_id,
        iteration=new_iteration,
        total_trials=len(all_trials),
        best_kpi=best_kpi,
    )

    return {
        "campaign_id": campaign_id,
        "appended": len(new_observations),
        "total": len(all_trials),
        "best_kpi": best_kpi,
    }


@router.get("/campaigns/{campaign_id}/diagnostics")
def get_diagnostics(campaign_id: str) -> dict[str, Any]:
    """Compute real-time diagnostic health metrics for a campaign."""
    snapshot = _load_snapshot(campaign_id)
    diag = _compute_diagnostics(snapshot)
    result: dict[str, Any] = diag.model_dump()
    n = len(snapshot.observations)
    if n < 3:
        result["warning"] = f"Only {n} observations — diagnostics may be unreliable"
    return result


@router.get("/campaigns/{campaign_id}/importance")
def get_importance(campaign_id: str) -> dict[str, Any]:
    """Compute parameter importance scores via fANOVA or correlation."""
    snapshot = _load_snapshot(campaign_id)
    n = len(snapshot.observations)
    if n < 5:
        # Return placeholder equal importances with a warning
        n_params = max(len(snapshot.parameter_specs), 1)
        equal_importance = round(1.0 / n_params, 6)
        return {
            "importances": [
                {"name": p.name, "importance": equal_importance}
                for p in snapshot.parameter_specs
            ],
            "warning": f"Only {n} observations — at least 5 needed for reliable importance estimates. Showing equal placeholder values.",
        }
    importance = _compute_importance(snapshot)
    return importance.model_dump()


@router.get("/campaigns/{campaign_id}/suggestions")
def get_suggestions(
    campaign_id: str,
    n: int = Query(default=5, ge=1, le=50),
) -> dict[str, Any]:
    """Generate next experiment suggestions based on campaign history."""
    snapshot = _load_snapshot(campaign_id)
    obs_count = len(snapshot.observations)
    if obs_count < 3:
        # Use only random sampling with a warning
        import random
        suggestions = []
        for _ in range(n):
            params: dict[str, Any] = {}
            for spec in snapshot.parameter_specs:
                if spec.type == VariableType.CATEGORICAL and spec.categories:
                    params[spec.name] = random.choice(spec.categories)
                elif spec.lower is not None and spec.upper is not None:
                    params[spec.name] = round(
                        spec.lower + random.random() * (spec.upper - spec.lower), 6
                    )
                else:
                    params[spec.name] = round(random.gauss(0, 1), 6)
            suggestions.append(params)
        return {
            "suggestions": suggestions,
            "predicted_values": [],
            "predicted_uncertainties": [],
            "backend_used": "random_fallback",
            "phase": "exploration",
            "warning": f"Using random sampling — insufficient data for model-based suggestions (have {obs_count}, need at least 3)",
        }
    result = _generate_suggestions(snapshot, n)
    return result.model_dump()


@router.post("/campaigns/{campaign_id}/steer")
def apply_steering(campaign_id: str, req: SteerRequest) -> dict[str, str]:
    """Apply a steering directive (focus region, avoid region, etc.)."""
    workspace = get_workspace()

    if not workspace.campaign_exists(campaign_id):
        raise HTTPException(status_code=404, detail=f"Campaign not found: {campaign_id}")

    # Load current spec and add steering constraint
    spec = workspace.load_spec(campaign_id)
    steering = spec.get("steering_directives", [])
    steering.append({
        "action": req.action,
        "region_bounds": req.region_bounds,
        "reason": req.reason,
        "timestamp": time.time(),
    })
    spec["steering_directives"] = steering
    workspace.save_spec(campaign_id, spec)

    return {"status": "accepted"}


@router.post("/chat/{campaign_id}")
def chat(campaign_id: str, req: ChatRequest) -> ChatResponse:
    """Process a chat message in the context of a campaign.

    Routes to appropriate analysis based on message intent.
    """
    snapshot = _load_snapshot(campaign_id)
    message = req.message.strip().lower()

    # Welcome message (empty input)
    if not message or not req.message.strip():
        obs_count = len(snapshot.observations)
        obj_names = ", ".join(snapshot.objective_names) if snapshot.objective_names else "none defined"
        param_count = len(snapshot.parameter_specs)
        return ChatResponse(
            reply=(
                f"Welcome! This campaign has {obs_count} observations, "
                f"{param_count} parameters, and objectives: {obj_names}. "
                f"Ask me to suggest next experiments, show diagnostics, "
                f"explain parameter importance, or focus on a specific region."
            ),
            role="system",
        )

    # Route based on intent

    if any(kw in message for kw in ["suggest", "next", "recommend", "what should"]):
        suggestions = _generate_suggestions(snapshot, n=5)
        formatted = []
        for i, s in enumerate(suggestions.suggestions[:5], 1):
            params_str = ", ".join(f"{k}={v}" for k, v in s.items())
            formatted.append(f"{i}. {params_str}")

        return ChatResponse(
            reply=f"Here are {len(formatted)} suggested experiments:\n\n" + "\n".join(formatted),
            role="suggestion",
            metadata={"suggestions": suggestions.suggestions[:5]},
        )

    if any(kw in message for kw in ["diagnostic", "health", "status", "how is"]):
        diag = _compute_diagnostics(snapshot)
        return ChatResponse(
            reply=(
                f"Campaign Diagnostics:\n"
                f"- Best KPI: {diag.best_kpi_value}\n"
                f"- Convergence trend: {diag.convergence_trend}\n"
                f"- Improvement velocity: {diag.improvement_velocity}\n"
                f"- Exploration coverage: {diag.exploration_coverage:.1%}\n"
                f"- Failure rate: {diag.failure_rate:.1%}\n"
                f"- Plateau length: {diag.plateau_length} iterations\n"
                f"- Signal/noise: {diag.signal_to_noise_ratio}"
            ),
            role="agent",
            metadata={"diagnostics": diag.model_dump()},
        )

    if any(kw in message for kw in ["importance", "which parameter", "most important", "fanova"]):
        importance = _compute_importance(snapshot)
        sorted_imp = sorted(importance.importances, key=lambda x: x["importance"], reverse=True)
        lines = [f"- {imp['name']}: {imp['importance']:.4f}" for imp in sorted_imp]
        return ChatResponse(
            reply="Parameter Importance (by contribution to objective variance):\n\n" + "\n".join(lines),
            role="agent",
            metadata={"importances": sorted_imp},
        )

    if any(kw in message for kw in ["why", "explain", "reason", "rationale"]):
        diag = _compute_diagnostics(snapshot)
        reasons = []
        if diag.plateau_length > 3:
            reasons.append(f"The optimization has plateaued for {diag.plateau_length} iterations. Consider increasing exploration.")
        if diag.failure_rate > 0.2:
            reasons.append(f"High failure rate ({diag.failure_rate:.0%}). Some parameter regions may be infeasible.")
        if diag.exploration_coverage < 0.3:
            reasons.append(f"Low exploration coverage ({diag.exploration_coverage:.0%}). The search space is underexplored.")
        if diag.noise_estimate > 0.5:
            reasons.append(f"High noise level detected (CV={diag.noise_estimate:.2f}). Results may benefit from replicate experiments.")
        if not reasons:
            reasons.append("The campaign is progressing well with no major concerns.")

        return ChatResponse(
            reply="Analysis:\n\n" + "\n".join(f"- {r}" for r in reasons),
            role="agent",
        )

    if any(kw in message for kw in ["export", "download", "save"]):
        return ChatResponse(
            reply="You can export campaign data using the Export tab in the workspace. Supported formats: CSV, JSON, and Excel.",
            role="system",
        )

    # Help intent — placed after specific intents to avoid keyword conflicts
    if any(kw in message for kw in ["help", "what can", "tutorial", "how do i", "how to"]):
        return ChatResponse(
            reply=(
                "I can help you with your optimization campaign. Try asking me:\n"
                "- 'Discover insights from data' -- Find patterns, correlations, and optimal regions\n"
                "- 'Suggest next experiments' -- Get AI-recommended parameter values\n"
                "- 'Show diagnostics' -- View campaign health metrics\n"
                "- 'Which parameter matters most?' -- See parameter importance ranking\n"
                "- 'Why is it stuck?' -- Understand optimization status\n"
                "- 'Focus on specific region' -- Steer the search direction\n"
                "- 'Export results' -- Download your data"
            ),
            role="system",
        )

    if any(kw in message for kw in ["insight", "discover", "pattern", "trend", "find", "learn"]):
        obs_count = len(snapshot.observations)
        if obs_count < 5:
            return ChatResponse(
                reply=f"I need at least 5 observations to discover meaningful insights. You currently have {obs_count}. Please add more experimental data.",
                role="agent",
            )
        from optimization_copilot.api.routes.insights import (
            _find_top_conditions,
            _compute_correlations,
            _find_optimal_regions,
            _detect_trends,
            _generate_summaries,
            _detect_interactions,
            _detect_failure_patterns,
        )
        top_cond = _find_top_conditions(snapshot, 5)
        corrs = _compute_correlations(snapshot)
        interactions = _detect_interactions(snapshot)
        opt_regions = _find_optimal_regions(snapshot)
        fail_patterns = _detect_failure_patterns(snapshot)
        trends = _detect_trends(snapshot)
        summaries = _generate_summaries(
            snapshot, top_cond, corrs, interactions, opt_regions, fail_patterns, trends
        )

        # Build rich text summary
        lines = []
        for s in summaries[:6]:
            lines.append(f"**{s.title}**: {s.body}")

        reply_text = f"I analyzed {len(snapshot.observations)} observations across {len(snapshot.parameter_specs)} parameters. Here are the key insights:\n\n" + "\n\n".join(lines)

        return ChatResponse(
            reply=reply_text,
            role="agent",
            metadata={
                "insights": [s.model_dump() for s in summaries[:6]],
                "correlations": [c.model_dump() for c in corrs[:6]],
                "top_conditions": [tc.model_dump() for tc in top_cond[:5]],
                "optimal_regions": [r.model_dump() for r in opt_regions[:5]],
                "interactions": [ix.model_dump() for ix in interactions[:5]],
                "trends": [t.model_dump() for t in trends[:3]],
                "failure_patterns": [fp.model_dump() for fp in fail_patterns[:3]],
            },
        )

    if any(kw in message for kw in ["focus", "narrow", "region", "constrain"]):
        return ChatResponse(
            reply=(
                "To focus the search on a specific region, you can use the steering controls. "
                "For example, tell me: 'Focus on temperature between 20-40' or "
                "'Avoid pressure above 100'. I'll adjust the optimization accordingly."
            ),
            role="agent",
        )

    # Default: summarize campaign state
    obs_count = len(snapshot.observations)
    best_obs = snapshot.successful_observations
    summary = f"Campaign has {obs_count} observations"
    if best_obs and snapshot.objective_names:
        primary = snapshot.objective_names[0]
        vals = [o.kpi_values.get(primary, 0) for o in best_obs if primary in o.kpi_values]
        if vals:
            direction = snapshot.objective_directions[0] if snapshot.objective_directions else "minimize"
            best = min(vals) if direction == "minimize" else max(vals)
            summary += f", best {primary} = {best:.4f}"

    return ChatResponse(
        reply=f"{summary}. I can help you analyze parameter importance, suggest next experiments, show diagnostics, or explain the optimization strategy. What would you like to know?",
        role="agent",
    )


@router.get("/campaigns/{campaign_id}/export/{fmt}")
def export_campaign(campaign_id: str, fmt: str) -> StreamingResponse:
    """Export campaign data as CSV, JSON, or XLSX."""
    snapshot = _load_snapshot(campaign_id)

    if fmt == "json":
        data = snapshot.to_dict()
        content = json.dumps(data, indent=2, ensure_ascii=False)
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={campaign_id}.json"},
        )

    if fmt == "csv":
        output = io.StringIO()
        if snapshot.observations:
            param_names = [p.name for p in snapshot.parameter_specs]
            obj_names = snapshot.objective_names
            fieldnames = ["iteration"] + param_names + obj_names

            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for obs in snapshot.observations:
                row: dict[str, Any] = {"iteration": obs.iteration}
                for pn in param_names:
                    row[pn] = obs.parameters.get(pn, "")
                for on in obj_names:
                    row[on] = obs.kpi_values.get(on, "")
                writer.writerow(row)

        content_bytes = output.getvalue().encode("utf-8")
        return StreamingResponse(
            io.BytesIO(content_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={campaign_id}.csv"},
        )

    if fmt == "xlsx":
        # Return CSV with xlsx extension as fallback (openpyxl not required)
        output = io.StringIO()
        if snapshot.observations:
            param_names = [p.name for p in snapshot.parameter_specs]
            obj_names = snapshot.objective_names
            fieldnames = ["iteration"] + param_names + obj_names

            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for obs in snapshot.observations:
                row: dict[str, Any] = {"iteration": obs.iteration}
                for pn in param_names:
                    row[pn] = obs.parameters.get(pn, "")
                for on in obj_names:
                    row[on] = obs.kpi_values.get(on, "")
                writer.writerow(row)

        content_bytes = output.getvalue().encode("utf-8")
        return StreamingResponse(
            io.BytesIO(content_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={campaign_id}.xlsx"},
        )

    raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}. Use csv, json, or xlsx.")


# ── Demo Datasets ───────────────────────────────────────────────────

_DEMO_DATASETS_DIR = Path(__file__).resolve().parents[3] / "data" / "gollum"

_DEMO_CATALOG: list[dict[str, Any]] = [
    {
        "id": "oer_catalyst",
        "filename": "oer_data.csv",
        "name": "OER Catalyst",
        "description": "Oxygen Evolution Reaction catalyst composition optimization. Explores metal loading ratios (Ni, Fe, Co, Mn, Ce, La) to minimize overpotential for water splitting electrocatalysis.",
        "tags": ["Chemistry", "Catalysis", "Materials"],
        "parameters": [
            {"name": "ni_load", "type": "continuous"},
            {"name": "fe_load", "type": "continuous"},
            {"name": "co_load", "type": "continuous"},
            {"name": "mn_load", "type": "continuous"},
            {"name": "ce_load", "type": "continuous"},
            {"name": "la_load", "type": "continuous"},
        ],
        "objectives": [{"name": "objective", "direction": "minimize"}],
        "metadata_cols": ["procedure"],
    },
    {
        "id": "suzuki_miyaura",
        "filename": "suzuki_miyaura_data.csv",
        "name": "Suzuki-Miyaura Coupling",
        "description": "Palladium-catalyzed cross-coupling reaction optimization. Varies reactants, catalysts, ligands, reagents, and solvents to maximize product yield.",
        "tags": ["Chemistry", "Organic", "Cross-coupling"],
        "parameters": [
            {"name": "reactant_1_smiles", "type": "categorical"},
            {"name": "reactant_2_smiles", "type": "categorical"},
            {"name": "catalyst_smiles", "type": "categorical"},
            {"name": "ligand_smiles", "type": "categorical"},
            {"name": "reagent_1_smiles", "type": "categorical"},
            {"name": "solvent_1_smiles", "type": "categorical"},
        ],
        "objectives": [{"name": "objective", "direction": "maximize"}],
        "metadata_cols": ["rxn", "product", "procedure"],
    },
    {
        "id": "hplc_separation",
        "filename": "hplc_data.csv",
        "name": "HPLC Separation",
        "description": "High-Performance Liquid Chromatography condition optimization. Tunes sample loop volume, flow rates, tubing parameters, and timing to maximize separation quality.",
        "tags": ["Analytical", "Chromatography"],
        "parameters": [
            {"name": "sample_loop", "type": "continuous"},
            {"name": "additional_volume", "type": "continuous"},
            {"name": "tubing_volume", "type": "continuous"},
            {"name": "sample_flow", "type": "continuous"},
            {"name": "push_speed", "type": "continuous"},
            {"name": "wait_time", "type": "continuous"},
        ],
        "objectives": [{"name": "objective", "direction": "maximize"}],
        "metadata_cols": ["procedure"],
    },
    {
        "id": "additives",
        "filename": "additives_plate_1.csv",
        "name": "Reaction Additives",
        "description": "Chemical additive screening for C-H functionalization reactions. Explores different aryl halides, acids, and additives to maximize product formation.",
        "tags": ["Chemistry", "Organic", "Screening"],
        "parameters": [
            {"name": "ArX_Smiles", "type": "categorical"},
            {"name": "Acid_Smiles", "type": "categorical"},
            {"name": "additives", "type": "categorical"},
        ],
        "objectives": [{"name": "objective", "direction": "maximize"}],
        "metadata_cols": ["product", "rxn"],
    },
    {
        "id": "c2_yield",
        "filename": "c2_yield_data.csv",
        "name": "C2 Yield (OCM)",
        "description": "Oxidative Coupling of Methane (OCM) catalyst and condition optimization. Optimizes metal compositions, reaction temperature, gas flow rates, and contact time to maximize C2 hydrocarbon yield.",
        "tags": ["Chemistry", "Catalysis", "Gas-phase"],
        "parameters": [
            {"name": "sup", "type": "categorical"},
            {"name": "m1", "type": "categorical"},
            {"name": "m1_mol", "type": "continuous"},
            {"name": "m2", "type": "categorical"},
            {"name": "m2_mol", "type": "continuous"},
            {"name": "m3", "type": "categorical"},
            {"name": "m3_mol", "type": "continuous"},
            {"name": "react_temp", "type": "continuous"},
            {"name": "flow_vol", "type": "continuous"},
            {"name": "ar_vol", "type": "continuous"},
            {"name": "ch4_vol", "type": "continuous"},
            {"name": "o2_vol", "type": "continuous"},
            {"name": "contact", "type": "continuous"},
        ],
        "objectives": [{"name": "objective", "direction": "maximize"}],
        "metadata_cols": ["name", "default_features", "procedure"],
    },
    {
        "id": "bh_reaction",
        "filename": "bh_reaction_1.csv",
        "name": "Buchwald-Hartwig Amination",
        "description": "Palladium-catalyzed C-N bond formation optimization. Screens ligands, additives, bases, and aryl halides to maximize amination reaction yield.",
        "tags": ["Chemistry", "Organic", "Cross-coupling"],
        "parameters": [
            {"name": "ligand", "type": "categorical"},
            {"name": "additive", "type": "categorical"},
            {"name": "base", "type": "categorical"},
            {"name": "aryl halide", "type": "categorical"},
        ],
        "objectives": [{"name": "objective", "direction": "maximize"}],
        "metadata_cols": ["rxn", "procedure"],
    },
    {
        "id": "vapor_diffusion",
        "filename": "vapdiff_data.csv",
        "name": "Vapor Diffusion Crystallization",
        "description": "Perovskite crystal growth via vapor diffusion. Optimizes organic compound, solvent, molarities, vial volumes, reaction time, and temperature to maximize crystal quality.",
        "tags": ["Materials", "Crystallography"],
        "parameters": [
            {"name": "organic", "type": "categorical"},
            {"name": "organic_molarity", "type": "continuous"},
            {"name": "solvent", "type": "categorical"},
            {"name": "solvent_molarity", "type": "continuous"},
            {"name": "inorganic_molarity", "type": "continuous"},
            {"name": "acid_molarity", "type": "continuous"},
            {"name": "alpha_vial_volume", "type": "continuous"},
            {"name": "beta_vial_volume", "type": "continuous"},
            {"name": "reaction_time", "type": "continuous"},
            {"name": "reaction_temperature", "type": "continuous"},
        ],
        "objectives": [{"name": "objective", "direction": "maximize"}],
        "metadata_cols": ["crystal_score", "default_features", "procedure"],
    },
]


def _read_demo_csv(filename: str, max_rows: int = 0) -> tuple[list[str], list[dict[str, str]], int]:
    """Read a demo CSV file and return (columns, rows, total_row_count).

    If max_rows > 0 only the first max_rows data rows are returned,
    but total_row_count always reflects the full file.
    """
    filepath = _DEMO_DATASETS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Demo dataset file not found: {filename}")

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        all_rows: list[dict[str, str]] = []
        total = 0
        for row in reader:
            total += 1
            if max_rows <= 0 or total <= max_rows:
                all_rows.append({k: (v or "") for k, v in row.items()})

    return list(columns), all_rows, total


def _auto_detect_bounds(
    rows: list[dict[str, str]], param_name: str
) -> tuple[float | None, float | None]:
    """Detect numeric bounds from data rows for a parameter."""
    values: list[float] = []
    for row in rows:
        raw = row.get(param_name, "")
        try:
            values.append(float(raw))
        except (ValueError, TypeError):
            pass
    if len(values) < 2:
        return None, None
    return round(min(values), 6), round(max(values), 6)


@router.get("/demo-datasets")
def list_demo_datasets() -> list[dict[str, Any]]:
    """List available demo datasets with metadata."""
    results: list[dict[str, Any]] = []
    for ds in _DEMO_CATALOG:
        filepath = _DEMO_DATASETS_DIR / ds["filename"]
        row_count = 0
        if filepath.exists():
            with open(filepath, encoding="utf-8") as f:
                row_count = sum(1 for _ in f) - 1  # subtract header

        results.append({
            "id": ds["id"],
            "name": ds["name"],
            "description": ds["description"],
            "tags": ds["tags"],
            "filename": ds["filename"],
            "row_count": row_count,
            "n_parameters": len(ds["parameters"]),
            "n_objectives": len(ds["objectives"]),
            "parameter_names": [p["name"] for p in ds["parameters"]],
            "objective_names": [o["name"] for o in ds["objectives"]],
        })
    return results


@router.get("/demo-datasets/{dataset_id}")
def get_demo_dataset(
    dataset_id: str,
    max_rows: int = Query(default=50, ge=1, le=10000),
) -> dict[str, Any]:
    """Return parsed demo dataset with suggested column mapping.

    Returns up to *max_rows* data rows (default 50) plus full metadata
    and a pre-built column mapping ready for ``/campaigns/from-upload``.
    """
    catalog_entry = None
    for ds in _DEMO_CATALOG:
        if ds["id"] == dataset_id:
            catalog_entry = ds
            break

    if catalog_entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown demo dataset: {dataset_id}")

    columns, rows, total_rows = _read_demo_csv(catalog_entry["filename"], max_rows=max_rows)

    # Build mapping with auto-detected bounds for continuous params
    param_mapping: list[dict[str, Any]] = []
    for p in catalog_entry["parameters"]:
        entry: dict[str, Any] = {"name": p["name"], "type": p["type"]}
        if p["type"] == "continuous":
            lo, hi = _auto_detect_bounds(rows, p["name"])
            if lo is not None and hi is not None:
                entry["lower"] = lo
                entry["upper"] = hi
        param_mapping.append(entry)

    obj_mapping = catalog_entry["objectives"]
    metadata_cols = [c for c in catalog_entry.get("metadata_cols", []) if c in columns]
    ignored_cols = [
        c for c in columns
        if c not in {p["name"] for p in catalog_entry["parameters"]}
        and c not in {o["name"] for o in catalog_entry["objectives"]}
        and c not in metadata_cols
    ]

    return {
        "id": catalog_entry["id"],
        "name": catalog_entry["name"],
        "description": catalog_entry["description"],
        "tags": catalog_entry["tags"],
        "columns": columns,
        "row_count": total_rows,
        "rows_returned": len(rows),
        "data": rows,
        "mapping": {
            "parameters": param_mapping,
            "objectives": obj_mapping,
            "metadata": metadata_cols,
            "ignored": ignored_cols,
        },
    }

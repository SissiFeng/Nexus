"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Campaign Schemas ──────────────────────────────────────────


class CreateCampaignRequest(BaseModel):
    spec: dict[str, Any] = Field(..., description="OptimizationSpec as dict")
    name: str = Field(default="", description="Campaign display name")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")


class CampaignResponse(BaseModel):
    campaign_id: str
    name: str
    status: str
    created_at: float
    updated_at: float
    iteration: int = 0
    best_kpi: float | None = None
    total_trials: int = 0
    error_message: str | None = None
    tags: list[str] = Field(default_factory=list)


class CampaignListResponse(BaseModel):
    campaigns: list[CampaignResponse]
    total: int


class CampaignDetailResponse(CampaignResponse):
    spec: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Trial Schemas ─────────────────────────────────────────────


class TrialResultSubmission(BaseModel):
    trial_id: str
    kpi_values: dict[str, float] = Field(default_factory=dict)
    is_failure: bool = False
    failure_reason: str | None = None
    metadata: dict[str, Any] | None = None


class SubmitTrialsRequest(BaseModel):
    results: list[TrialResultSubmission]


class TrialResponse(BaseModel):
    trial_id: str
    iteration: int
    parameters: dict[str, Any]
    state: str
    kpi_values: dict[str, float] = Field(default_factory=dict)


class BatchResponse(BaseModel):
    batch_id: str
    iteration: int
    trials: list[TrialResponse]


# ── Store Schemas ─────────────────────────────────────────────


class StoreSummaryResponse(BaseModel):
    n_experiments: int
    n_campaigns: int
    campaign_ids: list[str]
    n_artifacts: int
    parameter_names: list[str]
    kpi_names: list[str]


# ── Advice Schemas ────────────────────────────────────────────


class AdviceRequest(BaseModel):
    fingerprint: dict[str, str] = Field(
        ..., description="ProblemFingerprint as dict"
    )


class AdviceResponse(BaseModel):
    recommended_backends: list[str]
    scoring_weights: dict[str, float] | None = None
    switching_thresholds: dict[str, float] | None = None
    failure_adjustments: dict[str, str] = Field(default_factory=dict)
    drift_robust_backends: list[str] = Field(default_factory=list)
    confidence: float
    reason_codes: list[str] = Field(default_factory=list)


# ── Report Schemas ────────────────────────────────────────────


class CompareRequest(BaseModel):
    campaign_ids: list[str] = Field(
        ..., min_length=2, description="Campaign IDs to compare"
    )


class CompareResponse(BaseModel):
    campaign_ids: list[str]
    records: list[CampaignResponse]
    kpi_comparison: dict[str, list[float | None]]
    iteration_comparison: list[int]
    winner: str | None = None


# ── Search Schemas ────────────────────────────────────────────


class SearchResultItem(BaseModel):
    campaign_id: str
    field: str
    snippet: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int


# ── Auth Schemas ──────────────────────────────────────────────


class CreateKeyRequest(BaseModel):
    name: str
    role: str = "viewer"


class CreateKeyResponse(BaseModel):
    raw_key: str
    name: str
    role: str
    message: str = "Save this key — it won't be shown again."


class KeyListResponse(BaseModel):
    keys: list[dict[str, Any]]


# ── Generic ───────────────────────────────────────────────────


class StatusResponse(BaseModel):
    status: str
    message: str = ""


class ErrorResponse(BaseModel):
    detail: str

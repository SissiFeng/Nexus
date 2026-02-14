"""Platform data models for workspace, campaigns, auth, and search."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CampaignStatus(str, Enum):
    """Campaign lifecycle status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"
    ARCHIVED = "archived"


# Valid state transitions: from_status -> set of to_statuses
VALID_TRANSITIONS: dict[CampaignStatus, set[CampaignStatus]] = {
    CampaignStatus.DRAFT: {CampaignStatus.RUNNING, CampaignStatus.ARCHIVED},
    CampaignStatus.RUNNING: {
        CampaignStatus.PAUSED,
        CampaignStatus.COMPLETED,
        CampaignStatus.STOPPED,
        CampaignStatus.FAILED,
        CampaignStatus.ARCHIVED,
    },
    CampaignStatus.PAUSED: {CampaignStatus.RUNNING, CampaignStatus.ARCHIVED},
    CampaignStatus.COMPLETED: {CampaignStatus.ARCHIVED},
    CampaignStatus.STOPPED: {CampaignStatus.ARCHIVED},
    CampaignStatus.FAILED: {CampaignStatus.ARCHIVED},
    CampaignStatus.ARCHIVED: {CampaignStatus.DRAFT},
}


class Role(str, Enum):
    """User role for RBAC."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


# Role hierarchy: higher index = more privileges
ROLE_HIERARCHY: dict[Role, int] = {
    Role.VIEWER: 0,
    Role.OPERATOR: 1,
    Role.ADMIN: 2,
}


@dataclass
class CampaignRecord:
    """Persistent campaign metadata."""

    campaign_id: str
    name: str
    status: CampaignStatus
    spec: dict[str, Any]
    created_at: float
    updated_at: float
    iteration: int = 0
    best_kpi: float | None = None
    total_trials: int = 0
    error_message: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "status": self.status.value,
            "spec": self.spec,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "iteration": self.iteration,
            "best_kpi": self.best_kpi,
            "total_trials": self.total_trials,
            "error_message": self.error_message,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignRecord:
        return cls(
            campaign_id=data["campaign_id"],
            name=data["name"],
            status=CampaignStatus(data["status"]),
            spec=data["spec"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            iteration=data.get("iteration", 0),
            best_kpi=data.get("best_kpi"),
            total_trials=data.get("total_trials", 0),
            error_message=data.get("error_message"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ApiKey:
    """API key record (stores hash only, never the raw key)."""

    key_hash: str
    name: str
    role: Role
    created_at: float
    last_used: float | None = None
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_hash": self.key_hash,
            "name": self.name,
            "role": self.role.value,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApiKey:
        return cls(
            key_hash=data["key_hash"],
            name=data["name"],
            role=Role(data["role"]),
            created_at=data["created_at"],
            last_used=data.get("last_used"),
            active=data.get("active", True),
        )


@dataclass
class WorkspaceManifest:
    """Workspace root manifest."""

    workspace_id: str
    created_at: float
    version: str = "1.0.0"
    campaigns: dict[str, str] = field(default_factory=dict)  # id -> name

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "created_at": self.created_at,
            "version": self.version,
            "campaigns": dict(self.campaigns),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceManifest:
        return cls(
            workspace_id=data["workspace_id"],
            created_at=data["created_at"],
            version=data.get("version", "1.0.0"),
            campaigns=data.get("campaigns", {}),
        )


@dataclass
class SearchResult:
    """RAG search result."""

    campaign_id: str
    field: str
    snippet: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "field": self.field,
            "snippet": self.snippet,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResult:
        return cls(
            campaign_id=data["campaign_id"],
            field=data["field"],
            snippet=data["snippet"],
            score=data["score"],
        )


@dataclass
class CompareReport:
    """Side-by-side campaign comparison."""

    campaign_ids: list[str]
    records: list[CampaignRecord]
    kpi_comparison: dict[str, list[float | None]]  # kpi_name -> [value per campaign]
    iteration_comparison: list[int]
    winner: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_ids": list(self.campaign_ids),
            "records": [r.to_dict() for r in self.records],
            "kpi_comparison": {k: list(v) for k, v in self.kpi_comparison.items()},
            "iteration_comparison": list(self.iteration_comparison),
            "winner": self.winner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompareReport:
        return cls(
            campaign_ids=data["campaign_ids"],
            records=[CampaignRecord.from_dict(r) for r in data["records"]],
            kpi_comparison=data["kpi_comparison"],
            iteration_comparison=data["iteration_comparison"],
            winner=data.get("winner"),
        )

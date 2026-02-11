"""Platform service layer for Optimization Copilot.

Provides multi-campaign workspace management, authentication,
real-time event bus, and RAG search capabilities.
"""

from optimization_copilot.platform.models import (
    ApiKey,
    CampaignRecord,
    CampaignStatus,
    CompareReport,
    Role,
    SearchResult,
    WorkspaceManifest,
)

__all__ = [
    "ApiKey",
    "CampaignRecord",
    "CampaignStatus",
    "CompareReport",
    "Role",
    "SearchResult",
    "WorkspaceManifest",
]

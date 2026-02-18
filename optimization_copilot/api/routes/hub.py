"""API routes for Nexus Hub - campaign sharing with FAIR metadata."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from optimization_copilot.platform.hub_service import (
    NexusHubService,
    SharingVisibility,
    AccessLevel,
    QuotaExceededError,
)


router = APIRouter(prefix="/hub", tags=["hub"])

# Global hub service instance
_hub_service: NexusHubService | None = None


def get_hub_service() -> NexusHubService:
    """Get or create the global hub service."""
    global _hub_service
    if _hub_service is None:
        _hub_service = NexusHubService()
    return _hub_service


# ── Request/Response Models ────────────────────────────────────────────

class OwnerInfo(BaseModel):
    name: str
    orcid: str | None = None
    institution: str | None = None


class SharingOptions(BaseModel):
    title: str
    description: str = ""
    keywords: list[str] = Field(default_factory=lambda: ["optimization"])
    visibility: str = "unlisted"  # private/unlisted/public/collaborative
    access_level: str = "view"    # view/fork/comment/edit/admin
    license: str = "MIT"
    domain: str | None = None
    experimental_context: dict[str, Any] = Field(default_factory=dict)


class ShareCampaignRequest(BaseModel):
    campaign_data: dict[str, Any]
    owner_info: OwnerInfo
    sharing_options: SharingOptions


class ShareCampaignResponse(BaseModel):
    hub_id: str
    share_url: str
    embed_url: str
    fair_metadata: dict[str, Any]
    schemaorg_jsonld: str
    datacite_xml: str
    quota_remaining: dict[str, Any]


class QuotaStatusResponse(BaseModel):
    user_id: str
    shared_campaigns: int
    total_storage_mb: float
    limits: dict[str, int]
    remaining: dict[str, Any]


class HubCampaignResponse(BaseModel):
    hub_id: str
    campaign_id: str
    owner_id: str
    visibility: str
    access_level: str
    fair_metadata: dict[str, Any]
    view_count: int
    fork_count: int
    created_at: str


class SearchCampaignsResponse(BaseModel):
    results: list[dict[str, Any]]
    total: int
    offset: int
    limit: int


class AddCollaboratorRequest(BaseModel):
    collaborator_id: str
    access_level: str = "view"


# ── Hub API Endpoints ──────────────────────────────────────────────────

@router.post("/share", response_model=ShareCampaignResponse)
async def share_campaign(
    request: ShareCampaignRequest,
    user_id: str = "current_user",  # Would come from auth in production
    hub: NexusHubService = Depends(get_hub_service),
) -> ShareCampaignResponse:
    """Share a campaign on Nexus Hub with FAIR metadata.
    
    Generates FAIR-compliant metadata (Schema.org JSON-LD, DataCite XML)
    and creates a shareable link.
    """
    try:
        result = hub.share_campaign(
            user_id=user_id,
            campaign_data=request.campaign_data,
            owner_info=request.owner_info.model_dump(),
            sharing_options=request.sharing_options.model_dump(),
        )
        return ShareCampaignResponse(**result)
        
    except QuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to share campaign: {e}")


@router.get("/quota/{user_id}", response_model=QuotaStatusResponse)
async def get_quota_status(
    user_id: str,
    hub: NexusHubService = Depends(get_hub_service),
) -> QuotaStatusResponse:
    """Get user's free tier quota status."""
    quota = hub.get_quota_status(user_id)
    return QuotaStatusResponse(**quota)


@router.get("/campaign/{hub_id}", response_model=HubCampaignResponse)
async def get_shared_campaign(
    hub_id: str,
    requester_id: str | None = None,
    hub: NexusHubService = Depends(get_hub_service),
) -> HubCampaignResponse:
    """Get a shared campaign by its hub ID.
    
    Access depends on campaign visibility and requester permissions.
    """
    shared = hub.get_shared_campaign(hub_id, requester_id)
    
    if shared is None:
        raise HTTPException(status_code=404, detail="Campaign not found or access denied")
    
    return HubCampaignResponse(
        hub_id=shared.hub_id,
        campaign_id=shared.campaign_id,
        owner_id=shared.owner_id,
        visibility=shared.visibility.value,
        access_level=shared.access_level.value,
        fair_metadata=shared.fair_metadata.to_dict(),
        view_count=shared.view_count,
        fork_count=shared.fork_count,
        created_at=shared.created_at,
    )


@router.get("/campaign/{hub_id}/metadata.jsonld")
async def get_metadata_jsonld(
    hub_id: str,
    hub: NexusHubService = Depends(get_hub_service),
) -> str:
    """Get FAIR metadata as Schema.org JSON-LD."""
    shared = hub.get_shared_campaign(hub_id)
    
    if shared is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return shared.fair_metadata.to_schemaorg_jsonld()


@router.get("/campaign/{hub_id}/metadata.datacite.xml")
async def get_metadata_datacite(
    hub_id: str,
    hub: NexusHubService = Depends(get_hub_service),
) -> str:
    """Get FAIR metadata as DataCite XML (for DOI registration)."""
    shared = hub.get_shared_campaign(hub_id)
    
    if shared is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return shared.fair_metadata.to_datacite_xml()


@router.post("/campaign/{hub_id}/fork")
async def fork_campaign(
    hub_id: str,
    new_name: str | None = None,
    user_id: str = "current_user",
    hub: NexusHubService = Depends(get_hub_service),
) -> dict[str, Any]:
    """Fork (copy) a shared campaign to your workspace."""
    result = hub.fork_campaign(hub_id, user_id, new_name)
    
    if result is None:
        raise HTTPException(
            status_code=403,
            detail="Campaign not found or forking not allowed"
        )
    
    return result


@router.get("/search", response_model=SearchCampaignsResponse)
async def search_campaigns(
    q: str | None = None,
    keywords: str | None = None,  # comma-separated
    domain: str | None = None,
    limit: int = 20,
    offset: int = 0,
    hub: NexusHubService = Depends(get_hub_service),
) -> SearchCampaignsResponse:
    """Search public campaigns on Nexus Hub.
    
    Args:
        q: Free text search query
        keywords: Comma-separated list of keywords
        domain: Scientific domain filter
        limit: Max results to return
        offset: Pagination offset
    """
    keyword_list = keywords.split(",") if keywords else None
    
    results = hub.search_public_campaigns(
        query=q,
        keywords=keyword_list,
        domain=domain,
        limit=limit,
        offset=offset,
    )
    
    return SearchCampaignsResponse(
        results=results,
        total=len(results),  # Simplified; would need actual count
        offset=offset,
        limit=limit,
    )


@router.get("/user/{user_id}/campaigns")
async def get_user_campaigns(
    user_id: str,
    hub: NexusHubService = Depends(get_hub_service),
) -> list[dict[str, Any]]:
    """Get all campaigns shared by a user."""
    return hub.get_user_shared_campaigns(user_id)


@router.post("/campaign/{hub_id}/collaborators")
async def add_collaborator(
    hub_id: str,
    request: AddCollaboratorRequest,
    owner_id: str = "current_user",
    hub: NexusHubService = Depends(get_hub_service),
) -> dict[str, str]:
    """Add a collaborator to a shared campaign."""
    try:
        access_level = AccessLevel(request.access_level)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid access level")
    
    success = hub.add_collaborator(
        hub_id=hub_id,
        owner_id=owner_id,
        collaborator_id=request.collaborator_id,
        access_level=access_level,
    )
    
    if not success:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to add collaborators"
        )
    
    return {"status": "success", "message": f"Added {request.collaborator_id} as collaborator"}


@router.delete("/campaign/{hub_id}/collaborators/{collaborator_id}")
async def remove_collaborator(
    hub_id: str,
    collaborator_id: str,
    owner_id: str = "current_user",
    hub: NexusHubService = Depends(get_hub_service),
) -> dict[str, str]:
    """Remove a collaborator from a shared campaign."""
    success = hub.remove_collaborator(
        hub_id=hub_id,
        owner_id=owner_id,
        collaborator_id=collaborator_id,
    )
    
    if not success:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to remove collaborators"
        )
    
    return {"status": "success", "message": f"Removed {collaborator_id}"}


# ── One-Click Share Endpoint ───────────────────────────────────────────

@router.post("/share/one-click")
async def one_click_share(
    campaign_id: str,
    user_id: str = "current_user",
    hub: NexusHubService = Depends(get_hub_service),
) -> dict[str, Any]:
    """One-click share with sensible defaults.
    
    Automatically generates:
    - Title from campaign name
    - Keywords from domain knowledge
    - Unlisted visibility (link-only)
    - MIT license
    """
    # In production, would fetch campaign data from workspace
    # For now, create minimal campaign data
    campaign_data = {
        "campaign_id": campaign_id,
        "parameter_specs": [],
        "observations": [],
        "objective_names": ["objective"],
    }
    
    owner_info = {
        "name": "Nexus User",
        "orcid": "",
        "institution": "",
    }
    
    sharing_options = {
        "title": f"Campaign {campaign_id[:8]}",
        "description": "Shared via Nexus Hub one-click share",
        "keywords": ["optimization", "bayesian-optimization", "nexus"],
        "visibility": "unlisted",
        "access_level": "view",
        "license": "MIT",
    }
    
    try:
        result = hub.share_campaign(
            user_id=user_id,
            campaign_data=campaign_data,
            owner_info=owner_info,
            sharing_options=sharing_options,
        )
        
        return {
            "success": True,
            "hub_id": result["hub_id"],
            "share_url": result["share_url"],
            "message": "Campaign shared successfully!",
            "quota": result["quota_remaining"],
        }
        
    except QuotaExceededError as e:
        return {
            "success": False,
            "error": "quota_exceeded",
            "message": str(e),
        }


# ── Featured & Discovery ───────────────────────────────────────────────

@router.get("/featured")
async def get_featured_campaigns(
    limit: int = 6,
    hub: NexusHubService = Depends(get_hub_service),
) -> list[dict[str, Any]]:
    """Get featured campaigns for hub homepage."""
    # Get public campaigns, sorted by popularity
    campaigns = hub.search_public_campaigns(limit=limit)
    return campaigns


@router.get("/stats")
async def get_hub_stats(
    hub: NexusHubService = Depends(get_hub_service),
) -> dict[str, Any]:
    """Get Nexus Hub statistics."""
    # Count campaigns
    total_campaigns = len(hub._shared_campaigns)
    public_campaigns = len(hub._public_index)
    
    # Count unique users
    unique_users = len(hub._user_campaigns)
    
    # Sum view/fork counts
    total_views = sum(s.view_count for s in hub._shared_campaigns.values())
    total_forks = sum(s.fork_count for s in hub._shared_campaigns.values())
    
    return {
        "total_campaigns": total_campaigns,
        "public_campaigns": public_campaigns,
        "unique_users": unique_users,
        "total_views": total_views,
        "total_forks": total_forks,
    }

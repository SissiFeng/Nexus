"""Nexus Hub - Cloud sharing service for campaigns with FAIR metadata.

Provides:
- Campaign sharing with public/private visibility
- FAIR-compliant metadata generation
- Free tier quota management
- Collaborative access control
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from pathlib import Path


class SharingVisibility(Enum):
    """Campaign sharing visibility levels."""
    PRIVATE = "private"          # Only owner
    UNLISTED = "unlisted"        # Anyone with link
    PUBLIC = "public"            # Discoverable in hub
    COLLABORATIVE = "collaborative"  # Specific collaborators


class AccessLevel(Enum):
    """Access permission levels."""
    VIEW = "view"                # Read-only
    FORK = "fork"                # Can clone/copy
    COMMENT = "comment"          # Can add notes
    EDIT = "edit"                # Can modify
    ADMIN = "admin"              # Full control


@dataclass
class FAIRMetadata:
    """FAIR-compliant metadata for shared campaigns.
    
    Follows FAIR principles:
    - Findable: Identifiers, searchable metadata
    - Accessible: Retrievable by standard protocol
    - Interoperable: Uses formal knowledge representation
    - Reusable: Rich metadata with clear license
    """
    # Findable
    identifier: str                              # DOI-style unique ID
    title: str
    description: str
    creators: list[dict[str, str]]              # Authors with ORCID
    keywords: list[str]
    publication_date: str                        # ISO 8601
    modified_date: str
    
    # Accessible
    access_url: str
    download_url: str | None
    access_level: str                           # open/restricted/embargoed
    
    # Interoperable
    schema_version: str                         # Campaign schema version
    parameter_specs: list[dict[str, Any]]       # Formal parameter definitions
    objective_definitions: list[dict[str, Any]] # Formal objective definitions
    license: str                                # SPDX license identifier
    
    # Reusable
    methodology: str                            # Optimization method used
    software_version: str                       # Nexus version
    provenance: list[dict[str, Any]]            # Data lineage
    citation: str                               # Recommended citation
    
    # Domain-specific
    domain: str | None                          # Scientific domain
    experimental_context: dict[str, Any]        # Lab equipment, conditions
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "@context": "https://schema.org",
            "@type": "Dataset",
            "identifier": self.identifier,
            "name": self.title,
            "description": self.description,
            "creator": self.creators,
            "keywords": self.keywords,
            "datePublished": self.publication_date,
            "dateModified": self.modified_date,
            "url": self.access_url,
            "distribution": {
                "@type": "DataDownload",
                "contentUrl": self.download_url,
            } if self.download_url else None,
            "license": self.license,
            "variableMeasured": self.parameter_specs,
            "measurementTechnique": self.methodology,
            "softwareVersion": self.software_version,
            "additionalProperty": [
                {"@type": "PropertyValue", "name": "domain", "value": self.domain},
                {"@type": "PropertyValue", "name": "provenance", "value": self.provenance},
            ],
        }
    
    def to_schemaorg_jsonld(self) -> str:
        """Export as Schema.org JSON-LD format."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_datacite_xml(self) -> str:
        """Export as DataCite XML format (for DOI registration)."""
        creators_xml = "".join([
            f"<creator><creatorName>{c.get('name')}</creatorName>"
            f"<nameIdentifier nameIdentifierScheme=\"ORCID\">{c.get('orcid', '')}</nameIdentifier></creator>"
            for c in self.creators
        ])
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<resource xmlns="http://datacite.org/schema/kernel-4">
  <identifier identifierType="DOI">{self.identifier}</identifier>
  <creators>{creators_xml}</creators>
  <titles><title>{self.title}</title></titles>
  <publisher>Nexus Hub</publisher>
  <publicationYear>{self.publication_date[:4]}</publicationYear>
  <resourceType resourceTypeGeneral="Dataset">Optimization Campaign</resourceType>
  <descriptions>
    <description descriptionType="Abstract">{self.description}</description>
  </descriptions>
  <subjects>
    {''.join(f'<subject>{k}</subject>' for k in self.keywords)}
  </subjects>
  <rightsList>
    <rights rightsURI="https://spdx.org/licenses/{self.license}.html">{self.license}</rights>
  </rightsList>
</resource>"""


@dataclass
class SharedCampaign:
    """A campaign shared on Nexus Hub."""
    hub_id: str                              # Hub-specific ID
    campaign_id: str                         # Original campaign ID
    owner_id: str                            # User who shared it
    visibility: SharingVisibility
    access_level: AccessLevel
    fair_metadata: FAIRMetadata
    collaborators: dict[str, AccessLevel]    # user_id -> level
    created_at: str
    modified_at: str
    view_count: int = 0
    fork_count: int = 0
    citation_count: int = 0
    is_featured: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "hub_id": self.hub_id,
            "campaign_id": self.campaign_id,
            "owner_id": self.owner_id,
            "visibility": self.visibility.value,
            "access_level": self.access_level.value,
            "fair_metadata": self.fair_metadata.to_dict(),
            "collaborators": {k: v.value for k, v in self.collaborators.items()},
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "view_count": self.view_count,
            "fork_count": self.fork_count,
            "citation_count": self.citation_count,
            "is_featured": self.is_featured,
        }


@dataclass
class QuotaUsage:
    """Track user's free tier quota usage."""
    user_id: str
    shared_campaigns: int = 0
    total_storage_mb: float = 0.0
    api_calls_monthly: int = 0
    collaborators_total: int = 0
    last_reset: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Free tier limits
    MAX_SHARED_CAMPAIGNS = 10
    MAX_STORAGE_MB = 1000  # 1 GB
    MAX_API_CALLS_MONTHLY = 10000
    MAX_COLLABORATORS = 20
    
    @property
    def campaigns_remaining(self) -> int:
        return max(0, self.MAX_SHARED_CAMPAIGNS - self.shared_campaigns)
    
    @property
    def storage_remaining_mb(self) -> float:
        return max(0.0, self.MAX_STORAGE_MB - self.total_storage_mb)
    
    @property
    def api_calls_remaining(self) -> int:
        return max(0, self.MAX_API_CALLS_MONTHLY - self.api_calls_monthly)
    
    def can_share_new_campaign(self, estimated_size_mb: float = 10.0) -> bool:
        """Check if user can share another campaign."""
        return (
            self.shared_campaigns < self.MAX_SHARED_CAMPAIGNS and
            self.total_storage_mb + estimated_size_mb <= self.MAX_STORAGE_MB
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "shared_campaigns": self.shared_campaigns,
            "total_storage_mb": round(self.total_storage_mb, 2),
            "api_calls_monthly": self.api_calls_monthly,
            "collaborators_total": self.collaborators_total,
            "limits": {
                "max_campaigns": self.MAX_SHARED_CAMPAIGNS,
                "max_storage_mb": self.MAX_STORAGE_MB,
                "max_api_calls_monthly": self.MAX_API_CALLS_MONTHLY,
                "max_collaborators": self.MAX_COLLABORATORS,
            },
            "remaining": {
                "campaigns": self.campaigns_remaining,
                "storage_mb": round(self.storage_remaining_mb, 2),
                "api_calls": self.api_calls_remaining,
            },
            "last_reset": self.last_reset,
        }


class NexusHubService:
    """Service for sharing campaigns on Nexus Hub.
    
    Manages:
    - Campaign publishing with FAIR metadata
    - Access control and permissions
    - Free tier quota enforcement
    - Hub discovery and search
    """
    
    def __init__(self, hub_storage_path: str | Path | None = None) -> None:
        self._storage_path = Path(hub_storage_path) if hub_storage_path else None
        self._shared_campaigns: dict[str, SharedCampaign] = {}  # hub_id -> campaign
        self._user_campaigns: dict[str, list[str]] = {}  # user_id -> [hub_ids]
        self._quota_usage: dict[str, QuotaUsage] = {}  # user_id -> quota
        
        # In-memory indexes
        self._public_index: list[str] = []  # hub_ids of public campaigns
        self._keyword_index: dict[str, list[str]] = {}  # keyword -> hub_ids
    
    # ── FAIR Metadata Generation ─────────────────────────────────────────
    
    def generate_fair_metadata(
        self,
        campaign_data: dict[str, Any],
        owner_info: dict[str, str],
        sharing_options: dict[str, Any],
    ) -> FAIRMetadata:
        """Generate FAIR-compliant metadata from campaign data.
        
        Args:
            campaign_data: Full campaign snapshot and results
            owner_info: User info (name, orcid, institution)
            sharing_options: Sharing preferences (license, keywords, etc.)
        """
        now = datetime.now().isoformat()
        campaign_id = campaign_data.get("campaign_id", str(uuid.uuid4()))
        
        # Generate identifier (DOI-style)
        identifier = f"10.5281/nexus.{campaign_id[:8]}"
        
        # Extract parameter specs
        param_specs = campaign_data.get("parameter_specs", [])
        formal_params = [
            {
                "@type": "PropertyValue",
                "name": p.get("name"),
                "value": p.get("type"),
                "minValue": p.get("lower"),
                "maxValue": p.get("upper"),
            }
            for p in param_specs
        ]
        
        # Extract objectives
        objectives = campaign_data.get("objective_names", [])
        formal_objs = [
            {
                "@type": "PropertyValue",
                "name": obj,
                "description": f"Optimization objective: {obj}",
            }
            for obj in objectives
        ]
        
        # Build provenance
        provenance = []
        observations = campaign_data.get("observations", [])
        if observations:
            provenance.append({
                "activity": "Bayesian Optimization",
                "n_iterations": len(observations),
                "n_parameters": len(param_specs),
            })
        
        # Methodology
        backend = campaign_data.get("backend_name", "unknown")
        methodology = f"Bayesian optimization using {backend} algorithm"
        
        return FAIRMetadata(
            identifier=identifier,
            title=sharing_options.get("title", f"Campaign {campaign_id[:8]}"),
            description=sharing_options.get(
                "description",
                "Optimization campaign data shared via Nexus Hub"
            ),
            creators=[{
                "name": owner_info.get("name", "Anonymous"),
                "orcid": owner_info.get("orcid", ""),
                "affiliation": owner_info.get("institution", ""),
            }],
            keywords=sharing_options.get("keywords", ["optimization", "bayesian-optimization"]),
            publication_date=now,
            modified_date=now,
            access_url=f"https://hub.nexus.dev/campaign/{campaign_id}",
            download_url=None,  # Set when published
            access_level=sharing_options.get("access_level", "open"),
            schema_version="1.0",
            parameter_specs=formal_params,
            objective_definitions=formal_objs,
            license=sharing_options.get("license", "MIT"),
            methodology=methodology,
            software_version=campaign_data.get("nexus_version", "0.2.0"),
            provenance=provenance,
            citation=self._generate_citation(owner_info, now, identifier),
            domain=sharing_options.get("domain"),
            experimental_context=sharing_options.get("experimental_context", {}),
        )
    
    def _generate_citation(
        self,
        owner_info: dict[str, str],
        date: str,
        identifier: str,
    ) -> str:
        """Generate recommended citation string."""
        name = owner_info.get("name", "Anonymous")
        year = date[:4]
        return f"{name} ({year}). Optimization Campaign Data. Nexus Hub. https://doi.org/{identifier}"
    
    # ── Campaign Sharing ─────────────────────────────────────────────────
    
    def share_campaign(
        self,
        user_id: str,
        campaign_data: dict[str, Any],
        owner_info: dict[str, str],
        sharing_options: dict[str, Any],
    ) -> dict[str, Any]:
        """Share a campaign on Nexus Hub.
        
        Returns:
            Dict with hub_id, share_url, and metadata
        """
        # Check quota
        quota = self._get_quota(user_id)
        estimated_size = len(json.dumps(campaign_data)) / (1024 * 1024)  # MB
        
        if not quota.can_share_new_campaign(estimated_size):
            raise QuotaExceededError(
                f"Free tier limit reached. "
                f"Campaigns: {quota.shared_campaigns}/{quota.MAX_SHARED_CAMPAIGNS}, "
                f"Storage: {quota.total_storage_mb:.1f}/{quota.MAX_STORAGE_MB} MB"
            )
        
        # Generate FAIR metadata
        fair_metadata = self.generate_fair_metadata(
            campaign_data, owner_info, sharing_options
        )
        
        # Create shared campaign record
        hub_id = f"hub_{uuid.uuid4().hex[:12]}"
        campaign_id = campaign_data.get("campaign_id", str(uuid.uuid4()))
        
        visibility = SharingVisibility(
            sharing_options.get("visibility", "unlisted")
        )
        access_level = AccessLevel(
            sharing_options.get("access_level", "view")
        )
        
        shared = SharedCampaign(
            hub_id=hub_id,
            campaign_id=campaign_id,
            owner_id=user_id,
            visibility=visibility,
            access_level=access_level,
            fair_metadata=fair_metadata,
            collaborators={},  # Can add later
            created_at=datetime.now().isoformat(),
            modified_at=datetime.now().isoformat(),
        )
        
        # Store
        self._shared_campaigns[hub_id] = shared
        
        # Update indexes
        if user_id not in self._user_campaigns:
            self._user_campaigns[user_id] = []
        self._user_campaigns[user_id].append(hub_id)
        
        if visibility == SharingVisibility.PUBLIC:
            self._public_index.append(hub_id)
        
        # Update keyword index
        for keyword in fair_metadata.keywords:
            if keyword not in self._keyword_index:
                self._keyword_index[keyword] = []
            self._keyword_index[keyword].append(hub_id)
        
        # Update quota
        quota.shared_campaigns += 1
        quota.total_storage_mb += estimated_size
        
        return {
            "hub_id": hub_id,
            "share_url": f"https://hub.nexus.dev/c/{hub_id}",
            "embed_url": f"https://hub.nexus.dev/e/{hub_id}",
            "fair_metadata": fair_metadata.to_dict(),
            "schemaorg_jsonld": fair_metadata.to_schemaorg_jsonld(),
            "datacite_xml": fair_metadata.to_datacite_xml(),
            "quota_remaining": {
                "campaigns": quota.campaigns_remaining,
                "storage_mb": round(quota.storage_remaining_mb, 2),
            },
        }
    
    def get_shared_campaign(self, hub_id: str, requester_id: str | None = None) -> SharedCampaign | None:
        """Retrieve a shared campaign if accessible to requester."""
        if hub_id not in self._shared_campaigns:
            return None
        
        shared = self._shared_campaigns[hub_id]
        
        # Check access
        if shared.visibility == SharingVisibility.PRIVATE:
            if requester_id != shared.owner_id:
                return None
        
        if shared.visibility == SharingVisibility.COLLABORATIVE:
            if requester_id not in shared.collaborators and requester_id != shared.owner_id:
                return None
        
        # Update view count
        shared.view_count += 1
        
        return shared
    
    def fork_campaign(
        self,
        hub_id: str,
        new_owner_id: str,
        new_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Fork (copy) a shared campaign to user's workspace."""
        shared = self.get_shared_campaign(hub_id, new_owner_id)
        if shared is None:
            return None
        
        # Check if forking is allowed
        # Fork is allowed if:
        # 1. Public visibility - anyone can fork
        # 2. Access level allows fork/edit
        # 3. User is a collaborator with appropriate permissions
        can_fork = (
            shared.visibility == SharingVisibility.PUBLIC or
            shared.access_level in (AccessLevel.FORK, AccessLevel.EDIT, AccessLevel.ADMIN) or
            new_owner_id in shared.collaborators or
            new_owner_id == shared.owner_id
        )
        
        if not can_fork:
            return None
        
        shared.fork_count += 1
        
        return {
            "original_hub_id": hub_id,
            "campaign_id": shared.campaign_id,
            "new_name": new_name or f"Fork of {shared.fair_metadata.title}",
            "fair_metadata": shared.fair_metadata.to_dict(),
        }
    
    # ── Quota Management ─────────────────────────────────────────────────
    
    def _get_quota(self, user_id: str) -> QuotaUsage:
        """Get or create quota record for user."""
        if user_id not in self._quota_usage:
            self._quota_usage[user_id] = QuotaUsage(user_id=user_id)
        return self._quota_usage[user_id]
    
    def get_quota_status(self, user_id: str) -> dict[str, Any]:
        """Get user's current quota status."""
        quota = self._get_quota(user_id)
        return quota.to_dict()
    
    # ── Discovery & Search ───────────────────────────────────────────────
    
    def search_public_campaigns(
        self,
        query: str | None = None,
        keywords: list[str] | None = None,
        domain: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Search public campaigns on the hub."""
        results = []
        
        # Get candidate hub_ids
        candidate_ids = set(self._public_index)
        
        # Filter by keywords
        if keywords:
            keyword_matches = set()
            for kw in keywords:
                if kw in self._keyword_index:
                    keyword_matches.update(self._keyword_index[kw])
            candidate_ids &= keyword_matches
        
        # Filter and score
        for hub_id in candidate_ids:
            shared = self._shared_campaigns.get(hub_id)
            if not shared:
                continue
            
            # Domain filter
            if domain and shared.fair_metadata.domain != domain:
                continue
            
            # Text search (simple)
            if query:
                searchable = (
                    shared.fair_metadata.title.lower() +
                    shared.fair_metadata.description.lower() +
                    " ".join(shared.fair_metadata.keywords).lower()
                )
                if query.lower() not in searchable:
                    continue
            
            results.append({
                "hub_id": shared.hub_id,
                "title": shared.fair_metadata.title,
                "description": shared.fair_metadata.description[:200],
                "creators": shared.fair_metadata.creators,
                "keywords": shared.fair_metadata.keywords,
                "view_count": shared.view_count,
                "fork_count": shared.fork_count,
                "created_at": shared.created_at,
                "domain": shared.fair_metadata.domain,
            })
        
        # Sort by popularity (views + forks)
        results.sort(key=lambda x: x["view_count"] + x["fork_count"] * 2, reverse=True)
        
        return results[offset:offset + limit]
    
    def get_user_shared_campaigns(self, user_id: str) -> list[dict[str, Any]]:
        """Get all campaigns shared by a user."""
        hub_ids = self._user_campaigns.get(user_id, [])
        return [
            {
                "hub_id": h_id,
                "title": self._shared_campaigns[h_id].fair_metadata.title,
                "visibility": self._shared_campaigns[h_id].visibility.value,
                "view_count": self._shared_campaigns[h_id].view_count,
                "created_at": self._shared_campaigns[h_id].created_at,
            }
            for h_id in hub_ids
            if h_id in self._shared_campaigns
        ]
    
    # ── Collaboration ────────────────────────────────────────────────────
    
    def add_collaborator(
        self,
        hub_id: str,
        owner_id: str,
        collaborator_id: str,
        access_level: AccessLevel,
    ) -> bool:
        """Add a collaborator to a shared campaign."""
        if hub_id not in self._shared_campaigns:
            return False
        
        shared = self._shared_campaigns[hub_id]
        if shared.owner_id != owner_id:
            return False
        
        shared.collaborators[collaborator_id] = access_level
        shared.modified_at = datetime.now().isoformat()
        
        return True
    
    def remove_collaborator(
        self,
        hub_id: str,
        owner_id: str,
        collaborator_id: str,
    ) -> bool:
        """Remove a collaborator."""
        if hub_id not in self._shared_campaigns:
            return False
        
        shared = self._shared_campaigns[hub_id]
        if shared.owner_id != owner_id:
            return False
        
        if collaborator_id in shared.collaborators:
            del shared.collaborators[collaborator_id]
            shared.modified_at = datetime.now().isoformat()
        
        return True


class QuotaExceededError(Exception):
    """Raised when user exceeds free tier quota."""
    pass

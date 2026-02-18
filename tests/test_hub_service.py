"""Tests for Nexus Hub service."""

import pytest
from datetime import datetime

from optimization_copilot.platform.hub_service import (
    NexusHubService,
    FAIRMetadata,
    SharedCampaign,
    SharingVisibility,
    AccessLevel,
    QuotaUsage,
    QuotaExceededError,
)


class TestFAIRMetadata:
    """Test FAIR metadata generation and export."""
    
    def test_fair_metadata_creation(self):
        """Test creating FAIR metadata."""
        metadata = FAIRMetadata(
            identifier="10.5281/nexus.abc123",
            title="Test Optimization Campaign",
            description="A test campaign for catalyst optimization",
            creators=[{"name": "Test User", "orcid": "0000-0000-0000-0000"}],
            keywords=["optimization", "catalysis", "bayesian"],
            publication_date="2024-01-15T10:00:00",
            modified_date="2024-01-15T10:00:00",
            access_url="https://hub.nexus.dev/campaign/abc123",
            download_url=None,
            access_level="open",
            schema_version="1.0",
            parameter_specs=[{"name": "temperature", "type": "continuous"}],
            objective_definitions=[{"name": "yield"}],
            license="MIT",
            methodology="Bayesian optimization using TPE",
            software_version="0.2.0",
            provenance=[{"activity": "optimization"}],
            citation="Test User (2024). Optimization Campaign Data.",
            domain="chemistry",
            experimental_context={},
        )
        
        assert metadata.identifier == "10.5281/nexus.abc123"
        assert metadata.title == "Test Optimization Campaign"
        assert "optimization" in metadata.keywords
    
    def test_to_schemaorg_jsonld(self):
        """Test Schema.org JSON-LD export."""
        metadata = FAIRMetadata(
            identifier="10.5281/nexus.abc123",
            title="Test Campaign",
            description="Test",
            creators=[{"name": "Test User"}],
            keywords=["test"],
            publication_date="2024-01-15",
            modified_date="2024-01-15",
            access_url="https://hub.nexus.dev/test",
            download_url=None,
            access_level="open",
            schema_version="1.0",
            parameter_specs=[],
            objective_definitions=[],
            license="MIT",
            methodology="Test",
            software_version="0.2.0",
            provenance=[],
            citation="Test",
            domain=None,
            experimental_context={},
        )
        
        jsonld = metadata.to_schemaorg_jsonld()
        assert "@context" in jsonld
        assert "https://schema.org" in jsonld
        assert "Test Campaign" in jsonld
        assert "MIT" in jsonld
    
    def test_to_datacite_xml(self):
        """Test DataCite XML export."""
        metadata = FAIRMetadata(
            identifier="10.5281/nexus.abc123",
            title="Test Campaign",
            description="Test description",
            creators=[{"name": "Test User", "orcid": "0000-0000-0000-0001"}],
            keywords=["optimization", "test"],
            publication_date="2024-01-15",
            modified_date="2024-01-15",
            access_url="https://hub.nexus.dev/test",
            download_url=None,
            access_level="open",
            schema_version="1.0",
            parameter_specs=[],
            objective_definitions=[],
            license="MIT",
            methodology="Test",
            software_version="0.2.0",
            provenance=[],
            citation="Test",
            domain=None,
            experimental_context={},
        )
        
        xml = metadata.to_datacite_xml()
        assert "<?xml version" in xml
        assert "10.5281/nexus.abc123" in xml
        assert "Test Campaign" in xml
        assert "Test User" in xml
        assert "optimization" in xml
        assert "test" in xml


class TestQuotaManagement:
    """Test free tier quota management."""
    
    def test_quota_initialization(self):
        """Test quota initialization."""
        quota = QuotaUsage(user_id="user_123")
        
        assert quota.user_id == "user_123"
        assert quota.shared_campaigns == 0
        assert quota.total_storage_mb == 0.0
        assert quota.campaigns_remaining == 10
    
    def test_quota_tracking(self):
        """Test quota usage tracking."""
        quota = QuotaUsage(user_id="user_123")
        
        # Simulate sharing campaigns
        quota.shared_campaigns = 5
        quota.total_storage_mb = 500.0
        
        assert quota.campaigns_remaining == 5
        assert quota.storage_remaining_mb == 500.0
    
    def test_can_share_new_campaign(self):
        """Test campaign sharing permission."""
        quota = QuotaUsage(user_id="user_123")
        
        # Should allow initially
        assert quota.can_share_new_campaign(estimated_size_mb=50.0)
        
        # Fill up campaigns
        quota.shared_campaigns = 10
        assert not quota.can_share_new_campaign()
        
        # Reset and fill storage
        quota.shared_campaigns = 0
        quota.total_storage_mb = 950.0
        assert not quota.can_share_new_campaign(estimated_size_mb=100.0)
        assert quota.can_share_new_campaign(estimated_size_mb=40.0)
    
    def test_quota_to_dict(self):
        """Test quota serialization."""
        quota = QuotaUsage(user_id="user_123")
        quota.shared_campaigns = 3
        quota.total_storage_mb = 150.5
        
        data = quota.to_dict()
        
        assert data["user_id"] == "user_123"
        assert data["shared_campaigns"] == 3
        assert data["total_storage_mb"] == 150.5
        assert "limits" in data
        assert "remaining" in data
        assert data["remaining"]["campaigns"] == 7


class TestNexusHubService:
    """Test Nexus Hub service functionality."""
    
    def test_service_initialization(self):
        """Test hub service initialization."""
        hub = NexusHubService()
        
        assert len(hub._shared_campaigns) == 0
        assert len(hub._user_campaigns) == 0
    
    def test_generate_fair_metadata(self):
        """Test FAIR metadata generation."""
        hub = NexusHubService()
        
        campaign_data = {
            "campaign_id": "camp_abc123",
            "parameter_specs": [
                {"name": "temperature", "type": "continuous", "lower": 0, "upper": 100},
            ],
            "objective_names": ["yield"],
            "observations": [{"iteration": 1, "parameters": {}, "kpi_values": {"yield": 0.5}}],
            "backend_name": "tpe",
        }
        
        owner_info = {
            "name": "Test User",
            "orcid": "0000-0000-0000-0001",
            "institution": "Test University",
        }
        
        sharing_options = {
            "title": "My Catalyst Optimization",
            "description": "Optimizing catalyst composition",
            "keywords": ["catalysis", "optimization"],
            "visibility": "public",
            "access_level": "view",
            "license": "MIT",
            "domain": "chemistry",
        }
        
        metadata = hub.generate_fair_metadata(campaign_data, owner_info, sharing_options)
        
        assert metadata.title == "My Catalyst Optimization"
        assert "catalysis" in metadata.keywords
        assert metadata.creators[0]["name"] == "Test User"
        assert metadata.domain == "chemistry"
        assert len(metadata.parameter_specs) == 1
        assert "tpe" in metadata.methodology.lower()
    
    def test_share_campaign(self):
        """Test sharing a campaign."""
        hub = NexusHubService()
        
        campaign_data = {
            "campaign_id": "camp_abc123",
            "parameter_specs": [],
            "objective_names": ["objective"],
        }
        
        owner_info = {"name": "Test User"}
        
        sharing_options = {
            "title": "Test Campaign",
            "description": "A test campaign",
            "keywords": ["test"],
            "visibility": "public",
            "access_level": "view",
            "license": "MIT",
        }
        
        result = hub.share_campaign("user_123", campaign_data, owner_info, sharing_options)
        
        assert "hub_id" in result
        assert "share_url" in result
        assert "fair_metadata" in result
        assert result["share_url"].startswith("https://hub.nexus.dev/c/")
        
        # Check it was stored
        assert result["hub_id"] in hub._shared_campaigns
        assert result["hub_id"] in hub._public_index
    
    def test_share_campaign_quota_enforcement(self):
        """Test quota enforcement when sharing."""
        hub = NexusHubService()
        
        # Fill up quota
        quota = hub._get_quota("user_123")
        quota.shared_campaigns = 10
        
        campaign_data = {"campaign_id": "camp_xyz", "parameter_specs": []}
        owner_info = {"name": "Test User"}
        sharing_options = {
            "title": "Test",
            "keywords": ["test"],
            "visibility": "public",
            "license": "MIT",
        }
        
        with pytest.raises(QuotaExceededError) as exc_info:
            hub.share_campaign("user_123", campaign_data, owner_info, sharing_options)
        
        assert "Free tier limit reached" in str(exc_info.value)
    
    def test_get_shared_campaign_public(self):
        """Test retrieving a public shared campaign."""
        hub = NexusHubService()
        
        # Share a campaign
        campaign_data = {"campaign_id": "camp_abc", "parameter_specs": []}
        result = hub.share_campaign(
            "user_123",
            campaign_data,
            {"name": "Test"},
            {"title": "Test", "keywords": ["test"], "visibility": "public", "license": "MIT"},
        )
        
        hub_id = result["hub_id"]
        
        # Anyone should be able to view
        shared = hub.get_shared_campaign(hub_id, requester_id="any_user")
        assert shared is not None
        assert shared.hub_id == hub_id
        assert shared.view_count == 1  # Incremented
    
    def test_get_shared_campaign_private(self):
        """Test retrieving a private campaign."""
        hub = NexusHubService()
        
        # Share a private campaign
        campaign_data = {"campaign_id": "camp_private", "parameter_specs": []}
        result = hub.share_campaign(
            "user_123",
            campaign_data,
            {"name": "Test"},
            {"title": "Test", "keywords": ["test"], "visibility": "private", "license": "MIT"},
        )
        
        hub_id = result["hub_id"]
        
        # Owner should see it
        shared = hub.get_shared_campaign(hub_id, requester_id="user_123")
        assert shared is not None
        
        # Others should not
        shared = hub.get_shared_campaign(hub_id, requester_id="other_user")
        assert shared is None
    
    def test_fork_campaign(self):
        """Test forking a campaign."""
        hub = NexusHubService()
        
        # Share a campaign with fork access
        campaign_data = {"campaign_id": "camp_fork", "parameter_specs": []}
        result = hub.share_campaign(
            "user_123",
            campaign_data,
            {"name": "Test"},
            {"title": "Test", "keywords": ["test"], "visibility": "public", "license": "MIT"},
        )
        
        hub_id = result["hub_id"]
        
        # Fork it
        fork_result = hub.fork_campaign(hub_id, "user_456", "My Fork")
        
        assert fork_result is not None
        assert fork_result["original_hub_id"] == hub_id
        assert fork_result["new_name"] == "My Fork"
        
        # Check fork count incremented
        shared = hub._shared_campaigns[hub_id]
        assert shared.fork_count == 1
    
    def test_search_public_campaigns(self):
        """Test searching public campaigns."""
        hub = NexusHubService()
        
        # Share multiple campaigns
        for i in range(3):
            hub.share_campaign(
                f"user_{i}",
                {"campaign_id": f"camp_{i}", "parameter_specs": []},
                {"name": f"User {i}"},
                {
                    "title": f"Campaign {i}",
                    "keywords": ["optimization", f"tag_{i}"],
                    "visibility": "public",
                    "license": "MIT",
                },
            )
        
        # Search all
        results = hub.search_public_campaigns(limit=10)
        assert len(results) == 3
        
        # Search by keyword
        results = hub.search_public_campaigns(keywords=["tag_0"])
        assert len(results) == 1
        assert results[0]["title"] == "Campaign 0"
    
    def test_add_collaborator(self):
        """Test adding collaborators."""
        hub = NexusHubService()
        
        # Share a collaborative campaign
        campaign_data = {"campaign_id": "camp_collab", "parameter_specs": []}
        result = hub.share_campaign(
            "user_123",
            campaign_data,
            {"name": "Test"},
            {"title": "Test", "keywords": ["test"], "visibility": "collaborative", "license": "MIT"},
        )
        
        hub_id = result["hub_id"]
        
        # Add collaborator
        success = hub.add_collaborator(hub_id, "user_123", "user_456", AccessLevel.EDIT)
        assert success
        
        # Verify
        shared = hub._shared_campaigns[hub_id]
        assert "user_456" in shared.collaborators
        assert shared.collaborators["user_456"] == AccessLevel.EDIT
    
    def test_remove_collaborator(self):
        """Test removing collaborators."""
        hub = NexusHubService()
        
        # Share and add collaborator
        campaign_data = {"campaign_id": "camp_collab", "parameter_specs": []}
        result = hub.share_campaign(
            "user_123",
            campaign_data,
            {"name": "Test"},
            {"title": "Test", "keywords": ["test"], "visibility": "collaborative", "license": "MIT"},
        )
        
        hub_id = result["hub_id"]
        hub.add_collaborator(hub_id, "user_123", "user_456", AccessLevel.VIEW)
        
        # Remove collaborator
        success = hub.remove_collaborator(hub_id, "user_123", "user_456")
        assert success
        
        # Verify
        shared = hub._shared_campaigns[hub_id]
        assert "user_456" not in shared.collaborators
    
    def test_get_user_shared_campaigns(self):
        """Test getting user's shared campaigns."""
        hub = NexusHubService()
        
        # Share campaigns as different users
        for i in range(2):
            hub.share_campaign(
                "user_abc",
                {"campaign_id": f"camp_{i}", "parameter_specs": []},
                {"name": "Test"},
                {"title": f"Campaign {i}", "keywords": ["test"], "visibility": "public", "license": "MIT"},
            )
        
        hub.share_campaign(
            "user_xyz",
            {"campaign_id": "camp_other", "parameter_specs": []},
            {"name": "Test"},
            {"title": "Other", "keywords": ["test"], "visibility": "public", "license": "MIT"},
        )
        
        # Get user_abc's campaigns
        campaigns = hub.get_user_shared_campaigns("user_abc")
        assert len(campaigns) == 2
        
        # Get user_xyz's campaigns
        campaigns = hub.get_user_shared_campaigns("user_xyz")
        assert len(campaigns) == 1

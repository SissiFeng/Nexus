"""FAIR metadata generation for optimization campaigns.

Implements the FAIR (Findable, Accessible, Interoperable, Reusable)
principles by generating structured metadata records for campaign data.

Provides ``FAIRMetadata`` (structured metadata dataclass) and
``FAIRGenerator`` (factory for creating metadata from campaigns and
snapshots).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from optimization_copilot.core.models import CampaignSnapshot, ParameterSpec


# ---------------------------------------------------------------------------
# FAIRMetadata
# ---------------------------------------------------------------------------


@dataclass
class FAIRMetadata:
    """FAIR-compliant metadata record for an optimization campaign.

    Attributes
    ----------
    identifier : str
        Unique identifier (UUID or DOI).
    title : str
        Human-readable title for the dataset.
    creators : list[str]
        Names of the dataset creators.
    description : str
        Free-text description of the dataset.
    keywords : list[str]
        Searchable keywords for discoverability.
    license : str
        SPDX license identifier.
    version : str
        Semantic version of the dataset.
    created : str
        ISO 8601 creation timestamp.
    modified : str
        ISO 8601 last-modified timestamp.
    format : str
        MIME type of the primary data format.
    access_rights : str
        Access level (``open``, ``restricted``, ``embargoed``, ``closed``).
    """

    identifier: str
    title: str
    creators: list[str]
    description: str
    keywords: list[str] = field(default_factory=list)
    license: str = "CC-BY-4.0"
    version: str = "1.0"
    created: str = ""
    modified: str = ""
    format: str = "application/json"
    access_rights: str = "open"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "identifier": self.identifier,
            "title": self.title,
            "creators": list(self.creators),
            "description": self.description,
            "keywords": list(self.keywords),
            "license": self.license,
            "version": self.version,
            "created": self.created,
            "modified": self.modified,
            "format": self.format,
            "access_rights": self.access_rights,
        }

    def to_json(self) -> str:
        """Serialize to a JSON string with pretty-printing."""
        return json.dumps(self.to_dict(), indent=2)

    def to_jsonld(self) -> str:
        """Serialize to JSON-LD format with schema.org context.

        Returns a JSON-LD string suitable for linked data consumption,
        mapping FAIR metadata fields to schema.org vocabulary.
        """
        jsonld = {
            "@context": "https://schema.org",
            "@type": "Dataset",
            "identifier": self.identifier,
            "name": self.title,
            "creator": [
                {"@type": "Person", "name": name} for name in self.creators
            ],
            "description": self.description,
            "keywords": self.keywords,
            "license": self.license,
            "version": self.version,
            "dateCreated": self.created,
            "dateModified": self.modified,
            "encodingFormat": self.format,
            "accessMode": self.access_rights,
        }
        return json.dumps(jsonld, indent=2)


# ---------------------------------------------------------------------------
# FAIRGenerator
# ---------------------------------------------------------------------------


class FAIRGenerator:
    """Factory for creating FAIR metadata from campaign data.

    Provides convenience methods for generating ``FAIRMetadata`` records
    from explicit parameters or from ``CampaignSnapshot`` instances.
    """

    @staticmethod
    def generate(
        campaign_id: str,
        title: str,
        creators: list[str],
        description: str = "",
        n_observations: int = 0,
        keywords: list[str] | None = None,
    ) -> FAIRMetadata:
        """Generate FAIR metadata from explicit parameters.

        Auto-fills the identifier (UUID4), creation and modification
        timestamps (current UTC ISO 8601), and merges user-supplied
        keywords with contextual keywords derived from the campaign.

        Parameters
        ----------
        campaign_id : str
            Campaign identifier (included in keywords).
        title : str
            Human-readable title.
        creators : list[str]
            Creator names.
        description : str
            Free-text description.
        n_observations : int
            Number of observations (included in description if > 0).
        keywords : list[str] or None
            Additional keywords to include.

        Returns
        -------
        FAIRMetadata
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        merged_keywords = ["optimization", "self-driving-lab", campaign_id]
        if keywords:
            for kw in keywords:
                if kw not in merged_keywords:
                    merged_keywords.append(kw)

        if not description and n_observations > 0:
            description = (
                f"Optimization campaign '{campaign_id}' "
                f"with {n_observations} observations."
            )

        return FAIRMetadata(
            identifier=str(uuid.uuid4()),
            title=title,
            creators=list(creators),
            description=description,
            keywords=merged_keywords,
            created=now_iso,
            modified=now_iso,
        )

    @staticmethod
    def from_snapshot(
        snapshot: CampaignSnapshot,
        title: str,
        creators: list[str],
    ) -> FAIRMetadata:
        """Generate FAIR metadata from a ``CampaignSnapshot``.

        Extracts the description from ``snapshot.metadata`` (using the
        ``"description"`` key if present), counts observations, and
        builds keywords from parameter names and objective names.

        Parameters
        ----------
        snapshot : CampaignSnapshot
            The campaign snapshot to extract metadata from.
        title : str
            Human-readable title.
        creators : list[str]
            Creator names.

        Returns
        -------
        FAIRMetadata
        """
        description = snapshot.metadata.get("description", "")
        n_observations = len(snapshot.observations)

        if not description:
            description = (
                f"Optimization campaign '{snapshot.campaign_id}' "
                f"with {n_observations} observations."
            )

        # Build keywords from parameter names and objective names
        keywords: list[str] = []
        for spec in snapshot.parameter_specs:
            if spec.name not in keywords:
                keywords.append(spec.name)
        for obj_name in snapshot.objective_names:
            if obj_name not in keywords:
                keywords.append(obj_name)

        now_iso = datetime.now(timezone.utc).isoformat()

        merged_keywords = [
            "optimization",
            "self-driving-lab",
            snapshot.campaign_id,
        ]
        for kw in keywords:
            if kw not in merged_keywords:
                merged_keywords.append(kw)

        return FAIRMetadata(
            identifier=str(uuid.uuid4()),
            title=title,
            creators=list(creators),
            description=description,
            keywords=merged_keywords,
            created=now_iso,
            modified=now_iso,
        )

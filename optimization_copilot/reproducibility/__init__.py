"""Reproducibility package for optimization campaigns.

Re-exports the core public API:

- ``EventType`` -- campaign event classification enum
- ``CampaignEvent`` -- individual logged event with UUID and timestamp
- ``CampaignLogger`` -- append-only event log with JSONL persistence
- ``ReplayResult`` -- comparison summary from campaign replay
- ``CampaignReplayer`` -- replays events and verifies determinism
- ``FAIRMetadata`` -- FAIR-compliant metadata record
- ``FAIRGenerator`` -- factory for creating FAIR metadata
"""

from optimization_copilot.reproducibility.logger import (
    CampaignEvent,
    CampaignLogger,
    EventType,
)
from optimization_copilot.reproducibility.replay import (
    CampaignReplayer,
    ReplayResult,
)
from optimization_copilot.reproducibility.fair import (
    FAIRGenerator,
    FAIRMetadata,
)

__all__ = [
    "CampaignEvent",
    "CampaignLogger",
    "CampaignReplayer",
    "EventType",
    "FAIRGenerator",
    "FAIRMetadata",
    "ReplayResult",
]

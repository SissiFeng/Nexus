"""Data models for search-space surgery actions and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums ──────────────────────────────────────────────


class ActionType(str, Enum):
    """Types of surgery actions that can be applied to the search space."""
    TIGHTEN_RANGE = "tighten_range"
    FREEZE_PARAMETER = "freeze_parameter"
    CONDITIONAL_FREEZE = "conditional_freeze"
    MERGE_PARAMETERS = "merge_parameters"
    DERIVE_PARAMETER = "derive_parameter"
    REMOVE_PARAMETER = "remove_parameter"


class DerivedType(str, Enum):
    """Types of derived parameter transformations."""
    LOG = "log"
    RATIO = "ratio"
    DIFFERENCE = "difference"
    PRODUCT = "product"


# ── Dataclasses ────────────────────────────────────────


@dataclass
class SurgeryAction:
    """A single surgery action to apply to the search space."""
    action_type: ActionType
    target_params: list[str]
    new_lower: float | None = None
    new_upper: float | None = None
    freeze_value: Any = None
    condition_param: str | None = None
    condition_threshold: float | None = None
    condition_direction: str | None = None  # "above" or "below"
    merge_into: str | None = None
    derived_type: DerivedType | None = None
    derived_name: str | None = None
    derived_source_params: list[str] | None = None
    reason: str = ""
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "target_params": list(self.target_params),
            "new_lower": self.new_lower,
            "new_upper": self.new_upper,
            "freeze_value": self.freeze_value,
            "condition_param": self.condition_param,
            "condition_threshold": self.condition_threshold,
            "condition_direction": self.condition_direction,
            "merge_into": self.merge_into,
            "derived_type": self.derived_type.value if self.derived_type is not None else None,
            "derived_name": self.derived_name,
            "derived_source_params": list(self.derived_source_params) if self.derived_source_params is not None else None,
            "reason": self.reason,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurgeryAction:
        data = data.copy()
        data["action_type"] = ActionType(data["action_type"])
        if data.get("derived_type") is not None:
            data["derived_type"] = DerivedType(data["derived_type"])
        else:
            data["derived_type"] = None
        return cls(**data)


@dataclass
class SurgeryReport:
    """Complete report of surgery actions and their impact on the search space."""
    actions: list[SurgeryAction] = field(default_factory=list)
    original_dim: int = 0
    effective_dim: int = 0
    space_reduction_ratio: float = 0.0
    reason_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_actions(self) -> int:
        """Number of surgery actions in this report."""
        return len(self.actions)

    @property
    def has_actions(self) -> bool:
        """Whether this report contains any surgery actions."""
        return len(self.actions) > 0

    def actions_by_type(self, action_type: ActionType) -> list[SurgeryAction]:
        """Return all actions matching the given type."""
        return [a for a in self.actions if a.action_type == action_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [a.to_dict() for a in self.actions],
            "original_dim": self.original_dim,
            "effective_dim": self.effective_dim,
            "space_reduction_ratio": self.space_reduction_ratio,
            "reason_codes": list(self.reason_codes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurgeryReport:
        data = data.copy()
        data["actions"] = [SurgeryAction.from_dict(a) for a in data["actions"]]
        return cls(**data)

"""Data models for the Problem Builder module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.dsl.spec import Direction, ParamType
from optimization_copilot.ingestion.models import ColumnRole


class BuilderError(Exception):
    """Raised when ProblemBuilder.build() fails validation."""
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        formatted = "\n".join(f"  - {e}" for e in errors)
        super().__init__(f"Problem build failed with {len(errors)} error(s):\n{formatted}")


@dataclass
class ColumnSuggestion:
    """A suggestion for how to use a data column."""
    column_name: str
    suggested_role: ColumnRole
    confidence: float
    param_type: ParamType | None = None  # For PARAMETER suggestions
    direction: Direction | None = None   # For KPI suggestions
    lower: float | None = None
    upper: float | None = None
    categories: list[str] | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "suggested_role": self.suggested_role.value,
            "confidence": self.confidence,
            "param_type": self.param_type.value if self.param_type else None,
            "direction": self.direction.value if self.direction else None,
            "lower": self.lower,
            "upper": self.upper,
            "categories": self.categories,
            "reason": self.reason,
        }


@dataclass
class ProblemSuggestions:
    """Collection of column role suggestions from ProblemGuide."""
    campaign_id: str
    n_rows: int
    n_columns: int
    all_suggestions: list[ColumnSuggestion]
    input_suggestions: list[ColumnSuggestion] = field(default_factory=list)
    objective_suggestions: list[ColumnSuggestion] = field(default_factory=list)
    metadata_suggestions: list[ColumnSuggestion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "all_suggestions": [s.to_dict() for s in self.all_suggestions],
            "input_suggestions": [s.to_dict() for s in self.input_suggestions],
            "objective_suggestions": [s.to_dict() for s in self.objective_suggestions],
            "metadata_suggestions": [s.to_dict() for s in self.metadata_suggestions],
        }

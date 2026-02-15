"""Problem Builder — fluent and guided problem definition for optimization campaigns."""

from optimization_copilot.problem_builder.builder import ProblemBuilder
from optimization_copilot.problem_builder.guide import ProblemGuide
from optimization_copilot.problem_builder.models import (
    BuilderError,
    ColumnSuggestion,
    ProblemSuggestions,
)

__all__ = [
    "BuilderError",
    "ColumnSuggestion",
    "ProblemBuilder",
    "ProblemGuide",
    "ProblemSuggestions",
]

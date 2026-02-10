"""Auto Curriculum Optimization: progressive difficulty staging for optimization campaigns."""

from optimization_copilot.curriculum.engine import (
    CurriculumEngine,
    CurriculumPlan,
    CurriculumPolicy,
    CurriculumStage,
)

__all__ = [
    "CurriculumStage",
    "CurriculumPolicy",
    "CurriculumPlan",
    "CurriculumEngine",
]

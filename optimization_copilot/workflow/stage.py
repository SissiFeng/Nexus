"""Stage definitions and DAG for multi-stage experimental workflows.

Provides :class:`ExperimentStage` for modelling individual experiment
stages and :class:`StageDAG` for representing their dependency graph as
a directed acyclic graph (DAG).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentStage:
    """Represents a single stage in a multi-stage experimental workflow.

    Parameters
    ----------
    name : str
        Unique identifier for the stage.
    parameters : list[str]
        Parameter names relevant to this stage.
    kpis : list[str]
        KPIs measured at this stage.
    cost : float
        Relative cost of running this stage (default 1.0).
    duration_hours : float
        Estimated duration in hours (default 1.0).
    dependencies : list[str]
        Names of prerequisite stages that must complete before this one.
    metadata : dict[str, Any]
        Arbitrary metadata for the stage.
    """

    name: str
    parameters: list[str]
    kpis: list[str]
    cost: float = 1.0
    duration_hours: float = 1.0
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class StageDAG:
    """Directed Acyclic Graph of experiment stages.

    Manages stage registration, dependency validation, and topological
    ordering for multi-stage experimental workflows.
    """

    def __init__(self) -> None:
        self._stages: dict[str, ExperimentStage] = {}
        self._adjacency: dict[str, list[str]] = {}  # stage -> successors

    def add_stage(self, stage: ExperimentStage) -> None:
        """Add a stage to the DAG.

        Parameters
        ----------
        stage : ExperimentStage
            The stage to add.

        Raises
        ------
        ValueError
            If a stage with the same name already exists.
        """
        if stage.name in self._stages:
            raise ValueError(f"Stage '{stage.name}' already exists in DAG")
        self._stages[stage.name] = stage
        if stage.name not in self._adjacency:
            self._adjacency[stage.name] = []
        # Register edges from dependencies -> this stage
        for dep in stage.dependencies:
            if dep not in self._adjacency:
                self._adjacency[dep] = []
            self._adjacency[dep].append(stage.name)

    def get_stage(self, name: str) -> ExperimentStage:
        """Retrieve a stage by name.

        Parameters
        ----------
        name : str
            The stage name.

        Returns
        -------
        ExperimentStage

        Raises
        ------
        KeyError
            If the stage does not exist.
        """
        if name not in self._stages:
            raise KeyError(f"Stage '{name}' not found in DAG")
        return self._stages[name]

    def topological_order(self) -> list[str]:
        """Return stages in topological order using Kahn's algorithm.

        Returns
        -------
        list[str]
            Stage names in a valid execution order.

        Raises
        ------
        ValueError
            If the graph contains a cycle.
        """
        # Compute in-degree for all known stages
        in_degree: dict[str, int] = {name: 0 for name in self._stages}
        for stage in self._stages.values():
            for dep in stage.dependencies:
                if dep in self._stages:
                    # dep -> stage.name edge means stage.name has +1 in-degree
                    pass  # counted below
        # Recompute properly
        for name in self._stages:
            in_degree[name] = 0
        for stage in self._stages.values():
            for dep in stage.dependencies:
                if dep in self._stages:
                    in_degree[stage.name] += 1

        queue: deque[str] = deque()
        for name, deg in in_degree.items():
            if deg == 0:
                queue.append(name)

        result: list[str] = []
        while queue:
            node = queue.popleft()
            result.append(node)
            # Find successors: stages that depend on this node
            for stage in self._stages.values():
                if node in stage.dependencies and stage.name in in_degree:
                    in_degree[stage.name] -= 1
                    if in_degree[stage.name] == 0:
                        queue.append(stage.name)

        if len(result) != len(self._stages):
            raise ValueError("StageDAG contains a cycle")
        return result

    def get_ready_stages(self, completed: set[str]) -> list[str]:
        """Return stages whose dependencies are all satisfied.

        Parameters
        ----------
        completed : set[str]
            Set of stage names that have been completed.

        Returns
        -------
        list[str]
            Stage names that are ready to execute (not yet completed,
            all dependencies met).
        """
        ready: list[str] = []
        for name, stage in self._stages.items():
            if name in completed:
                continue
            if all(dep in completed for dep in stage.dependencies):
                ready.append(name)
        return ready

    def validate(self) -> bool:
        """Validate the DAG for cycles and missing dependencies.

        Returns
        -------
        bool
            True if the DAG is valid (no cycles, all dependencies present).

        Raises
        ------
        ValueError
            If the DAG contains a cycle or references missing stages.
        """
        # Check for missing dependencies
        for stage in self._stages.values():
            for dep in stage.dependencies:
                if dep not in self._stages:
                    raise ValueError(
                        f"Stage '{stage.name}' depends on '{dep}' "
                        f"which is not in the DAG"
                    )

        # Check for cycles via topological sort
        self.topological_order()
        return True

    def total_cost(self) -> float:
        """Return the total cost of all stages.

        Returns
        -------
        float
            Sum of all stage costs.
        """
        return sum(stage.cost for stage in self._stages.values())

    def stages(self) -> list[ExperimentStage]:
        """Return all stages in the DAG.

        Returns
        -------
        list[ExperimentStage]
            All registered stages (order not guaranteed).
        """
        return list(self._stages.values())

    def get_successors(self, name: str) -> list[str]:
        """Return the names of stages that directly depend on the given stage.

        Parameters
        ----------
        name : str
            The stage name.

        Returns
        -------
        list[str]
            Names of successor stages.
        """
        successors: list[str] = []
        for stage in self._stages.values():
            if name in stage.dependencies:
                successors.append(stage.name)
        return successors

    def get_predecessors(self, name: str) -> list[str]:
        """Return the names of stages that the given stage depends on.

        Parameters
        ----------
        name : str
            The stage name.

        Returns
        -------
        list[str]
            Names of predecessor stages.
        """
        stage = self.get_stage(name)
        return list(stage.dependencies)

    def __len__(self) -> int:
        return len(self._stages)

    def __contains__(self, name: str) -> bool:
        return name in self._stages


def _experiment_stage_from_fidelity_stage(cls: type, fidelity_stage: Any) -> ExperimentStage:
    """Create an ExperimentStage from a FidelityStage.

    Maps FidelityStage attributes to ExperimentStage fields.

    Parameters
    ----------
    fidelity_stage
        A ``workflow.fidelity_graph.FidelityStage`` instance.

    Returns
    -------
    ExperimentStage
    """
    return cls(
        name=fidelity_stage.name,
        parameters=[],
        kpis=list(fidelity_stage.kpis),
        cost=fidelity_stage.cost,
        metadata={"fidelity_level": fidelity_stage.fidelity_level},
    )


# Attach as classmethod on ExperimentStage
ExperimentStage.from_fidelity_stage = classmethod(_experiment_stage_from_fidelity_stage)  # type: ignore[attr-defined]

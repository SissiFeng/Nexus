"""Fidelity level definitions and cost model for multi-fidelity optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FidelityLevel:
    """A single fidelity level.

    Parameters
    ----------
    name : str
        Human-readable identifier for this fidelity level.
    fidelity : float
        Fidelity value in [0.0, 1.0] where 1.0 is the highest fidelity.
    cost : float
        Relative cost of evaluating at this fidelity level.
    noise_multiplier : float
        Noise scaling factor. Lower fidelity levels typically have more noise.
    metadata : dict
        Arbitrary metadata for this fidelity level.
    """

    name: str
    fidelity: float
    cost: float
    noise_multiplier: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.fidelity <= 1.0:
            raise ValueError(
                f"fidelity must be in [0.0, 1.0], got {self.fidelity}"
            )
        if self.cost <= 0.0:
            raise ValueError(f"cost must be positive, got {self.cost}")
        if self.noise_multiplier < 0.0:
            raise ValueError(
                f"noise_multiplier must be non-negative, got {self.noise_multiplier}"
            )


@dataclass
class FidelityConfig:
    """Configuration for multi-fidelity optimization.

    Parameters
    ----------
    levels : list[FidelityLevel]
        Fidelity levels ordered from lowest to highest fidelity.
    target_fidelity : str
        Name of the highest fidelity level (the one we ultimately care about).
    cost_budget : float
        Total cost budget for the optimization campaign.
    """

    levels: list[FidelityLevel]
    target_fidelity: str
    cost_budget: float = 100.0

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("At least one fidelity level is required.")
        names = [lv.name for lv in self.levels]
        if self.target_fidelity not in names:
            raise ValueError(
                f"target_fidelity '{self.target_fidelity}' not found in levels: {names}"
            )

    def get_level(self, name: str) -> FidelityLevel:
        """Return the fidelity level with the given name.

        Raises
        ------
        KeyError
            If no level with the given name exists.
        """
        for lv in self.levels:
            if lv.name == name:
                return lv
        raise KeyError(f"Fidelity level '{name}' not found")

    def get_cost_ratio(self, name: str) -> float:
        """Return the cost ratio of the named level relative to the most expensive.

        Returns
        -------
        float
            cost(name) / max_cost across all levels.
        """
        level = self.get_level(name)
        max_cost = max(lv.cost for lv in self.levels)
        return level.cost / max_cost if max_cost > 0 else 1.0

    @property
    def n_levels(self) -> int:
        """Number of fidelity levels."""
        return len(self.levels)

    @property
    def target_level(self) -> FidelityLevel:
        """Return the target (highest) fidelity level."""
        return self.get_level(self.target_fidelity)


class CostModel:
    """Tracks cost budget during multi-fidelity optimization.

    Parameters
    ----------
    budget : float
        Total cost budget available.
    """

    def __init__(self, budget: float) -> None:
        self._budget = budget
        self._spent = 0.0

    def spend(self, amount: float) -> None:
        """Record a cost expenditure.

        Parameters
        ----------
        amount : float
            Cost amount to deduct from budget.

        Raises
        ------
        ValueError
            If amount is negative.
        """
        if amount < 0:
            raise ValueError(f"Cannot spend negative amount: {amount}")
        self._spent += amount

    @property
    def remaining(self) -> float:
        """Remaining budget."""
        return max(0.0, self._budget - self._spent)

    @property
    def spent(self) -> float:
        """Total amount spent so far."""
        return self._spent

    def can_afford(self, amount: float) -> bool:
        """Check whether the given amount fits within the remaining budget."""
        return amount <= self.remaining


def _fidelity_config_to_graph(config: FidelityConfig) -> Any:
    """Convert a FidelityConfig to a FidelityGraph.

    Creates a linear chain of stages ordered by fidelity value (ascending).
    No gates are added — callers should add gates after construction.

    Returns
    -------
    FidelityGraph
        A graph with one FidelityStage per FidelityLevel.
    """
    from optimization_copilot.workflow.fidelity_graph import FidelityGraph, FidelityStage

    graph = FidelityGraph()
    sorted_levels = sorted(config.levels, key=lambda lv: lv.fidelity)
    for lv in sorted_levels:
        stage = FidelityStage(
            name=lv.name,
            fidelity_level=int(lv.fidelity * 100),
            cost=lv.cost,
            kpis=list(lv.metadata.get("kpis", [])),
        )
        graph.add_stage(stage)
    return graph


# Attach as method on FidelityConfig
FidelityConfig.to_fidelity_graph = _fidelity_config_to_graph  # type: ignore[attr-defined]

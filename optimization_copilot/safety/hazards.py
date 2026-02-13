"""Hazard classification for safe optimization.

Provides a registry of hazard specifications that define safe operating
ranges for experimental parameters. Hazards are classified by category
(thermal, pressure, toxicity, etc.) and severity level.

Integrates with the ConstraintEngine for hard constraint enforcement:
hazard specs define the physical safety boundaries, while the constraint
engine handles the optimization-level filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HazardLevel(Enum):
    """Severity level of a hazard.

    Ordered from least to most severe (value 0-4).
    """

    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


class HazardCategory(Enum):
    """Physical category of a hazard."""

    THERMAL = "thermal"
    PRESSURE = "pressure"
    TOXICITY = "toxicity"
    FLAMMABILITY = "flammability"
    CORROSION = "corrosion"
    REACTIVITY = "reactivity"
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"


@dataclass
class HazardSpec:
    """Specification of a single hazard for a parameter.

    Attributes
    ----------
    parameter_name : str
        Name of the parameter this hazard applies to.
    category : HazardCategory
        Physical category of the hazard.
    level : HazardLevel
        Severity level if the hazard is triggered.
    lower_safe : float
        Lower bound of the safe operating range (inclusive).
    upper_safe : float
        Upper bound of the safe operating range (inclusive).
    description : str
        Human-readable description of the hazard.
    """

    parameter_name: str
    category: HazardCategory
    level: HazardLevel
    lower_safe: float
    upper_safe: float
    description: str = ""


class HazardRegistry:
    """Registry of hazard specifications for an experiment.

    Maintains a collection of HazardSpec entries and provides methods
    to classify parameter points against registered hazards.

    Examples
    --------
    >>> registry = HazardRegistry()
    >>> registry.register(HazardSpec(
    ...     parameter_name="temperature",
    ...     category=HazardCategory.THERMAL,
    ...     level=HazardLevel.HIGH,
    ...     lower_safe=20.0,
    ...     upper_safe=200.0,
    ...     description="Thermal decomposition above 200C",
    ... ))
    >>> registry.max_hazard_level({"temperature": 250.0})
    <HazardLevel.HIGH: 3>
    """

    def __init__(self) -> None:
        self._hazards: list[HazardSpec] = []

    def register(self, spec: HazardSpec) -> None:
        """Register a hazard specification.

        Parameters
        ----------
        spec : HazardSpec
            The hazard specification to register.
        """
        self._hazards.append(spec)

    def get_hazards(self, parameter_name: str) -> list[HazardSpec]:
        """Get all hazard specs for a given parameter.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter to look up.

        Returns
        -------
        list[HazardSpec]
            All registered hazards for this parameter.
        """
        return [h for h in self._hazards if h.parameter_name == parameter_name]

    def all_hazards(self) -> list[HazardSpec]:
        """Return all registered hazard specifications.

        Returns
        -------
        list[HazardSpec]
            A copy of all registered hazards.
        """
        return list(self._hazards)

    def classify_point(self, params: dict[str, float]) -> list[HazardSpec]:
        """Return all hazard specs violated by the given parameter point.

        A hazard is violated when the parameter value falls outside
        the ``[lower_safe, upper_safe]`` range.

        Parameters
        ----------
        params : dict[str, float]
            Mapping of parameter names to their values.

        Returns
        -------
        list[HazardSpec]
            All hazard specs that are violated.
        """
        violated: list[HazardSpec] = []
        for hazard in self._hazards:
            value = params.get(hazard.parameter_name)
            if value is not None:
                if value < hazard.lower_safe or value > hazard.upper_safe:
                    violated.append(hazard)
        return violated

    def max_hazard_level(self, params: dict[str, float]) -> HazardLevel:
        """Return the highest hazard level violated by the given point.

        Parameters
        ----------
        params : dict[str, float]
            Mapping of parameter names to their values.

        Returns
        -------
        HazardLevel
            The maximum hazard level among all violations, or
            ``HazardLevel.NONE`` if no hazards are violated.
        """
        violated = self.classify_point(params)
        if not violated:
            return HazardLevel.NONE
        return max(violated, key=lambda h: h.level.value).level

    @property
    def n_hazards(self) -> int:
        """Total number of registered hazard specifications."""
        return len(self._hazards)

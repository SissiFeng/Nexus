"""Dimension Analyzer: detects fixed dimensions and computes effective dimensionality.

Analyses parameter variability across observations in a CampaignSnapshot
to identify which dimensions are effectively constant (fixed) and which
actually vary.  Produces a simplification hint that downstream components
can use to choose lighter-weight optimization strategies when the
effective search space is small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    ParameterSpec,
    VariableType,
)


@dataclass
class DimensionAnalysis:
    """Result of dimension analysis.

    Attributes
    ----------
    fixed_dimensions : list[str]
        Parameter names that are constant across all observations.
    effective_dimensions : list[str]
        Parameter names that actually vary across observations.
    n_original : int
        Total parameter count from the snapshot specification.
    n_effective : int
        Number of non-fixed dimensions.
    simplification_hint : str
        One of ``"degenerate"``, ``"ranking"``, ``"bandit"``,
        ``"line_search"``, ``"reduced_bo"``, or ``"full_bo"``.
    """

    fixed_dimensions: list[str] = field(default_factory=list)
    effective_dimensions: list[str] = field(default_factory=list)
    n_original: int = 0
    n_effective: int = 0
    simplification_hint: str = "full_bo"


class DimensionAnalyzer:
    """Detects fixed dimensions and computes effective dimensionality.

    For each parameter in the snapshot's ``parameter_specs``, counts the
    number of unique values observed across all observations.  A parameter
    with at most one unique observed value is classified as *fixed*.

    The simplification hint is determined by the number of effective
    (non-fixed) dimensions:

    * ``n_effective == 0`` -- ``"degenerate"`` (all dimensions fixed)
    * ``n_effective == 1`` and categorical -- ``"ranking"`` (>5 unique
      values) or ``"bandit"`` (<=5 unique values)
    * ``n_effective == 1`` and continuous/discrete -- ``"line_search"``
    * ``n_effective <= 3`` -- ``"reduced_bo"``
    * otherwise -- ``"full_bo"``
    """

    def analyze(self, snapshot: CampaignSnapshot) -> DimensionAnalysis:
        """Analyze dimensions in the campaign snapshot.

        Parameters
        ----------
        snapshot : CampaignSnapshot
            The campaign state to analyse.

        Returns
        -------
        DimensionAnalysis
            Classification of fixed vs. effective dimensions with a
            simplification hint.
        """
        fixed: list[str] = []
        effective: list[str] = []

        for spec in snapshot.parameter_specs:
            unique_values: set[str] = set()
            for obs in snapshot.observations:
                val = obs.parameters.get(spec.name)
                if val is not None:
                    # Use string representation for hashability
                    unique_values.add(str(val))

            if len(unique_values) <= 1:
                fixed.append(spec.name)
            else:
                effective.append(spec.name)

        n_original = len(snapshot.parameter_specs)
        n_effective = len(effective)

        # Determine simplification hint
        hint = self._compute_hint(snapshot, effective, n_effective)

        return DimensionAnalysis(
            fixed_dimensions=fixed,
            effective_dimensions=effective,
            n_original=n_original,
            n_effective=n_effective,
            simplification_hint=hint,
        )

    @staticmethod
    def _compute_hint(
        snapshot: CampaignSnapshot,
        effective: list[str],
        n_effective: int,
    ) -> str:
        """Compute the simplification hint based on effective dimensions."""
        if n_effective == 0:
            return "degenerate"

        if n_effective == 1:
            # Find the spec for the single effective dimension
            eff_name = effective[0]
            eff_spec: ParameterSpec | None = None
            for s in snapshot.parameter_specs:
                if s.name == eff_name:
                    eff_spec = s
                    break

            if eff_spec is not None and eff_spec.type == VariableType.CATEGORICAL:
                # Count unique values to determine ranking vs bandit
                unique_count = len(set(
                    str(obs.parameters.get(eff_name))
                    for obs in snapshot.observations
                    if obs.parameters.get(eff_name) is not None
                ))
                return "ranking" if unique_count > 5 else "bandit"

            return "line_search"

        if n_effective <= 3:
            return "reduced_bo"

        return "full_bo"

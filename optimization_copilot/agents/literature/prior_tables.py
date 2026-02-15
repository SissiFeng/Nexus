"""Preset prior tables for known experimental domains.

Contains empirical prior distributions for process parameters
based on published literature and domain expertise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PriorEntry:
    """A single parameter prior from literature or domain knowledge.

    Parameters
    ----------
    parameter : str
        Parameter name (matching optimization parameter names).
    prior_mean : float
        Expected value for this parameter.
    prior_std : float
        Standard deviation of the prior distribution.
    source : str
        Citation or ``"domain_knowledge"``.
    domain : str
        Domain this prior applies to.
    notes : str
        Additional context or explanation.
    """

    parameter: str
    prior_mean: float
    prior_std: float
    source: str
    domain: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Zinc electrodeposition priors
# ---------------------------------------------------------------------------

ZINC_PRIORS: list[PriorEntry] = [
    PriorEntry(
        parameter="additive_1",
        prior_mean=0.3,
        prior_std=0.15,
        source="domain_knowledge",
        domain="electrochemistry",
        notes="Typical brightener range for acid zinc plating (0.1-0.5 g/L)",
    ),
    PriorEntry(
        parameter="additive_2",
        prior_mean=2.0,
        prior_std=1.0,
        source="domain_knowledge",
        domain="electrochemistry",
        notes="Typical leveler/carrier range for acid zinc plating (1-4 g/L)",
    ),
    PriorEntry(
        parameter="current_density",
        prior_mean=3.0,
        prior_std=1.5,
        source="domain_knowledge",
        domain="electrochemistry",
        notes="Common current density for zinc plating (1-5 A/dm2)",
    ),
    PriorEntry(
        parameter="temperature",
        prior_mean=30.0,
        prior_std=10.0,
        source="domain_knowledge",
        domain="electrochemistry",
        notes="Typical bath temperature for acid zinc (20-40 C)",
    ),
    PriorEntry(
        parameter="pH",
        prior_mean=4.5,
        prior_std=0.5,
        source="domain_knowledge",
        domain="electrochemistry",
        notes="Optimal pH range for acid zinc sulfate baths (4.0-5.0)",
    ),
    PriorEntry(
        parameter="Zn_concentration",
        prior_mean=0.4,
        prior_std=0.15,
        source="domain_knowledge",
        domain="electrochemistry",
        notes="Typical ZnSO4 concentration (0.2-0.6 M)",
    ),
    PriorEntry(
        parameter="agitation",
        prior_mean=300.0,
        prior_std=100.0,
        source="domain_knowledge",
        domain="electrochemistry",
        notes="Typical stirring speed (200-500 rpm)",
    ),
]


# ---------------------------------------------------------------------------
# Catalysis priors (Suzuki-Miyaura coupling)
# ---------------------------------------------------------------------------

CATALYSIS_PRIORS: list[PriorEntry] = [
    PriorEntry(
        parameter="temperature",
        prior_mean=80.0,
        prior_std=20.0,
        source="domain_knowledge",
        domain="catalysis",
        notes="Optimal temperature range for Suzuki coupling (60-100 C)",
    ),
    PriorEntry(
        parameter="catalyst_loading",
        prior_mean=2.0,
        prior_std=1.0,
        source="domain_knowledge",
        domain="catalysis",
        notes="Typical Pd catalyst loading (1-5 mol%)",
    ),
    PriorEntry(
        parameter="base_equivalents",
        prior_mean=2.5,
        prior_std=0.5,
        source="domain_knowledge",
        domain="catalysis",
        notes="Base equivalents for Suzuki coupling (2-3 equiv)",
    ),
    PriorEntry(
        parameter="reaction_time",
        prior_mean=12.0,
        prior_std=6.0,
        source="domain_knowledge",
        domain="catalysis",
        notes="Typical reaction time (4-24 hours)",
    ),
    PriorEntry(
        parameter="substrate_concentration",
        prior_mean=0.2,
        prior_std=0.1,
        source="domain_knowledge",
        domain="catalysis",
        notes="Typical substrate concentration (0.1-0.5 M)",
    ),
    PriorEntry(
        parameter="water_fraction",
        prior_mean=0.15,
        prior_std=0.1,
        source="domain_knowledge",
        domain="catalysis",
        notes="Water content in mixed solvent systems (0-0.3)",
    ),
]


# ---------------------------------------------------------------------------
# Perovskite solar cell priors
# ---------------------------------------------------------------------------

PEROVSKITE_PRIORS: list[PriorEntry] = [
    PriorEntry(
        parameter="annealing_temperature",
        prior_mean=100.0,
        prior_std=15.0,
        source="domain_knowledge",
        domain="perovskite",
        notes="Annealing temperature for MAPbI3 films (80-120 C)",
    ),
    PriorEntry(
        parameter="annealing_time",
        prior_mean=10.0,
        prior_std=5.0,
        source="domain_knowledge",
        domain="perovskite",
        notes="Annealing time for perovskite crystallization (5-30 min)",
    ),
    PriorEntry(
        parameter="precursor_concentration",
        prior_mean=1.2,
        prior_std=0.3,
        source="domain_knowledge",
        domain="perovskite",
        notes="Typical precursor concentration in DMF (0.8-1.5 M)",
    ),
    PriorEntry(
        parameter="spin_speed",
        prior_mean=4000.0,
        prior_std=1000.0,
        source="domain_knowledge",
        domain="perovskite",
        notes="Spin coating speed (2000-6000 rpm)",
    ),
    PriorEntry(
        parameter="antisolvent_delay",
        prior_mean=8.0,
        prior_std=3.0,
        source="domain_knowledge",
        domain="perovskite",
        notes="Antisolvent dripping delay time (5-15 s)",
    ),
    PriorEntry(
        parameter="PbI2_excess",
        prior_mean=5.0,
        prior_std=3.0,
        source="domain_knowledge",
        domain="perovskite",
        notes="PbI2 molar excess percentage (0-10%)",
    ),
    PriorEntry(
        parameter="halide_ratio",
        prior_mean=0.8,
        prior_std=0.1,
        source="domain_knowledge",
        domain="perovskite",
        notes="I/(I+Br) ratio for mixed halide perovskites (0.6-1.0)",
    ),
]


# ---------------------------------------------------------------------------
# PriorTable class
# ---------------------------------------------------------------------------

_DOMAIN_TABLES: dict[str, list[PriorEntry]] = {
    "electrochemistry": ZINC_PRIORS,
    "catalysis": CATALYSIS_PRIORS,
    "perovskite": PEROVSKITE_PRIORS,
}


class PriorTable:
    """Look up prior information for parameters in a given domain.

    Provides access to domain-specific parameter priors from literature
    and domain expertise. Supports custom priors via ``add_custom_prior``.
    """

    def __init__(self) -> None:
        # Copy the tables to avoid mutating module-level data
        self._tables: dict[str, list[PriorEntry]] = {
            domain: list(entries)
            for domain, entries in _DOMAIN_TABLES.items()
        }

    def get_priors(self, domain: str) -> list[PriorEntry]:
        """Return all prior entries for a given domain.

        Parameters
        ----------
        domain : str
            Domain name.

        Returns
        -------
        list[PriorEntry]
            Prior entries, or empty list if domain unknown.
        """
        return list(self._tables.get(domain, []))

    def get_prior_for_parameter(
        self, domain: str, parameter: str
    ) -> PriorEntry | None:
        """Look up the prior for a specific parameter in a domain.

        Parameters
        ----------
        domain : str
            Domain name.
        parameter : str
            Parameter name.

        Returns
        -------
        PriorEntry | None
            Prior entry, or *None* if not found.
        """
        for entry in self._tables.get(domain, []):
            if entry.parameter == parameter:
                return entry
        return None

    def list_domains(self) -> list[str]:
        """Return sorted list of domains with available priors."""
        return sorted(self._tables.keys())

    def add_custom_prior(self, entry: PriorEntry) -> None:
        """Add a custom prior entry.

        If the domain does not exist yet, it is created. If a prior
        for the same parameter already exists in the domain, it is
        replaced.

        Parameters
        ----------
        entry : PriorEntry
            Custom prior to add.
        """
        domain = entry.domain
        if domain not in self._tables:
            self._tables[domain] = []

        # Replace existing entry for same parameter if present
        self._tables[domain] = [
            e for e in self._tables[domain]
            if e.parameter != entry.parameter
        ]
        self._tables[domain].append(entry)

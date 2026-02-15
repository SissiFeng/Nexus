"""Hypothesis templates for domain-specific mechanism matching.

Each template describes a known physical/chemical phenomenon with
the pattern it produces in optimization data and the evidence
required to confirm or refute the hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HypothesisTemplate:
    """A template describing a known domain-specific phenomenon.

    Parameters
    ----------
    name : str
        Unique identifier for this hypothesis template.
    domain : str
        Domain this template applies to (e.g. ``"electrochemistry"``).
    pattern : str
        Human-readable description of the data pattern this template matches.
    mechanism : str
        Proposed mechanism explanation.
    parameters_involved : list[str]
        Parameter names that are relevant to this mechanism.
    evidence_required : list[str]
        Data/observations that would confirm or refute the hypothesis.
    confidence_prior : float
        Base confidence before evidence matching (0.0 to 1.0).
    """

    name: str
    domain: str
    pattern: str
    mechanism: str
    parameters_involved: list[str] = field(default_factory=list)
    evidence_required: list[str] = field(default_factory=list)
    confidence_prior: float = 0.5


# ---------------------------------------------------------------------------
# Electrochemistry templates
# ---------------------------------------------------------------------------

ELECTROCHEMISTRY_TEMPLATES: list[HypothesisTemplate] = [
    HypothesisTemplate(
        name="passivation_threshold",
        domain="electrochemistry",
        pattern="Sudden CE drop above concentration X",
        mechanism=(
            "Additive forms passivation layer on cathode surface above a critical "
            "concentration, blocking active sites and reducing current efficiency."
        ),
        parameters_involved=["additive_concentration", "additive_1", "additive_2"],
        evidence_required=["EIS Rct increase", "CE drop > 20%"],
        confidence_prior=0.7,
    ),
    HypothesisTemplate(
        name="hydrogen_evolution_competition",
        domain="electrochemistry",
        pattern="CE decreases as current density increases beyond threshold",
        mechanism=(
            "At high current densities, hydrogen evolution reaction competes with "
            "zinc deposition, reducing cathodic efficiency. The HER overpotential "
            "becomes more favorable at higher polarization."
        ),
        parameters_involved=["current_density", "j"],
        evidence_required=["CE drop with j increase", "Gas bubble observation"],
        confidence_prior=0.8,
    ),
    HypothesisTemplate(
        name="mass_transport_limit",
        domain="electrochemistry",
        pattern="KPI plateau at high current density with agitation dependence",
        mechanism=(
            "Zinc ion depletion at the cathode surface limits deposition rate. "
            "The limiting current density depends on Zn2+ concentration and "
            "mass transport (stirring, flow rate)."
        ),
        parameters_involved=["current_density", "flow_rate", "Zn_concentration", "agitation"],
        evidence_required=["Plateau in deposition rate", "Dendritic morphology"],
        confidence_prior=0.75,
    ),
    HypothesisTemplate(
        name="additive_synergy",
        domain="electrochemistry",
        pattern="Performance improvement when two additives used together exceeds sum of individual effects",
        mechanism=(
            "Synergistic interaction between brightener and leveler additives: "
            "one additive enhances adsorption of the other, producing smoother "
            "deposits than either alone."
        ),
        parameters_involved=["additive_1", "additive_2", "brightener", "leveler"],
        evidence_required=[
            "Superlinear improvement with both additives",
            "Surface roughness decrease",
        ],
        confidence_prior=0.6,
    ),
    HypothesisTemplate(
        name="temperature_crystallization_transition",
        domain="electrochemistry",
        pattern="Abrupt change in deposit morphology at temperature threshold",
        mechanism=(
            "Temperature-dependent transition in preferred crystallographic "
            "orientation. Higher bath temperature promotes (002) basal plane "
            "growth, while lower temperature favors random orientation."
        ),
        parameters_involved=["temperature", "bath_temperature"],
        evidence_required=["XRD texture change", "SEM morphology transition"],
        confidence_prior=0.65,
    ),
]


# ---------------------------------------------------------------------------
# Catalysis templates
# ---------------------------------------------------------------------------

CATALYSIS_TEMPLATES: list[HypothesisTemplate] = [
    HypothesisTemplate(
        name="catalyst_poisoning",
        domain="catalysis",
        pattern="Yield drops sharply after initial good performance",
        mechanism=(
            "Product or byproduct poisons the catalyst active sites, "
            "reducing turnover. Common with Pd catalysts when halide "
            "byproducts accumulate."
        ),
        parameters_involved=["catalyst_loading", "substrate_concentration"],
        evidence_required=["Decreasing yield over time", "Catalyst color change"],
        confidence_prior=0.7,
    ),
    HypothesisTemplate(
        name="temperature_selectivity_tradeoff",
        domain="catalysis",
        pattern="Higher temperature increases conversion but decreases selectivity",
        mechanism=(
            "Elevated temperature activates undesired side reactions (homo-coupling, "
            "proto-dehalogenation) that compete with the target cross-coupling. "
            "The activation energy for side reactions is typically higher."
        ),
        parameters_involved=["temperature", "reaction_temperature"],
        evidence_required=["Conversion increase with T", "Selectivity decrease with T"],
        confidence_prior=0.75,
    ),
    HypothesisTemplate(
        name="base_stoichiometry_effect",
        domain="catalysis",
        pattern="Optimal yield at specific base equivalents with decline above/below",
        mechanism=(
            "Base is required for transmetalation step in Suzuki coupling. "
            "Insufficient base stalls the catalytic cycle; excess base "
            "promotes protodeboronation of the boronic acid partner."
        ),
        parameters_involved=["base_equivalents", "base_concentration", "base_amount"],
        evidence_required=["Yield peak at ~2-3 equiv base", "Boronic acid consumption"],
        confidence_prior=0.8,
    ),
    HypothesisTemplate(
        name="solvent_polarity_effect",
        domain="catalysis",
        pattern="Reaction rate varies systematically with solvent polarity",
        mechanism=(
            "Polar solvents stabilize charged intermediates in the catalytic cycle "
            "(oxidative addition, transmetalation), affecting reaction rate and "
            "selectivity. Optimal polarity depends on substrate electronics."
        ),
        parameters_involved=["solvent", "solvent_ratio", "water_fraction"],
        evidence_required=["Rate correlation with ET(30)", "Selectivity change with solvent"],
        confidence_prior=0.6,
    ),
    HypothesisTemplate(
        name="ligand_steric_electronic_balance",
        domain="catalysis",
        pattern="Non-monotonic yield with ligand bulkiness or electron density",
        mechanism=(
            "Bulky ligands accelerate reductive elimination but hinder oxidative "
            "addition. Electron-rich ligands favor oxidative addition but slow "
            "reductive elimination. Optimal ligand balances both steps."
        ),
        parameters_involved=["ligand", "ligand_loading", "ligand_type"],
        evidence_required=["Yield maximum at intermediate ligand properties"],
        confidence_prior=0.65,
    ),
]


# ---------------------------------------------------------------------------
# Perovskite templates
# ---------------------------------------------------------------------------

PEROVSKITE_TEMPLATES: list[HypothesisTemplate] = [
    HypothesisTemplate(
        name="antisolvent_crystallization_window",
        domain="perovskite",
        pattern="PCE is highly sensitive to antisolvent dripping timing",
        mechanism=(
            "Antisolvent must be applied during a narrow window of spin-coating "
            "when the wet film reaches the correct supersaturation level. "
            "Too early: excess solvent remains. Too late: uncontrolled nucleation."
        ),
        parameters_involved=[
            "antisolvent_delay", "spin_speed", "antisolvent_volume",
        ],
        evidence_required=["PCE peak at specific delay time", "Film uniformity correlation"],
        confidence_prior=0.8,
    ),
    HypothesisTemplate(
        name="halide_segregation",
        domain="perovskite",
        pattern="Mixed-halide perovskite shows PL red-shift under illumination",
        mechanism=(
            "Under illumination, mixed I/Br perovskites undergo halide segregation "
            "forming iodide-rich domains with lower bandgap. This creates charge "
            "traps and reduces Voc."
        ),
        parameters_involved=["halide_ratio", "I_fraction", "Br_fraction"],
        evidence_required=["PL red-shift", "Voc decrease under illumination"],
        confidence_prior=0.75,
    ),
    HypothesisTemplate(
        name="annealing_temperature_phase",
        domain="perovskite",
        pattern="Non-perovskite phase appears above/below annealing threshold",
        mechanism=(
            "Perovskite formation requires sufficient thermal energy for "
            "conversion, but excessive temperature causes decomposition to "
            "PbI2 or other non-perovskite phases. Optimal window is narrow."
        ),
        parameters_involved=["annealing_temperature", "annealing_time"],
        evidence_required=[
            "XRD non-perovskite peaks",
            "PCE drop at extreme temperatures",
        ],
        confidence_prior=0.8,
    ),
    HypothesisTemplate(
        name="precursor_stoichiometry_defects",
        domain="perovskite",
        pattern="Slight PbI2 excess improves performance",
        mechanism=(
            "Small excess of PbI2 passivates grain boundaries and reduces "
            "recombination losses. However, large excess creates insulating "
            "layers that block charge transport."
        ),
        parameters_involved=[
            "PbI2_excess", "precursor_ratio", "MAI_concentration",
        ],
        evidence_required=[
            "XRD PbI2 peak at 12.7 deg",
            "PCE maximum at ~5% excess",
        ],
        confidence_prior=0.7,
    ),
    HypothesisTemplate(
        name="humidity_degradation",
        domain="perovskite",
        pattern="Performance degrades rapidly in humid conditions",
        mechanism=(
            "Water molecules attack the organic cation (MA+) causing "
            "irreversible decomposition of the perovskite crystal structure "
            "to PbI2 and volatile organic compounds."
        ),
        parameters_involved=["humidity", "encapsulation", "environmental_rh"],
        evidence_required=[
            "Stability decrease with humidity",
            "Yellow PbI2 formation",
        ],
        confidence_prior=0.85,
    ),
]


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------

_ALL_TEMPLATES: dict[str, list[HypothesisTemplate]] = {
    "electrochemistry": ELECTROCHEMISTRY_TEMPLATES,
    "catalysis": CATALYSIS_TEMPLATES,
    "perovskite": PEROVSKITE_TEMPLATES,
}


def get_templates_for_domain(domain: str) -> list[HypothesisTemplate]:
    """Return hypothesis templates for a given domain.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"electrochemistry"``).

    Returns
    -------
    list[HypothesisTemplate]
        Templates for the domain, or an empty list if unknown.
    """
    return list(_ALL_TEMPLATES.get(domain, []))


def get_all_templates() -> list[HypothesisTemplate]:
    """Return all hypothesis templates across all domains."""
    result: list[HypothesisTemplate] = []
    for templates in _ALL_TEMPLATES.values():
        result.extend(templates)
    return result

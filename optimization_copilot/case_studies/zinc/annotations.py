"""Anomaly labels and known mechanism stubs for zinc electrodeposition.

Provides domain-expert annotations for future v6 agent verification:

* **Anomaly labels**: known failure/degradation modes and their conditions.
* **Known mechanisms**: additive roles and their electrochemical mechanisms.
* **Observation annotator**: flags observations with relevant domain knowledge.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Anomaly labels
# ---------------------------------------------------------------------------

ZINC_ANOMALY_LABELS: list[dict[str, Any]] = [
    {
        "type": "passivation",
        "additive_range": {"additive_1": (0.8, 1.0)},
        "description": (
            "Excessive additive_1 causes passivation layer formation on the "
            "cathode surface, reducing coulombic efficiency"
        ),
    },
    {
        "type": "dendrite_formation",
        "condition": "sum(additives) > 0.95",
        "description": (
            "High total additive loading promotes dendrite growth at the "
            "zinc electrode, leading to short circuits and capacity loss"
        ),
    },
    {
        "type": "hydrogen_evolution",
        "additive_range": {"additive_2": (0.0, 0.02)},
        "description": (
            "Very low additive_2 concentration fails to suppress parasitic "
            "hydrogen evolution reaction, reducing efficiency"
        ),
    },
    {
        "type": "bath_decomposition",
        "additive_range": {"additive_4": (0.7, 1.0), "additive_5": (0.7, 1.0)},
        "description": (
            "High concentrations of additive_4 and additive_5 together cause "
            "plating bath decomposition and precipitate formation"
        ),
    },
]


# ---------------------------------------------------------------------------
# Known mechanisms
# ---------------------------------------------------------------------------

ZINC_KNOWN_MECHANISMS: dict[str, dict[str, str]] = {
    "leveling": {
        "primary_additive": "additive_1",
        "mechanism": "adsorption-inhibition",
        "description": (
            "Additive_1 adsorbs preferentially on protrusions, inhibiting "
            "local deposition and producing a level zinc deposit"
        ),
    },
    "brightening": {
        "primary_additive": "additive_2",
        "mechanism": "grain_refinement",
        "description": (
            "Additive_2 promotes fine-grain nucleation, producing a bright "
            "and reflective zinc coating"
        ),
    },
    "throwing_power": {
        "primary_additive": "additive_3",
        "mechanism": "diffusion_control",
        "description": (
            "Additive_3 modifies the diffusion layer, improving the throwing "
            "power and deposit uniformity in recessed areas"
        ),
    },
    "corrosion_resistance": {
        "primary_additive": "additive_4",
        "mechanism": "alloy_incorporation",
        "description": (
            "Additive_4 co-deposits with zinc to form a more corrosion-"
            "resistant alloy phase"
        ),
    },
    "stress_relief": {
        "primary_additive": "additive_5",
        "mechanism": "lattice_relaxation",
        "description": (
            "Additive_5 relieves internal stress in the zinc deposit by "
            "modifying the crystal growth direction"
        ),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_anomaly_labels() -> list[dict[str, Any]]:
    """Return the list of known anomaly / failure mode labels.

    Returns
    -------
    list[dict]
        Each dict contains ``type``, ``additive_range`` or ``condition``,
        and ``description`` keys.
    """
    return list(ZINC_ANOMALY_LABELS)


def get_known_mechanisms() -> dict[str, dict[str, str]]:
    """Return the dictionary of known electrochemical mechanisms.

    Returns
    -------
    dict[str, dict[str, str]]
        Keyed by mechanism name (``leveling``, ``brightening``, etc.).
    """
    return dict(ZINC_KNOWN_MECHANISMS)


def annotate_observation(
    x: dict[str, float],
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Add annotation flags to an observation based on known mechanisms.

    Parameters
    ----------
    x : dict[str, float]
        Input point (parameter values keyed by name).
    result : dict | None
        Evaluation result (``{obj_name: {"value": ..., "variance": ...}}``)
        or ``None`` for a failed experiment.

    Returns
    -------
    dict[str, Any]
        Annotation dict with keys:
        - ``anomaly_flags``: list of triggered anomaly labels
        - ``mechanism_flags``: list of active mechanism names
        - ``failed``: whether the experiment failed (result is None)
    """
    anomaly_flags: list[dict[str, Any]] = []
    mechanism_flags: list[str] = []

    # -- Check anomaly conditions ------------------------------------------

    for label in ZINC_ANOMALY_LABELS:
        triggered = False

        # Range-based anomalies
        if "additive_range" in label:
            all_in_range = True
            for param, (lo, hi) in label["additive_range"].items():
                val = x.get(param, 0.0)
                if not (lo <= val <= hi):
                    all_in_range = False
                    break
            if all_in_range:
                triggered = True

        # Condition-based anomalies (sum constraint)
        if "condition" in label and "sum(additives)" in label["condition"]:
            # Parse threshold from condition string like "sum(additives) > 0.95"
            parts = label["condition"].split(">")
            if len(parts) == 2:
                try:
                    threshold = float(parts[1].strip())
                    additive_sum = sum(
                        x.get(f"additive_{i}", 0.0) for i in range(1, 8)
                    )
                    if additive_sum > threshold:
                        triggered = True
                except ValueError:
                    pass

        if triggered:
            anomaly_flags.append({
                "type": label["type"],
                "description": label["description"],
            })

    # -- Check which mechanisms are active ---------------------------------

    for mech_name, mech_info in ZINC_KNOWN_MECHANISMS.items():
        primary = mech_info["primary_additive"]
        val = x.get(primary, 0.0)
        # A mechanism is considered "active" when its primary additive is
        # present at a meaningful concentration (> 0.01).
        if val > 0.01:
            mechanism_flags.append(mech_name)

    return {
        "anomaly_flags": anomaly_flags,
        "mechanism_flags": mechanism_flags,
        "failed": result is None,
    }

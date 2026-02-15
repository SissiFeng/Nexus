"""Data loader for Suzuki-Miyaura catalysis coupling benchmark.

Generates synthetic 6D mixed-parameter data for Suzuki coupling yield
optimisation.  All data generation uses the Python stdlib ``random`` module
for reproducibility -- no external dependencies.

Search Space
------------
4 categorical + 2 continuous parameters:

* ``catalyst``: categorical, 4 options (CATALYSTS from domain knowledge)
* ``ligand``: categorical, 6 options (LIGANDS from domain knowledge)
* ``base``: categorical, 5 options (BASES from domain knowledge)
* ``solvent``: categorical, 4 options (THF, DMF, dioxane, toluene)
* ``temperature``: continuous, [40, 120] deg-C
* ``time``: continuous, [1, 24] hours

Objective
---------
``yield`` (maximize, unit "%", range 0-100 %).

Data Generation
---------------
* Seeded ``random.Random(seed)`` for full reproducibility.
* True function: base yield from catalyst-ligand lookup + temperature
  optimum around 80 deg-C (quadratic penalty) + diminishing log-shaped
  time benefit (plateaus at ~12 h) + solvent and base bonuses.
* Incompatible catalyst-ligand pairs (from domain knowledge) yield 0 %.
* Noise: uniform ~2 % std.
* Known optimum yield: ~95 %.
"""

from __future__ import annotations

import math
import random
from typing import Any

from optimization_copilot.domain_knowledge.catalysis import (
    CATALYSTS,
    LIGANDS,
    BASES,
    KNOWN_INCOMPATIBILITIES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOLVENTS: list[str] = ["THF", "DMF", "dioxane", "toluene"]

_TEMP_RANGE: tuple[float, float] = (40.0, 120.0)
_TIME_RANGE: tuple[float, float] = (1.0, 24.0)

_KNOWN_OPTIMUM_YIELD: float = 95.0

_NOISE_STD: float = 2.0  # ~2 % standard deviation


# ---------------------------------------------------------------------------
# Catalyst-ligand base yield lookup table
# ---------------------------------------------------------------------------

def _build_catalyst_ligand_table() -> dict[tuple[str, str], float]:
    """Build a lookup table for catalyst-ligand pair base yields.

    Best combinations:
    - Pd(PPh3)4 + XPhos: ~75 % base
    - Pd2(dba)3 + SPhos: ~75 % base

    Other combinations range from 40-70 %.
    """
    # Default base yields per catalyst (moderate)
    catalyst_base: dict[str, float] = {
        "Pd(OAc)2": 55.0,
        "Pd(PPh3)4": 65.0,
        "PdCl2": 50.0,
        "Pd2(dba)3": 62.0,
    }

    # Ligand modifiers (bonus on top of catalyst base)
    ligand_modifier: dict[str, float] = {
        "PPh3": 0.0,
        "XPhos": 5.0,
        "SPhos": 4.0,
        "BINAP": 2.0,
        "dppf": 3.0,
        "PCy3": 1.0,
    }

    # Special synergistic combinations get extra boost
    synergy_bonus: dict[tuple[str, str], float] = {
        ("Pd(PPh3)4", "XPhos"): 5.0,
        ("Pd2(dba)3", "SPhos"): 9.0,
        ("Pd(PPh3)4", "SPhos"): 3.0,
        ("Pd2(dba)3", "XPhos"): 4.0,
        ("Pd(OAc)2", "PPh3"): 2.0,
        ("Pd(OAc)2", "dppf"): 3.0,
    }

    table: dict[tuple[str, str], float] = {}
    for cat in CATALYSTS:
        for lig in LIGANDS:
            base = catalyst_base.get(cat, 50.0)
            mod = ligand_modifier.get(lig, 0.0)
            syn = synergy_bonus.get((cat, lig), 0.0)
            table[(cat, lig)] = base + mod + syn

    return table


_CATALYST_LIGAND_TABLE: dict[tuple[str, str], float] = (
    _build_catalyst_ligand_table()
)


# ---------------------------------------------------------------------------
# Solvent and base effect tables
# ---------------------------------------------------------------------------

_SOLVENT_BONUS: dict[str, float] = {
    "THF": 5.0,
    "DMF": 3.0,
    "dioxane": 1.0,
    "toluene": 0.0,
}

_BASE_BONUS: dict[str, float] = {
    "Cs2CO3": 5.0,
    "K2CO3": 3.0,
    "KOtBu": 1.5,
    "Et3N": 1.0,
    "DBU": 0.5,
}


# ---------------------------------------------------------------------------
# CatalysisDataLoader
# ---------------------------------------------------------------------------


class CatalysisDataLoader:
    """Generates synthetic Suzuki coupling reaction data.

    Parameters
    ----------
    n_points : int
        Number of data points to generate.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_points: int = 150, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._n_points = n_points
        self._data = self._generate()

    # -- public API --------------------------------------------------------

    def get_data(self) -> dict[str, Any]:
        """Return generated data in encoded form for surrogate fitting.

        Returns
        -------
        dict
            ``{"X": list[list[float]],
              "Y": {"yield": list[float]},
              "noise_levels": {"yield": float}}``

            X is encoded: one-hot for categoricals (catalyst 4, ligand 6,
            base 5, solvent 4) + raw continuous (temperature, time).
            Total dimension = 21.
        """
        return self._data

    def get_known_optimum(self) -> dict[str, float]:
        """Return the known optimum objective value.

        Returns
        -------
        dict[str, float]
            ``{"yield": 95.0}``
        """
        return {"yield": _KNOWN_OPTIMUM_YIELD}

    def get_search_space(self) -> dict[str, dict[str, Any]]:
        """Return the search space definition.

        Returns
        -------
        dict[str, dict]
            6 parameters: 4 categorical, 2 continuous.  The key ordering
            matches the encoding used in ``_encode_point`` and is consistent
            with ``ReplayBenchmark._encode()``.
        """
        # Use a regular dict -- Python 3.7+ preserves insertion order
        space: dict[str, dict[str, Any]] = {}
        space["catalyst"] = {
            "type": "categorical",
            "categories": list(CATALYSTS),
        }
        space["ligand"] = {
            "type": "categorical",
            "categories": list(LIGANDS),
        }
        space["base"] = {
            "type": "categorical",
            "categories": list(BASES),
        }
        space["solvent"] = {
            "type": "categorical",
            "categories": list(SOLVENTS),
        }
        space["temperature"] = {
            "type": "continuous",
            "range": list(_TEMP_RANGE),
        }
        space["time"] = {
            "type": "continuous",
            "range": list(_TIME_RANGE),
        }
        return space

    # -- true function (public for testing) --------------------------------

    def _true_function(
        self,
        catalyst: str,
        ligand: str,
        base: str,
        solvent: str,
        temperature: float,
        time: float,
    ) -> float:
        """Compute the noiseless true yield for a set of reaction conditions.

        Parameters
        ----------
        catalyst, ligand, base, solvent : str
            Categorical parameter values.
        temperature : float
            Reaction temperature in deg-C.
        time : float
            Reaction time in hours.

        Returns
        -------
        float
            Yield in [0, 100] %.
        """
        # Check incompatibility -- yield 0 % for known bad pairs
        for entry in KNOWN_INCOMPATIBILITIES:
            if entry["catalyst"] == catalyst and entry["ligand"] == ligand:
                return 0.0

        # 1. Base yield from catalyst-ligand combination
        base_yield = _CATALYST_LIGAND_TABLE.get(
            (catalyst, ligand), 50.0
        )

        # 2. Temperature effect: optimum at 80 deg-C, quadratic penalty
        temp_penalty = -0.005 * (temperature - 80.0) ** 2

        # 3. Time effect: diminishing log-shaped returns, plateaus at ~12 h
        #    Normalised so that time=12 gives +10 and time>=12 is capped.
        time_benefit = 10.0 * min(
            1.0, math.log(1.0 + time) / math.log(13.0)
        )

        # 4. Solvent bonus
        solvent_bonus = _SOLVENT_BONUS.get(solvent, 0.0)

        # 5. Base (reagent) bonus
        base_bonus = _BASE_BONUS.get(base, 0.0)

        # Combine
        y = base_yield + temp_penalty + time_benefit + solvent_bonus + base_bonus

        # Clamp to physical range
        return max(0.0, min(100.0, y))

    # -- encoding ----------------------------------------------------------

    def _encode_point(
        self,
        catalyst: str,
        ligand: str,
        base: str,
        solvent: str,
        temperature: float,
        time: float,
    ) -> list[float]:
        """Encode a single point into the numeric vector used by the surrogate.

        Encoding order must match ``get_search_space()`` iteration order:
        catalyst (4) | ligand (6) | base (5) | solvent (4) | temperature | time
        = 21 dimensions.
        """
        encoded: list[float] = []

        # Catalyst one-hot (4)
        for cat in CATALYSTS:
            encoded.append(1.0 if catalyst == cat else 0.0)

        # Ligand one-hot (6)
        for lig in LIGANDS:
            encoded.append(1.0 if ligand == lig else 0.0)

        # Base one-hot (5)
        for b in BASES:
            encoded.append(1.0 if base == b else 0.0)

        # Solvent one-hot (4)
        for sol in SOLVENTS:
            encoded.append(1.0 if solvent == sol else 0.0)

        # Continuous parameters (raw values)
        encoded.append(float(temperature))
        encoded.append(float(time))

        return encoded

    # -- data generation (internal) ----------------------------------------

    def _generate(self) -> dict[str, Any]:
        """Generate ``n_points`` synthetic Suzuki coupling data points.

        Returns
        -------
        dict
            ``{"X": [...], "Y": {"yield": [...]},
              "noise_levels": {"yield": float}}``
        """
        X: list[list[float]] = []
        Y_yield: list[float] = []

        for _ in range(self._n_points):
            catalyst = self._rng.choice(CATALYSTS)
            ligand = self._rng.choice(LIGANDS)
            base = self._rng.choice(BASES)
            solvent = self._rng.choice(SOLVENTS)
            temperature = self._rng.uniform(*_TEMP_RANGE)
            time = self._rng.uniform(*_TIME_RANGE)

            true_val = self._true_function(
                catalyst, ligand, base, solvent, temperature, time
            )
            noise = self._rng.gauss(0.0, _NOISE_STD)
            observed = max(0.0, min(100.0, true_val + noise))

            encoded = self._encode_point(
                catalyst, ligand, base, solvent, temperature, time
            )
            X.append(encoded)
            Y_yield.append(observed)

        return {
            "X": X,
            "Y": {"yield": Y_yield},
            "noise_levels": {"yield": _NOISE_STD},
        }

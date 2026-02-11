"""Data loader for perovskite thin film composition benchmark.

Generates synthetic realistic 8D data for perovskite solar cell composition
optimisation.  All data generation uses the Python stdlib ``random`` module
for reproducibility -- no external dependencies.

Search Space
------------
8 continuous parameters:

Cation fractions (simplex: FA + MA + Cs = 1.0):
- ``FA``  : formamidinium, [0, 1]
- ``MA``  : methylammonium, [0, 1]
- ``Cs``  : cesium, [0, 1]

Halide fractions (simplex: I + Br = 1.0):
- ``I``   : iodide, [0, 1]
- ``Br``  : bromide, [0, 1]

Process parameters:
- ``annealing_temp``  : [80, 200] deg C
- ``spin_speed``      : [1000, 6000] rpm
- ``precursor_conc``  : [0.5, 2.0] M

Objectives (multi-objective)
----------------------------
- ``PCE``       : power conversion efficiency (maximize, unit "%", range 0-33%)
- ``stability`` : operational stability (maximize, unit "hours", range 0-1000)

Data Generation
---------------
* Seeded ``random.Random(seed)`` for full reproducibility.
* True PCE function: quadratic penalties from optimal composition with
  process parameter effects.  Peak PCE ~23 % (realistic for perovskite).
* True stability function: Cs and Br fractions increase stability;
  trade-off with PCE.  Optimal stability ~800 hours.
* Phase stability violations cause dramatic stability drops.
* Noise: PCE ~0.5 % std, stability ~20 hours std.
"""

from __future__ import annotations

import math
import random
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CATION_NAMES: list[str] = ["FA", "MA", "Cs"]
_HALIDE_NAMES: list[str] = ["I", "Br"]
_PROCESS_NAMES: list[str] = ["annealing_temp", "spin_speed", "precursor_conc"]

_PARAM_NAMES: list[str] = _CATION_NAMES + _HALIDE_NAMES + _PROCESS_NAMES
_N_PARAMS: int = len(_PARAM_NAMES)

# Optimal composition
_OPTIMAL_FA: float = 0.80
_OPTIMAL_MA: float = 0.05
_OPTIMAL_CS: float = 0.15
_OPTIMAL_I: float = 0.85
_OPTIMAL_BR: float = 0.15

# Optimal process parameters
_OPTIMAL_ANNEAL: float = 130.0   # deg C
_OPTIMAL_SPIN: float = 4000.0    # rpm
_OPTIMAL_CONC: float = 1.2       # M

# Process parameter ranges (for normalisation)
_ANNEAL_RANGE: tuple[float, float] = (80.0, 200.0)
_SPIN_RANGE: tuple[float, float] = (1000.0, 6000.0)
_CONC_RANGE: tuple[float, float] = (0.5, 2.0)

# Objective targets
_PEAK_PCE: float = 23.0          # %
_PEAK_STABILITY: float = 800.0   # hours

# Noise standard deviations
_PCE_NOISE_STD: float = 0.5      # %
_STABILITY_NOISE_STD: float = 20.0  # hours


# ---------------------------------------------------------------------------
# PerovskiteDataLoader
# ---------------------------------------------------------------------------


class PerovskiteDataLoader:
    """Generates synthetic perovskite thin film composition data.

    Parameters
    ----------
    n_points : int
        Number of data points to generate.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_points: int = 120, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._n_points = n_points
        self._data = self._generate()

    # -- public API --------------------------------------------------------

    def get_data(self) -> dict[str, Any]:
        """Return generated data.

        Returns
        -------
        dict
            ``{"X": list[list[float]],
              "Y": {"PCE": list[float], "stability": list[float]},
              "noise_levels": {"PCE": float, "stability": float}}``
        """
        return self._data

    def get_known_optimum(self) -> dict[str, float]:
        """Return known optimum objective values.

        Returns
        -------
        dict[str, float]
            ``{"PCE": 23.0, "stability": 800.0}``
        """
        return {"PCE": _PEAK_PCE, "stability": _PEAK_STABILITY}

    def get_search_space(self) -> dict[str, dict[str, Any]]:
        """Return the search space definition.

        Returns
        -------
        dict[str, dict]
            8 continuous parameters with their ranges.
        """
        space: dict[str, dict[str, Any]] = {}
        # Cation fractions
        for name in _CATION_NAMES:
            space[name] = {"type": "continuous", "range": [0.0, 1.0]}
        # Halide fractions
        for name in _HALIDE_NAMES:
            space[name] = {"type": "continuous", "range": [0.0, 1.0]}
        # Process parameters
        space["annealing_temp"] = {
            "type": "continuous",
            "range": [_ANNEAL_RANGE[0], _ANNEAL_RANGE[1]],
        }
        space["spin_speed"] = {
            "type": "continuous",
            "range": [_SPIN_RANGE[0], _SPIN_RANGE[1]],
        }
        space["precursor_conc"] = {
            "type": "continuous",
            "range": [_CONC_RANGE[0], _CONC_RANGE[1]],
        }
        return space

    # -- simplex sampling --------------------------------------------------

    def _sample_simplex(self, n_components: int) -> list[float]:
        """Sample uniformly from the (n_components - 1)-simplex.

        Uses the sorted-uniform method: draw (n-1) uniform values on [0,1],
        sort them, and take consecutive differences.

        Parameters
        ----------
        n_components : int
            Number of components (sum to 1.0).

        Returns
        -------
        list[float]
            Fractions summing to 1.0, each in [0, 1].
        """
        cuts = sorted(self._rng.random() for _ in range(n_components - 1))
        fractions: list[float] = []
        prev = 0.0
        for c in cuts:
            fractions.append(c - prev)
            prev = c
        fractions.append(1.0 - prev)
        return fractions

    # -- true objective functions ------------------------------------------

    def _true_pce(self, x: list[float]) -> float:
        """Synthetic PCE landscape.

        Parameters
        ----------
        x : list[float]
            8-element vector [FA, MA, Cs, I, Br, annealing_temp, spin_speed,
            precursor_conc].

        Returns
        -------
        float
            True PCE value in % (before noise).
        """
        fa, ma, cs, i_frac, br = x[0], x[1], x[2], x[3], x[4]
        anneal, spin, conc = x[5], x[6], x[7]

        # Composition quadratic penalties
        comp_penalty = (
            15.0 * (fa - _OPTIMAL_FA) ** 2
            + 20.0 * (ma - _OPTIMAL_MA) ** 2
            + 18.0 * (cs - _OPTIMAL_CS) ** 2
            + 12.0 * (i_frac - _OPTIMAL_I) ** 2
            + 12.0 * (br - _OPTIMAL_BR) ** 2
        )

        # Process parameter penalties (normalised to [0, 1] range)
        anneal_norm = (anneal - _ANNEAL_RANGE[0]) / (
            _ANNEAL_RANGE[1] - _ANNEAL_RANGE[0]
        )
        anneal_opt_norm = (_OPTIMAL_ANNEAL - _ANNEAL_RANGE[0]) / (
            _ANNEAL_RANGE[1] - _ANNEAL_RANGE[0]
        )
        spin_norm = (spin - _SPIN_RANGE[0]) / (
            _SPIN_RANGE[1] - _SPIN_RANGE[0]
        )
        spin_opt_norm = (_OPTIMAL_SPIN - _SPIN_RANGE[0]) / (
            _SPIN_RANGE[1] - _SPIN_RANGE[0]
        )
        conc_norm = (conc - _CONC_RANGE[0]) / (
            _CONC_RANGE[1] - _CONC_RANGE[0]
        )
        conc_opt_norm = (_OPTIMAL_CONC - _CONC_RANGE[0]) / (
            _CONC_RANGE[1] - _CONC_RANGE[0]
        )

        process_penalty = (
            8.0 * (anneal_norm - anneal_opt_norm) ** 2
            + 6.0 * (spin_norm - spin_opt_norm) ** 2
            + 5.0 * (conc_norm - conc_opt_norm) ** 2
        )

        pce = _PEAK_PCE - comp_penalty - process_penalty

        # Clamp to physical range
        return max(0.0, min(33.0, pce))

    def _true_stability(self, x: list[float]) -> float:
        """Synthetic stability landscape.

        Parameters
        ----------
        x : list[float]
            8-element vector [FA, MA, Cs, I, Br, annealing_temp, spin_speed,
            precursor_conc].

        Returns
        -------
        float
            True stability value in hours (before noise).
        """
        fa, ma, cs, i_frac, br = x[0], x[1], x[2], x[3], x[4]
        anneal = x[5]

        # Base stability
        base = 400.0

        # Cs contribution: more Cs -> more stable (up to ~0.3)
        cs_bonus = 800.0 * cs * math.exp(-3.0 * max(cs - 0.30, 0.0))

        # Br contribution: more Br -> slightly more stable
        br_bonus = 200.0 * br * math.exp(-2.0 * max(br - 0.40, 0.0))

        # FA penalty: very high FA reduces stability
        fa_penalty = 300.0 * max(fa - 0.70, 0.0) ** 2

        # MA penalty: MA degrades under heat/moisture
        ma_penalty = 150.0 * ma

        # Annealing temperature effect: moderate annealing improves crystallinity
        anneal_norm = (anneal - _ANNEAL_RANGE[0]) / (
            _ANNEAL_RANGE[1] - _ANNEAL_RANGE[0]
        )
        anneal_bonus = 100.0 * math.exp(
            -8.0 * (anneal_norm - 0.5) ** 2
        )

        # Phase stability check: if FA > 0.85, delta-phase forms
        phase_penalty = 0.0
        if fa > 0.85:
            phase_penalty = 500.0
        # Poor phase stability: low Cs and low Br
        if cs < 0.05 and br < 0.1:
            phase_penalty += 300.0

        stability = (
            base
            + cs_bonus
            + br_bonus
            + anneal_bonus
            - fa_penalty
            - ma_penalty
            - phase_penalty
        )

        # Clamp to non-negative, capped at 1000
        return max(0.0, min(1000.0, stability))

    # -- data generation ---------------------------------------------------

    def _generate(self) -> dict[str, Any]:
        """Generate synthetic data points.

        Points are sampled with simplex constraints on cation and halide
        fractions, and uniform sampling on process parameters.
        """
        X: list[list[float]] = []
        Y_pce: list[float] = []
        Y_stab: list[float] = []

        for _ in range(self._n_points):
            # Sample cation fractions on 3-simplex
            cations = self._sample_simplex(3)  # [FA, MA, Cs]

            # Sample halide fractions on 2-simplex
            halides = self._sample_simplex(2)  # [I, Br]

            # Sample process parameters uniformly
            anneal = self._rng.uniform(*_ANNEAL_RANGE)
            spin = self._rng.uniform(*_SPIN_RANGE)
            conc = self._rng.uniform(*_CONC_RANGE)

            point = cations + halides + [anneal, spin, conc]

            # Compute true values
            true_pce = self._true_pce(point)
            true_stab = self._true_stability(point)

            # Add noise
            noisy_pce = true_pce + self._rng.gauss(0.0, _PCE_NOISE_STD)
            noisy_stab = true_stab + self._rng.gauss(0.0, _STABILITY_NOISE_STD)

            # Clamp after noise
            noisy_pce = max(0.0, min(33.0, noisy_pce))
            noisy_stab = max(0.0, min(1000.0, noisy_stab))

            X.append(point)
            Y_pce.append(noisy_pce)
            Y_stab.append(noisy_stab)

        return {
            "X": X,
            "Y": {"PCE": Y_pce, "stability": Y_stab},
            "noise_levels": {
                "PCE": _PCE_NOISE_STD,
                "stability": _STABILITY_NOISE_STD,
            },
        }

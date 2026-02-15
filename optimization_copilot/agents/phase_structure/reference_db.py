"""XRD reference database for phase identification.

Contains peak positions and relative intensities for common phases
encountered in zinc electrodeposition and related processes, sourced
from the Crystallography Open Database (COD) subset.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Reference peak data (COD subset)
# ---------------------------------------------------------------------------

REFERENCE_PEAKS: dict[str, dict[str, Any]] = {
    "Zn": {
        "crystal_system": "hexagonal",
        "space_group": "P63/mmc",
        "peaks_2theta": [36.3, 39.0, 43.2, 54.3, 70.1, 70.7, 77.0, 82.1, 86.6],
        "relative_intensity": [100, 28, 60, 40, 25, 21, 35, 8, 12],
        "miller_indices": [
            "(002)", "(100)", "(101)", "(102)", "(103)",
            "(110)", "(004)", "(112)", "(201)",
        ],
    },
    "ZnO": {
        "crystal_system": "hexagonal",
        "space_group": "P63mc",
        "peaks_2theta": [
            31.8, 34.4, 36.3, 47.5, 56.6, 62.9, 66.4, 67.9, 69.1, 72.6, 77.0,
        ],
        "relative_intensity": [57, 44, 100, 23, 32, 29, 4, 23, 11, 3, 4],
        "miller_indices": [
            "(100)", "(002)", "(101)", "(102)", "(110)", "(103)",
            "(200)", "(112)", "(201)", "(004)", "(202)",
        ],
    },
    "ZnOH2": {
        "crystal_system": "orthorhombic",
        "space_group": "Pmc21",
        "peaks_2theta": [20.2, 20.9, 27.3, 27.8, 32.5, 33.4, 34.6, 36.4, 40.1],
        "relative_intensity": [100, 45, 30, 25, 50, 35, 20, 15, 10],
        "miller_indices": [
            "(100)", "(020)", "(021)", "(110)", "(111)",
            "(200)", "(201)", "(002)", "(211)",
        ],
    },
    "ZnSO4": {
        "crystal_system": "orthorhombic",
        "space_group": "Pnma",
        "peaks_2theta": [14.8, 17.6, 21.0, 24.4, 25.1, 29.5, 30.2, 32.8, 34.0],
        "relative_intensity": [80, 60, 100, 45, 35, 70, 50, 30, 25],
        "miller_indices": [
            "(011)", "(101)", "(111)", "(020)", "(102)",
            "(121)", "(211)", "(022)", "(301)",
        ],
    },
}


# ---------------------------------------------------------------------------
# ReferenceDB class
# ---------------------------------------------------------------------------


class ReferenceDB:
    """XRD reference database for phase identification.

    Provides lookup and matching of observed XRD peaks against
    reference patterns from the COD subset.
    """

    def __init__(self, references: dict[str, dict[str, Any]] | None = None) -> None:
        """Initialize with reference data.

        Parameters
        ----------
        references : dict | None
            Custom reference data. Defaults to ``REFERENCE_PEAKS``.
        """
        self._references = references if references is not None else dict(REFERENCE_PEAKS)

    def get_reference(self, phase_name: str) -> dict[str, Any] | None:
        """Look up reference data for a given phase.

        Parameters
        ----------
        phase_name : str
            Phase identifier (e.g. ``"Zn"``, ``"ZnO"``).

        Returns
        -------
        dict | None
            Reference data dict, or *None* if phase is not in the database.
        """
        return self._references.get(phase_name)

    def list_phases(self) -> list[str]:
        """Return a sorted list of all phase names in the database."""
        return sorted(self._references.keys())

    def match_peaks(
        self,
        observed_peaks: list[float],
        tolerance: float = 0.3,
    ) -> list[tuple[str, list[tuple[float, float]], float]]:
        """Match observed peak positions against all reference phases.

        For each reference phase, counts how many observed peaks fall
        within ``tolerance`` degrees of a reference peak position.

        Parameters
        ----------
        observed_peaks : list[float]
            Observed 2-theta peak positions.
        tolerance : float
            Matching tolerance in degrees 2-theta.

        Returns
        -------
        list[tuple[str, list[tuple[float, float]], float]]
            List of ``(phase_name, matched_pairs, score)`` sorted by
            descending score. ``matched_pairs`` are ``(observed, reference)``
            pairs. ``score`` is the fraction of reference peaks matched
            (0.0 to 1.0).
        """
        if not observed_peaks:
            return []

        results: list[tuple[str, list[tuple[float, float]], float]] = []

        for phase_name, ref_data in self._references.items():
            ref_peaks = ref_data.get("peaks_2theta", [])
            if not ref_peaks:
                continue

            matched_pairs: list[tuple[float, float]] = []
            used_obs: set[int] = set()

            for ref_peak in ref_peaks:
                best_dist = tolerance + 1.0
                best_obs_idx = -1
                for obs_idx, obs_peak in enumerate(observed_peaks):
                    if obs_idx in used_obs:
                        continue
                    dist = abs(obs_peak - ref_peak)
                    if dist <= tolerance and dist < best_dist:
                        best_dist = dist
                        best_obs_idx = obs_idx

                if best_obs_idx >= 0:
                    matched_pairs.append(
                        (observed_peaks[best_obs_idx], ref_peak)
                    )
                    used_obs.add(best_obs_idx)

            score = len(matched_pairs) / len(ref_peaks) if ref_peaks else 0.0
            if matched_pairs:
                results.append((phase_name, matched_pairs, score))

        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def get_texture_coefficient(
        self,
        observed_peaks: list[float],
        observed_intensities: list[float],
        phase_name: str,
        hkl_num: str = "(002)",
        hkl_den: str = "(100)",
        tolerance: float = 0.3,
    ) -> float | None:
        """Compute texture coefficient for a phase.

        The texture coefficient is the ratio of observed intensities
        at two Miller indices normalised by the reference ratio::

            TC = (I_obs(hkl_num) / I_obs(hkl_den)) / (I_ref(hkl_num) / I_ref(hkl_den))

        A TC > 1 indicates preferred orientation along ``hkl_num``.

        Parameters
        ----------
        observed_peaks : list[float]
            Observed 2-theta positions.
        observed_intensities : list[float]
            Observed intensities corresponding to ``observed_peaks``.
        phase_name : str
            Phase to compute TC for.
        hkl_num : str
            Miller index for the numerator.
        hkl_den : str
            Miller index for the denominator.
        tolerance : float
            Matching tolerance in degrees.

        Returns
        -------
        float | None
            Texture coefficient, or *None* if required peaks are not found.
        """
        ref = self._references.get(phase_name)
        if ref is None:
            return None

        ref_peaks = ref.get("peaks_2theta", [])
        ref_intensities = ref.get("relative_intensity", [])
        ref_millers = ref.get("miller_indices", [])

        if not ref_peaks or not ref_intensities or not ref_millers:
            return None
        if len(observed_peaks) != len(observed_intensities):
            return None

        # Find reference positions and intensities for the two hkl indices
        def _find_ref(hkl: str) -> tuple[float, float] | None:
            for i, m in enumerate(ref_millers):
                if m == hkl:
                    return ref_peaks[i], float(ref_intensities[i])
            return None

        ref_num = _find_ref(hkl_num)
        ref_den = _find_ref(hkl_den)

        if ref_num is None or ref_den is None:
            return None
        if ref_den[1] < 1e-10:
            return None

        # Find observed intensities at those positions
        def _find_obs(target_2theta: float) -> float | None:
            best_dist = tolerance + 1.0
            best_intensity: float | None = None
            for i, obs_pos in enumerate(observed_peaks):
                dist = abs(obs_pos - target_2theta)
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_intensity = observed_intensities[i]
            return best_intensity

        obs_num_int = _find_obs(ref_num[0])
        obs_den_int = _find_obs(ref_den[0])

        if obs_num_int is None or obs_den_int is None:
            return None
        if obs_den_int < 1e-10:
            return None

        # TC = (I_obs_num / I_obs_den) / (I_ref_num / I_ref_den)
        obs_ratio = obs_num_int / obs_den_int
        ref_ratio = ref_num[1] / ref_den[1]

        if ref_ratio < 1e-10:
            return None

        return obs_ratio / ref_ratio

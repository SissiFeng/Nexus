"""PhaseStructureAgent -- XRD-based phase identification and structural analysis.

Matches observed XRD peaks against reference patterns and derives
structure-performance correlations for zinc electrodeposition.
"""

from __future__ import annotations

from typing import Any

from optimization_copilot.agents.base import (
    AgentContext,
    AgentMode,
    OptimizationFeedback,
    ScientificAgent,
    TriggerCondition,
)
from optimization_copilot.agents.phase_structure.reference_db import (
    REFERENCE_PEAKS,
    ReferenceDB,
)


# Keys in raw_data that indicate XRD data is present
_XRD_KEYS = {"xrd", "two_theta", "peaks", "2theta", "xrd_peaks"}

# Minimum match score to consider a phase identified
_MIN_MATCH_SCORE = 0.15


class PhaseStructureAgent(ScientificAgent):
    """Agent that performs XRD phase identification and structural analysis.

    Extracts XRD peak data from ``raw_data``, matches against a reference
    database, computes texture coefficients, and derives structure-performance
    insights specific to zinc electrodeposition.

    Parameters
    ----------
    reference_db : ReferenceDB | None
        Custom reference database. Defaults to the built-in COD subset.
    tolerance : float
        Peak matching tolerance in degrees 2-theta.
    mode : AgentMode
        Operational mode.
    """

    def __init__(
        self,
        reference_db: ReferenceDB | None = None,
        tolerance: float = 0.3,
        mode: AgentMode = AgentMode.PRAGMATIC,
    ) -> None:
        super().__init__(mode=mode)
        self._db = reference_db if reference_db is not None else ReferenceDB()
        self._tolerance = tolerance

        self._trigger_conditions = [
            TriggerCondition(
                name="xrd_data_available",
                check_fn_name="check_xrd_data",
                priority=5,
                description="Activates when raw_data contains XRD peak information",
            ),
            TriggerCondition(
                name="structural_analysis_needed",
                check_fn_name="check_structural_request",
                priority=3,
                description="Activates when structural characterization is requested",
            ),
        ]

    def name(self) -> str:
        return "phase_structure"

    def should_activate(self, context: AgentContext) -> bool:
        """Activate when raw_data contains XRD-like data."""
        if context.raw_data is None:
            return False
        raw_keys = set(context.raw_data.keys())
        return bool(raw_keys & _XRD_KEYS)

    def validate_context(self, context: AgentContext) -> bool:
        """Validate that extractable XRD peak data exists."""
        peaks, _ = self._extract_peaks(context)
        return len(peaks) > 0

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        """Run phase identification and structural analysis.

        Returns
        -------
        dict[str, Any]
            Keys: ``matched_phases``, ``texture_coefficients``,
            ``structural_insights``, ``quality_score``, ``n_peaks``.
        """
        peaks, intensities = self._extract_peaks(context)

        if not peaks:
            return {
                "matched_phases": [],
                "texture_coefficients": {},
                "structural_insights": [],
                "quality_score": 0.0,
                "n_peaks": 0,
            }

        # Phase matching
        matches = self._db.match_peaks(peaks, tolerance=self._tolerance)
        matched_phases: list[dict[str, Any]] = []
        for phase_name, pairs, score in matches:
            if score >= _MIN_MATCH_SCORE:
                ref = self._db.get_reference(phase_name)
                matched_phases.append({
                    "phase": phase_name,
                    "score": round(score, 3),
                    "matched_peaks": len(pairs),
                    "total_ref_peaks": len(ref.get("peaks_2theta", [])) if ref else 0,
                    "crystal_system": ref.get("crystal_system", "") if ref else "",
                    "space_group": ref.get("space_group", "") if ref else "",
                })

        # Texture coefficients (only if intensities available)
        texture_coefficients: dict[str, float] = {}
        if intensities and len(intensities) == len(peaks):
            for phase_info in matched_phases:
                phase_name = phase_info["phase"]
                if phase_name == "Zn":
                    tc = self._db.get_texture_coefficient(
                        peaks, intensities, phase_name,
                        hkl_num="(002)", hkl_den="(100)",
                        tolerance=self._tolerance,
                    )
                    if tc is not None:
                        texture_coefficients[f"Zn_(002)/(100)"] = round(tc, 3)

                    tc101 = self._db.get_texture_coefficient(
                        peaks, intensities, phase_name,
                        hkl_num="(002)", hkl_den="(101)",
                        tolerance=self._tolerance,
                    )
                    if tc101 is not None:
                        texture_coefficients[f"Zn_(002)/(101)"] = round(tc101, 3)

                elif phase_name == "ZnO":
                    tc = self._db.get_texture_coefficient(
                        peaks, intensities, phase_name,
                        hkl_num="(002)", hkl_den="(101)",
                        tolerance=self._tolerance,
                    )
                    if tc is not None:
                        texture_coefficients[f"ZnO_(002)/(101)"] = round(tc, 3)

        # Structural insights
        insights = self._derive_insights(matched_phases, texture_coefficients)

        # Quality score (0-1): based on how well the main phase is identified
        quality_score = 0.0
        if matched_phases:
            best_score = matched_phases[0]["score"]
            quality_score = min(1.0, best_score * 1.5)  # scale up

        return {
            "matched_phases": matched_phases,
            "texture_coefficients": texture_coefficients,
            "structural_insights": insights,
            "quality_score": round(quality_score, 3),
            "n_peaks": len(peaks),
        }

    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        """Convert structural analysis to optimization feedback."""
        insights = analysis_result.get("structural_insights", [])
        if not insights:
            return None

        quality = analysis_result.get("quality_score", 0.0)
        matched = analysis_result.get("matched_phases", [])
        texture = analysis_result.get("texture_coefficients", {})

        # Confidence based on quality score
        confidence = max(0.3, min(0.9, quality * 0.9))

        phase_names = [m["phase"] for m in matched]

        return OptimizationFeedback(
            agent_name=self.name(),
            feedback_type="hypothesis",
            confidence=confidence,
            payload={
                "phases": phase_names,
                "texture_coefficients": texture,
                "insights": insights,
                "quality_score": quality,
            },
            reasoning=(
                f"XRD phase analysis identified {', '.join(phase_names)}. "
                + " ".join(insights)
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_peaks(
        context: AgentContext,
    ) -> tuple[list[float], list[float]]:
        """Extract peak positions and intensities from context.raw_data.

        Tries several common data layouts:
        1. ``raw_data["peaks"]`` as list of dicts with ``two_theta`` and ``intensity``
        2. ``raw_data["xrd"]`` with sub-keys ``two_theta`` and ``intensities``
        3. ``raw_data["two_theta"]`` and ``raw_data["intensities"]`` as parallel lists
        4. ``raw_data["peaks"]`` or ``raw_data["xrd_peaks"]`` as plain list of floats

        Returns
        -------
        tuple[list[float], list[float]]
            ``(peak_positions, intensities)``. Intensities may be empty.
        """
        if context.raw_data is None:
            return [], []

        raw = context.raw_data

        # Layout 1: peaks as list of dicts
        peaks_data = raw.get("peaks")
        if isinstance(peaks_data, list) and peaks_data:
            if isinstance(peaks_data[0], dict):
                positions = []
                intensities = []
                for p in peaks_data:
                    pos = p.get("two_theta", p.get("2theta", p.get("position")))
                    if pos is not None:
                        try:
                            positions.append(float(pos))
                            inten = p.get("intensity", p.get("relative_intensity"))
                            if inten is not None:
                                intensities.append(float(inten))
                        except (TypeError, ValueError):
                            continue
                return positions, intensities if len(intensities) == len(positions) else []

            # Layout 4: peaks as plain list of floats
            try:
                positions = [float(v) for v in peaks_data]
                return positions, []
            except (TypeError, ValueError):
                pass

        # Layout 2: xrd sub-dict
        xrd_data = raw.get("xrd")
        if isinstance(xrd_data, dict):
            tt = xrd_data.get("two_theta", xrd_data.get("2theta", []))
            intensities = xrd_data.get("intensities", xrd_data.get("intensity", []))
            if tt:
                try:
                    positions = [float(v) for v in tt]
                    ints = [float(v) for v in intensities] if intensities else []
                    return positions, ints if len(ints) == len(positions) else []
                except (TypeError, ValueError):
                    pass

        # Layout 3: parallel lists
        tt = raw.get("two_theta", raw.get("2theta", []))
        intensities_raw = raw.get("intensities", raw.get("intensity", []))
        if tt:
            try:
                positions = [float(v) for v in tt]
                ints = [float(v) for v in intensities_raw] if intensities_raw else []
                return positions, ints if len(ints) == len(positions) else []
            except (TypeError, ValueError):
                pass

        # Layout 4: xrd_peaks as list of floats
        xrd_peaks = raw.get("xrd_peaks", [])
        if xrd_peaks:
            try:
                positions = [float(v) for v in xrd_peaks]
                return positions, []
            except (TypeError, ValueError):
                pass

        return [], []

    @staticmethod
    def _derive_insights(
        matched_phases: list[dict[str, Any]],
        texture_coefficients: dict[str, float],
    ) -> list[str]:
        """Derive structure-performance insights from phase/texture data.

        Applies domain rules for zinc electrodeposition:
        - High Zn(002)/Zn(100) ratio -> preferred basal orientation -> better corrosion resistance
        - ZnO presence -> possible passivation layer
        - Zn(OH)2 presence -> poor deposit quality indicator
        - ZnSO4 presence -> electrolyte salt inclusion
        """
        insights: list[str] = []
        phase_names = {m["phase"] for m in matched_phases}

        # Texture analysis for Zn
        tc_002_100 = texture_coefficients.get("Zn_(002)/(100)")
        if tc_002_100 is not None:
            if tc_002_100 > 2.0:
                insights.append(
                    f"Strong (002) preferred orientation (TC={tc_002_100:.2f}) indicates "
                    "basal plane texture. This correlates with improved corrosion "
                    "resistance and smoother deposit morphology."
                )
            elif tc_002_100 > 1.2:
                insights.append(
                    f"Moderate (002) texture (TC={tc_002_100:.2f}) suggests partial "
                    "basal orientation. Some improvement in corrosion performance expected."
                )
            elif tc_002_100 < 0.5:
                insights.append(
                    f"Low (002) texture (TC={tc_002_100:.2f}) indicates random or "
                    "non-basal orientation. Deposit may have higher surface roughness."
                )

        # Phase-specific insights
        if "ZnO" in phase_names:
            insights.append(
                "ZnO phase detected. This may indicate surface passivation or "
                "oxidation of the zinc deposit. Moderate ZnO can improve "
                "corrosion resistance but excessive amounts suggest overexposure."
            )

        if "ZnOH2" in phase_names:
            insights.append(
                "Zn(OH)2 detected, which is a poor-quality indicator. This phase "
                "forms when local pH is too high at the cathode surface. Consider "
                "reducing current density or adjusting electrolyte pH."
            )

        if "ZnSO4" in phase_names:
            insights.append(
                "ZnSO4 inclusion detected. This indicates incomplete rinsing "
                "or electrolyte salt entrapment in the deposit. Improve post-plating "
                "rinse protocol."
            )

        # Multi-phase assessment
        if len(phase_names) > 2:
            insights.append(
                f"Multiple phases detected ({', '.join(sorted(phase_names))}). "
                "Complex phase mixtures may indicate non-optimal deposition conditions."
            )

        return insights

"""Domain configuration loader -- aggregates instrument specs, physical constraints,
quality thresholds, and rules for a given experimental domain."""

from __future__ import annotations

from typing import Any


class DomainConfig:
    """Aggregates domain-specific configuration for a given experimental domain.

    Supported domains:
    - "electrochemistry" -- eis.py + dc_cycling.py + uv_vis.py + xrd.py
    - "catalysis" -- catalysis.py
    - "perovskite" -- perovskite.py

    Provides unified interface: get_instruments(), get_constraints(),
    get_quality_thresholds(), get_rules().
    """

    SUPPORTED_DOMAINS = ("electrochemistry", "catalysis", "perovskite")

    def __init__(self, domain_name: str) -> None:
        if domain_name not in self.SUPPORTED_DOMAINS:
            raise ValueError(
                f"Unknown domain: {domain_name!r}. "
                f"Supported: {self.SUPPORTED_DOMAINS}"
            )
        self.domain_name = domain_name
        self._config = self._load_domain(domain_name)

    def _load_domain(self, domain_name: str) -> dict[str, Any]:
        """Load and merge configs for the given domain."""
        if domain_name == "electrochemistry":
            return self._load_electrochemistry()
        elif domain_name == "catalysis":
            return self._load_catalysis()
        elif domain_name == "perovskite":
            return self._load_perovskite()
        raise ValueError(f"Unknown domain: {domain_name!r}")

    def _load_electrochemistry(self) -> dict[str, Any]:
        from optimization_copilot.domain_knowledge.eis import (
            DEFAULT_EIS_INSTRUMENT,
            EIS_PHYSICAL_CONSTRAINTS,
            EIS_QUALITY_THRESHOLDS,
            DEFAULT_EIS_CIRCUITS,
            DEFAULT_MODEL_SELECTION,
        )
        from optimization_copilot.domain_knowledge.dc_cycling import (
            DEFAULT_DC_INSTRUMENT,
            DC_PHYSICAL_CONSTRAINTS,
            DC_QUALITY_THRESHOLDS,
        )
        from optimization_copilot.domain_knowledge.uv_vis import (
            DEFAULT_UVVIS_INSTRUMENT,
            UVVIS_PHYSICAL_CONSTRAINTS,
            UVVIS_QUALITY_THRESHOLDS,
        )
        from optimization_copilot.domain_knowledge.xrd import (
            DEFAULT_XRD_INSTRUMENT,
            XRD_PHYSICAL_CONSTRAINTS,
            XRD_QUALITY_THRESHOLDS,
        )

        instruments = {
            "eis": DEFAULT_EIS_INSTRUMENT,
            "dc": DEFAULT_DC_INSTRUMENT,
            "uv_vis": DEFAULT_UVVIS_INSTRUMENT,
            "xrd": DEFAULT_XRD_INSTRUMENT,
        }

        physical_constraints: dict[str, Any] = {}
        physical_constraints.update(EIS_PHYSICAL_CONSTRAINTS)
        physical_constraints.update(DC_PHYSICAL_CONSTRAINTS)
        physical_constraints.update(UVVIS_PHYSICAL_CONSTRAINTS)
        physical_constraints.update(XRD_PHYSICAL_CONSTRAINTS)

        # Merge quality thresholds (use first seen for each key)
        quality_thresholds: dict[str, float] = {}
        for qt in [
            EIS_QUALITY_THRESHOLDS,
            DC_QUALITY_THRESHOLDS,
            UVVIS_QUALITY_THRESHOLDS,
            XRD_QUALITY_THRESHOLDS,
        ]:
            for k, v in qt.items():
                if k not in quality_thresholds:
                    quality_thresholds[k] = v

        return {
            "instruments": instruments,
            "physical_constraints": physical_constraints,
            "quality_thresholds": quality_thresholds,
            "circuits": DEFAULT_EIS_CIRCUITS,
            "model_selection": DEFAULT_MODEL_SELECTION,
            "rules": {},  # electrochemistry rules embedded in extractors
        }

    def _load_catalysis(self) -> dict[str, Any]:
        from optimization_copilot.domain_knowledge.catalysis import (
            DEFAULT_CATALYSIS_INSTRUMENTS,
            CATALYSIS_PHYSICAL_CONSTRAINTS,
            CATALYSIS_QUALITY_THRESHOLDS,
            KNOWN_INCOMPATIBILITIES,
            get_catalysis_rules,
        )
        return {
            "instruments": DEFAULT_CATALYSIS_INSTRUMENTS,
            "physical_constraints": CATALYSIS_PHYSICAL_CONSTRAINTS,
            "quality_thresholds": CATALYSIS_QUALITY_THRESHOLDS,
            "known_incompatibilities": KNOWN_INCOMPATIBILITIES,
            "rules": get_catalysis_rules(),
        }

    def _load_perovskite(self) -> dict[str, Any]:
        from optimization_copilot.domain_knowledge.perovskite import (
            DEFAULT_PEROVSKITE_INSTRUMENTS,
            PEROVSKITE_PHYSICAL_CONSTRAINTS,
            PEROVSKITE_QUALITY_THRESHOLDS,
            PHASE_STABILITY_RULES,
            get_perovskite_rules,
        )
        return {
            "instruments": DEFAULT_PEROVSKITE_INSTRUMENTS,
            "physical_constraints": PEROVSKITE_PHYSICAL_CONSTRAINTS,
            "quality_thresholds": PEROVSKITE_QUALITY_THRESHOLDS,
            "phase_stability_rules": PHASE_STABILITY_RULES,
            "rules": get_perovskite_rules(),
        }

    # -- Public accessors ---------------------------------------------------

    def get_instruments(self) -> dict[str, Any]:
        """Return instrument configurations for this domain."""
        return self._config.get("instruments", {})

    def get_constraints(self) -> dict[str, Any]:
        """Return physical constraints for this domain."""
        return self._config.get("physical_constraints", {})

    def get_quality_thresholds(self) -> dict[str, float]:
        """Return quality threshold values for this domain."""
        return self._config.get("quality_thresholds", {})

    def get_rules(self) -> dict[str, Any]:
        """Return validation rules for this domain."""
        return self._config.get("rules", {})

    def get_known_incompatibilities(self) -> list[dict[str, str]]:
        """Return known material incompatibilities (catalysis domain)."""
        return self._config.get("known_incompatibilities", [])

    @property
    def config(self) -> dict[str, Any]:
        """Return the full raw configuration dict."""
        return self._config

    def __repr__(self) -> str:
        n_inst = len(self.get_instruments())
        n_constr = len(self.get_constraints())
        return (
            f"DomainConfig(domain={self.domain_name!r}, "
            f"instruments={n_inst}, constraints={n_constr})"
        )

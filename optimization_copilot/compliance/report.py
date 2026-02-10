"""Compliance report generation for regulatory audits.

Produces a structured ``ComplianceReport`` from an audit log, capturing
the campaign summary, parameter specifications, decision log, final
recommendation, rule versions, and chain verification status.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.compliance.audit import ChainVerification


# ---------------------------------------------------------------------------
# ComplianceReport
# ---------------------------------------------------------------------------


@dataclass
class ComplianceReport:
    """Structured compliance report for a completed optimization campaign.

    Attributes
    ----------
    campaign_id : str
        Unique identifier of the campaign.
    campaign_summary : dict[str, Any]
        High-level summary (iteration count, timestamps, seed, etc.).
    parameter_specs : list[dict[str, Any]]
        Parameter specifications from the campaign spec.
    iteration_log : list[dict[str, Any]]
        Condensed per-iteration records (phase, backend, hashes, etc.).
    final_recommendation : dict[str, Any]
        The decision and context from the last iteration.
    rule_versions : dict[str, str]
        ``{rule_id: version}`` for every registered decision rule.
    chain_verification : ChainVerification
        Result of the hash chain integrity check.
    generation_timestamp : float
        Unix timestamp when the report was generated.
    report_version : str
        Semantic version of the report format.
    """

    campaign_id: str
    campaign_summary: dict[str, Any]
    parameter_specs: list[dict[str, Any]]
    iteration_log: list[dict[str, Any]]
    final_recommendation: dict[str, Any]
    rule_versions: dict[str, str]
    chain_verification: ChainVerification
    generation_timestamp: float = 0.0
    report_version: str = "1.0.0"

    def format_text(self) -> str:
        """Render the report as a human-readable plain-text document."""
        lines: list[str] = []

        lines.append("=" * 40)
        lines.append(f"COMPLIANCE REPORT: {self.campaign_id}")
        lines.append("=" * 40)
        lines.append("")

        # -- Summary --------------------------------------------------------
        lines.append("SUMMARY")
        lines.append("-" * 7)
        for key, value in self.campaign_summary.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # -- Parameter Specifications ---------------------------------------
        lines.append("PARAMETER SPECIFICATIONS")
        lines.append("-" * 24)
        for spec in self.parameter_specs:
            name = spec.get("name", "unknown")
            ptype = spec.get("type", "unknown")
            lower = spec.get("lower", "N/A")
            upper = spec.get("upper", "N/A")
            lines.append(f"  {name}: type={ptype}, lower={lower}, upper={upper}")
        lines.append("")

        # -- Decision Log ---------------------------------------------------
        lines.append("DECISION LOG")
        lines.append("-" * 12)
        for entry in self.iteration_log:
            iteration = entry.get("iteration", "?")
            phase = entry.get("phase", "?")
            backend = entry.get("backend_name", "?")
            reason_codes = entry.get("reason_codes", [])
            d_hash = entry.get("decision_hash", "?")
            c_hash = entry.get("chain_hash", "?")
            lines.append(
                f"  Iteration {iteration}: Phase={phase}, Backend={backend}"
            )
            lines.append(f"    Reason codes: {reason_codes}")
            lines.append(f"    Decision hash: {d_hash}")
            lines.append(f"    Chain hash: {c_hash}")
        lines.append("")

        # -- Final Recommendation -------------------------------------------
        lines.append("FINAL RECOMMENDATION")
        lines.append("-" * 20)
        for key, value in self.final_recommendation.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # -- Rule Versions --------------------------------------------------
        lines.append("RULE VERSIONS")
        lines.append("-" * 13)
        for rule_id, version in self.rule_versions.items():
            lines.append(f"  {rule_id}: {version}")
        lines.append("")

        # -- Chain Verification ---------------------------------------------
        lines.append("CHAIN VERIFICATION")
        lines.append("-" * 18)
        lines.append(f"  {self.chain_verification.summary()}")
        lines.append("")

        # -- Footer ---------------------------------------------------------
        lines.append("=" * 40)
        lines.append(f"Report version: {self.report_version}")
        lines.append(f"Generated: {self.generation_timestamp}")
        lines.append("=" * 40)

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "campaign_id": self.campaign_id,
            "campaign_summary": dict(self.campaign_summary),
            "parameter_specs": list(self.parameter_specs),
            "iteration_log": list(self.iteration_log),
            "final_recommendation": dict(self.final_recommendation),
            "rule_versions": dict(self.rule_versions),
            "chain_verification": self.chain_verification.to_dict(),
            "generation_timestamp": self.generation_timestamp,
            "report_version": self.report_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComplianceReport:
        """Deserialize from a plain dict."""
        return cls(
            campaign_id=data["campaign_id"],
            campaign_summary=data["campaign_summary"],
            parameter_specs=data["parameter_specs"],
            iteration_log=data["iteration_log"],
            final_recommendation=data["final_recommendation"],
            rule_versions=data["rule_versions"],
            chain_verification=ChainVerification.from_dict(
                data["chain_verification"]
            ),
            generation_timestamp=data.get("generation_timestamp", 0.0),
            report_version=data.get("report_version", "1.0.0"),
        )

"""Compliance engine orchestrating audit logging, chain verification, and report generation.

Ties together ``AuditLog``, ``AuditEntry``, ``verify_chain``, and
``ComplianceReport`` into a single high-level API that the optimization
pipeline uses to record, verify, and report on decision provenance.
"""

from __future__ import annotations

import time
from typing import Any

from optimization_copilot.compliance.audit import (
    AuditEntry,
    AuditLog,
    ChainVerification,
    verify_chain,
    _compute_content_hash,
    _compute_chain_hash,
)
from optimization_copilot.compliance.report import ComplianceReport
from optimization_copilot.replay.log import DecisionLogEntry
from optimization_copilot.replay.engine import ReplayEngine
from optimization_copilot.schema.rules import DecisionRuleEngine


class ComplianceEngine:
    """High-level compliance engine for optimization campaigns.

    Manages active audit logs, records hash-chained decisions, verifies
    chain integrity, and generates compliance reports.

    Parameters
    ----------
    replay_engine : ReplayEngine or None
        Optional replay engine for step-level replay.  When ``None``,
        ``replay_step`` will raise ``ValueError``.
    rule_engine : DecisionRuleEngine or None
        Rule engine whose registered rules are included in compliance
        reports.  A default engine is created if not provided.
    signer_id : str
        Identifier attached to every audit entry and log produced by
        this engine.
    """

    def __init__(
        self,
        replay_engine: ReplayEngine | None = None,
        rule_engine: DecisionRuleEngine | None = None,
        signer_id: str = "optimization_copilot",
    ) -> None:
        self._replay_engine = replay_engine
        self._rule_engine = rule_engine or DecisionRuleEngine()
        self._signer_id = signer_id
        self._active_logs: dict[str, AuditLog] = {}

    # -- lifecycle ----------------------------------------------------------

    def start_audit(
        self,
        campaign_id: str,
        spec: dict[str, Any],
        base_seed: int = 42,
        metadata: dict[str, Any] | None = None,
    ) -> AuditLog:
        """Begin a new audit log for *campaign_id*.

        Raises ``ValueError`` if an audit is already active for the
        given campaign.

        Parameters
        ----------
        campaign_id : str
            Unique campaign identifier.
        spec : dict[str, Any]
            Campaign specification (parameters, objectives, etc.).
        base_seed : int
            Base random seed for the campaign.
        metadata : dict or None
            Optional metadata to attach to the audit log.

        Returns
        -------
        AuditLog
        """
        if campaign_id in self._active_logs:
            raise ValueError(
                f"Audit already active for campaign '{campaign_id}'"
            )
        log = AuditLog(
            campaign_id=campaign_id,
            spec=spec,
            base_seed=base_seed,
            metadata=metadata or {},
            signer_id=self._signer_id,
        )
        self._active_logs[campaign_id] = log
        return log

    def finalize_audit(self, campaign_id: str) -> AuditLog:
        """Finalize and detach the audit log for *campaign_id*.

        Removes the log from the active set and returns it.  Raises
        ``KeyError`` if no active audit exists for the campaign.

        Parameters
        ----------
        campaign_id : str

        Returns
        -------
        AuditLog
        """
        return self._active_logs.pop(campaign_id)

    # -- recording ----------------------------------------------------------

    def record_decision(
        self,
        campaign_id: str,
        log_entry: DecisionLogEntry,
    ) -> AuditEntry:
        """Record a decision as a hash-chained audit entry.

        Computes the content hash from the log entry, chains it to the
        previous entry (if any), and appends the resulting ``AuditEntry``
        to the active audit log.

        Parameters
        ----------
        campaign_id : str
            Must match an active audit started via ``start_audit``.
        log_entry : DecisionLogEntry
            The decision log entry to record.

        Returns
        -------
        AuditEntry

        Raises
        ------
        KeyError
            If no active audit exists for *campaign_id*.
        """
        log = self._active_logs[campaign_id]

        # Previous chain hash (empty string for the first entry)
        prev_hash = log.entries[-1].chain_hash if log.entries else ""

        # Build the AuditEntry with an empty chain_hash first so that
        # content_hash() includes signer_id and entry_version -- matching
        # what verify_chain will recompute later.
        entry = AuditEntry.from_log_entry(
            log_entry,
            chain_hash="",
            signer_id=self._signer_id,
        )
        content_hash = entry.content_hash()
        entry.chain_hash = _compute_chain_hash(prev_hash, content_hash)

        log.append(entry)
        return entry

    # -- queries ------------------------------------------------------------

    def get_audit_log(self, campaign_id: str) -> AuditLog:
        """Return the active audit log for *campaign_id*.

        Raises ``KeyError`` if no active audit exists.
        """
        return self._active_logs[campaign_id]

    # -- verification -------------------------------------------------------

    def verify_chain(self, audit_log: AuditLog) -> ChainVerification:
        """Verify the hash chain integrity of an audit log.

        Delegates to the module-level ``verify_chain`` function.

        Parameters
        ----------
        audit_log : AuditLog

        Returns
        -------
        ChainVerification
        """
        return verify_chain(audit_log)

    # -- reporting ----------------------------------------------------------

    def generate_report(self, audit_log: AuditLog) -> ComplianceReport:
        """Generate a compliance report from an audit log.

        Assembles a ``ComplianceReport`` containing the campaign summary,
        parameter specifications, iteration log, final recommendation,
        rule versions, and chain verification result.

        Parameters
        ----------
        audit_log : AuditLog

        Returns
        -------
        ComplianceReport
        """
        # Campaign summary
        campaign_summary: dict[str, Any] = {
            "campaign_id": audit_log.campaign_id,
            "n_iterations": audit_log.n_entries,
            "base_seed": audit_log.base_seed,
        }
        if audit_log.entries:
            campaign_summary["first_timestamp"] = audit_log.entries[0].timestamp
            campaign_summary["last_timestamp"] = audit_log.entries[-1].timestamp

        # Parameter specifications
        parameter_specs: list[dict[str, Any]] = audit_log.spec.get(
            "parameter_specs", []
        )

        # Iteration log (condensed)
        iteration_log: list[dict[str, Any]] = []
        for entry in audit_log.entries:
            iteration_log.append({
                "iteration": entry.iteration,
                "phase": entry.phase,
                "backend_name": entry.backend_name,
                "reason_codes": list(entry.reason_codes),
                "decision_hash": entry.decision_hash,
                "chain_hash": entry.chain_hash,
                "snapshot_hash": entry.snapshot_hash,
            })

        # Final recommendation
        final_recommendation: dict[str, Any] = {}
        if audit_log.entries:
            last = audit_log.entries[-1]
            final_recommendation = {
                "final_decision": dict(last.decision),
                "final_phase": last.phase,
                "final_backend": last.backend_name,
            }

        # Rule versions
        rule_versions: dict[str, str] = {}
        for rule in self._rule_engine._rules.values():
            rule_versions[rule.rule_id] = rule.version

        # Chain verification
        chain_ver = verify_chain(audit_log)

        return ComplianceReport(
            campaign_id=audit_log.campaign_id,
            campaign_summary=campaign_summary,
            parameter_specs=parameter_specs,
            iteration_log=iteration_log,
            final_recommendation=final_recommendation,
            rule_versions=rule_versions,
            chain_verification=chain_ver,
            generation_timestamp=time.time(),
        )

    # -- replay -------------------------------------------------------------

    def replay_step(
        self,
        audit_log: AuditLog,
        iteration: int,
    ) -> dict[str, Any]:
        """Retrieve replay data for a single iteration.

        Returns the original entry data alongside a replay-availability
        indicator.  Requires a ``ReplayEngine`` to be configured.

        Parameters
        ----------
        audit_log : AuditLog
        iteration : int

        Returns
        -------
        dict[str, Any]

        Raises
        ------
        ValueError
            If no ``ReplayEngine`` was provided at construction.
        KeyError
            If no entry exists for the given *iteration*.
        """
        if self._replay_engine is None:
            raise ValueError("ReplayEngine not configured")

        entry = audit_log.get_entry(iteration)
        if entry is None:
            raise KeyError(f"No entry for iteration {iteration}")

        return {
            "original": entry.to_dict(),
            "iteration": iteration,
            "decision_hash": entry.decision_hash,
            "chain_hash": entry.chain_hash,
            "replay_available": True,
        }

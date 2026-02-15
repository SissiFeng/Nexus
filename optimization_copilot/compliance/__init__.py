"""Regulatory/Compliance package for optimization campaign audit trails.

Re-exports the core public API:

- ``AuditEntry`` -- hash-chained audit entry extending DecisionLogEntry
- ``AuditLog`` -- ordered collection of audit entries with chain integrity
- ``verify_chain`` -- validate hash chain integrity of an audit log
- ``ChainVerification`` -- result dataclass from chain verification
- ``ComplianceReport`` -- structured compliance report for regulatory audits
- ``ComplianceEngine`` -- high-level orchestrator for audit, verification, and reporting
"""

from optimization_copilot.compliance.audit import (
    AuditEntry,
    AuditLog,
    ChainVerification,
    verify_chain,
)
from optimization_copilot.compliance.report import ComplianceReport
from optimization_copilot.compliance.engine import ComplianceEngine

__all__ = [
    "AuditEntry",
    "AuditLog",
    "verify_chain",
    "ChainVerification",
    "ComplianceReport",
    "ComplianceEngine",
]

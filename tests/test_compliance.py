"""Tests for the Regulatory/Compliance Mode (Capability 15).

Covers:
- Module-level helpers: _compute_content_hash, _compute_chain_hash
- ChainVerification dataclass: construction, summary(), to_dict/from_dict
- AuditEntry dataclass: construction, from_log_entry(), content_hash(), to_dict/from_dict
- AuditLog dataclass: construction, append/get_entry, n_entries, chain_intact,
  to_dict/from_dict, to_json/from_json, save/load
- verify_chain() function: empty log, single entry, multi-entry, tampered chain,
  tampered content, broken_links reporting
- ComplianceReport dataclass: construction, format_text(), to_dict/from_dict
- ComplianceEngine class: constructor, start_audit, record_decision, get_audit_log,
  finalize_audit, verify_chain, generate_report, replay_step
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from optimization_copilot.compliance.audit import (
    AuditEntry,
    AuditLog,
    ChainVerification,
    verify_chain,
    _compute_content_hash,
    _compute_chain_hash,
)
from optimization_copilot.compliance.engine import ComplianceEngine
from optimization_copilot.compliance.report import ComplianceReport
from optimization_copilot.core.hashing import _sha256, _stable_serialize
from optimization_copilot.replay.log import DecisionLogEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_log_entry(
    iteration: int = 0,
    timestamp: float = 1000.0,
    phase: str = "learning",
    backend_name: str = "random_sampler",
    seed: int = 42,
) -> DecisionLogEntry:
    """Build a DecisionLogEntry with deterministic test data."""
    return DecisionLogEntry(
        iteration=iteration,
        timestamp=timestamp,
        snapshot_hash="snap_abc123",
        diagnostics_hash="diag_def456",
        diagnostics={"convergence_trend": 0.5, "model_uncertainty": 0.3},
        fingerprint={"dimensionality": "low", "noise_level": "medium"},
        decision={"backend_name": backend_name, "batch_size": 4, "phase": phase},
        decision_hash="dec_ghi789",
        suggested_candidates=[{"x": 1.0, "y": 2.0}],
        ingested_results=[{"parameters": {"x": 1.0}, "kpi_values": {"obj": 0.5}}],
        phase=phase,
        backend_name=backend_name,
        reason_codes=["cold_start", "exploration"],
        seed=seed,
    )


def _build_chained_audit_log(
    n_entries: int = 3,
    campaign_id: str = "test-campaign",
) -> AuditLog:
    """Build an AuditLog with n correctly chained entries."""
    log = AuditLog(
        campaign_id=campaign_id,
        spec={"parameter_specs": [{"name": "x", "type": "continuous", "lower": 0, "upper": 1}]},
        base_seed=42,
    )
    prev_chain_hash = ""
    for i in range(n_entries):
        log_entry = _make_log_entry(iteration=i, timestamp=1000.0 + i)
        entry = AuditEntry.from_log_entry(log_entry, chain_hash="")
        content_hash = entry.content_hash()
        entry.chain_hash = _compute_chain_hash(prev_chain_hash, content_hash)
        log.append(entry)
        prev_chain_hash = entry.chain_hash
    return log


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestComputeContentHash:
    """Tests for _compute_content_hash."""

    def test_deterministic(self):
        """Same dict produces same hash every time."""
        d = {"iteration": 0, "decision": {"x": 1}, "chain_hash": "ignored"}
        assert _compute_content_hash(d) == _compute_content_hash(d)

    def test_excludes_chain_hash(self):
        """chain_hash field is excluded from the hash computation."""
        d1 = {"iteration": 0, "value": 1, "chain_hash": "aaa"}
        d2 = {"iteration": 0, "value": 1, "chain_hash": "zzz"}
        assert _compute_content_hash(d1) == _compute_content_hash(d2)

    def test_different_content_different_hash(self):
        """Different content produces different hashes."""
        d1 = {"iteration": 0, "value": 1, "chain_hash": ""}
        d2 = {"iteration": 0, "value": 2, "chain_hash": ""}
        assert _compute_content_hash(d1) != _compute_content_hash(d2)


class TestComputeChainHash:
    """Tests for _compute_chain_hash."""

    def test_first_entry_no_previous(self):
        """When previous is empty, result is SHA-256 of content alone."""
        content = "abc123"
        result = _compute_chain_hash("", content)
        assert result == _sha256(content)

    def test_chained_entry(self):
        """When previous is non-empty, result is SHA-256 of prev+content."""
        prev = "prevhash"
        content = "contenthash"
        result = _compute_chain_hash(prev, content)
        assert result == _sha256(prev + content)

    def test_deterministic(self):
        """Same inputs always produce the same chain hash."""
        result1 = _compute_chain_hash("p", "c")
        result2 = _compute_chain_hash("p", "c")
        assert result1 == result2


# ---------------------------------------------------------------------------
# ChainVerification
# ---------------------------------------------------------------------------


class TestChainVerification:
    """Tests for ChainVerification dataclass."""

    def test_construction_valid(self):
        """Construct a valid ChainVerification."""
        cv = ChainVerification(
            valid=True,
            n_entries=5,
            n_broken_links=0,
            first_broken_link=None,
            broken_links=[],
        )
        assert cv.valid is True
        assert cv.n_entries == 5
        assert cv.n_broken_links == 0
        assert cv.first_broken_link is None
        assert cv.broken_links == []

    def test_construction_broken(self):
        """Construct a ChainVerification with broken links."""
        cv = ChainVerification(
            valid=False,
            n_entries=10,
            n_broken_links=2,
            first_broken_link=3,
            broken_links=[3, 7],
            details=[{"index": 2, "iteration": 3}],
        )
        assert cv.valid is False
        assert cv.n_broken_links == 2
        assert cv.first_broken_link == 3
        assert cv.broken_links == [3, 7]

    def test_summary_valid(self):
        """summary() returns PASSED message for valid chain."""
        cv = ChainVerification(
            valid=True, n_entries=5, n_broken_links=0,
            first_broken_link=None, broken_links=[],
        )
        s = cv.summary()
        assert "PASSED" in s
        assert "5 entries" in s

    def test_summary_invalid(self):
        """summary() returns FAILED message with broken link info."""
        cv = ChainVerification(
            valid=False, n_entries=10, n_broken_links=2,
            first_broken_link=3, broken_links=[3, 7],
        )
        s = cv.summary()
        assert "FAILED" in s
        assert "2/10" in s
        assert "iteration 3" in s

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict/from_dict produces identical object."""
        cv = ChainVerification(
            valid=False, n_entries=3, n_broken_links=1,
            first_broken_link=2, broken_links=[2],
            details=[{"index": 1, "iteration": 2, "expected": "aaa", "actual": "bbb"}],
        )
        restored = ChainVerification.from_dict(cv.to_dict())
        assert restored.valid == cv.valid
        assert restored.n_entries == cv.n_entries
        assert restored.n_broken_links == cv.n_broken_links
        assert restored.first_broken_link == cv.first_broken_link
        assert restored.broken_links == cv.broken_links
        assert restored.details == cv.details


# ---------------------------------------------------------------------------
# AuditEntry
# ---------------------------------------------------------------------------


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_construction_all_fields(self):
        """AuditEntry can be constructed with all required fields."""
        entry = AuditEntry(
            iteration=0, timestamp=1000.0,
            snapshot_hash="snap", diagnostics_hash="diag",
            diagnostics={"a": 1.0}, fingerprint={"b": "c"},
            decision={"x": 1}, decision_hash="dec",
            suggested_candidates=[], ingested_results=[],
            phase="learning", backend_name="random",
            reason_codes=["r1"], seed=42,
            chain_hash="ch", signer_id="test", entry_version="1.0.0",
        )
        assert entry.iteration == 0
        assert entry.chain_hash == "ch"
        assert entry.signer_id == "test"

    def test_from_log_entry(self):
        """from_log_entry copies all fields from DecisionLogEntry."""
        log_entry = _make_log_entry(iteration=5, timestamp=2000.0)
        audit_entry = AuditEntry.from_log_entry(
            log_entry, chain_hash="test_chain", signer_id="signer1",
        )
        assert audit_entry.iteration == 5
        assert audit_entry.timestamp == 2000.0
        assert audit_entry.snapshot_hash == log_entry.snapshot_hash
        assert audit_entry.diagnostics_hash == log_entry.diagnostics_hash
        assert audit_entry.diagnostics == log_entry.diagnostics
        assert audit_entry.fingerprint == log_entry.fingerprint
        assert audit_entry.decision == log_entry.decision
        assert audit_entry.decision_hash == log_entry.decision_hash
        assert audit_entry.suggested_candidates == log_entry.suggested_candidates
        assert audit_entry.ingested_results == log_entry.ingested_results
        assert audit_entry.phase == log_entry.phase
        assert audit_entry.backend_name == log_entry.backend_name
        assert audit_entry.reason_codes == log_entry.reason_codes
        assert audit_entry.seed == log_entry.seed
        assert audit_entry.chain_hash == "test_chain"
        assert audit_entry.signer_id == "signer1"

    def test_content_hash_deterministic(self):
        """content_hash() is deterministic for the same entry."""
        log_entry = _make_log_entry()
        entry = AuditEntry.from_log_entry(log_entry, chain_hash="")
        h1 = entry.content_hash()
        h2 = entry.content_hash()
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) > 0

    def test_content_hash_excludes_chain_hash(self):
        """content_hash() is the same regardless of chain_hash value."""
        log_entry = _make_log_entry()
        entry1 = AuditEntry.from_log_entry(log_entry, chain_hash="")
        entry2 = AuditEntry.from_log_entry(log_entry, chain_hash="different_chain")
        assert entry1.content_hash() == entry2.content_hash()

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict/from_dict produces identical AuditEntry."""
        log_entry = _make_log_entry(iteration=3)
        original = AuditEntry.from_log_entry(
            log_entry, chain_hash="test_ch", signer_id="s1",
        )
        d = original.to_dict()
        restored = AuditEntry.from_dict(d)
        assert restored.iteration == original.iteration
        assert restored.timestamp == original.timestamp
        assert restored.chain_hash == original.chain_hash
        assert restored.signer_id == original.signer_id
        assert restored.diagnostics == original.diagnostics
        assert restored.decision == original.decision
        assert restored.content_hash() == original.content_hash()


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------


class TestAuditLog:
    """Tests for AuditLog dataclass."""

    def test_construction(self):
        """AuditLog can be constructed with required fields."""
        log = AuditLog(
            campaign_id="camp-1",
            spec={"parameters": []},
            base_seed=99,
        )
        assert log.campaign_id == "camp-1"
        assert log.base_seed == 99
        assert log.entries == []
        assert log.n_entries == 0

    def test_append_and_get_entry(self):
        """append() adds entries and get_entry() retrieves by iteration."""
        log = AuditLog(campaign_id="c", spec={}, base_seed=0)
        entry0 = AuditEntry.from_log_entry(_make_log_entry(iteration=0), chain_hash="h0")
        entry1 = AuditEntry.from_log_entry(_make_log_entry(iteration=1), chain_hash="h1")
        log.append(entry0)
        log.append(entry1)
        assert log.n_entries == 2
        assert log.get_entry(0) is entry0
        assert log.get_entry(1) is entry1
        assert log.get_entry(99) is None

    def test_n_entries_property(self):
        """n_entries returns the correct count."""
        log = _build_chained_audit_log(n_entries=5)
        assert log.n_entries == 5

    def test_chain_intact_true(self):
        """chain_intact returns True for correctly chained log."""
        log = _build_chained_audit_log(n_entries=3)
        assert log.chain_intact is True

    def test_chain_intact_false_on_tamper(self):
        """chain_intact returns False when chain_hash is tampered."""
        log = _build_chained_audit_log(n_entries=3)
        log.entries[1].chain_hash = "tampered_hash"
        assert log.chain_intact is False

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict/from_dict produces identical AuditLog."""
        log = _build_chained_audit_log(n_entries=2)
        log.metadata = {"key": "value"}
        log.signer_id = "test_signer"
        d = log.to_dict()
        restored = AuditLog.from_dict(d)
        assert restored.campaign_id == log.campaign_id
        assert restored.base_seed == log.base_seed
        assert restored.n_entries == log.n_entries
        assert restored.metadata == log.metadata
        assert restored.signer_id == log.signer_id
        assert restored.entries[0].chain_hash == log.entries[0].chain_hash

    def test_to_json_from_json_roundtrip(self):
        """to_json/from_json produces identical AuditLog."""
        log = _build_chained_audit_log(n_entries=2)
        json_str = log.to_json()
        restored = AuditLog.from_json(json_str)
        assert restored.campaign_id == log.campaign_id
        assert restored.n_entries == log.n_entries
        # Verify chain integrity is preserved through JSON
        assert restored.chain_intact is True

    def test_save_load_roundtrip(self, tmp_path):
        """save/load produces identical AuditLog from file."""
        log = _build_chained_audit_log(n_entries=3)
        filepath = tmp_path / "audit_log.json"
        log.save(filepath)
        assert filepath.exists()
        restored = AuditLog.load(filepath)
        assert restored.campaign_id == log.campaign_id
        assert restored.n_entries == log.n_entries
        assert restored.chain_intact is True
        for i in range(log.n_entries):
            assert restored.entries[i].chain_hash == log.entries[i].chain_hash


# ---------------------------------------------------------------------------
# verify_chain()
# ---------------------------------------------------------------------------


class TestVerifyChain:
    """Tests for the verify_chain() function."""

    def test_empty_log(self):
        """Empty log returns valid=True with n_entries=0."""
        log = AuditLog(campaign_id="empty", spec={}, base_seed=0)
        result = verify_chain(log)
        assert result.valid is True
        assert result.n_entries == 0
        assert result.n_broken_links == 0
        assert result.first_broken_link is None
        assert result.broken_links == []

    def test_single_entry_valid(self):
        """Single correctly chained entry is valid."""
        log = _build_chained_audit_log(n_entries=1)
        result = verify_chain(log)
        assert result.valid is True
        assert result.n_entries == 1
        assert result.n_broken_links == 0

    def test_multiple_entries_valid(self):
        """Multiple correctly chained entries all validate."""
        log = _build_chained_audit_log(n_entries=5)
        result = verify_chain(log)
        assert result.valid is True
        assert result.n_entries == 5
        assert result.n_broken_links == 0
        assert result.broken_links == []

    def test_tampered_chain_hash(self):
        """Tampered chain_hash is detected as broken link."""
        log = _build_chained_audit_log(n_entries=3)
        # Tamper the second entry's chain hash
        log.entries[1].chain_hash = "tampered"
        result = verify_chain(log)
        assert result.valid is False
        assert result.n_broken_links >= 1
        assert 1 in result.broken_links  # iteration 1 is broken
        assert result.first_broken_link == 1

    def test_tampered_content(self):
        """Modified decision field causes content hash mismatch."""
        log = _build_chained_audit_log(n_entries=3)
        # Tamper content of entry at iteration 1
        log.entries[1].decision = {"tampered": True}
        result = verify_chain(log)
        assert result.valid is False
        assert 1 in result.broken_links

    def test_tampered_chain_hash_cascades(self):
        """Tampering entry i breaks entry i and may break i+1 (cascade)."""
        log = _build_chained_audit_log(n_entries=4)
        # Tamper entry 1's chain hash -- this breaks entry 1 itself
        # and also entry 2 (because entry 2's expected hash depends on entry 1's chain_hash)
        log.entries[1].chain_hash = "bad"
        result = verify_chain(log)
        assert result.valid is False
        # At minimum entry 1 is broken; entry 2 also depends on entry 1's chain_hash
        assert 1 in result.broken_links
        assert 2 in result.broken_links

    def test_reports_correct_first_broken_link(self):
        """first_broken_link points to the earliest broken iteration."""
        log = _build_chained_audit_log(n_entries=5)
        # Tamper entry 3
        log.entries[3].chain_hash = "bad"
        result = verify_chain(log)
        assert result.first_broken_link == 3

    def test_details_populated_for_broken_links(self):
        """details list contains info for each broken link."""
        log = _build_chained_audit_log(n_entries=3)
        log.entries[1].chain_hash = "bad"
        result = verify_chain(log)
        assert len(result.details) >= 1
        detail = result.details[0]
        assert "iteration" in detail
        assert "expected_chain_hash" in detail
        assert "actual_chain_hash" in detail
        assert detail["actual_chain_hash"] == "bad"


# ---------------------------------------------------------------------------
# ComplianceReport
# ---------------------------------------------------------------------------


class TestComplianceReport:
    """Tests for ComplianceReport dataclass."""

    def _make_report(self) -> ComplianceReport:
        cv = ChainVerification(
            valid=True, n_entries=2, n_broken_links=0,
            first_broken_link=None, broken_links=[],
        )
        return ComplianceReport(
            campaign_id="test-camp",
            campaign_summary={"n_iterations": 2, "base_seed": 42},
            parameter_specs=[{"name": "x", "type": "continuous", "lower": 0, "upper": 1}],
            iteration_log=[
                {"iteration": 0, "phase": "learning", "backend_name": "random",
                 "reason_codes": ["cold_start"], "decision_hash": "dh0", "chain_hash": "ch0"},
                {"iteration": 1, "phase": "exploitation", "backend_name": "tpe",
                 "reason_codes": ["convergence"], "decision_hash": "dh1", "chain_hash": "ch1"},
            ],
            final_recommendation={"final_decision": {"x": 0.5}, "final_phase": "exploitation"},
            rule_versions={"rule_a": "1.0.0", "rule_b": "2.0.0"},
            chain_verification=cv,
            generation_timestamp=1700000000.0,
            report_version="1.0.0",
        )

    def test_construction(self):
        """ComplianceReport can be constructed with all fields."""
        report = self._make_report()
        assert report.campaign_id == "test-camp"
        assert report.report_version == "1.0.0"
        assert report.chain_verification.valid is True

    def test_format_text_contains_sections(self):
        """format_text() returns string with expected section headers."""
        report = self._make_report()
        text = report.format_text()
        assert "COMPLIANCE REPORT" in text
        assert "SUMMARY" in text
        assert "PARAMETER SPECIFICATIONS" in text
        assert "DECISION LOG" in text
        assert "FINAL RECOMMENDATION" in text
        assert "RULE VERSIONS" in text
        assert "CHAIN VERIFICATION" in text
        assert "PASSED" in text  # chain is valid

    def test_format_text_with_failed_chain(self):
        """format_text() shows FAILED when chain is broken."""
        report = self._make_report()
        report.chain_verification = ChainVerification(
            valid=False, n_entries=2, n_broken_links=1,
            first_broken_link=1, broken_links=[1],
        )
        text = report.format_text()
        assert "FAILED" in text

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict/from_dict produces identical ComplianceReport."""
        report = self._make_report()
        d = report.to_dict()
        restored = ComplianceReport.from_dict(d)
        assert restored.campaign_id == report.campaign_id
        assert restored.campaign_summary == report.campaign_summary
        assert restored.parameter_specs == report.parameter_specs
        assert restored.iteration_log == report.iteration_log
        assert restored.final_recommendation == report.final_recommendation
        assert restored.rule_versions == report.rule_versions
        assert restored.chain_verification.valid == report.chain_verification.valid
        assert restored.generation_timestamp == report.generation_timestamp
        assert restored.report_version == report.report_version


# ---------------------------------------------------------------------------
# ComplianceEngine
# ---------------------------------------------------------------------------


class TestComplianceEngine:
    """Tests for ComplianceEngine class."""

    def test_constructor_defaults(self):
        """ComplianceEngine can be constructed with defaults."""
        engine = ComplianceEngine()
        assert engine._signer_id == "optimization_copilot"
        assert engine._replay_engine is None

    def test_start_audit(self):
        """start_audit creates and stores an AuditLog."""
        engine = ComplianceEngine(signer_id="test")
        spec = {"parameter_specs": []}
        log = engine.start_audit("camp-1", spec, base_seed=10)
        assert isinstance(log, AuditLog)
        assert log.campaign_id == "camp-1"
        assert log.base_seed == 10
        assert log.signer_id == "test"

    def test_start_audit_duplicate_raises(self):
        """start_audit raises ValueError for duplicate campaign_id."""
        engine = ComplianceEngine()
        engine.start_audit("camp-1", {})
        with pytest.raises(ValueError, match="already active"):
            engine.start_audit("camp-1", {})

    def test_record_decision_creates_entry(self):
        """record_decision creates an AuditEntry with chain hash."""
        engine = ComplianceEngine(signer_id="test_signer")
        engine.start_audit("camp-1", {})
        log_entry = _make_log_entry(iteration=0)
        audit_entry = engine.record_decision("camp-1", log_entry)
        assert isinstance(audit_entry, AuditEntry)
        assert audit_entry.iteration == 0
        assert audit_entry.signer_id == "test_signer"
        assert audit_entry.chain_hash != ""
        # First entry: chain_hash == SHA-256(content_hash)
        expected_chain = _compute_chain_hash("", audit_entry.content_hash())
        assert audit_entry.chain_hash == expected_chain

    def test_record_decision_chain_links(self):
        """Multiple record_decision calls produce correctly linked chain hashes."""
        engine = ComplianceEngine()
        engine.start_audit("camp-1", {})
        entry0 = engine.record_decision("camp-1", _make_log_entry(iteration=0))
        entry1 = engine.record_decision("camp-1", _make_log_entry(iteration=1, timestamp=1001.0))
        # entry1's chain hash should depend on entry0's chain hash
        expected = _compute_chain_hash(entry0.chain_hash, entry1.content_hash())
        assert entry1.chain_hash == expected
        # Verify the full chain
        log = engine.get_audit_log("camp-1")
        result = verify_chain(log)
        assert result.valid is True

    def test_get_audit_log(self):
        """get_audit_log returns the active log."""
        engine = ComplianceEngine()
        engine.start_audit("camp-1", {"key": "val"})
        log = engine.get_audit_log("camp-1")
        assert log.campaign_id == "camp-1"

    def test_get_audit_log_unknown_raises(self):
        """get_audit_log raises KeyError for unknown campaign."""
        engine = ComplianceEngine()
        with pytest.raises(KeyError):
            engine.get_audit_log("nonexistent")

    def test_finalize_audit(self):
        """finalize_audit removes from active and returns the log."""
        engine = ComplianceEngine()
        engine.start_audit("camp-1", {})
        engine.record_decision("camp-1", _make_log_entry(iteration=0))
        log = engine.finalize_audit("camp-1")
        assert isinstance(log, AuditLog)
        assert log.n_entries == 1
        # No longer active
        with pytest.raises(KeyError):
            engine.get_audit_log("camp-1")

    def test_verify_chain_delegates(self):
        """ComplianceEngine.verify_chain delegates to module-level verify_chain."""
        engine = ComplianceEngine()
        log = _build_chained_audit_log(n_entries=3)
        result = engine.verify_chain(log)
        assert result.valid is True
        assert result.n_entries == 3

    def test_generate_report(self):
        """generate_report returns a ComplianceReport with correct fields."""
        engine = ComplianceEngine()
        spec = {
            "parameter_specs": [
                {"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0},
            ],
        }
        engine.start_audit("camp-rpt", spec, base_seed=42)
        engine.record_decision("camp-rpt", _make_log_entry(iteration=0))
        engine.record_decision("camp-rpt", _make_log_entry(iteration=1, timestamp=1001.0))
        log = engine.get_audit_log("camp-rpt")
        report = engine.generate_report(log)

        assert isinstance(report, ComplianceReport)
        assert report.campaign_id == "camp-rpt"
        assert report.campaign_summary["n_iterations"] == 2
        assert report.campaign_summary["base_seed"] == 42
        assert len(report.parameter_specs) == 1
        assert len(report.iteration_log) == 2
        assert "final_decision" in report.final_recommendation
        assert report.chain_verification.valid is True
        assert report.generation_timestamp > 0

    def test_replay_step_without_engine_raises(self):
        """replay_step raises ValueError when no ReplayEngine configured."""
        engine = ComplianceEngine(replay_engine=None)
        log = _build_chained_audit_log(n_entries=2)
        with pytest.raises(ValueError, match="ReplayEngine not configured"):
            engine.replay_step(log, iteration=0)

    def test_replay_step_with_mock_engine(self):
        """replay_step returns dict with original entry data when ReplayEngine present."""

        class _FakeReplayEngine:
            """Minimal stand-in to satisfy the replay_engine is not None check."""
            pass

        engine = ComplianceEngine(replay_engine=_FakeReplayEngine())
        log = _build_chained_audit_log(n_entries=2)
        result = engine.replay_step(log, iteration=0)

        assert isinstance(result, dict)
        assert result["iteration"] == 0
        assert "original" in result
        assert result["original"]["iteration"] == 0
        assert "decision_hash" in result
        assert "chain_hash" in result
        assert result["replay_available"] is True

    def test_replay_step_unknown_iteration_raises(self):
        """replay_step raises KeyError for nonexistent iteration."""

        class _FakeReplayEngine:
            pass

        engine = ComplianceEngine(replay_engine=_FakeReplayEngine())
        log = _build_chained_audit_log(n_entries=2)
        with pytest.raises(KeyError, match="No entry for iteration 99"):
            engine.replay_step(log, iteration=99)

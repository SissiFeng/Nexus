"""Tests for the optimization_copilot.safety package (hazards, monitor, emergency)."""

from __future__ import annotations

import time

import pytest

from optimization_copilot.safety import (
    EmergencyAction,
    EmergencyEvaluation,
    EmergencyLog,
    EmergencyProtocol,
    HazardCategory,
    HazardLevel,
    HazardRegistry,
    HazardSpec,
    SafetyEvent,
    SafetyMonitor,
    SafetyStatus,
)


# ── Helpers ───────────────────────────────────────────


def _make_registry() -> HazardRegistry:
    """Create a HazardRegistry with temperature and pressure hazards."""
    registry = HazardRegistry()
    registry.register(HazardSpec(
        parameter_name="temperature",
        category=HazardCategory.THERMAL,
        level=HazardLevel.HIGH,
        lower_safe=20.0,
        upper_safe=200.0,
        description="Thermal decomposition above 200C",
    ))
    registry.register(HazardSpec(
        parameter_name="pressure",
        category=HazardCategory.PRESSURE,
        level=HazardLevel.MODERATE,
        lower_safe=1.0,
        upper_safe=10.0,
        description="Pressure vessel limit",
    ))
    return registry


def _make_safety_event(
    status: SafetyStatus = SafetyStatus.WARNING,
    parameter: str = "temperature",
    value: float = 195.0,
    limit: float = 200.0,
) -> SafetyEvent:
    """Create a SafetyEvent for testing."""
    return SafetyEvent(
        timestamp=time.time(),
        status=status,
        parameter=parameter,
        value=value,
        limit=limit,
        message=f"{parameter}={value} near limit {limit}",
    )


# ── hazards: HazardLevel enum ────────────────────────


def test_hazard_level_values():
    """HazardLevel enum has integer values NONE=0 through CRITICAL=4."""
    assert HazardLevel.NONE.value == 0
    assert HazardLevel.LOW.value == 1
    assert HazardLevel.MODERATE.value == 2
    assert HazardLevel.HIGH.value == 3
    assert HazardLevel.CRITICAL.value == 4


def test_hazard_level_ordering():
    """HazardLevel values allow severity comparison."""
    assert HazardLevel.NONE.value < HazardLevel.LOW.value
    assert HazardLevel.LOW.value < HazardLevel.MODERATE.value
    assert HazardLevel.MODERATE.value < HazardLevel.HIGH.value
    assert HazardLevel.HIGH.value < HazardLevel.CRITICAL.value


# ── hazards: HazardCategory enum ─────────────────────


def test_hazard_category_members():
    """HazardCategory has expected physical categories."""
    categories = {c.value for c in HazardCategory}
    assert "thermal" in categories
    assert "pressure" in categories
    assert "toxicity" in categories


# ── hazards: HazardRegistry ──────────────────────────


def test_registry_register_and_count():
    """HazardRegistry.register adds specs; n_hazards reflects count."""
    registry = HazardRegistry()
    assert registry.n_hazards == 0

    registry.register(HazardSpec(
        parameter_name="temperature",
        category=HazardCategory.THERMAL,
        level=HazardLevel.HIGH,
        lower_safe=20.0,
        upper_safe=200.0,
    ))
    assert registry.n_hazards == 1


def test_registry_classify_point_safe():
    """classify_point returns empty list for in-bounds parameters."""
    registry = _make_registry()
    violated = registry.classify_point({"temperature": 100.0, "pressure": 5.0})
    assert violated == []


def test_registry_classify_point_violated():
    """classify_point returns violated specs for out-of-bounds parameters."""
    registry = _make_registry()
    violated = registry.classify_point({"temperature": 250.0, "pressure": 5.0})
    assert len(violated) == 1
    assert violated[0].parameter_name == "temperature"


def test_registry_max_hazard_level_none():
    """max_hazard_level returns NONE for safe parameters."""
    registry = _make_registry()
    level = registry.max_hazard_level({"temperature": 100.0, "pressure": 5.0})
    assert level == HazardLevel.NONE


def test_registry_max_hazard_level_returns_highest():
    """max_hazard_level returns the highest among violated hazards."""
    registry = _make_registry()
    # Both violated: temperature is HIGH (3), pressure is MODERATE (2)
    level = registry.max_hazard_level({"temperature": 250.0, "pressure": 15.0})
    assert level == HazardLevel.HIGH


# ── monitor: SafetyMonitor.check_point ───────────────


def test_monitor_check_point_safe():
    """check_point returns SAFE and no events for in-bounds parameters."""
    registry = _make_registry()
    monitor = SafetyMonitor(registry, warning_margin=0.1)

    status, events = monitor.check_point({"temperature": 100.0, "pressure": 5.0})
    assert status == SafetyStatus.SAFE
    assert events == []


def test_monitor_check_point_warning():
    """check_point returns WARNING when value is within warning margin."""
    registry = _make_registry()
    # warning_margin=0.1: safe range is 180 (200-20), warning band = 18
    # So temperature > 200-18 = 182 triggers WARNING
    monitor = SafetyMonitor(registry, warning_margin=0.1)

    status, events = monitor.check_point({"temperature": 195.0, "pressure": 5.0})
    assert status == SafetyStatus.WARNING
    assert len(events) == 1
    assert events[0].status == SafetyStatus.WARNING
    assert events[0].parameter == "temperature"


def test_monitor_check_point_danger():
    """check_point returns DANGER when value exceeds safe boundary."""
    registry = _make_registry()
    monitor = SafetyMonitor(registry, warning_margin=0.1)

    status, events = monitor.check_point({"temperature": 250.0, "pressure": 5.0})
    assert status == SafetyStatus.DANGER
    assert len(events) == 1
    assert events[0].status == SafetyStatus.DANGER


def test_monitor_check_point_emergency_escalation():
    """Multiple DANGER events escalate to EMERGENCY status."""
    registry = _make_registry()
    monitor = SafetyMonitor(registry, warning_margin=0.1)

    # Both parameters out of range -> 2 DANGER events -> EMERGENCY
    status, events = monitor.check_point({"temperature": 250.0, "pressure": 15.0})
    assert status == SafetyStatus.EMERGENCY
    assert len(events) == 2


# ── monitor: SafetyMonitor.check_batch ───────────────


def test_monitor_check_batch_filters_unsafe():
    """check_batch excludes points with DANGER or EMERGENCY status."""
    registry = _make_registry()
    monitor = SafetyMonitor(registry, warning_margin=0.1)

    batch = [
        {"temperature": 100.0, "pressure": 5.0},   # SAFE
        {"temperature": 250.0, "pressure": 5.0},    # DANGER
        {"temperature": 190.0, "pressure": 5.0},    # WARNING (within margin)
    ]
    safe_points = monitor.check_batch(batch)
    # SAFE and WARNING are kept; DANGER is excluded
    assert len(safe_points) == 2
    assert safe_points[0]["temperature"] == 100.0
    assert safe_points[1]["temperature"] == 190.0


# ── monitor: SafetyMonitor.safety_margin ─────────────


def test_monitor_safety_margin_in_bounds():
    """safety_margin returns positive value for in-bounds parameters."""
    registry = _make_registry()
    monitor = SafetyMonitor(registry)

    # temperature: range = 180, value = 110 -> dist_lower=90, dist_upper=90 -> margin=90/180=0.5
    margins = monitor.safety_margin({"temperature": 110.0, "pressure": 5.5})
    assert margins["temperature"] == pytest.approx(0.5, abs=0.01)
    assert margins["pressure"] > 0


def test_monitor_safety_margin_out_of_bounds():
    """safety_margin returns negative value for out-of-bounds parameters."""
    registry = _make_registry()
    monitor = SafetyMonitor(registry)

    # temperature=210: dist_lower=190, dist_upper=-10 -> margin=-10/180 < 0
    margins = monitor.safety_margin({"temperature": 210.0})
    assert margins["temperature"] < 0


# ── emergency: EmergencyProtocol.evaluate ────────────


def test_emergency_protocol_continue_no_events():
    """EmergencyProtocol.evaluate returns CONTINUE for empty event list."""
    protocol = EmergencyProtocol()
    evaluation = protocol.evaluate([])
    assert evaluation.action == EmergencyAction.CONTINUE
    assert evaluation.n_warnings == 0
    assert evaluation.n_dangers == 0
    assert evaluation.n_emergencies == 0


def test_emergency_protocol_pause_on_warnings():
    """PAUSE is triggered when warning count meets threshold."""
    protocol = EmergencyProtocol(n_warnings_to_pause=2, n_dangers_to_stop=2)
    events = [
        _make_safety_event(SafetyStatus.WARNING, "temperature", 195.0, 200.0),
        _make_safety_event(SafetyStatus.WARNING, "pressure", 9.5, 10.0),
    ]
    evaluation = protocol.evaluate(events)
    assert evaluation.action == EmergencyAction.PAUSE
    assert evaluation.n_warnings == 2


def test_emergency_protocol_fallback_on_danger():
    """FALLBACK is triggered when danger events exist below stop threshold."""
    # n_dangers_to_stop=2 means 1 danger -> FALLBACK, not STOP
    protocol = EmergencyProtocol(n_dangers_to_stop=2)
    events = [
        _make_safety_event(SafetyStatus.DANGER, "temperature", 250.0, 200.0),
    ]
    evaluation = protocol.evaluate(events)
    assert evaluation.action == EmergencyAction.FALLBACK
    assert evaluation.n_dangers == 1


def test_emergency_protocol_stop_on_danger_threshold():
    """STOP is triggered when danger count meets threshold."""
    protocol = EmergencyProtocol(n_dangers_to_stop=1)
    events = [
        _make_safety_event(SafetyStatus.DANGER, "temperature", 250.0, 200.0),
    ]
    evaluation = protocol.evaluate(events)
    assert evaluation.action == EmergencyAction.STOP


def test_emergency_protocol_stop_on_emergency():
    """STOP is triggered immediately for any EMERGENCY event."""
    protocol = EmergencyProtocol()
    events = [
        _make_safety_event(SafetyStatus.EMERGENCY, "temperature", 300.0, 200.0),
    ]
    evaluation = protocol.evaluate(events)
    assert evaluation.action == EmergencyAction.STOP
    assert evaluation.n_emergencies == 1


# ── emergency: EmergencyLog ──────────────────────────


def test_emergency_log_recording():
    """EmergencyLog records evaluations and supports retrieval."""
    log = EmergencyLog()
    assert len(log) == 0
    assert log.has_stop() is False

    protocol = EmergencyProtocol()

    # Log a CONTINUE evaluation
    eval_continue = protocol.evaluate([])
    log.log(eval_continue)
    assert len(log) == 1
    assert log.has_stop() is False
    assert log.latest().action == EmergencyAction.CONTINUE

    # Log a STOP evaluation
    eval_stop = protocol.evaluate([
        _make_safety_event(SafetyStatus.EMERGENCY, "temperature", 300.0, 200.0),
    ])
    log.log(eval_stop)
    assert len(log) == 2
    assert log.has_stop() is True
    assert log.latest().action == EmergencyAction.STOP


def test_emergency_log_get_log_returns_copy():
    """EmergencyLog.get_log returns a copy of entries."""
    log = EmergencyLog()
    protocol = EmergencyProtocol()
    evaluation = protocol.evaluate([])
    log.log(evaluation)

    entries = log.get_log()
    assert len(entries) == 1
    # Modifying returned list should not affect internal state
    entries.clear()
    assert len(log) == 1

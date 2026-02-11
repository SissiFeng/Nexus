"""Tests for SDL (Self-Driving Lab) monitoring visualization module.

Covers all four panels from v3 spec section 6:
- 6.1 Experiment Status Dashboard
- 6.2 Safety Monitoring
- 6.3 Human-in-the-Loop
- 6.4 Continuous Operation Timeline
"""

from __future__ import annotations

import pytest

from optimization_copilot.visualization.sdl_monitor import (
    SDLDashboardData,
    plot_continuous_operation_timeline,
    plot_experiment_status_dashboard,
    plot_human_in_the_loop,
    plot_safety_monitoring,
)


# ── 6.1 Experiment Status Dashboard ─────────────────────────────────────────


class TestExperimentStatusDashboard:
    """Tests for plot_experiment_status_dashboard."""

    def test_empty_inputs_returns_valid_plotdata(self):
        """Empty queues and no hardware should still produce valid PlotData."""
        result = plot_experiment_status_dashboard(
            queue=[], running=[], completed=[], hardware_status={},
        )
        assert result.plot_type == "sdl_status_dashboard"
        assert result.data["n_queued"] == 0
        assert result.data["n_running"] == 0
        assert result.data["n_completed"] == 0
        assert result.data["n_total"] == 0
        assert result.svg is not None

    def test_counts_correct(self):
        """Verify queue/running/completed counts."""
        queue = [
            {"id": "q1", "params": {}, "priority": 1, "scheduled_time": ""},
            {"id": "q2", "params": {}, "priority": 2, "scheduled_time": ""},
        ]
        running = [
            {"id": "r1", "params": {}, "start_time": "", "progress": 0.5},
        ]
        completed = [
            {"id": "c1", "params": {}, "result": 1.0, "end_time": ""},
            {"id": "c2", "params": {}, "result": 2.0, "end_time": ""},
            {"id": "c3", "params": {}, "result": 3.0, "end_time": ""},
        ]
        result = plot_experiment_status_dashboard(
            queue=queue, running=running, completed=completed,
            hardware_status={},
        )
        assert result.data["n_queued"] == 2
        assert result.data["n_running"] == 1
        assert result.data["n_completed"] == 3
        assert result.data["n_total"] == 6

    def test_hardware_status_color_coding(self):
        """Idle should map to green, error to red."""
        hw = {
            "mixer": {"state": "idle"},
            "reactor": {"state": "error", "error": "overheated"},
        }
        result = plot_experiment_status_dashboard(
            queue=[], running=[], completed=[], hardware_status=hw,
        )
        entries = result.data["hardware_entries"]
        mixer_entry = next(e for e in entries if e["device"] == "mixer")
        reactor_entry = next(e for e in entries if e["device"] == "reactor")
        assert mixer_entry["color"] == "#4CAF50"   # green
        assert reactor_entry["color"] == "#F44336"  # red

    def test_multiple_hardware_devices(self):
        """Multiple devices should all appear in hardware entries."""
        hw = {
            "pump_A": {"state": "running"},
            "pump_B": {"state": "idle"},
            "sensor_1": {"state": "warning"},
            "heater": {"state": "offline"},
        }
        result = plot_experiment_status_dashboard(
            queue=[], running=[], completed=[], hardware_status=hw,
        )
        assert result.metadata["n_hardware_devices"] == 4
        device_names = {e["device"] for e in result.data["hardware_entries"]}
        assert device_names == {"pump_A", "pump_B", "sensor_1", "heater"}

    def test_svg_generated_with_correct_dimensions(self):
        """SVG string should contain the dashboard dimensions."""
        result = plot_experiment_status_dashboard(
            queue=[], running=[], completed=[],
            hardware_status={"dev1": {"state": "ok"}},
        )
        assert result.svg is not None
        assert 'width="900"' in result.svg
        assert 'height="700"' in result.svg
        assert "</svg>" in result.svg

    def test_dashboard_data_structure_completeness(self):
        """Data dict should contain all required keys for all quadrants."""
        queue = [{"id": "q1", "params": {}, "priority": 0, "scheduled_time": ""}]
        running = [{"id": "r1", "params": {}, "start_time": "", "progress": 0.3}]
        completed = [{"id": "c1", "params": {}, "result": 42, "end_time": ""}]
        hw = {"dev1": {"state": "idle"}}
        result = plot_experiment_status_dashboard(
            queue=queue, running=running, completed=completed,
            hardware_status=hw,
        )
        assert "queue_items" in result.data
        assert "hardware_entries" in result.data
        assert "anomaly_alerts" in result.data
        assert "n_queued" in result.data
        assert "n_running" in result.data
        assert "n_completed" in result.data
        assert "n_total" in result.data
        assert "quadrants" in result.metadata
        assert len(result.metadata["quadrants"]) == 4


# ── 6.2 Safety Monitoring ───────────────────────────────────────────────────


class TestSafetyMonitoring:
    """Tests for plot_safety_monitoring."""

    def test_all_constraints_satisfied_no_violations(self):
        """When all constraints are satisfied, violation count is zero."""
        history = [
            {"time": "2024-01-01T00:00:00", "constraint_name": "temp",
             "satisfied": True, "value": 25.0, "limit": 50.0},
            {"time": "2024-01-01T00:01:00", "constraint_name": "pressure",
             "satisfied": True, "value": 1.0, "limit": 3.0},
        ]
        result = plot_safety_monitoring(history, anomaly_scores=[0.5, 1.0])
        assert result.plot_type == "sdl_safety_monitoring"
        assert result.data["n_violations"] == 0
        assert result.data["violation_rate"] == 0.0

    def test_mixed_satisfied_violated_constraints(self):
        """Correctly count violations in a mixed set."""
        history = [
            {"time": "2024-01-01T00:00:00", "constraint_name": "temp",
             "satisfied": True, "value": 25.0, "limit": 50.0},
            {"time": "2024-01-01T00:01:00", "constraint_name": "temp",
             "satisfied": False, "value": 55.0, "limit": 50.0},
            {"time": "2024-01-01T00:02:00", "constraint_name": "pressure",
             "satisfied": False, "value": 4.0, "limit": 3.0},
            {"time": "2024-01-01T00:03:00", "constraint_name": "pressure",
             "satisfied": True, "value": 2.5, "limit": 3.0},
        ]
        result = plot_safety_monitoring(history, anomaly_scores=[])
        assert result.data["n_violations"] == 2
        assert result.data["n_total_constraints"] == 4
        assert result.data["violation_rate"] == pytest.approx(50.0)

    def test_anomaly_scores_below_threshold_no_alerts(self):
        """Scores below threshold produce zero anomaly alerts."""
        result = plot_safety_monitoring(
            constraint_history=[],
            anomaly_scores=[0.5, 1.0, 2.0, 2.9],
            threshold=3.0,
        )
        assert result.data["n_anomaly_alerts"] == 0

    def test_anomaly_scores_above_threshold_alerts(self):
        """Scores above threshold should be counted as alerts."""
        result = plot_safety_monitoring(
            constraint_history=[],
            anomaly_scores=[1.0, 3.5, 2.0, 4.0, 5.0],
            threshold=3.0,
        )
        assert result.data["n_anomaly_alerts"] == 3
        assert result.data["max_anomaly_score"] == pytest.approx(5.0)

    def test_empty_constraint_history(self):
        """Empty constraint history should produce valid result."""
        result = plot_safety_monitoring(
            constraint_history=[],
            anomaly_scores=[1.0, 2.0],
        )
        assert result.data["n_total_constraints"] == 0
        assert result.data["n_violations"] == 0
        assert result.data["violation_rate"] == 0.0
        assert result.svg is not None

    def test_threshold_edge_case(self):
        """Score exactly at threshold should NOT be an alert (strict >)."""
        result = plot_safety_monitoring(
            constraint_history=[],
            anomaly_scores=[3.0],
            threshold=3.0,
        )
        assert result.data["n_anomaly_alerts"] == 0


# ── 6.3 Human-in-the-Loop ──────────────────────────────────────────────────


class TestHumanInTheLoop:
    """Tests for plot_human_in_the_loop."""

    def test_all_approved_100_percent(self):
        """All approve decisions yield 100% approval rate."""
        experiments = [
            {"id": i, "params": {}, "algorithm_reasoning": ""}
            for i in range(5)
        ]
        decisions = ["approve"] * 5
        result = plot_human_in_the_loop(experiments, decisions)
        assert result.plot_type == "sdl_human_in_loop"
        assert result.data["approval_rate"] == pytest.approx(100.0)
        assert result.data["override_rate"] == pytest.approx(0.0)
        assert result.data["n_approve"] == 5

    def test_mixed_approve_reject_modify(self):
        """Mixed decisions produce correct rates."""
        experiments = [
            {"id": i, "params": {}, "algorithm_reasoning": ""}
            for i in range(10)
        ]
        # 5 approve, 3 reject, 2 modify
        decisions = (
            ["approve"] * 5 + ["reject"] * 3 + ["modify"] * 2
        )
        result = plot_human_in_the_loop(experiments, decisions)
        assert result.data["n_approve"] == 5
        assert result.data["n_reject"] == 3
        assert result.data["n_modify"] == 2
        assert result.data["approval_rate"] == pytest.approx(50.0)
        assert result.data["rejection_rate"] == pytest.approx(30.0)
        assert result.data["modification_rate"] == pytest.approx(20.0)
        assert result.data["override_rate"] == pytest.approx(50.0)

    def test_override_rate_computation(self):
        """Override rate is reject + modify as percentage."""
        experiments = [
            {"id": i, "params": {}, "algorithm_reasoning": ""}
            for i in range(4)
        ]
        decisions = ["approve", "reject", "modify", "reject"]
        result = plot_human_in_the_loop(experiments, decisions)
        # 3 out of 4 are overrides
        assert result.data["override_rate"] == pytest.approx(75.0)
        # Windowed rates should exist
        assert len(result.data["windowed_override_rates"]) == 4

    def test_empty_experiments_list(self):
        """Empty input should produce valid PlotData with zero rates."""
        result = plot_human_in_the_loop(
            proposed_experiments=[], human_decisions=[],
        )
        assert result.data["n_total"] == 0
        assert result.data["approval_rate"] == pytest.approx(0.0)
        assert result.data["override_rate"] == pytest.approx(0.0)
        assert result.svg is not None

    def test_with_reasoning(self):
        """When reasoning is provided, it should appear in details."""
        experiments = [
            {"id": 1, "params": {"x": 1}, "algorithm_reasoning": "explore"},
        ]
        decisions = ["reject"]
        reasoning = ["out of safety bounds"]
        result = plot_human_in_the_loop(experiments, decisions, reasoning)
        assert result.metadata["has_reasoning"] is True
        detail = result.data["decision_details"][0]
        assert detail["human_reasoning"] == "out of safety bounds"
        assert detail["decision"] == "reject"

    def test_without_reasoning(self):
        """Without reasoning, metadata flag should be False."""
        experiments = [
            {"id": 1, "params": {}, "algorithm_reasoning": ""},
        ]
        decisions = ["approve"]
        result = plot_human_in_the_loop(experiments, decisions)
        assert result.metadata["has_reasoning"] is False

    def test_decision_counts_correct(self):
        """Verify individual decision counts match input."""
        experiments = [
            {"id": i, "params": {}, "algorithm_reasoning": ""}
            for i in range(6)
        ]
        decisions = ["approve", "approve", "reject", "modify", "modify", "approve"]
        result = plot_human_in_the_loop(experiments, decisions)
        assert result.data["n_approve"] == 3
        assert result.data["n_reject"] == 1
        assert result.data["n_modify"] == 2
        assert result.data["n_total"] == 6


# ── 6.4 Continuous Operation Timeline ───────────────────────────────────────


class TestContinuousOperationTimeline:
    """Tests for plot_continuous_operation_timeline."""

    def test_empty_operations_list(self):
        """Empty operations should produce valid PlotData."""
        result = plot_continuous_operation_timeline(
            start_time="2024-01-01T00:00:00",
            operations=[],
            failures=[],
            recovery_actions=[],
        )
        assert result.plot_type == "sdl_operation_timeline"
        assert result.data["n_operations"] == 0
        assert result.data["devices"] == []
        assert result.svg is not None

    def test_single_device_swim_lane(self):
        """Single device should create one swim lane."""
        ops = [
            {"time": "2024-01-01T00:00:00", "type": "measure",
             "device": "sensor_1", "status": "ok", "details": ""},
            {"time": "2024-01-01T00:05:00", "type": "measure",
             "device": "sensor_1", "status": "ok", "details": ""},
        ]
        result = plot_continuous_operation_timeline(
            start_time="2024-01-01T00:00:00",
            operations=ops, failures=[], recovery_actions=[],
        )
        assert result.data["devices"] == ["sensor_1"]
        assert result.metadata["n_devices"] == 1
        assert len(result.data["swim_lanes"]["sensor_1"]) == 2

    def test_multiple_device_swim_lanes(self):
        """Multiple devices should create separate swim lanes."""
        ops = [
            {"time": "2024-01-01T00:00:00", "type": "measure",
             "device": "sensor_1", "status": "ok"},
            {"time": "2024-01-01T00:01:00", "type": "dispense",
             "device": "pump_A", "status": "ok"},
            {"time": "2024-01-01T00:02:00", "type": "heat",
             "device": "heater", "status": "ok"},
        ]
        result = plot_continuous_operation_timeline(
            start_time="2024-01-01T00:00:00",
            operations=ops, failures=[], recovery_actions=[],
        )
        assert set(result.data["devices"]) == {"sensor_1", "pump_A", "heater"}
        assert result.metadata["n_devices"] == 3

    def test_failures_marked_correctly(self):
        """Failure entries should be captured with severity colours."""
        ops = [
            {"time": "2024-01-01T00:00:00", "type": "measure",
             "device": "sensor_1", "status": "ok"},
        ]
        failures = [
            {"time": "2024-01-01T00:10:00", "device": "sensor_1",
             "error": "overheated", "severity": "critical"},
            {"time": "2024-01-01T00:20:00", "device": "sensor_1",
             "error": "drift", "severity": "medium"},
        ]
        result = plot_continuous_operation_timeline(
            start_time="2024-01-01T00:00:00",
            operations=ops, failures=failures, recovery_actions=[],
        )
        assert result.data["n_failures"] == 2
        f_entries = result.data["failure_entries"]
        assert len(f_entries) == 2
        # Critical -> red, medium -> orange
        assert f_entries[0]["color"] == "#F44336"
        assert f_entries[1]["color"] == "#FF9800"

    def test_recovery_actions_tracked(self):
        """Recovery actions should be counted and success rates computed."""
        recoveries = [
            {"time": "2024-01-01T00:15:00", "action": "restart", "success": True},
            {"time": "2024-01-01T00:25:00", "action": "recalibrate", "success": False},
            {"time": "2024-01-01T00:30:00", "action": "restart", "success": True},
        ]
        result = plot_continuous_operation_timeline(
            start_time="2024-01-01T00:00:00",
            operations=[], failures=[], recovery_actions=recoveries,
        )
        assert result.data["n_recovery_actions"] == 3
        assert result.data["n_successful_recoveries"] == 2
        assert result.data["recovery_rate"] == pytest.approx(200 / 3)

    def test_throughput_computation(self):
        """Throughput should be operations per hour."""
        # 6 operations over 2 hours = 3 ops/hr
        ops = [
            {"time": "2024-01-01T00:00:00", "type": "measure",
             "device": "dev1", "status": "ok"},
            {"time": "2024-01-01T00:20:00", "type": "measure",
             "device": "dev1", "status": "ok"},
            {"time": "2024-01-01T00:40:00", "type": "measure",
             "device": "dev1", "status": "ok"},
            {"time": "2024-01-01T01:00:00", "type": "measure",
             "device": "dev1", "status": "ok"},
            {"time": "2024-01-01T01:20:00", "type": "measure",
             "device": "dev1", "status": "ok"},
            {"time": "2024-01-01T02:00:00", "type": "measure",
             "device": "dev1", "status": "ok"},
        ]
        result = plot_continuous_operation_timeline(
            start_time="2024-01-01T00:00:00",
            operations=ops, failures=[], recovery_actions=[],
        )
        # 6 ops over 2 hours
        assert result.data["throughput_ops_per_hour"] == pytest.approx(3.0)
        assert result.data["total_hours"] == pytest.approx(2.0)

    def test_sdl_dashboard_data_autonomy_default(self):
        """SDLDashboardData should default to autonomy_level=3."""
        data = SDLDashboardData()
        assert data.autonomy_level == 3
        assert data.refresh_interval_s == 5.0
        assert data.alert_channels == []

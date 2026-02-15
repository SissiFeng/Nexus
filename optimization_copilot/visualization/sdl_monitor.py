"""SDL (Self-Driving Lab) real-time monitoring visualization module.

Implements four monitoring panels for autonomous laboratory operations
per v3 specification section 6:

* **6.1 Experiment Status Dashboard** -- 4-quadrant overview of queue,
  hardware, progress, and anomaly alerts.
* **6.2 Safety Monitoring** -- constraint satisfaction timeline and
  anomaly score chart with configurable thresholds.
* **6.3 Human-in-the-Loop** -- decision summary with approval, rejection,
  and modification rates plus windowed override tracking.
* **6.4 Continuous Operation Timeline** -- swim-lane timeline per device
  with failure markers, recovery actions, and throughput computation.

All functions return :class:`~optimization_copilot.visualization.models.PlotData`
instances and use :class:`~optimization_copilot.visualization.svg_renderer.SVGCanvas`
for SVG generation.  Pure Python stdlib only -- no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from optimization_copilot.visualization.models import PlotData
from optimization_copilot.visualization.svg_renderer import SVGCanvas


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DASHBOARD_WIDTH = 900
_DASHBOARD_HEIGHT = 700
_SAFETY_WIDTH = 800
_SAFETY_HEIGHT = 500
_HITL_WIDTH = 700
_HITL_HEIGHT = 400
_TIMELINE_WIDTH = 900
_TIMELINE_HEIGHT = 500

_COLOR_GREEN = "#4CAF50"
_COLOR_YELLOW = "#FFC107"
_COLOR_RED = "#F44336"
_COLOR_BLUE = "#2196F3"
_COLOR_GRAY = "#9E9E9E"
_COLOR_ORANGE = "#FF9800"
_COLOR_LIGHT_GREEN = "#C8E6C9"
_COLOR_LIGHT_RED = "#FFCDD2"
_COLOR_DARK_TEXT = "#212121"
_COLOR_BG = "#FAFAFA"
_COLOR_BORDER = "#E0E0E0"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-format timestamp string to datetime.

    Handles both ``YYYY-MM-DDTHH:MM:SS`` and ``YYYY-MM-DDTHH:MM:SSZ``
    formats.
    """
    cleaned = ts.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts!r}")


def _hardware_color(state: str) -> str:
    """Map a hardware state string to a status colour."""
    state_lower = state.lower()
    if state_lower in ("idle", "ready", "ok", "online"):
        return _COLOR_GREEN
    if state_lower in ("busy", "running", "warming", "warning"):
        return _COLOR_YELLOW
    if state_lower in ("error", "fault", "offline", "failed"):
        return _COLOR_RED
    return _COLOR_GRAY


def _severity_color(severity: str) -> str:
    """Map a failure severity string to a colour."""
    severity_lower = severity.lower()
    if severity_lower in ("critical", "high"):
        return _COLOR_RED
    if severity_lower in ("medium", "warning"):
        return _COLOR_ORANGE
    return _COLOR_YELLOW


# ---------------------------------------------------------------------------
# 6.1  Experiment Status Dashboard
# ---------------------------------------------------------------------------


def plot_experiment_status_dashboard(
    queue: list[dict[str, Any]],
    running: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    hardware_status: dict[str, dict[str, Any]],
) -> PlotData:
    """Build a 4-quadrant experiment status dashboard.

    Parameters
    ----------
    queue:
        Pending experiments, each with keys ``id``, ``params``,
        ``priority``, ``scheduled_time``.
    running:
        Currently executing experiments, each with ``id``, ``params``,
        ``start_time``, ``progress`` (0-1 float).
    completed:
        Finished experiments, each with ``id``, ``params``, ``result``,
        ``end_time``.
    hardware_status:
        Mapping of device name to status dict with keys ``state`` (str),
        and optionally ``progress`` (float), ``error`` (str), ``value``
        (any).

    Returns
    -------
    PlotData
        Dashboard data with ``plot_type="sdl_status_dashboard"``.
    """
    n_queued = len(queue)
    n_running = len(running)
    n_completed = len(completed)
    n_total = n_queued + n_running + n_completed

    # -- Collect anomaly / error alerts from hardware -------------------------
    anomaly_alerts: list[dict[str, Any]] = []
    hw_entries: list[dict[str, Any]] = []
    for device_name, status in hardware_status.items():
        state = status.get("state", "unknown")
        color = _hardware_color(state)
        entry: dict[str, Any] = {
            "device": device_name,
            "state": state,
            "color": color,
        }
        if "progress" in status:
            entry["progress"] = status["progress"]
        if "value" in status:
            entry["value"] = status["value"]
        hw_entries.append(entry)

        if status.get("error"):
            anomaly_alerts.append({
                "device": device_name,
                "error": status["error"],
                "severity": "error",
            })

    # -- Queue Gantt data (Q1) ------------------------------------------------
    queue_items: list[dict[str, Any]] = []
    for exp in queue:
        queue_items.append({
            "id": exp.get("id", "?"),
            "priority": exp.get("priority", 0),
            "scheduled_time": exp.get("scheduled_time", ""),
        })

    # -- Build SVG ------------------------------------------------------------
    canvas = SVGCanvas(_DASHBOARD_WIDTH, _DASHBOARD_HEIGHT, background=_COLOR_BG)
    mid_x = _DASHBOARD_WIDTH // 2
    mid_y = _DASHBOARD_HEIGHT // 2
    pad = 10

    # Title
    canvas.text(
        _DASHBOARD_WIDTH / 2, 25,
        "SDL Experiment Status Dashboard",
        font_size=16, fill=_COLOR_DARK_TEXT, text_anchor="middle",
    )

    # Divider lines
    canvas.line(mid_x, 40, mid_x, _DASHBOARD_HEIGHT - pad,
                stroke=_COLOR_BORDER, stroke_width=1)
    canvas.line(pad, mid_y, _DASHBOARD_WIDTH - pad, mid_y,
                stroke=_COLOR_BORDER, stroke_width=1)

    # Q1 -- Experiment Queue (top-left)
    canvas.text(pad + 10, 55, "Experiment Queue", font_size=13,
                fill=_COLOR_DARK_TEXT)
    y_offset = 75
    for i, item in enumerate(queue_items[:8]):
        bar_width = max(30, min(200, 200 - item["priority"] * 20))
        canvas.rect(pad + 10, y_offset + i * 30, bar_width, 20,
                    fill=_COLOR_BLUE, rx=3, ry=3)
        canvas.text(pad + 15, y_offset + i * 30 + 14,
                    f"Exp {item['id']} (p={item['priority']})",
                    font_size=10, fill="white")

    # Q2 -- Hardware Status (top-right)
    canvas.text(mid_x + 10, 55, "Hardware Status", font_size=13,
                fill=_COLOR_DARK_TEXT)
    y_offset = 75
    for i, hw in enumerate(hw_entries[:8]):
        # Status light
        canvas.circle(mid_x + 25, y_offset + i * 30 + 10, 8,
                      fill=hw["color"], stroke=_COLOR_DARK_TEXT,
                      stroke_width=0.5)
        label = f"{hw['device']}: {hw['state']}"
        canvas.text(mid_x + 40, y_offset + i * 30 + 14, label,
                    font_size=10, fill=_COLOR_DARK_TEXT)

    # Q3 -- Progress Overview (bottom-left)
    canvas.text(pad + 10, mid_y + 15, "Progress Overview", font_size=13,
                fill=_COLOR_DARK_TEXT)
    bar_y = mid_y + 40
    bar_max_w = mid_x - 2 * pad - 20
    for label, count, color in [
        ("Completed", n_completed, _COLOR_GREEN),
        ("Running", n_running, _COLOR_YELLOW),
        ("Queued", n_queued, _COLOR_BLUE),
    ]:
        bar_w = (count / max(n_total, 1)) * bar_max_w
        canvas.rect(pad + 10, bar_y, max(bar_w, 2), 25,
                    fill=color, rx=3, ry=3)
        canvas.text(pad + 15, bar_y + 17,
                    f"{label}: {count}",
                    font_size=10, fill=_COLOR_DARK_TEXT)
        bar_y += 35

    # Q4 -- Anomaly Alerts (bottom-right)
    canvas.text(mid_x + 10, mid_y + 15, "Anomaly Alerts", font_size=13,
                fill=_COLOR_DARK_TEXT)
    y_offset = mid_y + 40
    if not anomaly_alerts:
        canvas.text(mid_x + 20, y_offset + 15,
                    "No active alerts",
                    font_size=11, fill=_COLOR_GREEN)
    else:
        for i, alert in enumerate(anomaly_alerts[:6]):
            canvas.rect(mid_x + 10, y_offset + i * 28, 420, 22,
                        fill=_COLOR_LIGHT_RED, rx=3, ry=3)
            canvas.text(mid_x + 15, y_offset + i * 28 + 15,
                        f"[{alert['device']}] {alert['error']}",
                        font_size=10, fill=_COLOR_RED)

    svg_str = canvas.to_string()

    return PlotData(
        plot_type="sdl_status_dashboard",
        data={
            "queue_items": queue_items,
            "hardware_entries": hw_entries,
            "anomaly_alerts": anomaly_alerts,
            "n_queued": n_queued,
            "n_running": n_running,
            "n_completed": n_completed,
            "n_total": n_total,
        },
        metadata={
            "n_hardware_devices": len(hardware_status),
            "quadrants": [
                "experiment_queue",
                "hardware_status",
                "progress_overview",
                "anomaly_alerts",
            ],
        },
        svg=svg_str,
    )


# ---------------------------------------------------------------------------
# 6.2  Safety Monitoring
# ---------------------------------------------------------------------------


def plot_safety_monitoring(
    constraint_history: list[dict[str, Any]],
    anomaly_scores: list[float],
    threshold: float = 3.0,
) -> PlotData:
    """Build a safety monitoring visualisation.

    Parameters
    ----------
    constraint_history:
        Time-ordered entries, each with ``time`` (str ISO), ``constraint_name``
        (str), ``satisfied`` (bool), ``value`` (float), ``limit`` (float).
    anomaly_scores:
        Anomaly detection scores over time (one per time step).
    threshold:
        Standard-deviation threshold above which anomaly scores trigger alerts.

    Returns
    -------
    PlotData
        Safety dashboard with ``plot_type="sdl_safety_monitoring"``.
    """
    # -- Constraint analysis --------------------------------------------------
    n_total_constraints = len(constraint_history)
    n_violations = sum(1 for c in constraint_history if not c.get("satisfied", True))
    violation_rate = (n_violations / n_total_constraints * 100.0
                      if n_total_constraints > 0 else 0.0)

    constraint_segments: list[dict[str, Any]] = []
    seen_constraints: set[str] = set()
    for entry in constraint_history:
        name = entry.get("constraint_name", "unknown")
        seen_constraints.add(name)
        constraint_segments.append({
            "time": entry.get("time", ""),
            "constraint_name": name,
            "satisfied": entry.get("satisfied", True),
            "value": entry.get("value", 0.0),
            "limit": entry.get("limit", 0.0),
            "color": _COLOR_GREEN if entry.get("satisfied", True) else _COLOR_RED,
        })

    # -- Anomaly analysis -----------------------------------------------------
    n_anomaly_alerts = sum(1 for s in anomaly_scores if s > threshold)
    max_anomaly_score = max(anomaly_scores) if anomaly_scores else 0.0

    # -- Build SVG ------------------------------------------------------------
    canvas = SVGCanvas(_SAFETY_WIDTH, _SAFETY_HEIGHT, background=_COLOR_BG)

    # Title
    canvas.text(
        _SAFETY_WIDTH / 2, 25,
        "SDL Safety Monitoring",
        font_size=16, fill=_COLOR_DARK_TEXT, text_anchor="middle",
    )

    # Divider
    upper_h = 220
    canvas.line(10, upper_h + 30, _SAFETY_WIDTH - 10, upper_h + 30,
                stroke=_COLOR_BORDER, stroke_width=1)

    # -- Upper panel: constraint satisfaction timeline ------------------------
    canvas.text(15, 50, "Constraint Satisfaction Timeline", font_size=12,
                fill=_COLOR_DARK_TEXT)

    if constraint_segments:
        lane_h = 16
        lane_gap = 4
        x_start = 15
        x_end = _SAFETY_WIDTH - 15
        n_segments = len(constraint_segments)
        seg_w = max(2, (x_end - x_start) / max(n_segments, 1))

        # Group by constraint name for swim lanes
        constraint_names = sorted(seen_constraints)
        for lane_idx, cname in enumerate(constraint_names[:8]):
            y = 65 + lane_idx * (lane_h + lane_gap)
            canvas.text(x_start, y + 12, cname, font_size=9,
                        fill=_COLOR_DARK_TEXT)
            seg_count = 0
            for seg in constraint_segments:
                if seg["constraint_name"] == cname:
                    sx = x_start + 120 + seg_count * seg_w
                    canvas.rect(sx, y, max(seg_w - 1, 1), lane_h,
                                fill=seg["color"], rx=1, ry=1)
                    seg_count += 1

    # -- Lower panel: anomaly score line chart --------------------------------
    lower_y_start = upper_h + 50
    canvas.text(15, lower_y_start, "Anomaly Scores", font_size=12,
                fill=_COLOR_DARK_TEXT)

    chart_x = 50
    chart_y = lower_y_start + 20
    chart_w = _SAFETY_WIDTH - 100
    chart_h = _SAFETY_HEIGHT - lower_y_start - 60

    # Axes
    canvas.line(chart_x, chart_y, chart_x, chart_y + chart_h,
                stroke=_COLOR_DARK_TEXT, stroke_width=1)
    canvas.line(chart_x, chart_y + chart_h,
                chart_x + chart_w, chart_y + chart_h,
                stroke=_COLOR_DARK_TEXT, stroke_width=1)

    if anomaly_scores:
        max_score = max(max(anomaly_scores), threshold * 1.2, 1.0)
        n_scores = len(anomaly_scores)
        dx = chart_w / max(n_scores - 1, 1)

        # Threshold line
        threshold_y = chart_y + chart_h - (threshold / max_score) * chart_h
        canvas.line(chart_x, threshold_y, chart_x + chart_w, threshold_y,
                    stroke=_COLOR_RED, stroke_width=1,
                    stroke_dasharray="5,3")
        canvas.text(chart_x + chart_w + 5, threshold_y + 4,
                    f"threshold={threshold}",
                    font_size=8, fill=_COLOR_RED)

        # Score polyline
        points: list[tuple[float, float]] = []
        for i, score in enumerate(anomaly_scores):
            px = chart_x + i * dx
            py = chart_y + chart_h - (score / max_score) * chart_h
            points.append((px, py))
        canvas.polyline(points, stroke=_COLOR_BLUE, stroke_width=1.5)

        # Highlight points above threshold
        for i, score in enumerate(anomaly_scores):
            if score > threshold:
                px = chart_x + i * dx
                py = chart_y + chart_h - (score / max_score) * chart_h
                canvas.circle(px, py, 3, fill=_COLOR_RED)

    svg_str = canvas.to_string()

    return PlotData(
        plot_type="sdl_safety_monitoring",
        data={
            "constraint_segments": constraint_segments,
            "constraint_names": sorted(seen_constraints),
            "anomaly_scores": list(anomaly_scores),
            "threshold": threshold,
            "n_total_constraints": n_total_constraints,
            "n_violations": n_violations,
            "violation_rate": violation_rate,
            "n_anomaly_alerts": n_anomaly_alerts,
            "max_anomaly_score": max_anomaly_score,
        },
        metadata={
            "n_constraint_types": len(seen_constraints),
            "panels": ["constraint_timeline", "anomaly_scores"],
        },
        svg=svg_str,
    )


# ---------------------------------------------------------------------------
# 6.3  Human-in-the-Loop
# ---------------------------------------------------------------------------


def plot_human_in_the_loop(
    proposed_experiments: list[dict[str, Any]],
    human_decisions: list[str],
    reasoning: list[str] | None = None,
) -> PlotData:
    """Build a human-in-the-loop decision summary.

    Parameters
    ----------
    proposed_experiments:
        Algorithm-proposed experiments, each with ``id``, ``params``,
        ``algorithm_reasoning``.
    human_decisions:
        One decision per experiment: ``"approve"``, ``"reject"``, or
        ``"modify"``.
    reasoning:
        Optional human reasoning strings, one per decision.

    Returns
    -------
    PlotData
        Decision summary with ``plot_type="sdl_human_in_loop"``.
    """
    n_total = len(human_decisions)

    # -- Compute rates --------------------------------------------------------
    n_approve = sum(1 for d in human_decisions if d == "approve")
    n_reject = sum(1 for d in human_decisions if d == "reject")
    n_modify = sum(1 for d in human_decisions if d == "modify")

    approval_rate = (n_approve / n_total * 100.0) if n_total > 0 else 0.0
    rejection_rate = (n_reject / n_total * 100.0) if n_total > 0 else 0.0
    modification_rate = (n_modify / n_total * 100.0) if n_total > 0 else 0.0

    # Override rate = anything that is not "approve"
    n_override = n_reject + n_modify
    override_rate = (n_override / n_total * 100.0) if n_total > 0 else 0.0

    # -- Windowed override rate (window of 5) ---------------------------------
    window_size = 5
    windowed_override_rates: list[float] = []
    for i in range(len(human_decisions)):
        start = max(0, i - window_size + 1)
        window = human_decisions[start:i + 1]
        w_overrides = sum(1 for d in window if d != "approve")
        windowed_override_rates.append(w_overrides / len(window) * 100.0)

    # -- Decision details -----------------------------------------------------
    decision_details: list[dict[str, Any]] = []
    for i, exp in enumerate(proposed_experiments):
        detail: dict[str, Any] = {
            "experiment_id": exp.get("id", i),
            "params": exp.get("params", {}),
            "algorithm_reasoning": exp.get("algorithm_reasoning", ""),
            "decision": human_decisions[i] if i < n_total else "pending",
        }
        if reasoning and i < len(reasoning):
            detail["human_reasoning"] = reasoning[i]
        decision_details.append(detail)

    # -- Build SVG ------------------------------------------------------------
    canvas = SVGCanvas(_HITL_WIDTH, _HITL_HEIGHT, background=_COLOR_BG)

    # Title
    canvas.text(
        _HITL_WIDTH / 2, 25,
        "Human-in-the-Loop Decision Summary",
        font_size=16, fill=_COLOR_DARK_TEXT, text_anchor="middle",
    )

    # Bar chart for approval / rejection / modification
    bar_x = 50
    bar_y = 55
    bar_max_w = 300
    bar_h = 28

    for label, count, rate, color in [
        ("Approved", n_approve, approval_rate, _COLOR_GREEN),
        ("Rejected", n_reject, rejection_rate, _COLOR_RED),
        ("Modified", n_modify, modification_rate, _COLOR_ORANGE),
    ]:
        bar_w = (rate / 100.0) * bar_max_w if n_total > 0 else 0
        canvas.rect(bar_x + 80, bar_y, max(bar_w, 2), bar_h,
                    fill=color, rx=3, ry=3)
        canvas.text(bar_x, bar_y + 19, label, font_size=11,
                    fill=_COLOR_DARK_TEXT)
        canvas.text(bar_x + 85 + bar_w, bar_y + 19,
                    f"{count} ({rate:.1f}%)",
                    font_size=10, fill=_COLOR_DARK_TEXT)
        bar_y += 40

    # Override rate summary
    canvas.text(bar_x, bar_y + 15,
                f"Overall Override Rate: {override_rate:.1f}%",
                font_size=12, fill=_COLOR_DARK_TEXT)

    # Windowed override rate line (if data exists)
    if windowed_override_rates:
        chart_x = 50
        chart_y_start = 230
        chart_w = _HITL_WIDTH - 100
        chart_h = 120

        canvas.text(chart_x, chart_y_start - 5,
                    "Windowed Override Rate (window=5)",
                    font_size=11, fill=_COLOR_DARK_TEXT)

        # Axes
        canvas.line(chart_x, chart_y_start, chart_x, chart_y_start + chart_h,
                    stroke=_COLOR_DARK_TEXT, stroke_width=1)
        canvas.line(chart_x, chart_y_start + chart_h,
                    chart_x + chart_w, chart_y_start + chart_h,
                    stroke=_COLOR_DARK_TEXT, stroke_width=1)

        n_pts = len(windowed_override_rates)
        dx = chart_w / max(n_pts - 1, 1)
        points: list[tuple[float, float]] = []
        for i, rate_val in enumerate(windowed_override_rates):
            px = chart_x + i * dx
            py = chart_y_start + chart_h - (rate_val / 100.0) * chart_h
            points.append((px, py))
        canvas.polyline(points, stroke=_COLOR_ORANGE, stroke_width=1.5)

    svg_str = canvas.to_string()

    return PlotData(
        plot_type="sdl_human_in_loop",
        data={
            "decision_details": decision_details,
            "n_total": n_total,
            "n_approve": n_approve,
            "n_reject": n_reject,
            "n_modify": n_modify,
            "approval_rate": approval_rate,
            "rejection_rate": rejection_rate,
            "modification_rate": modification_rate,
            "override_rate": override_rate,
            "windowed_override_rates": windowed_override_rates,
        },
        metadata={
            "window_size": window_size,
            "has_reasoning": reasoning is not None,
            "n_experiments": len(proposed_experiments),
        },
        svg=svg_str,
    )


# ---------------------------------------------------------------------------
# 6.4  Continuous Operation Timeline
# ---------------------------------------------------------------------------


def plot_continuous_operation_timeline(
    start_time: str,
    operations: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    recovery_actions: list[dict[str, Any]],
) -> PlotData:
    """Build a swim-lane continuous operation timeline.

    Parameters
    ----------
    start_time:
        ISO-format start timestamp for the monitoring window.
    operations:
        Time-ordered events, each with ``time`` (str ISO), ``type`` (str),
        ``device`` (str), ``status`` (str), and optionally ``details``
        (str).
    failures:
        Failure events, each with ``time`` (str ISO), ``device`` (str),
        ``error`` (str), ``severity`` (str).
    recovery_actions:
        Recovery attempts, each with ``time`` (str ISO), ``action`` (str),
        ``success`` (bool).

    Returns
    -------
    PlotData
        Timeline data with ``plot_type="sdl_operation_timeline"``.
    """
    # -- Identify unique devices for swim lanes --------------------------------
    devices: list[str] = []
    seen_devices: set[str] = set()
    for op in operations:
        dev = op.get("device", "unknown")
        if dev not in seen_devices:
            devices.append(dev)
            seen_devices.add(dev)
    for f in failures:
        dev = f.get("device", "unknown")
        if dev not in seen_devices:
            devices.append(dev)
            seen_devices.add(dev)

    # -- Parse time range ------------------------------------------------------
    try:
        t_start = _parse_iso(start_time)
    except (ValueError, TypeError):
        t_start = datetime(2024, 1, 1)

    # Collect all timestamps to compute range
    all_times: list[datetime] = [t_start]
    for op in operations:
        try:
            all_times.append(_parse_iso(op.get("time", start_time)))
        except (ValueError, TypeError):
            pass
    for f in failures:
        try:
            all_times.append(_parse_iso(f.get("time", start_time)))
        except (ValueError, TypeError):
            pass
    for r in recovery_actions:
        try:
            all_times.append(_parse_iso(r.get("time", start_time)))
        except (ValueError, TypeError):
            pass

    t_min = min(all_times)
    t_max = max(all_times)
    total_seconds = max((t_max - t_min).total_seconds(), 1.0)
    total_hours = total_seconds / 3600.0

    # -- Compute throughput (operations per hour) -----------------------------
    throughput = len(operations) / max(total_hours, 0.001)

    # -- Build swim-lane data -------------------------------------------------
    swim_lanes: dict[str, list[dict[str, Any]]] = {d: [] for d in devices}
    for op in operations:
        dev = op.get("device", "unknown")
        try:
            t = _parse_iso(op.get("time", start_time))
            offset = (t - t_min).total_seconds()
        except (ValueError, TypeError):
            offset = 0.0
        lane_entry = {
            "time": op.get("time", ""),
            "time_offset_s": offset,
            "type": op.get("type", ""),
            "status": op.get("status", ""),
            "details": op.get("details", ""),
            "is_failure": False,
            "is_recovery": False,
        }
        if dev in swim_lanes:
            swim_lanes[dev].append(lane_entry)

    # -- Mark failures ---------------------------------------------------------
    failure_entries: list[dict[str, Any]] = []
    for f in failures:
        dev = f.get("device", "unknown")
        try:
            t = _parse_iso(f.get("time", start_time))
            offset = (t - t_min).total_seconds()
        except (ValueError, TypeError):
            offset = 0.0
        entry = {
            "time": f.get("time", ""),
            "time_offset_s": offset,
            "device": dev,
            "error": f.get("error", ""),
            "severity": f.get("severity", "medium"),
            "color": _severity_color(f.get("severity", "medium")),
        }
        failure_entries.append(entry)

    # -- Track recovery actions ------------------------------------------------
    recovery_entries: list[dict[str, Any]] = []
    n_successful_recoveries = 0
    for r in recovery_actions:
        try:
            t = _parse_iso(r.get("time", start_time))
            offset = (t - t_min).total_seconds()
        except (ValueError, TypeError):
            offset = 0.0
        success = r.get("success", False)
        if success:
            n_successful_recoveries += 1
        recovery_entries.append({
            "time": r.get("time", ""),
            "time_offset_s": offset,
            "action": r.get("action", ""),
            "success": success,
            "color": _COLOR_GREEN if success else _COLOR_RED,
        })

    recovery_rate = (
        n_successful_recoveries / len(recovery_actions) * 100.0
        if recovery_actions else 0.0
    )

    # -- Build SVG ------------------------------------------------------------
    canvas = SVGCanvas(_TIMELINE_WIDTH, _TIMELINE_HEIGHT, background=_COLOR_BG)

    # Title
    canvas.text(
        _TIMELINE_WIDTH / 2, 25,
        "SDL Continuous Operation Timeline",
        font_size=16, fill=_COLOR_DARK_TEXT, text_anchor="middle",
    )

    # Throughput annotation
    canvas.text(
        _TIMELINE_WIDTH - 15, 25,
        f"Throughput: {throughput:.1f} ops/hr",
        font_size=10, fill=_COLOR_DARK_TEXT, text_anchor="end",
    )

    # Swim lanes
    lane_h = 40
    lane_gap = 10
    lane_start_y = 55
    chart_x = 120
    chart_w = _TIMELINE_WIDTH - chart_x - 30

    for lane_idx, device in enumerate(devices[:8]):
        y = lane_start_y + lane_idx * (lane_h + lane_gap)

        # Lane background
        bg_color = "#F5F5F5" if lane_idx % 2 == 0 else _COLOR_BG
        canvas.rect(chart_x, y, chart_w, lane_h,
                    fill=bg_color, stroke=_COLOR_BORDER, stroke_width=0.5)

        # Device label
        canvas.text(10, y + lane_h / 2 + 4, device,
                    font_size=10, fill=_COLOR_DARK_TEXT)

        # Operation marks
        for op_entry in swim_lanes.get(device, []):
            frac = op_entry["time_offset_s"] / total_seconds
            px = chart_x + frac * chart_w
            canvas.circle(px, y + lane_h / 2, 4,
                          fill=_COLOR_BLUE, stroke_width=0.5)

    # Failure markers (red diamonds)
    for f_entry in failure_entries:
        dev = f_entry["device"]
        if dev not in devices:
            continue
        lane_idx = devices.index(dev)
        if lane_idx >= 8:
            continue
        y = lane_start_y + lane_idx * (lane_h + lane_gap) + lane_h / 2
        frac = f_entry["time_offset_s"] / total_seconds
        px = chart_x + frac * chart_w
        # Diamond marker
        size = 6
        canvas.polygon(
            [(px, y - size), (px + size, y), (px, y + size), (px - size, y)],
            fill=f_entry["color"],
        )

    # Recovery markers (green/red triangles)
    for r_entry in recovery_entries:
        frac = r_entry["time_offset_s"] / total_seconds
        px = chart_x + frac * chart_w
        # Place below swim lanes
        ry = lane_start_y + len(devices[:8]) * (lane_h + lane_gap) + 15
        size = 5
        canvas.polygon(
            [(px, ry - size), (px + size, ry + size), (px - size, ry + size)],
            fill=r_entry["color"],
        )

    svg_str = canvas.to_string()

    return PlotData(
        plot_type="sdl_operation_timeline",
        data={
            "devices": devices,
            "swim_lanes": swim_lanes,
            "failure_entries": failure_entries,
            "recovery_entries": recovery_entries,
            "throughput_ops_per_hour": throughput,
            "total_hours": total_hours,
            "n_operations": len(operations),
            "n_failures": len(failures),
            "n_recovery_actions": len(recovery_actions),
            "n_successful_recoveries": n_successful_recoveries,
            "recovery_rate": recovery_rate,
        },
        metadata={
            "start_time": start_time,
            "n_devices": len(devices),
            "total_seconds": total_seconds,
        },
        svg=svg_str,
    )


# ---------------------------------------------------------------------------
# SDLDashboardData container
# ---------------------------------------------------------------------------


@dataclass
class SDLDashboardData:
    """Container for SDL dashboard configuration.

    Parameters
    ----------
    autonomy_level : int
        Autonomy level from 1 (fully manual) to 5 (fully autonomous).
        Default is 3 (human-in-the-loop with approval gates).
    refresh_interval_s : float
        Dashboard refresh interval in seconds.
    alert_channels : list[str]
        Notification channels for critical alerts.
    """

    autonomy_level: int = 3
    refresh_interval_s: float = 5.0
    alert_channels: list[str] = field(default_factory=list)

"""SVG visualization for multi-stage experiment workflows.

Provides functions to render stage DAG flowcharts and cost savings
reports as SVG graphics using the existing SVGCanvas.
"""

from __future__ import annotations

from typing import Any

from optimization_copilot.visualization.svg_renderer import SVGCanvas
from optimization_copilot.workflow.stage import StageDAG


# Color palette
_COLORS = {
    "pending": "#e0e0e0",
    "completed": "#4caf50",
    "active": "#2196f3",
    "border": "#333333",
    "text": "#222222",
    "arrow": "#666666",
    "background": "#ffffff",
    "bar_cost_spent": "#f44336",
    "bar_cost_saved": "#4caf50",
    "bar_evaluations": "#2196f3",
}


def render_stage_flow(
    dag: StageDAG,
    completed: set[str] | None = None,
) -> str:
    """Render a DAG flowchart as SVG.

    Stages are drawn as rounded rectangles arranged by topological order,
    with arrows showing dependencies. Completed stages are highlighted
    in green.

    Parameters
    ----------
    dag : StageDAG
        The stage DAG to visualize.
    completed : set[str] | None
        Set of completed stage names. If None, no highlighting.

    Returns
    -------
    str
        SVG XML string.
    """
    if completed is None:
        completed = set()

    try:
        topo_order = dag.topological_order()
    except ValueError:
        topo_order = [s.name for s in dag.stages()]

    n_stages = len(topo_order)
    if n_stages == 0:
        canvas = SVGCanvas(width=200, height=100, background=_COLORS["background"])
        canvas.text(100, 50, "No stages", font_size=14, text_anchor="middle",
                    dominant_baseline="middle", fill=_COLORS["text"])
        return canvas.to_string()

    # Layout parameters
    box_w = 140
    box_h = 50
    h_gap = 60
    v_margin = 40
    h_margin = 40

    canvas_w = h_margin * 2 + n_stages * box_w + (n_stages - 1) * h_gap
    canvas_h = v_margin * 2 + box_h + 40  # extra space for labels

    canvas = SVGCanvas(width=int(canvas_w), height=int(canvas_h),
                       background=_COLORS["background"])

    # Draw title
    canvas.text(canvas_w / 2, 18, "Stage Workflow", font_size=16,
                text_anchor="middle", fill=_COLORS["text"])

    # Compute positions
    positions: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(topo_order):
        x = h_margin + i * (box_w + h_gap)
        y = v_margin + 10
        positions[name] = (x, y)

    # Draw arrows first (behind boxes)
    for name in topo_order:
        stage = dag.get_stage(name)
        x2, y2 = positions[name]
        for dep in stage.dependencies:
            if dep in positions:
                x1, y1 = positions[dep]
                # Arrow from right edge of dep to left edge of this stage
                canvas.line(
                    x1 + box_w, y1 + box_h / 2,
                    x2, y2 + box_h / 2,
                    stroke=_COLORS["arrow"],
                    stroke_width=2,
                )
                # Simple arrowhead
                arrow_size = 8
                canvas.polygon(
                    [
                        (x2, y2 + box_h / 2),
                        (x2 - arrow_size, y2 + box_h / 2 - arrow_size / 2),
                        (x2 - arrow_size, y2 + box_h / 2 + arrow_size / 2),
                    ],
                    fill=_COLORS["arrow"],
                )

    # Draw boxes
    for name in topo_order:
        x, y = positions[name]
        if name in completed:
            fill = _COLORS["completed"]
            text_fill = "#ffffff"
        else:
            fill = _COLORS["pending"]
            text_fill = _COLORS["text"]

        canvas.rect(x, y, box_w, box_h, fill=fill,
                    stroke=_COLORS["border"], stroke_width=1.5, rx=8, ry=8)

        # Stage name
        canvas.text(x + box_w / 2, y + box_h / 2, name, font_size=12,
                    text_anchor="middle", dominant_baseline="middle",
                    fill=text_fill)

        # Cost label below box
        stage = dag.get_stage(name)
        canvas.text(x + box_w / 2, y + box_h + 14,
                    f"cost: {stage.cost:.1f}",
                    font_size=9, text_anchor="middle", fill="#888888")

    return canvas.to_string()


def render_savings_report(report: dict[str, Any]) -> str:
    """Render a cost savings report as an SVG bar chart.

    Parameters
    ----------
    report : dict[str, Any]
        Report from ``MultiStageBayesianOptimizer.get_savings_report()``.

    Returns
    -------
    str
        SVG XML string.
    """
    canvas_w = 500
    canvas_h = 300
    canvas = SVGCanvas(width=canvas_w, height=canvas_h,
                       background=_COLORS["background"])

    # Title
    canvas.text(canvas_w / 2, 25, "Cost Savings Report", font_size=16,
                text_anchor="middle", fill=_COLORS["text"])

    # Extract values
    cost_spent = report.get("total_cost_spent", 0.0)
    cost_saved = report.get("total_cost_saved", 0.0)
    total_evals = report.get("total_evaluations", 0)
    early_terms = report.get("early_terminations", 0)
    savings_ratio = report.get("savings_ratio", 0.0)

    # Bar chart area
    chart_left = 120
    chart_top = 50
    chart_right = canvas_w - 40
    chart_bottom = canvas_h - 60
    chart_width = chart_right - chart_left
    chart_height = chart_bottom - chart_top

    # Determine max value for scaling
    max_val = max(cost_spent, cost_saved, 1.0)

    # Draw bars
    bar_items = [
        ("Cost Spent", cost_spent, _COLORS["bar_cost_spent"]),
        ("Cost Saved", cost_saved, _COLORS["bar_cost_saved"]),
    ]

    n_bars = len(bar_items)
    bar_gap = 20
    bar_h = min(40, (chart_height - (n_bars - 1) * bar_gap) / n_bars)

    for i, (label, value, color) in enumerate(bar_items):
        y = chart_top + i * (bar_h + bar_gap)
        bar_width = (value / max_val) * chart_width if max_val > 0 else 0

        # Label
        canvas.text(chart_left - 10, y + bar_h / 2, label, font_size=11,
                    text_anchor="end", dominant_baseline="middle",
                    fill=_COLORS["text"])

        # Bar
        canvas.rect(chart_left, y, max(bar_width, 1), bar_h,
                    fill=color, rx=3, ry=3)

        # Value text
        canvas.text(chart_left + bar_width + 8, y + bar_h / 2,
                    f"{value:.1f}", font_size=11,
                    dominant_baseline="middle", fill=_COLORS["text"])

    # Summary stats
    summary_y = chart_top + n_bars * (bar_h + bar_gap) + 20
    stats = [
        f"Total evaluations: {total_evals}",
        f"Early terminations: {early_terms}",
        f"Savings ratio: {savings_ratio:.1%}",
    ]
    for i, stat in enumerate(stats):
        canvas.text(chart_left, summary_y + i * 20, stat,
                    font_size=11, fill=_COLORS["text"])

    return canvas.to_string()

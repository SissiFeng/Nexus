"""Standard 8-figure SVG campaign report template.

Produces paper-grade fixed deliverables for each optimization campaign:
best-so-far convergence, calibration curve, drift timeline, batch
comparison, recommendation coverage, uncertainty coverage, feature
importance, and Pareto front.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.visualization.svg_renderer import SVGCanvas
from optimization_copilot.visualization.models import PlotData


# -- Shared helpers ----------------------------------------------------------

def _scale_linear(
    value: float, data_min: float, data_max: float,
    pixel_min: float, pixel_max: float,
) -> float:
    """Map a data value to pixel coordinates."""
    if abs(data_max - data_min) < 1e-15:
        return (pixel_min + pixel_max) / 2.0
    return pixel_min + (value - data_min) / (data_max - data_min) * (pixel_max - pixel_min)


def _compute_nice_ticks(data_min: float, data_max: float, n_ticks: int = 5) -> list[float]:
    """Compute nicely-spaced tick values."""
    if abs(data_max - data_min) < 1e-15:
        return [data_min]
    raw_step = (data_max - data_min) / max(n_ticks - 1, 1)
    # Round to a nice number
    magnitude = 10 ** math.floor(math.log10(max(abs(raw_step), 1e-15)))
    normalized = raw_step / magnitude
    if normalized <= 1.5:
        nice_step = 1.0 * magnitude
    elif normalized <= 3.5:
        nice_step = 2.0 * magnitude
    elif normalized <= 7.5:
        nice_step = 5.0 * magnitude
    else:
        nice_step = 10.0 * magnitude

    start = math.floor(data_min / nice_step) * nice_step
    ticks: list[float] = []
    val = start
    while val <= data_max + nice_step * 0.01:
        if val >= data_min - nice_step * 0.01:
            ticks.append(round(val, 10))
        val += nice_step
    return ticks if ticks else [data_min, data_max]


def _draw_axes(
    canvas: SVGCanvas,
    x_min: float, x_max: float, y_min: float, y_max: float,
    plot_left: float, plot_right: float,
    plot_top: float, plot_bottom: float,
    x_label: str = "", y_label: str = "",
    title: str = "",
) -> None:
    """Draw axes with ticks and labels on a canvas."""
    # Axes lines
    canvas.line(plot_left, plot_bottom, plot_right, plot_bottom, stroke="#333", stroke_width=1.5)
    canvas.line(plot_left, plot_top, plot_left, plot_bottom, stroke="#333", stroke_width=1.5)

    # X ticks
    x_ticks = _compute_nice_ticks(x_min, x_max)
    for tick in x_ticks:
        px = _scale_linear(tick, x_min, x_max, plot_left, plot_right)
        canvas.line(px, plot_bottom, px, plot_bottom + 5, stroke="#333", stroke_width=1)
        label = f"{tick:.3g}"
        canvas.text(px, plot_bottom + 18, label, font_size=10, text_anchor="middle", fill="#555")

    # Y ticks
    y_ticks = _compute_nice_ticks(y_min, y_max)
    for tick in y_ticks:
        py = _scale_linear(tick, y_min, y_max, plot_bottom, plot_top)
        canvas.line(plot_left - 5, py, plot_left, py, stroke="#333", stroke_width=1)
        label = f"{tick:.3g}"
        canvas.text(plot_left - 8, py, label, font_size=10, text_anchor="end", dominant_baseline="middle", fill="#555")

    # Labels
    if x_label:
        mid_x = (plot_left + plot_right) / 2
        canvas.text(mid_x, plot_bottom + 35, x_label, font_size=12, text_anchor="middle", fill="#333")
    if y_label:
        mid_y = (plot_top + plot_bottom) / 2
        canvas.text(plot_left - 45, mid_y, y_label, font_size=12, text_anchor="middle", fill="#333",
                    transform=f"rotate(-90,{plot_left - 45},{mid_y})")
    if title:
        mid_x = (plot_left + plot_right) / 2
        canvas.text(mid_x, plot_top - 10, title, font_size=14, text_anchor="middle", fill="#222")


# -- Plot margins (shared) ---------------------------------------------------

_W, _H = 400, 300
_MARGIN = {"left": 60, "right": 20, "top": 30, "bottom": 45}

def _plot_area() -> tuple[float, float, float, float]:
    return _MARGIN["left"], _W - _MARGIN["right"], _MARGIN["top"], _H - _MARGIN["bottom"]


# -- 8 figure functions ------------------------------------------------------

def plot_convergence_curve(
    iterations: list[int], best_values: list[float]
) -> PlotData:
    """Figure 1: Best-so-far convergence curve."""
    canvas = SVGCanvas(_W, _H, background="white")
    pl, pr, pt, pb = _plot_area()

    if not iterations or not best_values:
        canvas.text(_W / 2, _H / 2, "No data", font_size=14, text_anchor="middle", fill="#999")
        return PlotData(plot_type="convergence_curve", data={}, svg=canvas.to_string())

    x_min, x_max = min(iterations), max(iterations)
    y_min, y_max = min(best_values), max(best_values)
    # Add 5% padding
    y_range = y_max - y_min if y_max != y_min else 1.0
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range

    _draw_axes(canvas, x_min, x_max, y_min, y_max, pl, pr, pt, pb,
               x_label="Iteration", y_label="Best Objective", title="Best-so-far Convergence")

    # Plot line
    points = []
    for i, (it, bv) in enumerate(zip(iterations, best_values)):
        px = _scale_linear(it, x_min, x_max, pl, pr)
        py = _scale_linear(bv, y_min, y_max, pb, pt)
        points.append((px, py))
    canvas.polyline(points, stroke="#2196F3", stroke_width=2)

    return PlotData(
        plot_type="convergence_curve",
        data={"iterations": iterations, "best_values": best_values},
        metadata={"final_best": best_values[-1]},
        svg=canvas.to_string(),
    )


def plot_calibration_curve(
    predicted: list[float], actual: list[float]
) -> PlotData:
    """Figure 2: Calibration curve (predicted vs actual) with R² and RMSE."""
    canvas = SVGCanvas(_W, _H, background="white")
    pl, pr, pt, pb = _plot_area()

    if not predicted or not actual:
        canvas.text(_W / 2, _H / 2, "No data", font_size=14, text_anchor="middle", fill="#999")
        return PlotData(plot_type="calibration_curve", data={}, svg=canvas.to_string())

    all_vals = predicted + actual
    v_min, v_max = min(all_vals), max(all_vals)
    v_range = v_max - v_min if v_max != v_min else 1.0
    v_min -= 0.05 * v_range
    v_max += 0.05 * v_range

    _draw_axes(canvas, v_min, v_max, v_min, v_max, pl, pr, pt, pb,
               x_label="Predicted", y_label="Actual", title="Calibration")

    # Diagonal reference line
    px1 = _scale_linear(v_min, v_min, v_max, pl, pr)
    py1 = _scale_linear(v_min, v_min, v_max, pb, pt)
    px2 = _scale_linear(v_max, v_min, v_max, pl, pr)
    py2 = _scale_linear(v_max, v_min, v_max, pb, pt)
    canvas.line(px1, py1, px2, py2, stroke="#ccc", stroke_width=1, stroke_dasharray="4,4")

    # Scatter points
    for p, a in zip(predicted, actual):
        px = _scale_linear(p, v_min, v_max, pl, pr)
        py = _scale_linear(a, v_min, v_max, pb, pt)
        canvas.circle(px, py, 3, fill="#E91E63", opacity=0.7)

    # Compute R² and RMSE
    n = len(predicted)
    mean_a = sum(actual) / n
    ss_res = sum((actual[i] - predicted[i]) ** 2 for i in range(n))
    ss_tot = sum((a - mean_a) ** 2 for a in actual)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    rmse = math.sqrt(ss_res / n)

    canvas.text(pr - 5, pt + 15, f"R²={r_squared:.3f}", font_size=10, text_anchor="end", fill="#333")
    canvas.text(pr - 5, pt + 28, f"RMSE={rmse:.3f}", font_size=10, text_anchor="end", fill="#333")

    return PlotData(
        plot_type="calibration_curve",
        data={"predicted": predicted, "actual": actual},
        metadata={"r_squared": r_squared, "rmse": rmse},
        svg=canvas.to_string(),
    )


def plot_drift_timeline(
    iterations: list[int], model_errors: list[float]
) -> PlotData:
    """Figure 3: Drift timeline showing model error over time with trend."""
    canvas = SVGCanvas(_W, _H, background="white")
    pl, pr, pt, pb = _plot_area()

    if not iterations or not model_errors:
        canvas.text(_W / 2, _H / 2, "No data", font_size=14, text_anchor="middle", fill="#999")
        return PlotData(plot_type="drift_timeline", data={}, svg=canvas.to_string())

    x_min, x_max = min(iterations), max(iterations)
    y_min, y_max = min(model_errors), max(model_errors)
    y_range = y_max - y_min if y_max != y_min else 1.0
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range

    _draw_axes(canvas, x_min, x_max, y_min, y_max, pl, pr, pt, pb,
               x_label="Iteration", y_label="Model Error", title="Drift Timeline")

    # Data points + line
    points = []
    for it, err in zip(iterations, model_errors):
        px = _scale_linear(it, x_min, x_max, pl, pr)
        py = _scale_linear(err, y_min, y_max, pb, pt)
        points.append((px, py))
        canvas.circle(px, py, 2.5, fill="#FF9800", opacity=0.8)
    canvas.polyline(points, stroke="#FF9800", stroke_width=1.5, opacity=0.6)

    # Linear trend line
    n = len(iterations)
    x_mean = sum(iterations) / n
    y_mean = sum(model_errors) / n
    cov = sum((iterations[i] - x_mean) * (model_errors[i] - y_mean) for i in range(n))
    var_x = sum((it - x_mean) ** 2 for it in iterations)
    slope = cov / var_x if var_x > 1e-15 else 0.0
    intercept = y_mean - slope * x_mean

    trend_y0 = slope * x_min + intercept
    trend_y1 = slope * x_max + intercept
    tx0 = _scale_linear(x_min, x_min, x_max, pl, pr)
    ty0 = _scale_linear(trend_y0, y_min, y_max, pb, pt)
    tx1 = _scale_linear(x_max, x_min, x_max, pl, pr)
    ty1 = _scale_linear(trend_y1, y_min, y_max, pb, pt)
    canvas.line(tx0, ty0, tx1, ty1, stroke="#F44336", stroke_width=2, stroke_dasharray="6,3")

    canvas.text(pr - 5, pt + 15, f"slope={slope:.4f}", font_size=10, text_anchor="end", fill="#F44336")

    return PlotData(
        plot_type="drift_timeline",
        data={"iterations": iterations, "model_errors": model_errors},
        metadata={"trend_slope": slope, "trend_intercept": intercept},
        svg=canvas.to_string(),
    )


def plot_batch_comparison(
    labels: list[str], means: list[float], stds: list[float]
) -> PlotData:
    """Figure 4: Batch comparison bar chart with error bars."""
    canvas = SVGCanvas(_W, _H, background="white")
    pl, pr, pt, pb = _plot_area()

    if not labels or not means:
        canvas.text(_W / 2, _H / 2, "No data", font_size=14, text_anchor="middle", fill="#999")
        return PlotData(plot_type="batch_comparison", data={}, svg=canvas.to_string())

    n = len(labels)
    y_max_val = max(m + s for m, s in zip(means, stds)) if stds else max(means)
    y_min_val = min(0.0, min(m - s for m, s in zip(means, stds)) if stds else min(means))
    y_range = y_max_val - y_min_val if y_max_val != y_min_val else 1.0
    y_min_val -= 0.05 * y_range
    y_max_val += 0.1 * y_range

    # Axes
    canvas.line(pl, pb, pr, pb, stroke="#333", stroke_width=1.5)
    canvas.line(pl, pt, pl, pb, stroke="#333", stroke_width=1.5)

    y_ticks = _compute_nice_ticks(y_min_val, y_max_val)
    for tick in y_ticks:
        py = _scale_linear(tick, y_min_val, y_max_val, pb, pt)
        canvas.line(pl - 5, py, pl, py, stroke="#333", stroke_width=1)
        canvas.text(pl - 8, py, f"{tick:.2g}", font_size=10, text_anchor="end", dominant_baseline="middle", fill="#555")

    canvas.text(_W / 2, pt - 10, "Batch Comparison", font_size=14, text_anchor="middle", fill="#222")

    # Bars
    bar_width = (pr - pl) / (n * 1.5 + 0.5)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4", "#795548", "#607D8B"]

    for i in range(n):
        x_center = pl + (i * 1.5 + 1.0) * bar_width
        bar_h = _scale_linear(means[i], y_min_val, y_max_val, pb, pt)
        zero_y = _scale_linear(0.0, y_min_val, y_max_val, pb, pt)
        color = colors[i % len(colors)]

        top = min(bar_h, zero_y)
        height = abs(bar_h - zero_y)
        canvas.rect(x_center - bar_width / 2, top, bar_width, height, fill=color, opacity=0.8)

        # Error bar
        if i < len(stds) and stds[i] > 0:
            err_top = _scale_linear(means[i] + stds[i], y_min_val, y_max_val, pb, pt)
            err_bot = _scale_linear(means[i] - stds[i], y_min_val, y_max_val, pb, pt)
            canvas.line(x_center, err_top, x_center, err_bot, stroke="#333", stroke_width=1.5)
            canvas.line(x_center - 4, err_top, x_center + 4, err_top, stroke="#333", stroke_width=1.5)
            canvas.line(x_center - 4, err_bot, x_center + 4, err_bot, stroke="#333", stroke_width=1.5)

        # Label
        canvas.text(x_center, pb + 15, labels[i], font_size=9, text_anchor="middle", fill="#333")

    return PlotData(
        plot_type="batch_comparison",
        data={"labels": labels, "means": means, "stds": stds},
        metadata={},
        svg=canvas.to_string(),
    )


def plot_recommendation_coverage(
    candidate_xy: list[tuple[float, float]],
    observed_xy: list[tuple[float, float]],
) -> PlotData:
    """Figure 5: Recommendation coverage scatter plot."""
    canvas = SVGCanvas(_W, _H, background="white")
    pl, pr, pt, pb = _plot_area()

    all_points = list(candidate_xy) + list(observed_xy)
    if not all_points:
        canvas.text(_W / 2, _H / 2, "No data", font_size=14, text_anchor="middle", fill="#999")
        return PlotData(plot_type="recommendation_coverage", data={}, svg=canvas.to_string())

    x_vals = [p[0] for p in all_points]
    y_vals = [p[1] for p in all_points]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    x_min -= 0.05 * x_range; x_max += 0.05 * x_range
    y_min -= 0.05 * y_range; y_max += 0.05 * y_range

    _draw_axes(canvas, x_min, x_max, y_min, y_max, pl, pr, pt, pb,
               x_label="Dim 1", y_label="Dim 2", title="Recommendation Coverage")

    # Candidates (light)
    for cx, cy in candidate_xy:
        px = _scale_linear(cx, x_min, x_max, pl, pr)
        py = _scale_linear(cy, y_min, y_max, pb, pt)
        canvas.circle(px, py, 3, fill="#90CAF9", opacity=0.5)

    # Observed (bold)
    for ox, oy in observed_xy:
        px = _scale_linear(ox, x_min, x_max, pl, pr)
        py = _scale_linear(oy, y_min, y_max, pb, pt)
        canvas.circle(px, py, 5, fill="#F44336", stroke="#B71C1C", stroke_width=1, opacity=0.9)

    return PlotData(
        plot_type="recommendation_coverage",
        data={
            "candidate_xy": [list(p) for p in candidate_xy],
            "observed_xy": [list(p) for p in observed_xy],
        },
        metadata={"n_candidates": len(candidate_xy), "n_observed": len(observed_xy)},
        svg=canvas.to_string(),
    )


def plot_uncertainty_coverage(
    grid_x: list[float], grid_y: list[float],
    uncertainty: list[list[float]],
) -> PlotData:
    """Figure 6: Uncertainty heatmap on a 2D grid."""
    canvas = SVGCanvas(_W, _H, background="white")
    pl, pr, pt, pb = _plot_area()

    if not grid_x or not grid_y or not uncertainty:
        canvas.text(_W / 2, _H / 2, "No data", font_size=14, text_anchor="middle", fill="#999")
        return PlotData(plot_type="uncertainty_coverage", data={}, svg=canvas.to_string())

    nx, ny = len(grid_x), len(grid_y)
    x_min, x_max = min(grid_x), max(grid_x)
    y_min, y_max = min(grid_y), max(grid_y)

    canvas.text(_W / 2, pt - 10, "Uncertainty Coverage", font_size=14, text_anchor="middle", fill="#222")

    # Flatten uncertainty for min/max
    flat_u = [u for row in uncertainty for u in row]
    u_min, u_max = min(flat_u), max(flat_u)

    cell_w = (pr - pl) / max(nx - 1, 1) if nx > 1 else (pr - pl)
    cell_h = (pb - pt) / max(ny - 1, 1) if ny > 1 else (pb - pt)

    for i in range(min(ny, len(uncertainty))):
        for j in range(min(nx, len(uncertainty[i]) if i < len(uncertainty) else 0)):
            val = uncertainty[i][j]
            # Map to color intensity (0=light, 1=dark blue)
            t = (val - u_min) / (u_max - u_min) if u_max > u_min else 0.5
            r = int(255 * (1 - t * 0.8))
            g = int(255 * (1 - t * 0.6))
            b = 255
            color = f"#{r:02x}{g:02x}{b:02x}"

            px = pl + j * cell_w - cell_w / 2 if nx > 1 else pl
            py = pt + i * cell_h - cell_h / 2 if ny > 1 else pt
            canvas.rect(px, py, cell_w + 1, cell_h + 1, fill=color)

    # Axes labels
    canvas.text(_W / 2, pb + 20, "Dim 1", font_size=11, text_anchor="middle", fill="#333")

    return PlotData(
        plot_type="uncertainty_coverage",
        data={"grid_x": grid_x, "grid_y": grid_y, "uncertainty": uncertainty},
        metadata={"u_min": u_min, "u_max": u_max},
        svg=canvas.to_string(),
    )


def plot_feature_importance(
    names: list[str], importances: list[float]
) -> PlotData:
    """Figure 7: Horizontal bar chart of feature importances."""
    canvas = SVGCanvas(_W, _H, background="white")
    pl, pr, pt, pb = _plot_area()

    if not names or not importances:
        canvas.text(_W / 2, _H / 2, "No data", font_size=14, text_anchor="middle", fill="#999")
        return PlotData(plot_type="feature_importance", data={}, svg=canvas.to_string())

    n = len(names)
    max_imp = max(abs(v) for v in importances) if importances else 1.0

    canvas.text(_W / 2, pt - 10, "Feature Importance", font_size=14, text_anchor="middle", fill="#222")

    bar_height = (pb - pt) / (n + 1)

    # Sort by importance descending
    paired = sorted(zip(importances, names), reverse=True)

    for i, (imp, name) in enumerate(paired):
        y_center = pt + (i + 1) * bar_height
        bar_len = abs(imp) / max_imp * (pr - pl - 80)
        color = "#4CAF50" if imp >= 0 else "#F44336"

        canvas.rect(pl + 70, y_center - bar_height * 0.35, bar_len, bar_height * 0.7,
                     fill=color, opacity=0.8, rx=2, ry=2)
        canvas.text(pl + 65, y_center, name, font_size=10, text_anchor="end",
                    dominant_baseline="middle", fill="#333")
        canvas.text(pl + 73 + bar_len, y_center, f"{imp:.3f}", font_size=9,
                    dominant_baseline="middle", fill="#555")

    return PlotData(
        plot_type="feature_importance",
        data={"names": names, "importances": importances},
        metadata={"max_importance": max_imp},
        svg=canvas.to_string(),
    )


def plot_pareto_front(
    obj1: list[float], obj2: list[float],
    is_dominated: list[bool],
) -> PlotData:
    """Figure 8: Pareto front scatter plot."""
    canvas = SVGCanvas(_W, _H, background="white")
    pl, pr, pt, pb = _plot_area()

    if not obj1 or not obj2:
        canvas.text(_W / 2, _H / 2, "No data", font_size=14, text_anchor="middle", fill="#999")
        return PlotData(plot_type="pareto_front", data={}, svg=canvas.to_string())

    x_min, x_max = min(obj1), max(obj1)
    y_min, y_max = min(obj2), max(obj2)
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    x_min -= 0.05 * x_range; x_max += 0.05 * x_range
    y_min -= 0.05 * y_range; y_max += 0.05 * y_range

    _draw_axes(canvas, x_min, x_max, y_min, y_max, pl, pr, pt, pb,
               x_label="Objective 1", y_label="Objective 2", title="Pareto Front")

    # Pareto front line (non-dominated points, sorted by obj1)
    pareto_points = []
    for i in range(len(obj1)):
        if i < len(is_dominated) and not is_dominated[i]:
            pareto_points.append((obj1[i], obj2[i]))
    pareto_points.sort()

    if len(pareto_points) > 1:
        line_pts = [
            (_scale_linear(px, x_min, x_max, pl, pr),
             _scale_linear(py, y_min, y_max, pb, pt))
            for px, py in pareto_points
        ]
        canvas.polyline(line_pts, stroke="#F44336", stroke_width=2)

    # All points
    for i in range(len(obj1)):
        px = _scale_linear(obj1[i], x_min, x_max, pl, pr)
        py = _scale_linear(obj2[i], y_min, y_max, pb, pt)
        dom = is_dominated[i] if i < len(is_dominated) else True
        if dom:
            canvas.circle(px, py, 3, fill="#90A4AE", opacity=0.5)
        else:
            canvas.circle(px, py, 5, fill="#F44336", stroke="#B71C1C", stroke_width=1)

    n_pareto = sum(1 for d in is_dominated if not d)
    return PlotData(
        plot_type="pareto_front",
        data={"obj1": obj1, "obj2": obj2, "is_dominated": is_dominated},
        metadata={"n_pareto": n_pareto, "n_total": len(obj1)},
        svg=canvas.to_string(),
    )


# -- Campaign Report Data ---------------------------------------------------

@dataclass
class CampaignReportData:
    """All-optional data container for campaign report generation.

    Empty lists = skip that figure.
    """
    # Fig 1: Convergence
    iterations: list[int] = field(default_factory=list)
    best_values: list[float] = field(default_factory=list)
    # Fig 2: Calibration
    predicted: list[float] = field(default_factory=list)
    actual: list[float] = field(default_factory=list)
    # Fig 3: Drift
    drift_iterations: list[int] = field(default_factory=list)
    model_errors: list[float] = field(default_factory=list)
    # Fig 4: Batch comparison
    batch_labels: list[str] = field(default_factory=list)
    batch_means: list[float] = field(default_factory=list)
    batch_stds: list[float] = field(default_factory=list)
    # Fig 5: Recommendation coverage
    candidate_xy: list[tuple[float, float]] = field(default_factory=list)
    observed_xy: list[tuple[float, float]] = field(default_factory=list)
    # Fig 6: Uncertainty
    uncertainty_grid_x: list[float] = field(default_factory=list)
    uncertainty_grid_y: list[float] = field(default_factory=list)
    uncertainty_values: list[list[float]] = field(default_factory=list)
    # Fig 7: Feature importance
    feature_names: list[str] = field(default_factory=list)
    feature_importances: list[float] = field(default_factory=list)
    # Fig 8: Pareto front
    pareto_obj1: list[float] = field(default_factory=list)
    pareto_obj2: list[float] = field(default_factory=list)
    pareto_dominated: list[bool] = field(default_factory=list)


class CampaignReport:
    """Orchestrator that generates the standard 8-figure campaign report."""

    def generate(self, data: CampaignReportData) -> dict[str, PlotData]:
        """Generate individual figures, skipping those with no data."""
        figures: dict[str, PlotData] = {}

        if data.iterations and data.best_values:
            figures["convergence"] = plot_convergence_curve(data.iterations, data.best_values)
        if data.predicted and data.actual:
            figures["calibration"] = plot_calibration_curve(data.predicted, data.actual)
        if data.drift_iterations and data.model_errors:
            figures["drift"] = plot_drift_timeline(data.drift_iterations, data.model_errors)
        if data.batch_labels and data.batch_means:
            figures["batch"] = plot_batch_comparison(data.batch_labels, data.batch_means, data.batch_stds)
        if data.candidate_xy or data.observed_xy:
            figures["coverage"] = plot_recommendation_coverage(data.candidate_xy, data.observed_xy)
        if data.uncertainty_grid_x and data.uncertainty_values:
            figures["uncertainty"] = plot_uncertainty_coverage(
                data.uncertainty_grid_x, data.uncertainty_grid_y, data.uncertainty_values
            )
        if data.feature_names and data.feature_importances:
            figures["importance"] = plot_feature_importance(data.feature_names, data.feature_importances)
        if data.pareto_obj1 and data.pareto_obj2:
            figures["pareto"] = plot_pareto_front(data.pareto_obj1, data.pareto_obj2, data.pareto_dominated)

        return figures

    def generate_combined_svg(self, data: CampaignReportData) -> str:
        """Generate a 2x4 tiled grid SVG combining all figures."""
        figures = self.generate(data)

        cols, rows = 2, 4
        total_w = cols * _W
        total_h = rows * _H
        canvas = SVGCanvas(total_w, total_h, background="white")

        # Order of figures in the grid
        figure_keys = [
            "convergence", "calibration",
            "drift", "batch",
            "coverage", "uncertainty",
            "importance", "pareto",
        ]

        for idx, key in enumerate(figure_keys):
            col = idx % cols
            row = idx // cols
            x_off = col * _W
            y_off = row * _H

            if key in figures and figures[key].svg:
                # Embed the individual SVG content as a group with translation
                canvas.group_start(transform=f"translate({x_off},{y_off})")
                # Extract inner content from the SVG (skip outer svg tags)
                svg_str = figures[key].svg
                # Find content between first > and last </svg>
                start = svg_str.find(">") + 1
                end = svg_str.rfind("</svg>")
                if start > 0 and end > start:
                    inner = svg_str[start:end]
                    canvas.raw(inner)
                canvas.group_end()
            else:
                # Empty placeholder
                canvas.rect(x_off + 10, y_off + 10, _W - 20, _H - 20,
                           fill="#f5f5f5", stroke="#ddd", stroke_width=1, rx=4, ry=4)
                canvas.text(x_off + _W / 2, y_off + _H / 2, f"({key})",
                           font_size=12, text_anchor="middle", fill="#999")

        return canvas.to_string()

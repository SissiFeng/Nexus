"""Uncertainty flow visualizations -- budget, noise impact, reliability, heatmap.

All four chart functions accept uncertainty-related data and return a
:class:`PlotData` instance with pre-rendered SVG (via :class:`SVGCanvas`).
No external dependencies.
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.uncertainty.types import (
    MeasurementWithUncertainty,
    ObservationWithNoise,
    UncertaintyBudget,
)
from optimization_copilot.visualization.models import PlotData
from optimization_copilot.visualization.svg_renderer import SVGCanvas


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _confidence_to_color(confidence: float) -> str:
    """Map confidence in [0, 1] to a red-yellow-green colour string.

    0.0 -> red (#EF4444), 0.5 -> yellow (#EAB308), 1.0 -> green (#22C55E).
    """
    c = max(0.0, min(1.0, confidence))
    if c <= 0.5:
        # red to yellow
        t = c / 0.5
        r = int(239 + (234 - 239) * t)
        g = int(68 + (179 - 68) * t)
        b = int(68 + (8 - 68) * t)
    else:
        # yellow to green
        t = (c - 0.5) / 0.5
        r = int(234 + (34 - 234) * t)
        g = int(179 + (197 - 179) * t)
        b = int(8 + (94 - 8) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def _noise_to_color(noise_fraction: float) -> str:
    """Map normalised noise fraction in [0, 1] to green-to-red.

    0.0 (low noise) -> green (#22C55E), 1.0 (high noise) -> red (#EF4444).
    """
    t = max(0.0, min(1.0, noise_fraction))
    r = int(34 + (239 - 34) * t)
    g = int(197 + (68 - 197) * t)
    b = int(94 + (68 - 94) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------------------
# 1. Uncertainty Budget
# ---------------------------------------------------------------------------

def plot_uncertainty_budget(
    measurements: list[MeasurementWithUncertainty],
    observation: ObservationWithNoise | None = None,
) -> PlotData:
    """Horizontal bar chart of each KPI's variance contribution.

    Parameters
    ----------
    measurements : list[MeasurementWithUncertainty]
        Measurements with source labels and variances.
    observation : ObservationWithNoise | None
        If provided and has an uncertainty_budget, that budget is used directly.
        Otherwise the budget is computed from *measurements*.

    Returns
    -------
    PlotData
        ``plot_type="uncertainty_budget"`` with SVG.
    """
    # Determine the budget.
    budget: UncertaintyBudget | None = None
    if observation is not None and observation.uncertainty_budget is not None:
        budget = observation.uncertainty_budget
    else:
        contributions: dict[str, float] = {}
        for m in measurements:
            contributions[m.source] = contributions.get(m.source, 0.0) + m.variance
        budget = UncertaintyBudget.from_contributions(contributions)

    sources = list(budget.contributions.keys())
    n = len(sources)

    # --- SVG layout ---
    margin_left = 160
    margin_right = 80
    margin_top = 40
    margin_bottom = 30
    bar_height = 28
    row_gap = 8
    chart_width = 600
    chart_height = max(n * (bar_height + row_gap) + margin_top + margin_bottom, 120)

    canvas = SVGCanvas(chart_width, chart_height, background="white")

    # Title.
    canvas.text(
        chart_width / 2, 20,
        "Uncertainty Budget",
        font_size=14, text_anchor="middle", fill="#333",
    )

    if n == 0:
        canvas.text(
            chart_width / 2, chart_height / 2,
            "No sources to display",
            font_size=12, text_anchor="middle", fill="#999",
        )
        return PlotData(
            plot_type="uncertainty_budget",
            data={"sources": [], "fractions": [], "dominant_source": ""},
            metadata={"total_variance": 0.0},
            svg=canvas.to_string(),
        )

    plot_width = chart_width - margin_left - margin_right
    fractions = [budget.fraction(s) for s in sources]

    for i, source in enumerate(sources):
        frac = fractions[i]
        y_top = margin_top + i * (bar_height + row_gap)
        bar_w = max(frac * plot_width, 1)

        is_dominant = source == budget.dominant_source
        fill = "#EF4444" if is_dominant else "#3B82F6"

        canvas.rect(margin_left, y_top, bar_w, bar_height, fill=fill, rx=3, ry=3)

        # Source label.
        canvas.text(
            margin_left - 8, y_top + bar_height / 2 + 4,
            source[:20],
            font_size=11, text_anchor="end", fill="#333",
        )

        # Percentage label.
        pct = frac * 100
        canvas.text(
            margin_left + bar_w + 6, y_top + bar_height / 2 + 4,
            f"{pct:.1f}%",
            font_size=10, fill="#555",
        )

    # Legend note for dominant.
    canvas.text(
        chart_width / 2, chart_height - 10,
        f"Dominant: {budget.dominant_source} | Total var: {budget.total_variance:.4g}",
        font_size=10, text_anchor="middle", fill="#888",
    )

    return PlotData(
        plot_type="uncertainty_budget",
        data={
            "sources": sources,
            "fractions": fractions,
            "dominant_source": budget.dominant_source,
            "contributions": dict(budget.contributions),
        },
        metadata={
            "total_variance": budget.total_variance,
            "n_sources": n,
        },
        svg=canvas.to_string(),
    )


# ---------------------------------------------------------------------------
# 2. Noise Impact
# ---------------------------------------------------------------------------

def plot_noise_impact(
    observations: list[float] | list[ObservationWithNoise],
    noise_variances: list[float] | None = None,
) -> PlotData:
    """Scatter plot of observations with noise-weighted point sizes.

    Parameters
    ----------
    observations : list[float] | list[ObservationWithNoise]
        Objective values (scalars or ObservationWithNoise objects).
    noise_variances : list[float] | None
        Noise variances for each observation. Required if *observations*
        are plain floats; otherwise extracted from .noise_variance.

    Returns
    -------
    PlotData
        ``plot_type="noise_impact"`` with SVG.
    """
    # Normalise inputs.
    obj_values: list[float] = []
    variances: list[float] = []

    if observations and isinstance(observations[0], ObservationWithNoise):
        for obs in observations:
            obs_typed: ObservationWithNoise = obs  # type: ignore[assignment]
            obj_values.append(obs_typed.objective_value)
            variances.append(obs_typed.noise_variance)
    else:
        obj_values = list(observations)  # type: ignore[arg-type]
        variances = list(noise_variances) if noise_variances is not None else [0.0] * len(obj_values)

    n = len(obj_values)

    # --- SVG layout ---
    margin_left = 70
    margin_right = 40
    margin_top = 40
    margin_bottom = 50
    chart_width = 600
    chart_height = 400
    plot_width = chart_width - margin_left - margin_right
    plot_height = chart_height - margin_top - margin_bottom

    canvas = SVGCanvas(chart_width, chart_height, background="white")
    canvas.text(
        chart_width / 2, 20,
        "Noise Impact",
        font_size=14, text_anchor="middle", fill="#333",
    )

    if n == 0:
        canvas.text(
            chart_width / 2, chart_height / 2,
            "No observations to display",
            font_size=12, text_anchor="middle", fill="#999",
        )
        return PlotData(
            plot_type="noise_impact",
            data={"observations": [], "noise_variances": []},
            metadata={"n_observations": 0},
            svg=canvas.to_string(),
        )

    # Compute median noise variance for sizing.
    sorted_var = sorted(variances)
    if n % 2 == 0:
        median_var = (sorted_var[n // 2 - 1] + sorted_var[n // 2]) / 2.0
    else:
        median_var = sorted_var[n // 2]

    # Normalise noise for colouring.
    max_var = max(variances) if variances else 1.0
    if max_var < 1e-30:
        max_var = 1.0

    # Axis ranges.
    x_min_val = 0.5
    x_max_val = n + 0.5
    y_vals = obj_values
    y_min = min(y_vals)
    y_max = max(y_vals)
    y_pad = max((y_max - y_min) * 0.1, 0.01)
    y_min -= y_pad
    y_max += y_pad
    y_range = y_max - y_min if y_max != y_min else 1.0

    def x_pos(idx: int) -> float:
        return margin_left + (idx + 1 - x_min_val) / (x_max_val - x_min_val) * plot_width

    def y_pos(v: float) -> float:
        return margin_top + plot_height - (v - y_min) / y_range * plot_height

    # Axes.
    canvas.line(
        margin_left, margin_top + plot_height,
        margin_left + plot_width, margin_top + plot_height,
        stroke="#999",
    )
    canvas.line(
        margin_left, margin_top,
        margin_left, margin_top + plot_height,
        stroke="#999",
    )

    # Draw points with error bars.
    for i in range(n):
        cx = x_pos(i)
        cy = y_pos(obj_values[i])
        sigma = math.sqrt(variances[i]) if variances[i] > 0 else 0.0

        # Point size inversely proportional to noise.
        if median_var > 1e-30:
            size = 8.0 / (1.0 + variances[i] / median_var)
        else:
            size = 8.0
        size = max(size, 2.0)

        # Colour by noise level.
        noise_frac = variances[i] / max_var
        color = _noise_to_color(noise_frac)

        # Error bars: +/- 2 sigma.
        if sigma > 0:
            y_lo = y_pos(obj_values[i] - 2 * sigma)
            y_hi = y_pos(obj_values[i] + 2 * sigma)
            canvas.line(cx, y_lo, cx, y_hi, stroke=color, stroke_width=1, opacity=0.5)
            # Caps.
            cap_w = 3
            canvas.line(cx - cap_w, y_lo, cx + cap_w, y_lo, stroke=color, stroke_width=1, opacity=0.5)
            canvas.line(cx - cap_w, y_hi, cx + cap_w, y_hi, stroke=color, stroke_width=1, opacity=0.5)

        canvas.circle(cx, cy, size, fill=color, opacity=0.8)

    # Axis labels.
    canvas.text(
        chart_width / 2, chart_height - 8,
        "Observation index", font_size=11, text_anchor="middle", fill="#666",
    )
    canvas.text(
        12, chart_height / 2,
        "Objective value", font_size=11, text_anchor="middle", fill="#666",
        transform=f"rotate(-90 12 {chart_height / 2})",
    )

    # Legend.
    canvas.circle(chart_width - margin_right - 60, margin_top + 10, 5, fill="#22C55E")
    canvas.text(chart_width - margin_right - 50, margin_top + 14, "Low noise", font_size=8, fill="#666")
    canvas.circle(chart_width - margin_right - 60, margin_top + 26, 5, fill="#EF4444")
    canvas.text(chart_width - margin_right - 50, margin_top + 30, "High noise", font_size=8, fill="#666")

    return PlotData(
        plot_type="noise_impact",
        data={
            "observations": obj_values,
            "noise_variances": variances,
            "median_noise_variance": median_var,
        },
        metadata={
            "n_observations": n,
            "max_noise_variance": max_var,
        },
        svg=canvas.to_string(),
    )


# ---------------------------------------------------------------------------
# 3. Measurement Reliability Timeline
# ---------------------------------------------------------------------------

def plot_measurement_reliability_timeline(
    history: list[list[MeasurementWithUncertainty]],
) -> PlotData:
    """Line chart tracking measurement reliability over iterations.

    Parameters
    ----------
    history : list[list[MeasurementWithUncertainty]]
        One list of measurements per iteration.

    Returns
    -------
    PlotData
        ``plot_type="reliability_timeline"`` with SVG.
    """
    n_iterations = len(history)

    # --- SVG layout ---
    margin_left = 70
    margin_right = 140
    margin_top = 40
    margin_bottom = 50
    chart_width = 600
    chart_height = 400
    plot_width = chart_width - margin_left - margin_right
    plot_height = chart_height - margin_top - margin_bottom

    canvas = SVGCanvas(chart_width, chart_height, background="white")
    canvas.text(
        chart_width / 2, 20,
        "Measurement Reliability Timeline",
        font_size=14, text_anchor="middle", fill="#333",
    )

    if n_iterations == 0:
        canvas.text(
            chart_width / 2, chart_height / 2,
            "No history to display",
            font_size=12, text_anchor="middle", fill="#999",
        )
        return PlotData(
            plot_type="reliability_timeline",
            data={"sources": {}, "n_iterations": 0, "unreliable_points": []},
            metadata={},
            svg=canvas.to_string(),
        )

    # Collect all source names.
    all_sources: set[str] = set()
    for iteration_measurements in history:
        for m in iteration_measurements:
            all_sources.add(m.source)
    sources = sorted(all_sources)

    # Build per-source time series of relative_uncertainty.
    source_series: dict[str, list[float | None]] = {s: [] for s in sources}
    source_confidence: dict[str, list[float | None]] = {s: [] for s in sources}
    for iteration_measurements in history:
        found: dict[str, tuple[float, float]] = {}
        for m in iteration_measurements:
            found[m.source] = (m.relative_uncertainty, m.confidence)
        for s in sources:
            if s in found:
                source_series[s].append(found[s][0])
                source_confidence[s].append(found[s][1])
            else:
                source_series[s].append(None)
                source_confidence[s].append(None)

    # y-axis: relative uncertainty range.
    all_ru: list[float] = []
    for s in sources:
        for v in source_series[s]:
            if v is not None and math.isfinite(v):
                all_ru.append(v)

    if not all_ru:
        all_ru = [0.0, 1.0]

    y_min = 0.0
    y_max = max(max(all_ru) * 1.2, 0.6)
    y_range = y_max - y_min if y_max > y_min else 1.0

    def x_pos(iteration: int) -> float:
        if n_iterations <= 1:
            return margin_left + plot_width / 2
        return margin_left + iteration / (n_iterations - 1) * plot_width

    def y_pos(v: float) -> float:
        return margin_top + plot_height - (v - y_min) / y_range * plot_height

    # Axes.
    canvas.line(
        margin_left, margin_top + plot_height,
        margin_left + plot_width, margin_top + plot_height,
        stroke="#999",
    )
    canvas.line(
        margin_left, margin_top,
        margin_left, margin_top + plot_height,
        stroke="#999",
    )

    # Warning line at relative_uncertainty = 0.5.
    warning_y = y_pos(0.5)
    if margin_top <= warning_y <= margin_top + plot_height:
        canvas.line(
            margin_left, warning_y,
            margin_left + plot_width, warning_y,
            stroke="#EAB308", stroke_width=1, stroke_dasharray="6,3",
        )
        canvas.text(
            margin_left + plot_width + 4, warning_y + 4,
            "warning=0.5",
            font_size=8, fill="#EAB308",
        )

    # Colour palette for sources.
    palette = [
        "#3B82F6", "#EF4444", "#22C55E", "#A855F7",
        "#F97316", "#06B6D4", "#EC4899", "#84CC16",
    ]

    unreliable_points: list[dict[str, Any]] = []

    for si, source in enumerate(sources):
        color = palette[si % len(palette)]
        series = source_series[source]
        conf_series = source_confidence[source]

        # Draw line segments between non-None points.
        points: list[tuple[float, float]] = []
        for t in range(n_iterations):
            v = series[t]
            if v is not None and math.isfinite(v):
                points.append((x_pos(t), y_pos(v)))

        if len(points) >= 2:
            canvas.polyline(points, stroke=color, stroke_width=1.5, fill="none")

        # Draw points, marking unreliable ones in red.
        for t in range(n_iterations):
            v = series[t]
            c = conf_series[t]
            if v is not None and math.isfinite(v):
                cx = x_pos(t)
                cy = y_pos(v)
                is_unreliable = c is not None and c < 0.5
                pt_color = "#EF4444" if is_unreliable else color
                pt_r = 4 if is_unreliable else 3
                canvas.circle(cx, cy, pt_r, fill=pt_color, opacity=0.9)
                if is_unreliable:
                    unreliable_points.append({
                        "source": source,
                        "iteration": t,
                        "relative_uncertainty": v,
                        "confidence": c,
                    })

    # Legend.
    legend_x = margin_left + plot_width + 10
    for si, source in enumerate(sources):
        color = palette[si % len(palette)]
        ly = margin_top + si * 16
        canvas.line(legend_x, ly + 6, legend_x + 14, ly + 6, stroke=color, stroke_width=2)
        canvas.text(legend_x + 18, ly + 10, source[:15], font_size=9, fill="#555")

    # Axis labels.
    canvas.text(
        chart_width / 2, chart_height - 8,
        "Iteration", font_size=11, text_anchor="middle", fill="#666",
    )
    canvas.text(
        12, chart_height / 2,
        "Relative uncertainty", font_size=11, text_anchor="middle", fill="#666",
        transform=f"rotate(-90 12 {chart_height / 2})",
    )

    # Serialise source data.
    data_sources: dict[str, list[float | None]] = {}
    for s in sources:
        data_sources[s] = source_series[s]

    return PlotData(
        plot_type="reliability_timeline",
        data={
            "sources": data_sources,
            "n_iterations": n_iterations,
            "unreliable_points": unreliable_points,
        },
        metadata={
            "source_names": sources,
            "y_range": [y_min, y_max],
        },
        svg=canvas.to_string(),
    )


# ---------------------------------------------------------------------------
# 4. Confidence Heatmap
# ---------------------------------------------------------------------------

def plot_confidence_heatmap(
    measurements_grid: list[list[float]],
    param_names: list[str],
) -> PlotData:
    """Grid visualization of confidence across parameter space.

    Parameters
    ----------
    measurements_grid : list[list[float]]
        2D grid of confidence values (rows x cols).
    param_names : list[str]
        Two parameter names: [row_param, col_param].

    Returns
    -------
    PlotData
        ``plot_type="confidence_heatmap"`` with SVG.
    """
    n_rows = len(measurements_grid)
    n_cols = len(measurements_grid[0]) if n_rows > 0 else 0

    # --- SVG layout ---
    margin_left = 80
    margin_right = 40
    margin_top = 50
    margin_bottom = 60
    chart_width = 600
    chart_height = 400

    canvas = SVGCanvas(chart_width, chart_height, background="white")
    canvas.text(
        chart_width / 2, 22,
        "Confidence Heatmap",
        font_size=14, text_anchor="middle", fill="#333",
    )

    if n_rows == 0 or n_cols == 0:
        canvas.text(
            chart_width / 2, chart_height / 2,
            "No grid data to display",
            font_size=12, text_anchor="middle", fill="#999",
        )
        return PlotData(
            plot_type="confidence_heatmap",
            data={"grid": [], "param_names": param_names},
            metadata={"n_rows": 0, "n_cols": 0},
            svg=canvas.to_string(),
        )

    plot_width = chart_width - margin_left - margin_right
    plot_height = chart_height - margin_top - margin_bottom

    cell_w = plot_width / n_cols
    cell_h = plot_height / n_rows

    for r in range(n_rows):
        for c in range(n_cols):
            val = measurements_grid[r][c]
            clamped = max(0.0, min(1.0, val))
            color = _confidence_to_color(clamped)
            x = margin_left + c * cell_w
            y = margin_top + r * cell_h

            canvas.rect(x, y, cell_w, cell_h, fill=color, stroke="white", stroke_width=1)

            # Text label -- choose dark or light text based on brightness.
            text_fill = "#333" if clamped > 0.4 else "#FFF"
            canvas.text(
                x + cell_w / 2, y + cell_h / 2 + 4,
                f"{val:.2f}",
                font_size=min(10, cell_h * 0.4),
                text_anchor="middle", fill=text_fill,
            )

    # Axis labels from param_names.
    row_label = param_names[0] if len(param_names) > 0 else "rows"
    col_label = param_names[1] if len(param_names) > 1 else "cols"

    canvas.text(
        chart_width / 2, chart_height - 10,
        col_label,
        font_size=11, text_anchor="middle", fill="#666",
    )
    canvas.text(
        12, chart_height / 2,
        row_label,
        font_size=11, text_anchor="middle", fill="#666",
        transform=f"rotate(-90 12 {chart_height / 2})",
    )

    # Column indices.
    for c in range(n_cols):
        cx = margin_left + c * cell_w + cell_w / 2
        canvas.text(
            cx, margin_top + plot_height + 16,
            str(c), font_size=9, text_anchor="middle", fill="#888",
        )

    # Row indices.
    for r in range(n_rows):
        ry = margin_top + r * cell_h + cell_h / 2 + 4
        canvas.text(
            margin_left - 8, ry,
            str(r), font_size=9, text_anchor="end", fill="#888",
        )

    # Colour legend.
    legend_y = margin_top - 18
    for lv, lbl in [(0.0, "Low"), (0.5, "Med"), (1.0, "High")]:
        lx = chart_width - margin_right - 120 + lv * 100
        canvas.rect(lx, legend_y, 12, 12, fill=_confidence_to_color(lv))
        canvas.text(lx + 15, legend_y + 10, lbl, font_size=8, fill="#888")

    return PlotData(
        plot_type="confidence_heatmap",
        data={
            "grid": measurements_grid,
            "param_names": param_names,
        },
        metadata={
            "n_rows": n_rows,
            "n_cols": n_cols,
        },
        svg=canvas.to_string(),
    )

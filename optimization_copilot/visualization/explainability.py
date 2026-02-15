"""SHAP explainability visualizations -- waterfall, beeswarm, dependence, force.

All four chart functions accept pre-computed SHAP values and return a
:class:`PlotData` instance with pre-rendered SVG (via :class:`SVGCanvas`).
No external dependencies.
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.visualization.models import PlotData
from optimization_copilot.visualization.svg_renderer import SVGCanvas


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _value_to_color(t: float) -> str:
    """Map normalised value *t* in [0, 1] to a blue-to-red colour string.

    0.0 -> blue (#3B82F6), 0.5 -> gray (#9CA3AF), 1.0 -> red (#EF4444).
    """
    t = max(0.0, min(1.0, t))
    if t <= 0.5:
        # blue to gray
        s = t / 0.5
        r = int(59 + (156 - 59) * s)
        g = int(130 + (163 - 130) * s)
        b = int(246 + (175 - 246) * s)
    else:
        # gray to red
        s = (t - 0.5) / 0.5
        r = int(156 + (239 - 156) * s)
        g = int(163 + (68 - 163) * s)
        b = int(175 + (68 - 175) * s)
    return f"#{r:02X}{g:02X}{b:02X}"


def _normalise_values(values: list[float]) -> list[float]:
    """Normalise to [0, 1] range.  Constant lists map to 0.5."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _pearson_correlation(a: list[float], b: list[float]) -> float:
    """Absolute Pearson correlation between two equal-length lists."""
    n = len(a)
    if n < 2:
        return 0.0
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    var_a = sum((a[i] - mean_a) ** 2 for i in range(n))
    var_b = sum((b[i] - mean_b) ** 2 for i in range(n))
    denom = math.sqrt(var_a * var_b)
    if denom < 1e-12:
        return 0.0
    return abs(cov / denom)


# ---------------------------------------------------------------------------
# 1. SHAP Waterfall
# ---------------------------------------------------------------------------

def plot_shap_waterfall(
    trial_index: int,
    shap_values: list[float],
    feature_names: list[str],
    base_value: float,
) -> PlotData:
    """Waterfall chart showing how each feature pushes the prediction.

    Features are sorted by absolute SHAP value (largest at top).
    Cumulative bars start from *base_value* and end at the final prediction.

    Parameters
    ----------
    trial_index : int
        Index of the trial being explained (for labelling).
    shap_values : list[float]
        One SHAP value per feature.
    feature_names : list[str]
        Human-readable feature names (same order as *shap_values*).
    base_value : float
        Expected model output E[f(X)].

    Returns
    -------
    PlotData
        ``plot_type="shap_waterfall"`` with SVG.
    """
    d = len(shap_values)
    if d == 0:
        canvas = SVGCanvas(800, 100, background="white")
        canvas.text(400, 50, "No features to display", text_anchor="middle")
        return PlotData(
            plot_type="shap_waterfall",
            data={"sorted_features": [], "sorted_shap": [], "cumulative": []},
            metadata={"trial_index": trial_index, "base_value": base_value},
            svg=canvas.to_string(),
        )

    # Sort by |SHAP| descending.
    indexed = sorted(
        enumerate(shap_values), key=lambda p: abs(p[1]), reverse=True
    )
    sorted_names = [feature_names[i] for i, _ in indexed]
    sorted_shap = [v for _, v in indexed]

    # Cumulative values starting from base.
    cumulative = [base_value]
    for v in sorted_shap:
        cumulative.append(cumulative[-1] + v)
    final_value = cumulative[-1]

    # --- SVG rendering ---
    margin_left = 150
    margin_right = 80
    margin_top = 40
    margin_bottom = 40
    bar_height = 24
    row_gap = 6
    chart_height = d * (bar_height + row_gap) + margin_top + margin_bottom + 30
    chart_width = 800

    canvas = SVGCanvas(chart_width, chart_height, background="white")

    # Title
    canvas.text(
        chart_width / 2, 20,
        f"SHAP Waterfall - Trial {trial_index}",
        font_size=14, text_anchor="middle", fill="#333",
    )

    # Determine x-axis range.
    all_vals = cumulative
    x_min = min(all_vals) - abs(max(sorted_shap, key=abs)) * 0.1
    x_max = max(all_vals) + abs(max(sorted_shap, key=abs)) * 0.1
    x_range = x_max - x_min if x_max != x_min else 1.0
    plot_width = chart_width - margin_left - margin_right

    def x_pos(val: float) -> float:
        return margin_left + (val - x_min) / x_range * plot_width

    # Base value line.
    bx = x_pos(base_value)
    canvas.line(
        bx, margin_top, bx, chart_height - margin_bottom,
        stroke="#999", stroke_width=1, stroke_dasharray="4,2",
    )

    # Draw bars.
    for i in range(d):
        y_top = margin_top + i * (bar_height + row_gap)
        start_val = cumulative[i]
        end_val = cumulative[i + 1]

        x1 = x_pos(min(start_val, end_val))
        x2 = x_pos(max(start_val, end_val))
        bar_w = max(x2 - x1, 1)

        color = "#EF4444" if sorted_shap[i] >= 0 else "#3B82F6"

        canvas.rect(x1, y_top, bar_w, bar_height, fill=color, rx=2, ry=2)

        # Feature name label.
        canvas.text(
            margin_left - 8, y_top + bar_height / 2 + 4,
            sorted_names[i][:18],
            font_size=11, text_anchor="end", fill="#333",
        )

        # Value label.
        label = f"{sorted_shap[i]:+.3f}"
        canvas.text(
            x2 + 4, y_top + bar_height / 2 + 4,
            label, font_size=10, fill="#555",
        )

    # Final prediction label.
    canvas.text(
        chart_width / 2, chart_height - 12,
        f"f(x) = {final_value:.4f}  (base = {base_value:.4f})",
        font_size=11, text_anchor="middle", fill="#666",
    )

    return PlotData(
        plot_type="shap_waterfall",
        data={
            "sorted_features": sorted_names,
            "sorted_shap": sorted_shap,
            "cumulative": cumulative,
        },
        metadata={
            "trial_index": trial_index,
            "base_value": base_value,
            "final_value": final_value,
            "n_features": d,
        },
        svg=canvas.to_string(),
    )


# ---------------------------------------------------------------------------
# 2. SHAP Beeswarm
# ---------------------------------------------------------------------------

def plot_shap_beeswarm(
    shap_matrix: list[list[float]],
    feature_values: list[list[float]],
    feature_names: list[str],
) -> PlotData:
    """Beeswarm plot showing SHAP value distributions for all features.

    Features are ranked by mean absolute SHAP value.  Points are coloured
    by the raw feature value (blue = low, red = high).

    Parameters
    ----------
    shap_matrix : list[list[float]]
        Shape (n_trials, d) -- SHAP values per trial per feature.
    feature_values : list[list[float]]
        Shape (n_trials, d) -- raw feature values per trial per feature.
    feature_names : list[str]
        Length d.

    Returns
    -------
    PlotData
    """
    if not shap_matrix or not shap_matrix[0]:
        canvas = SVGCanvas(800, 100, background="white")
        canvas.text(400, 50, "No data to display", text_anchor="middle")
        return PlotData(
            plot_type="shap_beeswarm",
            data={"feature_order": [], "mean_abs_shap": []},
            metadata={},
            svg=canvas.to_string(),
        )

    n = len(shap_matrix)
    d = len(shap_matrix[0])

    # Mean |SHAP| per feature.
    mean_abs = [
        sum(abs(shap_matrix[t][j]) for t in range(n)) / n for j in range(d)
    ]
    # Sort features by importance descending.
    order = sorted(range(d), key=lambda j: mean_abs[j], reverse=True)
    ordered_names = [feature_names[j] for j in order]
    ordered_mean = [mean_abs[j] for j in order]

    # --- SVG ---
    margin_left = 150
    margin_right = 40
    margin_top = 40
    margin_bottom = 50
    row_height = 30
    chart_height = max(d * row_height + margin_top + margin_bottom, 150)
    chart_width = 800
    plot_width = chart_width - margin_left - margin_right

    canvas = SVGCanvas(chart_width, chart_height, background="white")
    canvas.text(
        chart_width / 2, 18,
        "SHAP Beeswarm", font_size=14, text_anchor="middle", fill="#333",
    )

    # x-axis: SHAP value range.
    all_shap = [shap_matrix[t][j] for t in range(n) for j in range(d)]
    if all_shap:
        x_min = min(all_shap)
        x_max = max(all_shap)
    else:
        x_min, x_max = -1.0, 1.0
    x_pad = max((x_max - x_min) * 0.1, 0.01)
    x_min -= x_pad
    x_max += x_pad
    x_range = x_max - x_min

    def x_pos(v: float) -> float:
        return margin_left + (v - x_min) / x_range * plot_width

    # Vertical center line at SHAP = 0.
    x_zero = x_pos(0.0)
    canvas.line(
        x_zero, margin_top, x_zero, chart_height - margin_bottom,
        stroke="#DDD", stroke_width=1,
    )

    # Normalise feature values per feature for colouring.
    for rank, j in enumerate(order):
        y_center = margin_top + rank * row_height + row_height / 2
        fv_col = [feature_values[t][j] for t in range(n)]
        normed = _normalise_values(fv_col)

        # Feature label.
        canvas.text(
            margin_left - 8, y_center + 4,
            ordered_names[rank][:18],
            font_size=11, text_anchor="end", fill="#333",
        )

        # Jitter deterministically based on trial index.
        for t in range(n):
            sv = shap_matrix[t][j]
            cx = x_pos(sv)
            jitter = ((t * 7 + j * 3) % 17 - 8) / 8.0 * (row_height * 0.35)
            cy = y_center + jitter
            color = _value_to_color(normed[t])
            canvas.circle(cx, cy, 3, fill=color, opacity=0.7)

    # x-axis label.
    canvas.text(
        chart_width / 2, chart_height - 10,
        "SHAP value", font_size=11, text_anchor="middle", fill="#666",
    )

    # Colour legend.
    legend_x = chart_width - margin_right - 80
    canvas.text(legend_x, margin_top - 5, "Feature value", font_size=9, fill="#888")
    canvas.text(legend_x, margin_top + 8, "Low", font_size=8, fill="#3B82F6")
    canvas.text(legend_x + 60, margin_top + 8, "High", font_size=8, fill="#EF4444")

    return PlotData(
        plot_type="shap_beeswarm",
        data={
            "feature_order": ordered_names,
            "mean_abs_shap": ordered_mean,
            "n_trials": n,
            "n_features": d,
        },
        metadata={"x_range": [x_min, x_max]},
        svg=canvas.to_string(),
    )


# ---------------------------------------------------------------------------
# 3. SHAP Dependence
# ---------------------------------------------------------------------------

def plot_shap_dependence(
    feature_idx: int,
    shap_values: list[float],
    feature_values: list[float],
    interaction_feature_values: list[float] | None = None,
    interaction_name: str | None = None,
) -> PlotData:
    """Scatter plot of feature value vs. its SHAP contribution.

    Optionally colours points by an interaction feature.  If no interaction
    feature is specified explicitly, the function auto-selects one.

    Parameters
    ----------
    feature_idx : int
        Index of the feature being plotted.
    shap_values : list[float]
        SHAP values for this feature across all trials.
    feature_values : list[float]
        Raw values for this feature across all trials.
    interaction_feature_values : list[float] | None
        Raw values of an interaction feature for colouring (same length).
    interaction_name : str | None
        Label for the interaction feature.

    Returns
    -------
    PlotData
    """
    n = len(shap_values)

    if n == 0:
        canvas = SVGCanvas(800, 400, background="white")
        canvas.text(400, 200, "No data to display", text_anchor="middle")
        return PlotData(
            plot_type="shap_dependence",
            data={"feature_idx": feature_idx, "points": []},
            metadata={},
            svg=canvas.to_string(),
        )

    # --- SVG ---
    margin_left = 70
    margin_right = 60
    margin_top = 40
    margin_bottom = 50
    chart_width = 800
    chart_height = 500
    plot_width = chart_width - margin_left - margin_right
    plot_height = chart_height - margin_top - margin_bottom

    canvas = SVGCanvas(chart_width, chart_height, background="white")
    canvas.text(
        chart_width / 2, 20,
        f"SHAP Dependence - Feature {feature_idx}",
        font_size=14, text_anchor="middle", fill="#333",
    )

    # Axis ranges.
    fv_min = min(feature_values)
    fv_max = max(feature_values)
    sv_min = min(shap_values)
    sv_max = max(shap_values)
    fv_pad = max((fv_max - fv_min) * 0.05, 1e-6)
    sv_pad = max((sv_max - sv_min) * 0.05, 1e-6)
    fv_min -= fv_pad
    fv_max += fv_pad
    sv_min -= sv_pad
    sv_max += sv_pad
    fv_range = fv_max - fv_min
    sv_range = sv_max - sv_min

    def x_pos(v: float) -> float:
        return margin_left + (v - fv_min) / fv_range * plot_width

    def y_pos(v: float) -> float:
        return margin_top + plot_height - (v - sv_min) / sv_range * plot_height

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

    # Zero line for SHAP.
    if sv_min <= 0 <= sv_max:
        canvas.line(
            margin_left, y_pos(0), margin_left + plot_width, y_pos(0),
            stroke="#DDD", stroke_dasharray="4,2",
        )

    # Colour by interaction feature.
    has_interaction = interaction_feature_values is not None and len(
        interaction_feature_values
    ) == n
    if has_interaction:
        normed_interaction = _normalise_values(interaction_feature_values)  # type: ignore[arg-type]
    else:
        normed_interaction = [0.5] * n

    points_data: list[dict[str, float]] = []
    for i in range(n):
        cx = x_pos(feature_values[i])
        cy = y_pos(shap_values[i])
        color = _value_to_color(normed_interaction[i])
        canvas.circle(cx, cy, 3.5, fill=color, opacity=0.65)
        points_data.append({
            "feature_value": feature_values[i],
            "shap_value": shap_values[i],
        })

    # Axis labels.
    canvas.text(
        chart_width / 2, chart_height - 8,
        "Feature value", font_size=11, text_anchor="middle", fill="#666",
    )
    canvas.text(
        12, chart_height / 2,
        "SHAP value", font_size=11, text_anchor="middle", fill="#666",
        transform=f"rotate(-90 12 {chart_height / 2})",
    )

    # Interaction legend.
    if has_interaction and interaction_name:
        canvas.text(
            chart_width - margin_right + 5, margin_top + 10,
            interaction_name, font_size=9, fill="#888",
        )

    return PlotData(
        plot_type="shap_dependence",
        data={
            "feature_idx": feature_idx,
            "points": points_data,
            "n_points": n,
            "has_interaction": has_interaction,
            "interaction_name": interaction_name,
        },
        metadata={
            "x_range": [fv_min, fv_max],
            "y_range": [sv_min, sv_max],
        },
        svg=canvas.to_string(),
    )


def auto_select_interaction(
    feature_idx: int,
    shap_matrix: list[list[float]],
    feature_values_matrix: list[list[float]],
) -> int:
    """Select the best interaction feature via max correlation with SHAP residuals.

    Parameters
    ----------
    feature_idx : int
        The primary feature index.
    shap_matrix : list[list[float]]
        Shape (n_trials, d).
    feature_values_matrix : list[list[float]]
        Shape (n_trials, d).

    Returns
    -------
    int
        Index of the feature with highest absolute correlation to the
        SHAP values of *feature_idx*.  Returns 0 if no better option found.
    """
    n = len(shap_matrix)
    d = len(shap_matrix[0]) if n > 0 else 0
    if d <= 1:
        return 0

    shap_col = [shap_matrix[t][feature_idx] for t in range(n)]

    best_idx = 0
    best_corr = -1.0
    for j in range(d):
        if j == feature_idx:
            continue
        fv_col = [feature_values_matrix[t][j] for t in range(n)]
        corr = _pearson_correlation(shap_col, fv_col)
        if corr > best_corr:
            best_corr = corr
            best_idx = j

    return best_idx


# ---------------------------------------------------------------------------
# 4. SHAP Force Plot
# ---------------------------------------------------------------------------

def plot_shap_force(
    trial_index: int,
    shap_values: list[float],
    feature_names: list[str],
    feature_values: list[float],
    base_value: float,
) -> PlotData:
    """Horizontal force plot showing positive (right) and negative (left) contributions.

    Parameters
    ----------
    trial_index : int
        Trial being explained.
    shap_values : list[float]
        One SHAP value per feature.
    feature_names : list[str]
        Feature labels.
    feature_values : list[float]
        Raw feature values (for annotation).
    base_value : float
        E[f(X)].

    Returns
    -------
    PlotData
    """
    d = len(shap_values)
    final_value = base_value + sum(shap_values)

    # Separate positive and negative.
    positive: list[tuple[int, float]] = []
    negative: list[tuple[int, float]] = []
    for i, sv in enumerate(shap_values):
        if sv >= 0:
            positive.append((i, sv))
        else:
            negative.append((i, sv))

    # Sort: positive by value descending, negative by |value| descending.
    positive.sort(key=lambda p: p[1], reverse=True)
    negative.sort(key=lambda p: abs(p[1]), reverse=True)

    if d == 0:
        canvas = SVGCanvas(800, 120, background="white")
        canvas.text(400, 60, "No features to display", text_anchor="middle")
        return PlotData(
            plot_type="shap_force",
            data={
                "positive": [],
                "negative": [],
                "base_value": base_value,
                "final_value": final_value,
            },
            metadata={"trial_index": trial_index},
            svg=canvas.to_string(),
        )

    # --- SVG ---
    chart_width = 800
    chart_height = 180
    margin_left = 60
    margin_right = 60
    bar_y = 70
    bar_height = 40
    plot_width = chart_width - margin_left - margin_right

    canvas = SVGCanvas(chart_width, chart_height, background="white")
    canvas.text(
        chart_width / 2, 20,
        f"SHAP Force Plot - Trial {trial_index}",
        font_size=14, text_anchor="middle", fill="#333",
    )

    # Total positive / negative contributions.
    total_pos = sum(v for _, v in positive)
    total_neg = sum(abs(v) for _, v in negative)
    total_range = total_pos + total_neg
    if total_range < 1e-12:
        total_range = 1.0

    # The bar spans from base_value to final_value conceptually.
    # We render positive segments on the right, negative on the left.
    center_x = margin_left + plot_width / 2

    # Scale: map total contributions to pixel width.
    scale = plot_width / total_range if total_range > 0 else 1.0

    # Draw positive segments (right of center, or stacked from base).
    # Layout: center_x is the dividing point.  Positive grows right,
    # negative grows left.
    x_cursor = center_x

    pos_data: list[dict[str, Any]] = []
    for idx, sv in positive:
        seg_w = sv * scale
        if seg_w < 0.5:
            seg_w = 0.5
        canvas.rect(x_cursor, bar_y, seg_w, bar_height, fill="#EF4444", opacity=0.8)
        # Label if segment is wide enough.
        if seg_w > 30:
            label = feature_names[idx][:10]
            canvas.text(
                x_cursor + seg_w / 2, bar_y + bar_height / 2 + 4,
                label, font_size=8, text_anchor="middle", fill="white",
            )
        pos_data.append({
            "feature": feature_names[idx],
            "value": feature_values[idx],
            "shap": sv,
        })
        x_cursor += seg_w

    # Draw negative segments (left of center).
    x_cursor = center_x
    neg_data: list[dict[str, Any]] = []
    for idx, sv in negative:
        seg_w = abs(sv) * scale
        if seg_w < 0.5:
            seg_w = 0.5
        x_cursor -= seg_w
        canvas.rect(x_cursor, bar_y, seg_w, bar_height, fill="#3B82F6", opacity=0.8)
        if seg_w > 30:
            label = feature_names[idx][:10]
            canvas.text(
                x_cursor + seg_w / 2, bar_y + bar_height / 2 + 4,
                label, font_size=8, text_anchor="middle", fill="white",
            )
        neg_data.append({
            "feature": feature_names[idx],
            "value": feature_values[idx],
            "shap": sv,
        })

    # Base and final value annotations.
    canvas.text(
        margin_left, bar_y + bar_height + 20,
        f"Base: {base_value:.4f}",
        font_size=10, fill="#888",
    )
    canvas.text(
        chart_width - margin_right, bar_y + bar_height + 20,
        f"f(x) = {final_value:.4f}",
        font_size=10, text_anchor="end", fill="#333",
    )

    # Legend.
    canvas.rect(margin_left, chart_height - 25, 12, 12, fill="#EF4444")
    canvas.text(margin_left + 16, chart_height - 15, "Positive", font_size=9, fill="#666")
    canvas.rect(margin_left + 80, chart_height - 25, 12, 12, fill="#3B82F6")
    canvas.text(margin_left + 96, chart_height - 15, "Negative", font_size=9, fill="#666")

    return PlotData(
        plot_type="shap_force",
        data={
            "positive": pos_data,
            "negative": neg_data,
            "base_value": base_value,
            "final_value": final_value,
        },
        metadata={
            "trial_index": trial_index,
            "n_features": d,
            "n_positive": len(positive),
            "n_negative": len(negative),
        },
        svg=canvas.to_string(),
    )

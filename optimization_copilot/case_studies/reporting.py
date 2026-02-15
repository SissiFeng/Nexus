"""SVG chart generation for case study reports.

Uses :class:`~optimization_copilot.visualization.svg_renderer.SVGCanvas`
to produce standalone SVG strings.  All functions return SVG markup as a
plain string -- no external dependencies required.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from optimization_copilot.visualization.svg_renderer import SVGCanvas

# ---------------------------------------------------------------------------
# Default colour palette
# ---------------------------------------------------------------------------

_DEFAULT_COLORS = [
    "#4e79a7",  # blue
    "#f28e2b",  # orange
    "#e15759",  # red
    "#76b7b2",  # teal
    "#59a14f",  # green
    "#edc948",  # yellow
    "#b07aa1",  # purple
    "#ff9da7",  # pink
    "#9c755f",  # brown
    "#bab0ac",  # grey
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _quantile(values: List[float], q: float) -> float:
    """Linear interpolation quantile (like numpy ``method='linear'``)."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n == 1:
        return s[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[lo] + frac * (s[hi] - s[lo])


def _nice_axis_range(lo: float, hi: float, n_ticks: int = 5):
    """Return (nice_lo, nice_hi, step) that bracket [lo, hi]."""
    if hi == lo:
        if lo == 0:
            return -1.0, 1.0, 0.5
        delta = abs(lo) * 0.1
        return lo - delta, hi + delta, delta / 2.0

    raw_step = (hi - lo) / max(n_ticks - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1.0
    residual = raw_step / magnitude
    if residual <= 1.0:
        nice_step = magnitude
    elif residual <= 2.0:
        nice_step = 2.0 * magnitude
    elif residual <= 5.0:
        nice_step = 5.0 * magnitude
    else:
        nice_step = 10.0 * magnitude

    nice_lo = math.floor(lo / nice_step) * nice_step
    nice_hi = math.ceil(hi / nice_step) * nice_step
    if nice_lo == nice_hi:
        nice_hi += nice_step
    return nice_lo, nice_hi, nice_step


# ---------------------------------------------------------------------------
# 1. Convergence curves
# ---------------------------------------------------------------------------


def plot_convergence_curves(
    curves: Dict[str, List[List[float]]],
    title: str = "Convergence",
    width: int = 800,
    height: int = 500,
    colors: Optional[List[str]] = None,
) -> str:
    """Plot convergence curves with mean +/- shaded std.

    Parameters
    ----------
    curves : dict[str, list[list[float]]]
        ``{strategy_name: [curve_repeat_1, curve_repeat_2, ...]}``.
        Each curve is a list of best-so-far values indexed by iteration.
    title : str
        Chart title.
    width, height : int
        SVG dimensions in pixels.
    colors : list[str] | None
        Colour per strategy.  Falls back to the built-in palette.

    Returns
    -------
    str
        SVG markup.
    """
    if colors is None:
        colors = _DEFAULT_COLORS

    canvas = SVGCanvas(width=width, height=height, background="white")

    # Margins
    ml, mr, mt, mb = 70, 30, 50, 60

    plot_w = width - ml - mr
    plot_h = height - mt - mb

    # Title
    canvas.text(
        width / 2, mt / 2 + 4, title,
        font_size=16, text_anchor="middle", fill="#333",
    )

    # Handle empty input gracefully
    if not curves:
        canvas.text(
            width / 2, height / 2, "(no data)",
            font_size=14, text_anchor="middle", fill="#999",
        )
        return canvas.to_string()

    # Compute per-strategy mean and std curves
    strategy_stats: Dict[str, dict] = {}
    max_len = 0
    all_vals: List[float] = []

    for name, repeats in curves.items():
        if not repeats:
            continue
        n_iters = max(len(r) for r in repeats)
        max_len = max(max_len, n_iters)
        means = []
        stds = []
        for i in range(n_iters):
            col = [r[i] for r in repeats if i < len(r)]
            m = _mean(col)
            s = _std(col) if len(col) >= 2 else 0.0
            means.append(m)
            stds.append(s)
            all_vals.extend(col)
        strategy_stats[name] = {"mean": means, "std": stds, "n": n_iters}

    if not all_vals or max_len == 0:
        canvas.text(
            width / 2, height / 2, "(no data)",
            font_size=14, text_anchor="middle", fill="#999",
        )
        return canvas.to_string()

    # Axis ranges
    y_lo = min(all_vals)
    y_hi = max(all_vals)
    # Expand for std bands
    for info in strategy_stats.values():
        for m, s in zip(info["mean"], info["std"]):
            y_lo = min(y_lo, m - s)
            y_hi = max(y_hi, m + s)
    nice_y_lo, nice_y_hi, y_step = _nice_axis_range(y_lo, y_hi)

    x_lo, x_hi = 0.0, float(max_len - 1) if max_len > 1 else 1.0

    def tx(v: float) -> float:
        return ml + (v - x_lo) / (x_hi - x_lo) * plot_w if x_hi != x_lo else ml + plot_w / 2

    def ty(v: float) -> float:
        return mt + (1.0 - (v - nice_y_lo) / (nice_y_hi - nice_y_lo)) * plot_h if nice_y_hi != nice_y_lo else mt + plot_h / 2

    # Grid and axes
    canvas.rect(ml, mt, plot_w, plot_h, fill="#fafafa", stroke="#ccc")

    # Y ticks
    y_val = nice_y_lo
    while y_val <= nice_y_hi + y_step * 0.01:
        yy = ty(y_val)
        if mt <= yy <= mt + plot_h:
            canvas.line(ml, yy, ml + plot_w, yy, stroke="#eee", stroke_width=1)
            label = f"{y_val:.4g}"
            canvas.text(ml - 5, yy + 4, label, font_size=10, text_anchor="end", fill="#666")
        y_val += y_step

    # X ticks
    n_xticks = min(max_len, 6)
    for i in range(n_xticks):
        xv = x_lo + i * (x_hi - x_lo) / max(n_xticks - 1, 1)
        xx = tx(xv)
        canvas.line(xx, mt + plot_h, xx, mt + plot_h + 5, stroke="#666")
        canvas.text(xx, mt + plot_h + 18, str(int(round(xv))),
                    font_size=10, text_anchor="middle", fill="#666")

    # Axis labels
    canvas.text(width / 2, height - 8, "Iteration",
                font_size=12, text_anchor="middle", fill="#333")
    canvas.text(
        14, height / 2, "Best value",
        font_size=12, text_anchor="middle", fill="#333",
        transform=f"rotate(-90,14,{height / 2})",
    )

    # Draw curves
    for idx, (name, info) in enumerate(strategy_stats.items()):
        color = colors[idx % len(colors)]
        means = info["mean"]
        stds = info["std"]
        n = info["n"]

        # Shaded std region (polygon)
        upper_pts = []
        lower_pts = []
        for i in range(n):
            xi = tx(float(i))
            upper_pts.append((xi, ty(means[i] + stds[i])))
            lower_pts.append((xi, ty(means[i] - stds[i])))
        # polygon: upper forward, lower backward
        shade_pts = upper_pts + list(reversed(lower_pts))
        if shade_pts:
            canvas.polygon(shade_pts, fill=color, opacity=0.15)

        # Mean line
        line_pts = [(tx(float(i)), ty(means[i])) for i in range(n)]
        if line_pts:
            canvas.polyline(line_pts, stroke=color, stroke_width=2)

    # Legend
    lx = ml + 10
    ly = mt + 15
    for idx, name in enumerate(strategy_stats):
        color = colors[idx % len(colors)]
        canvas.rect(lx, ly + idx * 20, 14, 10, fill=color)
        canvas.text(lx + 18, ly + idx * 20 + 9, name, font_size=11, fill="#333")

    return canvas.to_string()


# ---------------------------------------------------------------------------
# 2. Box comparison
# ---------------------------------------------------------------------------


def plot_box_comparison(
    data: Dict[str, List[float]],
    title: str = "Final Performance",
    width: int = 700,
    height: int = 400,
) -> str:
    """Box plots comparing final performance of strategies.

    For each strategy: box (Q1-Q3), whiskers (1.5 IQR), median line,
    outlier dots.

    Returns SVG string.
    """
    canvas = SVGCanvas(width=width, height=height, background="white")
    ml, mr, mt, mb = 70, 30, 50, 60
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    canvas.text(width / 2, mt / 2 + 4, title,
                font_size=16, text_anchor="middle", fill="#333")

    names = list(data.keys())
    if not names:
        canvas.text(width / 2, height / 2, "(no data)",
                    font_size=14, text_anchor="middle", fill="#999")
        return canvas.to_string()

    # Compute stats per group
    group_stats = []
    all_vals: List[float] = []
    for name in names:
        vals = sorted(data[name])
        if not vals:
            group_stats.append(None)
            continue
        q1 = _quantile(vals, 0.25)
        q3 = _quantile(vals, 0.75)
        med = _median(vals)
        iqr = q3 - q1
        lo_w = max(vals[0], q1 - 1.5 * iqr)
        hi_w = min(vals[-1], q3 + 1.5 * iqr)
        # Find the actual closest data values within whisker range
        lo_w = min(v for v in vals if v >= q1 - 1.5 * iqr)
        hi_w = max(v for v in vals if v <= q3 + 1.5 * iqr)
        outliers = [v for v in vals if v < lo_w or v > hi_w]
        group_stats.append({
            "q1": q1, "q3": q3, "med": med,
            "lo_w": lo_w, "hi_w": hi_w, "outliers": outliers,
        })
        all_vals.extend(vals)

    if not all_vals:
        canvas.text(width / 2, height / 2, "(no data)",
                    font_size=14, text_anchor="middle", fill="#999")
        return canvas.to_string()

    y_lo, y_hi = min(all_vals), max(all_vals)
    nice_y_lo, nice_y_hi, y_step = _nice_axis_range(y_lo, y_hi)

    def ty(v: float) -> float:
        r = (v - nice_y_lo) / (nice_y_hi - nice_y_lo) if nice_y_hi != nice_y_lo else 0.5
        return mt + (1.0 - r) * plot_h

    # Background + Y grid
    canvas.rect(ml, mt, plot_w, plot_h, fill="#fafafa", stroke="#ccc")
    y_val = nice_y_lo
    while y_val <= nice_y_hi + y_step * 0.01:
        yy = ty(y_val)
        if mt <= yy <= mt + plot_h:
            canvas.line(ml, yy, ml + plot_w, yy, stroke="#eee")
            canvas.text(ml - 5, yy + 4, f"{y_val:.4g}",
                        font_size=10, text_anchor="end", fill="#666")
        y_val += y_step

    n = len(names)
    slot_w = plot_w / n
    box_w = max(slot_w * 0.5, 8)

    for i, (name, gs) in enumerate(zip(names, group_stats)):
        cx = ml + slot_w * (i + 0.5)
        # Label
        canvas.text(cx, mt + plot_h + 18, name,
                    font_size=10, text_anchor="middle", fill="#333")
        if gs is None:
            continue
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]

        bx = cx - box_w / 2
        by_q3 = ty(gs["q3"])
        by_q1 = ty(gs["q1"])
        box_h = by_q1 - by_q3

        # Box
        canvas.rect(bx, by_q3, box_w, max(box_h, 0.5),
                     fill=color, stroke="#333", opacity=0.4)

        # Median line
        my = ty(gs["med"])
        canvas.line(bx, my, bx + box_w, my, stroke="#333", stroke_width=2)

        # Whiskers
        canvas.line(cx, by_q3, cx, ty(gs["hi_w"]), stroke="#333")
        canvas.line(cx - box_w * 0.3, ty(gs["hi_w"]),
                    cx + box_w * 0.3, ty(gs["hi_w"]), stroke="#333")
        canvas.line(cx, by_q1, cx, ty(gs["lo_w"]), stroke="#333")
        canvas.line(cx - box_w * 0.3, ty(gs["lo_w"]),
                    cx + box_w * 0.3, ty(gs["lo_w"]), stroke="#333")

        # Outliers
        for v in gs["outliers"]:
            canvas.circle(cx, ty(v), 3, fill=color, stroke="#333")

    return canvas.to_string()


# ---------------------------------------------------------------------------
# 3. Significance heatmap
# ---------------------------------------------------------------------------


def plot_significance_heatmap(
    comparison: Dict[str, Dict[str, Dict[str, float]]],
    title: str = "Statistical Significance",
    width: int = 600,
    height: int = 600,
) -> str:
    """Heatmap of p-values from pairwise comparisons.

    Colour coding: green (<0.01), yellow (<0.05), orange (<0.1),
    red (>=0.1).  Shows p-value text in each cell.

    Returns SVG string.
    """
    canvas = SVGCanvas(width=width, height=height, background="white")

    canvas.text(width / 2, 24, title,
                font_size=16, text_anchor="middle", fill="#333")

    names = sorted(comparison.keys())
    n = len(names)
    if n == 0:
        canvas.text(width / 2, height / 2, "(no data)",
                    font_size=14, text_anchor="middle", fill="#999")
        return canvas.to_string()

    ml, mt_map = 100, 60
    cell_size = min((width - ml - 20) / n, (height - mt_map - 40) / n, 80)
    grid_w = cell_size * n
    grid_h = cell_size * n

    def _p_color(p: float) -> str:
        if p < 0.01:
            return "#59a14f"  # green
        elif p < 0.05:
            return "#edc948"  # yellow
        elif p < 0.1:
            return "#f28e2b"  # orange
        return "#e15759"  # red

    for ri, row_name in enumerate(names):
        # Row label
        canvas.text(
            ml - 5, mt_map + ri * cell_size + cell_size / 2 + 4,
            row_name, font_size=11, text_anchor="end", fill="#333",
        )
        for ci, col_name in enumerate(names):
            x0 = ml + ci * cell_size
            y0 = mt_map + ri * cell_size

            entry = comparison.get(row_name, {}).get(col_name, {})
            p = entry.get("p_value", 1.0)
            color = _p_color(p)

            canvas.rect(x0, y0, cell_size, cell_size,
                         fill=color, stroke="white", stroke_width=1)

            # p-value label
            label = f"{p:.3f}" if isinstance(p, float) else str(p)
            canvas.text(
                x0 + cell_size / 2, y0 + cell_size / 2 + 4,
                label, font_size=10, text_anchor="middle", fill="#222",
            )

    # Column headers (top)
    for ci, col_name in enumerate(names):
        cx = ml + ci * cell_size + cell_size / 2
        canvas.text(
            cx, mt_map - 6, col_name,
            font_size=11, text_anchor="middle", fill="#333",
        )

    return canvas.to_string()


# ---------------------------------------------------------------------------
# 4. Ablation bars
# ---------------------------------------------------------------------------


def plot_ablation_bars(
    data: Dict[str, float],
    baseline_name: str,
    title: str = "Ablation Study",
    width: int = 700,
    height: int = 400,
) -> str:
    """Grouped bar chart for ablation studies.

    Shows baseline vs each ablation variant.  Colour-codes bars by
    whether they outperform the baseline (green) or not (red/orange).

    Returns SVG string.
    """
    canvas = SVGCanvas(width=width, height=height, background="white")
    ml, mr, mt, mb = 70, 30, 50, 60
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    canvas.text(width / 2, mt / 2 + 4, title,
                font_size=16, text_anchor="middle", fill="#333")

    names = list(data.keys())
    if not names:
        canvas.text(width / 2, height / 2, "(no data)",
                    font_size=14, text_anchor="middle", fill="#999")
        return canvas.to_string()

    baseline_val = data.get(baseline_name, 0.0)
    vals = list(data.values())
    y_lo = min(0.0, min(vals))
    y_hi = max(vals)
    nice_y_lo, nice_y_hi, y_step = _nice_axis_range(y_lo, y_hi)

    def ty(v: float) -> float:
        r = (v - nice_y_lo) / (nice_y_hi - nice_y_lo) if nice_y_hi != nice_y_lo else 0.5
        return mt + (1.0 - r) * plot_h

    # Background
    canvas.rect(ml, mt, plot_w, plot_h, fill="#fafafa", stroke="#ccc")

    # Y grid
    y_val = nice_y_lo
    while y_val <= nice_y_hi + y_step * 0.01:
        yy = ty(y_val)
        if mt <= yy <= mt + plot_h:
            canvas.line(ml, yy, ml + plot_w, yy, stroke="#eee")
            canvas.text(ml - 5, yy + 4, f"{y_val:.4g}",
                        font_size=10, text_anchor="end", fill="#666")
        y_val += y_step

    n = len(names)
    slot_w = plot_w / n
    bar_w = max(slot_w * 0.6, 8)
    zero_y = ty(0.0)

    for i, name in enumerate(names):
        v = data[name]
        cx = ml + slot_w * (i + 0.5)
        bx = cx - bar_w / 2
        top_y = ty(v)

        # Colour: baseline highlighted in blue, better in green, worse in orange
        if name == baseline_name:
            color = "#4e79a7"  # blue - baseline
        elif v <= baseline_val:
            color = "#59a14f"  # green - outperforms (lower is better)
        else:
            color = "#f28e2b"  # orange - underperforms

        # Bar from zero line to value
        bar_top = min(top_y, zero_y)
        bar_h = abs(top_y - zero_y)
        canvas.rect(bx, bar_top, bar_w, max(bar_h, 1), fill=color, stroke="#333",
                     stroke_width=0.5)

        # Label
        canvas.text(cx, mt + plot_h + 18, name,
                    font_size=10, text_anchor="middle", fill="#333")

        # Value on top of bar
        canvas.text(cx, top_y - 5, f"{v:.3g}",
                    font_size=9, text_anchor="middle", fill="#333")

    # Baseline reference line
    canvas.line(ml, ty(baseline_val), ml + plot_w, ty(baseline_val),
                stroke="#333", stroke_width=1, stroke_dasharray="4,3")

    return canvas.to_string()

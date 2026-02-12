"""Cross-fidelity diagnostics and SVG visualization.

Provides correlation analysis across fidelity levels and SVG
renderings for cost allocation and fidelity comparison.
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.visualization.svg_renderer import SVGCanvas


def cross_fidelity_correlation(
    observations_by_fidelity: dict[str, list[tuple[list[float], float]]],
) -> dict[tuple[str, str], float]:
    """Compute Spearman rank correlation between fidelity levels.

    For each pair of fidelity levels that share overlapping input
    locations (matched by nearest-neighbour within tolerance), compute
    the rank correlation of their objective values.

    Parameters
    ----------
    observations_by_fidelity : dict
        Mapping from fidelity name to list of ``(x_vector, y_value)``
        tuples.

    Returns
    -------
    dict[tuple[str, str], float]
        Rank correlations for each pair of fidelity levels.
        Self-correlations are 1.0.
    """
    levels = sorted(observations_by_fidelity.keys())
    correlations: dict[tuple[str, str], float] = {}

    for i, lv1 in enumerate(levels):
        correlations[(lv1, lv1)] = 1.0
        for j in range(i + 1, len(levels)):
            lv2 = levels[j]
            obs1 = observations_by_fidelity[lv1]
            obs2 = observations_by_fidelity[lv2]

            # Match observations by nearest input location
            matched_y1: list[float] = []
            matched_y2: list[float] = []
            tol = 1e-4

            for x1, y1 in obs1:
                for x2, y2 in obs2:
                    dist_sq = sum(
                        (a - b) ** 2 for a, b in zip(x1, x2)
                    )
                    if dist_sq < tol:
                        matched_y1.append(y1)
                        matched_y2.append(y2)
                        break

            if len(matched_y1) < 2:
                # Not enough matched data -- use all values and
                # compute Pearson-like correlation on sorted values
                vals1 = [y for _, y in obs1]
                vals2 = [y for _, y in obs2]
                min_len = min(len(vals1), len(vals2))
                if min_len < 2:
                    correlations[(lv1, lv2)] = 0.0
                    correlations[(lv2, lv1)] = 0.0
                    continue
                vals1 = sorted(vals1)[:min_len]
                vals2 = sorted(vals2)[:min_len]
                corr = _rank_correlation(vals1, vals2)
            else:
                corr = _rank_correlation(matched_y1, matched_y2)

            correlations[(lv1, lv2)] = corr
            correlations[(lv2, lv1)] = corr

    return correlations


def _rank_correlation(a: list[float], b: list[float]) -> float:
    """Compute Spearman rank correlation between two equal-length lists."""
    n = len(a)
    if n < 2:
        return 0.0

    # Compute ranks (average rank for ties)
    ranks_a = _compute_ranks(a)
    ranks_b = _compute_ranks(b)

    # Pearson correlation on ranks
    mean_a = sum(ranks_a) / n
    mean_b = sum(ranks_b) / n

    num = sum((ranks_a[i] - mean_a) * (ranks_b[i] - mean_b) for i in range(n))
    denom_a = math.sqrt(sum((ranks_a[i] - mean_a) ** 2 for i in range(n)))
    denom_b = math.sqrt(sum((ranks_b[i] - mean_b) ** 2 for i in range(n)))

    if denom_a < 1e-12 or denom_b < 1e-12:
        return 0.0

    return num / (denom_a * denom_b)


def _compute_ranks(values: list[float]) -> list[float]:
    """Compute ranks with average-rank tie-breaking."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-based
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks


def render_fidelity_comparison(
    observations_by_fidelity: dict[str, list[tuple[list[float], float]]],
    parameter_names: list[str],
) -> str:
    """Render an SVG scatter plot comparing objective values across fidelity levels.

    Shows a scatter of observations per fidelity level, using the first
    parameter dimension on the x-axis and the objective value on the y-axis.

    Parameters
    ----------
    observations_by_fidelity : dict
        Mapping from fidelity name to list of ``(x_vector, y_value)``.
    parameter_names : list[str]
        Names of the parameters (used for axis labels).

    Returns
    -------
    str
        SVG XML string.
    """
    width, height = 600, 400
    margin = 60
    canvas = SVGCanvas(width=width, height=height, background="white")

    levels = sorted(observations_by_fidelity.keys())
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    # Collect all x0 and y values for scaling
    all_x: list[float] = []
    all_y: list[float] = []
    for lv in levels:
        for x, y in observations_by_fidelity[lv]:
            if x:
                all_x.append(x[0])
            all_y.append(y)

    if not all_x or not all_y:
        canvas.text(width / 2, height / 2, "No data to display",
                    text_anchor="middle", font_size=14)
        return canvas.to_string()

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    def to_px(xv: float, yv: float) -> tuple[float, float]:
        px = margin + (xv - x_min) / x_range * plot_w
        py = margin + plot_h - (yv - y_min) / y_range * plot_h
        return px, py

    # Axes
    canvas.line(margin, margin, margin, height - margin, stroke="#333", stroke_width=1)
    canvas.line(margin, height - margin, width - margin, height - margin,
                stroke="#333", stroke_width=1)

    # Axis labels
    x_label = parameter_names[0] if parameter_names else "x0"
    canvas.text(width / 2, height - 10, x_label,
                text_anchor="middle", font_size=12, fill="#333")
    canvas.text(15, height / 2, "Objective",
                text_anchor="middle", font_size=12, fill="#333",
                transform=f"rotate(-90,15,{height / 2})")

    # Title
    canvas.text(width / 2, 20, "Fidelity Comparison",
                text_anchor="middle", font_size=14, fill="#333")

    # Plot points
    for idx, lv in enumerate(levels):
        color = colors[idx % len(colors)]
        for x, y in observations_by_fidelity[lv]:
            x0 = x[0] if x else 0.0
            px, py = to_px(x0, y)
            canvas.circle(px, py, 4, fill=color, opacity=0.7)

    # Legend
    for idx, lv in enumerate(levels):
        color = colors[idx % len(colors)]
        lx = width - margin + 10
        ly = margin + idx * 20
        canvas.circle(lx, ly, 4, fill=color)
        canvas.text(lx + 10, ly + 4, lv, font_size=10, fill="#333")

    return canvas.to_string()


def render_cost_allocation(cost_report: dict[str, Any]) -> str:
    """Render an SVG bar chart of cost allocation across fidelity levels.

    Parameters
    ----------
    cost_report : dict
        Output from ``MultiFidelityBackend.get_cost_report()``.

    Returns
    -------
    str
        SVG XML string.
    """
    width, height = 500, 350
    margin = 60
    canvas = SVGCanvas(width=width, height=height, background="white")

    per_level = cost_report.get("per_level", {})
    budget = cost_report.get("budget", 0.0)
    spent = cost_report.get("spent", 0.0)

    levels = sorted(per_level.keys())
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    if not levels:
        canvas.text(width / 2, height / 2, "No cost data",
                    text_anchor="middle", font_size=14)
        return canvas.to_string()

    # Title
    canvas.text(width / 2, 25, "Cost Allocation by Fidelity Level",
                text_anchor="middle", font_size=14, fill="#333")
    canvas.text(width / 2, 42,
                f"Budget: {budget:.1f} | Spent: {spent:.1f} | Remaining: {budget - spent:.1f}",
                text_anchor="middle", font_size=10, fill="#666")

    costs = [per_level[lv].get("total_cost", 0.0) for lv in levels]
    max_cost = max(max(costs), 1e-6)

    plot_w = width - 2 * margin
    plot_h = height - 2 * margin - 30  # extra space for title
    bar_width = plot_w / (len(levels) * 1.5 + 0.5)
    gap = bar_width * 0.5

    top_margin = margin + 30

    # Axes
    canvas.line(margin, top_margin, margin, height - margin,
                stroke="#333", stroke_width=1)
    canvas.line(margin, height - margin, width - margin, height - margin,
                stroke="#333", stroke_width=1)

    for idx, lv in enumerate(levels):
        cost = costs[idx]
        bar_h = (cost / max_cost) * plot_h if max_cost > 0 else 0
        x = margin + gap + idx * (bar_width + gap)
        y = height - margin - bar_h
        color = colors[idx % len(colors)]

        canvas.rect(x, y, bar_width, bar_h, fill=color, opacity=0.8)

        # Level name
        canvas.text(x + bar_width / 2, height - margin + 15, lv,
                    text_anchor="middle", font_size=10, fill="#333")

        # Cost value on bar
        if bar_h > 15:
            canvas.text(x + bar_width / 2, y + 15, f"{cost:.1f}",
                        text_anchor="middle", font_size=10, fill="white")

        # N evaluations
        n_evals = per_level[lv].get("n_evaluations", 0)
        canvas.text(x + bar_width / 2, y - 5, f"n={n_evals}",
                    text_anchor="middle", font_size=9, fill="#666")

    return canvas.to_string()

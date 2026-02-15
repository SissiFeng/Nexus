"""Hexagonal binning coverage view for parameter-space exploration.

Implements a hexbin plot that visualises how densely an optimisation
campaign has sampled a 2-D projection of the search space.  Each hex
cell can be coloured by observation density, surrogate predicted mean,
or surrogate uncertainty.

The hex grid uses *axial coordinates* (q, r) and flat-top orientation.
Conversion formulae follow the Red Blob Games hex-grid reference
(https://www.redblobgames.com/grids/hexagons/).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.visualization.models import PlotData, SurrogateModel
from optimization_copilot.visualization.svg_renderer import SVGCanvas


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HexCell:
    """A single hexagonal cell in the coverage grid.

    Parameters
    ----------
    q : int
        Axial q coordinate.
    r : int
        Axial r coordinate.
    center_x : float
        Pixel / world-space centre x.
    center_y : float
        Pixel / world-space centre y.
    count : int
        Number of observed points assigned to this hex.
    value : float
        Aggregated value used for colouring (meaning depends on *color_by*).
    """

    q: int
    r: int
    center_x: float
    center_y: float
    count: int = 0
    value: float = 0.0


# ---------------------------------------------------------------------------
# Hex-grid geometry helpers
# ---------------------------------------------------------------------------

def _hex_center(q: int, r: int, hex_size: float) -> tuple[float, float]:
    """Return the pixel centre of the hex at axial *(q, r)*.

    Uses flat-top orientation:
        x = hex_size * (3/2 * q)
        y = hex_size * (sqrt(3)/2 * q + sqrt(3) * r)
    """
    x = hex_size * (1.5 * q)
    y = hex_size * (math.sqrt(3) / 2.0 * q + math.sqrt(3) * r)
    return x, y


def _pixel_to_hex(x: float, y: float, hex_size: float) -> tuple[int, int]:
    """Convert pixel coordinates to the nearest axial hex (q, r).

    Inverse of ``_hex_center``.  Uses the cube-rounding approach.
    """
    q_frac = (2.0 / 3.0) * x / hex_size
    r_frac = (-1.0 / 3.0 * x + math.sqrt(3) / 3.0 * y) / hex_size
    return _cube_round(q_frac, r_frac)


def _cube_round(frac_q: float, frac_r: float) -> tuple[int, int]:
    """Round fractional axial coordinates to the nearest integer hex.

    Internally works in cube coordinates (q, r, s) where s = -q - r.
    """
    frac_s = -frac_q - frac_r

    rq = round(frac_q)
    rr = round(frac_r)
    rs = round(frac_s)

    dq = abs(rq - frac_q)
    dr = abs(rr - frac_r)
    ds = abs(rs - frac_s)

    # Reset the component with the largest rounding error.
    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    # else: rs = -rq - rr  (not needed since we return q, r)

    return int(rq), int(rr)


def _hex_vertices(
    cx: float,
    cy: float,
    hex_size: float,
) -> list[tuple[float, float]]:
    """Return the 6 vertices of a flat-top regular hexagon centred at *(cx, cy)*."""
    vertices: list[tuple[float, float]] = []
    for i in range(6):
        angle_deg = 60.0 * i
        angle_rad = math.radians(angle_deg)
        vx = cx + hex_size * math.cos(angle_rad)
        vy = cy + hex_size * math.sin(angle_rad)
        vertices.append((vx, vy))
    return vertices


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _density_color(normalised: float) -> str:
    """White-to-blue gradient.  *normalised* is in [0, 1]."""
    r = int(255 * (1.0 - normalised))
    g = int(255 * (1.0 - normalised))
    b = 255
    return f"#{r:02X}{g:02X}{b:02X}"


def _viridis_approx(normalised: float) -> str:
    """Simplified viridis-style gradient.  *normalised* is in [0, 1]."""
    # 5-stop linear interpolation matching the stops in colormaps.py.
    stops = [
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ]
    t = max(0.0, min(1.0, normalised))

    # Find surrounding stops.
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
            r = int(c0[0] + (c1[0] - c0[0]) * f)
            g = int(c0[1] + (c1[1] - c0[1]) * f)
            b = int(c0[2] + (c1[2] - c0[2]) * f)
            return f"#{r:02X}{g:02X}{b:02X}"

    # Fallback to last colour.
    c = stops[-1][1]
    return f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_hexbin_coverage(
    search_space: dict[str, tuple[float, float]],
    observed_points: list[dict[str, float]],
    predicted_surface: SurrogateModel | None = None,
    hex_size: float = 0.1,
    color_by: str = "density",
    param_x: str | None = None,
    param_y: str | None = None,
) -> PlotData:
    """Generate a hexagonal-binning coverage view of the 2-D parameter space.

    Parameters
    ----------
    search_space:
        ``{param_name: (lower, upper)}`` defining the full design space.
        Must contain at least 2 parameters.
    observed_points:
        List of observation dicts, e.g. ``[{"x1": 0.2, "x2": 0.5}, ...]``.
    predicted_surface:
        Optional surrogate model exposing ``predict(x) -> (mean, unc)``.
    hex_size:
        Hex cell radius as a *fraction* of the space extent (0, 1].
    color_by:
        ``"density"`` | ``"predicted_mean"`` | ``"uncertainty"``.
    param_x:
        Parameter mapped to the x-axis (default: first key).
    param_y:
        Parameter mapped to the y-axis (default: second key).

    Returns
    -------
    PlotData
        With ``plot_type="hexbin_coverage"`` and an embedded SVG.

    Raises
    ------
    ValueError
        If *search_space* has fewer than 2 parameters.
    """

    if len(search_space) < 2:
        raise ValueError(
            "Hexbin coverage requires at least 2 parameters in search_space, "
            f"got {len(search_space)}."
        )

    param_names = list(search_space.keys())
    if param_x is None:
        param_x = param_names[0]
    if param_y is None:
        param_y = param_names[1]

    x_lo, x_hi = search_space[param_x]
    y_lo, y_hi = search_space[param_y]
    x_span = x_hi - x_lo
    y_span = y_hi - y_lo

    # Absolute hex radius in parameter-space units.
    abs_hex_size = hex_size * max(x_span, y_span)
    if abs_hex_size <= 0:
        abs_hex_size = 1.0  # degenerate space guard

    # ------------------------------------------------------------------
    # 1. Build hex grid covering the bounding box
    # ------------------------------------------------------------------
    # Determine q, r ranges by converting bounding-box corners.
    corners = [
        (x_lo, y_lo),
        (x_hi, y_lo),
        (x_lo, y_hi),
        (x_hi, y_hi),
    ]
    q_vals: list[int] = []
    r_vals: list[int] = []
    for cx, cy in corners:
        q, r = _pixel_to_hex(cx, cy, abs_hex_size)
        q_vals.append(q)
        r_vals.append(r)

    q_min, q_max = min(q_vals) - 1, max(q_vals) + 1
    r_min, r_max = min(r_vals) - 1, max(r_vals) + 1

    grid: dict[tuple[int, int], HexCell] = {}
    for q in range(q_min, q_max + 1):
        for r in range(r_min, r_max + 1):
            hx, hy = _hex_center(q, r, abs_hex_size)
            # Only keep hexes whose centre falls within (or near) the space.
            if (x_lo - abs_hex_size <= hx <= x_hi + abs_hex_size
                    and y_lo - abs_hex_size <= hy <= y_hi + abs_hex_size):
                grid[(q, r)] = HexCell(q=q, r=r, center_x=hx, center_y=hy)

    # ------------------------------------------------------------------
    # 2. Assign observed points to hexes
    # ------------------------------------------------------------------
    for pt in observed_points:
        px = pt.get(param_x, 0.0)
        py = pt.get(param_y, 0.0)
        q, r = _pixel_to_hex(px, py, abs_hex_size)
        if (q, r) in grid:
            grid[(q, r)].count += 1

    # ------------------------------------------------------------------
    # 3. Compute cell values based on color_by mode
    # ------------------------------------------------------------------
    max_count = max((c.count for c in grid.values()), default=1) or 1

    for cell in grid.values():
        if color_by == "density":
            cell.value = cell.count / max_count
        elif color_by == "predicted_mean" and predicted_surface is not None:
            mean, _unc = predicted_surface.predict(
                [cell.center_x, cell.center_y]
            )
            cell.value = mean
        elif color_by == "uncertainty" and predicted_surface is not None:
            _mean, unc = predicted_surface.predict(
                [cell.center_x, cell.center_y]
            )
            cell.value = unc
        else:
            # Fallback: use density when surrogate is unavailable.
            cell.value = cell.count / max_count

    # Normalise predicted / uncertainty values to [0, 1].
    if color_by in ("predicted_mean", "uncertainty") and predicted_surface is not None:
        vals = [c.value for c in grid.values()]
        v_min = min(vals) if vals else 0.0
        v_max = max(vals) if vals else 1.0
        v_range = v_max - v_min if v_max != v_min else 1.0
        for cell in grid.values():
            cell.value = (cell.value - v_min) / v_range

    # ------------------------------------------------------------------
    # 4. Render SVG
    # ------------------------------------------------------------------
    margin = 60
    canvas_w = 800
    canvas_h = 600
    plot_w = canvas_w - 2 * margin
    plot_h = canvas_h - 2 * margin

    def _to_svg_x(val: float) -> float:
        return margin + (val - x_lo) / x_span * plot_w if x_span else margin + plot_w / 2

    def _to_svg_y(val: float) -> float:
        # SVG y-axis is inverted.
        return margin + plot_h - (val - y_lo) / y_span * plot_h if y_span else margin + plot_h / 2

    svg_hex_size = abs_hex_size / max(x_span, y_span) * max(plot_w, plot_h) if max(x_span, y_span) else 20

    canvas = SVGCanvas(width=canvas_w, height=canvas_h, background="white")

    # Axis labels.
    canvas.text(
        canvas_w / 2, canvas_h - 10, param_x,
        font_size=14, text_anchor="middle",
    )
    canvas.text(
        15, canvas_h / 2, param_y,
        font_size=14, text_anchor="middle",
        transform=f"rotate(-90, 15, {canvas_h / 2})",
    )

    # Draw hex cells.
    color_fn = _density_color if color_by == "density" else _viridis_approx
    for cell in grid.values():
        sx = _to_svg_x(cell.center_x)
        sy = _to_svg_y(cell.center_y)
        verts = _hex_vertices(sx, sy, svg_hex_size)
        fill = color_fn(cell.value)
        canvas.polygon(verts, fill=fill, stroke="#CCCCCC", stroke_width=0.5)

    # Draw observed points.
    for pt in observed_points:
        px = _to_svg_x(pt.get(param_x, 0.0))
        py = _to_svg_y(pt.get(param_y, 0.0))
        canvas.circle(px, py, 3, fill="#333333", opacity=0.7)

    svg_string = canvas.to_string()

    # ------------------------------------------------------------------
    # 5. Build PlotData
    # ------------------------------------------------------------------
    hex_data: list[dict[str, Any]] = [
        {
            "q": c.q,
            "r": c.r,
            "center_x": c.center_x,
            "center_y": c.center_y,
            "count": c.count,
            "value": c.value,
        }
        for c in grid.values()
    ]

    return PlotData(
        plot_type="hexbin_coverage",
        data={
            "hex_cells": hex_data,
            "observed_points": observed_points,
            "n_hexes": len(grid),
            "n_points": len(observed_points),
            "max_count": max_count,
        },
        metadata={
            "param_x": param_x,
            "param_y": param_y,
            "hex_size": hex_size,
            "color_by": color_by,
            "search_space": {k: list(v) for k, v in search_space.items()},
        },
        svg=svg_string,
    )

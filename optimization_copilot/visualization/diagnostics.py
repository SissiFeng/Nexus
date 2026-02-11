"""Space-filling quality metrics for design-of-experiments diagnostics.

Implements three complementary measures of how well a set of sample
points covers a bounded design space:

* **Star discrepancy** -- measures uniformity against the ideal
  uniform distribution.  Exact for d <= 5, randomised approximation
  for higher dimensions.
* **Coverage** -- percentage of grid cells that contain at least one
  sample point.
* **Minimum distance** -- smallest pairwise Euclidean distance in the
  normalised [0,1]^d hypercube (detects near-duplicates).

All three are exposed through the single public entry point
:func:`plot_space_filling_metrics`, which returns a
:class:`~optimization_copilot.visualization.models.PlotData` instance.
"""

from __future__ import annotations

import math
import random
from itertools import product as iterproduct

from optimization_copilot.visualization.models import PlotData

# ---------------------------------------------------------------------------
# Default metric names
# ---------------------------------------------------------------------------

_DEFAULT_METRICS: list[str] = ["discrepancy", "coverage", "min_distance"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise(
    points: list[list[float]],
    bounds: list[tuple[float, float]],
) -> list[list[float]]:
    """Map *points* into the [0, 1]^d unit hypercube using *bounds*."""
    normed: list[list[float]] = []
    for pt in points:
        row: list[float] = []
        for j, (lo, hi) in enumerate(bounds):
            span = hi - lo
            if span == 0.0:
                row.append(0.5)
            else:
                row.append((pt[j] - lo) / span)
        normed.append(row)
    return normed


# ---------------------------------------------------------------------------
# Star discrepancy  D*
# ---------------------------------------------------------------------------


def _compute_star_discrepancy(
    points: list[list[float]],
    bounds: list[tuple[float, float]],
) -> float:
    """Compute the star discrepancy D* of *points* within *bounds*.

    For dimensionality d <= 5 the exact value is computed by examining
    every anchored box whose upper corner is defined by a sample point
    coordinate (plus the unit boundary).  For d > 5 a Monte-Carlo
    approximation with 10 000 random corners is used instead.

    Parameters
    ----------
    points:
        Sample points in the original (un-normalised) space.
    bounds:
        Lower/upper bounds for each dimension.

    Returns
    -------
    float
        The star discrepancy value in [0, 1].  Returns 0.0 when
        *points* is empty.
    """
    if not points:
        return 0.0

    n = len(points)
    d = len(bounds)
    normed = _normalise(points, bounds)

    def _empirical_and_volume(corner: list[float]) -> float:
        """Return |F_n(corner) - Vol([0, corner])|."""
        # Volume of the anchored box [0, corner].
        vol = 1.0
        for c in corner:
            vol *= c

        # Fraction of points dominated by *corner*.
        count = 0
        for pt in normed:
            if all(pt[j] <= corner[j] for j in range(d)):
                count += 1
        f_n = count / n
        return abs(f_n - vol)

    if d <= 5:
        # Exact: enumerate all vertices formed by point coordinates + 1.0
        coords_per_dim: list[list[float]] = []
        for j in range(d):
            vals = sorted({pt[j] for pt in normed})
            if not vals or vals[-1] < 1.0:
                vals.append(1.0)
            coords_per_dim.append(vals)

        max_disc = 0.0
        for corner in iterproduct(*coords_per_dim):
            disc = _empirical_and_volume(list(corner))
            if disc > max_disc:
                max_disc = disc
        return max_disc
    else:
        # Random approximation for high dimensions.
        rng = random.Random(42)
        max_disc = 0.0
        for _ in range(10_000):
            corner = [rng.random() for _ in range(d)]
            disc = _empirical_and_volume(corner)
            if disc > max_disc:
                max_disc = disc
        return max_disc


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def _compute_coverage(
    points: list[list[float]],
    bounds: list[tuple[float, float]],
    resolution: int = 50,
) -> float:
    """Compute grid coverage percentage of *points* within *bounds*.

    Divides each dimension into *resolution* equal bins and counts the
    fraction of cells that contain at least one sample point.

    For d > 6 the resolution is adaptively lowered to
    ``max(5, int(100 ** (1/d)))`` to keep the total cell count
    tractable.

    Parameters
    ----------
    points:
        Sample points in the original space.
    bounds:
        Lower/upper bounds for each dimension.
    resolution:
        Number of bins per dimension (before adaptive reduction).

    Returns
    -------
    float
        Coverage percentage in [0, 100].  Returns 0.0 when *points* is
        empty.
    """
    if not points:
        return 0.0

    d = len(bounds)
    if d > 6:
        resolution = max(5, int(100 ** (1.0 / d)))

    total_cells = resolution ** d

    occupied: set[tuple[int, ...]] = set()
    for pt in points:
        cell: list[int] = []
        for j, (lo, hi) in enumerate(bounds):
            span = hi - lo
            if span == 0.0:
                cell.append(0)
            else:
                idx = int((pt[j] - lo) / span * resolution)
                # Clamp to valid range [0, resolution - 1].
                idx = max(0, min(resolution - 1, idx))
                cell.append(idx)
        occupied.add(tuple(cell))

    return (len(occupied) / total_cells) * 100.0


# ---------------------------------------------------------------------------
# Minimum pairwise distance
# ---------------------------------------------------------------------------


def _compute_min_distance(
    points: list[list[float]],
    bounds: list[tuple[float, float]],
) -> float:
    """Compute the minimum pairwise L2 distance in normalised space.

    Parameters
    ----------
    points:
        Sample points in the original space.
    bounds:
        Lower/upper bounds for each dimension.

    Returns
    -------
    float
        The minimum distance, or ``float('inf')`` if fewer than two
        points are provided.
    """
    if len(points) < 2:
        return float("inf")

    normed = _normalise(points, bounds)
    n = len(normed)
    d = len(bounds)
    min_dist = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = sum(
                (normed[i][k] - normed[j][k]) ** 2 for k in range(d)
            )
            dist = math.sqrt(dist_sq)
            if dist < min_dist:
                min_dist = dist
    return min_dist


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_METRIC_DISPATCH: dict[str, object] = {
    "discrepancy": _compute_star_discrepancy,
    "coverage": _compute_coverage,
    "min_distance": _compute_min_distance,
}


def plot_space_filling_metrics(
    points: list[list[float]],
    bounds: list[tuple[float, float]],
    metrics: list[str] | None = None,
) -> PlotData:
    """Compute space-filling quality metrics and wrap them in a PlotData.

    Parameters
    ----------
    points:
        Sample points, each a list of floats with length matching
        *bounds*.
    bounds:
        ``(lower, upper)`` for every dimension.
    metrics:
        Which metrics to compute.  Defaults to
        ``["discrepancy", "coverage", "min_distance"]``.

    Returns
    -------
    PlotData
        A ``PlotData`` with ``plot_type="space_filling_metrics"``
        containing the computed metrics in its ``data`` dict.
    """
    if metrics is None:
        metrics = list(_DEFAULT_METRICS)

    data: dict[str, object] = {}
    for name in metrics:
        fn = _METRIC_DISPATCH.get(name)
        if fn is None:
            raise ValueError(
                f"Unknown metric {name!r}. "
                f"Choose from {sorted(_METRIC_DISPATCH)}."
            )
        data[name] = fn(points, bounds)  # type: ignore[operator]

    data["n_points"] = len(points)
    data["n_dims"] = len(bounds)

    return PlotData(
        plot_type="space_filling_metrics",
        data=data,  # type: ignore[arg-type]
        metadata={"bounds": bounds, "metrics_computed": metrics},
    )

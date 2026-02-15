"""Design-space exploration visualizations (v3 spec section 7).

Provides four complementary views of the optimization design space:

* **Low-dimensional projection** (PCA / t-SNE) -- reduces high-dimensional
  parameter spaces to 2-D scatter plots coloured by objective, uncertainty,
  or iteration index.
* **iSOM landscape** -- trains a self-organising map on the design space and
  colours each node by the mean objective of its mapped data points, with a
  U-matrix overlay showing cluster boundaries.
* **Forward / inverse design** -- sweeps a surrogate model over a parameter
  grid to build prediction surfaces, and optionally highlights the feasible
  region that satisfies target objectives within a tolerance.

All functions return a :class:`~optimization_copilot.visualization.models.PlotData`
instance and are pure Python with zero external dependencies.
"""

from __future__ import annotations

import math
import random

from optimization_copilot.visualization.models import PlotData, SurrogateModel
from optimization_copilot.visualization.svg_renderer import SVGCanvas
from optimization_copilot.backends._math.linalg import eigen_symmetric, vec_dot

# Try to import the colormap module for viridis-style colouring; fall back to
# a simple green-to-blue gradient if unavailable.
try:
    from optimization_copilot.visualization.colormaps import VSUPColorMap, color_to_hex

    _HAS_COLORMAPS = True
except ImportError:  # pragma: no cover
    _HAS_COLORMAPS = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _value_to_hex(value: float, lo: float, hi: float) -> str:
    """Map a scalar *value* in [*lo*, *hi*] to a viridis-style hex colour.

    Falls back to a simple green-blue gradient when the colormaps module is
    not available.
    """
    span = hi - lo
    if span == 0.0:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - lo) / span))

    if _HAS_COLORMAPS:
        cm = VSUPColorMap(value_cmap="viridis")
        r, g, b, _a = cm.map(t, uncertainty=0.0)
        return color_to_hex(r, g, b)

    # Minimal fallback: dark purple (low) -> teal (high).
    r = int(68 + (33 - 68) * t)
    g = int(1 + (145 - 1) * t)
    b = int(84 + (140 - 84) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def _pairwise_sq_distances(X: list[list[float]]) -> list[list[float]]:
    """Return squared Euclidean distance matrix for rows of *X*."""
    n = len(X)
    d = len(X[0]) if n > 0 else 0
    D: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = 0.0
            for k in range(d):
                diff = X[i][k] - X[j][k]
                s += diff * diff
            D[i][j] = s
            D[j][i] = s
    return D


# ---------------------------------------------------------------------------
# 7.1  Low-dimensional projection  (PCA / t-SNE)
# ---------------------------------------------------------------------------

def _pca_project(
    X: list[list[float]],
    seed: int = 42,
) -> tuple[list[list[float]], list[float]]:
    """Project *X* to 2-D via PCA.

    Returns ``(Z, explained_variance_ratios)`` where *Z* has shape (n, 2).
    """
    n = len(X)
    d = len(X[0]) if n > 0 else 0

    if n == 0 or d == 0:
        return [], [0.0, 0.0]

    # 1. Centre data (subtract column means).
    means = [0.0] * d
    for row in X:
        for j in range(d):
            means[j] += row[j]
    means = [m / n for m in means]

    X_c = [[row[j] - means[j] for j in range(d)] for row in X]

    # 2. Covariance matrix  (d x d).
    cov: list[list[float]] = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            s = 0.0
            for row in X_c:
                s += row[i] * row[j]
            val = s / max(n - 1, 1)
            cov[i][j] = val
            cov[j][i] = val

    # 3. Top-2 eigenvectors.
    k = min(2, d)
    eigenvalues, eigenvectors = eigen_symmetric(cov, k=k, seed=seed)

    # Explained variance ratios.
    total_var = sum(cov[i][i] for i in range(d))
    if total_var == 0.0:
        ev_ratios = [0.0] * k
    else:
        ev_ratios = [ev / total_var for ev in eigenvalues]

    # Pad to length 2 if d == 1.
    while len(eigenvectors) < 2:
        eigenvectors.append([0.0] * d)
        ev_ratios.append(0.0)

    # 4. Project: Z = X_c @ V^T  (each eigenvector is a row in V).
    Z: list[list[float]] = []
    for row in X_c:
        z0 = vec_dot(row, eigenvectors[0])
        z1 = vec_dot(row, eigenvectors[1])
        Z.append([z0, z1])

    return Z, ev_ratios[:2]


def _tsne_project(
    X: list[list[float]],
    seed: int = 42,
    perplexity: float = 30.0,
    n_iter: int = 300,
    learning_rate: float = 200.0,
) -> list[list[float]]:
    """Simplified t-SNE projection of *X* to 2-D.

    Pure Python, O(n^2) per iteration -- suitable for n < 500.
    """
    n = len(X)
    if n == 0:
        return []
    if n == 1:
        return [[0.0, 0.0]]

    rng = random.Random(seed)

    # Effective perplexity must be < n.
    perp = min(perplexity, n - 1)
    if perp < 1.0:
        perp = 1.0

    # --- Pairwise distances --------------------------------------------------
    D_sq = _pairwise_sq_distances(X)

    # --- Compute conditional probabilities p(j|i) via binary search ----------
    P: list[list[float]] = [[0.0] * n for _ in range(n)]
    log_perp = math.log(perp)

    for i in range(n):
        lo_beta, hi_beta = 1e-20, 1e10
        beta = 1.0  # beta = 1 / (2 * sigma_i^2)

        for _attempt in range(50):  # binary search iterations
            # Compute conditional probabilities for point i.
            denom = 0.0
            for j in range(n):
                if j == i:
                    continue
                denom += math.exp(-D_sq[i][j] * beta)
            if denom == 0.0:
                denom = 1e-300

            # Entropy H and p(j|i).
            H = 0.0
            for j in range(n):
                if j == i:
                    P[i][j] = 0.0
                    continue
                pji = math.exp(-D_sq[i][j] * beta) / denom
                P[i][j] = max(pji, 1e-300)
                H += -pji * math.log(max(pji, 1e-300))

            # Check perplexity.
            diff = H - log_perp
            if abs(diff) < 1e-5:
                break
            if diff > 0.0:
                lo_beta = beta
                beta = (beta + hi_beta) / 2.0 if hi_beta < 1e9 else beta * 2.0
            else:
                hi_beta = beta
                beta = (beta + lo_beta) / 2.0

    # --- Symmetrise: P_sym = (P_ij + P_ji) / (2n) ---------------------------
    for i in range(n):
        for j in range(i + 1, n):
            val = (P[i][j] + P[j][i]) / (2.0 * n)
            val = max(val, 1e-12)
            P[i][j] = val
            P[j][i] = val

    # --- Initialise Y randomly -----------------------------------------------
    Y: list[list[float]] = [[rng.gauss(0, 1e-4), rng.gauss(0, 1e-4)] for _ in range(n)]
    Y_prev: list[list[float]] = [[y[0], y[1]] for y in Y]

    # --- Gradient descent -----------------------------------------------------
    for it in range(n_iter):
        # Early exaggeration (4x) for first 100 iterations.
        exag = 4.0 if it < 100 else 1.0
        momentum = 0.5 if it < 100 else 0.8

        # Compute q_{ij} (Student-t with 1 DoF).
        Q_num: list[list[float]] = [[0.0] * n for _ in range(n)]
        q_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dy0 = Y[i][0] - Y[j][0]
                dy1 = Y[i][1] - Y[j][1]
                val = 1.0 / (1.0 + dy0 * dy0 + dy1 * dy1)
                Q_num[i][j] = val
                Q_num[j][i] = val
                q_sum += 2.0 * val
        if q_sum == 0.0:
            q_sum = 1e-300

        # Gradient.
        grad: list[list[float]] = [[0.0, 0.0] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                q_ij = Q_num[i][j] / q_sum
                pq = (exag * P[i][j] - q_ij) * Q_num[i][j]
                grad[i][0] += 4.0 * pq * (Y[i][0] - Y[j][0])
                grad[i][1] += 4.0 * pq * (Y[i][1] - Y[j][1])

        # Update with momentum.
        for i in range(n):
            new0 = Y[i][0] - learning_rate * grad[i][0] + momentum * (Y[i][0] - Y_prev[i][0])
            new1 = Y[i][1] - learning_rate * grad[i][1] + momentum * (Y[i][1] - Y_prev[i][1])
            Y_prev[i][0] = Y[i][0]
            Y_prev[i][1] = Y[i][1]
            Y[i][0] = new0
            Y[i][1] = new1

    return Y


def plot_latent_space_exploration(
    X_observed: list[list[float]],
    Y_observed: list[float],
    method: str = "pca",
    color_by: str = "objective",
    seed: int = 42,
) -> PlotData:
    """Low-dimensional projection of the parameter space (v3 spec 7.1).

    Parameters
    ----------
    X_observed:
        Observed parameter-space points, each a list of floats.
    Y_observed:
        Objective values corresponding to each row of *X_observed*.
    method:
        ``"pca"`` or ``"tsne"``.
    color_by:
        ``"objective"``, ``"uncertainty"``, or ``"iteration"``.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    PlotData
        A ``PlotData`` with ``plot_type="latent_space_exploration"``.
    """
    n = len(X_observed)
    metadata: dict = {"method": method, "color_by": color_by, "seed": seed}
    ev_ratios: list[float] = []

    # --- Projection -----------------------------------------------------------
    if method == "pca":
        Z, ev_ratios = _pca_project(X_observed, seed=seed)
        metadata["explained_variance_ratios"] = ev_ratios
    elif method == "tsne":
        Z = _tsne_project(X_observed, seed=seed)
    else:
        raise ValueError(f"Unknown projection method {method!r}. Use 'pca' or 'tsne'.")

    # --- Colour mapping -------------------------------------------------------
    if color_by == "iteration":
        color_values = list(range(n))
    elif color_by == "uncertainty":
        # Uncertainty colouring uses the same values as objective (caller may
        # supply uncertainty in Y_observed for this mode).
        color_values = list(Y_observed)
    else:  # "objective"
        color_values = list(Y_observed)

    c_lo = min(color_values) if color_values else 0.0
    c_hi = max(color_values) if color_values else 1.0

    hex_colors: list[str] = [_value_to_hex(v, c_lo, c_hi) for v in color_values]

    # --- SVG rendering --------------------------------------------------------
    canvas = SVGCanvas(width=600, height=500, background="white")

    if Z:
        margin = 60
        plot_w = 600 - 2 * margin
        plot_h = 500 - 2 * margin

        z0_vals = [p[0] for p in Z]
        z1_vals = [p[1] for p in Z]
        z0_lo, z0_hi = min(z0_vals), max(z0_vals)
        z1_lo, z1_hi = min(z1_vals), max(z1_vals)
        z0_span = z0_hi - z0_lo if z0_hi != z0_lo else 1.0
        z1_span = z1_hi - z1_lo if z1_hi != z1_lo else 1.0

        # Axes.
        canvas.line(margin, 500 - margin, 600 - margin, 500 - margin, stroke="#888", stroke_width=1)
        canvas.line(margin, margin, margin, 500 - margin, stroke="#888", stroke_width=1)
        label0 = "PC1" if method == "pca" else "Dim 1"
        label1 = "PC2" if method == "pca" else "Dim 2"
        canvas.text(300, 495, label0, font_size=11, text_anchor="middle", fill="#555")
        canvas.text(15, 250, label1, font_size=11, text_anchor="middle", fill="#555",
                    transform="rotate(-90 15 250)")

        for i, (z, col) in enumerate(zip(Z, hex_colors)):
            sx = margin + (z[0] - z0_lo) / z0_span * plot_w
            sy = (500 - margin) - (z[1] - z1_lo) / z1_span * plot_h
            canvas.circle(sx, sy, 4, fill=col, stroke="#333", stroke_width=0.5, opacity=0.85)

    data_dict: dict = {
        "points_2d": Z,
        "color_values": color_values,
        "hex_colors": hex_colors,
        "n_points": n,
    }
    if ev_ratios:
        data_dict["explained_variance_ratios"] = ev_ratios

    return PlotData(
        plot_type="latent_space_exploration",
        data=data_dict,
        metadata=metadata,
        svg=canvas.to_string(),
    )


# ---------------------------------------------------------------------------
# 7.2  iSOM landscape
# ---------------------------------------------------------------------------

def plot_isom_landscape(
    X_observed: list[list[float]],
    Y_observed: list[float],
    grid_size: tuple[int, int] = (10, 10),
    n_iterations: int | None = None,
    seed: int = 42,
) -> PlotData:
    """Interpolated Self-Organising Map landscape (v3 spec 7.2).

    Parameters
    ----------
    X_observed:
        Observed parameter-space points.
    Y_observed:
        Objective values.
    grid_size:
        ``(width, height)`` of the SOM grid.
    n_iterations:
        Training iterations.  Defaults to ``500 * grid_w * grid_h``.
    seed:
        Random seed.

    Returns
    -------
    PlotData
        A ``PlotData`` with ``plot_type="isom_landscape"``.
    """
    rng = random.Random(seed)
    n = len(X_observed)
    d = len(X_observed[0]) if n > 0 else 1
    gw, gh = grid_size

    if n_iterations is None:
        n_iterations = 500 * gw * gh

    # --- Determine data range per dimension ----------------------------------
    lo = [float("inf")] * d
    hi = [float("-inf")] * d
    for pt in X_observed:
        for j in range(d):
            if pt[j] < lo[j]:
                lo[j] = pt[j]
            if pt[j] > hi[j]:
                hi[j] = pt[j]
    # Handle degenerate ranges.
    for j in range(d):
        if lo[j] == hi[j]:
            lo[j] -= 0.5
            hi[j] += 0.5

    # --- Initialise prototype vectors ----------------------------------------
    prototypes: list[list[float]] = []
    for _ in range(gw * gh):
        proto = [rng.uniform(lo[j], hi[j]) for j in range(d)]
        prototypes.append(proto)

    # --- Competitive learning ------------------------------------------------
    initial_lr = 0.5
    initial_radius = max(gw, gh) / 2.0

    for it in range(n_iterations):
        t_frac = it / max(n_iterations - 1, 1)

        # Decaying learning rate and neighbourhood radius.
        lr = initial_lr * (1.0 - t_frac)
        radius = initial_radius * math.exp(-3.0 * t_frac)
        if radius < 0.5:
            radius = 0.5

        # Pick a random data point.
        idx = rng.randint(0, n - 1) if n > 0 else 0
        if n == 0:
            break
        x = X_observed[idx]

        # Find BMU (best matching unit).
        best_dist = float("inf")
        bmu = 0
        for k in range(gw * gh):
            dist = 0.0
            for j in range(d):
                diff = x[j] - prototypes[k][j]
                dist += diff * diff
            if dist < best_dist:
                best_dist = dist
                bmu = k

        bmu_row, bmu_col = divmod(bmu, gw)

        # Update BMU and neighbours.
        for k in range(gw * gh):
            k_row, k_col = divmod(k, gw)
            grid_dist_sq = (k_row - bmu_row) ** 2 + (k_col - bmu_col) ** 2
            # Gaussian neighbourhood.
            h = math.exp(-grid_dist_sq / (2.0 * radius * radius))
            if h < 1e-6:
                continue
            for j in range(d):
                prototypes[k][j] += lr * h * (x[j] - prototypes[k][j])

    # --- Map data points to their BMU ----------------------------------------
    bmu_map: dict[int, list[int]] = {}  # node_idx -> list of data indices
    for i in range(n):
        best_dist = float("inf")
        bmu = 0
        for k in range(gw * gh):
            dist = 0.0
            for j in range(d):
                diff = X_observed[i][j] - prototypes[k][j]
                dist += diff * diff
            if dist < best_dist:
                best_dist = dist
                bmu = k
        bmu_map.setdefault(bmu, []).append(i)

    # --- Colour each node by mean Y of mapped points ------------------------
    node_colors: list[float | None] = [None] * (gw * gh)
    for k in range(gw * gh):
        indices = bmu_map.get(k, [])
        if indices:
            node_colors[k] = sum(Y_observed[i] for i in indices) / len(indices)

    # --- Compute U-matrix (mean distance to grid neighbours) -----------------
    u_matrix: list[float] = [0.0] * (gw * gh)
    for k in range(gw * gh):
        k_row, k_col = divmod(k, gw)
        neighbours: list[int] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = k_row + dr, k_col + dc
            if 0 <= nr < gh and 0 <= nc < gw:
                neighbours.append(nr * gw + nc)
        if neighbours:
            total = 0.0
            for nk in neighbours:
                dist = 0.0
                for j in range(d):
                    diff = prototypes[k][j] - prototypes[nk][j]
                    dist += diff * diff
                total += math.sqrt(dist)
            u_matrix[k] = total / len(neighbours)

    # --- SVG rendering -------------------------------------------------------
    canvas = SVGCanvas(width=600, height=600, background="white")
    margin = 50
    cell_w = (600 - 2 * margin) / gw
    cell_h = (600 - 2 * margin) / gh

    # Colour range from node colours.
    valid_colors = [c for c in node_colors if c is not None]
    c_lo = min(valid_colors) if valid_colors else 0.0
    c_hi = max(valid_colors) if valid_colors else 1.0

    for k in range(gw * gh):
        k_row, k_col = divmod(k, gw)
        cx = margin + k_col * cell_w
        cy = margin + k_row * cell_h
        if node_colors[k] is not None:
            fill = _value_to_hex(node_colors[k], c_lo, c_hi)
        else:
            fill = "#EEEEEE"
        canvas.rect(cx, cy, cell_w, cell_h, fill=fill, stroke="#CCC", stroke_width=0.5)

    canvas.text(300, 20, "iSOM Landscape", font_size=14, text_anchor="middle", fill="#333")

    data_dict: dict = {
        "grid_size": list(grid_size),
        "prototypes": prototypes,
        "node_colors": node_colors,
        "u_matrix": u_matrix,
        "bmu_map": {str(k): v for k, v in bmu_map.items()},
        "n_points": n,
    }

    return PlotData(
        plot_type="isom_landscape",
        data=data_dict,
        metadata={
            "grid_size": list(grid_size),
            "n_iterations": n_iterations,
            "seed": seed,
        },
        svg=canvas.to_string(),
    )


# ---------------------------------------------------------------------------
# 7.3  Forward / inverse design
# ---------------------------------------------------------------------------

def plot_forward_inverse_design(
    parameter_space: dict[str, tuple[float, float]],
    objective_space: dict[str, tuple[float, float]],
    mapping_model: SurrogateModel,
    target_objectives: list[float] | None = None,
    grid_resolution: int = 20,
    tolerance: float = 0.1,
) -> PlotData:
    """Forward and inverse design-space exploration (v3 spec 7.3).

    Parameters
    ----------
    parameter_space:
        ``{name: (lower, upper)}`` for each design parameter.
    objective_space:
        ``{name: (lower, upper)}`` for each objective dimension.
    mapping_model:
        A surrogate model satisfying the ``SurrogateModel`` protocol.
    target_objectives:
        Desired objective values for inverse design.  When provided the
        function identifies parameter-grid points whose predicted
        objectives fall within *tolerance* of these targets.
    grid_resolution:
        Number of grid points per parameter dimension.
    tolerance:
        Acceptable deviation from *target_objectives* for feasibility.

    Returns
    -------
    PlotData
        A ``PlotData`` with ``plot_type="forward_inverse_design"``.
    """
    param_names = list(parameter_space.keys())
    param_bounds = [parameter_space[p] for p in param_names]
    n_params = len(param_names)

    # --- Build grid -----------------------------------------------------------
    axes: list[list[float]] = []
    for lo, hi in param_bounds:
        if grid_resolution <= 1:
            axes.append([(lo + hi) / 2.0])
        else:
            step = (hi - lo) / (grid_resolution - 1)
            axes.append([lo + i * step for i in range(grid_resolution)])

    # Generate grid via itertools-style Cartesian product (pure Python).
    grid_points: list[list[float]] = [[]]
    for ax in axes:
        new_points: list[list[float]] = []
        for pt in grid_points:
            for v in ax:
                new_points.append(pt + [v])
        grid_points = new_points

    # --- Forward: predict over grid ------------------------------------------
    predictions: list[float] = []
    uncertainties: list[float] = []
    for pt in grid_points:
        mean, unc = mapping_model.predict(pt)
        predictions.append(mean)
        uncertainties.append(unc)

    # --- Inverse: filter feasible points -------------------------------------
    feasible_indices: list[int] = []
    if target_objectives is not None and len(target_objectives) > 0:
        target_val = target_objectives[0]  # First objective target.
        for idx, pred in enumerate(predictions):
            if abs(pred - target_val) <= tolerance:
                feasible_indices.append(idx)

    feasible_points = [grid_points[i] for i in feasible_indices]
    feasible_predictions = [predictions[i] for i in feasible_indices]

    # --- SVG rendering -------------------------------------------------------
    canvas = SVGCanvas(width=650, height=500, background="white")
    margin = 60

    if n_params >= 1 and grid_points:
        plot_w = 650 - 2 * margin
        plot_h = 500 - 2 * margin

        # Use first two parameter dimensions for scatter axes.
        dim0 = [pt[0] for pt in grid_points]
        dim1 = [pt[1] for pt in grid_points] if n_params >= 2 else predictions

        d0_lo, d0_hi = min(dim0), max(dim0)
        d1_lo, d1_hi = min(dim1), max(dim1)
        d0_span = d0_hi - d0_lo if d0_hi != d0_lo else 1.0
        d1_span = d1_hi - d1_lo if d1_hi != d1_lo else 1.0

        p_lo = min(predictions) if predictions else 0.0
        p_hi = max(predictions) if predictions else 1.0

        # Axes.
        canvas.line(margin, 500 - margin, 650 - margin, 500 - margin, stroke="#888")
        canvas.line(margin, margin, margin, 500 - margin, stroke="#888")
        xlabel = param_names[0] if n_params >= 1 else "x0"
        ylabel = param_names[1] if n_params >= 2 else "Prediction"
        canvas.text(350, 495, xlabel, font_size=11, text_anchor="middle", fill="#555")
        canvas.text(15, 250, ylabel, font_size=11, text_anchor="middle", fill="#555",
                    transform="rotate(-90 15 250)")

        # Forward scatter -- sample a subset for large grids.
        max_render = 2000
        step = max(1, len(grid_points) // max_render)
        for idx in range(0, len(grid_points), step):
            sx = margin + (dim0[idx] - d0_lo) / d0_span * plot_w
            sy = (500 - margin) - (dim1[idx] - d1_lo) / d1_span * plot_h
            fill = _value_to_hex(predictions[idx], p_lo, p_hi)
            canvas.circle(sx, sy, 2.5, fill=fill, opacity=0.6)

        # Feasible region highlight.
        feasible_set = set(feasible_indices)
        for idx in feasible_indices:
            sx = margin + (dim0[idx] - d0_lo) / d0_span * plot_w
            sy = (500 - margin) - (dim1[idx] - d1_lo) / d1_span * plot_h
            canvas.circle(sx, sy, 5, fill="none", stroke="#FF4444", stroke_width=1.5, opacity=0.9)

    title = "Forward Design"
    if target_objectives is not None:
        title = "Forward / Inverse Design"
    canvas.text(325, 20, title, font_size=14, text_anchor="middle", fill="#333")

    data_dict: dict = {
        "grid_points": grid_points,
        "predictions": predictions,
        "uncertainties": uncertainties,
        "feasible_indices": feasible_indices,
        "feasible_points": feasible_points,
        "feasible_predictions": feasible_predictions,
        "n_grid_points": len(grid_points),
        "grid_resolution": grid_resolution,
    }

    metadata: dict = {
        "parameter_names": param_names,
        "parameter_bounds": param_bounds,
        "objective_names": list(objective_space.keys()),
        "target_objectives": target_objectives,
        "tolerance": tolerance,
    }

    return PlotData(
        plot_type="forward_inverse_design",
        data=data_dict,
        metadata=metadata,
        svg=canvas.to_string(),
    )

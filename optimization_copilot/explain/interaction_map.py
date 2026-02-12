"""fANOVA decomposition and 2D interaction heatmaps.

Provides an ``InteractionMap`` class that builds simple regression tree stumps
to estimate per-feature main effects and pairwise interaction strengths.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from optimization_copilot.visualization.svg_renderer import SVGCanvas


# ---------------------------------------------------------------------------
# Internal tree structure
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """A single axis-aligned decision stump."""

    feature: int
    threshold: float
    left_value: float  # mean y for left partition
    right_value: float  # mean y for right partition
    variance_reduction: float
    n_left: int
    n_right: int


# ---------------------------------------------------------------------------
# InteractionMap
# ---------------------------------------------------------------------------

class InteractionMap:
    """fANOVA-style decomposition of feature importances and interactions.

    Uses an ensemble of simple decision-tree stumps to estimate main effects
    and pairwise interaction strengths.

    Parameters
    ----------
    n_trees : int
        Number of stumps in the ensemble.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_trees: int = 50, seed: int = 42) -> None:
        self.n_trees = n_trees
        self.seed = seed
        self._rng = random.Random(seed)
        self._trees: list[TreeNode] = []
        self._X: list[list[float]] = []
        self._y: list[float] = []
        self._n_features: int = 0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        """Build regression tree stumps for fANOVA decomposition.

        Parameters
        ----------
        X : list[list[float]]
            Feature matrix, shape ``(n_samples, n_features)``.
        y : list[float]
            Target values, length ``n_samples``.
        """
        if not X or not y:
            self._fitted = True
            self._X = X
            self._y = y
            self._n_features = 0
            self._trees = []
            return

        self._X = [list(row) for row in X]
        self._y = list(y)
        n = len(X)
        self._n_features = len(X[0])
        self._trees = []

        if self._n_features == 0 or n < 2:
            self._fitted = True
            return

        total_var = _variance(y)
        if total_var < 1e-15:
            # Constant target — build trivial trees with zero variance reduction
            for _ in range(self.n_trees):
                feat = self._rng.randint(0, self._n_features - 1)
                self._trees.append(TreeNode(
                    feature=feat,
                    threshold=0.0,
                    left_value=y[0],
                    right_value=y[0],
                    variance_reduction=0.0,
                    n_left=n,
                    n_right=0,
                ))
            self._fitted = True
            return

        for _ in range(self.n_trees):
            feat = self._rng.randint(0, self._n_features - 1)
            # Collect unique feature values and choose a threshold
            vals = sorted({X[i][feat] for i in range(n)})
            if len(vals) < 2:
                # Cannot split on this feature — pick a trivial split
                self._trees.append(TreeNode(
                    feature=feat,
                    threshold=vals[0] if vals else 0.0,
                    left_value=_mean(y),
                    right_value=_mean(y),
                    variance_reduction=0.0,
                    n_left=n,
                    n_right=0,
                ))
                continue

            # Try a random subset of midpoints and pick the best split
            n_candidates = min(len(vals) - 1, 10)
            candidate_indices = self._rng.sample(range(len(vals) - 1), n_candidates)
            best_vr = -1.0
            best_thresh = vals[0]
            best_left_val = 0.0
            best_right_val = 0.0
            best_nl = 0
            best_nr = 0

            for ci in candidate_indices:
                thresh = (vals[ci] + vals[ci + 1]) / 2.0
                left_y = [y[i] for i in range(n) if X[i][feat] <= thresh]
                right_y = [y[i] for i in range(n) if X[i][feat] > thresh]
                if not left_y or not right_y:
                    continue
                vr = _variance_reduction(y, left_y, right_y)
                if vr > best_vr:
                    best_vr = vr
                    best_thresh = thresh
                    best_left_val = _mean(left_y)
                    best_right_val = _mean(right_y)
                    best_nl = len(left_y)
                    best_nr = len(right_y)

            self._trees.append(TreeNode(
                feature=feat,
                threshold=best_thresh,
                left_value=best_left_val,
                right_value=best_right_val,
                variance_reduction=max(best_vr, 0.0),
                n_left=best_nl,
                n_right=best_nr,
            ))

        self._fitted = True

    # ------------------------------------------------------------------
    # Main effects
    # ------------------------------------------------------------------

    def compute_main_effects(self) -> dict[int, float]:
        """Return per-feature importance scores normalised to sum to 1.

        Returns
        -------
        dict[int, float]
            Feature index to importance score mapping.
        """
        if not self._trees or self._n_features == 0:
            return {}

        raw: dict[int, float] = {i: 0.0 for i in range(self._n_features)}
        for tree in self._trees:
            raw[tree.feature] += tree.variance_reduction

        total = sum(raw.values())
        if total < 1e-15:
            # Uniform importance when no variance reduction
            n = self._n_features
            return {i: 1.0 / n for i in range(n)}

        return {i: v / total for i, v in raw.items()}

    # ------------------------------------------------------------------
    # Pairwise interactions
    # ------------------------------------------------------------------

    def compute_interactions(self) -> dict[tuple[int, int], float]:
        """Return pairwise interaction scores.

        For each pair ``(i, j)`` the interaction is estimated as the
        additional variance explained when splitting on feature *i* then
        conditioning on feature *j*, minus the individual main effects.

        Returns
        -------
        dict[tuple[int, int], float]
            Pair ``(i, j)`` to interaction strength (non-negative).
        """
        if not self._fitted or self._n_features < 2 or len(self._y) < 4:
            return {}

        X = self._X
        y = self._y
        n = len(y)
        total_var = _variance(y)
        if total_var < 1e-15:
            return {(i, j): 0.0
                    for i in range(self._n_features)
                    for j in range(i + 1, self._n_features)}

        main = self.compute_main_effects()
        interactions: dict[tuple[int, int], float] = {}

        for i in range(self._n_features):
            for j in range(i + 1, self._n_features):
                # Split on i then j: compute variance explained jointly
                joint_var = self._joint_variance_explained(X, y, n, i, j)
                interaction = max(joint_var - main.get(i, 0.0) - main.get(j, 0.0), 0.0)
                interactions[(i, j)] = interaction

        # Normalise interactions relative to total
        total_inter = sum(interactions.values())
        if total_inter > 1e-15:
            interactions = {k: v / total_inter for k, v in interactions.items()}

        return interactions

    def _joint_variance_explained(
        self,
        X: list[list[float]],
        y: list[float],
        n: int,
        feat_i: int,
        feat_j: int,
    ) -> float:
        """Estimate the fraction of variance explained by jointly splitting on two features."""
        # Find median splits for each feature
        vals_i = sorted({X[k][feat_i] for k in range(n)})
        vals_j = sorted({X[k][feat_j] for k in range(n)})
        if len(vals_i) < 2 or len(vals_j) < 2:
            return 0.0

        thresh_i = (vals_i[len(vals_i) // 2 - 1] + vals_i[len(vals_i) // 2]) / 2.0
        thresh_j = (vals_j[len(vals_j) // 2 - 1] + vals_j[len(vals_j) // 2]) / 2.0

        # Partition into 4 quadrants
        buckets: list[list[float]] = [[], [], [], []]
        for k in range(n):
            left_i = X[k][feat_i] <= thresh_i
            left_j = X[k][feat_j] <= thresh_j
            idx = (0 if left_i else 1) * 2 + (0 if left_j else 1)
            buckets[idx].append(y[k])

        # Variance explained = 1 - weighted_within_var / total_var
        total_var = _variance(y)
        if total_var < 1e-15:
            return 0.0

        within_var = 0.0
        for bucket in buckets:
            if bucket:
                within_var += len(bucket) * _variance(bucket)
        within_var /= n

        return max(1.0 - within_var / total_var, 0.0)

    # ------------------------------------------------------------------
    # Top interactions
    # ------------------------------------------------------------------

    def get_top_interactions(self, k: int = 5) -> list[tuple[int, int, float]]:
        """Return top-k interaction pairs sorted by interaction strength.

        Parameters
        ----------
        k : int
            Number of top interactions to return.

        Returns
        -------
        list[tuple[int, int, float]]
            List of ``(feature_i, feature_j, strength)`` tuples.
        """
        interactions = self.compute_interactions()
        ranked = sorted(interactions.items(), key=lambda t: t[1], reverse=True)
        return [(i, j, v) for (i, j), v in ranked[:k]]

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def render_heatmap(self, feature_names: list[str] | None = None) -> str:
        """Render interaction matrix as an SVG heatmap.

        Parameters
        ----------
        feature_names : list[str] | None
            Optional human-readable feature names.

        Returns
        -------
        str
            SVG XML string.
        """
        if self._n_features == 0:
            canvas = SVGCanvas(width=200, height=100, background="#ffffff")
            canvas.text(100, 50, "No features", font_size=14, fill="#666",
                        text_anchor="middle")
            return canvas.to_string()

        n = self._n_features
        names = feature_names if feature_names and len(feature_names) == n else [
            f"x{i}" for i in range(n)
        ]

        interactions = self.compute_interactions()
        main = self.compute_main_effects()

        # Build full matrix including main effects on diagonal
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = main.get(i, 0.0)
        for (i, j), v in interactions.items():
            matrix[i][j] = v
            matrix[j][i] = v

        # SVG layout
        cell_size = 50
        label_margin = 80
        width = label_margin + n * cell_size + 20
        height = label_margin + n * cell_size + 20

        canvas = SVGCanvas(width=int(width), height=int(height), background="#ffffff")

        # Title
        canvas.text(width / 2, 15, "Feature Interaction Heatmap",
                    font_size=14, fill="#333", text_anchor="middle")

        # Find max for colour scaling
        max_val = max(
            (matrix[i][j] for i in range(n) for j in range(n)),
            default=1.0,
        )
        if max_val < 1e-15:
            max_val = 1.0

        # Draw cells
        for i in range(n):
            for j in range(n):
                x = label_margin + j * cell_size
                y = label_margin + i * cell_size - 40
                intensity = matrix[i][j] / max_val
                r = int(255 * (1.0 - intensity))
                g = int(255 * (1.0 - 0.5 * intensity))
                b = 255
                fill_colour = f"rgb({r},{g},{b})"
                canvas.rect(x, y, cell_size, cell_size,
                            fill=fill_colour, stroke="#ccc")
                canvas.text(x + cell_size / 2, y + cell_size / 2 + 4,
                            f"{matrix[i][j]:.2f}",
                            font_size=9, fill="#333", text_anchor="middle")

        # Row and column labels
        for i in range(n):
            # Row label
            canvas.text(label_margin - 5,
                        label_margin + i * cell_size + cell_size / 2 - 36,
                        names[i], font_size=10, fill="#333", text_anchor="end")
            # Column label
            canvas.text(label_margin + i * cell_size + cell_size / 2,
                        label_margin - 45,
                        names[i], font_size=10, fill="#333", text_anchor="middle")

        return canvas.to_string()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    """Compute the arithmetic mean of a list of floats."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    """Compute the population variance of a list of floats."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def _variance_reduction(
    parent_y: list[float],
    left_y: list[float],
    right_y: list[float],
) -> float:
    """Compute variance reduction from a binary split."""
    n = len(parent_y)
    if n == 0:
        return 0.0
    parent_var = _variance(parent_y)
    left_var = _variance(left_y) * len(left_y) / n
    right_var = _variance(right_y) * len(right_y) / n
    return parent_var - left_var - right_var

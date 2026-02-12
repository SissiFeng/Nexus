"""Hypothesis generation from multiple analysis sources.

Generates competing hypotheses from symbolic regression Pareto fronts,
causal graph structures, fANOVA importance decompositions, and simple
correlation analysis -- all using plain Python data structures.
"""

from __future__ import annotations

import math
import random
from collections import deque

from optimization_copilot.hypothesis.models import Hypothesis


class HypothesisGenerator:
    """Generate competing hypotheses from multiple analysis sources.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._counter = 0
        self._seed = seed
        self._rng = random.Random(seed)

    def _next_id(self) -> str:
        self._counter += 1
        return f"H{self._counter:04d}"

    # -- From symbolic regression Pareto front ---------------------------------

    def from_symreg(
        self,
        pareto_front: list[dict],
        var_names: list[str] | None = None,
    ) -> list[Hypothesis]:
        """Each Pareto-optimal equation becomes a hypothesis.

        Parameters
        ----------
        pareto_front : list[dict]
            Each entry has ``"equation"`` (str), ``"complexity"`` (int),
            and optionally ``"r_squared"`` (float).
        var_names : list[str] | None
            Variable names for human-readable descriptions.
        """
        hypotheses: list[Hypothesis] = []
        for entry in pareto_front:
            eq = entry["equation"]
            complexity = entry.get("complexity", 1)
            r2 = entry.get("r_squared", None)
            desc = f"Symbolic regression model: y = {eq}"
            if r2 is not None:
                desc += f" (R²={r2:.4f})"
            h = Hypothesis(
                id=self._next_id(),
                description=desc,
                equation=eq,
                source="symreg",
                n_parameters=complexity,
            )
            if r2 is not None:
                # Store R² in bic_score field as a proxy until BIC is computed
                h.bic_score = None
            hypotheses.append(h)
        return hypotheses

    # -- From causal graph -----------------------------------------------------

    def from_causal_graph(
        self, graph_dict: dict, target: str
    ) -> list[Hypothesis]:
        """Each causal path to *target* becomes a mechanism hypothesis.

        Parameters
        ----------
        graph_dict : dict
            Serialized ``CausalGraph`` (from ``graph.to_dict()``).
            Expected keys: ``"nodes"`` (dict), ``"edges"`` (list of dicts
            with ``"source"`` and ``"target"``).
        target : str
            Name of the target variable.
        """
        # Build adjacency: parent -> children
        children_map: dict[str, list[str]] = {}
        all_nodes: set[str] = set()
        for name in graph_dict.get("nodes", {}):
            all_nodes.add(name)
            children_map.setdefault(name, [])
        for edge in graph_dict.get("edges", []):
            src, tgt = edge["source"], edge["target"]
            children_map.setdefault(src, []).append(tgt)
            all_nodes.add(src)
            all_nodes.add(tgt)

        if target not in all_nodes:
            return []

        # Find all paths ending at target using BFS from each non-target node
        hypotheses: list[Hypothesis] = []
        for start in sorted(all_nodes):
            if start == target:
                continue
            # BFS to find paths from start to target
            paths = self._find_paths(children_map, start, target)
            for path in paths:
                mechanism = " -> ".join(path)
                desc = (
                    f"{start} causes {target} through path {mechanism}"
                )
                h = Hypothesis(
                    id=self._next_id(),
                    description=desc,
                    causal_mechanism=mechanism,
                    source="causal",
                    n_parameters=len(path) - 1,
                )
                hypotheses.append(h)
        return hypotheses

    @staticmethod
    def _find_paths(
        children_map: dict[str, list[str]],
        start: str,
        target: str,
        max_depth: int = 10,
    ) -> list[list[str]]:
        """Find all simple paths from *start* to *target* (BFS)."""
        paths: list[list[str]] = []
        queue: deque[list[str]] = deque([[start]])
        while queue:
            path = queue.popleft()
            if len(path) > max_depth:
                continue
            current = path[-1]
            if current == target and len(path) > 1:
                paths.append(path)
                continue
            for child in children_map.get(current, []):
                if child not in path:  # avoid cycles
                    queue.append(path + [child])
        return paths

    # -- From fANOVA -----------------------------------------------------------

    def from_fanova(
        self,
        main_effects: dict[str, float],
        interactions: list[dict] | None = None,
        threshold: float = 0.1,
    ) -> list[Hypothesis]:
        """Important features and interactions become hypotheses.

        Parameters
        ----------
        main_effects : dict[str, float]
            ``{var_name: importance_score}``.
        interactions : list[dict] | None
            Each entry has ``"vars"`` (list of two str) and
            ``"importance"`` (float).
        threshold : float
            Minimum importance to generate a hypothesis.
        """
        hypotheses: list[Hypothesis] = []
        for var, imp in sorted(main_effects.items(), key=lambda x: -x[1]):
            if imp >= threshold:
                h = Hypothesis(
                    id=self._next_id(),
                    description=(
                        f"{var} is a significant driver "
                        f"(importance={imp:.4f})"
                    ),
                    source="fanova",
                    n_parameters=1,
                )
                hypotheses.append(h)

        if interactions:
            for inter in sorted(
                interactions, key=lambda x: -x["importance"]
            ):
                if inter["importance"] >= threshold:
                    v1, v2 = inter["vars"]
                    h = Hypothesis(
                        id=self._next_id(),
                        description=(
                            f"Interaction between {v1} and {v2} is "
                            f"significant (importance={inter['importance']:.4f})"
                        ),
                        equation=f"{v1} * {v2}",
                        source="fanova",
                        n_parameters=2,
                    )
                    hypotheses.append(h)
        return hypotheses

    # -- From correlations -----------------------------------------------------

    def from_correlation(
        self,
        correlations: dict[str, float],
        threshold: float = 0.3,
    ) -> list[Hypothesis]:
        """Strong correlations become candidate hypotheses.

        Parameters
        ----------
        correlations : dict[str, float]
            ``{var_name: correlation_with_target}``.
        threshold : float
            Minimum absolute correlation to generate a hypothesis.
        """
        hypotheses: list[Hypothesis] = []
        for var, corr in sorted(
            correlations.items(), key=lambda x: -abs(x[1])
        ):
            if abs(corr) >= threshold:
                direction = "positively" if corr > 0 else "negatively"
                h = Hypothesis(
                    id=self._next_id(),
                    description=(
                        f"{var} is {direction} correlated with target "
                        f"(r={corr:.4f})"
                    ),
                    equation=f"{corr:.4f} * {var}",
                    source="correlation",
                    n_parameters=1,
                )
                hypotheses.append(h)
        return hypotheses

    # -- Lightweight auto-generation -------------------------------------------

    def generate_competing(
        self,
        data: list[list[float]],
        var_names: list[str],
        target_index: int = -1,
    ) -> list[Hypothesis]:
        """Generate competing hypotheses using lightweight analysis.

        Computes simple variance-based importance and pairwise correlations
        from the raw data (no heavy dependencies).

        Parameters
        ----------
        data : list[list[float]]
            Row-major dataset.
        var_names : list[str]
            Column names.
        target_index : int
            Index of the target column (default: last).
        """
        if not data or not data[0]:
            return []

        n = len(data)
        p = len(data[0])
        tidx = target_index if target_index >= 0 else p + target_index

        # Extract target
        y = [row[tidx] for row in data]
        y_mean = sum(y) / n
        y_var = sum((v - y_mean) ** 2 for v in y) / n
        if y_var == 0:
            return []

        # Feature indices excluding target
        feat_indices = [i for i in range(p) if i != tidx]
        feat_names = [var_names[i] for i in feat_indices]

        # Compute correlations
        correlations: dict[str, float] = {}
        for fi, fname in zip(feat_indices, feat_names):
            x_col = [row[fi] for row in data]
            x_mean = sum(x_col) / n
            x_var = sum((v - x_mean) ** 2 for v in x_col) / n
            if x_var == 0:
                correlations[fname] = 0.0
                continue
            cov = sum((x_col[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / n
            correlations[fname] = cov / math.sqrt(x_var * y_var)

        # Compute simple variance-based importance (squared correlation)
        main_effects: dict[str, float] = {
            name: corr ** 2 for name, corr in correlations.items()
        }

        # Generate hypotheses from both sources
        hypotheses: list[Hypothesis] = []
        hypotheses.extend(self.from_correlation(correlations, threshold=0.3))
        hypotheses.extend(self.from_fanova(main_effects, threshold=0.1))

        return hypotheses

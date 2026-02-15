"""PC algorithm for causal structure learning.

Learns a causal DAG from observational data using the PC (Peter-Clark)
constraint-based algorithm.  All computations use pure Python standard
library operations plus the project's internal ``_math`` helpers.
"""

from __future__ import annotations

import math
from itertools import combinations

from optimization_copilot.backends._math.linalg import mat_inv
from optimization_copilot.backends._math.stats import norm_cdf
from optimization_copilot.causal.models import CausalEdge, CausalGraph, CausalNode


class CausalStructureLearner:
    """PC algorithm for causal structure learning from observational data.

    Parameters
    ----------
    alpha : float
        Significance level for the conditional independence test.
    max_cond_set : int
        Maximum size of the conditioning set to consider.
    """

    def __init__(self, alpha: float = 0.05, max_cond_set: int = 3) -> None:
        self.alpha = alpha
        self.max_cond_set = max_cond_set

    # -- Public API -------------------------------------------------------------

    def learn(
        self,
        data: list[list[float]],
        var_names: list[str],
    ) -> CausalGraph:
        """Learn a causal graph from observational data using the PC algorithm.

        Parameters
        ----------
        data : list[list[float]]
            Observations as a list of rows, each row containing one value per
            variable.  Shape ``(n_samples, n_vars)``.
        var_names : list[str]
            Variable names corresponding to the columns of *data*.

        Returns
        -------
        CausalGraph
            The learned causal graph with oriented edges.
        """
        n_vars = len(var_names)
        n_samples = len(data)

        # Step 0: Compute correlation matrix and precision matrix
        corr_mat = self._correlation_matrix(data)

        # Regularize before inversion for numerical stability
        reg = 1e-6
        reg_corr = [
            [corr_mat[i][j] + (reg if i == j else 0.0) for j in range(n_vars)]
            for i in range(n_vars)
        ]
        precision_mat = mat_inv(reg_corr)

        # Step 1: Start with complete undirected graph (adjacency sets)
        adj: dict[int, set[int]] = {i: set() for i in range(n_vars)}
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                adj[i].add(j)
                adj[j].add(i)

        # Separation sets: record which conditioning set made (i,j) independent
        sep_set: dict[tuple[int, int], set[int]] = {}

        # Step 2: Skeleton discovery via conditional independence testing
        for cond_size in range(self.max_cond_set + 1):
            # Iterate over all current edges
            pairs_to_check: list[tuple[int, int]] = []
            for i in range(n_vars):
                for j in sorted(adj[i]):
                    if i < j:
                        pairs_to_check.append((i, j))

            for i, j in pairs_to_check:
                if j not in adj[i]:
                    continue  # already removed

                # Neighbors of i (excluding j) for conditioning
                neighbors_i = sorted(adj[i] - {j})

                if len(neighbors_i) < cond_size:
                    continue

                found_independent = False
                for subset in combinations(neighbors_i, cond_size):
                    cond_set = set(subset)

                    # Compute partial correlation
                    pcorr = self._partial_correlation(
                        precision_mat, i, j, cond_set, corr_mat, n_vars,
                    )

                    # Fisher z-test
                    p_value = self._fisher_z_test(pcorr, n_samples, len(cond_set))

                    if p_value > self.alpha:
                        # Conditionally independent: remove edge
                        adj[i].discard(j)
                        adj[j].discard(i)
                        sep_set[(i, j)] = cond_set
                        sep_set[(j, i)] = cond_set
                        found_independent = True
                        break

                if found_independent:
                    continue

                # Also try neighbors of j
                neighbors_j = sorted(adj[j] - {i})
                if len(neighbors_j) < cond_size:
                    continue

                for subset in combinations(neighbors_j, cond_size):
                    cond_set = set(subset)
                    pcorr = self._partial_correlation(
                        precision_mat, i, j, cond_set, corr_mat, n_vars,
                    )
                    p_value = self._fisher_z_test(pcorr, n_samples, len(cond_set))

                    if p_value > self.alpha:
                        adj[i].discard(j)
                        adj[j].discard(i)
                        sep_set[(i, j)] = cond_set
                        sep_set[(j, i)] = cond_set
                        break

        # Step 3: Orient v-structures (colliders)
        # directed[i][j] = True means edge is oriented i -> j
        directed: dict[int, set[int]] = {i: set() for i in range(n_vars)}

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                for k in range(n_vars):
                    if k == i or k == j:
                        continue
                    # Check for X - Z - Y pattern where X and Y are not adjacent
                    if (
                        j in adj[i]
                        and j in adj[k]
                        and k not in adj[i]
                    ):
                        # X-Z-Y unshielded triple, X and Y not adjacent
                        sep = sep_set.get((i, k), sep_set.get((k, i), set()))
                        if j not in sep:
                            # Z is not in sep set -> collider: X -> Z <- Y
                            directed[i].add(j)
                            directed[k].add(j)

        # Step 4: Apply Meek rules to orient remaining edges
        changed = True
        while changed:
            changed = False

            for i in range(n_vars):
                for j in sorted(adj[i]):
                    if j in directed[i] or i in directed[j]:
                        continue  # already oriented

                    # Rule R1: If there exists k such that k -> i and k not adj j
                    # then orient i -> j
                    for k in range(n_vars):
                        if k == i or k == j:
                            continue
                        if i in directed[k] and j not in adj[k]:
                            directed[i].add(j)
                            changed = True
                            break

                    if j in directed[i]:
                        continue

                    # Rule R2: If there is a directed path i -> k -> j
                    # then orient i -> j
                    for k in range(n_vars):
                        if k == i or k == j:
                            continue
                        if k in directed[i] and j in directed[k]:
                            directed[i].add(j)
                            changed = True
                            break

                    if j in directed[i]:
                        continue

                    # Rule R3: If there exist k, l such that
                    # k - i, l - i, k -> j, l -> j, k not adj l
                    # then orient i -> j
                    neighbors_of_i = [
                        n for n in adj[i]
                        if n != j and n not in directed[i] and i not in directed[n]
                    ]
                    for idx_k, k in enumerate(neighbors_of_i):
                        for l in neighbors_of_i[idx_k + 1:]:  # noqa: E741
                            if (
                                j in directed[k]
                                and j in directed[l]
                                and l not in adj[k]
                            ):
                                directed[i].add(j)
                                changed = True
                                break
                        if j in directed[i]:
                            break

        # Build the CausalGraph
        graph = CausalGraph()
        for idx, name in enumerate(var_names):
            graph.add_node(CausalNode(name=name))

        # Add directed edges
        added_edges: set[tuple[int, int]] = set()
        for i in range(n_vars):
            for j in sorted(directed[i]):
                if (i, j) not in added_edges:
                    graph.add_edge(CausalEdge(
                        source=var_names[i],
                        target=var_names[j],
                    ))
                    added_edges.add((i, j))

        # Remaining undirected edges: orient arbitrarily (lower index -> higher)
        for i in range(n_vars):
            for j in sorted(adj[i]):
                if i < j and (i, j) not in added_edges and (j, i) not in added_edges:
                    if j not in directed[i] and i not in directed[j]:
                        graph.add_edge(CausalEdge(
                            source=var_names[i],
                            target=var_names[j],
                        ))
                        added_edges.add((i, j))

        return graph

    # -- Internal helpers -------------------------------------------------------

    def _correlation_matrix(self, data: list[list[float]]) -> list[list[float]]:
        """Compute the Pearson correlation matrix from columnar data.

        Parameters
        ----------
        data : list[list[float]]
            Rows of observations.

        Returns
        -------
        list[list[float]]
            Symmetric correlation matrix of shape ``(p, p)``.
        """
        n = len(data)
        if n == 0:
            return []
        p = len(data[0])

        # Compute means
        means = [0.0] * p
        for row in data:
            for j in range(p):
                means[j] += row[j]
        means = [m / n for m in means]

        # Compute std devs
        stds = [0.0] * p
        for row in data:
            for j in range(p):
                stds[j] += (row[j] - means[j]) ** 2
        stds = [math.sqrt(s / n) for s in stds]

        # Compute correlation matrix
        corr = [[0.0] * p for _ in range(p)]
        for i in range(p):
            corr[i][i] = 1.0
            for j in range(i + 1, p):
                if stds[i] < 1e-12 or stds[j] < 1e-12:
                    corr[i][j] = 0.0
                    corr[j][i] = 0.0
                    continue
                cov = 0.0
                for row in data:
                    cov += (row[i] - means[i]) * (row[j] - means[j])
                cov /= n
                r = cov / (stds[i] * stds[j])
                corr[i][j] = r
                corr[j][i] = r

        return corr

    def _partial_correlation(
        self,
        precision_mat: list[list[float]],
        i: int,
        j: int,
        cond_set: set[int],
        corr_mat: list[list[float]],
        n_vars: int,
    ) -> float:
        """Compute partial correlation between variables i and j.

        For empty conditioning set, returns the simple correlation.
        For non-empty conditioning sets, uses the precision matrix approach:
        ``partial_corr(i, j) = -P[i][j] / sqrt(P[i][i] * P[j][j])``.

        For conditioning on a subset, we invert the relevant sub-matrix
        of the correlation matrix.
        """
        if not cond_set:
            return corr_mat[i][j]

        # Build the sub-matrix for {i, j} union cond_set
        indices = sorted({i, j} | cond_set)
        k = len(indices)
        sub_corr = [[0.0] * k for _ in range(k)]
        for ri, ii in enumerate(indices):
            for ci, jj in enumerate(indices):
                sub_corr[ri][ci] = corr_mat[ii][jj]

        # Regularize for stability
        reg = 1e-6
        for ri in range(k):
            sub_corr[ri][ri] += reg

        try:
            sub_precision = mat_inv(sub_corr)
        except Exception:
            return 0.0

        # Find the positions of i and j in the sub-matrix
        idx_i = indices.index(i)
        idx_j = indices.index(j)

        denom = sub_precision[idx_i][idx_i] * sub_precision[idx_j][idx_j]
        if denom <= 0:
            return 0.0

        return -sub_precision[idx_i][idx_j] / math.sqrt(denom)

    def _fisher_z_test(self, r: float, n: int, cond_size: int) -> float:
        """Fisher z-transform test for partial correlation.

        Parameters
        ----------
        r : float
            Partial correlation coefficient.
        n : int
            Number of observations.
        cond_size : int
            Size of the conditioning set.

        Returns
        -------
        float
            Two-sided p-value from the standard normal distribution.
        """
        # Clamp r to avoid log(0)
        r = max(-0.9999, min(0.9999, r))

        # Fisher z-transform
        z = 0.5 * math.log((1.0 + r) / (1.0 - r))

        # Degrees of freedom adjustment
        df = n - cond_size - 3
        if df < 1:
            return 1.0  # not enough data

        # Test statistic
        test_stat = abs(z) * math.sqrt(df)

        # Two-sided p-value from standard normal
        p_value = 2.0 * (1.0 - norm_cdf(test_stat))
        return p_value

"""Built-in optimization backends using only the Python standard library."""

from __future__ import annotations

import math
import random
from typing import Any

from optimization_copilot.core.models import Observation, ParameterSpec, VariableType
from optimization_copilot.plugins.base import AlgorithmPlugin
from optimization_copilot.backends._math.linalg import (
    vec_dot as _vec_dot,
    mat_mul as _mat_mul,
    mat_vec as _mat_vec,
    cholesky as _cholesky,
    solve_lower as _solve_lower,
    solve_upper as _solve_upper,
    solve_cholesky as _solve_cholesky,
    transpose as _transpose,
    identity as _identity,
)
from optimization_copilot.backends._math.stats import (
    norm_pdf as _norm_pdf,
    norm_cdf as _norm_cdf,
)
from optimization_copilot.backends._math.sobol import (
    sobol_sequence as _sobol_sequence,
    SOBOL_DIRECTION_NUMBERS as _SOBOL_DIRECTION_NUMBERS,
)


# ── helpers ───────────────────────────────────────────────────────────

def _sample_param(spec: ParameterSpec, rng: random.Random) -> Any:
    """Draw one random value for *spec* using the given RNG."""
    if spec.type == VariableType.CATEGORICAL:
        return rng.choice(spec.categories)
    if spec.type == VariableType.DISCRETE:
        return rng.randint(int(spec.lower), int(spec.upper))
    # CONTINUOUS (and fallback)
    return rng.uniform(spec.lower, spec.upper)


def _clamp(value: float, spec: ParameterSpec) -> Any:
    """Clamp *value* to the bounds of *spec*."""
    if spec.type == VariableType.CATEGORICAL:
        return value  # no numeric bounds
    if spec.type == VariableType.DISCRETE:
        return max(int(spec.lower), min(int(spec.upper), int(round(value))))
    return max(spec.lower, min(spec.upper, value))


# ── RandomSampler ─────────────────────────────────────────────────────

class RandomSampler(AlgorithmPlugin):
    """Uniform random sampling within parameter bounds.

    Useful as a baseline and during cold-start phases when no prior
    observations are available.
    """

    def __init__(self) -> None:
        self._specs: list[ParameterSpec] = []

    def name(self) -> str:
        return "random_sampler"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = list(parameter_specs)

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        suggestions: list[dict[str, Any]] = []
        for _ in range(n_suggestions):
            point = {spec.name: _sample_param(spec, rng) for spec in self._specs}
            suggestions.append(point)
        return suggestions

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": False,
            "max_dimensions": None,
        }


# ── LatinHypercubeSampler ─────────────────────────────────────────────

class LatinHypercubeSampler(AlgorithmPlugin):
    """Latin Hypercube Sampling (LHS) for space-filling experimental design.

    Divides each dimension into *n* equal strata and places exactly one
    sample in each stratum.  Strata assignments are shuffled independently
    per dimension.  For categorical parameters each category is assigned
    to strata as evenly as possible.
    """

    def __init__(self) -> None:
        self._specs: list[ParameterSpec] = []

    def name(self) -> str:
        return "latin_hypercube_sampler"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = list(parameter_specs)

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        n = n_suggestions

        # For each spec, build a column of n values (one per stratum).
        columns: dict[str, list[Any]] = {}
        for spec in self._specs:
            if spec.type == VariableType.CATEGORICAL:
                # Distribute categories across strata as evenly as possible.
                cats = spec.categories
                col = [cats[i % len(cats)] for i in range(n)]
                rng.shuffle(col)
            elif spec.type == VariableType.DISCRETE:
                lo, hi = int(spec.lower), int(spec.upper)
                # Strata boundaries in continuous space, then round.
                width = (hi - lo + 1) / n
                col: list[Any] = []
                indices = list(range(n))
                rng.shuffle(indices)
                for idx in indices:
                    low_edge = lo + idx * width
                    high_edge = lo + (idx + 1) * width
                    val = rng.uniform(low_edge, high_edge)
                    col.append(max(lo, min(hi, int(round(val)))))
                columns[spec.name] = col
                continue
            else:
                # Continuous: divide [lower, upper] into n equal strata.
                lo, hi = spec.lower, spec.upper
                width = (hi - lo) / n
                col = []
                indices = list(range(n))
                rng.shuffle(indices)
                for idx in indices:
                    low_edge = lo + idx * width
                    high_edge = lo + (idx + 1) * width
                    col.append(rng.uniform(low_edge, high_edge))
            columns[spec.name] = col

        # Assemble per-sample dicts.
        return [
            {spec.name: columns[spec.name][i] for spec in self._specs}
            for i in range(n)
        ]

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": False,
            "max_dimensions": None,
        }


# ── TPESampler ────────────────────────────────────────────────────────

class TPESampler(AlgorithmPlugin):
    """Simplified Tree-structured Parzen Estimator (TPE).

    Splits historical observations into *good* (top percentile) and *bad*
    groups, then samples new candidates from the good region with small
    jitter.  Falls back to uniform random when insufficient observations
    are available.
    """

    def __init__(self, gamma: float = 0.25) -> None:
        self._specs: list[ParameterSpec] = []
        self._observations: list[Observation] = []
        self._gamma = gamma  # fraction considered "good"

    def name(self) -> str:
        return "tpe_sampler"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = list(parameter_specs)
        self._observations = [o for o in observations if not o.is_failure]

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)

        # Not enough observations — fall back to uniform random.
        if len(self._observations) < 4:
            return [
                {s.name: _sample_param(s, rng) for s in self._specs}
                for _ in range(n_suggestions)
            ]

        # Rank observations by their first KPI value (lower is better for
        # a simplified implementation; the meta-controller handles direction).
        sorted_obs = sorted(
            self._observations,
            key=lambda o: list(o.kpi_values.values())[0],
        )
        n_good = max(1, int(len(sorted_obs) * self._gamma))
        good_obs = sorted_obs[:n_good]

        suggestions: list[dict[str, Any]] = []
        for _ in range(n_suggestions):
            # Pick a random observation from the good set and jitter it.
            base = rng.choice(good_obs)
            point: dict[str, Any] = {}
            for spec in self._specs:
                base_val = base.parameters.get(spec.name)
                if spec.type == VariableType.CATEGORICAL:
                    # With small probability, explore a different category.
                    if rng.random() < 0.2:
                        point[spec.name] = rng.choice(spec.categories)
                    else:
                        point[spec.name] = base_val
                elif spec.type == VariableType.DISCRETE:
                    lo, hi = int(spec.lower), int(spec.upper)
                    spread = max(1.0, (hi - lo) * 0.1)
                    jittered = base_val + rng.gauss(0, spread)
                    point[spec.name] = max(lo, min(hi, int(round(jittered))))
                else:
                    # Continuous: Gaussian jitter proportional to range.
                    lo, hi = spec.lower, spec.upper
                    spread = (hi - lo) * 0.1
                    jittered = base_val + rng.gauss(0, spread)
                    point[spec.name] = max(lo, min(hi, jittered))
            suggestions.append(point)
        return suggestions

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": True,
            "max_dimensions": None,
        }


# ── SobolSampler ─────────────────────────────────────────────────────

class SobolSampler(AlgorithmPlugin):
    """Quasi-random Sobol sequence sampler for space-filling designs.

    Uses bit manipulation with gray-code ordering for efficient
    low-discrepancy sequence generation.  Supports up to 21 dimensions.
    For categorical parameters, categories are mapped to uniform intervals.
    """

    def __init__(self) -> None:
        self._specs: list[ParameterSpec] = []

    def name(self) -> str:
        return "sobol_sampler"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = list(parameter_specs)

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        n_dims = len(self._specs)
        if n_dims == 0:
            return [{}] * n_suggestions

        # Generate Sobol points; skip `seed` points to vary the sequence
        raw = _sobol_sequence(n_suggestions + seed, n_dims)
        raw = raw[seed:]  # skip first `seed` points for variety

        suggestions: list[dict[str, Any]] = []
        for pt in raw:
            point: dict[str, Any] = {}
            for d, spec in enumerate(self._specs):
                u = pt[d] if d < len(pt) else 0.5
                if spec.type == VariableType.CATEGORICAL:
                    idx = int(u * len(spec.categories))
                    idx = min(idx, len(spec.categories) - 1)
                    point[spec.name] = spec.categories[idx]
                elif spec.type == VariableType.DISCRETE:
                    lo, hi = int(spec.lower), int(spec.upper)
                    point[spec.name] = max(lo, min(hi, int(round(lo + u * (hi - lo)))))
                else:
                    point[spec.name] = spec.lower + u * (spec.upper - spec.lower)
            suggestions.append(point)
        return suggestions

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": False,
            "max_dimensions": 21,
        }


# ── GaussianProcessBO ────────────────────────────────────────────────

class GaussianProcessBO(AlgorithmPlugin):
    """Gaussian Process Bayesian Optimization with Expected Improvement.

    Implements a simple GP surrogate with RBF kernel, Cholesky-based
    posterior inference, and EI acquisition.  Falls back to random
    sampling when fewer than 3 observations are available or when
    categorical parameters are present.
    """

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-4) -> None:
        self._specs: list[ParameterSpec] = []
        self._observations: list[Observation] = []
        self._length_scale = length_scale
        self._noise = noise

    def name(self) -> str:
        return "gaussian_process_bo"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = list(parameter_specs)
        self._observations = [o for o in observations if not o.is_failure]

    def _has_categorical(self) -> bool:
        return any(s.type == VariableType.CATEGORICAL for s in self._specs)

    def _to_vec(self, params: dict[str, Any]) -> list[float]:
        """Convert parameter dict to numeric vector (skip categoricals)."""
        vec: list[float] = []
        for s in self._specs:
            if s.type == VariableType.CATEGORICAL:
                continue
            vec.append(float(params.get(s.name, 0.0)))
        return vec

    def _numeric_specs(self) -> list[ParameterSpec]:
        return [s for s in self._specs if s.type != VariableType.CATEGORICAL]

    def _rbf_kernel(self, x1: list[float], x2: list[float]) -> float:
        ls2 = self._length_scale ** 2
        sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return math.exp(-0.5 * sq_dist / ls2)

    def _build_kernel_matrix(self, X: list[list[float]]) -> list[list[float]]:
        n = len(X)
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                k = self._rbf_kernel(X[i], X[j])
                if i == j:
                    k += self._noise
                K[i][j] = k
                K[j][i] = k
        return K

    def _expected_improvement(self, mu: float, sigma: float, best_y: float) -> float:
        if sigma < 1e-12:
            return 0.0
        z = (best_y - mu) / sigma
        return (best_y - mu) * _norm_cdf(z) + sigma * _norm_pdf(z)

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)

        # Fallback: not enough observations or has categorical
        if len(self._observations) < 3 or self._has_categorical():
            return [
                {s.name: _sample_param(s, rng) for s in self._specs}
                for _ in range(n_suggestions)
            ]

        # Build GP from observations
        X_train = [self._to_vec(o.parameters) for o in self._observations]
        y_train = [list(o.kpi_values.values())[0] for o in self._observations]
        best_y = min(y_train)

        # Auto-scale length scale
        n_dims = len(X_train[0]) if X_train else 1
        ranges = []
        for d in range(n_dims):
            vals = [x[d] for x in X_train]
            r = max(vals) - min(vals)
            ranges.append(r if r > 0 else 1.0)
        self._length_scale = sum(ranges) / len(ranges) * 0.5

        K = self._build_kernel_matrix(X_train)
        L = _cholesky(K)
        alpha = _solve_cholesky(L, y_train)

        numeric_specs = self._numeric_specs()
        cat_specs = [s for s in self._specs if s.type == VariableType.CATEGORICAL]

        # Generate candidates via random sampling and pick best EI
        n_candidates = max(100, 20 * n_dims)
        suggestions: list[dict[str, Any]] = []

        for _ in range(n_suggestions):
            best_ei = -1.0
            best_point: dict[str, Any] = {}

            for _ in range(n_candidates):
                # Random candidate
                cand_vec: list[float] = []
                cand_point: dict[str, Any] = {}
                for s in self._specs:
                    if s.type == VariableType.CATEGORICAL:
                        cand_point[s.name] = rng.choice(s.categories)
                    else:
                        val = _sample_param(s, rng)
                        cand_point[s.name] = val
                        cand_vec.append(float(val))

                # GP posterior
                k_star = [self._rbf_kernel(cand_vec, xt) for xt in X_train]
                mu = _vec_dot(k_star, alpha)
                v = _solve_lower(L, k_star)
                k_ss = self._rbf_kernel(cand_vec, cand_vec)
                sigma2 = max(k_ss - _vec_dot(v, v), 1e-12)
                sigma = math.sqrt(sigma2)

                ei = self._expected_improvement(mu, sigma, best_y)
                if ei > best_ei:
                    best_ei = ei
                    best_point = cand_point

            suggestions.append(best_point)

        return suggestions

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": False,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": True,
            "max_dimensions": None,
        }


# ── RandomForestBO ───────────────────────────────────────────────────

class _DecisionTree:
    """Minimal decision tree for regression (axis-aligned splits)."""

    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._tree: dict[str, Any] | None = None

    def fit(self, X: list[list[float]], y: list[float], rng: random.Random) -> None:
        self._tree = self._build(X, y, depth=0, rng=rng)

    def _build(
        self,
        X: list[list[float]],
        y: list[float],
        depth: int,
        rng: random.Random,
    ) -> dict[str, Any]:
        n = len(y)
        if n <= self.min_samples_leaf or depth >= self.max_depth:
            return {"leaf": True, "value": sum(y) / max(n, 1)}

        n_features = len(X[0]) if X else 0
        # Subsample features (sqrt(n_features))
        n_try = max(1, int(math.sqrt(n_features)))
        features = rng.sample(range(n_features), min(n_try, n_features))

        best_score = float("inf")
        best_feat = 0
        best_thresh = 0.0

        for feat in features:
            vals = sorted(set(row[feat] for row in X))
            if len(vals) < 2:
                continue
            # Try a few threshold candidates
            thresholds = []
            if len(vals) <= 10:
                thresholds = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
            else:
                indices = rng.sample(range(len(vals) - 1), min(10, len(vals) - 1))
                thresholds = [(vals[i] + vals[i + 1]) / 2 for i in indices]

            for thresh in thresholds:
                left_y = [y[i] for i in range(n) if X[i][feat] <= thresh]
                right_y = [y[i] for i in range(n) if X[i][feat] > thresh]
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                score = self._variance(left_y) * len(left_y) + self._variance(right_y) * len(right_y)
                if score < best_score:
                    best_score = score
                    best_feat = feat
                    best_thresh = thresh

        if best_score == float("inf"):
            return {"leaf": True, "value": sum(y) / max(n, 1)}

        left_idx = [i for i in range(n) if X[i][best_feat] <= best_thresh]
        right_idx = [i for i in range(n) if X[i][best_feat] > best_thresh]

        return {
            "leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "left": self._build([X[i] for i in left_idx], [y[i] for i in left_idx], depth + 1, rng),
            "right": self._build([X[i] for i in right_idx], [y[i] for i in right_idx], depth + 1, rng),
        }

    @staticmethod
    def _variance(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        return sum((v - mean) ** 2 for v in vals) / len(vals)

    def predict(self, x: list[float]) -> float:
        node = self._tree
        while node and not node["leaf"]:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"] if node else 0.0


class RandomForestBO(AlgorithmPlugin):
    """Random Forest surrogate for Bayesian Optimization (SMAC-style).

    Builds an ensemble of decision trees, uses ensemble mean for
    predicted objective and ensemble variance (disagreement) as
    uncertainty.  Acquisition is EI-like based on ensemble statistics.
    Works with all variable types including categorical (one-hot encoded).
    """

    def __init__(self, n_trees: int = 10, max_depth: int = 5) -> None:
        self._specs: list[ParameterSpec] = []
        self._observations: list[Observation] = []
        self._n_trees = n_trees
        self._max_depth = max_depth
        self._forest: list[_DecisionTree] = []
        self._X_train: list[list[float]] = []
        self._y_train: list[float] = []

    def name(self) -> str:
        return "random_forest_bo"

    def _encode(self, params: dict[str, Any]) -> list[float]:
        """Encode parameters to numeric vector.  Categorical -> one-hot."""
        vec: list[float] = []
        for s in self._specs:
            val = params.get(s.name)
            if s.type == VariableType.CATEGORICAL:
                for cat in s.categories:
                    vec.append(1.0 if val == cat else 0.0)
            elif s.type == VariableType.DISCRETE:
                vec.append(float(val))
            else:
                vec.append(float(val))
        return vec

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = list(parameter_specs)
        self._observations = [o for o in observations if not o.is_failure]
        if len(self._observations) >= 3:
            self._X_train = [self._encode(o.parameters) for o in self._observations]
            self._y_train = [list(o.kpi_values.values())[0] for o in self._observations]

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)

        # Fallback
        if len(self._observations) < 3:
            return [
                {s.name: _sample_param(s, rng) for s in self._specs}
                for _ in range(n_suggestions)
            ]

        # Build forest with bootstrap
        self._forest = []
        for t in range(self._n_trees):
            tree_rng = random.Random(seed + t + 1)
            n = len(self._X_train)
            indices = [tree_rng.randint(0, n - 1) for _ in range(n)]
            X_boot = [self._X_train[i] for i in indices]
            y_boot = [self._y_train[i] for i in indices]
            tree = _DecisionTree(max_depth=self._max_depth)
            tree.fit(X_boot, y_boot, tree_rng)
            self._forest.append(tree)

        best_y = min(self._y_train)
        n_candidates = max(100, 20 * len(self._specs))
        suggestions: list[dict[str, Any]] = []

        for _ in range(n_suggestions):
            best_ei = -1.0
            best_point: dict[str, Any] = {}

            for _ in range(n_candidates):
                cand = {s.name: _sample_param(s, rng) for s in self._specs}
                cand_vec = self._encode(cand)

                # Ensemble prediction
                preds = [tree.predict(cand_vec) for tree in self._forest]
                mu = sum(preds) / len(preds)
                var = sum((p - mu) ** 2 for p in preds) / max(len(preds) - 1, 1)
                sigma = math.sqrt(max(var, 1e-12))

                # EI
                if sigma < 1e-12:
                    ei = 0.0
                else:
                    z = (best_y - mu) / sigma
                    ei = (best_y - mu) * _norm_cdf(z) + sigma * _norm_pdf(z)

                if ei > best_ei:
                    best_ei = ei
                    best_point = cand

            suggestions.append(best_point)

        return suggestions

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": True,
            "max_dimensions": None,
        }


# ── CMAESSampler ─────────────────────────────────────────────────────

class CMAESSampler(AlgorithmPlugin):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

    Maintains a mean vector, covariance matrix, and step size that
    adapt over successive calls to ``fit()`` / ``suggest()``.  Uses
    rank-1 and rank-mu updates.  Supports continuous and discrete
    parameters; falls back to random for categorical-only problems.
    """

    def __init__(self, population_size: int | None = None) -> None:
        self._specs: list[ParameterSpec] = []
        self._observations: list[Observation] = []
        self._pop_size = population_size
        # CMA state
        self._mean: list[float] | None = None
        self._C: list[list[float]] | None = None  # covariance matrix
        self._sigma: float = 0.3  # step size
        self._pc: list[float] | None = None  # evolution path for rank-1
        self._ps: list[float] | None = None  # evolution path for sigma
        self._gen: int = 0

    def name(self) -> str:
        return "cmaes_sampler"

    def _numeric_specs(self) -> list[ParameterSpec]:
        return [s for s in self._specs if s.type != VariableType.CATEGORICAL]

    def _has_categorical_only(self) -> bool:
        return all(s.type == VariableType.CATEGORICAL for s in self._specs)

    def _n_dims(self) -> int:
        return len(self._numeric_specs())

    def _to_vec(self, params: dict[str, Any]) -> list[float]:
        vec: list[float] = []
        for s in self._numeric_specs():
            lo = float(s.lower)
            hi = float(s.upper)
            val = float(params.get(s.name, (lo + hi) / 2))
            # Normalize to [0, 1]
            vec.append((val - lo) / max(hi - lo, 1e-12))
        return vec

    def _from_vec(self, vec: list[float], rng: random.Random) -> dict[str, Any]:
        point: dict[str, Any] = {}
        idx = 0
        for s in self._specs:
            if s.type == VariableType.CATEGORICAL:
                point[s.name] = rng.choice(s.categories)
            else:
                v = vec[idx] if idx < len(vec) else 0.5
                lo = float(s.lower)
                hi = float(s.upper)
                raw = lo + v * (hi - lo)
                point[s.name] = _clamp(raw, s)
                idx += 1
        return point

    def _init_state(self, n: int) -> None:
        self._mean = [0.5] * n
        self._C = _identity(n)
        self._pc = [0.0] * n
        self._ps = [0.0] * n
        self._sigma = 0.3
        self._gen = 0

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = list(parameter_specs)
        self._observations = [o for o in observations if not o.is_failure]

        n = self._n_dims()
        if n == 0:
            return

        if self._mean is None or len(self._mean) != n:
            self._init_state(n)

        # If enough observations, update CMA state
        if len(self._observations) < 3:
            return

        lam = self._pop_size or max(4, 4 + int(3 * math.log(n)))
        mu = lam // 2

        # Get recent observations sorted by objective
        recent = sorted(
            self._observations[-lam * 2:],
            key=lambda o: list(o.kpi_values.values())[0],
        )[:mu]

        # Weights for rank-mu update
        raw_w = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
        w_sum = sum(raw_w)
        weights = [w / w_sum for w in raw_w]

        # Compute new mean
        vecs = [self._to_vec(o.parameters) for o in recent]
        new_mean = [0.0] * n
        for i, w in enumerate(weights):
            for d in range(n):
                new_mean[d] += w * vecs[i][d]

        # Rank-1 update
        c1 = 2.0 / ((n + 1.3) ** 2 + mu)
        cmu = min(1.0 - c1, 2.0 * (mu - 2.0 + 1.0 / mu) / ((n + 2.0) ** 2 + mu))
        cc = 4.0 / (n + 4.0)
        cs = (mu + 2.0) / (n + mu + 5.0)
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mu - 1.0) / (n + 1.0)) - 1.0) + cs

        # Update evolution path
        mean_shift = [(new_mean[d] - self._mean[d]) / max(self._sigma, 1e-12) for d in range(n)]
        for d in range(n):
            self._ps[d] = (1.0 - cs) * self._ps[d] + math.sqrt(cs * (2.0 - cs) * mu) * mean_shift[d]
            self._pc[d] = (1.0 - cc) * self._pc[d] + math.sqrt(cc * (2.0 - cc) * mu) * mean_shift[d]

        # Update covariance matrix
        for i in range(n):
            for j in range(n):
                rank1 = c1 * self._pc[i] * self._pc[j]
                rank_mu = 0.0
                for k, w in enumerate(weights):
                    diff_i = (vecs[k][i] - self._mean[i]) / max(self._sigma, 1e-12)
                    diff_j = (vecs[k][j] - self._mean[j]) / max(self._sigma, 1e-12)
                    rank_mu += w * diff_i * diff_j
                rank_mu *= cmu
                self._C[i][j] = (1.0 - c1 - cmu) * self._C[i][j] + rank1 + rank_mu

        # Update step size
        ps_norm = math.sqrt(sum(p * p for p in self._ps))
        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self._sigma *= math.exp((cs / damps) * (ps_norm / chi_n - 1.0))
        self._sigma = max(1e-8, min(self._sigma, 2.0))

        self._mean = new_mean
        self._gen += 1

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)

        if self._has_categorical_only() or self._mean is None:
            return [
                {s.name: _sample_param(s, rng) for s in self._specs}
                for _ in range(n_suggestions)
            ]

        n = self._n_dims()
        # Sample from N(mean, sigma^2 * C)
        try:
            L = _cholesky(self._C)
        except Exception:
            L = _identity(n)

        suggestions: list[dict[str, Any]] = []
        for _ in range(n_suggestions):
            z = [rng.gauss(0, 1) for _ in range(n)]
            scaled = _mat_vec(L, z)
            sample = [
                self._mean[d] + self._sigma * scaled[d]
                for d in range(n)
            ]
            # Clamp to [0, 1]
            sample = [max(0.0, min(1.0, s)) for s in sample]
            suggestions.append(self._from_vec(sample, rng))
        return suggestions

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": False,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": True,
            "max_dimensions": None,
        }


# ── DifferentialEvolution ────────────────────────────────────────────

class DifferentialEvolution(AlgorithmPlugin):
    """Classic Differential Evolution (DE/rand/1/bin).

    Maintains a population that evolves across ``fit()`` calls.
    Mutation: ``v = x_r1 + F * (x_r2 - x_r3)``, binomial crossover,
    and greedy selection.  Works with continuous and discrete parameters.
    """

    def __init__(
        self,
        population_size: int = 30,
        F: float = 0.8,
        CR: float = 0.9,
    ) -> None:
        self._specs: list[ParameterSpec] = []
        self._observations: list[Observation] = []
        self._pop_size = population_size
        self._F = F
        self._CR = CR
        self._population: list[dict[str, Any]] = []
        self._fitness: list[float] = []

    def name(self) -> str:
        return "differential_evolution"

    def _to_vec(self, params: dict[str, Any]) -> list[float]:
        vec: list[float] = []
        for s in self._specs:
            if s.type == VariableType.CATEGORICAL:
                idx = s.categories.index(params.get(s.name, s.categories[0]))
                vec.append(float(idx))
            else:
                vec.append(float(params.get(s.name, 0.0)))
        return vec

    def _from_vec(self, vec: list[float]) -> dict[str, Any]:
        point: dict[str, Any] = {}
        for i, s in enumerate(self._specs):
            v = vec[i] if i < len(vec) else 0.0
            if s.type == VariableType.CATEGORICAL:
                idx = int(round(v)) % len(s.categories)
                idx = max(0, min(len(s.categories) - 1, idx))
                point[s.name] = s.categories[idx]
            else:
                point[s.name] = _clamp(v, s)
        return point

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = list(parameter_specs)
        self._observations = [o for o in observations if not o.is_failure]

        # Initialize population from observations if available
        if not self._population or len(self._population) != self._pop_size:
            self._population = []
            self._fitness = []
            if self._observations:
                # Seed population from best observations
                sorted_obs = sorted(
                    self._observations,
                    key=lambda o: list(o.kpi_values.values())[0],
                )
                for obs in sorted_obs[: self._pop_size]:
                    self._population.append(dict(obs.parameters))
                    self._fitness.append(list(obs.kpi_values.values())[0])

        # Update fitness for existing population members from observations
        if self._observations and self._population:
            obs_map: dict[str, float] = {}
            for o in self._observations:
                key = str(sorted(o.parameters.items()))
                obs_map[key] = list(o.kpi_values.values())[0]

            for i, member in enumerate(self._population):
                key = str(sorted(member.items()))
                if key in obs_map:
                    self._fitness[i] = obs_map[key]

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        n_dims = len(self._specs)

        # Fill population to target size with random members
        while len(self._population) < self._pop_size:
            member = {s.name: _sample_param(s, rng) for s in self._specs}
            self._population.append(member)
            self._fitness.append(float("inf"))

        if n_dims == 0:
            return [{}] * n_suggestions

        suggestions: list[dict[str, Any]] = []
        pop_vecs = [self._to_vec(p) for p in self._population]

        for _ in range(n_suggestions):
            # Pick a random target from population
            target_idx = rng.randint(0, len(pop_vecs) - 1)
            target = pop_vecs[target_idx]

            # Pick three distinct random indices (different from target)
            available = [i for i in range(len(pop_vecs)) if i != target_idx]
            if len(available) < 3:
                suggestions.append({s.name: _sample_param(s, rng) for s in self._specs})
                continue
            r1, r2, r3 = rng.sample(available, 3)

            # Mutation: v = x_r1 + F * (x_r2 - x_r3)
            mutant = [
                pop_vecs[r1][d] + self._F * (pop_vecs[r2][d] - pop_vecs[r3][d])
                for d in range(n_dims)
            ]

            # Binomial crossover
            j_rand = rng.randint(0, n_dims - 1)
            trial = [
                mutant[d] if (rng.random() < self._CR or d == j_rand) else target[d]
                for d in range(n_dims)
            ]

            suggestions.append(self._from_vec(trial))

        return suggestions

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": False,
            "max_dimensions": None,
        }


# ── NSGA2Sampler ─────────────────────────────────────────────────────

class NSGA2Sampler(AlgorithmPlugin):
    """Non-dominated Sorting Genetic Algorithm II (NSGA-II).

    Multi-objective optimizer using fast non-dominated sorting,
    crowding distance, tournament selection, Simulated Binary
    Crossover (SBX), and polynomial mutation.
    """

    def __init__(
        self,
        population_size: int = 40,
        crossover_prob: float = 0.9,
        mutation_prob: float | None = None,
        eta_c: float = 20.0,  # SBX distribution index
        eta_m: float = 20.0,  # mutation distribution index
    ) -> None:
        self._specs: list[ParameterSpec] = []
        self._observations: list[Observation] = []
        self._pop_size = population_size
        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
        self._eta_c = eta_c
        self._eta_m = eta_m
        self._population: list[dict[str, Any]] = []

    def name(self) -> str:
        return "nsga2_sampler"

    def _dominates(self, obj_a: list[float], obj_b: list[float]) -> bool:
        """Return True if obj_a dominates obj_b (all <= and at least one <)."""
        at_least_one_better = False
        for a, b in zip(obj_a, obj_b):
            if a > b:
                return False
            if a < b:
                at_least_one_better = True
        return at_least_one_better

    def _fast_nondominated_sort(
        self, objectives: list[list[float]]
    ) -> list[list[int]]:
        """Fast non-dominated sort.  Returns list of fronts (index lists)."""
        n = len(objectives)
        domination_count = [0] * n
        dominated_set: list[list[int]] = [[] for _ in range(n)]
        fronts: list[list[int]] = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dominates(objectives[p], objectives[q]):
                    dominated_set[p].append(q)
                elif self._dominates(objectives[q], objectives[p]):
                    domination_count[p] += 1
            if domination_count[p] == 0:
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front: list[int] = []
            for p in fronts[i]:
                for q in dominated_set[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return [f for f in fronts if f]

    def _crowding_distance(self, objectives: list[list[float]], front: list[int]) -> list[float]:
        """Compute crowding distance for a front."""
        n = len(front)
        if n <= 2:
            return [float("inf")] * n

        n_obj = len(objectives[0]) if objectives else 0
        distances = [0.0] * n

        for m in range(n_obj):
            # Sort by objective m
            sorted_idx = sorted(range(n), key=lambda i: objectives[front[i]][m])
            distances[sorted_idx[0]] = float("inf")
            distances[sorted_idx[-1]] = float("inf")
            obj_range = objectives[front[sorted_idx[-1]]][m] - objectives[front[sorted_idx[0]]][m]
            if obj_range < 1e-12:
                continue
            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (
                    objectives[front[sorted_idx[i + 1]]][m]
                    - objectives[front[sorted_idx[i - 1]]][m]
                ) / obj_range

        return distances

    def _sbx_crossover(
        self, p1: list[float], p2: list[float], rng: random.Random
    ) -> tuple[list[float], list[float]]:
        """Simulated Binary Crossover."""
        c1, c2 = list(p1), list(p2)
        if rng.random() > self._crossover_prob:
            return c1, c2
        for i in range(len(p1)):
            if rng.random() > 0.5:
                continue
            if abs(p1[i] - p2[i]) < 1e-14:
                continue
            u = rng.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (self._eta_c + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self._eta_c + 1.0))
            c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        return c1, c2

    def _polynomial_mutation(
        self, x: list[float], lowers: list[float], uppers: list[float], rng: random.Random
    ) -> list[float]:
        """Polynomial mutation."""
        n = len(x)
        prob = self._mutation_prob if self._mutation_prob is not None else 1.0 / max(n, 1)
        result = list(x)
        for i in range(n):
            if rng.random() > prob:
                continue
            delta = uppers[i] - lowers[i]
            if delta < 1e-14:
                continue
            u = rng.random()
            if u < 0.5:
                deltaq = (2.0 * u) ** (1.0 / (self._eta_m + 1.0)) - 1.0
            else:
                deltaq = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self._eta_m + 1.0))
            result[i] = x[i] + deltaq * delta
            result[i] = max(lowers[i], min(uppers[i], result[i]))
        return result

    def _to_vec(self, params: dict[str, Any]) -> list[float]:
        vec: list[float] = []
        for s in self._specs:
            if s.type == VariableType.CATEGORICAL:
                idx = s.categories.index(params.get(s.name, s.categories[0]))
                vec.append(float(idx))
            else:
                vec.append(float(params.get(s.name, 0.0)))
        return vec

    def _from_vec(self, vec: list[float], rng: random.Random) -> dict[str, Any]:
        point: dict[str, Any] = {}
        for i, s in enumerate(self._specs):
            v = vec[i] if i < len(vec) else 0.0
            if s.type == VariableType.CATEGORICAL:
                idx = int(round(v)) % len(s.categories)
                idx = max(0, min(len(s.categories) - 1, idx))
                point[s.name] = s.categories[idx]
            else:
                point[s.name] = _clamp(v, s)
        return point

    def _get_bounds(self) -> tuple[list[float], list[float]]:
        lowers, uppers = [], []
        for s in self._specs:
            if s.type == VariableType.CATEGORICAL:
                lowers.append(0.0)
                uppers.append(float(len(s.categories) - 1))
            else:
                lowers.append(float(s.lower))
                uppers.append(float(s.upper))
        return lowers, uppers

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = list(parameter_specs)
        self._observations = [o for o in observations if not o.is_failure]

        # Build initial population from observations
        if self._observations and not self._population:
            for obs in self._observations[: self._pop_size]:
                self._population.append(dict(obs.parameters))

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        n_dims = len(self._specs)

        if n_dims == 0:
            return [{}] * n_suggestions

        lowers, uppers = self._get_bounds()

        # Ensure population is filled
        while len(self._population) < self._pop_size:
            member = {s.name: _sample_param(s, rng) for s in self._specs}
            self._population.append(member)

        # Not enough observations for NSGA-II selection — just produce offspring
        if len(self._observations) < 4:
            suggestions: list[dict[str, Any]] = []
            for _ in range(n_suggestions):
                p = rng.choice(self._population)
                vec = self._to_vec(p)
                child = self._polynomial_mutation(vec, lowers, uppers, rng)
                suggestions.append(self._from_vec(child, rng))
            return suggestions

        # Compute objectives for population members
        obs_map: dict[str, list[float]] = {}
        for o in self._observations:
            key = str(sorted(o.parameters.items()))
            obs_map[key] = list(o.kpi_values.values())

        # Use only population members we have objectives for
        pop_with_obj: list[tuple[int, list[float]]] = []
        for i, member in enumerate(self._population):
            key = str(sorted(member.items()))
            if key in obs_map:
                pop_with_obj.append((i, obs_map[key]))

        if len(pop_with_obj) < 4:
            return [
                {s.name: _sample_param(s, rng) for s in self._specs}
                for _ in range(n_suggestions)
            ]

        indices = [p[0] for p in pop_with_obj]
        objectives = [p[1] for p in pop_with_obj]

        # Non-dominated sort
        fronts = self._fast_nondominated_sort(objectives)

        # Assign rank and crowding distance
        rank = [0] * len(indices)
        crowd = [0.0] * len(indices)
        local_to_pop = {local: pop_idx for local, (pop_idx, _) in enumerate(pop_with_obj)}

        for front_rank, front in enumerate(fronts):
            cd = self._crowding_distance(objectives, front)
            for local_idx_in_front, local_idx in enumerate(front):
                rank[local_idx] = front_rank
                crowd[local_idx] = cd[local_idx_in_front]

        # Tournament selection + crossover + mutation
        suggestions: list[dict[str, Any]] = []
        pop_vecs = [self._to_vec(self._population[indices[i]]) for i in range(len(indices))]

        def tournament() -> int:
            a, b = rng.sample(range(len(indices)), 2)
            if rank[a] < rank[b]:
                return a
            elif rank[b] < rank[a]:
                return b
            return a if crowd[a] > crowd[b] else b

        for _ in range(n_suggestions):
            p1_idx = tournament()
            p2_idx = tournament()
            c1, c2 = self._sbx_crossover(pop_vecs[p1_idx], pop_vecs[p2_idx], rng)
            child = self._polynomial_mutation(c1, lowers, uppers, rng)
            suggestions.append(self._from_vec(child, rng))

        return suggestions

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": True,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "supports_multi_objective": True,
            "requires_observations": True,
            "max_dimensions": None,
        }


# ── TuRBOSampler ────────────────────────────────────────────────────

class TuRBOSampler(AlgorithmPlugin):
    """Trust Region Bayesian Optimization (TuRBO).

    Maintains a local trust region centered on the best observed point.
    The trust region shrinks after consecutive failures and grows after
    consecutive successes.  Inside the trust region, candidates are
    selected using a simplified GP surrogate with Thompson sampling.
    Continuous-focused; falls back to random for categorical-only problems.
    """

    def __init__(
        self,
        length_init: float = 0.8,
        length_min: float = 0.01,
        length_max: float = 1.6,
        success_tol: int = 3,
        failure_tol: int = 5,
    ) -> None:
        self._specs: list[ParameterSpec] = []
        self._observations: list[Observation] = []
        self._length = length_init
        self._length_init = length_init
        self._length_min = length_min
        self._length_max = length_max
        self._success_tol = success_tol
        self._failure_tol = failure_tol
        self._n_successes = 0
        self._n_failures = 0
        self._best_value: float | None = None

    def name(self) -> str:
        return "turbo_sampler"

    def _numeric_specs(self) -> list[ParameterSpec]:
        return [s for s in self._specs if s.type != VariableType.CATEGORICAL]

    def _has_categorical_only(self) -> bool:
        return all(s.type == VariableType.CATEGORICAL for s in self._specs)

    def _to_unit(self, params: dict[str, Any]) -> list[float]:
        """Convert to [0, 1] unit hypercube (numeric dims only)."""
        vec: list[float] = []
        for s in self._numeric_specs():
            lo, hi = float(s.lower), float(s.upper)
            val = float(params.get(s.name, (lo + hi) / 2))
            vec.append((val - lo) / max(hi - lo, 1e-12))
        return vec

    def _from_unit(self, vec: list[float], rng: random.Random) -> dict[str, Any]:
        """Convert from [0, 1] unit hypercube back to parameter dict."""
        point: dict[str, Any] = {}
        idx = 0
        for s in self._specs:
            if s.type == VariableType.CATEGORICAL:
                point[s.name] = rng.choice(s.categories)
            else:
                v = vec[idx] if idx < len(vec) else 0.5
                v = max(0.0, min(1.0, v))
                lo, hi = float(s.lower), float(s.upper)
                raw = lo + v * (hi - lo)
                point[s.name] = _clamp(raw, s)
                idx += 1
        return point

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        self._specs = list(parameter_specs)
        prev_best = self._best_value
        self._observations = [o for o in observations if not o.is_failure]

        if not self._observations:
            return

        # Track best value
        current_best = min(
            list(o.kpi_values.values())[0] for o in self._observations
        )

        # Update trust region based on improvement
        if prev_best is not None:
            if current_best < prev_best - 1e-8:
                self._n_successes += 1
                self._n_failures = 0
            else:
                self._n_failures += 1
                self._n_successes = 0

            if self._n_successes >= self._success_tol:
                self._length = min(self._length * 2.0, self._length_max)
                self._n_successes = 0
            elif self._n_failures >= self._failure_tol:
                self._length = max(self._length / 2.0, self._length_min)
                self._n_failures = 0

        self._best_value = current_best

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)

        if self._has_categorical_only():
            return [
                {s.name: _sample_param(s, rng) for s in self._specs}
                for _ in range(n_suggestions)
            ]

        # Fallback with few observations
        if len(self._observations) < 3:
            return [
                {s.name: _sample_param(s, rng) for s in self._specs}
                for _ in range(n_suggestions)
            ]

        n_dims = len(self._numeric_specs())

        # Find best observation (center of trust region)
        best_obs = min(
            self._observations,
            key=lambda o: list(o.kpi_values.values())[0],
        )
        center = self._to_unit(best_obs.parameters)

        # Build a local GP using observations within the trust region
        X_all = [self._to_unit(o.parameters) for o in self._observations]
        y_all = [list(o.kpi_values.values())[0] for o in self._observations]

        # Filter to trust region
        tr_half = self._length / 2.0
        X_local, y_local = [], []
        for x, y in zip(X_all, y_all):
            if all(abs(x[d] - center[d]) <= tr_half for d in range(n_dims)):
                X_local.append(x)
                y_local.append(y)

        # If not enough local points, use all
        if len(X_local) < 3:
            X_local, y_local = X_all, y_all

        best_y = min(y_local)

        # Build simple GP kernel matrix
        ls = self._length * 0.5
        n_train = len(X_local)
        noise = 1e-4

        K = [[0.0] * n_train for _ in range(n_train)]
        for i in range(n_train):
            for j in range(i, n_train):
                sq = sum((X_local[i][d] - X_local[j][d]) ** 2 for d in range(n_dims))
                k = math.exp(-0.5 * sq / max(ls * ls, 1e-12))
                if i == j:
                    k += noise
                K[i][j] = k
                K[j][i] = k

        try:
            L = _cholesky(K)
            alpha = _solve_cholesky(L, y_local)
        except Exception:
            # Fallback: sample uniformly in trust region
            suggestions: list[dict[str, Any]] = []
            for _ in range(n_suggestions):
                vec = [
                    max(0.0, min(1.0, center[d] + rng.uniform(-tr_half, tr_half)))
                    for d in range(n_dims)
                ]
                suggestions.append(self._from_unit(vec, rng))
            return suggestions

        # Thompson sampling: sample GP posterior at candidates in trust region
        n_candidates = max(100, 20 * n_dims)
        suggestions = []

        for _ in range(n_suggestions):
            best_thompson = float("inf")
            best_point: dict[str, Any] = {}

            for _ in range(n_candidates):
                # Sample candidate within trust region
                cand = [
                    max(0.0, min(1.0, center[d] + rng.uniform(-tr_half, tr_half)))
                    for d in range(n_dims)
                ]

                # GP posterior
                k_star = [
                    math.exp(-0.5 * sum((cand[d] - X_local[j][d]) ** 2 for d in range(n_dims)) / max(ls * ls, 1e-12))
                    for j in range(n_train)
                ]
                mu = _vec_dot(k_star, alpha)
                v = _solve_lower(L, k_star)
                k_ss = 1.0 + noise
                sigma2 = max(k_ss - _vec_dot(v, v), 1e-12)
                sigma = math.sqrt(sigma2)

                # Thompson sample
                thompson = mu + sigma * rng.gauss(0, 1)

                if thompson < best_thompson:
                    best_thompson = thompson
                    best_point = self._from_unit(cand, rng)

            suggestions.append(best_point)

        return suggestions

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_categorical": False,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": True,
            "max_dimensions": None,
        }

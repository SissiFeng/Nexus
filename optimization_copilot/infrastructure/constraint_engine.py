"""Constraint optimization framework.

Three-tier constraint model:
1. KNOWN_HARD: Known explicit constraints — hard filter before suggest
2. KNOWN_SOFT: Known soft constraints — penalty in acquisition function
3. UNKNOWN: Unknown constraints — learned via GP classifier from observations

References:
- SafeOpt: Safe constraint handling with safety probabilities
- Anubis (Digital Discovery 2025): Feasibility-aware acquisition
- IISE 2025: Entropy-based constraint query for active learning
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ConstraintType(Enum):
    """Three levels of constraint knowledge."""
    KNOWN_HARD = "known_hard"    # Violation = experiment failure/danger
    KNOWN_SOFT = "known_soft"    # Violation = undesirable but acceptable
    UNKNOWN = "unknown"          # Learned during optimization


class ConstraintStatus(Enum):
    """Evaluation result of a constraint."""
    FEASIBLE = "feasible"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass
class Constraint:
    """Definition of a single constraint."""
    name: str
    constraint_type: ConstraintType

    # Known constraints: evaluation function (params_dict -> bool)
    evaluate: Callable[[dict[str, Any]], bool] | None = None

    # Unknown constraints: observations (params_as_list, feasible)
    observations: list[tuple[list[float], bool]] = field(default_factory=list)

    # Soft constraint tolerance (0 = strict, 1 = very lenient)
    tolerance: float = 0.0

    # Safety probability threshold (SafeOpt style)
    safety_probability: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "constraint_type": self.constraint_type.value,
            "tolerance": self.tolerance,
            "safety_probability": self.safety_probability,
            "n_observations": len(self.observations),
        }


@dataclass
class ConstraintEvaluation:
    """Result of evaluating all constraints on a candidate."""
    candidate: dict[str, Any]
    is_feasible: bool
    constraint_results: dict[str, ConstraintStatus] = field(default_factory=dict)
    feasibility_probabilities: dict[str, float] = field(default_factory=dict)
    overall_feasibility_probability: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_feasible": self.is_feasible,
            "constraint_results": {k: v.value for k, v in self.constraint_results.items()},
            "feasibility_probabilities": dict(self.feasibility_probabilities),
            "overall_feasibility_probability": self.overall_feasibility_probability,
        }


class _FeasibilityGP:
    """Minimal GP classifier for learning unknown constraint boundaries.

    Uses GP with probit approximation (Laplace approximation)
    for binary classification of feasible/infeasible regions.

    Pure Python stdlib implementation.
    """

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-4):
        self._length_scale = length_scale
        self._noise = noise
        self._X: list[list[float]] = []
        self._y: list[float] = []  # 1.0 = feasible, 0.0 = infeasible
        self._alpha: list[float] | None = None
        self._L: list[list[float]] | None = None
        self._fitted = False

    def _rbf_kernel(self, x1: list[float], x2: list[float]) -> float:
        """RBF kernel."""
        ls2 = self._length_scale ** 2
        sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return math.exp(-0.5 * sq_dist / ls2)

    def _build_K(self, X: list[list[float]]) -> list[list[float]]:
        """Build kernel matrix."""
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

    @staticmethod
    def _cholesky(A: list[list[float]]) -> list[list[float]]:
        """Cholesky decomposition A = L L^T."""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] - s
                    L[i][j] = math.sqrt(max(val, 1e-12))
                else:
                    L[i][j] = (A[i][j] - s) / max(L[j][j], 1e-12)
        return L

    @staticmethod
    def _solve_lower(L: list[list[float]], b: list[float]) -> list[float]:
        """Forward substitution."""
        n = len(b)
        x = [0.0] * n
        for i in range(n):
            x[i] = (b[i] - sum(L[i][j] * x[j] for j in range(i))) / max(L[i][i], 1e-12)
        return x

    @staticmethod
    def _solve_upper(U: list[list[float]], b: list[float]) -> list[float]:
        """Back substitution."""
        n = len(b)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / max(U[i][i], 1e-12)
        return x

    def _solve_cholesky(self, L: list[list[float]], b: list[float]) -> list[float]:
        """Solve A x = b given L where A = L L^T."""
        y = self._solve_lower(L, b)
        n = len(L)
        Lt = [[L[j][i] for j in range(n)] for i in range(n)]
        return self._solve_upper(Lt, y)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid function (logistic) as probit approximation."""
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        """Train GP classifier. y in {0.0, 1.0}."""
        self._X = [list(x) for x in X]
        self._y = list(y)

        if len(X) < 2:
            self._fitted = False
            return

        # Auto-scale length scale
        n_dims = len(X[0]) if X else 1
        ranges = []
        for d in range(n_dims):
            vals = [x[d] for x in X]
            r = max(vals) - min(vals) if vals else 1.0
            ranges.append(r if r > 0 else 1.0)
        self._length_scale = sum(ranges) / len(ranges) * 0.5

        # Transform y to {-1, +1} for GP regression approximation
        y_centered = [2.0 * yi - 1.0 for yi in y]  # {0,1} -> {-1,+1}

        K = self._build_K(X)
        try:
            self._L = self._cholesky(K)
            self._alpha = self._solve_cholesky(self._L, y_centered)
            self._fitted = True
        except Exception:
            self._fitted = False

    def predict_probability(self, x_new: list[float] | dict) -> float:
        """Predict P(feasible | x_new).

        For dict input, extracts float values in order.
        """
        if isinstance(x_new, dict):
            x_new = [float(v) for v in x_new.values() if isinstance(v, (int, float))]

        if not self._fitted or self._alpha is None or self._L is None:
            return 0.5  # No data: uniform prior

        # GP posterior mean
        k_star = [self._rbf_kernel(x_new, xt) for xt in self._X]
        f_mean = sum(k * a for k, a in zip(k_star, self._alpha))

        # GP posterior variance
        v = self._solve_lower(self._L, k_star)
        k_ss = self._rbf_kernel(x_new, x_new)
        f_var = max(k_ss - sum(vi * vi for vi in v), 1e-12)
        f_std = math.sqrt(f_var)

        # Probit: P(feasible) = Phi(f_mean / sqrt(1 + f_var))
        # Approximation: sigmoid scaling
        p = self._norm_cdf(f_mean / math.sqrt(1.0 + f_var))
        return max(0.001, min(0.999, p))

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_observations(self) -> int:
        return len(self._X)


class ConstraintEngine:
    """Unified constraint handling engine.

    Capabilities:
    1. Known hard constraints -> candidate filtering (pre-suggest)
    2. Known soft constraints -> penalty injection into acquisition
    3. Unknown constraints -> GP classifier learns feasibility boundary
    4. Constraint-weighted acquisition (Anubis-style)
    5. Active learning for constraint exploration (entropy-based)
    6. Feasibility summary statistics
    """

    def __init__(self, constraints: list[Constraint] | None = None):
        self._constraints = list(constraints) if constraints else []
        self._feasibility_models: dict[str, _FeasibilityGP] = {}
        self._evaluation_history: list[ConstraintEvaluation] = []

    @property
    def constraints(self) -> list[Constraint]:
        return list(self._constraints)

    @property
    def n_constraints(self) -> int:
        return len(self._constraints)

    @property
    def has_unknown_constraints(self) -> bool:
        return any(c.constraint_type == ConstraintType.UNKNOWN for c in self._constraints)

    @property
    def has_hard_constraints(self) -> bool:
        return any(c.constraint_type == ConstraintType.KNOWN_HARD for c in self._constraints)

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the engine."""
        self._constraints.append(constraint)

    def remove_constraint(self, name: str) -> bool:
        """Remove a constraint by name. Returns True if found."""
        for i, c in enumerate(self._constraints):
            if c.name == name:
                self._constraints.pop(i)
                self._feasibility_models.pop(name, None)
                return True
        return False

    def filter_candidates(
        self,
        candidates: list[dict[str, Any]],
        parameter_specs: list | None = None,
    ) -> list[dict[str, Any]]:
        """Filter candidates using known hard constraints.

        O(n * c) where n = len(candidates), c = number of hard constraints.
        """
        if not self._constraints:
            return list(candidates)

        result: list[dict[str, Any]] = []
        for x in candidates:
            feasible = True
            for c in self._constraints:
                if c.constraint_type == ConstraintType.KNOWN_HARD:
                    if c.evaluate is not None and not c.evaluate(x):
                        feasible = False
                        break
            if feasible:
                result.append(x)
        return result

    def constraint_weighted_acquisition(
        self,
        acquisition_values: list[float],
        candidates: list[dict[str, Any]],
    ) -> list[float]:
        """Weight acquisition values by constraint feasibility.

        For unknown constraints: weighted_acq(x) = acq(x) * P(feasible|x)
        For soft constraints: weighted_acq(x) = acq(x) * penalty(x)

        Reference: Anubis (Digital Discovery 2025) feasibility-aware acquisition.
        """
        weighted = list(acquisition_values)
        for i, x in enumerate(candidates):
            for c in self._constraints:
                if c.constraint_type == ConstraintType.UNKNOWN:
                    p_feasible = self._predict_feasibility(c.name, x)
                    weighted[i] *= p_feasible
                elif c.constraint_type == ConstraintType.KNOWN_SOFT:
                    if c.evaluate is not None and not c.evaluate(x):
                        penalty = max(0.01, 1.0 - c.tolerance)
                        weighted[i] *= penalty
        return weighted

    def evaluate_candidate(self, x: dict[str, Any]) -> ConstraintEvaluation:
        """Evaluate all constraints on a single candidate.

        Returns detailed evaluation including per-constraint status
        and feasibility probabilities.
        """
        results: dict[str, ConstraintStatus] = {}
        probs: dict[str, float] = {}
        overall_prob = 1.0
        is_feasible = True

        for c in self._constraints:
            if c.constraint_type in (ConstraintType.KNOWN_HARD, ConstraintType.KNOWN_SOFT):
                if c.evaluate is not None:
                    if c.evaluate(x):
                        results[c.name] = ConstraintStatus.FEASIBLE
                        probs[c.name] = 1.0
                    else:
                        results[c.name] = ConstraintStatus.VIOLATED
                        probs[c.name] = 0.0
                        if c.constraint_type == ConstraintType.KNOWN_HARD:
                            is_feasible = False
                        overall_prob *= 0.0 if c.constraint_type == ConstraintType.KNOWN_HARD else max(0.01, 1.0 - c.tolerance)
                else:
                    results[c.name] = ConstraintStatus.UNKNOWN
                    probs[c.name] = 0.5
            elif c.constraint_type == ConstraintType.UNKNOWN:
                p = self._predict_feasibility(c.name, x)
                probs[c.name] = p
                if p >= c.safety_probability:
                    results[c.name] = ConstraintStatus.FEASIBLE
                elif p <= (1.0 - c.safety_probability):
                    results[c.name] = ConstraintStatus.VIOLATED
                    is_feasible = False
                else:
                    results[c.name] = ConstraintStatus.UNKNOWN
                overall_prob *= p

        evaluation = ConstraintEvaluation(
            candidate=x,
            is_feasible=is_feasible,
            constraint_results=results,
            feasibility_probabilities=probs,
            overall_feasibility_probability=overall_prob,
        )
        self._evaluation_history.append(evaluation)
        return evaluation

    def update_unknown_constraints(
        self,
        x: list[float] | dict[str, Any],
        constraint_results: dict[str, bool],
    ) -> None:
        """Update unknown constraint models with experimental feedback.

        Called after each experiment with feasibility results.

        Args:
            x: The evaluated point (as list of floats or parameter dict)
            constraint_results: {constraint_name: feasible}
        """
        if isinstance(x, dict):
            x_list = [float(v) for v in x.values() if isinstance(v, (int, float))]
        else:
            x_list = list(x)

        for name, feasible in constraint_results.items():
            constraint = self._get_constraint(name)
            if constraint is not None and constraint.constraint_type == ConstraintType.UNKNOWN:
                constraint.observations.append((x_list, feasible))
                self._update_feasibility_model(name)

    def suggest_constraint_exploration(
        self,
        candidates: list[dict[str, Any]],
        budget: int = 1,
    ) -> list[dict[str, Any]]:
        """Active learning: select points that maximize constraint information gain.

        Selects candidates where P(feasible) is closest to 0.5
        (i.e., near the feasibility boundary, maximum entropy).

        Reference: IISE 2025 entropy-based constraint query.
        """
        if not self.has_unknown_constraints or not candidates:
            return candidates[:budget]

        info_gains: list[tuple[float, dict[str, Any]]] = []
        for x in candidates:
            total_entropy = 0.0
            for c in self._constraints:
                if c.constraint_type == ConstraintType.UNKNOWN:
                    p = self._predict_feasibility(c.name, x)
                    # Binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)
                    if 0 < p < 1:
                        total_entropy += -(p * math.log(p) + (1 - p) * math.log(1 - p))
            info_gains.append((total_entropy, x))

        info_gains.sort(key=lambda t: t[0], reverse=True)
        return [x for _, x in info_gains[:budget]]

    def feasibility_summary(self) -> dict[str, Any]:
        """Summary statistics for all constraints."""
        summary: dict[str, Any] = {
            "n_constraints": self.n_constraints,
            "n_hard": sum(1 for c in self._constraints if c.constraint_type == ConstraintType.KNOWN_HARD),
            "n_soft": sum(1 for c in self._constraints if c.constraint_type == ConstraintType.KNOWN_SOFT),
            "n_unknown": sum(1 for c in self._constraints if c.constraint_type == ConstraintType.UNKNOWN),
            "constraints": {},
        }

        for c in self._constraints:
            info: dict[str, Any] = {
                "type": c.constraint_type.value,
                "tolerance": c.tolerance,
                "safety_probability": c.safety_probability,
            }
            if c.constraint_type == ConstraintType.UNKNOWN:
                n_obs = len(c.observations)
                n_feasible = sum(1 for _, f in c.observations if f)
                info["n_observations"] = n_obs
                info["feasibility_rate"] = n_feasible / n_obs if n_obs > 0 else None
                info["model_fitted"] = c.name in self._feasibility_models and self._feasibility_models[c.name].is_fitted
            summary["constraints"][c.name] = info

        return summary

    def feasibility_rate(self) -> float | None:
        """Overall feasibility rate from evaluation history."""
        if not self._evaluation_history:
            return None
        n_feasible = sum(1 for e in self._evaluation_history if e.is_feasible)
        return n_feasible / len(self._evaluation_history)

    def _predict_feasibility(self, constraint_name: str, x: dict[str, Any] | list[float]) -> float:
        """Predict feasibility probability using GP classifier."""
        model = self._feasibility_models.get(constraint_name)
        if model is None or not model.is_fitted:
            return 0.5  # No data: optimistic prior
        return model.predict_probability(x)

    def _update_feasibility_model(self, constraint_name: str) -> None:
        """Retrain the GP classifier for an unknown constraint."""
        constraint = self._get_constraint(constraint_name)
        if constraint is None or len(constraint.observations) < 2:
            return

        model = _FeasibilityGP()
        X = [obs[0] for obs in constraint.observations]
        y = [1.0 if obs[1] else 0.0 for obs in constraint.observations]
        model.fit(X, y)
        self._feasibility_models[constraint_name] = model

    def _get_constraint(self, name: str) -> Constraint | None:
        """Look up a constraint by name."""
        for c in self._constraints:
            if c.name == name:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "constraints": [c.to_dict() for c in self._constraints],
            "feasibility_summary": self.feasibility_summary(),
        }

    @classmethod
    def from_constraints(cls, constraint_defs: list[dict[str, Any]]) -> ConstraintEngine:
        """Create from serialized constraint definitions.

        Note: evaluate functions cannot be serialized, so known constraints
        created this way will need their evaluate functions re-attached.
        """
        constraints: list[Constraint] = []
        for d in constraint_defs:
            constraints.append(Constraint(
                name=d["name"],
                constraint_type=ConstraintType(d["constraint_type"]),
                tolerance=d.get("tolerance", 0.0),
                safety_probability=d.get("safety_probability", 0.95),
            ))
        return cls(constraints=constraints)

"""Pure-Python symbolic regression via tree-based genetic programming.

Discovers interpretable mathematical expressions that approximate
the relationship between input features and a target variable.
Results are returned as a Pareto front of accuracy vs. complexity.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Expression tree node
# ---------------------------------------------------------------------------

_UNARY_OPS = {"exp", "log", "sqrt", "abs", "neg"}
_BINARY_OPS = {"+", "-", "*", "/"}
_LEAF_OPS = {"const", "var"}


@dataclass
class ExprNode:
    """A single node in a symbolic expression tree.

    Supported ``op`` values:

    - Binary: ``"+"``, ``"-"``, ``"*"``, ``"/"``
    - Unary: ``"exp"``, ``"log"``, ``"sqrt"``, ``"abs"``, ``"neg"``
    - Leaf: ``"const"`` (uses *value*), ``"var"`` (uses *var_index*)
    """

    op: str
    value: float | None = None
    var_index: int | None = None
    left: ExprNode | None = None
    right: ExprNode | None = None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, x: list[float]) -> float:
        """Evaluate the expression tree at a single data point.

        Parameters
        ----------
        x : list[float]
            Feature vector.

        Returns
        -------
        float
            The computed value, or ``float('inf')`` on error.
        """
        try:
            return self._eval(x)
        except Exception:
            return float("inf")

    def _eval(self, x: list[float]) -> float:
        """Recursive evaluation (may raise)."""
        if self.op == "const":
            return self.value if self.value is not None else 0.0
        if self.op == "var":
            idx = self.var_index if self.var_index is not None else 0
            if idx < 0 or idx >= len(x):
                return 0.0
            return x[idx]

        # Unary operators
        if self.op in _UNARY_OPS:
            a = self.left._eval(x) if self.left else 0.0
            if not math.isfinite(a):
                return float("inf")
            if self.op == "exp":
                if a > 500.0:
                    return float("inf")
                return math.exp(a)
            if self.op == "log":
                if a <= 0.0:
                    return float("inf")
                return math.log(a)
            if self.op == "sqrt":
                if a < 0.0:
                    return float("inf")
                return math.sqrt(a)
            if self.op == "abs":
                return abs(a)
            if self.op == "neg":
                return -a

        # Binary operators
        a = self.left._eval(x) if self.left else 0.0
        b = self.right._eval(x) if self.right else 0.0
        if not math.isfinite(a) or not math.isfinite(b):
            return float("inf")

        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            result = a * b
            if not math.isfinite(result):
                return float("inf")
            return result
        if self.op == "/":
            if abs(b) < 1e-12:
                return float("inf")
            return a / b

        return float("inf")

    # ------------------------------------------------------------------
    # Complexity
    # ------------------------------------------------------------------

    def complexity(self) -> int:
        """Count the total number of nodes in the tree."""
        count = 1
        if self.left is not None:
            count += self.left.complexity()
        if self.right is not None:
            count += self.right.complexity()
        return count

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def to_string(self, var_names: list[str] | None = None) -> str:
        """Convert the expression tree to a human-readable string.

        Parameters
        ----------
        var_names : list[str] | None
            Optional variable names; defaults to ``x0, x1, ...``.

        Returns
        -------
        str
            Infix expression string.
        """
        if self.op == "const":
            v = self.value if self.value is not None else 0.0
            # Format nicely
            if v == int(v) and abs(v) < 1e6:
                return str(int(v))
            return f"{v:.4g}"
        if self.op == "var":
            idx = self.var_index if self.var_index is not None else 0
            if var_names and 0 <= idx < len(var_names):
                return var_names[idx]
            return f"x{idx}"

        if self.op in _UNARY_OPS:
            child_str = self.left.to_string(var_names) if self.left else "0"
            return f"{self.op}({child_str})"

        # Binary
        left_str = self.left.to_string(var_names) if self.left else "0"
        right_str = self.right.to_string(var_names) if self.right else "0"
        return f"({left_str} {self.op} {right_str})"

    def __repr__(self) -> str:
        return self.to_string()


# ---------------------------------------------------------------------------
# Pareto solution
# ---------------------------------------------------------------------------

@dataclass
class ParetoSolution:
    """A single solution on the accuracy-complexity Pareto front."""

    expression: ExprNode
    mse: float
    complexity: int
    equation_string: str


# ---------------------------------------------------------------------------
# EquationDiscovery
# ---------------------------------------------------------------------------

class EquationDiscovery:
    """Symbolic regression using tree-based genetic programming.

    Parameters
    ----------
    population_size : int
        Number of individuals per generation.
    n_generations : int
        Number of evolutionary generations.
    tournament_size : int
        Tournament selection size.
    max_depth : int
        Maximum tree depth for random tree generation.
    operators : list[str] | None
        Allowed binary operators.  Defaults to ``["+", "-", "*", "/"]``.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        population_size: int = 200,
        n_generations: int = 50,
        tournament_size: int = 5,
        max_depth: int = 5,
        operators: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.population_size = population_size
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.operators = operators or ["+", "-", "*", "/"]
        self.seed = seed
        self._rng = random.Random(seed)
        self._pareto_front: list[ParetoSolution] = []
        self._var_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Random tree generation
    # ------------------------------------------------------------------

    def _random_tree(self, depth: int, n_vars: int) -> ExprNode:
        """Generate a random expression tree using the grow method.

        Parameters
        ----------
        depth : int
            Maximum remaining depth.
        n_vars : int
            Number of input variables.

        Returns
        -------
        ExprNode
            A randomly generated expression tree.
        """
        # Terminal with increasing probability as depth decreases
        if depth <= 1 or (depth < self.max_depth and self._rng.random() < 0.3):
            return self._random_leaf(n_vars)

        op = self._rng.choice(self.operators)
        left = self._random_tree(depth - 1, n_vars)
        right = self._random_tree(depth - 1, n_vars)
        return ExprNode(op=op, left=left, right=right)

    def _random_leaf(self, n_vars: int) -> ExprNode:
        """Generate a random leaf node (constant or variable)."""
        if n_vars > 0 and self._rng.random() < 0.6:
            return ExprNode(op="var", var_index=self._rng.randint(0, n_vars - 1))
        else:
            # Random constant in a reasonable range
            c = self._rng.uniform(-5.0, 5.0)
            return ExprNode(op="const", value=round(c, 2))

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def _evaluate_fitness(
        self,
        tree: ExprNode,
        X: list[list[float]],
        y: list[float],
    ) -> float:
        """Compute MSE between tree predictions and target values.

        Returns
        -------
        float
            Mean squared error, or ``float('inf')`` on failure.
        """
        n = len(y)
        if n == 0:
            return float("inf")

        total = 0.0
        for i in range(n):
            pred = tree.evaluate(X[i])
            if not math.isfinite(pred):
                return float("inf")
            diff = pred - y[i]
            total += diff * diff

        mse = total / n
        if not math.isfinite(mse):
            return float("inf")
        return mse

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _tournament_select(
        self,
        population: list[ExprNode],
        fitnesses: list[float],
    ) -> ExprNode:
        """Tournament selection.

        Returns
        -------
        ExprNode
            A deep copy of the selected individual.
        """
        n = len(population)
        k = min(self.tournament_size, n)
        indices = self._rng.sample(range(n), k)
        best_idx = min(indices, key=lambda i: fitnesses[i])
        return copy.deepcopy(population[best_idx])

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def _crossover(self, parent1: ExprNode, parent2: ExprNode) -> ExprNode:
        """Subtree crossover.

        Pick a random node in *parent1* and replace it with a random
        subtree from *parent2*.

        Returns
        -------
        ExprNode
            A new offspring tree (deep copies used to avoid aliasing).
        """
        child = copy.deepcopy(parent1)
        donor = copy.deepcopy(parent2)

        # Collect all nodes (and parent references) in child
        child_nodes = _collect_nodes(child)
        donor_nodes = _collect_nodes(donor)

        if len(child_nodes) <= 1:
            return donor
        if not donor_nodes:
            return child

        # Pick a random non-root crossover point in child
        cx_idx = self._rng.randint(1, len(child_nodes) - 1) if len(child_nodes) > 1 else 0
        donor_idx = self._rng.randint(0, len(donor_nodes) - 1)

        target_node, parent_node, attr = child_nodes[cx_idx]
        donor_subtree = donor_nodes[donor_idx][0]

        if parent_node is not None:
            setattr(parent_node, attr, copy.deepcopy(donor_subtree))

        return child

    def _mutate(self, tree: ExprNode, n_vars: int) -> ExprNode:
        """Point mutation: replace a random node with a small random subtree.

        Returns
        -------
        ExprNode
            Mutated tree (deep copy).
        """
        mutant = copy.deepcopy(tree)
        nodes = _collect_nodes(mutant)

        if not nodes:
            return self._random_tree(2, n_vars)

        idx = self._rng.randint(0, len(nodes) - 1)
        _node, parent, attr = nodes[idx]
        new_sub = self._random_tree(self._rng.randint(1, 2), n_vars)

        if parent is None:
            return new_sub
        setattr(parent, attr, new_sub)
        return mutant

    # ------------------------------------------------------------------
    # Physics filter
    # ------------------------------------------------------------------

    def _physics_filter(self, tree: ExprNode, X: list[list[float]]) -> bool:
        """Check that the expression produces bounded output on training data.

        Returns
        -------
        bool
            *True* if the tree produces finite outputs on all data points.
        """
        for row in X:
            val = tree.evaluate(row)
            if not math.isfinite(val) or abs(val) > 1e12:
                return False
        return True

    # ------------------------------------------------------------------
    # Main algorithm
    # ------------------------------------------------------------------

    def fit(
        self,
        X: list[list[float]],
        y: list[float],
        var_names: list[str] | None = None,
    ) -> list[ParetoSolution]:
        """Run the genetic programming algorithm.

        Parameters
        ----------
        X : list[list[float]]
            Feature matrix, shape ``(n_samples, n_features)``.
        y : list[float]
            Target values.
        var_names : list[str] | None
            Optional variable names for equation strings.

        Returns
        -------
        list[ParetoSolution]
            Pareto front of (accuracy, complexity) trade-offs.
        """
        self._var_names = var_names

        if not X or not y:
            self._pareto_front = []
            return []

        n_vars = len(X[0]) if X else 1

        # Step 1: Generate initial population
        population: list[ExprNode] = []
        for _ in range(self.population_size):
            tree = self._random_tree(self._rng.randint(2, self.max_depth), n_vars)
            population.append(tree)

        # Step 2: Evolve
        elite_count = max(1, self.population_size // 20)  # top 5%

        for _gen in range(self.n_generations):
            # Evaluate fitness
            fitnesses = [self._evaluate_fitness(t, X, y) for t in population]

            # Sort by fitness for elitism
            indices_sorted = sorted(range(len(population)), key=lambda i: fitnesses[i])

            # Elitism: carry over top individuals
            new_pop: list[ExprNode] = []
            for i in indices_sorted[:elite_count]:
                new_pop.append(copy.deepcopy(population[i]))

            # Generate rest of new population
            while len(new_pop) < self.population_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)

                r = self._rng.random()
                if r < 0.8:
                    # Crossover
                    child = self._crossover(p1, p2)
                elif r < 0.95:
                    # Mutation
                    child = self._mutate(p1, n_vars)
                else:
                    # Reproduction
                    child = p1

                # Occasional mutation after crossover
                if self._rng.random() < 0.2:
                    child = self._mutate(child, n_vars)

                # Apply physics filter
                if self._physics_filter(child, X):
                    new_pop.append(child)
                else:
                    # Replace with a random tree
                    new_pop.append(
                        self._random_tree(self._rng.randint(2, self.max_depth), n_vars)
                    )

            population = new_pop

        # Step 3: Extract Pareto front
        fitnesses = [self._evaluate_fitness(t, X, y) for t in population]
        self._pareto_front = _extract_pareto_front(population, fitnesses, var_names)
        return list(self._pareto_front)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_pareto_front(self) -> list[ParetoSolution]:
        """Return the stored Pareto front from the last ``fit`` call."""
        return list(self._pareto_front)

    def best_equation(self) -> ParetoSolution | None:
        """Return the Pareto solution with the best accuracy-complexity balance.

        The "knee point" is the solution that minimises
        ``mse * complexity``.

        Returns
        -------
        ParetoSolution | None
            Best balanced solution, or *None* if no front exists.
        """
        if not self._pareto_front:
            return None

        best = None
        best_score = float("inf")
        for sol in self._pareto_front:
            score = sol.mse * sol.complexity
            if score < best_score:
                best_score = score
                best = sol
        return best


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _collect_nodes(
    root: ExprNode,
) -> list[tuple[ExprNode, ExprNode | None, str]]:
    """Collect all nodes as ``(node, parent, attr_name)`` tuples.

    ``attr_name`` is ``"left"`` or ``"right"`` — the attribute on the
    parent that points to this node.  For the root the parent is *None*
    and attr_name is ``""``.
    """
    result: list[tuple[ExprNode, ExprNode | None, str]] = []

    def _walk(node: ExprNode, parent: ExprNode | None, attr: str) -> None:
        result.append((node, parent, attr))
        if node.left is not None:
            _walk(node.left, node, "left")
        if node.right is not None:
            _walk(node.right, node, "right")

    _walk(root, None, "")
    return result


def _dominates(a_mse: float, a_cplx: int, b_mse: float, b_cplx: int) -> bool:
    """Check whether solution *a* dominates solution *b* in (MSE, complexity) space."""
    return (a_mse <= b_mse and a_cplx <= b_cplx) and (a_mse < b_mse or a_cplx < b_cplx)


def _extract_pareto_front(
    population: list[ExprNode],
    fitnesses: list[float],
    var_names: list[str] | None = None,
) -> list[ParetoSolution]:
    """Extract the Pareto front from a population.

    Non-dominated solutions in ``(MSE, complexity)`` space are returned.
    """
    candidates: list[tuple[int, float, int]] = []
    for i, tree in enumerate(population):
        mse = fitnesses[i]
        if not math.isfinite(mse):
            continue
        cplx = tree.complexity()
        candidates.append((i, mse, cplx))

    if not candidates:
        return []

    # Find non-dominated set
    front: list[ParetoSolution] = []
    for idx, mse, cplx in candidates:
        dominated = False
        for idx2, mse2, cplx2 in candidates:
            if idx2 == idx:
                continue
            if _dominates(mse2, cplx2, mse, cplx):
                dominated = True
                break
        if not dominated:
            tree = population[idx]
            front.append(ParetoSolution(
                expression=tree,
                mse=mse,
                complexity=cplx,
                equation_string=tree.to_string(var_names),
            ))

    # Sort by complexity for readability
    front.sort(key=lambda s: s.complexity)

    # Deduplicate by equation string
    seen: set[str] = set()
    unique: list[ParetoSolution] = []
    for sol in front:
        if sol.equation_string not in seen:
            seen.add(sol.equation_string)
            unique.append(sol)

    return unique

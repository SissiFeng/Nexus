"""Hypothesis testing, BIC scoring, and falsification.

Provides methods to evaluate hypotheses against data, compute Bayesian
Information Criterion scores, approximate Bayes factors, perform
sequential updates, and check falsification criteria.
"""

from __future__ import annotations

import math
import operator
import re

from optimization_copilot.hypothesis.models import (
    Evidence,
    Hypothesis,
    Prediction,
)


# ---------------------------------------------------------------------------
# Safe expression evaluator
# ---------------------------------------------------------------------------

# Supported tokens for the restricted expression parser
_ALLOWED_FUNCS: dict[str, object] = {
    "sin": math.sin,
    "cos": math.cos,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
    "pi": math.pi,
    "e": math.e,
}

# Token types
_TOKEN_NUMBER = "NUMBER"
_TOKEN_IDENT = "IDENT"
_TOKEN_OP = "OP"
_TOKEN_LPAREN = "LPAREN"
_TOKEN_RPAREN = "RPAREN"
_TOKEN_END = "END"

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<number>[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)
      | (?P<ident>[a-zA-Z_][a-zA-Z_0-9]*)
      | (?P<starstar>\*\*)
      | (?P<op>[+\-*/^])
      | (?P<lparen>\()
      | (?P<rparen>\))
    )\s*
    """,
    re.VERBOSE,
)


def _tokenize(expr: str) -> list[tuple[str, str]]:
    """Tokenize a simple math expression into (type, value) pairs."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        # Skip whitespace
        while pos < len(expr) and expr[pos].isspace():
            pos += 1
        if pos >= len(expr):
            break
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise ValueError(f"Unexpected character at position {pos}: {expr[pos:]!r}")
        if m.group("starstar"):
            tokens.append((_TOKEN_OP, "**"))
        elif m.group("number"):
            tokens.append((_TOKEN_NUMBER, m.group("number")))
        elif m.group("ident"):
            tokens.append((_TOKEN_IDENT, m.group("ident")))
        elif m.group("op"):
            tokens.append((_TOKEN_OP, m.group("op")))
        elif m.group("lparen"):
            tokens.append((_TOKEN_LPAREN, "("))
        elif m.group("rparen"):
            tokens.append((_TOKEN_RPAREN, ")"))
        pos = m.end()
    tokens.append((_TOKEN_END, ""))
    return tokens


class _Parser:
    """Recursive-descent parser for simple math expressions.

    Grammar (precedence low-to-high):
        expr     := term (('+' | '-') term)*
        term     := unary (('*' | '/') unary)*
        unary    := '-' unary | power
        power    := atom ('**' | '^') unary | atom
        atom     := NUMBER | IDENT '(' expr ')' | IDENT | '(' expr ')'
    """

    def __init__(
        self,
        tokens: list[tuple[str, str]],
        variables: dict[str, float],
    ) -> None:
        self.tokens = tokens
        self.pos = 0
        self.variables = variables

    def _peek(self) -> tuple[str, str]:
        return self.tokens[self.pos]

    def _advance(self) -> tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> float:
        result = self._expr()
        if self._peek()[0] != _TOKEN_END:
            raise ValueError(
                f"Unexpected token after expression: {self._peek()}"
            )
        return result

    def _expr(self) -> float:
        left = self._term()
        while self._peek() == (_TOKEN_OP, "+") or self._peek() == (_TOKEN_OP, "-"):
            op = self._advance()[1]
            right = self._term()
            if op == "+":
                left = left + right
            else:
                left = left - right
        return left

    def _term(self) -> float:
        left = self._unary()
        while self._peek() == (_TOKEN_OP, "*") or self._peek() == (_TOKEN_OP, "/"):
            op = self._advance()[1]
            right = self._unary()
            if op == "*":
                left = left * right
            else:
                if right == 0:
                    left = float("inf")
                else:
                    left = left / right
        return left

    def _unary(self) -> float:
        if self._peek() == (_TOKEN_OP, "-"):
            self._advance()
            return -self._unary()
        return self._power()

    def _power(self) -> float:
        base = self._atom()
        if self._peek() == (_TOKEN_OP, "**") or self._peek() == (_TOKEN_OP, "^"):
            self._advance()
            exp = self._unary()  # right-associative
            try:
                return base ** exp
            except (OverflowError, ValueError):
                return float("inf")
        return base

    def _atom(self) -> float:
        tok_type, tok_val = self._peek()
        if tok_type == _TOKEN_NUMBER:
            self._advance()
            return float(tok_val)
        if tok_type == _TOKEN_IDENT:
            self._advance()
            # Check for function call
            if self._peek()[0] == _TOKEN_LPAREN:
                # function call
                func = _ALLOWED_FUNCS.get(tok_val)
                if func is None:
                    raise ValueError(f"Unknown function: {tok_val}")
                if not callable(func):
                    raise ValueError(f"{tok_val} is not callable")
                self._advance()  # consume '('
                arg = self._expr()
                if self._peek()[0] != _TOKEN_RPAREN:
                    raise ValueError("Expected ')'")
                self._advance()  # consume ')'
                try:
                    return float(func(arg))  # type: ignore[operator]
                except (ValueError, OverflowError):
                    return float("inf")
            # Check for constant
            if tok_val in _ALLOWED_FUNCS:
                val = _ALLOWED_FUNCS[tok_val]
                if isinstance(val, (int, float)):
                    return float(val)
            # Variable lookup
            if tok_val in self.variables:
                return self.variables[tok_val]
            raise ValueError(f"Unknown variable: {tok_val}")
        if tok_type == _TOKEN_LPAREN:
            self._advance()
            result = self._expr()
            if self._peek()[0] != _TOKEN_RPAREN:
                raise ValueError("Expected ')'")
            self._advance()
            return result
        raise ValueError(f"Unexpected token: {self._peek()}")


def _safe_eval(
    equation: str, variables: dict[str, float]
) -> float:
    """Safely evaluate a simple symbolic math expression.

    Supports ``+``, ``-``, ``*``, ``/``, ``**`` (or ``^``), parentheses,
    variable names, numeric literals, and math functions
    (``sin``, ``cos``, ``exp``, ``log``, ``sqrt``, ``abs``).

    Raises ``ValueError`` for unrecognized tokens or variables.
    """
    tokens = _tokenize(equation)
    parser = _Parser(tokens, variables)
    return parser.parse()


# ---------------------------------------------------------------------------
# HypothesisTester
# ---------------------------------------------------------------------------


class HypothesisTester:
    """Test hypotheses against data using BIC and falsification."""

    def compute_bic(
        self,
        hypothesis: Hypothesis,
        X: list[list[float]],
        y: list[float],
        var_names: list[str] | None = None,
    ) -> float:
        """Compute Bayesian Information Criterion for a hypothesis.

        ``BIC = n * ln(RSS / n) + k * ln(n)``

        If the hypothesis has an equation, it is evaluated at each row
        of *X*.  Otherwise *n_parameters* is used as a proxy and RSS
        is computed from the mean prediction.

        Parameters
        ----------
        hypothesis : Hypothesis
            Hypothesis to score.
        X : list[list[float]]
            Feature matrix (row-major).
        y : list[float]
            Target values.
        var_names : list[str] | None
            Column names for *X*.  Falls back to ``x0, x1, ...``.

        Returns
        -------
        float
            BIC score (lower is better).
        """
        n = len(y)
        if n == 0:
            return float("inf")

        k = hypothesis.n_parameters

        if hypothesis.equation:
            rss = 0.0
            for i in range(n):
                pred = self._evaluate_equation(
                    hypothesis.equation, X[i], var_names
                )
                if not math.isfinite(pred):
                    return float("inf")
                rss += (y[i] - pred) ** 2
        else:
            # Fallback: use mean prediction
            y_mean = sum(y) / n
            rss = sum((yi - y_mean) ** 2 for yi in y)

        if rss <= 0:
            rss = 1e-300  # avoid log(0)

        bic = n * math.log(rss / n) + k * math.log(n)
        hypothesis.bic_score = bic
        return bic

    def _evaluate_equation(
        self,
        equation: str,
        x_row: list[float],
        var_names: list[str] | None = None,
    ) -> float:
        """Safely evaluate a simple symbolic equation at one data point.

        Supports ``+, -, *, /, **, ^``, parentheses, and math functions.
        Variables are referenced by name (e.g. ``x0``, ``x1``, ...) or
        by the names given in *var_names*.
        """
        variables: dict[str, float] = {}
        for i, val in enumerate(x_row):
            variables[f"x{i}"] = val
        if var_names is not None:
            for i, name in enumerate(var_names):
                if i < len(x_row):
                    variables[name] = x_row[i]

        try:
            return _safe_eval(equation, variables)
        except (ValueError, ZeroDivisionError, OverflowError):
            return float("inf")

    def bayes_factor(
        self,
        h1: Hypothesis,
        h2: Hypothesis,
        X: list[list[float]],
        y: list[float],
        var_names: list[str] | None = None,
    ) -> float:
        """Approximate Bayes factor between two hypotheses.

        ``BF = exp((BIC_h2 - BIC_h1) / 2)``

        BF > 1 favours *h1*, BF < 1 favours *h2*.
        """
        bic1 = self.compute_bic(h1, X, y, var_names)
        bic2 = self.compute_bic(h2, X, y, var_names)
        try:
            return math.exp((bic2 - bic1) / 2.0)
        except OverflowError:
            return float("inf")

    def sequential_update(
        self,
        hypothesis: Hypothesis,
        new_x: list[float],
        new_y: float,
        var_names: list[str] | None = None,
    ) -> Hypothesis:
        """Update *hypothesis* with a single new observation.

        1. Generate a prediction from the hypothesis equation.
        2. Compare with the observed value.
        3. Create :class:`Evidence` and add it to the hypothesis.

        Returns the (mutated) hypothesis for convenience.
        """
        if hypothesis.equation:
            pred_val = self._evaluate_equation(
                hypothesis.equation, new_x, var_names
            )
        else:
            pred_val = 0.0

        # Simple confidence interval: +/- 1.0 (default band)
        ci = (pred_val - 1.0, pred_val + 1.0)
        prediction = Prediction(
            hypothesis_id=hypothesis.id,
            variable="y",
            predicted_value=pred_val,
            confidence_interval=ci,
        )
        evidence = Evidence(prediction=prediction, observed_value=new_y)
        hypothesis.add_evidence(evidence)
        return hypothesis

    def check_falsification(
        self, hypothesis: Hypothesis, threshold: int = 3
    ) -> bool:
        """Return ``True`` if the hypothesis should be falsified.

        Falsified when the last *threshold* consecutive evidence entries
        are all refuting (outside the confidence interval).
        """
        if len(hypothesis.evidence) < threshold:
            return False
        recent = hypothesis.evidence[-threshold:]
        return all(not e.within_ci for e in recent)

    def compare_all(
        self,
        hypotheses: list[Hypothesis],
        X: list[list[float]],
        y: list[float],
        var_names: list[str] | None = None,
    ) -> list[dict]:
        """Rank hypotheses by BIC.

        Returns a sorted list of dicts with ``"hypothesis_id"``,
        ``"bic"``, ``"rank"``, and ``"bayes_factor_vs_best"``.
        """
        scored: list[tuple[Hypothesis, float]] = []
        for h in hypotheses:
            bic = self.compute_bic(h, X, y, var_names)
            scored.append((h, bic))

        scored.sort(key=lambda x: x[1])

        results: list[dict] = []
        best_bic = scored[0][1] if scored else 0.0
        for rank, (h, bic) in enumerate(scored, start=1):
            delta = bic - best_bic
            try:
                bf_vs_best = math.exp(delta / 2.0)
            except OverflowError:
                bf_vs_best = float("inf")
            results.append(
                {
                    "hypothesis_id": h.id,
                    "bic": bic,
                    "rank": rank,
                    "bayes_factor_vs_best": bf_vs_best,
                }
            )
        return results

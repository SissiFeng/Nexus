"""Theory (mechanistic) models for hybrid optimization.

Provides abstract base class and concrete implementations of
physics-based models: Arrhenius kinetics, Michaelis-Menten enzyme
kinetics, power-law relationships, and ODE-based models.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable


class TheoryModel(ABC):
    """Abstract base for mechanistic/theory models.

    Subclasses encode domain knowledge as deterministic predictions.
    The residual between theory and observations is learned by a GP.
    """

    @abstractmethod
    def predict(self, X: list[list[float]]) -> list[float]:
        """Deterministic theory prediction for each row in X.

        Parameters
        ----------
        X : list[list[float]]
            Input matrix where each row is a feature vector.

        Returns
        -------
        list[float]
            Predicted value for each input row.
        """

    @abstractmethod
    def n_parameters(self) -> int:
        """Return the number of model parameters."""

    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Return the names of model parameters."""


class ArrheniusModel(TheoryModel):
    """Arrhenius kinetics: Rate = A * exp(-Ea / (R * T)).

    X rows: the column at ``temp_index`` is temperature T (Kelvin).

    Parameters
    ----------
    A : float
        Pre-exponential factor.
    Ea : float
        Activation energy (J/mol).
    R : float
        Gas constant (default 8.314 J/(mol*K)).
    temp_index : int
        Column index in X for temperature (default 0).
    """

    def __init__(
        self,
        A: float = 1.0,
        Ea: float = 50000.0,
        R: float = 8.314,
        temp_index: int = 0,
    ) -> None:
        self.A = A
        self.Ea = Ea
        self.R = R
        self.temp_index = temp_index

    def predict(self, X: list[list[float]]) -> list[float]:
        results: list[float] = []
        for row in X:
            T = row[self.temp_index]
            if T <= 0.0:
                # Avoid division by zero / negative temperature
                results.append(0.0)
            else:
                exponent = -self.Ea / (self.R * T)
                # Clamp exponent to avoid overflow
                exponent = max(exponent, -700.0)
                results.append(self.A * math.exp(exponent))
        return results

    def n_parameters(self) -> int:
        return 3

    def parameter_names(self) -> list[str]:
        return ["A", "Ea", "R"]


class MichaelisMentenModel(TheoryModel):
    """Michaelis-Menten kinetics: V = Vmax * S / (Km + S).

    X rows: the column at ``substrate_index`` is substrate
    concentration S.

    Parameters
    ----------
    Vmax : float
        Maximum reaction rate.
    Km : float
        Michaelis constant (substrate concentration at half Vmax).
    substrate_index : int
        Column index in X for substrate concentration (default 0).
    """

    def __init__(
        self,
        Vmax: float = 1.0,
        Km: float = 1.0,
        substrate_index: int = 0,
    ) -> None:
        self.Vmax = Vmax
        self.Km = Km
        self.substrate_index = substrate_index

    def predict(self, X: list[list[float]]) -> list[float]:
        results: list[float] = []
        for row in X:
            S = row[self.substrate_index]
            denom = self.Km + S
            if abs(denom) < 1e-15:
                results.append(0.0)
            else:
                results.append(self.Vmax * S / denom)
        return results

    def n_parameters(self) -> int:
        return 2

    def parameter_names(self) -> list[str]:
        return ["Vmax", "Km"]


class PowerLawModel(TheoryModel):
    """Power-law relationship: y = a * x^b.

    X rows: the column at ``var_index`` is the independent variable x.

    Parameters
    ----------
    a : float
        Scale coefficient.
    b : float
        Exponent.
    var_index : int
        Column index in X for the independent variable (default 0).
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        var_index: int = 0,
    ) -> None:
        self.a = a
        self.b = b
        self.var_index = var_index

    def predict(self, X: list[list[float]]) -> list[float]:
        results: list[float] = []
        for row in X:
            x = row[self.var_index]
            if x < 0.0 and self.b != int(self.b):
                # Fractional power of negative number: use abs
                results.append(self.a * (-((-x) ** self.b)))
            else:
                results.append(self.a * (x ** self.b))
        return results

    def n_parameters(self) -> int:
        return 2

    def parameter_names(self) -> list[str]:
        return ["a", "b"]


class ODEModel(TheoryModel):
    """Wraps RK4Solver for mechanistic ODE models.

    For each row x in X, solves the ODE from ``t_span[0]`` to
    ``x[0]`` (or the full ``t_span`` if configured) and extracts
    the state component at ``output_index`` at the final time.

    Parameters
    ----------
    rhs_fn : callable
        Right-hand side function ``f(t, y) -> dy/dt``.
    y0 : list[float]
        Initial condition vector.
    t_span : tuple[float, float]
        Default integration interval ``(t0, t1)``.
    output_index : int
        Which component of the state vector to return (default 0).
    n_steps : int
        Number of RK4 integration steps (default 100).
    """

    def __init__(
        self,
        rhs_fn: Callable[[float, list[float]], list[float]],
        y0: list[float],
        t_span: tuple[float, float] = (0.0, 1.0),
        output_index: int = 0,
        n_steps: int = 100,
    ) -> None:
        self._rhs_fn = rhs_fn
        self._y0 = list(y0)
        self._t_span = t_span
        self._output_index = output_index
        self._n_steps = n_steps

    def predict(self, X: list[list[float]]) -> list[float]:
        from optimization_copilot.physics.ode_solver import RK4Solver

        solver = RK4Solver()
        results: list[float] = []
        for row in X:
            # Use X[i][0] as the time endpoint
            t_end = row[0]
            t0 = self._t_span[0]
            if t_end <= t0:
                # No integration needed; return initial condition
                results.append(self._y0[self._output_index])
                continue
            _, y_values = solver.solve(
                self._rhs_fn,
                self._y0,
                (t0, t_end),
                n_steps=self._n_steps,
            )
            results.append(y_values[-1][self._output_index])
        return results

    def n_parameters(self) -> int:
        return len(self._y0)

    def parameter_names(self) -> list[str]:
        return [f"y0_{i}" for i in range(len(self._y0))]

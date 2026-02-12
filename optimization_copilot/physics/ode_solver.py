"""Pure-Python ODE solvers for physics-informed modeling.

Provides a 4th-order Runge-Kutta integrator and steady-state solver.
All operations use plain Python lists to avoid external dependencies.
"""

from __future__ import annotations

import math
from typing import Callable


class RK4Solver:
    """4th-order Runge-Kutta ODE integrator, pure Python.

    Solves initial value problems of the form:

        dy/dt = f(t, y)

    using the classical RK4 method with fixed step size.
    """

    def solve(
        self,
        f: Callable[[float, list[float]], list[float]],
        y0: list[float],
        t_span: tuple[float, float],
        n_steps: int = 100,
    ) -> tuple[list[float], list[list[float]]]:
        """Solve dy/dt = f(t, y) from t_span[0] to t_span[1].

        Parameters
        ----------
        f : callable
            Right-hand side function with signature
            ``f(t: float, y: list[float]) -> list[float]``.
        y0 : list[float]
            Initial state vector.
        t_span : tuple[float, float]
            Integration interval ``(t0, t1)``.
        n_steps : int
            Number of integration steps (default 100).

        Returns
        -------
        tuple[list[float], list[list[float]]]
            ``(t_values, y_values)`` where ``y_values[i]`` is the
            state vector at ``t_values[i]``.

        Notes
        -----
        RK4 update rule::

            h = (t1 - t0) / n_steps
            k1 = h * f(tn, yn)
            k2 = h * f(tn + h/2, yn + k1/2)
            k3 = h * f(tn + h/2, yn + k2/2)
            k4 = h * f(tn + h, yn + k3)
            yn+1 = yn + (k1 + 2*k2 + 2*k3 + k4) / 6
        """
        t0, t1 = t_span
        h = (t1 - t0) / max(n_steps, 1)
        dim = len(y0)

        t_values: list[float] = [t0]
        y_values: list[list[float]] = [list(y0)]

        y = list(y0)
        t = t0

        for _ in range(n_steps):
            k1 = f(t, y)
            k1 = [h * ki for ki in k1]

            y_temp = [y[i] + 0.5 * k1[i] for i in range(dim)]
            k2 = f(t + 0.5 * h, y_temp)
            k2 = [h * ki for ki in k2]

            y_temp = [y[i] + 0.5 * k2[i] for i in range(dim)]
            k3 = f(t + 0.5 * h, y_temp)
            k3 = [h * ki for ki in k3]

            y_temp = [y[i] + k3[i] for i in range(dim)]
            k4 = f(t + h, y_temp)
            k4 = [h * ki for ki in k4]

            y = [
                y[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
                for i in range(dim)
            ]
            t = t + h

            t_values.append(t)
            y_values.append(list(y))

        return t_values, y_values

    def solve_to_steady_state(
        self,
        f: Callable[[float, list[float]], list[float]],
        y0: list[float],
        dt: float = 0.01,
        max_steps: int = 10000,
        tol: float = 1e-6,
    ) -> list[float]:
        """Integrate until steady state is reached.

        Integrates forward in time using RK4 steps of size *dt* until
        the norm of the derivative ``||f(t, y)||`` is below *tol* or
        *max_steps* is reached.

        Parameters
        ----------
        f : callable
            Right-hand side function with signature
            ``f(t: float, y: list[float]) -> list[float]``.
        y0 : list[float]
            Initial state vector.
        dt : float
            Time step size (default 0.01).
        max_steps : int
            Maximum number of integration steps (default 10000).
        tol : float
            Convergence tolerance on ``||dy/dt||`` (default 1e-6).

        Returns
        -------
        list[float]
            The steady-state solution vector.
        """
        dim = len(y0)
        y = list(y0)
        t = 0.0
        h = dt

        for _ in range(max_steps):
            # Check convergence before stepping
            dy = f(t, y)
            norm_dy = math.sqrt(sum(d * d for d in dy))
            if norm_dy < tol:
                break

            # RK4 step
            k1 = [h * di for di in dy]

            y_temp = [y[i] + 0.5 * k1[i] for i in range(dim)]
            k2_raw = f(t + 0.5 * h, y_temp)
            k2 = [h * ki for ki in k2_raw]

            y_temp = [y[i] + 0.5 * k2[i] for i in range(dim)]
            k3_raw = f(t + 0.5 * h, y_temp)
            k3 = [h * ki for ki in k3_raw]

            y_temp = [y[i] + k3[i] for i in range(dim)]
            k4_raw = f(t + h, y_temp)
            k4 = [h * ki for ki in k4_raw]

            y = [
                y[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
                for i in range(dim)
            ]
            t += h

        return y

    def __repr__(self) -> str:
        return "RK4Solver()"

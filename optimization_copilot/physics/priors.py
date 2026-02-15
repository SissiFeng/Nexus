"""Physics-informed prior mean functions for Gaussian processes.

Each prior encodes a well-known physical relationship as a callable
that maps input matrices to prior mean vectors. Pure Python stdlib only.
"""

from __future__ import annotations

import math


class ArrheniusPrior:
    """Arrhenius equation prior mean function.

    Models temperature-dependent reaction rates:

        m(T) = A * exp(-Ea / (R * T))

    Parameters
    ----------
    A : float
        Pre-exponential factor (default 1.0).
    Ea : float
        Activation energy in J/mol (default 50000.0).
    R : float
        Universal gas constant in J/(mol*K) (default 8.314).
    """

    def __init__(
        self,
        A: float = 1.0,
        Ea: float = 50000.0,
        R: float = 8.314,
    ) -> None:
        self.A = A
        self.Ea = Ea
        self.R = R

    def __call__(
        self,
        X: list[list[float]],
        temp_index: int = 0,
    ) -> list[float]:
        """Return prior mean for each row in X using temperature at temp_index.

        Parameters
        ----------
        X : list[list[float]]
            Input matrix where each row is an observation.
        temp_index : int
            Column index of the temperature variable (default 0).

        Returns
        -------
        list[float]
            Prior mean values, one per row.
        """
        result: list[float] = []
        for row in X:
            T = row[temp_index]
            if T <= 0.0:
                # Avoid division by zero / negative temperature
                result.append(0.0)
            else:
                exponent = -self.Ea / (self.R * T)
                # Clamp exponent to avoid overflow
                exponent = max(exponent, -500.0)
                result.append(self.A * math.exp(exponent))
        return result

    def gradient(
        self,
        X: list[list[float]],
        temp_index: int = 0,
    ) -> list[float]:
        """Derivative d/dT of the Arrhenius function.

        d/dT [A * exp(-Ea/(R*T))] = A * (Ea / (R * T^2)) * exp(-Ea/(R*T))

        Parameters
        ----------
        X : list[list[float]]
            Input matrix.
        temp_index : int
            Column index of the temperature variable (default 0).

        Returns
        -------
        list[float]
            Gradient values, one per row.
        """
        result: list[float] = []
        for row in X:
            T = row[temp_index]
            if T <= 0.0:
                result.append(0.0)
            else:
                exponent = -self.Ea / (self.R * T)
                exponent = max(exponent, -500.0)
                grad = self.A * (self.Ea / (self.R * T * T)) * math.exp(exponent)
                result.append(grad)
        return result

    def __repr__(self) -> str:
        return f"ArrheniusPrior(A={self.A}, Ea={self.Ea}, R={self.R})"


class MichaelisMentenPrior:
    """Michaelis-Menten kinetics prior mean function.

    Models enzyme-catalyzed reaction rates:

        m(S) = Vmax * S / (Km + S)

    Parameters
    ----------
    Vmax : float
        Maximum reaction rate (default 1.0).
    Km : float
        Michaelis constant (substrate concentration at half-max rate)
        (default 1.0).
    """

    def __init__(self, Vmax: float = 1.0, Km: float = 1.0) -> None:
        self.Vmax = Vmax
        self.Km = Km

    def __call__(
        self,
        X: list[list[float]],
        substrate_index: int = 0,
    ) -> list[float]:
        """Return prior mean for each row in X using substrate at substrate_index.

        Parameters
        ----------
        X : list[list[float]]
            Input matrix where each row is an observation.
        substrate_index : int
            Column index of the substrate concentration (default 0).

        Returns
        -------
        list[float]
            Prior mean values, one per row.
        """
        result: list[float] = []
        for row in X:
            S = row[substrate_index]
            if S < 0.0:
                result.append(0.0)
            else:
                denom = self.Km + S
                if abs(denom) < 1e-12:
                    result.append(0.0)
                else:
                    result.append(self.Vmax * S / denom)
        return result

    def gradient(
        self,
        X: list[list[float]],
        substrate_index: int = 0,
    ) -> list[float]:
        """Derivative d/dS of the Michaelis-Menten function.

        d/dS [Vmax * S / (Km + S)] = Vmax * Km / (Km + S)^2

        Parameters
        ----------
        X : list[list[float]]
            Input matrix.
        substrate_index : int
            Column index of the substrate concentration (default 0).

        Returns
        -------
        list[float]
            Gradient values, one per row.
        """
        result: list[float] = []
        for row in X:
            S = row[substrate_index]
            denom = (self.Km + S) ** 2
            if abs(denom) < 1e-12 or S < 0.0:
                result.append(0.0)
            else:
                result.append(self.Vmax * self.Km / denom)
        return result

    def __repr__(self) -> str:
        return f"MichaelisMentenPrior(Vmax={self.Vmax}, Km={self.Km})"


class PowerLawPrior:
    """Power-law prior mean function.

    Models power-law relationships:

        m(x) = a * x^b

    Parameters
    ----------
    a : float
        Coefficient (default 1.0).
    b : float
        Exponent (default 1.0).
    """

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        self.a = a
        self.b = b

    def __call__(
        self,
        X: list[list[float]],
        var_index: int = 0,
    ) -> list[float]:
        """Return prior mean for each row in X using variable at var_index.

        Parameters
        ----------
        X : list[list[float]]
            Input matrix where each row is an observation.
        var_index : int
            Column index of the variable (default 0).

        Returns
        -------
        list[float]
            Prior mean values, one per row.
        """
        result: list[float] = []
        for row in X:
            x = row[var_index]
            if x <= 0.0 and self.b != int(self.b):
                # Fractional exponent of non-positive number is undefined
                result.append(0.0)
            elif x == 0.0:
                result.append(0.0)
            else:
                result.append(self.a * (abs(x) ** self.b) * (1.0 if x > 0 else (-1.0 if int(self.b) % 2 != 0 else 1.0)))
        return result

    def gradient(
        self,
        X: list[list[float]],
        var_index: int = 0,
    ) -> list[float]:
        """Derivative d/dx of the power-law function.

        d/dx [a * x^b] = a * b * x^(b-1)

        Parameters
        ----------
        X : list[list[float]]
            Input matrix.
        var_index : int
            Column index of the variable (default 0).

        Returns
        -------
        list[float]
            Gradient values, one per row.
        """
        result: list[float] = []
        for row in X:
            x = row[var_index]
            if x <= 0.0:
                result.append(0.0)
            elif self.b == 0.0:
                result.append(0.0)
            else:
                result.append(self.a * self.b * (x ** (self.b - 1.0)))
        return result

    def __repr__(self) -> str:
        return f"PowerLawPrior(a={self.a}, b={self.b})"

"""VSUP (Value-Suppressing Uncertainty Palettes) colormap module.

Implements perceptual colormaps where high uncertainty suppresses saturation,
pushing colours toward neutral gray.  Pure Python -- no external dependencies.

References
----------
Michael Correll, Dominik Moritz, Jeffrey Heer.
"Value-Suppressing Uncertainty Palettes", CHI 2018.
"""

from __future__ import annotations


# ── Colormap stop tables ─────────────────────────────────────────────────────
# Each entry maps a normalised position [0, 1] to an (R, G, B) triplet.

_COLORMAP_STOPS: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    "viridis": [
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ],
    "plasma": [
        (0.00, (13, 8, 135)),
        (0.25, (126, 3, 168)),
        (0.50, (204, 71, 120)),
        (0.75, (248, 149, 64)),
        (1.00, (240, 249, 33)),
    ],
    "inferno": [
        (0.00, (0, 0, 4)),
        (0.25, (87, 16, 110)),
        (0.50, (188, 55, 84)),
        (0.75, (249, 142, 9)),
        (1.00, (252, 255, 164)),
    ],
}

_GRAY = 200  # neutral gray target for full suppression


# ── Helper ────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *v* to the closed interval [*lo*, *hi*]."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _lerp(a: int, b: int, t: float) -> int:
    """Linearly interpolate between integers *a* and *b* by factor *t*."""
    return int(round(a + (b - a) * t))


def color_to_hex(r: int, g: int, b: int, a: int = 255) -> str:
    """Return a CSS hex colour string.

    Returns ``"#RRGGBB"`` when *a* is 255 (fully opaque),
    otherwise ``"#RRGGBBAA"``.
    """
    if a == 255:
        return f"#{r:02X}{g:02X}{b:02X}"
    return f"#{r:02X}{g:02X}{b:02X}{a:02X}"


# ── VSUPColorMap ──────────────────────────────────────────────────────────────

class VSUPColorMap:
    """Value-Suppressing Uncertainty Palette.

    Maps a ``(value, uncertainty)`` pair to an RGBA colour.  Low uncertainty
    preserves the full base colour; high uncertainty suppresses saturation,
    blending toward neutral gray.

    Parameters
    ----------
    value_cmap : str
        Name of the base colormap (``"viridis"``, ``"plasma"``, ``"inferno"``).
    uncertainty_range : tuple[float, float]
        Reserved for future use (input normalisation bounds).
    """

    def __init__(
        self,
        value_cmap: str = "viridis",
        uncertainty_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        if value_cmap not in _COLORMAP_STOPS:
            raise ValueError(
                f"Unknown colormap {value_cmap!r}. "
                f"Choose from {sorted(_COLORMAP_STOPS)}."
            )
        self.value_cmap = value_cmap
        self.uncertainty_range = uncertainty_range
        self._stops = _COLORMAP_STOPS[value_cmap]

    # -- public API -----------------------------------------------------------

    def map(self, value: float, uncertainty: float) -> tuple[int, int, int, int]:
        """Return an RGBA tuple (each component 0-255).

        Parameters
        ----------
        value : float
            Normalised value in [0, 1].  Clamped if outside range.
        uncertainty : float
            Normalised uncertainty in [0, 1].  Clamped if outside range.
            0 = full colour, 1 = fully gray (maximum suppression).
        """
        value = _clamp(value)
        uncertainty = _clamp(uncertainty)

        r, g, b = self._value_to_rgb(value)

        # Suppress saturation proportional to uncertainty.
        suppress = uncertainty  # 0 -> keep colour, 1 -> all gray
        r = _lerp(r, _GRAY, suppress)
        g = _lerp(g, _GRAY, suppress)
        b = _lerp(b, _GRAY, suppress)

        return (r, g, b, 255)

    def batch_map(
        self,
        values: list[float],
        uncertainties: list[float],
    ) -> list[tuple[int, int, int, int]]:
        """Vectorised version of :meth:`map`.

        Parameters
        ----------
        values : list[float]
            Sequence of normalised values.
        uncertainties : list[float]
            Sequence of normalised uncertainties (same length as *values*).

        Returns
        -------
        list[tuple[int, int, int, int]]
            One RGBA tuple per input pair.
        """
        return [
            self.map(v, u) for v, u in zip(values, uncertainties)
        ]

    # -- internal -------------------------------------------------------------

    def _value_to_rgb(self, value: float) -> tuple[int, int, int]:
        """Interpolate the base colormap at *value* (already clamped)."""
        stops = self._stops

        # Exact-or-past last stop.
        if value >= stops[-1][0]:
            return stops[-1][1]

        # Walk stops to find the surrounding interval.
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t0 <= value <= t1:
                t = (value - t0) / (t1 - t0) if t1 != t0 else 0.0
                return (
                    _lerp(c0[0], c1[0], t),
                    _lerp(c0[1], c1[1], t),
                    _lerp(c0[2], c1[2], t),
                )

        # Fallback (should not be reached after clamping).
        return stops[0][1]  # pragma: no cover

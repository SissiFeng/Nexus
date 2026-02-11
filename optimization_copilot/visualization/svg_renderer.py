"""Minimal pure-Python SVG builder — zero external dependencies.

Produces valid SVG XML strings suitable for embedding in HTML reports
or saving as standalone ``.svg`` files.  All styling uses inline
attributes for maximum portability (no CSS / JS).
"""

from __future__ import annotations

from typing import Any


def _attr(key: str, value: Any) -> str:
    """Format a single SVG attribute."""
    return f'{key}="{value}"'


def _attrs(**kwargs: Any) -> str:
    """Format keyword arguments as SVG attribute string."""
    parts: list[str] = []
    for k, v in kwargs.items():
        if v is None:
            continue
        # Convert Python snake_case to SVG kebab-case.
        svg_key = k.replace("_", "-")
        parts.append(_attr(svg_key, v))
    return " ".join(parts)


class SVGCanvas:
    """Accumulates SVG elements and serialises them to an XML string.

    Parameters
    ----------
    width : int
        Canvas width in pixels.
    height : int
        Canvas height in pixels.
    background : str | None
        Optional background colour (CSS colour string).
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        background: str | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.background = background
        self._elements: list[str] = []
        self._defs: list[str] = []

    # -- primitive drawing methods -------------------------------------------

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float | None = None,
        opacity: float | None = None,
        rx: float | None = None,
        ry: float | None = None,
    ) -> None:
        """Add a rectangle."""
        a = _attrs(
            x=x, y=y, width=width, height=height,
            fill=fill, stroke=stroke,
            stroke_width=stroke_width, opacity=opacity,
            rx=rx, ry=ry,
        )
        self._elements.append(f"<rect {a}/>")

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float | None = None,
        opacity: float | None = None,
    ) -> None:
        """Add a circle."""
        a = _attrs(
            cx=cx, cy=cy, r=r,
            fill=fill, stroke=stroke,
            stroke_width=stroke_width, opacity=opacity,
        )
        self._elements.append(f"<circle {a}/>")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = "black",
        stroke_width: float = 1,
        stroke_dasharray: str | None = None,
        opacity: float | None = None,
    ) -> None:
        """Add a line segment."""
        a = _attrs(
            x1=x1, y1=y1, x2=x2, y2=y2,
            stroke=stroke, stroke_width=stroke_width,
            stroke_dasharray=stroke_dasharray, opacity=opacity,
        )
        self._elements.append(f"<line {a}/>")

    def polyline(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str = "none",
        stroke: str = "black",
        stroke_width: float = 1,
        opacity: float | None = None,
    ) -> None:
        """Add a polyline (open path through *points*)."""
        pts = " ".join(f"{x},{y}" for x, y in points)
        a = _attrs(
            fill=fill, stroke=stroke,
            stroke_width=stroke_width, opacity=opacity,
        )
        self._elements.append(f'<polyline points="{pts}" {a}/>')

    def polygon(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float | None = None,
        opacity: float | None = None,
    ) -> None:
        """Add a closed polygon."""
        pts = " ".join(f"{x},{y}" for x, y in points)
        a = _attrs(
            fill=fill, stroke=stroke,
            stroke_width=stroke_width, opacity=opacity,
        )
        self._elements.append(f'<polygon points="{pts}" {a}/>')

    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        font_size: float = 12,
        font_family: str = "sans-serif",
        fill: str = "black",
        text_anchor: str | None = None,
        dominant_baseline: str | None = None,
        transform: str | None = None,
    ) -> None:
        """Add a text element."""
        a = _attrs(
            x=x, y=y,
            font_size=font_size, font_family=font_family,
            fill=fill, text_anchor=text_anchor,
            dominant_baseline=dominant_baseline,
            transform=transform,
        )
        # Escape basic XML entities in content.
        safe = (
            content
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        self._elements.append(f"<text {a}>{safe}</text>")

    def path(
        self,
        d: str,
        *,
        fill: str = "none",
        stroke: str = "black",
        stroke_width: float = 1,
        opacity: float | None = None,
    ) -> None:
        """Add an SVG path element with a raw *d* attribute string."""
        a = _attrs(
            fill=fill, stroke=stroke,
            stroke_width=stroke_width, opacity=opacity,
        )
        self._elements.append(f'<path d="{d}" {a}/>')

    # -- grouping & structure ------------------------------------------------

    def group_start(self, **kwargs: Any) -> None:
        """Open a ``<g>`` group.  Close with :meth:`group_end`."""
        a = _attrs(**kwargs)
        self._elements.append(f"<g {a}>")

    def group_end(self) -> None:
        """Close a ``<g>`` group."""
        self._elements.append("</g>")

    def add_def(self, definition: str) -> None:
        """Add a raw string to the ``<defs>`` block."""
        self._defs.append(definition)

    def raw(self, svg_fragment: str) -> None:
        """Insert a raw SVG fragment (use with care)."""
        self._elements.append(svg_fragment)

    # -- serialisation -------------------------------------------------------

    def to_string(self) -> str:
        """Produce the complete SVG XML string."""
        parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">',
        ]

        # <defs> block.
        if self._defs:
            parts.append("<defs>")
            parts.extend(self._defs)
            parts.append("</defs>")

        # Background rect.
        if self.background:
            parts.append(
                f'<rect width="{self.width}" height="{self.height}" '
                f'fill="{self.background}"/>'
            )

        # Child elements.
        parts.extend(self._elements)
        parts.append("</svg>")
        return "\n".join(parts)

"""Core data models for the visualization layer.

Defines ``PlotData`` (the universal chart container) and ``SurrogateModel``
(the protocol any predictive model must satisfy to integrate with SHAP,
hexbin, and design-space visualizations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class PlotData:
    """Universal container for a single chart / visualization.

    Parameters
    ----------
    plot_type : str
        Identifier for the chart type, e.g. ``"shap_waterfall"``,
        ``"hexbin"``, ``"vsup_heatmap"``.
    data : dict[str, Any]
        Chart-specific data payload (series, points, polygons, …).
    metadata : dict[str, Any]
        Rendering hints, labels, axis config, etc.
    svg : str | None
        Pre-rendered SVG string (optional — may be populated lazily).
    """

    plot_type: str
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    svg: str | None = None

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "plot_type": self.plot_type,
            "data": self.data,
            "metadata": self.metadata,
            "svg": self.svg,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlotData:
        """Deserialise from a dictionary."""
        return cls(
            plot_type=d["plot_type"],
            data=d.get("data", {}),
            metadata=d.get("metadata", {}),
            svg=d.get("svg"),
        )


@runtime_checkable
class SurrogateModel(Protocol):
    """Structural typing protocol for any predictive surrogate.

    Any object that exposes a ``predict(x) -> (mean, uncertainty)`` method
    satisfies this protocol — no inheritance required.
    """

    def predict(self, x: list[float]) -> tuple[float, float]:
        """Return ``(predicted_mean, predicted_uncertainty)`` for *x*."""
        ...

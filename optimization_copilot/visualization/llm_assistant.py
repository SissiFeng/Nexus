"""LLM-driven natural language visualization assistant.

Phase 5+ skeleton -- provides the interface and data structures for future
LLM integration.  Currently returns NotImplementedError for actual LLM calls
but provides the PlotSpec schema and validation logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from optimization_copilot.visualization.models import PlotData


@dataclass
class PlotSpec:
    """Structured specification for a chart, parseable from LLM output.

    This serves as the intermediate representation between natural language
    queries and the PlotData rendering format.
    """

    plot_type: str  # target chart type
    filters: dict[str, Any] = field(default_factory=dict)  # trial filters
    parameters: list[str] = field(default_factory=list)  # which params to show
    color_by: str | None = None  # colouring strategy
    aggregation: str | None = None  # aggregation mode
    title: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlotSpec:
        """Deserialise from a dictionary."""
        return cls(
            plot_type=d["plot_type"],
            filters=d.get("filters", {}),
            parameters=d.get("parameters", []),
            color_by=d.get("color_by"),
            aggregation=d.get("aggregation"),
            title=d.get("title"),
        )


class LLMVisualizationAssistant:
    """LLM-driven natural language visualization.

    Planned workflow:
    1. User inputs natural language query
    2. LLM parses into PlotSpec (JSON Schema)
    3. VisualizationEngine renders PlotSpec -> PlotData
    """

    SUPPORTED_PLOT_TYPES: list[str] = [
        "scatter",
        "line",
        "bar",
        "heatmap",
        "parallel_coordinates",
        "shap_waterfall",
        "shap_beeswarm",
        "hexbin",
        "space_filling_metrics",
        "latent_space_exploration",
        "isom_landscape",
        "sdl_status_dashboard",
    ]

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def query_to_plot(self, query: str, study_data: dict) -> PlotData:
        """Convert natural language query to rendered PlotData.

        Currently raises NotImplementedError -- will be implemented when
        LLM integration is available (Phase 5+).
        """
        plot_spec = self._parse_query(query, study_data)
        return self._render(plot_spec)

    def _parse_query(self, query: str, study_data: dict) -> PlotSpec:
        """Parse natural language into PlotSpec.

        Phase 5+: This will call an LLM API.
        Currently: raises NotImplementedError.
        """
        raise NotImplementedError(
            "LLM integration not yet available. "
            "Use PlotSpec directly for programmatic chart creation."
        )

    def _render(self, spec: PlotSpec) -> PlotData:
        """Render a PlotSpec into PlotData.

        This is a dispatcher that routes to the appropriate chart function
        based on spec.plot_type.
        """
        if spec.plot_type not in self.SUPPORTED_PLOT_TYPES:
            raise ValueError(f"Unsupported plot type: {spec.plot_type}")
        return PlotData(
            plot_type=spec.plot_type,
            data={"spec": spec.to_dict()},
            metadata={"source": "llm_assistant", "title": spec.title},
        )

    def validate_spec(self, spec: PlotSpec) -> list[str]:
        """Validate a PlotSpec and return list of error messages (empty = valid)."""
        errors: list[str] = []
        if not spec.plot_type:
            errors.append("plot_type is required")
        if spec.plot_type and spec.plot_type not in self.SUPPORTED_PLOT_TYPES:
            errors.append(f"Unsupported plot_type: {spec.plot_type}")
        return errors

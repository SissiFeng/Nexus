"""Tests for visualization.llm_assistant (PlotSpec + LLMVisualizationAssistant)."""

from __future__ import annotations

import pytest

from optimization_copilot.visualization.llm_assistant import (
    LLMVisualizationAssistant,
    PlotSpec,
)
from optimization_copilot.visualization.models import PlotData


# -- PlotSpec ----------------------------------------------------------------


class TestPlotSpec:
    def test_defaults(self):
        spec = PlotSpec(plot_type="scatter")
        assert spec.plot_type == "scatter"
        assert spec.filters == {}
        assert spec.parameters == []
        assert spec.color_by is None
        assert spec.aggregation is None
        assert spec.title is None

    def test_to_dict(self):
        spec = PlotSpec(
            plot_type="bar",
            filters={"status": "complete"},
            parameters=["lr", "batch_size"],
            color_by="objective",
            aggregation="mean",
            title="My Chart",
        )
        d = spec.to_dict()
        assert d["plot_type"] == "bar"
        assert d["filters"] == {"status": "complete"}
        assert d["parameters"] == ["lr", "batch_size"]
        assert d["color_by"] == "objective"
        assert d["aggregation"] == "mean"
        assert d["title"] == "My Chart"

    def test_from_dict(self):
        d = {
            "plot_type": "heatmap",
            "filters": {"trial_id": 5},
            "parameters": ["x1"],
            "color_by": "feasibility",
            "aggregation": None,
            "title": "Heat",
        }
        spec = PlotSpec.from_dict(d)
        assert spec.plot_type == "heatmap"
        assert spec.filters == {"trial_id": 5}
        assert spec.parameters == ["x1"]
        assert spec.color_by == "feasibility"
        assert spec.title == "Heat"

    def test_from_dict_missing_optional(self):
        spec = PlotSpec.from_dict({"plot_type": "line"})
        assert spec.filters == {}
        assert spec.parameters == []
        assert spec.color_by is None
        assert spec.aggregation is None
        assert spec.title is None

    def test_roundtrip(self):
        spec = PlotSpec(
            plot_type="parallel_coordinates",
            filters={"status": "complete", "objective_lt": 0.5},
            parameters=["lr", "dropout", "hidden_dim"],
            color_by="objective",
            aggregation="median",
            title="Parallel Coords",
        )
        restored = PlotSpec.from_dict(spec.to_dict())
        assert restored.to_dict() == spec.to_dict()


# -- LLMVisualizationAssistant ----------------------------------------------


class TestLLMVisualizationAssistant:
    def test_construction_default(self):
        assistant = LLMVisualizationAssistant()
        assert assistant.api_key is None

    def test_construction_with_key(self):
        assistant = LLMVisualizationAssistant(api_key="sk-test-123")
        assert assistant.api_key == "sk-test-123"

    def test_supported_plot_types_nonempty(self):
        assert len(LLMVisualizationAssistant.SUPPORTED_PLOT_TYPES) > 0
        assert isinstance(LLMVisualizationAssistant.SUPPORTED_PLOT_TYPES, list)

    def test_query_to_plot_raises_not_implemented(self):
        assistant = LLMVisualizationAssistant()
        with pytest.raises(NotImplementedError, match="LLM integration not yet available"):
            assistant.query_to_plot("show me a scatter plot", {"trials": []})

    def test_render_valid_spec(self):
        assistant = LLMVisualizationAssistant()
        spec = PlotSpec(plot_type="scatter", title="Test Scatter")
        result = assistant._render(spec)
        assert isinstance(result, PlotData)
        assert result.plot_type == "scatter"
        assert result.data["spec"]["plot_type"] == "scatter"
        assert result.metadata["source"] == "llm_assistant"
        assert result.metadata["title"] == "Test Scatter"

    def test_render_invalid_plot_type(self):
        assistant = LLMVisualizationAssistant()
        spec = PlotSpec(plot_type="pie_chart_3d")
        with pytest.raises(ValueError, match="Unsupported plot type: pie_chart_3d"):
            assistant._render(spec)

    def test_validate_spec_valid(self):
        assistant = LLMVisualizationAssistant()
        spec = PlotSpec(plot_type="heatmap")
        errors = assistant.validate_spec(spec)
        assert errors == []

    def test_validate_spec_unsupported_type(self):
        assistant = LLMVisualizationAssistant()
        spec = PlotSpec(plot_type="unknown_chart")
        errors = assistant.validate_spec(spec)
        assert len(errors) == 1
        assert "Unsupported plot_type: unknown_chart" in errors[0]

    def test_validate_spec_empty_type(self):
        assistant = LLMVisualizationAssistant()
        spec = PlotSpec(plot_type="")
        errors = assistant.validate_spec(spec)
        assert "plot_type is required" in errors

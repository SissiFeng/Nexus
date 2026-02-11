"""Tests for visualization.models (PlotData + SurrogateModel protocol)."""

from __future__ import annotations

import pytest

from optimization_copilot.visualization.models import PlotData, SurrogateModel
from optimization_copilot.visualization.svg_renderer import SVGCanvas


# ── PlotData ────────────────────────────────────────────────────────────────

class TestPlotData:
    def test_basic_construction(self):
        pd = PlotData(plot_type="scatter", data={"x": [1, 2]}, metadata={"title": "t"})
        assert pd.plot_type == "scatter"
        assert pd.data == {"x": [1, 2]}
        assert pd.svg is None

    def test_defaults(self):
        pd = PlotData(plot_type="bar")
        assert pd.data == {}
        assert pd.metadata == {}
        assert pd.svg is None

    def test_with_svg(self):
        pd = PlotData(plot_type="line", svg="<svg></svg>")
        assert pd.svg == "<svg></svg>"

    def test_to_dict(self):
        pd = PlotData(plot_type="heatmap", data={"z": [[1]]}, metadata={"cmap": "v"})
        d = pd.to_dict()
        assert d["plot_type"] == "heatmap"
        assert d["data"] == {"z": [[1]]}
        assert d["metadata"] == {"cmap": "v"}
        assert d["svg"] is None

    def test_from_dict(self):
        d = {"plot_type": "hexbin", "data": {"cells": []}, "metadata": {}, "svg": "<svg/>"}
        pd = PlotData.from_dict(d)
        assert pd.plot_type == "hexbin"
        assert pd.svg == "<svg/>"

    def test_roundtrip(self):
        pd = PlotData(
            plot_type="shap_waterfall",
            data={"values": [0.1, -0.2]},
            metadata={"trial": 5},
            svg="<svg>test</svg>",
        )
        assert PlotData.from_dict(pd.to_dict()).to_dict() == pd.to_dict()

    def test_from_dict_missing_optional(self):
        pd = PlotData.from_dict({"plot_type": "empty"})
        assert pd.data == {}
        assert pd.metadata == {}
        assert pd.svg is None


# ── SurrogateModel protocol ─────────────────────────────────────────────────

class _MockSurrogate:
    """Satisfies the SurrogateModel protocol."""

    def predict(self, x: list[float]) -> tuple[float, float]:
        return (sum(x), 0.1)


class _BadModel:
    """Does NOT satisfy the protocol."""

    def evaluate(self, x: list[float]) -> float:
        return sum(x)


class TestSurrogateModelProtocol:
    def test_mock_satisfies_protocol(self):
        m = _MockSurrogate()
        assert isinstance(m, SurrogateModel)

    def test_bad_model_not_protocol(self):
        m = _BadModel()
        assert not isinstance(m, SurrogateModel)

    def test_predict_returns_tuple(self):
        m = _MockSurrogate()
        mean, unc = m.predict([1.0, 2.0, 3.0])
        assert mean == pytest.approx(6.0)
        assert unc == pytest.approx(0.1)


# ── SVGCanvas ───────────────────────────────────────────────────────────────

class TestSVGCanvas:
    def test_empty_canvas(self):
        c = SVGCanvas(400, 300)
        svg = c.to_string()
        assert 'width="400"' in svg
        assert 'height="300"' in svg
        assert "</svg>" in svg

    def test_background(self):
        c = SVGCanvas(100, 100, background="white")
        svg = c.to_string()
        assert 'fill="white"' in svg

    def test_rect(self):
        c = SVGCanvas(200, 200)
        c.rect(10, 20, 50, 30, fill="red")
        svg = c.to_string()
        assert "<rect" in svg
        assert 'fill="red"' in svg

    def test_circle(self):
        c = SVGCanvas(200, 200)
        c.circle(100, 100, 25, fill="blue", stroke="black")
        svg = c.to_string()
        assert "<circle" in svg
        assert 'r="25"' in svg

    def test_line(self):
        c = SVGCanvas(200, 200)
        c.line(0, 0, 100, 100, stroke="green", stroke_width=2)
        svg = c.to_string()
        assert "<line" in svg
        assert 'stroke="green"' in svg

    def test_polyline(self):
        c = SVGCanvas(200, 200)
        c.polyline([(0, 0), (50, 50), (100, 0)], stroke="purple")
        svg = c.to_string()
        assert "<polyline" in svg
        assert "0,0 50,50 100,0" in svg

    def test_polygon(self):
        c = SVGCanvas(200, 200)
        c.polygon([(0, 0), (100, 0), (50, 87)], fill="yellow")
        svg = c.to_string()
        assert "<polygon" in svg
        assert 'fill="yellow"' in svg

    def test_text(self):
        c = SVGCanvas(200, 200)
        c.text(10, 20, "Hello <World>", font_size=14)
        svg = c.to_string()
        assert "<text" in svg
        assert "Hello &lt;World&gt;" in svg

    def test_text_escaping(self):
        c = SVGCanvas(100, 100)
        c.text(0, 0, "a & b < c > d")
        svg = c.to_string()
        assert "a &amp; b &lt; c &gt; d" in svg

    def test_path(self):
        c = SVGCanvas(200, 200)
        c.path("M 10 10 L 100 100", stroke="black")
        svg = c.to_string()
        assert "<path" in svg
        assert "M 10 10 L 100 100" in svg

    def test_group(self):
        c = SVGCanvas(200, 200)
        c.group_start(opacity=0.5)
        c.rect(0, 0, 10, 10, fill="red")
        c.group_end()
        svg = c.to_string()
        assert "<g" in svg
        assert "</g>" in svg

    def test_defs(self):
        c = SVGCanvas(200, 200)
        c.add_def('<linearGradient id="g1"></linearGradient>')
        svg = c.to_string()
        assert "<defs>" in svg
        assert "linearGradient" in svg

    def test_raw(self):
        c = SVGCanvas(100, 100)
        c.raw('<custom-elem attr="val"/>')
        svg = c.to_string()
        assert '<custom-elem attr="val"/>' in svg

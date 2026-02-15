"""Tests for the VSUP colormap module."""

from __future__ import annotations

import pytest

from optimization_copilot.visualization.colormaps import (
    VSUPColorMap,
    color_to_hex,
)


# ── Construction ──────────────────────────────────────────────────────────────


class TestConstruction:
    def test_default_construction(self):
        cm = VSUPColorMap()
        assert cm.value_cmap == "viridis"
        assert cm.uncertainty_range == (0.0, 1.0)

    def test_unknown_colormap_raises(self):
        with pytest.raises(ValueError, match="Unknown colormap"):
            VSUPColorMap(value_cmap="turbo")


# ── Viridis stop interpolation ────────────────────────────────────────────────


class TestViridisStops:
    """Verify exact stop colours at the five anchor points."""

    @pytest.fixture()
    def cm(self) -> VSUPColorMap:
        return VSUPColorMap()

    @pytest.mark.parametrize(
        "value, expected_rgb",
        [
            (0.00, (68, 1, 84)),
            (0.25, (59, 82, 139)),
            (0.50, (33, 145, 140)),
            (0.75, (94, 201, 98)),
            (1.00, (253, 231, 37)),
        ],
    )
    def test_exact_stops(self, cm: VSUPColorMap, value: float, expected_rgb: tuple):
        rgba = cm.map(value, uncertainty=0.0)
        assert rgba[:3] == expected_rgb
        assert rgba[3] == 255

    def test_midpoint_interpolation(self, cm: VSUPColorMap):
        """At value=0.125 (midpoint of first segment), expect average of stops 0 & 1."""
        rgba = cm.map(0.125, uncertainty=0.0)
        # Exact halfway: round((68+59)/2)=64, round((1+82)/2)=42, round((84+139)/2)=112
        assert rgba[:3] == (64, 42, 112)

    def test_three_quarter_interpolation(self, cm: VSUPColorMap):
        """At value=0.375 (midpoint of second segment), check interpolation."""
        rgba = cm.map(0.375, uncertainty=0.0)
        # Halfway between stop-1 (59,82,139) and stop-2 (33,145,140):
        # round((59+33)/2)=46, round((82+145)/2)=114, round((139+140)/2)=140
        assert rgba[:3] == (46, 114, 140)


# ── Uncertainty suppression ───────────────────────────────────────────────────


class TestUncertaintySuppression:
    @pytest.fixture()
    def cm(self) -> VSUPColorMap:
        return VSUPColorMap()

    def test_zero_uncertainty_full_colour(self, cm: VSUPColorMap):
        """uncertainty=0 -> no suppression, colour unchanged from base."""
        rgba = cm.map(0.5, uncertainty=0.0)
        assert rgba == (33, 145, 140, 255)

    def test_full_uncertainty_gray(self, cm: VSUPColorMap):
        """uncertainty=1 -> complete suppression, all channels go to gray=200."""
        rgba = cm.map(0.5, uncertainty=1.0)
        assert rgba == (200, 200, 200, 255)

    def test_half_uncertainty_midway(self, cm: VSUPColorMap):
        """uncertainty=0.5 -> halfway between base and gray for each channel."""
        rgba = cm.map(0.5, uncertainty=0.5)
        # Base is (33, 145, 140).  Gray is 200.
        # R: round(33 + (200-33)*0.5) = round(33 + 83.5) = round(116.5) = 116 (banker's rounding)
        # G: round(145 + (200-145)*0.5) = round(145 + 27.5) = round(172.5) = 172
        # B: round(140 + (200-140)*0.5) = round(140 + 30) = 170
        assert rgba == (116, 172, 170, 255)

    def test_alpha_always_255(self, cm: VSUPColorMap):
        """Alpha channel is always 255 regardless of uncertainty."""
        for unc in [0.0, 0.3, 0.7, 1.0]:
            rgba = cm.map(0.5, unc)
            assert rgba[3] == 255


# ── Input clamping ────────────────────────────────────────────────────────────


class TestClamping:
    @pytest.fixture()
    def cm(self) -> VSUPColorMap:
        return VSUPColorMap()

    def test_value_below_zero_clamped(self, cm: VSUPColorMap):
        assert cm.map(-0.5, 0.0) == cm.map(0.0, 0.0)

    def test_value_above_one_clamped(self, cm: VSUPColorMap):
        assert cm.map(1.5, 0.0) == cm.map(1.0, 0.0)

    def test_uncertainty_below_zero_clamped(self, cm: VSUPColorMap):
        assert cm.map(0.5, -0.3) == cm.map(0.5, 0.0)

    def test_uncertainty_above_one_clamped(self, cm: VSUPColorMap):
        assert cm.map(0.5, 2.0) == cm.map(0.5, 1.0)


# ── batch_map ─────────────────────────────────────────────────────────────────


class TestBatchMap:
    @pytest.fixture()
    def cm(self) -> VSUPColorMap:
        return VSUPColorMap()

    def test_batch_correctness(self, cm: VSUPColorMap):
        vals = [0.0, 0.5, 1.0]
        uncs = [0.0, 0.5, 1.0]
        results = cm.batch_map(vals, uncs)
        assert len(results) == 3
        for i, (v, u) in enumerate(zip(vals, uncs)):
            assert results[i] == cm.map(v, u)

    def test_empty_batch(self, cm: VSUPColorMap):
        assert cm.batch_map([], []) == []


# ── color_to_hex ──────────────────────────────────────────────────────────────


class TestColorToHex:
    def test_opaque(self):
        assert color_to_hex(255, 0, 128) == "#FF0080"

    def test_with_alpha(self):
        assert color_to_hex(255, 0, 128, 200) == "#FF0080C8"

    def test_alpha_255_no_suffix(self):
        result = color_to_hex(0, 0, 0, 255)
        assert result == "#000000"
        assert len(result) == 7  # "#RRGGBB", no alpha suffix

    def test_zero_values(self):
        assert color_to_hex(0, 0, 0, 0) == "#00000000"


# ── Alternate colormaps ───────────────────────────────────────────────────────


class TestAlternateColormaps:
    def test_plasma_construction(self):
        cm = VSUPColorMap(value_cmap="plasma")
        assert cm.value_cmap == "plasma"

    def test_plasma_stop_zero(self):
        cm = VSUPColorMap(value_cmap="plasma")
        rgba = cm.map(0.0, 0.0)
        assert rgba[:3] == (13, 8, 135)
        assert rgba[3] == 255

    def test_inferno_construction(self):
        cm = VSUPColorMap(value_cmap="inferno")
        assert cm.value_cmap == "inferno"

    def test_inferno_stop_one(self):
        cm = VSUPColorMap(value_cmap="inferno")
        rgba = cm.map(1.0, 0.0)
        assert rgba[:3] == (252, 255, 164)
        assert rgba[3] == 255

    def test_inferno_with_uncertainty(self):
        cm = VSUPColorMap(value_cmap="inferno")
        rgba = cm.map(0.0, 1.0)
        # Base is (0,0,4), full suppression -> gray 200
        assert rgba == (200, 200, 200, 255)

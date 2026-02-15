"""Comprehensive tests for domain_encoding and robust_optimizer modules.

Tests cover:
- Encoding ABC interface
- OneHotEncoding, OrdinalEncoding, CustomDescriptorEncoding, SpatialEncoding
- EncodingPipeline encode/decode round-trips
- RobustOptimizer perturbation, robustification, sensitivity, and serialization
- Edge cases: single category, boundary coordinates, zero noise, determinism
"""

from __future__ import annotations

import math

import pytest

from optimization_copilot.infrastructure.domain_encoding import (
    CustomDescriptorEncoding,
    Encoding,
    EncodingPipeline,
    OneHotEncoding,
    OrdinalEncoding,
    SpatialEncoding,
)
from optimization_copilot.infrastructure.robust_optimizer import RobustOptimizer


# ---------------------------------------------------------------------------
# Encoding ABC
# ---------------------------------------------------------------------------

class TestEncodingABC:
    """Verify that Encoding cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Encoding()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# OneHotEncoding
# ---------------------------------------------------------------------------

class TestOneHotEncoding:

    def test_encode_single_category(self):
        enc = OneHotEncoding(["only"])
        assert enc.encode("only") == [1.0]

    def test_encode_multiple_categories(self):
        enc = OneHotEncoding(["red", "green", "blue"])
        assert enc.encode("red") == [1.0, 0.0, 0.0]
        assert enc.encode("green") == [0.0, 1.0, 0.0]
        assert enc.encode("blue") == [0.0, 0.0, 1.0]

    def test_decode_exact(self):
        enc = OneHotEncoding(["a", "b", "c"])
        assert enc.decode([1.0, 0.0, 0.0]) == "a"
        assert enc.decode([0.0, 0.0, 1.0]) == "c"

    def test_decode_approximate_uses_argmax(self):
        enc = OneHotEncoding(["a", "b", "c"])
        assert enc.decode([0.1, 0.9, 0.2]) == "b"

    def test_encode_decode_roundtrip(self):
        cats = ["x", "y", "z"]
        enc = OneHotEncoding(cats)
        for cat in cats:
            features = enc.encode(cat)
            assert enc.decode(features) == cat

    def test_n_features_matches_num_categories(self):
        enc = OneHotEncoding(["a", "b", "c", "d"])
        assert enc.n_features == 4

    def test_unknown_category_raises(self):
        enc = OneHotEncoding(["a", "b"])
        with pytest.raises(ValueError, match="Unknown category"):
            enc.encode("c")

    def test_decode_wrong_length_raises(self):
        enc = OneHotEncoding(["a", "b"])
        with pytest.raises(ValueError, match="Expected 2 features"):
            enc.decode([1.0])

    def test_empty_categories_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            OneHotEncoding([])

    def test_duplicate_categories_raises(self):
        with pytest.raises(ValueError, match="unique"):
            OneHotEncoding(["a", "a"])

    def test_to_dict(self):
        enc = OneHotEncoding(["a", "b"])
        d = enc.to_dict()
        assert d["type"] == "OneHotEncoding"
        assert d["categories"] == ["a", "b"]
        assert d["n_features"] == 2

    def test_from_dict_roundtrip(self):
        enc = OneHotEncoding(["alpha", "beta", "gamma"])
        d = enc.to_dict()
        restored = OneHotEncoding.from_dict(d)
        assert restored.categories == enc.categories
        assert restored.encode("beta") == enc.encode("beta")

    def test_categories_returns_copy(self):
        enc = OneHotEncoding(["a", "b"])
        cats = enc.categories
        cats.append("c")
        assert len(enc.categories) == 2  # original unaffected


# ---------------------------------------------------------------------------
# OrdinalEncoding
# ---------------------------------------------------------------------------

class TestOrdinalEncoding:

    def test_encode_first_category(self):
        enc = OrdinalEncoding(["low", "medium", "high"])
        assert enc.encode("low") == [0.0]

    def test_encode_last_category(self):
        enc = OrdinalEncoding(["low", "medium", "high"])
        assert enc.encode("high") == [1.0]

    def test_encode_middle_category(self):
        enc = OrdinalEncoding(["low", "medium", "high"])
        assert enc.encode("medium") == [0.5]

    def test_decode_exact(self):
        enc = OrdinalEncoding(["low", "medium", "high"])
        assert enc.decode([0.0]) == "low"
        assert enc.decode([0.5]) == "medium"
        assert enc.decode([1.0]) == "high"

    def test_decode_approximate(self):
        enc = OrdinalEncoding(["low", "medium", "high"])
        assert enc.decode([0.6]) == "medium"
        assert enc.decode([0.3]) == "medium"

    def test_encode_decode_roundtrip(self):
        cats = ["low", "medium", "high"]
        enc = OrdinalEncoding(cats)
        for cat in cats:
            features = enc.encode(cat)
            assert enc.decode(features) == cat

    def test_preserves_ordering(self):
        enc = OrdinalEncoding(["cold", "warm", "hot"])
        vals = [enc.encode(c)[0] for c in ["cold", "warm", "hot"]]
        assert vals[0] < vals[1] < vals[2]

    def test_n_features_is_one(self):
        enc = OrdinalEncoding(["a", "b", "c", "d", "e"])
        assert enc.n_features == 1

    def test_single_category(self):
        enc = OrdinalEncoding(["only"])
        assert enc.encode("only") == [0.0]
        assert enc.decode([0.0]) == "only"

    def test_unknown_category_raises(self):
        enc = OrdinalEncoding(["a", "b"])
        with pytest.raises(ValueError, match="Unknown category"):
            enc.encode("c")

    def test_decode_wrong_length_raises(self):
        enc = OrdinalEncoding(["a", "b"])
        with pytest.raises(ValueError):
            enc.decode([0.1, 0.2])

    def test_to_dict_from_dict_roundtrip(self):
        enc = OrdinalEncoding(["low", "medium", "high"])
        d = enc.to_dict()
        assert d["type"] == "OrdinalEncoding"
        assert d["ordered_categories"] == ["low", "medium", "high"]
        restored = OrdinalEncoding.from_dict(d)
        assert restored.encode("medium") == enc.encode("medium")


# ---------------------------------------------------------------------------
# CustomDescriptorEncoding
# ---------------------------------------------------------------------------

class TestCustomDescriptorEncoding:

    @pytest.fixture()
    def solvent_enc(self) -> CustomDescriptorEncoding:
        return CustomDescriptorEncoding({
            "ethanol": [46.07, -0.31, 20.23],
            "methanol": [32.04, -0.74, 20.23],
            "acetone": [58.08, -0.24, 17.07],
        })

    def test_encode_returns_descriptor_vector(self, solvent_enc: CustomDescriptorEncoding):
        assert solvent_enc.encode("ethanol") == [46.07, -0.31, 20.23]

    def test_n_features_matches_descriptor_dims(self, solvent_enc: CustomDescriptorEncoding):
        assert solvent_enc.n_features == 3

    def test_decode_finds_nearest_neighbor(self, solvent_enc: CustomDescriptorEncoding):
        # Exact match
        assert solvent_enc.decode([46.07, -0.31, 20.23]) == "ethanol"
        # Slightly perturbed, still closest to ethanol
        assert solvent_enc.decode([46.0, -0.3, 20.2]) == "ethanol"

    def test_encode_decode_roundtrip(self, solvent_enc: CustomDescriptorEncoding):
        for cat in solvent_enc.categories:
            features = solvent_enc.encode(cat)
            assert solvent_enc.decode(features) == cat

    def test_unknown_category_raises(self, solvent_enc: CustomDescriptorEncoding):
        with pytest.raises(ValueError, match="Unknown category"):
            solvent_enc.encode("water")

    def test_decode_wrong_length_raises(self, solvent_enc: CustomDescriptorEncoding):
        with pytest.raises(ValueError, match="Expected 3 features"):
            solvent_enc.decode([1.0, 2.0])

    def test_empty_descriptor_table_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            CustomDescriptorEncoding({})

    def test_inconsistent_descriptor_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            CustomDescriptorEncoding({
                "a": [1.0, 2.0],
                "b": [3.0],
            })

    def test_single_category_single_dim(self):
        enc = CustomDescriptorEncoding({"only": [42.0]})
        assert enc.n_features == 1
        assert enc.encode("only") == [42.0]
        assert enc.decode([99.0]) == "only"  # only one option

    def test_to_dict_from_dict_roundtrip(self, solvent_enc: CustomDescriptorEncoding):
        d = solvent_enc.to_dict()
        assert d["type"] == "CustomDescriptorEncoding"
        assert "descriptor_table" in d
        restored = CustomDescriptorEncoding.from_dict(d)
        assert restored.encode("methanol") == solvent_enc.encode("methanol")
        assert restored.n_features == solvent_enc.n_features


# ---------------------------------------------------------------------------
# SpatialEncoding
# ---------------------------------------------------------------------------

class TestSpatialEncoding:

    def test_equator_prime_meridian(self):
        enc = SpatialEncoding()
        features = enc.encode((0.0, 0.0))
        assert len(features) == 3
        assert features[0] == pytest.approx(1.0, abs=1e-10)
        assert features[1] == pytest.approx(0.0, abs=1e-10)
        assert features[2] == pytest.approx(0.0, abs=1e-10)

    def test_north_pole(self):
        enc = SpatialEncoding()
        features = enc.encode((90.0, 0.0))
        assert features[0] == pytest.approx(0.0, abs=1e-10)
        assert features[1] == pytest.approx(0.0, abs=1e-10)
        assert features[2] == pytest.approx(1.0, abs=1e-10)

    def test_south_pole(self):
        enc = SpatialEncoding()
        features = enc.encode((-90.0, 0.0))
        assert features[2] == pytest.approx(-1.0, abs=1e-10)

    def test_date_line_positive(self):
        enc = SpatialEncoding()
        features = enc.encode((0.0, 180.0))
        assert features[0] == pytest.approx(-1.0, abs=1e-10)
        assert abs(features[1]) < 1e-10

    def test_date_line_negative(self):
        enc = SpatialEncoding()
        features = enc.encode((0.0, -180.0))
        assert features[0] == pytest.approx(-1.0, abs=1e-10)
        assert abs(features[1]) < 1e-10

    def test_unit_sphere_norm(self):
        enc = SpatialEncoding()
        for lat, lon in [(37.7749, -122.4194), (51.5074, -0.1278), (-33.8688, 151.2093)]:
            features = enc.encode((lat, lon))
            norm = math.sqrt(sum(f * f for f in features))
            assert norm == pytest.approx(1.0, abs=1e-10)

    def test_encode_decode_roundtrip(self):
        enc = SpatialEncoding()
        coords = [(0.0, 0.0), (45.0, 90.0), (-30.0, -60.0), (89.9, 179.9)]
        for lat, lon in coords:
            features = enc.encode((lat, lon))
            decoded_lat, decoded_lon = enc.decode(features)
            assert decoded_lat == pytest.approx(lat, abs=1e-6)
            assert decoded_lon == pytest.approx(lon, abs=1e-6)

    def test_n_features_is_three(self):
        enc = SpatialEncoding()
        assert enc.n_features == 3

    def test_decode_zero_vector(self):
        enc = SpatialEncoding()
        lat, lon = enc.decode([0.0, 0.0, 0.0])
        assert lat == 0.0
        assert lon == 0.0

    def test_invalid_coord_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            SpatialEncoding(coord_type="utm")

    def test_encode_invalid_input_raises(self):
        enc = SpatialEncoding()
        with pytest.raises(ValueError):
            enc.encode(42)  # not a 2-element sequence

    def test_decode_wrong_length_raises(self):
        enc = SpatialEncoding()
        with pytest.raises(ValueError, match="Expected 3 features"):
            enc.decode([1.0, 2.0])

    def test_to_dict_from_dict_roundtrip(self):
        enc = SpatialEncoding()
        d = enc.to_dict()
        assert d["type"] == "SpatialEncoding"
        assert d["coord_type"] == "latlon"
        restored = SpatialEncoding.from_dict(d)
        original = enc.encode((45.0, 90.0))
        restored_feat = restored.encode((45.0, 90.0))
        assert original == restored_feat


# ---------------------------------------------------------------------------
# EncodingPipeline
# ---------------------------------------------------------------------------

class TestEncodingPipeline:

    def test_encode_point_with_multiple_encodings(self):
        pipeline = EncodingPipeline()
        pipeline.add_encoding("color", OneHotEncoding(["red", "green", "blue"]))
        pipeline.add_encoding("size", OrdinalEncoding(["small", "medium", "large"]))

        features = pipeline.encode_params({"color": "green", "size": "large"})
        # 3 for one-hot + 1 for ordinal = 4
        assert len(features) == 4
        assert features[:3] == [0.0, 1.0, 0.0]  # green
        assert features[3] == pytest.approx(1.0)  # large

    def test_encode_with_passthrough_numeric(self):
        pipeline = EncodingPipeline()
        pipeline.add_encoding("color", OneHotEncoding(["red", "blue"]))

        features = pipeline.encode_params({"color": "red", "pressure": 1.5})
        assert len(features) == 3  # 2 (one-hot) + 1 (pass-through)
        assert features[:2] == [1.0, 0.0]
        assert features[2] == pytest.approx(1.5)

    def test_passthrough_non_numeric_raises(self):
        pipeline = EncodingPipeline()
        with pytest.raises(TypeError, match="numeric"):
            pipeline.encode_params({"name": "test_string"})

    def test_decode_roundtrip(self):
        pipeline = EncodingPipeline()
        pipeline.add_encoding("color", OneHotEncoding(["red", "green"]))
        pipeline.add_encoding("grade", OrdinalEncoding(["low", "high"]))

        params = {"color": "red", "grade": "high", "pressure": 3.14}
        features = pipeline.encode_params(params)
        decoded = pipeline.decode_features(
            features,
            param_names=["color", "grade", "pressure"],
        )
        assert decoded["color"] == "red"
        assert decoded["grade"] == "high"
        assert decoded["pressure"] == pytest.approx(3.14)

    def test_total_features_sum(self):
        pipeline = EncodingPipeline()
        pipeline.add_encoding("a", OneHotEncoding(["x", "y", "z"]))  # 3
        pipeline.add_encoding("b", OrdinalEncoding(["lo", "hi"]))    # 1
        pipeline.add_encoding("c", SpatialEncoding())                 # 3
        assert pipeline.total_features() == 7

    def test_empty_pipeline(self):
        pipeline = EncodingPipeline()
        assert pipeline.total_features() == 0
        features = pipeline.encode_params({"x": 1.0})
        assert features == [1.0]

    def test_remove_encoding(self):
        pipeline = EncodingPipeline()
        pipeline.add_encoding("a", OneHotEncoding(["x", "y"]))
        assert pipeline.remove_encoding("a") is True
        assert pipeline.remove_encoding("a") is False
        assert pipeline.total_features() == 0

    def test_param_names(self):
        pipeline = EncodingPipeline()
        pipeline.add_encoding("color", OneHotEncoding(["r"]))
        pipeline.add_encoding("size", OrdinalEncoding(["s"]))
        assert pipeline.param_names == ["color", "size"]

    def test_to_dict(self):
        pipeline = EncodingPipeline()
        pipeline.add_encoding("a", OneHotEncoding(["x", "y"]))
        d = pipeline.to_dict()
        assert "encodings" in d
        assert "a" in d["encodings"]
        assert d["total_features"] == 2

    def test_decode_features_too_short_raises(self):
        pipeline = EncodingPipeline()
        pipeline.add_encoding("color", OneHotEncoding(["r", "g", "b"]))
        with pytest.raises(ValueError, match="too short"):
            pipeline.decode_features([1.0], param_names=["color"])

    def test_init_with_encodings_dict(self):
        encs = {"c": OneHotEncoding(["a", "b"])}
        pipeline = EncodingPipeline(encodings=encs)
        assert pipeline.total_features() == 2


# ---------------------------------------------------------------------------
# RobustOptimizer
# ---------------------------------------------------------------------------

class TestRobustOptimizer:

    def test_initialization_defaults(self):
        opt = RobustOptimizer()
        assert opt.n_perturbations == 20
        assert opt.noise_config == {}

    def test_initialization_custom(self):
        opt = RobustOptimizer(
            input_noise={"x": 0.1, "y": 0.2},
            n_perturbations=50,
            seed=42,
        )
        assert opt.n_perturbations == 50
        assert opt.noise_config == {"x": 0.1, "y": 0.2}

    def test_n_perturbations_clamped_to_at_least_one(self):
        opt = RobustOptimizer(n_perturbations=0)
        assert opt.n_perturbations >= 1

    def test_robustify_acquisition_no_noise_passthrough(self):
        """With no noise config, robustify should return raw acquisition values."""
        opt = RobustOptimizer(input_noise={})
        candidates = [{"x": 1.0}, {"x": 2.0}]

        def acq(c):
            return c["x"] ** 2

        specs = [{"name": "x", "min": 0.0, "max": 10.0}]
        result = opt.robustify_acquisition(candidates, acq, specs)
        assert result == [1.0, 4.0]

    def test_robustify_acquisition_returns_correct_length(self):
        opt = RobustOptimizer(input_noise={"x": 0.1}, n_perturbations=10, seed=1)
        candidates = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]

        def acq(c):
            return c["x"]

        specs = [{"name": "x", "min": 0.0, "max": 10.0}]
        result = opt.robustify_acquisition(candidates, acq, specs)
        assert len(result) == 3

    def test_robustify_acquisition_single_candidate(self):
        opt = RobustOptimizer(input_noise={"x": 0.01}, n_perturbations=5, seed=42)
        candidates = [{"x": 5.0}]

        def acq(c):
            return c["x"]

        specs = [{"name": "x", "min": 0.0, "max": 10.0}]
        result = opt.robustify_acquisition(candidates, acq, specs)
        assert len(result) == 1
        # With small noise, should be close to original value
        assert result[0] == pytest.approx(5.0, abs=0.5)

    def test_robustify_acquisition_averages_near_original(self):
        """With small noise, robust value should be close to noiseless value."""
        opt = RobustOptimizer(input_noise={"x": 0.001}, n_perturbations=100, seed=99)
        candidates = [{"x": 5.0}]

        def acq(c):
            return c["x"]

        specs = [{"name": "x", "min": 0.0, "max": 10.0}]
        result = opt.robustify_acquisition(candidates, acq, specs)
        assert result[0] == pytest.approx(5.0, abs=0.05)

    def test_perturb_stays_within_bounds(self):
        opt = RobustOptimizer(input_noise={"x": 100.0}, n_perturbations=50, seed=7)
        specs = [{"name": "x", "min": 0.0, "max": 1.0}]
        candidate = {"x": 0.5}
        for _ in range(50):
            perturbed = opt._perturb(candidate, specs)
            assert 0.0 <= perturbed["x"] <= 1.0

    def test_perturb_leaves_non_noisy_params_unchanged(self):
        opt = RobustOptimizer(input_noise={"x": 0.1}, n_perturbations=10, seed=1)
        specs = [
            {"name": "x", "min": 0.0, "max": 10.0},
            {"name": "y", "min": 0.0, "max": 10.0},
        ]
        candidate = {"x": 5.0, "y": 3.0}
        perturbed = opt._perturb(candidate, specs)
        assert perturbed["y"] == 3.0  # y has no noise, should be unchanged

    def test_perturb_leaves_categorical_unchanged(self):
        opt = RobustOptimizer(input_noise={"cat": 0.5}, n_perturbations=5, seed=1)
        specs = [{"name": "cat"}]
        candidate = {"cat": "red"}
        perturbed = opt._perturb(candidate, specs)
        assert perturbed["cat"] == "red"  # non-numeric, should not be perturbed

    def test_zero_noise_no_perturbation(self):
        opt = RobustOptimizer(input_noise={"x": 0.0}, n_perturbations=10, seed=1)
        specs = [{"name": "x", "min": 0.0, "max": 10.0}]
        candidate = {"x": 5.0}
        for _ in range(10):
            perturbed = opt._perturb(candidate, specs)
            assert perturbed["x"] == 5.0  # noise_std <= 0 means no perturbation

    def test_determinism_same_seed(self):
        """Same seed should produce identical results."""
        noise = {"x": 0.5}
        specs = [{"name": "x", "min": 0.0, "max": 10.0}]
        candidates = [{"x": 5.0}]

        def acq(c):
            return c["x"] ** 2

        opt1 = RobustOptimizer(input_noise=noise, n_perturbations=20, seed=42)
        result1 = opt1.robustify_acquisition(candidates, acq, specs)

        opt2 = RobustOptimizer(input_noise=noise, n_perturbations=20, seed=42)
        result2 = opt2.robustify_acquisition(candidates, acq, specs)

        assert result1 == result2

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different results."""
        noise = {"x": 1.0}
        specs = [{"name": "x", "min": 0.0, "max": 10.0}]
        candidates = [{"x": 5.0}]

        def acq(c):
            return c["x"] ** 2

        opt1 = RobustOptimizer(input_noise=noise, n_perturbations=20, seed=1)
        result1 = opt1.robustify_acquisition(candidates, acq, specs)

        opt2 = RobustOptimizer(input_noise=noise, n_perturbations=20, seed=2)
        result2 = opt2.robustify_acquisition(candidates, acq, specs)

        assert result1 != result2

    def test_robustify_candidates_no_noise_passthrough(self):
        opt = RobustOptimizer(input_noise={})
        candidates = [{"x": 1.0}]
        acq_values = [10.0]
        specs = [{"name": "x", "min": 0.0, "max": 10.0}]
        result = opt.robustify_candidates(candidates, acq_values, specs)
        assert result == [10.0]

    def test_robustify_candidates_penalizes(self):
        """Candidate near boundary should receive a penalty."""
        opt = RobustOptimizer(input_noise={"x": 2.0}, n_perturbations=10, seed=1)
        specs = [{"name": "x", "min": 0.0, "max": 10.0}]
        # Candidate at boundary
        candidates = [{"x": 0.0}]
        acq_values = [10.0]
        result = opt.robustify_candidates(candidates, acq_values, specs)
        # Should be penalized (less than 10.0)
        assert result[0] < 10.0

    def test_robustify_candidates_center_less_penalized(self):
        """Candidate at center should be penalized less than at boundary."""
        opt = RobustOptimizer(input_noise={"x": 1.0}, n_perturbations=10, seed=1)
        specs = [{"name": "x", "min": 0.0, "max": 10.0}]
        center_cand = [{"x": 5.0}]
        edge_cand = [{"x": 0.0}]
        acq_values = [10.0]

        center_result = opt.robustify_candidates(center_cand, acq_values, specs)
        edge_result = opt.robustify_candidates(edge_cand, acq_values, specs)
        assert center_result[0] >= edge_result[0]

    def test_sensitivity_analysis_returns_per_param(self):
        opt = RobustOptimizer(
            input_noise={"x": 0.5, "y": 0.1},
            n_perturbations=20,
            seed=42,
        )
        candidate = {"x": 5.0, "y": 3.0}

        def acq(c):
            return c["x"] ** 2 + c["y"]

        specs = [
            {"name": "x", "min": 0.0, "max": 10.0},
            {"name": "y", "min": 0.0, "max": 10.0},
        ]
        sensitivities = opt.sensitivity_analysis(candidate, acq, specs)
        assert "x" in sensitivities
        assert "y" in sensitivities
        # x has larger noise and quadratic effect, should be more sensitive
        assert sensitivities["x"] > sensitivities["y"]

    def test_sensitivity_analysis_zero_noise_param(self):
        opt = RobustOptimizer(
            input_noise={"x": 0.0},
            n_perturbations=10,
            seed=1,
        )
        candidate = {"x": 5.0}

        def acq(c):
            return c["x"]

        specs = [{"name": "x", "min": 0.0, "max": 10.0}]
        sensitivities = opt.sensitivity_analysis(candidate, acq, specs)
        assert sensitivities["x"] == 0.0

    def test_to_dict(self):
        opt = RobustOptimizer(input_noise={"x": 0.1}, n_perturbations=30)
        d = opt.to_dict()
        assert d["input_noise"] == {"x": 0.1}
        assert d["n_perturbations"] == 30

    def test_from_dict_roundtrip(self):
        opt = RobustOptimizer(input_noise={"x": 0.5, "y": 0.2}, n_perturbations=15)
        d = opt.to_dict()
        restored = RobustOptimizer.from_dict(d)
        assert restored.noise_config == opt.noise_config
        assert restored.n_perturbations == opt.n_perturbations

    def test_many_candidates(self):
        """Ensure correct behavior with many candidates."""
        opt = RobustOptimizer(input_noise={"x": 0.1}, n_perturbations=5, seed=1)
        candidates = [{"x": float(i)} for i in range(100)]

        def acq(c):
            return c["x"]

        specs = [{"name": "x", "min": 0.0, "max": 100.0}]
        result = opt.robustify_acquisition(candidates, acq, specs)
        assert len(result) == 100

    def test_single_dimension(self):
        opt = RobustOptimizer(input_noise={"x": 0.01}, n_perturbations=10, seed=5)
        candidates = [{"x": 0.5}]

        def acq(c):
            return c["x"]

        specs = [{"name": "x", "min": 0.0, "max": 1.0}]
        result = opt.robustify_acquisition(candidates, acq, specs)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.5, abs=0.1)

    def test_noise_config_returns_copy(self):
        opt = RobustOptimizer(input_noise={"x": 0.1})
        cfg = opt.noise_config
        cfg["x"] = 999.0
        assert opt.noise_config["x"] == 0.1  # original unaffected

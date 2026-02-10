"""Feature Extractor Registry for curve-as-objective optimization.

When the KPI is not a scalar but a curve (EIS, UV-Vis, rheology, etc.),
this module extracts named features from the curve to enable standard
scalar-based optimization over curve-derived quantities.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.hashing import _stable_serialize, _sha256


# ── Data Classes ──────────────────────────────────────────


@dataclass
class CurveData:
    """A measured curve (x, y) with optional metadata.

    Parameters
    ----------
    x_values : list[float]
        Independent variable (e.g. frequency, wavelength, time).
    y_values : list[float]
        Dependent variable (e.g. impedance, absorbance, viscosity).
    metadata : dict
        Arbitrary metadata (e.g. {"type": "eis", "unit": "ohm"}).
    """

    x_values: list[float]
    y_values: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.x_values) != len(self.y_values):
            raise ValueError(
                f"x_values length ({len(self.x_values)}) != "
                f"y_values length ({len(self.y_values)})"
            )


@dataclass
class ExtractedFeatures:
    """Result of applying a :class:`FeatureExtractor` to a :class:`CurveData`.

    Parameters
    ----------
    features : dict[str, float]
        Named scalar features extracted from the curve.
    extractor_name : str
        Name of the extractor that produced these features.
    extractor_version : str
        Version of the extractor for reproducibility tracking.
    feature_hash : str
        SHA-256 (truncated) of the input data for reproducibility audit.
    """

    features: dict[str, float]
    extractor_name: str
    extractor_version: str
    feature_hash: str


# ── Abstract Base ─────────────────────────────────────────


class FeatureExtractor(ABC):
    """Abstract base class for curve feature extractors."""

    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this extractor."""

    @abstractmethod
    def version(self) -> str:
        """Semantic version string for reproducibility."""

    @abstractmethod
    def extract(self, curve: CurveData) -> ExtractedFeatures:
        """Extract scalar features from *curve*.

        Implementations must populate :attr:`ExtractedFeatures.feature_hash`
        using :func:`_compute_curve_hash`.
        """

    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return the list of feature names this extractor produces."""


# ── Hashing helper ────────────────────────────────────────


def _compute_curve_hash(curve: CurveData) -> str:
    """Deterministic hash of a CurveData for reproducibility audit."""
    payload = {
        "x_values": curve.x_values,
        "y_values": curve.y_values,
        "metadata": curve.metadata,
    }
    return _sha256(_stable_serialize(payload))


# ── Built-in Extractors ──────────────────────────────────


class BasicCurveExtractor(FeatureExtractor):
    """Extracts fundamental statistical features from a curve.

    Features
    --------
    peak_value : float
        Maximum y value.
    peak_position : float
        x value at which the maximum y occurs.
    area_under_curve : float
        Trapezoidal integration of the curve.
    mean_slope : float
        Average slope (delta-y / delta-x) across consecutive points.
    start_value : float
        First y value.
    end_value : float
        Last y value.
    range : float
        Difference between max and min y values.
    std_dev : float
        Population standard deviation of y values.
    """

    def name(self) -> str:
        return "basic_curve"

    def version(self) -> str:
        return "1.0.0"

    def feature_names(self) -> list[str]:
        return [
            "peak_value",
            "peak_position",
            "area_under_curve",
            "mean_slope",
            "start_value",
            "end_value",
            "range",
            "std_dev",
        ]

    def extract(self, curve: CurveData) -> ExtractedFeatures:
        xs, ys = curve.x_values, curve.y_values
        n = len(ys)

        if n == 0:
            features = {name: 0.0 for name in self.feature_names()}
            return ExtractedFeatures(
                features=features,
                extractor_name=self.name(),
                extractor_version=self.version(),
                feature_hash=_compute_curve_hash(curve),
            )

        peak_idx = 0
        for i in range(1, n):
            if ys[i] > ys[peak_idx]:
                peak_idx = i

        peak_value = ys[peak_idx]
        peak_position = xs[peak_idx]

        # Trapezoidal area under curve
        area = 0.0
        for i in range(n - 1):
            area += 0.5 * abs(xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1])

        # Mean slope
        if n >= 2:
            slopes = []
            for i in range(n - 1):
                dx = xs[i + 1] - xs[i]
                if dx != 0.0:
                    slopes.append((ys[i + 1] - ys[i]) / dx)
            mean_slope = sum(slopes) / len(slopes) if slopes else 0.0
        else:
            mean_slope = 0.0

        start_value = ys[0]
        end_value = ys[-1]

        y_min = min(ys)
        y_max = max(ys)
        y_range = y_max - y_min

        # Population standard deviation
        mean_y = sum(ys) / n
        variance = sum((y - mean_y) ** 2 for y in ys) / n
        std_dev = math.sqrt(variance)

        features = {
            "peak_value": peak_value,
            "peak_position": peak_position,
            "area_under_curve": area,
            "mean_slope": mean_slope,
            "start_value": start_value,
            "end_value": end_value,
            "range": y_range,
            "std_dev": std_dev,
        }

        return ExtractedFeatures(
            features=features,
            extractor_name=self.name(),
            extractor_version=self.version(),
            feature_hash=_compute_curve_hash(curve),
        )


class ThresholdExtractor(FeatureExtractor):
    """Extracts threshold-crossing features from a curve.

    Parameters
    ----------
    threshold : float
        The y-value threshold to evaluate crossings against.

    Features
    --------
    time_to_threshold : float
        First x-value where y crosses (or equals) the threshold.
        Returns ``float('inf')`` if the threshold is never reached.
    fraction_above_threshold : float
        Fraction of data points where y >= threshold.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self._threshold = threshold

    def name(self) -> str:
        return "threshold"

    def version(self) -> str:
        return "1.0.0"

    def feature_names(self) -> list[str]:
        return ["time_to_threshold", "fraction_above_threshold"]

    def extract(self, curve: CurveData) -> ExtractedFeatures:
        xs, ys = curve.x_values, curve.y_values
        n = len(ys)
        threshold = self._threshold

        if n == 0:
            features = {
                "time_to_threshold": float("inf"),
                "fraction_above_threshold": 0.0,
            }
            return ExtractedFeatures(
                features=features,
                extractor_name=self.name(),
                extractor_version=self.version(),
                feature_hash=_compute_curve_hash(curve),
            )

        # First x where y >= threshold
        time_to_threshold = float("inf")
        for i in range(n):
            if ys[i] >= threshold:
                time_to_threshold = xs[i]
                break

        # Fraction of points above (or equal to) threshold
        count_above = sum(1 for y in ys if y >= threshold)
        fraction_above = count_above / n

        features = {
            "time_to_threshold": time_to_threshold,
            "fraction_above_threshold": fraction_above,
        }

        return ExtractedFeatures(
            features=features,
            extractor_name=self.name(),
            extractor_version=self.version(),
            feature_hash=_compute_curve_hash(curve),
        )


# ── Registry ──────────────────────────────────────────────


class FeatureExtractorRegistry:
    """Central registry for :class:`FeatureExtractor` implementations.

    Usage::

        registry = FeatureExtractorRegistry()
        registry.register(BasicCurveExtractor)
        extractor = registry.get("basic_curve")
        result = extractor.extract(my_curve)
    """

    def __init__(self) -> None:
        self._extractors: dict[str, FeatureExtractor] = {}

    def register(self, extractor_class: type[FeatureExtractor], **kwargs: Any) -> None:
        """Register an extractor class (instantiated with *kwargs*).

        Raises
        ------
        TypeError
            If *extractor_class* is not a subclass of :class:`FeatureExtractor`.
        ValueError
            If an extractor with the same name is already registered.
        """
        if not (isinstance(extractor_class, type) and issubclass(extractor_class, FeatureExtractor)):
            raise TypeError(
                f"Expected a subclass of FeatureExtractor, got {extractor_class!r}"
            )
        instance = extractor_class(**kwargs)
        ext_name = instance.name()
        if ext_name in self._extractors:
            raise ValueError(f"Extractor '{ext_name}' is already registered")
        self._extractors[ext_name] = instance

    def get(self, name: str) -> FeatureExtractor:
        """Return the registered extractor instance for *name*.

        Raises
        ------
        KeyError
            If no extractor with that name exists.
        """
        if name not in self._extractors:
            raise KeyError(
                f"Unknown extractor '{name}'. "
                f"Available: {self.list_extractors()}"
            )
        return self._extractors[name]

    def list_extractors(self) -> list[str]:
        """Return sorted list of all registered extractor names."""
        return sorted(self._extractors)

    def extract_all(self, curve: CurveData) -> dict[str, ExtractedFeatures]:
        """Run every registered extractor on *curve*.

        Returns a dict mapping extractor name to its :class:`ExtractedFeatures`.
        """
        results: dict[str, ExtractedFeatures] = {}
        for name in sorted(self._extractors):
            results[name] = self._extractors[name].extract(curve)
        return results


# ── Helper: Curve Stability Signal ────────────────────────


def curve_stability_signal(
    curves: list[CurveData],
    extractor: FeatureExtractor,
) -> dict[str, float]:
    """Compute coefficient-of-variation (CV) for each feature across *curves*.

    For each feature produced by *extractor*, this function extracts the
    feature from every curve, then computes the CV (std / |mean|) as a
    "feature drift" signal.  Higher CV indicates less stability.

    Parameters
    ----------
    curves : list[CurveData]
        A collection of curves (e.g. replicate measurements).
    extractor : FeatureExtractor
        The extractor to apply to each curve.

    Returns
    -------
    dict[str, float]
        Mapping of ``feature_name -> CV``.  Returns 0.0 for a feature
        when the mean is zero (undefined CV).
    """
    if not curves:
        return {name: 0.0 for name in extractor.feature_names()}

    # Collect feature values across all curves
    all_features: dict[str, list[float]] = {
        name: [] for name in extractor.feature_names()
    }
    for curve in curves:
        result = extractor.extract(curve)
        for fname, fval in result.features.items():
            if fname in all_features:
                all_features[fname].append(fval)

    # Compute CV per feature
    cv_signals: dict[str, float] = {}
    for fname, values in all_features.items():
        n = len(values)
        if n == 0:
            cv_signals[fname] = 0.0
            continue
        mean = sum(values) / n
        if n < 2:
            cv_signals[fname] = 0.0
            continue
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance)
        if abs(mean) < 1e-15:
            cv_signals[fname] = 0.0
        else:
            cv_signals[fname] = std / abs(mean)

    return cv_signals

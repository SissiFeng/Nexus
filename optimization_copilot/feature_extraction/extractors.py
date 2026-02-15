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


# ── Domain-specific extractors ───────────────────────────


class EISNyquistExtractor(FeatureExtractor):
    """Extract features from Electrochemical Impedance Spectroscopy Nyquist plots.

    Expects ``x_values`` = real impedance (Z'), ``y_values`` = -imaginary (Z'').

    Features
    --------
    r_solution : float
        Solution (or series) resistance — smallest real impedance.
    r_polarization : float
        Polarization resistance — largest real impedance.
    semicircle_diameter : float
        r_polarization - r_solution (charge transfer proxy).
    max_imaginary : float
        Peak of -Z'' (semicircle apex).
    peak_frequency_position : float
        X-position (Z') at the -Z'' maximum.
    low_freq_tail_slope : float
        Slope of the last 20% of the curve (Warburg / diffusion indicator).
    """

    def name(self) -> str:
        return "eis_nyquist"

    def version(self) -> str:
        return "1.0.0"

    def feature_names(self) -> list[str]:
        return [
            "r_solution",
            "r_polarization",
            "semicircle_diameter",
            "max_imaginary",
            "peak_frequency_position",
            "low_freq_tail_slope",
        ]

    def extract(self, curve: CurveData) -> ExtractedFeatures:
        xs, ys = curve.x_values, curve.y_values
        n = len(xs)
        h = _compute_curve_hash(curve)

        if n == 0:
            return ExtractedFeatures(
                features={k: 0.0 for k in self.feature_names()},
                extractor_name=self.name(),
                extractor_version=self.version(),
                feature_hash=h,
            )

        r_solution = min(xs)
        r_polarization = max(xs)
        semicircle_diameter = r_polarization - r_solution

        max_imag = max(ys)
        peak_idx = ys.index(max_imag)
        peak_freq_pos = xs[peak_idx]

        # Low-frequency tail slope (last 20% of points)
        tail_start = max(1, n - max(1, n // 5))
        if tail_start < n - 1:
            dx = xs[-1] - xs[tail_start]
            dy = ys[-1] - ys[tail_start]
            tail_slope = dy / dx if dx != 0 else 0.0
        else:
            tail_slope = 0.0

        return ExtractedFeatures(
            features={
                "r_solution": r_solution,
                "r_polarization": r_polarization,
                "semicircle_diameter": semicircle_diameter,
                "max_imaginary": max_imag,
                "peak_frequency_position": peak_freq_pos,
                "low_freq_tail_slope": tail_slope,
            },
            extractor_name=self.name(),
            extractor_version=self.version(),
            feature_hash=h,
        )


class UVVisExtractor(FeatureExtractor):
    """Extract features from UV-Vis absorbance spectra.

    Expects ``x_values`` = wavelength (nm), ``y_values`` = absorbance.

    Features
    --------
    peak_wavelength : float
        Wavelength of maximum absorbance.
    peak_absorbance : float
        Maximum absorbance value.
    fwhm : float
        Full width at half maximum of the dominant peak.
    total_absorbance : float
        Trapezoidal integral (total spectral area).
    baseline_level : float
        Mean of the lowest 10% of absorbance values.
    peak_prominence : float
        peak_absorbance - baseline_level.
    """

    def name(self) -> str:
        return "uv_vis"

    def version(self) -> str:
        return "1.0.0"

    def feature_names(self) -> list[str]:
        return [
            "peak_wavelength",
            "peak_absorbance",
            "fwhm",
            "total_absorbance",
            "baseline_level",
            "peak_prominence",
        ]

    def extract(self, curve: CurveData) -> ExtractedFeatures:
        xs, ys = curve.x_values, curve.y_values
        n = len(xs)
        h = _compute_curve_hash(curve)

        if n == 0:
            return ExtractedFeatures(
                features={k: 0.0 for k in self.feature_names()},
                extractor_name=self.name(),
                extractor_version=self.version(),
                feature_hash=h,
            )

        peak_idx = 0
        for i in range(1, n):
            if ys[i] > ys[peak_idx]:
                peak_idx = i

        peak_wavelength = xs[peak_idx]
        peak_absorbance = ys[peak_idx]

        # Baseline: mean of lowest 10% of values
        sorted_y = sorted(ys)
        n_baseline = max(1, n // 10)
        baseline_level = sum(sorted_y[:n_baseline]) / n_baseline

        peak_prominence = peak_absorbance - baseline_level

        # FWHM: find half-max boundaries around peak
        half_max = baseline_level + peak_prominence / 2.0
        left_x = xs[0]
        for i in range(peak_idx, -1, -1):
            if ys[i] <= half_max:
                left_x = xs[i]
                break
        right_x = xs[-1]
        for i in range(peak_idx, n):
            if ys[i] <= half_max:
                right_x = xs[i]
                break
        fwhm = abs(right_x - left_x)

        # Trapezoidal integral
        total = 0.0
        for i in range(n - 1):
            total += 0.5 * abs(xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1])

        return ExtractedFeatures(
            features={
                "peak_wavelength": peak_wavelength,
                "peak_absorbance": peak_absorbance,
                "fwhm": fwhm,
                "total_absorbance": total,
                "baseline_level": baseline_level,
                "peak_prominence": peak_prominence,
            },
            extractor_name=self.name(),
            extractor_version=self.version(),
            feature_hash=h,
        )


class XRDPatternExtractor(FeatureExtractor):
    """Extract features from X-ray Diffraction patterns.

    Expects ``x_values`` = 2-theta (degrees), ``y_values`` = intensity (counts).

    Features
    --------
    primary_peak_angle : float
        2-theta of the strongest peak.
    primary_peak_intensity : float
        Intensity of the strongest peak.
    n_peaks : float
        Number of detected peaks (local maxima above baseline).
    crystallinity_index : float
        Ratio of peak area to total area (higher → more crystalline).
    mean_peak_width : float
        Average FWHM of detected peaks (Scherrer-size proxy).
    background_level : float
        Estimated background (mean of lowest 20% of intensities).
    """

    def __init__(self, min_prominence_fraction: float = 0.1) -> None:
        self._min_prominence_fraction = min_prominence_fraction

    def name(self) -> str:
        return "xrd_pattern"

    def version(self) -> str:
        return "1.0.0"

    def feature_names(self) -> list[str]:
        return [
            "primary_peak_angle",
            "primary_peak_intensity",
            "n_peaks",
            "crystallinity_index",
            "mean_peak_width",
            "background_level",
        ]

    def extract(self, curve: CurveData) -> ExtractedFeatures:
        xs, ys = curve.x_values, curve.y_values
        n = len(xs)
        h = _compute_curve_hash(curve)

        if n == 0:
            return ExtractedFeatures(
                features={k: 0.0 for k in self.feature_names()},
                extractor_name=self.name(),
                extractor_version=self.version(),
                feature_hash=h,
            )

        # Background estimate
        sorted_y = sorted(ys)
        n_bg = max(1, n // 5)
        background = sum(sorted_y[:n_bg]) / n_bg

        # Peak detection: local maxima above background + prominence threshold
        y_range = max(ys) - background
        min_prom = y_range * self._min_prominence_fraction
        peaks: list[int] = []
        for i in range(1, n - 1):
            if ys[i] > ys[i - 1] and ys[i] > ys[i + 1]:
                if ys[i] - background > min_prom:
                    peaks.append(i)
        # Handle boundary: check if first/last is highest
        if n >= 2:
            if ys[0] > ys[1] and ys[0] - background > min_prom:
                peaks.insert(0, 0)
            if ys[-1] > ys[-2] and ys[-1] - background > min_prom:
                peaks.append(n - 1)

        if not peaks:
            # Fallback: use global max
            max_idx = ys.index(max(ys))
            peaks = [max_idx]

        # Primary peak
        primary_idx = max(peaks, key=lambda i: ys[i])
        primary_angle = xs[primary_idx]
        primary_intensity = ys[primary_idx]

        # Mean peak width (estimate FWHM per peak)
        widths: list[float] = []
        for pi in peaks:
            half_h = background + (ys[pi] - background) / 2.0
            left_x = xs[max(0, pi - 1)]
            for j in range(pi, -1, -1):
                if ys[j] <= half_h:
                    left_x = xs[j]
                    break
            right_x = xs[min(n - 1, pi + 1)]
            for j in range(pi, n):
                if ys[j] <= half_h:
                    right_x = xs[j]
                    break
            widths.append(abs(right_x - left_x))
        mean_width = sum(widths) / len(widths) if widths else 0.0

        # Crystallinity index: sum of peak regions / total area
        total_area = 0.0
        for i in range(n - 1):
            total_area += 0.5 * abs(xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1])

        # Estimate peak area as area above background
        peak_area = 0.0
        for i in range(n - 1):
            above_a = max(0.0, ys[i] - background)
            above_b = max(0.0, ys[i + 1] - background)
            peak_area += 0.5 * abs(xs[i + 1] - xs[i]) * (above_a + above_b)

        cryst_index = peak_area / total_area if total_area > 0 else 0.0

        return ExtractedFeatures(
            features={
                "primary_peak_angle": primary_angle,
                "primary_peak_intensity": primary_intensity,
                "n_peaks": float(len(peaks)),
                "crystallinity_index": cryst_index,
                "mean_peak_width": mean_width,
                "background_level": background,
            },
            extractor_name=self.name(),
            extractor_version=self.version(),
            feature_hash=h,
        )


# ── Version-locked registry ──────────────────────────────


class VersionLockedRegistry(FeatureExtractorRegistry):
    """Registry with allowlist/denylist and version locking.

    Extends :class:`FeatureExtractorRegistry` with:
    - Allowlist/denylist filtering of extractors
    - Version pinning: lock an extractor to a specific version
    - Consistency checking for determinism verification
    """

    def __init__(
        self,
        allowlist: list[str] | None = None,
        denylist: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._allowlist: set[str] | None = set(allowlist) if allowlist else None
        self._denylist: set[str] = set(denylist) if denylist else set()
        self._version_locks: dict[str, str] = {}

    def register(self, extractor_class: type[FeatureExtractor], **kwargs: Any) -> None:
        """Register an extractor, subject to allowlist/denylist."""
        if not (isinstance(extractor_class, type) and issubclass(extractor_class, FeatureExtractor)):
            raise TypeError(
                f"Expected a subclass of FeatureExtractor, got {extractor_class!r}"
            )
        instance = extractor_class(**kwargs)
        ext_name = instance.name()

        if self._allowlist is not None and ext_name not in self._allowlist:
            raise ValueError(
                f"Extractor '{ext_name}' is not in the allowlist"
            )
        if ext_name in self._denylist:
            raise ValueError(
                f"Extractor '{ext_name}' is in the denylist"
            )
        if ext_name in self._extractors:
            raise ValueError(f"Extractor '{ext_name}' is already registered")
        self._extractors[ext_name] = instance

    def lock_version(self, name: str, version: str) -> None:
        """Pin an extractor to a specific version.

        Raises :class:`KeyError` if the extractor is not registered.
        Raises :class:`ValueError` if the registered version differs.
        """
        ext = self.get(name)
        if ext.version() != version:
            raise ValueError(
                f"Extractor '{name}' is version {ext.version()!r}, "
                f"cannot lock to {version!r}"
            )
        self._version_locks[name] = version

    def get_locked_versions(self) -> dict[str, str]:
        """Return a copy of all version locks."""
        return dict(self._version_locks)

    def check_version_compliance(self) -> list[str]:
        """Return list of violations where registered version != locked version."""
        violations: list[str] = []
        for name, locked_ver in self._version_locks.items():
            if name not in self._extractors:
                violations.append(f"Locked extractor '{name}' is not registered")
                continue
            actual = self._extractors[name].version()
            if actual != locked_ver:
                violations.append(
                    f"Extractor '{name}': locked={locked_ver}, actual={actual}"
                )
        return violations


# ── Consistency checking ─────────────────────────────────


@dataclass
class ConsistencyReport:
    """Result of checking extractor determinism.

    Attributes
    ----------
    is_consistent : bool
        True if all runs produced identical results.
    n_runs : int
        Number of runs performed.
    feature_deltas : dict[str, float]
        Maximum absolute delta per feature across runs (0.0 if consistent).
    """
    is_consistent: bool
    n_runs: int
    feature_deltas: dict[str, float] = field(default_factory=dict)


def check_extractor_consistency(
    curve: CurveData,
    extractor: FeatureExtractor,
    n_runs: int = 3,
) -> ConsistencyReport:
    """Verify that an extractor produces identical results on repeated runs.

    Parameters
    ----------
    curve :
        Input curve.
    extractor :
        Extractor to test.
    n_runs :
        Number of times to extract (default 3).

    Returns
    -------
    ConsistencyReport
    """
    results: list[ExtractedFeatures] = []
    for _ in range(n_runs):
        results.append(extractor.extract(curve))

    # Compare all against first run
    ref = results[0]
    deltas: dict[str, float] = {k: 0.0 for k in ref.features}
    consistent = True

    for run in results[1:]:
        if run.features != ref.features:
            consistent = False
        for k in ref.features:
            d = abs(run.features.get(k, 0.0) - ref.features[k])
            if d > deltas[k]:
                deltas[k] = d
        if run.feature_hash != ref.feature_hash:
            consistent = False

    return ConsistencyReport(
        is_consistent=consistent,
        n_runs=n_runs,
        feature_deltas=deltas,
    )


# ── Curve embedding (dimensionality reduction) ───────────


@dataclass
class CurveEmbedding:
    """Low-dimensional embedding of a curve.

    Attributes
    ----------
    components : list[float]
        Latent vector (length = n_components).
    explained_variance : list[float]
        Variance explained by each component.
    reconstruction_error : float
        Mean squared error of reconstructing the original from the embedding.
    """
    components: list[float]
    explained_variance: list[float]
    reconstruction_error: float


class CurveEmbedder:
    """Simple PCA-based dimensionality reduction for curves.

    Fits a principal component decomposition on a collection of curves
    (resampled to uniform length), then projects new curves into the
    latent space.

    Parameters
    ----------
    n_components : int
        Number of latent dimensions (default 3).
    n_points : int
        Number of points to resample each curve to (default 100).
    """

    def __init__(self, n_components: int = 3, n_points: int = 100) -> None:
        self._n_components = n_components
        self._n_points = n_points
        self._mean: list[float] | None = None
        self._components: list[list[float]] | None = None
        self._explained_variance: list[float] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None

    def fit(self, curves: list[CurveData]) -> None:
        """Fit the embedder on a collection of curves."""
        if not curves:
            raise ValueError("Cannot fit on empty curve collection")

        # Resample all curves to uniform length
        matrix = [self._resample(c) for c in curves]
        n = len(matrix)
        d = self._n_points

        # Compute mean
        self._mean = [0.0] * d
        for row in matrix:
            for j in range(d):
                self._mean[j] += row[j]
        for j in range(d):
            self._mean[j] /= n

        # Center the data
        centered = []
        for row in matrix:
            centered.append([row[j] - self._mean[j] for j in range(d)])

        # Compute covariance matrix (d x d) via power iteration for top-k eigenvectors
        # For efficiency, we compute C = X^T X / n where X is (n x d)
        k = min(self._n_components, d, n)
        self._components = []
        self._explained_variance = []

        residual = [row[:] for row in centered]
        for _ in range(k):
            comp, var = self._power_iteration(residual, d)
            self._components.append(comp)
            self._explained_variance.append(var)
            # Deflate: subtract projection onto this component
            for i in range(len(residual)):
                proj = sum(residual[i][j] * comp[j] for j in range(d))
                for j in range(d):
                    residual[i][j] -= proj * comp[j]

    def transform(self, curve: CurveData) -> CurveEmbedding:
        """Project a single curve into the latent space.

        Raises :class:`RuntimeError` if :meth:`fit` has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError("CurveEmbedder has not been fitted yet")

        resampled = self._resample(curve)
        d = self._n_points
        centered = [resampled[j] - self._mean[j] for j in range(d)]

        components: list[float] = []
        for comp in self._components:
            proj = sum(centered[j] * comp[j] for j in range(d))
            components.append(proj)

        # Reconstruction error
        reconstructed = list(self._mean)
        for ci, comp in enumerate(self._components):
            for j in range(d):
                reconstructed[j] += components[ci] * comp[j]

        mse = sum((resampled[j] - reconstructed[j]) ** 2 for j in range(d)) / d

        return CurveEmbedding(
            components=components,
            explained_variance=list(self._explained_variance),
            reconstruction_error=mse,
        )

    def fit_transform(self, curves: list[CurveData]) -> list[CurveEmbedding]:
        """Fit on *curves* and return embeddings for all."""
        self.fit(curves)
        return [self.transform(c) for c in curves]

    # ── Internal helpers ──────────────────────────────────

    def _resample(self, curve: CurveData) -> list[float]:
        """Resample curve to self._n_points via linear interpolation."""
        xs, ys = curve.x_values, curve.y_values
        n = len(xs)
        if n == 0:
            return [0.0] * self._n_points
        if n == 1:
            return [ys[0]] * self._n_points

        # Linear interpolation at uniform points
        x_min, x_max = xs[0], xs[-1]
        if x_min == x_max:
            return [ys[0]] * self._n_points

        step = (x_max - x_min) / (self._n_points - 1)
        result: list[float] = []
        j = 0
        for i in range(self._n_points):
            target_x = x_min + i * step
            while j < n - 2 and xs[j + 1] < target_x:
                j += 1
            if xs[j + 1] == xs[j]:
                result.append(ys[j])
            else:
                t = (target_x - xs[j]) / (xs[j + 1] - xs[j])
                result.append(ys[j] + t * (ys[j + 1] - ys[j]))
        return result

    @staticmethod
    def _power_iteration(
        data: list[list[float]], d: int, max_iter: int = 100, tol: float = 1e-10,
    ) -> tuple[list[float], float]:
        """Find the top eigenvector of X^T X / n via power iteration."""
        n = len(data)
        if n == 0:
            return [0.0] * d, 0.0

        # Initialize with first data row (or ones)
        v = [1.0 / math.sqrt(d)] * d

        for _ in range(max_iter):
            # w = X^T (X v) / n
            xv = [sum(data[i][j] * v[j] for j in range(d)) for i in range(n)]
            w = [0.0] * d
            for i in range(n):
                for j in range(d):
                    w[j] += data[i][j] * xv[i]
            for j in range(d):
                w[j] /= n

            # Normalize
            norm = math.sqrt(sum(wj ** 2 for wj in w))
            if norm < tol:
                return [0.0] * d, 0.0
            v_new = [wj / norm for wj in w]

            # Check convergence
            diff = sum((v_new[j] - v[j]) ** 2 for j in range(d))
            v = v_new
            if diff < tol:
                break

        # Eigenvalue = v^T (X^T X / n) v = ||X v||^2 / n
        xv = [sum(data[i][j] * v[j] for j in range(d)) for i in range(n)]
        eigenvalue = sum(x ** 2 for x in xv) / n

        return v, eigenvalue

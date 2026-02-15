"""Domain-specific parameter encoding framework.

Maps raw parameter values to feature representations suitable for
surrogate models. Supports one-hot, ordinal, custom descriptor,
and spatial encodings with a pipeline for multi-parameter encoding.

Key insight from BayBE (ACS Cent. Sci. 2024): domain-specific encodings
(e.g., chemical descriptors instead of one-hot) can yield 2x speedup
in optimization convergence.

References:
- BayBE: Domain-aware Bayesian optimization with descriptor encodings
- Molecular fingerprints: ECFP, MACCS, RDKit descriptors
- Spatial embeddings: 3D sphere projection for geographic coordinates
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any


class Encoding(ABC):
    """Base class for parameter encodings.

    Maps raw parameter values to feature representations suitable for
    surrogate models. Each encoding must implement encode/decode and
    declare its output dimensionality via n_features.

    Subclasses handle different parameter types:
    - Categorical (unordered): OneHotEncoding
    - Categorical (ordered): OrdinalEncoding
    - Categorical (with descriptors): CustomDescriptorEncoding
    - Spatial coordinates: SpatialEncoding
    """

    @abstractmethod
    def encode(self, value: Any) -> list[float]:
        """Encode a single value to feature vector.

        Args:
            value: Raw parameter value.

        Returns:
            List of floats representing the encoded features.
        """

    @abstractmethod
    def decode(self, features: list[float]) -> Any:
        """Decode feature vector back to original value.

        Args:
            features: List of floats from encode().

        Returns:
            Reconstructed parameter value (may be approximate).
        """

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of features after encoding."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize encoding configuration.

        Returns:
            Dictionary with encoding type and configuration.
        """
        return {
            "type": self.__class__.__name__,
            "n_features": self.n_features,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_features={self.n_features})"


class OneHotEncoding(Encoding):
    """Standard one-hot encoding for categorical parameters.

    Maps each category to a binary vector where exactly one position
    is 1.0 and all others are 0.0. Decoding uses argmax.

    Example:
        >>> enc = OneHotEncoding(["red", "green", "blue"])
        >>> enc.encode("green")
        [0.0, 1.0, 0.0]
        >>> enc.decode([0.1, 0.9, 0.2])
        'green'
    """

    def __init__(self, categories: list) -> None:
        """Initialize one-hot encoding.

        Args:
            categories: List of category values. Order determines encoding.

        Raises:
            ValueError: If categories is empty or contains duplicates.
        """
        if not categories:
            raise ValueError("categories must be non-empty")
        if len(set(str(c) for c in categories)) != len(categories):
            raise ValueError("categories must be unique")
        self._categories = list(categories)
        self._cat_to_idx: dict[Any, int] = {c: i for i, c in enumerate(categories)}

    @property
    def categories(self) -> list:
        """Return copy of categories list."""
        return list(self._categories)

    def encode(self, value: Any) -> list[float]:
        """Encode value as one-hot vector.

        Args:
            value: Must be one of the known categories.

        Returns:
            Binary vector with 1.0 at the category index.

        Raises:
            ValueError: If value is not a known category.
        """
        if value not in self._cat_to_idx:
            raise ValueError(
                f"Unknown category {value!r}. "
                f"Known categories: {self._categories}"
            )
        vec = [0.0] * len(self._categories)
        vec[self._cat_to_idx[value]] = 1.0
        return vec

    def decode(self, features: list[float]) -> Any:
        """Decode one-hot vector via argmax.

        Args:
            features: Feature vector of length n_features.

        Returns:
            Category corresponding to the maximum value index.

        Raises:
            ValueError: If features length does not match n_features.
        """
        if len(features) != len(self._categories):
            raise ValueError(
                f"Expected {len(self._categories)} features, "
                f"got {len(features)}"
            )
        max_idx = 0
        max_val = features[0]
        for i in range(1, len(features)):
            if features[i] > max_val:
                max_val = features[i]
                max_idx = i
        return self._categories[max_idx]

    @property
    def n_features(self) -> int:
        """Number of features equals number of categories."""
        return len(self._categories)

    def to_dict(self) -> dict[str, Any]:
        """Serialize one-hot encoding configuration."""
        base = super().to_dict()
        base["categories"] = list(self._categories)
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OneHotEncoding:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with 'categories' key.

        Returns:
            Restored OneHotEncoding instance.
        """
        return cls(categories=data["categories"])


class OrdinalEncoding(Encoding):
    """Ordinal encoding for ordered categories.

    Maps ordered categories to evenly-spaced values in [0, 1].
    Preserves ordering information that one-hot encoding loses.

    Example:
        >>> enc = OrdinalEncoding(["low", "medium", "high"])
        >>> enc.encode("medium")
        [0.5]
        >>> enc.decode([0.6])
        'medium'
    """

    def __init__(self, ordered_categories: list) -> None:
        """Initialize ordinal encoding.

        Args:
            ordered_categories: Categories in ascending order.

        Raises:
            ValueError: If ordered_categories is empty or contains duplicates.
        """
        if not ordered_categories:
            raise ValueError("ordered_categories must be non-empty")
        if len(set(str(c) for c in ordered_categories)) != len(ordered_categories):
            raise ValueError("ordered_categories must be unique")
        self._categories = list(ordered_categories)
        self._cat_to_idx: dict[Any, int] = {
            c: i for i, c in enumerate(ordered_categories)
        }

    @property
    def categories(self) -> list:
        """Return copy of ordered categories list."""
        return list(self._categories)

    def encode(self, value: Any) -> list[float]:
        """Encode value as normalized ordinal in [0, 1].

        Single category maps to [0.0]. Multiple categories map to
        index / (n - 1).

        Args:
            value: Must be one of the known ordered categories.

        Returns:
            Single-element list with the normalized ordinal value.

        Raises:
            ValueError: If value is not a known category.
        """
        if value not in self._cat_to_idx:
            raise ValueError(
                f"Unknown category {value!r}. "
                f"Known categories: {self._categories}"
            )
        idx = self._cat_to_idx[value]
        n = len(self._categories)
        if n == 1:
            return [0.0]
        return [idx / (n - 1)]

    def decode(self, features: list[float]) -> Any:
        """Decode normalized ordinal back to nearest category.

        Args:
            features: Single-element list with value in [0, 1].

        Returns:
            Nearest category by rounding to the closest index.

        Raises:
            ValueError: If features length is not 1.
        """
        if len(features) != 1:
            raise ValueError(f"Expected 1 feature, got {len(features)}")
        val = features[0]
        n = len(self._categories)
        if n == 1:
            return self._categories[0]
        # Map [0, 1] back to index
        idx = round(val * (n - 1))
        idx = max(0, min(n - 1, idx))
        return self._categories[idx]

    @property
    def n_features(self) -> int:
        """Ordinal encoding always produces 1 feature."""
        return 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize ordinal encoding configuration."""
        base = super().to_dict()
        base["ordered_categories"] = list(self._categories)
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrdinalEncoding:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with 'ordered_categories' key.

        Returns:
            Restored OrdinalEncoding instance.
        """
        return cls(ordered_categories=data["ordered_categories"])


class CustomDescriptorEncoding(Encoding):
    """User-provided descriptor encoding for categorical parameters.

    Maps categories to user-defined feature vectors (descriptors).
    This enables domain knowledge to improve surrogate model quality.

    BayBE demonstrated 2x optimization speedup using chemical descriptors
    (e.g., molecular weight, logP, TPSA) instead of one-hot encoding.

    Example:
        >>> descriptors = {
        ...     "ethanol": [46.07, -0.31, 20.23],
        ...     "methanol": [32.04, -0.74, 20.23],
        ...     "acetone": [58.08, -0.24, 17.07],
        ... }
        >>> enc = CustomDescriptorEncoding(descriptors)
        >>> enc.encode("ethanol")
        [46.07, -0.31, 20.23]
    """

    def __init__(self, descriptor_table: dict[str, list[float]]) -> None:
        """Initialize custom descriptor encoding.

        Args:
            descriptor_table: Mapping from category name to descriptor vector.
                All descriptor vectors must have the same length.

        Raises:
            ValueError: If table is empty or descriptor lengths are inconsistent.
        """
        if not descriptor_table:
            raise ValueError("descriptor_table must be non-empty")
        self._table: dict[str, list[float]] = {
            k: list(v) for k, v in descriptor_table.items()
        }
        lengths = {len(v) for v in self._table.values()}
        if len(lengths) > 1:
            raise ValueError(
                f"All descriptor vectors must have the same length, "
                f"got lengths: {lengths}"
            )
        self._n_features = lengths.pop()

    @property
    def categories(self) -> list[str]:
        """Return list of category names."""
        return list(self._table.keys())

    def encode(self, value: Any) -> list[float]:
        """Look up descriptor vector for the given category.

        Args:
            value: Category name (must be a key in descriptor_table).

        Returns:
            Copy of the descriptor vector for this category.

        Raises:
            ValueError: If value is not in the descriptor table.
        """
        key = str(value)
        if key not in self._table:
            raise ValueError(
                f"Unknown category {value!r}. "
                f"Known categories: {list(self._table.keys())}"
            )
        return list(self._table[key])

    def decode(self, features: list[float]) -> Any:
        """Decode by finding nearest neighbor in descriptor table.

        Uses Euclidean distance to find the closest known descriptor.

        Args:
            features: Feature vector of length n_features.

        Returns:
            Category name whose descriptor is closest.

        Raises:
            ValueError: If features length does not match n_features.
        """
        if len(features) != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, "
                f"got {len(features)}"
            )
        best_cat = None
        best_dist = float("inf")
        for cat, desc in self._table.items():
            dist = sum((a - b) ** 2 for a, b in zip(features, desc))
            if dist < best_dist:
                best_dist = dist
                best_cat = cat
        return best_cat

    @property
    def n_features(self) -> int:
        """Number of features from descriptor vector length."""
        return self._n_features

    def to_dict(self) -> dict[str, Any]:
        """Serialize custom descriptor encoding configuration."""
        base = super().to_dict()
        base["descriptor_table"] = {k: list(v) for k, v in self._table.items()}
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CustomDescriptorEncoding:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with 'descriptor_table' key.

        Returns:
            Restored CustomDescriptorEncoding instance.
        """
        return cls(descriptor_table=data["descriptor_table"])


class SpatialEncoding(Encoding):
    """Spatial coordinate encoding using 3D sphere embedding.

    Converts latitude/longitude to 3D Cartesian coordinates on
    the unit sphere. This preserves spatial distance relationships
    that are lost in raw lat/lon representation (e.g., wrapping
    at the date line).

    Encoding: (lat, lon) -> (cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat))
    where lat and lon are in degrees.

    Example:
        >>> enc = SpatialEncoding()
        >>> features = enc.encode((0.0, 0.0))  # Equator, prime meridian
        >>> len(features)
        3
        >>> abs(features[0] - 1.0) < 1e-10  # cos(0)*cos(0) = 1
        True
    """

    def __init__(self, coord_type: str = "latlon") -> None:
        """Initialize spatial encoding.

        Args:
            coord_type: Coordinate system type. Currently supports "latlon"
                (latitude/longitude in degrees).

        Raises:
            ValueError: If coord_type is not supported.
        """
        supported = ("latlon",)
        if coord_type not in supported:
            raise ValueError(
                f"Unsupported coord_type {coord_type!r}. "
                f"Supported: {supported}"
            )
        self._type = coord_type

    def encode(self, value: Any) -> list[float]:
        """Encode spatial coordinates to 3D unit sphere.

        Args:
            value: Tuple of (latitude, longitude) in degrees.
                Latitude in [-90, 90], longitude in [-180, 180].

        Returns:
            3D Cartesian coordinates [x, y, z] on unit sphere.

        Raises:
            ValueError: If value is not a 2-element sequence.
        """
        if not hasattr(value, "__len__") or len(value) != 2:
            raise ValueError(
                f"Expected (latitude, longitude) tuple, got {value!r}"
            )
        lat_deg, lon_deg = float(value[0]), float(value[1])
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)
        return [x, y, z]

    def decode(self, features: list[float]) -> tuple[float, float]:
        """Decode 3D unit sphere coordinates back to lat/lon.

        Projects the 3D point onto the unit sphere first (normalizes),
        then converts to latitude/longitude.

        Args:
            features: 3D Cartesian coordinates [x, y, z].

        Returns:
            Tuple of (latitude, longitude) in degrees.

        Raises:
            ValueError: If features length is not 3.
        """
        if len(features) != 3:
            raise ValueError(f"Expected 3 features, got {len(features)}")
        x, y, z = features

        # Normalize to unit sphere
        norm = math.sqrt(x * x + y * y + z * z)
        if norm < 1e-12:
            return (0.0, 0.0)
        x /= norm
        y /= norm
        z /= norm

        # Clamp z for numerical safety
        z = max(-1.0, min(1.0, z))
        lat = math.degrees(math.asin(z))
        lon = math.degrees(math.atan2(y, x))
        return (lat, lon)

    @property
    def n_features(self) -> int:
        """Spatial encoding always produces 3 features (x, y, z)."""
        return 3

    def to_dict(self) -> dict[str, Any]:
        """Serialize spatial encoding configuration."""
        base = super().to_dict()
        base["coord_type"] = self._type
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialEncoding:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with 'coord_type' key.

        Returns:
            Restored SpatialEncoding instance.
        """
        return cls(coord_type=data.get("coord_type", "latlon"))


class EncodingPipeline:
    """Pipeline to encode/decode multiple parameters with different encodings.

    Orchestrates encoding of full parameter dictionaries, mapping each
    parameter through its assigned encoding to produce a flat feature vector.
    Parameters without an explicit encoding are treated as pass-through
    (must be numeric).

    Example:
        >>> pipeline = EncodingPipeline()
        >>> pipeline.add_encoding("solvent", OneHotEncoding(["water", "ethanol"]))
        >>> pipeline.add_encoding("temp_class", OrdinalEncoding(["low", "high"]))
        >>> features = pipeline.encode_params({
        ...     "solvent": "water",
        ...     "temp_class": "high",
        ...     "pressure": 1.5,
        ... })
        >>> len(features)  # 2 (one-hot) + 1 (ordinal) + 1 (pass-through)
        4
    """

    def __init__(self, encodings: dict[str, Encoding] | None = None) -> None:
        """Initialize encoding pipeline.

        Args:
            encodings: Optional mapping from parameter name to Encoding.
                Can be populated later via add_encoding().
        """
        self._encodings: dict[str, Encoding] = dict(encodings) if encodings else {}

    @property
    def encodings(self) -> dict[str, Encoding]:
        """Return copy of the encoding mapping."""
        return dict(self._encodings)

    @property
    def param_names(self) -> list[str]:
        """Return list of parameter names with explicit encodings."""
        return list(self._encodings.keys())

    def add_encoding(self, param_name: str, encoding: Encoding) -> None:
        """Register an encoding for a parameter.

        Args:
            param_name: Name of the parameter.
            encoding: Encoding instance to use for this parameter.
        """
        self._encodings[param_name] = encoding

    def remove_encoding(self, param_name: str) -> bool:
        """Remove an encoding for a parameter.

        Args:
            param_name: Name of the parameter.

        Returns:
            True if the encoding was found and removed.
        """
        if param_name in self._encodings:
            del self._encodings[param_name]
            return True
        return False

    def encode_params(self, params: dict[str, Any]) -> list[float]:
        """Encode full parameter dict to flat feature vector.

        Parameters with registered encodings are encoded using their
        respective Encoding. Parameters without encoding are treated
        as pass-through and must be numeric (int or float).

        The output order is: encoded parameters (in registration order),
        then pass-through parameters (in dict iteration order).

        Args:
            params: Dictionary of parameter name to value.

        Returns:
            Flat list of float features.

        Raises:
            TypeError: If a pass-through parameter is not numeric.
        """
        features: list[float] = []

        # First: parameters with explicit encodings (in registration order)
        for name, encoding in self._encodings.items():
            if name in params:
                encoded = encoding.encode(params[name])
                features.extend(encoded)

        # Second: pass-through parameters (no encoding registered)
        for name, value in params.items():
            if name not in self._encodings:
                if not isinstance(value, (int, float)):
                    raise TypeError(
                        f"Pass-through parameter {name!r} must be numeric, "
                        f"got {type(value).__name__}"
                    )
                features.append(float(value))

        return features

    def decode_features(
        self,
        features: list[float],
        param_names: list[str],
    ) -> dict[str, Any]:
        """Decode feature vector back to parameter dict.

        Reverses the operation of encode_params(). Requires knowing
        the original parameter names and their order.

        Args:
            features: Flat feature vector from encode_params().
            param_names: List of all parameter names in the same order
                they were encoded (encoded params first, then pass-through).

        Returns:
            Dictionary of parameter name to decoded value.

        Raises:
            ValueError: If feature vector length does not match expected.
        """
        result: dict[str, Any] = {}
        offset = 0

        for name in param_names:
            if name in self._encodings:
                encoding = self._encodings[name]
                n = encoding.n_features
                if offset + n > len(features):
                    raise ValueError(
                        f"Feature vector too short: need {offset + n} features "
                        f"for parameter {name!r}, got {len(features)}"
                    )
                chunk = features[offset : offset + n]
                result[name] = encoding.decode(chunk)
                offset += n
            else:
                # Pass-through: single float
                if offset >= len(features):
                    raise ValueError(
                        f"Feature vector too short: need feature at index {offset} "
                        f"for parameter {name!r}, got {len(features)}"
                    )
                result[name] = features[offset]
                offset += 1

        return result

    def total_features(self) -> int:
        """Total number of encoded features from all registered encodings.

        Note: This only counts features from registered encodings.
        Pass-through parameters are not included since the pipeline
        does not know about them until encode_params() is called.

        Returns:
            Sum of n_features across all registered encodings.
        """
        return sum(enc.n_features for enc in self._encodings.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize pipeline configuration.

        Returns:
            Dictionary with encoding configurations per parameter.
        """
        return {
            "encodings": {
                name: enc.to_dict() for name, enc in self._encodings.items()
            },
            "total_features": self.total_features(),
        }

    def __repr__(self) -> str:
        enc_strs = [f"{name}={enc!r}" for name, enc in self._encodings.items()]
        return f"EncodingPipeline({', '.join(enc_strs)})"

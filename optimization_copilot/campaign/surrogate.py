"""Fingerprint-based GP surrogate for molecular optimization.

Wraps :class:`NGramTanimoto` for SMILES encoding and uses the shared
``_math`` package (RBF kernel, Cholesky decomposition) for Gaussian
process inference.  Produces posterior mean and standard deviation
for untested candidates — the core input to acquisition functions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SurrogateFitResult:
    """Summary of a surrogate model fit.

    Parameters
    ----------
    n_training : int
        Number of training observations.
    y_mean : float
        Mean of observed objective values.
    y_std : float
        Std-dev of observed objective values.
    n_features : int
        Fingerprint dimensionality.
    objective_name : str
        Which objective was fitted.
    duration_ms : float
        Wall-clock time for fit (milliseconds).
    """

    n_training: int
    y_mean: float
    y_std: float
    n_features: int
    objective_name: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_training": self.n_training,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "n_features": self.n_features,
            "objective_name": self.objective_name,
            "duration_ms": self.duration_ms,
        }


@dataclass
class PredictionResult:
    """Prediction for a single candidate.

    Parameters
    ----------
    smiles : str
        SMILES string of the candidate.
    mean : float
        Posterior mean (original scale).
    std : float
        Posterior standard deviation (original scale).
    """

    smiles: str
    mean: float
    std: float

    def to_dict(self) -> dict[str, Any]:
        return {"smiles": self.smiles, "mean": self.mean, "std": self.std}


class FingerprintSurrogate:
    """GP surrogate using SMILES fingerprint features.

    Encodes SMILES via :class:`NGramTanimoto` into binary fingerprints,
    then fits a Gaussian process with an RBF kernel.  Predictions return
    posterior (mean, std) on the original objective scale.

    Parameters
    ----------
    n_gram : int
        N-gram size for fingerprinting (default 3).
    fp_size : int
        Fingerprint bit-vector length (default 128).
    length_scale : float
        RBF kernel length-scale (default 1.0).
    noise : float
        Diagonal noise for numerical stability (default 1e-4).
    seed : int
        Random seed (unused currently, reserved for future).
    """

    def __init__(
        self,
        n_gram: int = 3,
        fp_size: int = 128,
        length_scale: float = 1.0,
        noise: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self._n_gram = n_gram
        self._fp_size = fp_size
        self._length_scale = length_scale
        self._noise = noise
        self._seed = seed

        # Fitted state (lazy-populated by fit())
        self._X_train: list[list[float]] = []
        self._y_norm: list[float] = []
        self._alpha: list[float] = []
        self._L: list[list[float]] = []
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._fitted: bool = False
        self._fit_result: SurrogateFitResult | None = None

    @property
    def fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._fitted

    @property
    def fit_result(self) -> SurrogateFitResult | None:
        """Summary from most recent fit, or ``None``."""
        return self._fit_result

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        smiles_list: list[str],
        y_values: list[float],
        objective_name: str = "",
    ) -> SurrogateFitResult:
        """Fit the GP on observed SMILES and objective values.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings for observed molecules.
        y_values : list[float]
            Corresponding objective values.
        objective_name : str
            Label for the objective (informational).

        Returns
        -------
        SurrogateFitResult
            Summary statistics of the fit.

        Raises
        ------
        ValueError
            If fewer than 2 observations, or mismatched lengths.
        """
        if len(smiles_list) != len(y_values):
            raise ValueError(
                f"Length mismatch: {len(smiles_list)} SMILES vs {len(y_values)} y-values"
            )
        if len(smiles_list) < 2:
            raise ValueError(
                f"Need at least 2 observations, got {len(smiles_list)}"
            )

        from optimization_copilot.representation.ngram_tanimoto import NGramTanimoto
        from optimization_copilot.backends._math import (
            cholesky,
            kernel_matrix,
            rbf_kernel,
            solve_cholesky,
        )

        t0 = time.monotonic()

        # 1. Encode SMILES → fingerprints
        encoder = NGramTanimoto(n=self._n_gram, fingerprint_size=self._fp_size)
        X = encoder.encode(smiles_list)

        # 2. Normalise y to zero-mean, unit-variance
        n = len(y_values)
        self._y_mean = sum(y_values) / n
        variance = sum((y - self._y_mean) ** 2 for y in y_values) / max(n - 1, 1)
        self._y_std = math.sqrt(variance) if variance > 1e-12 else 1.0
        self._y_norm = [(y - self._y_mean) / self._y_std for y in y_values]

        # 3. Build kernel matrix K + noise*I
        ls = self._length_scale
        kern_fn = lambda x1, x2: rbf_kernel(x1, x2, length_scale=ls)
        K = kernel_matrix(X, kern_fn, noise=self._noise)

        # 4. Cholesky decomposition: L such that K = L L^T
        self._L = cholesky(K)

        # 5. Solve for alpha: K^{-1} y = (L L^T)^{-1} y
        self._alpha = solve_cholesky(self._L, self._y_norm)

        self._X_train = X
        self._fitted = True

        elapsed = (time.monotonic() - t0) * 1000.0
        self._fit_result = SurrogateFitResult(
            n_training=n,
            y_mean=self._y_mean,
            y_std=self._y_std,
            n_features=self._fp_size,
            objective_name=objective_name,
            duration_ms=elapsed,
        )
        return self._fit_result

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        smiles_list: list[str],
    ) -> list[PredictionResult]:
        """Predict posterior (mean, std) for candidate SMILES.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings for untested candidates.

        Returns
        -------
        list[PredictionResult]
            One prediction per candidate, on the original objective scale.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted — call fit() first")

        from optimization_copilot.representation.ngram_tanimoto import NGramTanimoto
        from optimization_copilot.backends._math import (
            rbf_kernel,
            solve_lower,
            vec_dot,
        )

        encoder = NGramTanimoto(n=self._n_gram, fingerprint_size=self._fp_size)
        X_cand = encoder.encode(smiles_list)

        ls = self._length_scale
        results: list[PredictionResult] = []

        for idx, x_star in enumerate(X_cand):
            # k_star = [k(x_star, x_i) for each training point]
            k_star = [rbf_kernel(x_star, xi, length_scale=ls) for xi in self._X_train]

            # Posterior mean (normalised): mu* = k_star^T alpha
            mu_norm = vec_dot(k_star, self._alpha)

            # Posterior variance: sigma*^2 = k(x*, x*) - v^T v
            #   where v = L^{-1} k_star
            v = solve_lower(self._L, k_star)
            k_ss = rbf_kernel(x_star, x_star, length_scale=ls) + self._noise
            var = k_ss - vec_dot(v, v)
            sigma_norm = math.sqrt(max(var, 1e-10))

            # De-normalise to original scale
            mu = mu_norm * self._y_std + self._y_mean
            sigma = sigma_norm * self._y_std

            results.append(PredictionResult(
                smiles=smiles_list[idx],
                mean=mu,
                std=sigma,
            ))

        return results

    # ------------------------------------------------------------------
    # Cross-validation error (leave-one-out approximate)
    # ------------------------------------------------------------------

    def loo_errors(
        self,
        smiles_list: list[str],
        y_values: list[float],
    ) -> list[dict[str, Any]]:
        """Approximate leave-one-out prediction errors.

        Fits the model n times (each time leaving one observation out)
        and compares the held-out prediction to the actual value.

        Parameters
        ----------
        smiles_list : list[str]
            All observed SMILES.
        y_values : list[float]
            Corresponding objective values.

        Returns
        -------
        list[dict]
            One dict per observation: ``{"smiles", "actual", "predicted", "error"}``.
        """
        if len(smiles_list) < 3:
            return []

        errors: list[dict[str, Any]] = []
        for i in range(len(smiles_list)):
            train_smiles = smiles_list[:i] + smiles_list[i + 1:]
            train_y = y_values[:i] + y_values[i + 1:]
            test_smiles = smiles_list[i]
            test_y = y_values[i]

            clone = FingerprintSurrogate(
                n_gram=self._n_gram,
                fp_size=self._fp_size,
                length_scale=self._length_scale,
                noise=self._noise,
                seed=self._seed,
            )
            try:
                clone.fit(train_smiles, train_y)
                preds = clone.predict([test_smiles])
                pred_mu = preds[0].mean
                errors.append({
                    "smiles": test_smiles,
                    "actual": test_y,
                    "predicted": pred_mu,
                    "error": test_y - pred_mu,
                })
            except Exception:
                errors.append({
                    "smiles": test_smiles,
                    "actual": test_y,
                    "predicted": None,
                    "error": None,
                })

        return errors

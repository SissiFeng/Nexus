"""Pure-Python statistical distribution helpers (no external dependencies).

Provides standard normal PDF, CDF, inverse CDF, and related functions.
"""

from __future__ import annotations

import math


def norm_pdf(x: float) -> float:
    """Standard normal probability density function.

    Parameters
    ----------
    x : float
        Point at which to evaluate the PDF.

    Returns
    -------
    float
        The density phi(x) = exp(-x^2/2) / sqrt(2*pi).
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function (approximation).

    Uses ``math.erf`` for a closed-form approximation.

    Parameters
    ----------
    x : float
        Point at which to evaluate the CDF.

    Returns
    -------
    float
        The cumulative probability Phi(x).
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (percent point function).

    Uses the rational approximation by Peter Acklam, accurate to
    approximately 1.15e-9 in the full range (0, 1).

    Parameters
    ----------
    p : float
        Probability in (0, 1).

    Returns
    -------
    float
        The value x such that Phi(x) = p.

    Raises
    ------
    ValueError
        If *p* is not in the open interval (0, 1).
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Coefficients for the rational approximation
    a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ]
    d = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # Rational approximation for lower region
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        # Rational approximation for central region
        q = p - 0.5
        r = q * q
        return (
            ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
        ) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    else:
        # Rational approximation for upper region
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )


def norm_logpdf(x: float) -> float:
    """Log of the standard normal probability density function.

    More numerically stable than ``math.log(norm_pdf(x))`` for large |x|.

    Parameters
    ----------
    x : float
        Point at which to evaluate the log-PDF.

    Returns
    -------
    float
        The value ``-0.5 * x^2 - 0.5 * log(2*pi)``.
    """
    return -0.5 * x * x - 0.5 * math.log(2.0 * math.pi)


def binary_entropy(p: float) -> float:
    """Binary entropy function H(p) = -p*log2(p) - (1-p)*log2(1-p).

    Parameters
    ----------
    p : float
        Probability in [0, 1].

    Returns
    -------
    float
        The binary entropy in bits.  Returns 0.0 at the boundaries
        p=0 and p=1.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

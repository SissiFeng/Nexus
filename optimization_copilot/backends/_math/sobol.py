"""Pure-Python Sobol quasi-random sequence generator.

Provides direction numbers and the gray-code-ordered Sobol sequence
supporting up to 21 dimensions.
"""

from __future__ import annotations


# Pre-computed direction numbers for dimensions 1-21 (simplified Joe-Kuo).
# Dimension 0 uses the Van der Corput sequence.
SOBOL_DIRECTION_NUMBERS: list[list[int]] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # dim 1 (poly=0, s=1, a=0)
    [1, 1, 1, 3, 1, 3, 1, 3, 3, 1],  # dim 2
    [1, 3, 1, 3, 3, 1, 3, 3, 1, 3],  # dim 3
    [1, 3, 5, 13, 7, 11, 1, 3, 7, 5],  # dim 4
    [1, 1, 5, 5, 15, 1, 9, 7, 5, 11],  # dim 5
    [1, 3, 1, 7, 9, 13, 11, 1, 15, 7],  # dim 6
    [1, 1, 3, 7, 13, 3, 15, 1, 9, 5],  # dim 7
    [1, 3, 3, 9, 3, 5, 1, 15, 13, 9],  # dim 8
    [1, 3, 7, 7, 1, 15, 9, 13, 7, 3],  # dim 9
    [1, 1, 5, 11, 1, 3, 7, 9, 3, 15],  # dim 10
    [1, 3, 5, 5, 3, 15, 7, 1, 13, 9],  # dim 11
    [1, 1, 1, 15, 15, 3, 9, 7, 5, 11],  # dim 12
    [1, 3, 7, 3, 13, 1, 5, 9, 15, 7],  # dim 13
    [1, 1, 3, 9, 7, 5, 13, 1, 15, 3],  # dim 14
    [1, 3, 1, 13, 5, 11, 7, 15, 9, 1],  # dim 15
    [1, 1, 7, 11, 3, 13, 15, 5, 1, 9],  # dim 16
    [1, 3, 5, 9, 15, 7, 1, 11, 3, 13],  # dim 17
    [1, 1, 1, 3, 11, 9, 13, 15, 7, 5],  # dim 18
    [1, 3, 3, 5, 1, 7, 11, 9, 13, 15],  # dim 19
    [1, 1, 5, 7, 9, 11, 3, 13, 15, 1],  # dim 20
]


def sobol_sequence(n_points: int, n_dims: int) -> list[list[float]]:
    """Generate *n_points* Sobol quasi-random numbers in *n_dims* dimensions.

    Uses gray-code ordering for efficiency.  Supports up to 21 dimensions.

    Parameters
    ----------
    n_points : int
        Number of points to generate.
    n_dims : int
        Number of dimensions (capped at 21).

    Returns
    -------
    list[list[float]]
        A list of *n_points* vectors, each of length *n_dims*, with
        components in [0, 1).
    """
    max_bits = 30
    n_dims = min(n_dims, 21)

    # Build direction number tables (V[dim][bit])
    V: list[list[int]] = []
    for d in range(n_dims):
        v = [0] * (max_bits + 1)
        if d == 0:
            # Van der Corput in base 2
            for i in range(1, max_bits + 1):
                v[i] = 1 << (max_bits - i)
        else:
            dn = SOBOL_DIRECTION_NUMBERS[d - 1] if d - 1 < len(SOBOL_DIRECTION_NUMBERS) else [1] * 10
            s = min(len(dn), max_bits)
            for i in range(1, s + 1):
                v[i] = dn[i - 1] << (max_bits - i)
            for i in range(s + 1, max_bits + 1):
                v[i] = v[i - s] ^ (v[i - s] >> s)
                for k in range(1, s):
                    v[i] ^= ((dn[k - 1] >> (s - 1 - k)) & 1) * v[i - k] if k < len(dn) else 0
        V.append(v)

    denom = float(1 << max_bits)
    points: list[list[float]] = []
    x = [0] * n_dims  # current point as integers

    for i in range(n_points):
        if i == 0:
            points.append([0.0] * n_dims)
        else:
            # Find rightmost zero bit of (i - 1)
            c = 1
            val = i - 1
            while val & 1:
                val >>= 1
                c += 1
            for d in range(n_dims):
                x[d] ^= V[d][c] if c <= max_bits else V[d][max_bits]
            points.append([x[d] / denom for d in range(n_dims)])

    return points

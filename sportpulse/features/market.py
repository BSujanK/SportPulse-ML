"""Bookmaker odds -> fair probabilities.

Shin (1993) de-vigging models the bookmaker margin as protection against
insider trading and is the standard, best-calibrated method for 1X2 markets
(Štrumbelj 2014, "On determining probability forecasts from betting odds").
"""
from __future__ import annotations

import numpy as np


def shin_probabilities(odds: np.ndarray, iters: int = 60) -> np.ndarray:
    """Vectorised Shin de-vigging.

    odds: (n, k) decimal odds, NaN rows allowed. Returns (n, k) probabilities
    summing to 1 (NaN where input invalid).
    """
    odds = np.asarray(odds, dtype=float)
    q = 1.0 / odds
    s = q.sum(axis=1, keepdims=True)
    valid = np.isfinite(s[:, 0]) & (s[:, 0] > 1.0)

    def probs(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / s) - z) / (2 * (1 - z))

    lo = np.zeros((len(q), 1))
    hi = np.full((len(q), 1), 0.4)
    for _ in range(iters):  # bisection on the insider fraction z
        mid = (lo + hi) / 2
        total = np.nansum(probs(mid), axis=1, keepdims=True)
        too_big = total > 1
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    p = probs((lo + hi) / 2)
    p = p / p.sum(axis=1, keepdims=True)
    # Books with no margin (s <= 1): fall back to simple normalisation.
    p_norm = q / s
    p = np.where(valid[:, None], p, p_norm)
    p[~np.isfinite(s[:, 0])] = np.nan
    return p


def two_way_probabilities(odds: np.ndarray) -> np.ndarray:
    """Proportional de-vig for two-outcome markets (over/under, Asian handicap)."""
    q = 1.0 / np.asarray(odds, dtype=float)
    return q / q.sum(axis=1, keepdims=True)

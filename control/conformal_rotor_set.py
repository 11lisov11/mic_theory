"""Convert a modulo-angle posterior to a fixed-threshold ranked set.

The runtime functions use only posterior probabilities and a threshold
calibrated outside the controller. They never receive the true rotor angle.
"""

from __future__ import annotations

import math

import numpy as np


def rank_mass(marginal: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(marginal, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) < 2:
        raise ValueError("a one-dimensional posterior with at least two bins is required")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("posterior probabilities must be finite and nonnegative")
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("posterior mass must be positive")
    probabilities = probabilities / total
    order = np.argsort(-probabilities, kind="stable")
    scores = np.empty(len(probabilities), dtype=float)
    scores[order] = np.r_[0.0, np.cumsum(probabilities[order[:-1]])]
    return scores


def ranked_set(marginal: np.ndarray, threshold: float, period_rad: float) -> tuple[np.ndarray, float]:
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    if not math.isfinite(period_rad) or period_rad <= 0.0:
        raise ValueError("angle period must be finite and positive")
    indices = np.flatnonzero(rank_mass(marginal) <= threshold)
    return indices, float(len(indices) * period_rad / len(marginal))

"""A posterior-width availability gate for modulo-angle estimates.

Runtime decisions use only the current angle marginal and a frozen width
threshold. Error labels are deliberately absent from this module.
"""

from __future__ import annotations

import math

import numpy as np


def hpd_width_deg(marginal: np.ndarray, period_deg: float, mass: float = 0.95) -> float:
    probabilities = np.asarray(marginal, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) < 2:
        raise ValueError("a one-dimensional posterior with at least two bins is required")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("posterior probabilities must be finite and nonnegative")
    total = float(np.sum(probabilities))
    if total <= 0.0 or not math.isfinite(period_deg) or period_deg <= 0.0:
        raise ValueError("posterior mass and period must be positive and finite")
    if not 0.0 < mass < 1.0:
        raise ValueError("mass must be in (0,1)")
    sorted_mass = np.sort(probabilities / total)[::-1]
    count = int(np.searchsorted(np.cumsum(sorted_mass), mass, side="left")) + 1
    return float(period_deg * count / len(probabilities))


def accept_estimate(marginal: np.ndarray, period_deg: float,
                    max_width_deg: float, mass: float = 0.95) -> bool:
    if not math.isfinite(max_width_deg) or max_width_deg <= 0.0:
        raise ValueError("maximum width must be finite and positive")
    return hpd_width_deg(marginal, period_deg, mass) <= max_width_deg

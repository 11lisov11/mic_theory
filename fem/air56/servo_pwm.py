"""Centered five-segment space-vector PWM for an ideal two-level three-phase bridge."""

import numpy as np

ACTIVE = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]])
CLARKE = (2 / 3) * np.array([[1, -0.5, -0.5], [0, np.sqrt(3) / 2, -np.sqrt(3) / 2]])


def segments(voltage, period, dc=310.0):
    if np.linalg.norm(voltage) < 1e-11:
        return [(np.zeros(2), period)]
    sector = int(np.floor((np.arctan2(voltage[1], voltage[0]) % (2 * np.pi)) / (np.pi / 3))) % 6
    states = ACTIVE[[sector, (sector + 1) % 6]]
    vectors = dc * states @ CLARKE.T
    d1, d2 = np.maximum(0.0, np.linalg.solve(vectors.T, voltage))
    if d1 + d2 > 1 + 1e-10:
        raise ValueError("Voltage command outside bridge hexagon")
    z = max(0.0, 1 - d1 - d2)
    # The zero state adjacent to the first active state is 000 or 111.
    zero = np.full(3, int(states[0].sum() == 2))
    pattern = [(zero, z / 2), (states[0], d1 / 2), (states[1], d2), (states[0], d1 / 2), (zero, z / 2)]
    return [(dc * (CLARKE @ s), period * d) for s, d in pattern if d > 1e-10]


def segment_tangents(voltage, period, dc=310.0):
    """Partial derivatives of the active and zero dwell lengths in the current sector."""
    sector = int(np.floor((np.arctan2(voltage[1], voltage[0]) % (2 * np.pi)) / (np.pi / 3))) % 6
    v = dc * ACTIVE[[sector, (sector + 1) % 6]] @ CLARKE.T
    inv = np.linalg.inv(v.T)
    d1, d2 = np.maximum(0, inv @ voltage)
    dz = -inv.sum(axis=0)
    z = max(0.0, 1.0 - d1 - d2)
    return [
        (vec, period * d, period * der)
        for vec, d, der in [
            (np.zeros(2), z / 2, dz / 2),
            (v[0], d1 / 2, inv[0] / 2),
            (v[1], d2, inv[1]),
            (v[0], d1 / 2, inv[0] / 2),
            (np.zeros(2), z / 2, dz / 2),
        ]
        if d > 1e-10
    ]

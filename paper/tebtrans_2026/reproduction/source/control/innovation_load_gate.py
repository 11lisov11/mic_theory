"""Causal load-uncertainty gate using measured-current innovations only."""

from collections import deque
import math

import numpy as np


class InnovationLoadGate:
    def __init__(self, *, mode="coherence", window=32, threshold=20.0,
                 hold_steps=100, low=0.0001, high=0.1):
        if mode not in ("coherence", "energy"):
            raise ValueError("unknown gate mode")
        if window < 2 or hold_steps < 1 or not 0 <= low <= high or threshold <= 0:
            raise ValueError("invalid gate configuration")
        self.mode, self.window, self.threshold = mode, window, threshold
        self.low, self.high, self.hold_steps = low, high, hold_steps
        self.values = deque(maxlen=window)
        self.remaining = 0
        self.statistic = 0.0
        self.active_steps = 0

    def update(self, innovation, covariance, estimated_flux, *, enabled):
        if not enabled:
            self.values.clear()
            self.remaining = 0
            return self.low
        angle = math.atan2(estimated_flux[1], estimated_flux[0])
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, s], [-s, c]])
        # Rotate before whitening: covariance anisotropy rotates with the frame.
        local = rotation @ np.asarray(innovation)
        local_covariance = rotation @ covariance @ rotation.T
        white = np.linalg.solve(np.linalg.cholesky(local_covariance), local)
        self.values.append(white)
        if len(self.values) == self.window:
            values = np.asarray(self.values)
            if self.mode == "coherence":
                self.statistic = float(np.sum(values, axis=0) @ np.sum(values, axis=0) / self.window)
            else:
                self.statistic = float(np.mean(np.sum(values**2, axis=1)))
            if self.statistic > self.threshold:
                self.remaining = self.hold_steps
        active = self.remaining > 0
        self.remaining = max(0, self.remaining - 1)
        self.active_steps += int(active)
        return self.high if active else self.low

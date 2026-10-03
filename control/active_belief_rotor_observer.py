"""Grid posterior for rotor angle modulo a measured slot harmonic.

Only current increments, commanded voltages and elapsed time enter this API.
True angle/speed live exclusively in the experiment's metrics layer.
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np

from models.rotor_slot_saliency import SlotSaliency


class ActiveBeliefRotorObserver:
    def __init__(
        self,
        saliency: SlotSaliency,
        *,
        noise_std_a: float,
        n_angle: int = 48,
        speed_hypotheses: tuple[float, ...] = (-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0),
        gain_hypotheses: tuple[float, ...] = (0.8, 1.0, 1.2),
        max_lag: int = 8,
    ) -> None:
        if noise_std_a <= 0 or n_angle < 8 or not speed_hypotheses or not gain_hypotheses:
            raise ValueError("invalid observer grid or noise")
        self.saliency = saliency
        self.noise_std_a = float(noise_std_a)
        self.angles = np.arange(n_angle, dtype=float) * saliency.period_rad / n_angle
        self.speeds = np.asarray(speed_hypotheses, dtype=float)
        self.gains = np.asarray(gain_hypotheses, dtype=float)
        self.weights = np.full((n_angle, len(self.speeds), len(self.gains)), 1.0 / (n_angle * len(self.speeds) * len(self.gains)))
        self.history: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=max_lag + 1)

    @property
    def period_rad(self) -> float:
        return self.saliency.period_rad

    def _transport(self, values: np.ndarray, dt_s: float, *, backward: bool = False) -> np.ndarray:
        out = np.empty_like(values)
        bin_width = self.period_rad / len(self.angles)
        for j, speed in enumerate(self.speeds):
            shift = speed * dt_s / bin_width
            whole = math.floor(shift)
            frac = shift - whole
            if backward:
                out[:, j, :] = (1.0 - frac) * np.roll(values[:, j, :], -whole, axis=0) + frac * np.roll(values[:, j, :], -whole - 1, axis=0)
            else:
                out[:, j, :] = (1.0 - frac) * np.roll(values[:, j, :], whole, axis=0) + frac * np.roll(values[:, j, :], whole + 1, axis=0)
        return out

    def predict_response(self, voltage: tuple[float, float], dwell_s: float) -> np.ndarray:
        va, vb = voltage
        phase = self.saliency.harmonic * self.angles
        c, s = np.cos(phase), np.sin(phase)
        scale = dwell_s * self.gains[None, None, :] / self.saliency.differential_inductance_h
        anis_a = c[:, None, None] * va + s[:, None, None] * vb
        anis_b = s[:, None, None] * va - c[:, None, None] * vb
        shape = self.weights.shape
        a = np.broadcast_to(scale * (va + self.saliency.strength * anis_a), shape)
        b = np.broadcast_to(scale * (vb + self.saliency.strength * anis_b), shape)
        return np.stack((a, b), axis=-1)

    def update_observations(self, observations: list[tuple[tuple[float, float], tuple[float, float], float]], dt_s: float) -> None:
        if not observations:
            raise ValueError("at least one measurement is required")
        prior = self._transport(self.weights, dt_s)
        log_like = np.zeros_like(prior)
        for measurement, voltage, dwell_s in observations:
            pred = self.predict_response(voltage, dwell_s)
            innovation = pred - np.asarray(measurement, dtype=float)
            log_like -= 0.5 * np.sum(innovation * innovation, axis=-1) / self.noise_std_a**2
        like = np.exp(log_like - np.max(log_like))
        post = prior * like
        norm = float(np.sum(post))
        if norm <= 0.0 or not math.isfinite(norm):
            raise FloatingPointError("posterior normalization failed")
        self.weights = post / norm
        self.history.append((self.weights.copy(), like.copy(), dt_s))

    def update(self, measurement: tuple[float, float], voltage: tuple[float, float], dwell_s: float, dt_s: float) -> None:
        self.update_observations([(measurement, voltage, dwell_s)], dt_s)

    def angle_marginal(self, lag: int = 0) -> np.ndarray:
        if lag < 0 or lag >= len(self.history) and lag != 0:
            raise ValueError("lag outside stored history")
        if lag == 0:
            return np.sum(self.weights, axis=(1, 2))
        items = list(self.history)
        alpha = items[-lag - 1][0]
        beta = np.ones_like(alpha)
        for _filtered, likelihood, dt_s in reversed(items[-lag:]):
            beta = self._transport(beta * likelihood, dt_s, backward=True)
            beta /= max(float(np.max(beta)), 1e-300)
        smooth = alpha * beta
        smooth /= np.sum(smooth)
        return np.sum(smooth, axis=(1, 2))

    def map_angle(self, lag: int = 0) -> float:
        return float(self.angles[int(np.argmax(self.angle_marginal(lag)))])

    def credible_set(self, mass: float, lag: int = 0) -> tuple[np.ndarray, float]:
        if not 0.0 < mass < 1.0:
            raise ValueError("mass must be in (0, 1)")
        marginal = self.angle_marginal(lag)
        order = np.argsort(-marginal, kind="stable")
        count = int(np.searchsorted(np.cumsum(marginal[order]), mass, side="left")) + 1
        chosen = order[:count]
        return chosen, float(count * self.period_rad / len(self.angles))

    def entropy(self) -> float:
        w = self.weights
        return float(-np.sum(w * np.log(np.maximum(w, 1e-300))))


__all__ = ["ActiveBeliefRotorObserver"]

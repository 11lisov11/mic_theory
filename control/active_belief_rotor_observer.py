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
        noise_std_axes_a: tuple[float, float] | None = None,
        n_angle: int = 48,
        speed_hypotheses: tuple[float, ...] = (-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0),
        gain_hypotheses: tuple[float, ...] = (0.8, 1.0, 1.2),
        max_lag: int = 8,
        angle_diffusion: float = 0.0,
        speed_diffusion: float = 0.0,
        gain_diffusion: float = 0.0,
        noise_beta_ratio_candidates: tuple[float, ...] | None = None,
        likelihood_temperature: float = 1.0,
    ) -> None:
        if noise_std_a <= 0 or n_angle < 8 or not speed_hypotheses or not gain_hypotheses:
            raise ValueError("invalid observer grid or noise")
        if noise_std_axes_a is not None and (len(noise_std_axes_a) != 2 or any(x <= 0 for x in noise_std_axes_a)):
            raise ValueError("axis noise standard deviations must be positive")
        if any(not 0.0 <= value <= 0.5 for value in (angle_diffusion, speed_diffusion, gain_diffusion)):
            raise ValueError("diffusion probabilities must be in [0, 0.5]")
        if tuple(sorted(set(speed_hypotheses))) != speed_hypotheses:
            raise ValueError("speed hypotheses must be sorted and unique")
        if tuple(sorted(set(gain_hypotheses))) != gain_hypotheses:
            raise ValueError("gain hypotheses must be sorted and unique")
        if noise_beta_ratio_candidates is not None and (
            not noise_beta_ratio_candidates
            or any(not math.isfinite(value) or value <= 0.0 for value in noise_beta_ratio_candidates)
        ):
            raise ValueError("noise-ratio candidates must be finite and positive")
        if not math.isfinite(likelihood_temperature) or likelihood_temperature < 1.0:
            raise ValueError("likelihood temperature must be finite and at least one")
        self.saliency = saliency
        self.noise_std_a = float(noise_std_a)
        self.noise_std_axes_a = np.asarray(noise_std_axes_a or (noise_std_a, noise_std_a), dtype=float)
        self.noise_beta_ratio_candidates = noise_beta_ratio_candidates
        self.likelihood_temperature = float(likelihood_temperature)
        self.angles = np.arange(n_angle, dtype=float) * saliency.period_rad / n_angle
        self.speeds = np.asarray(speed_hypotheses, dtype=float)
        self.gains = np.asarray(gain_hypotheses, dtype=float)
        self.angle_diffusion = float(angle_diffusion)
        self.speed_diffusion = float(speed_diffusion)
        self.gain_diffusion = float(gain_diffusion)
        # Relative evidence of the tempered likelihood; Gaussian constants are omitted.
        self.last_predictive_log_evidence: float | None = None
        self.weights = np.full((n_angle, len(self.speeds), len(self.gains)), 1.0 / (n_angle * len(self.speeds) * len(self.gains)))
        self.history: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=max_lag + 1)

    @property
    def period_rad(self) -> float:
        return self.saliency.period_rad

    @staticmethod
    def _neighbor_diffusion(values: np.ndarray, axis: int, probability: float) -> np.ndarray:
        if probability == 0.0 or values.shape[axis] < 2:
            return values
        shifted = np.moveaxis(values, axis, 0)
        out = shifted.copy()
        transfer = probability * (shifted[:-1] - shifted[1:])
        out[:-1] -= transfer
        out[1:] += transfer
        return np.moveaxis(out, 0, axis)

    def _angle_diffuse(self, values: np.ndarray) -> np.ndarray:
        p = self.angle_diffusion
        return ((1.0 - 2.0 * p) * values + p * np.roll(values, 1, axis=0)
                + p * np.roll(values, -1, axis=0)) if p else values

    def _transport(self, values: np.ndarray, dt_s: float, *, backward: bool = False) -> np.ndarray:
        if backward:
            values = self._angle_diffuse(values)
            values = self._neighbor_diffusion(values, 2, self.gain_diffusion)
            values = self._neighbor_diffusion(values, 1, self.speed_diffusion)
        bin_width = self.period_rad / len(self.angles)
        shifts = self.speeds * dt_s / bin_width
        whole = np.floor(shifts).astype(int)
        frac = (shifts - whole)[None, :, None]
        angle_index = np.arange(len(self.angles))[:, None]
        speed_index = np.arange(len(self.speeds))[None, :]
        source0 = (angle_index + whole if backward else angle_index - whole) % len(self.angles)
        source1 = (source0 + 1 if backward else source0 - 1) % len(self.angles)
        out = (1.0 - frac) * values[source0, speed_index, :] + frac * values[source1, speed_index, :]
        if backward:
            return out
        out = self._neighbor_diffusion(out, 1, self.speed_diffusion)
        out = self._neighbor_diffusion(out, 2, self.gain_diffusion)
        return self._angle_diffuse(out)

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
        if self.noise_beta_ratio_candidates is None:
            log_like = np.zeros_like(prior)
            for measurement, voltage, dwell_s in observations:
                pred = self.predict_response(voltage, dwell_s)
                innovation = pred - np.asarray(measurement, dtype=float)
                log_like -= 0.5 * np.sum((innovation / self.noise_std_axes_a) ** 2, axis=-1)
        else:
            component_logs = []
            for ratio in self.noise_beta_ratio_candidates:
                sigma = np.asarray((self.noise_std_a, self.noise_std_a * ratio))
                component = np.full_like(prior, -len(observations) * math.log(ratio))
                for measurement, voltage, dwell_s in observations:
                    pred = self.predict_response(voltage, dwell_s)
                    innovation = pred - np.asarray(measurement, dtype=float)
                    component -= 0.5 * np.sum((innovation / sigma) ** 2, axis=-1)
                component_logs.append(component)
            components = np.stack(component_logs)
            pivot = np.max(components, axis=0)
            log_like = pivot + np.log(np.sum(np.exp(components - pivot), axis=0))
        if self.likelihood_temperature != 1.0:
            log_like /= self.likelihood_temperature
        log_like_max = float(np.max(log_like))
        like = np.exp(log_like - log_like_max)
        post = prior * like
        norm = float(np.sum(post))
        if norm <= 0.0 or not math.isfinite(norm):
            raise FloatingPointError("posterior normalization failed")
        self.last_predictive_log_evidence = log_like_max + math.log(norm)
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

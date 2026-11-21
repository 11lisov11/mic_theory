"""Rotor belief with a trial-persistent, unknown current-noise mode.

The noise mode is inferred only from commanded voltages and current increments.
It is not allowed to change during a trial; no true rotor state enters this API.
"""

from __future__ import annotations

import math

import numpy as np

from control.active_belief_rotor_observer import ActiveBeliefRotorObserver
from models.rotor_slot_saliency import SlotSaliency


class PersistentNoiseRotorObserver(ActiveBeliefRotorObserver):
    def __init__(self, saliency: SlotSaliency, *, noise_std_a: float,
                 noise_ratios: tuple[float, ...], **kwargs: object) -> None:
        if (not noise_ratios or len(set(noise_ratios)) != len(noise_ratios)
                or any(not math.isfinite(ratio) or ratio <= 0.0 for ratio in noise_ratios)):
            raise ValueError("noise ratios must be distinct, finite and positive")
        if kwargs.get("noise_beta_ratio_candidates") is not None or kwargs.get("noise_std_axes_a") is not None:
            raise ValueError("persistent noise modes replace per-block mixtures and fixed axis noise")
        super().__init__(saliency, noise_std_a=noise_std_a, **kwargs)
        self.noise_ratios = tuple(noise_ratios)
        self._joint_weights = np.repeat(self.weights[None, ...] / len(noise_ratios),
                                        len(noise_ratios), axis=0)

    def update_observations(self, observations: list[tuple[tuple[float, float], tuple[float, float], float]],
                            dt_s: float) -> None:
        if not observations:
            raise ValueError("at least one measurement is required")
        prior = np.stack([self._transport(mode, dt_s) for mode in self._joint_weights])
        squared_innovations = []
        for measurement, voltage, dwell_s in observations:
            innovation = self.predict_response(voltage, dwell_s) - np.asarray(measurement, dtype=float)
            squared_innovations.append(innovation * innovation)
        log_likes = []
        for ratio in self.noise_ratios:
            sigma_squared = np.asarray((self.noise_std_a ** 2, (self.noise_std_a * ratio) ** 2))
            log_like = np.full_like(self.weights, -len(observations) * math.log(ratio))
            for squared in squared_innovations:
                log_like -= 0.5 * np.sum(squared / sigma_squared, axis=-1)
            log_likes.append(log_like / self.likelihood_temperature)
        log_likes_array = np.stack(log_likes)
        likelihood = np.exp(log_likes_array - np.max(log_likes_array))
        posterior = prior * likelihood
        norm = float(np.sum(posterior))
        if norm <= 0.0 or not math.isfinite(norm):
            raise FloatingPointError("joint posterior normalization failed")
        self._joint_weights = posterior / norm
        self.weights = np.sum(self._joint_weights, axis=0)
        self.history.append((self.weights.copy(), np.ones_like(self.weights), dt_s))

    def noise_mode_probabilities(self) -> np.ndarray:
        return np.sum(self._joint_weights, axis=(1, 2, 3))

    def angle_marginal(self, lag: int = 0) -> np.ndarray:
        if lag != 0:
            raise ValueError("fixed-lag smoothing is not implemented for the persistent noise model")
        return super().angle_marginal(0)

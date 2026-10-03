"""Rotor belief with a heavy-tailed beta-current innovation model.

Only measured current increments, commanded voltages and elapsed time enter
the update. This is a host research likelihood, not a hardware noise model.
"""

from __future__ import annotations

import math

import numpy as np

from control.active_belief_rotor_observer import ActiveBeliefRotorObserver
from models.rotor_slot_saliency import SlotSaliency


class StudentTNoiseRotorObserver(ActiveBeliefRotorObserver):
    def __init__(self, saliency: SlotSaliency, *, noise_std_a: float,
                 beta_df: float = 3.0, beta_scale_ratio: float = 2.0,
                 **kwargs: object) -> None:
        if not math.isfinite(beta_df) or beta_df <= 2.0:
            raise ValueError("Student-t degrees of freedom must be finite and greater than two")
        if not math.isfinite(beta_scale_ratio) or beta_scale_ratio <= 0.0:
            raise ValueError("beta scale ratio must be finite and positive")
        if kwargs.get("noise_beta_ratio_candidates") is not None or kwargs.get("noise_std_axes_a") is not None:
            raise ValueError("Student-t beta noise replaces Gaussian axis noise and mixtures")
        super().__init__(saliency, noise_std_a=noise_std_a, **kwargs)
        self.beta_df = float(beta_df)
        self.beta_scale_ratio = float(beta_scale_ratio)

    def update_observations(self, observations: list[tuple[tuple[float, float], tuple[float, float], float]],
                            dt_s: float) -> None:
        if not observations:
            raise ValueError("at least one measurement is required")
        prior = self._transport(self.weights, dt_s)
        log_like = np.zeros_like(prior)
        beta_scale = self.noise_std_a * self.beta_scale_ratio
        for measurement, voltage, dwell_s in observations:
            innovation = self.predict_response(voltage, dwell_s) - np.asarray(measurement, dtype=float)
            log_like -= 0.5 * (innovation[..., 0] / self.noise_std_a) ** 2
            log_like -= 0.5 * (self.beta_df + 1.0) * np.log1p(
                (innovation[..., 1] / beta_scale) ** 2 / self.beta_df)
        log_like /= self.likelihood_temperature
        like = np.exp(log_like - np.max(log_like))
        post = prior * like
        norm = float(np.sum(post))
        if norm <= 0.0 or not math.isfinite(norm):
            raise FloatingPointError("Student-t posterior normalization failed")
        self.weights = post / norm
        self.history.append((self.weights.copy(), like.copy(), dt_s))

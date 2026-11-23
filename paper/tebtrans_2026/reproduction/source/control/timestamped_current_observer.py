"""Fixed-lag, chronological current correction with causal prediction replay."""

from collections import OrderedDict
import math

import numpy as np

from control.current_only_induction_ekf import CurrentOnlyInductionEKF


class TimestampedCurrentObserver:
    def __init__(self, observer: CurrentOnlyInductionEKF, *, history_steps=16,
                 robust_nis_limit=None, persistent_samples=None):
        if history_steps < 1:
            raise ValueError("history must be positive")
        if robust_nis_limit is not None and (not math.isfinite(robust_nis_limit) or robust_nis_limit <= 0):
            raise ValueError("invalid robust NIS threshold")
        if persistent_samples is not None and (not isinstance(persistent_samples, int) or persistent_samples < 2):
            raise ValueError("persistence count must be an integer >= 2")
        self.observer = observer
        self.history_steps = history_steps
        self.robust_nis_limit = robust_nis_limit
        self.persistent_samples = persistent_samples
        self.high_nis_run = 0
        self.index = 0
        self.last_sample = -1
        self.priors = OrderedDict({0: (observer.x.copy(), observer.p.copy())})
        self.inputs = OrderedDict()
        self.downweighted = 0
        self.raw_nis = 0.0
        self.replay_steps = 0

    def correct(self, current, sample_index):
        if (not isinstance(sample_index, int) or sample_index < 0
                or sample_index <= self.last_sample or sample_index > self.index):
            raise ValueError("samples must be chronological, unique and not in the future")
        if sample_index not in self.priors:
            raise ValueError("sample older than retained history")
        y = np.asarray(current, dtype=float)
        if y.shape != (2,) or not np.all(np.isfinite(y)):
            raise ValueError("invalid measured current")
        obs = self.observer
        obs.x, obs.p = (value.copy() for value in self.priors[sample_index])
        innovation = y - obs.x[:2]
        covariance = obs.p[:2, :2] + obs.r
        self.raw_nis = float(innovation @ np.linalg.solve(covariance, innovation))
        nominal_r = obs.r.copy()
        high_nis = self.robust_nis_limit is not None and self.raw_nis > self.robust_nis_limit
        if sample_index != self.last_sample+1:
            self.high_nis_run = 0
        self.high_nis_run = self.high_nis_run+1 if high_nis else 0
        persistent = self.persistent_samples is not None and self.high_nis_run >= self.persistent_samples
        if high_nis and not persistent:
            obs.r = nominal_r * (self.raw_nis / self.robust_nis_limit)
            self.downweighted += 1
        try:
            obs.correct(tuple(y))
        finally:
            obs.r = nominal_r
        current_q = obs.q_spectral.copy()
        try:
            for index in range(sample_index, self.index):
                voltage, dt, q = self.inputs[index]
                # Replay the uncertainty chosen at that interval, never today's Q.
                obs.q_spectral = q.copy()
                obs.predict(voltage, dt)
                self.priors[index+1] = (obs.x.copy(), obs.p.copy())
                self.replay_steps += 1
        finally:
            obs.q_spectral = current_q
        self.last_sample = sample_index

    def predict(self, voltage, dt):
        obs = self.observer
        self.inputs[self.index] = (tuple(voltage), dt, obs.q_spectral.copy())
        obs.predict(voltage, dt)
        self.index += 1
        self.priors[self.index] = (obs.x.copy(), obs.p.copy())
        first_retained = max(0, self.index-self.history_steps)
        while next(iter(self.priors)) < first_retained:
            self.priors.popitem(last=False)
        while self.inputs and next(iter(self.inputs)) < first_retained:
            self.inputs.popitem(last=False)

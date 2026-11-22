"""Phase-synchronous near-zero detector with an explicit standstill hypothesis."""

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct
from scipy.special import logsumexp

from fem.air56.current_tracking import anisotropic_current


BIN_SECONDS = .3
PAIRS_PER_BIN = 9
SPEED_STEP = .025
SPEED_GRID = np.arange(-50, 51)*SPEED_STEP
PHASE_COUNT = 120
BIAS_PRIOR_A = .001
UNCERTAIN = 2


def phase_bins(phasors):
    """Average three repeats of each of the three working-phase pairs."""
    features = anisotropic_current(phasors)
    count = features.shape[-2]//PAIRS_PER_BIN
    if count*PAIRS_PER_BIN != features.shape[-2]:
        raise ValueError("each bin must contain nine phase-locked observations")
    shape = features.shape[:-2]+(count, 3, 3, 4)
    return features.reshape(shape).mean(axis=-3).reshape(features.shape[:-2]+(count, 12))


def ramp_profile(t, start_s, ramp_s, initial_rpm, final_rpm):
    """Return instantaneous speed and exact mechanical displacement of a linear ramp."""
    t = np.asarray(t)
    fraction = np.clip((t-start_s)/ramp_s, 0., 1.)
    delta = final_rpm-initial_rpm
    rpm = initial_rpm+delta*fraction
    turns = (initial_rpm*t+delta*(.5*ramp_s*fraction**2+np.maximum(t-start_s-ramp_s, 0.)))/60
    return rpm, 2*np.pi*turns


@dataclass(frozen=True)
class Detector:
    templates: np.ndarray
    power: np.ndarray
    reference: np.ndarray
    bins: int

    @staticmethod
    def transform(values, reference):
        count = min(values.shape[-2], 12)
        return dct(values-reference, axis=-2, norm="ortho")[..., :count, :].astype(np.float32)

    @classmethod
    def build(cls, calibration, sample_interval, observe, bins):
        times = np.arange(bins*PAIRS_PER_BIN)*sample_interval
        phases = np.arange(PHASE_COUNT)*calibration.period/PHASE_COUNT
        center = phase_bins(observe(calibration, phases, np.zeros_like(phases), times[:PAIRS_PER_BIN], 100., 1.))
        reference = center.mean(axis=0)
        blocks = []
        for grid in np.array_split(SPEED_GRID, 11):
            waveforms = observe(calibration, phases[None, :], grid[:, None], times, 100., 1.)
            blocks.append(cls.transform(phase_bins(waveforms), reference))
        models = np.concatenate(blocks).reshape(len(SPEED_GRID)*PHASE_COUNT, -1, 12)
        return cls(models, np.einsum("ijk,ijk->ij", models, models), reference, bins)

    def estimate(self, observations, noise_a):
        if observations.shape[-2] != self.bins:
            raise ValueError("observations must use the trained window length")
        # Independent white noise averages across the three repeats. A
        # separate unknown offset for each phase feature acts on its mean.
        white = noise_a/np.sqrt(3)
        weight = white**2/(white**2+self.bins*BIAS_PRIOR_A**2)
        template = self.templates.copy()
        template[:, 0] *= weight
        power = weight*self.power[:, 0]+self.power[:, 1:].sum(axis=1)
        template = template.reshape(len(template), -1)
        observed = self.transform(observations, self.reference).reshape(len(observations), -1)
        signs, estimates, movement = [], [], []
        for start in range(0, len(observed), 160):
            batch = observed[start:start+160]
            score = (batch @ template.T-.5*power)/white**2
            score = logsumexp(score.reshape(len(batch), len(SPEED_GRID), PHASE_COUNT), axis=-1)
            zero = len(SPEED_GRID)//2
            prior = np.log(zero)
            classes = np.column_stack((logsumexp(score[:, :zero], axis=1)-prior,
                                       score[:, zero], logsumexp(score[:, zero+1:], axis=1)-prior))
            probability = np.exp(classes-logsumexp(classes, axis=1, keepdims=True))
            direction = probability.argmax(axis=1)-1
            signs.append(np.where(probability.max(axis=1)>=.95, direction, UNCERTAIN))
            estimates.append(SPEED_GRID[score.argmax(axis=1)])
            movement.append(1-probability[:, 1])
        return np.concatenate(signs), np.concatenate(estimates), np.concatenate(movement)

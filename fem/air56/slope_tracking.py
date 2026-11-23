"""Flux-increment residual and quasi-direct rotor angle tracking for two-axis PWM probes."""

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import t as student_t


SYMMETRIC = np.array([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
ANGLE_GRID = np.arange(900)*np.pi/900
WORK_PHASES = np.arange(6)*np.pi/6


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack((np.stack((c, -s), axis=-1), np.stack((s, c), axis=-1)), axis=-2)


def inductance_terms(theta):
    rotor = rotation(theta)[..., None, :, :]
    rotated = (rotor @ SYMMETRIC @ rotor.swapaxes(-1, -2))[..., None, :, :, :]
    relative = 2*(WORK_PHASES-theta[..., None])
    cos, sin = np.cos(relative)[..., None, None, None], np.sin(relative)[..., None, None, None]
    return np.concatenate((np.broadcast_to(rotated, (*relative.shape, 3, 2, 2)), cos*rotated, sin*rotated), axis=-3)


def slope_parts(observation):
    return observation[..., :2], observation[..., 2:4], 1/observation[..., 4]


@dataclass(frozen=True)
class FluxIncrement:
    coefficients: np.ndarray

    @classmethod
    def fit(cls, angles, data):
        current, voltage, weight = slope_parts(data)
        # Equation (30) in rotor coordinates, with sine and cosine terms
        # so the saturation amplitude and phase can differ by matrix entry.
        design = np.einsum("apmij,apbj->apbim", inductance_terms(angles), current)
        w = np.sqrt(weight)[..., None]
        coefficients = np.linalg.lstsq((design*w[..., None]).reshape(-1, 9),
                                       (voltage*w).ravel(), rcond=None)[0]
        return cls(coefficients)

    def inductance(self, angles):
        return np.einsum("...pmij,m->...pij", inductance_terms(angles), self.coefficients)

    def acquire(self, data):
        """Global initialization of (27) modulo the twofold rotor symmetry."""
        current, voltage, weight = slope_parts(data)
        matrices = self.inductance(ANGLE_GRID)
        power = matrices.swapaxes(-1, -2) @ matrices
        linear = np.einsum("...pbi,...pbj,...pb->...pij", voltage, current, weight)
        quadratic = np.einsum("...pbi,...pbj,...pb->...pij", current, current, weight)
        score = linear.reshape(-1, 24) @ matrices.reshape(len(matrices), 24).T
        score -= .5*(quadratic.reshape(-1, 24) @ power.reshape(len(matrices), 24).T)
        return ANGLE_GRID[score.argmax(axis=1)].reshape(data.shape[:-3])

    def update(self, estimate, data):
        """Equations (28)–(29), relinearized once per observation block."""
        current, voltage, weight = slope_parts(data)
        residual = voltage-np.einsum("...pij,...pbj->...pbi", self.inductance(estimate), current)
        rotor = rotation(estimate)
        magnetics = np.einsum("m,mij->ij", self.coefficients[:3], SYMMETRIC)
        turn = np.array([[0., -1.], [1., 0.]])
        derivative = rotor @ (turn @ magnetics-magnetics @ turn) @ rotor.swapaxes(-1, -2)
        jacobian = -np.einsum("...ij,...pbj->...pbi", derivative, current)
        gradient = np.einsum("...pbi,...pbi,...pb->...", jacobian, residual, weight)
        curvature = np.einsum("...pbi,...pbi,...pb->...", jacobian, jacobian, weight)
        error = np.zeros_like(estimate)
        for step in np.deg2rad(4)/2.**np.arange(7):
            error -= step*np.sign(gradient+curvature*error)
        return (estimate+error) % np.pi

    def sequence(self, observations):
        estimates = np.empty(observations.shape[:2])
        estimates[:, 0] = self.acquire(observations[:, 0])
        for frame in range(1, observations.shape[1]):
            estimates[:, frame] = self.update(estimates[:, frame-1], observations[:, frame])
        return estimates


def causal_speed(angles, frames, sample_seconds):
    """Fit a causal local angular velocity and gate the sign at 95 percent."""
    unwrapped = np.unwrap(2*angles, axis=-1)/2
    windows = sliding_window_view(unwrapped, frames, axis=-1)
    times = (np.arange(frames)-(frames-1)/2)*sample_seconds
    time_power = times @ times
    centered = windows-windows.mean(axis=-1, keepdims=True)
    slope = centered @ times/time_power
    residual = centered-slope[..., None]*times
    uncertainty = np.sqrt(np.sum(residual**2, axis=-1)/((frames-2)*time_power))
    confident = abs(slope) > student_t.ppf(.975, frames-2)*uncertainty
    sign = np.where(confident, np.sign(slope), 0).astype(int)
    return slope*60/(2*np.pi), sign

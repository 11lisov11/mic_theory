"""Speed and direction from two staggered current phasors per observation."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logsumexp


@dataclass(frozen=True)
class CurrentMap:
    period: float
    admittance: np.ndarray
    offset: float = 0.0

    @classmethod
    def from_impedance(cls, period, impedance, offset=0.0):
        return cls(float(period), np.linalg.inv(impedance), float(offset))

    def sample(self, theta):
        position = (np.asarray(theta)-self.offset) % self.period
        x = position*len(self.admittance)/self.period
        lower = np.floor(x).astype(int)
        fraction = (x-lower)[..., None, None]
        return (1-fraction)*self.admittance[lower % len(self.admittance)]+fraction*self.admittance[(lower+1) % len(self.admittance)]

    def centered(self):
        return self.admittance-self.admittance.mean(axis=0)

    def contrast(self):
        return float(np.linalg.norm(self.centered(), axis=(1, 2)).max())

    def rank(self):
        spectrum = np.linalg.svd(real_features(self.centered()), compute_uv=False)
        return float(spectrum[1]/max(1e-16, spectrum[0]))


def real_features(phasor):
    return np.concatenate((phasor.real, phasor.imag), axis=-1).reshape(*phasor.shape[:-2], 8)


def observations(table, theta0, rpm, times, probe_hz, volts):
    omega = np.asarray(rpm)*2*np.pi/60
    angles = np.asarray(theta0)[..., None]+omega[..., None]*times
    a = table.sample(angles)[..., :, 0]
    b = table.sample(angles+omega[..., None]/probe_hz)[..., :, 1]
    return volts*real_features(np.stack((a, b), axis=-1))


def centered(x):
    return x-x.mean(axis=-2, keepdims=True)


def score_motion(calibration, truth, frequency, volts, sample_noise, seeds, steps=120,
                 scenario_rpms=(0., -30., -10., -3., -1., 1., 3., 10., 30.)):
    """Profile out constant current offsets. Confidence integrates over phase."""
    times = np.arange(steps)*.01
    speeds = np.arange(-36., 36.01, 0.5)
    phase_count = 196 if calibration.period > 1.0 else 48
    phases = np.arange(phase_count)*calibration.period/phase_count
    prediction = observations(calibration, phases[None, :], speeds[:, None], times, frequency, volts)
    prediction = centered(prediction).reshape(len(speeds)*len(phases), -1).astype(np.float32)
    model_power = np.einsum("ij,ij->i", prediction, prediction)
    scenarios = np.asarray(scenario_rpms, dtype=float)
    rows, truths, measurements = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, truth.period, len(scenarios))
        signal = observations(truth, theta, scenarios, times, frequency, volts)
        white = rng.normal(0, sample_noise, signal.shape)
        offsets = rng.normal(0, 5*sample_noise, (len(theta), 1, signal.shape[-1]))
        rows.extend(dict(seed=int(seed), true_rpm=float(rpm), start_deg=float(np.rad2deg(phase)))
                    for rpm, phase in zip(scenarios, theta))
        measurements.append(signal+offsets+white)
        truths.append(signal)
    y = centered(np.concatenate(measurements)).reshape(len(rows), -1).astype(np.float32)
    traces = np.concatenate(truths)
    likelihood = (y @ prediction.T - 0.5*model_power)/sample_noise**2
    score = logsumexp(likelihood.reshape(len(rows), len(speeds), len(phases)), axis=2)
    p = np.exp(score-logsumexp(score, axis=1, keepdims=True))
    classes = np.stack((p[:, speeds < -0.5].sum(axis=1),
                        p[:, abs(speeds) <= 0.5].sum(axis=1), p[:, speeds > 0.5].sum(axis=1)), axis=1)
    sign = np.argmax(classes, axis=1)-1
    confidence = classes.max(axis=1)
    maps = speeds[np.argmax(p, axis=1)]
    zero = len(speeds)//2
    absolute_p = p[:, zero:].copy()
    absolute_p[:, 1:] += p[:, :zero][:, ::-1]
    amplitude_maps = speeds[zero:][np.argmax(absolute_p, axis=1)]
    moving = abs(speeds) > 0.5
    # Compare mean likelihoods with equal prior odds for motion and rest.
    # Counting every grid point with equal mass would give motion a 98% prior.
    moving_score = logsumexp(score[:, moving], axis=1)-np.log(moving.sum())
    rest_score = logsumexp(score[:, ~moving], axis=1)-np.log((~moving).sum())
    moving_p = expit(moving_score-rest_score)
    for i, row in enumerate(rows):
        row.update(est_rpm=float(maps[i]), declared_direction=int(sign[i]) if confidence[i] >= 0.95 else None,
                   confidence=float(confidence[i]), sample_noise_a=float(sample_noise),
                   est_abs_rpm=float(amplitude_maps[i]), motion_probability=float(moving_p[i]),
                   centered_signal_rms_a=float(np.sqrt(np.mean(centered(traces[i])**2))))
    return rows


def summarize_motion(rows):
    result = []
    for speed in sorted({r["true_rpm"] for r in rows}):
        group = [r for r in rows if r["true_rpm"] == speed]
        accepted = [r for r in group if r["declared_direction"] is not None]
        errors = [abs(r["est_rpm"]-speed) for r in group]
        correct = sum(r["declared_direction"] == np.sign(speed) for r in accepted)
        result.append(dict(rpm=speed, trials=len(group), declared=len(accepted),
                           correct=int(correct), incorrect=len(accepted)-int(correct),
                           mae_rpm=float(np.mean(errors)), p95_rpm=float(np.quantile(errors, .95))))
    return result

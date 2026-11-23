"""Disambiguate two half-turn hypotheses using an independent saturating voltage burst."""

import numpy as np
from scipy.stats import chi2


def capture_quality(cost, noise, readings, margin):
    """Residual compatibility and separation from the opposite-polarity fit."""
    dof = readings - 1  # Two parameters, plus one Gaussian load-prior residual.
    normalized = cost / noise**2
    return normalized / dof, bool(normalized <= chi2.ppf(0.999, dof) and margin >= 2 * np.log(1000))


def resolving_pulses():
    directions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    # Consecutive equal voltage segments raise the field before polarity reverses.
    return [(v, 0.001) for v in 115.0 * np.repeat(directions, 2, axis=0)]


def resolve(model, centers, pulses, measured, noise, *, fit_load=False):
    best = []
    for start in centers:
        angle = start
        load = 0.0
        load_sd = 0.05
        trial_best = None
        for _ in range(5):
            state = model.equilibrium(angle)
            eps = 3e-5
            derivative = ((model.equilibrium(angle + eps) - model.equilibrium(angle - eps)) / (2 * eps))[
                :, None
            ]
            if fit_load:
                state[-1] = load
                derivative = np.column_stack((derivative, np.zeros_like(state)))
                derivative[-1, 1] = 1.0
            currents = []
            jac = []
            for u, h in pulses:
                prev = state
                state, _, fac = model.step(prev, u, h, tangent=True)
                derivative, _ = model.differentiate(
                    prev, state, fac, derivative, np.zeros((2, derivative.shape[1])), h
                )
                currents.extend(state[-5:-3])
                jac.extend(derivative[-5:-3])
            residual = np.ravel(measured) - currents
            cost = residual @ residual
            jac = np.asarray(jac)
            if fit_load:
                # An unknown constant signed torque during the short acquisition.
                # The prior is centred on zero for either sign of the applied load.
                residual = np.r_[residual, -noise * load / load_sd]
                jac = np.vstack((jac, [0.0, noise / load_sd]))
                cost = residual @ residual
            if trial_best is None or cost < trial_best[0]:
                if fit_load:
                    _, singular, vt = np.linalg.svd(jac, full_matrices=False)
                    uncertainty = derivative @ (vt.T * (noise / np.maximum(singular, 1e-10)))
                else:
                    uncertainty = derivative[:, 0].copy()
                trial_best = (cost, angle, state.copy(), uncertainty)
            if fit_load:
                shift = np.linalg.lstsq(jac, residual, rcond=None)[0]
                change = np.clip(shift[0], -0.048, 0.048)
                load += np.clip(shift[1], -0.06, 0.06)
            else:
                slope = jac[:, 0]
                change = np.clip(slope @ residual / (slope @ slope + 1e-7), -0.048, 0.048)
            angle += change
            if abs(change) < 0.0002 and (not fit_load or abs(shift[1]) < 0.0001):
                break
        best.append(trial_best)
    best.sort(key=lambda row: row[0])
    return best[0], (best[1][0] - best[0][0]) / noise**2

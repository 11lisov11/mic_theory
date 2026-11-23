"""Disambiguate two half-turn hypotheses using an independent saturating voltage burst."""

import numpy as np


def resolving_pulses():
    directions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    # Consecutive equal voltage segments raise the field before polarity reverses.
    return [(v, 0.001) for v in 115.0 * np.repeat(directions, 2, axis=0)]


def resolve(model, centers, pulses, measured, noise):
    best = []
    for start in centers:
        angle = start
        trial_best = None
        for _ in range(5):
            state = model.equilibrium(angle)
            eps = 3e-5
            derivative = ((model.equilibrium(angle + eps) - model.equilibrium(angle - eps)) / (2 * eps))[
                :, None
            ]
            currents = []
            jac = []
            for u, h in pulses:
                prev = state
                state, _, fac = model.step(prev, u, h, tangent=True)
                derivative, _ = model.differentiate(prev, state, fac, derivative, np.zeros((2, 1)), h)
                currents.extend(state[-5:-3])
                jac.extend(derivative[-5:-3, 0])
            residual = np.ravel(measured) - currents
            cost = residual @ residual
            if trial_best is None or cost < trial_best[0]:
                trial_best = (cost, angle, state.copy(), derivative[:, 0].copy())
            jac = np.asarray(jac)
            change = np.clip(jac @ residual / (jac @ jac + 1e-7), -0.048, 0.048)
            angle += change
            if abs(change) < 0.0002:
                break
        best.append(trial_best)
    best.sort(key=lambda row: row[0])
    return best[0], (best[1][0] - best[0][0]) / noise**2

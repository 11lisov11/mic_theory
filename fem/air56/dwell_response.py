"""Identify the electrical time response from a blind, switched current capture."""

import numpy as np
from scipy.optimize import least_squares


# Independent response calculations, indexed by the first-order BE error coordinate.
# The set spans a single implicit step up to a more resolved field evolution.
RATES = (1, 3, 7)


def weights(inverse_steps, rates=RATES):
    nodes = 1 / np.asarray(rates, dtype=float)
    values = []
    for j, x in enumerate(nodes):
        rest = np.delete(nodes, j)
        values.append(np.prod((inverse_steps - rest) / (x - rest)))
    return np.asarray(values)


def fit_capture(model, initial, load, pulses, readings, noise, propagate, workers=None):
    """Fit an angle, signed disturbance and damping coordinate from winding currents."""
    readings = np.asarray(readings).reshape(-1, 2)

    # Successive linearizations are evaluated on full switched current histories.
    def signatures(angle, disturbance):
        pending = workers.signatures(angle, disturbance, pulses) if workers else {}
        result = {}
        for rate in RATES:
            if rate in pending:
                continue
            state = model.equilibrium(angle)
            state[-1] = disturbance
            trace = []
            for voltage, dt in pulses:
                state, _ = propagate(model, state, voltage, dt, disturbance, rate)
                trace.append(state[-5:-3].copy())
            result[rate] = trace
        result.update((rate, future.result()) for rate, future in pending.items())
        return np.asarray([result[n] for n in RATES]).reshape(len(RATES), -1)

    angle = initial
    disturbance = load
    coordinate = 0.35
    for _ in range(2):
        signature = signatures(angle, disturbance)
        dangle = (signatures(angle + 0.001, disturbance) - signature) / 0.001
        dload = (signatures(angle, disturbance + 0.002) - signature) / 0.002

        def residual(params):
            correction, changed_load, x = params
            predicted = weights(x) @ (signature + correction * dangle + changed_load * dload)
            return np.r_[(predicted - readings.ravel()) / noise, (disturbance + changed_load) / 0.05]

        fit = least_squares(
            residual, [0.0, 0.0, coordinate], bounds=([-0.04, -0.12, 0.0], [0.04, 0.12, 1.0])
        )
        angle += fit.x[0]
        disturbance += fit.x[1]
        coordinate = fit.x[2]
    return angle, disturbance, coordinate, np.sqrt(np.mean(fit.fun[:-1] ** 2)) * noise


def blended_step(
    model, state, factors, voltage, dt, blend, advance, propagate, history=None, slope=None, workers=None
):
    """One independent elapsed-time map. Bounds and bus states are supplied by the inverter."""
    # Propagate covariance along the central branch. Mean state/torque use all three.
    pending = workers.predictions(state, voltage, dt, history is not None) if workers else {}
    currents = [state[-5:-3].copy()]
    base, factors, mean = advance(model, state, factors, voltage, dt, RATES[1], currents)
    result = blend[1] * base
    mean *= blend[1]
    derivative = slope[1] * base if slope is not None else None
    histories = [None, currents, None]
    for j in (0, 2):
        branch = [state[-5:-3].copy()] if history is not None else None
        if pending:
            alternate, torque, _, branch = pending[RATES[j]].result()
        else:
            alternate, torque = propagate(model, state, voltage, dt, state[-1], RATES[j], currents=branch)
        result += blend[j] * alternate
        mean += blend[j] * torque
        if derivative is not None:
            derivative += slope[j] * alternate
        histories[j] = branch
    if history is not None:
        # Preserve the breakpoints of all three piecewise-linear current predictions.
        # 21 common subintervals give 22 samples including zero and one.
        times = np.linspace(0, 1, 22)
        waveform = np.zeros((len(times), 2))
        for weight, branch in zip(blend, histories):
            values = np.asarray(branch)
            local = np.linspace(0, 1, len(values))
            waveform += weight * np.column_stack([np.interp(times, local, v) for v in values.T])
        history.extend(waveform[1:])
    if derivative is not None:
        return result, factors, mean, derivative
    return result, factors, mean

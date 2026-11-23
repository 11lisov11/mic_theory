"""Control the integrated torque of switched voltage plateaus using only observer states."""

import numpy as np

from fem.air56.servo_pwm import segment_tangents


def impulse_gradient(model, state, stages, divisions, interval):
    sens = np.zeros((len(state), 2))
    impulse = 0.0
    gradient = np.zeros(2)
    zero = np.zeros((2, 2))
    for voltage, length, slope in stages:
        h, dh = length / divisions, slope / divisions
        for _ in range(divisions):
            previous = state
            state, torque, jacobian = model.step(previous, voltage, h, tangent=True)
            sens, tangent = model.differentiate(previous, state, jacobian, sens, zero, h, dh)
            impulse += h * torque / interval
            gradient += (h * tangent + torque * dh) / interval
    return impulse, state[-6], gradient, sens[-6]


def preview(model, state, demand, previous_voltage, probe, interval, blend, workers):
    # A periodic endpoint torque may differ from the mean. Penalize *change* of
    # endpoint torque to damp alternating commands without shifting the DC moment.
    endpoint_weight = 0.25
    u = previous_voltage + probe
    best = None
    for iteration in range(3):
        stages = segment_tangents(u, interval)
        pending = workers.impulses(state, [(v, h) for v, h, _ in stages], interval)
        middle, end, dm, de = impulse_gradient(model, state, stages, 3, interval)
        means = [middle] * 3
        endpoints = [end] * 3
        for j, n in ((0, 1), (2, 7)):
            means[j], endpoints[j] = pending[n].result()
        mean = blend @ means
        error = demand - (mean + endpoint_weight * (blend @ endpoints - state[-6]))
        if best is None or abs(error) < best[0]:
            best = (abs(error), u.copy(), mean)
        if abs(error) < 0.00012 or iteration == 2:
            break
        g = dm + endpoint_weight * de
        delta = error * g / (g @ g + 2e-9)
        u = u + delta * min(1.0, 8.0 / max(np.linalg.norm(delta), 1e-12))
        u *= min(1.0, 80.0 / max(np.linalg.norm(u), 1e-12))
    return best[1], best[2]

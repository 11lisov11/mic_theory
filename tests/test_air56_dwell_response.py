"""Independent checks of current-only calibration and the dwell defect prediction."""

import numpy as np
import pytest

from fem.air56.dwell_response import RATES, blended_step, fit_capture, weights
from tools.run_air56_terminal_servo import Machine, advance_observer, advance_plant


def test_calibration_identifies_signed_load_and_response_from_distinct_current_patterns():
    class CalibrationMotor:
        def equilibrium(self, angle):
            return np.array([0.0, 0.0, angle, 0.0, 0.0])

    def propagate(model, old, voltage, dt, load, rate):
        state = old.copy()
        ip = 1.0 / rate
        state[:2] += dt * (
            voltage * np.sin(state[-3])
            + np.roll(voltage, 1) * load
            + (ip + ip * ip / 3) * np.array([1.0, -2.0])
        )
        return state, 0.0

    motor = CalibrationMotor()
    volts = np.array([[1.2, 2.0], [-1.0, 0.5], [0.4, -0.8], [1.1, -2.3]])
    pulses = list(zip(volts, [0.6, 0.7, 0.5, 0.4]))
    state = motor.equilibrium(0.42)
    trace = []
    # A withheld continuous coordinate between calibration points.
    for v, dt in pulses:
        state, _ = propagate(motor, state, v, dt, -0.022, 1 / 0.208)
        trace.extend(state[:2])
    angle, load, coordinate, rms = fit_capture(motor, 0.44, 0.005, pulses, trace, 1e-5, propagate)
    assert angle == pytest.approx(0.42, abs=1e-5)
    assert load == pytest.approx(-0.022, abs=1e-5)
    assert coordinate == pytest.approx(0.208, abs=1e-5)
    assert rms < 1e-6


def test_dwell_current_endpoints_and_response_sensitivity_on_the_field():
    model = Machine(3, 0, False)
    model.trapezoid = True
    state = model.equilibrium(0.51)
    state[-1] = 0.017
    voltage = np.array([7.0, -5.0])
    dt = 0.0015
    coordinate = 0.22

    def solve(q, history=None, sensitivity=False):
        slope = weights(q + 0.5) - weights(q - 0.5) if sensitivity else None
        return blended_step(
            model,
            state,
            np.zeros((len(state), 1)),
            voltage,
            dt,
            weights(q),
            advance_observer,
            advance_plant,
            history,
            slope,
        )

    history = [state[-5:-3].copy()]
    result, _, _, derivative = solve(coordinate, history, True)
    np.testing.assert_allclose(history[-1], result[-5:-3], atol=1e-9)
    eps = 0.001
    finite = (solve(coordinate + eps)[0] - solve(coordinate - eps)[0]) / (2 * eps)
    np.testing.assert_allclose(derivative[-6:], finite[-6:], rtol=0.002, atol=5e-5)
    for j, rate in enumerate(RATES):
        np.testing.assert_allclose(weights(1 / rate), np.eye(3)[j], atol=1e-15)


def test_parallel_physical_and_observer_branches_keep_separate_states():
    from fem.air56.parallel_dwell import DwellWorkers

    model = Machine(3, 0, False)
    model.trapezoid = True
    estimate = model.equilibrium(0.57)
    estimate[-1] = -0.013
    truth = model.equilibrium(0.59)
    voltage = np.array([5.3, 3.1])
    interval = 0.0011
    cov = np.zeros((len(truth), 1))
    blend = weights(0.27)
    standalone = blended_step(
        model, estimate, cov, voltage, interval, blend, advance_observer, advance_plant
    )
    actual, torque = advance_plant(model, truth, voltage, interval, 0.02, 4)
    with DwellWorkers(3, 3, 0, False) as workers:
        future = workers.reference(truth, voltage, interval, 0.02, 4)
        combined = blended_step(
            model, estimate, cov, voltage, interval, blend, advance_observer, advance_plant, workers=workers
        )
        repeated, repeated_torque, angles, currents = future.result()
    np.testing.assert_allclose(repeated[-5:], actual[-5:], atol=3e-6)
    assert repeated_torque == pytest.approx(torque, abs=1e-6)
    np.testing.assert_allclose(combined[0][-6:], standalone[0][-6:], atol=3e-6)
    assert combined[2] == pytest.approx(standalone[2], abs=1e-6)
    assert len(angles) == 4 and len(currents) == 5


def test_refined_impulse_gradient_includes_duration_and_endpoint_penalty():
    from fem.air56.impulse_control import impulse_gradient
    from fem.air56.servo_pwm import segment_tangents

    model = Machine(3, 0, False)
    model.trapezoid = True
    state = model.equilibrium(0.55)
    state[-1] = 0.018
    u = np.array([4.2, -1.6])
    period = 0.004

    def value(v):
        stages = segment_tangents(v, period)
        mean, end, dm, de = impulse_gradient(model, state, stages, 3, period)
        return mean + 0.25 * (end - state[-6]), dm + 0.25 * de

    _, gradient = value(u)
    for axis, du in enumerate(0.001 * np.eye(2)):
        expected = (value(u + du)[0] - value(u - du)[0]) / 0.002
        assert gradient[axis] == pytest.approx(expected, abs=2e-6)

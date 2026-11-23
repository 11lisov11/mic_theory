"""Cross-check the winding/shaft Jacobian and the switched source independently."""

import numpy as np
import pytest
from fem.air56.servo_pwm import ACTIVE, CLARKE, segments
from tools.run_air56_terminal_servo import Machine
from tools.run_air56_free_field import BandField, SkewBand
from tools.score_air56_terminal_servo import score


@pytest.mark.parametrize("skew", [False, True])
def test_shared_winding_step_and_differential(skew):
    motor = Machine(3, 0, skew)
    theta = 0.47
    initial = motor.equilibrium(theta)
    voltage = np.array([14.0, -8.0])
    interval = 0.003
    state, torque, fac = motor.step(initial, voltage, interval, tangent=True)
    reference = SkewBand(theta, 3) if skew else BandField(0, 3)
    reference.update(theta)
    other = reference.advance(reference.equilibrium(), voltage, interval)
    np.testing.assert_allclose(motor.electromagnetic(state), other, atol=2e-7)
    np.testing.assert_allclose(torque, reference.torque(other[:-3]), atol=1e-7)
    perturb = np.zeros((len(initial), 2))
    perturb[-3, 0] = 1.0
    perturb[-1, 1] = 1.0
    inputs = np.array([[0.0, 1.0], [0.0, -0.3]])
    answer, dtau = motor.differentiate(initial, state, fac, perturb, inputs, interval)
    eps = 1e-5
    for k in range(2):
        positive, plus, _ = motor.step(
            initial + eps * perturb[:, k], voltage + eps * inputs[:, k], interval
        )
        negative, minus, _ = motor.step(
            initial - eps * perturb[:, k], voltage - eps * inputs[:, k], interval
        )
        np.testing.assert_allclose((positive[-5:] - negative[-5:]) / (2 * eps), answer[-5:, k], atol=1e-4)
        np.testing.assert_allclose((plus - minus) / (2 * eps), dtau[k], atol=2e-6)


def test_switched_vectors_reproduce_voltage_and_period():
    levels = np.vstack((np.zeros(2), 310.0 * ACTIVE @ CLARKE.T))
    for angle in np.arange(24) * np.pi / 12:
        command = 75 * np.array([np.cos(angle), np.sin(angle)])
        pattern = segments(command, 0.004)
        assert sum(h for _, h in pattern) == pytest.approx(0.004)
        assert all(h > 0 and np.min(np.linalg.norm(levels - v, axis=1)) < 1e-8 for v, h in pattern)
        np.testing.assert_allclose(sum(v * h for v, h in pattern) / 0.004, command, atol=1e-10)


def test_score_requires_motion_of_the_plant():
    from tools.run_air56_terminal_servo import profile

    t = np.arange(0.01, 11.51, 0.01)
    x = np.zeros((len(t), 15))
    x[:, 0] = t
    x[:, 2] = x[:, 3] = [profile(i, 0.3, 4.2)[0] for i in t]
    x[:, 1] = x[:, 2]
    assert score(x)["passes"]
    assert score(x, seconds=12.5) == {"complete": False}
    x[:, 1] = 0.3
    assert not score(x)["passes"]


def test_torque_history_uses_last_converged_field():
    motor = Machine(3, 0, False, True)
    motor.trapezoid = True
    initial = motor.equilibrium(0.47)
    initial[-6] += 0.01
    voltage = np.array([4.0, -3.0])
    dt = 0.003
    state, torque, fac = motor.step(initial, voltage, dt, tangent=True)
    assert torque == pytest.approx((initial[-6] + state[-6]) / 2)
    d = np.zeros((len(initial), 1))
    d[-6] = 1.0
    d[-3] = 0.2
    d[-2] = 0.4
    du = np.array([[0.2], [-0.15]])
    tangent, mt = motor.differentiate(initial, state, fac, d, du, dt)
    eps = 3e-5
    a, ma, _ = motor.step(initial + eps * d[:, 0], voltage + eps * du[:, 0], dt)
    b, mb, _ = motor.step(initial - eps * d[:, 0], voltage - eps * du[:, 0], dt)
    np.testing.assert_allclose((a[-3:] - b[-3:]) / (2 * eps), tangent[-3:, 0], atol=1e-4)
    np.testing.assert_allclose((ma - mb) / (2 * eps), mt[0], atol=5e-6)


def test_pwm_impulse_derivative_includes_switching_times():
    from fem.air56.servo_pwm import segment_tangents

    motor = Machine(3, 0, False, True)
    motor.trapezoid = True
    initial = motor.equilibrium(0.47)
    interval = 0.004

    def simulate(command, tangent):
        state = initial.copy()
        response = np.zeros((len(state), 2))
        impulse = 0.0
        derivative = np.zeros(2)
        for voltage, h, dh in segment_tangents(command, interval):
            previous = state
            state, torque, factors = motor.step(previous, voltage, h, tangent=tangent)
            if tangent:
                response, dt = motor.differentiate(
                    previous, state, factors, response, np.zeros((2, 2)), h, dh
                )
                derivative += (h * dt + torque * dh) / interval
            impulse += h * torque / interval
        return impulse, derivative

    command = np.array([2.1, -0.65])
    _, derivative = simulate(command, True)
    step = 0.001
    for j, offset in enumerate(step * np.eye(2)):
        plus, _ = simulate(command + offset, False)
        minus, _ = simulate(command - offset, False)
        assert derivative[j] == pytest.approx((plus - minus) / (2 * step), abs=1e-6)

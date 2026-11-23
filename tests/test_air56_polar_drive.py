"""Checks of the passive drive dynamics and current-only readout."""

import numpy as np
import pytest
from tools.run_air56_direct_polar_drive import dynamics, LD, LQ, LM, LR, K, PSI, RS, RR, J, DRAG
from tools.run_air56_calibrated_motion import speed, confirmed_labels
from tools.score_air56_polar_burst import table_estimate


def test_coupled_flux_energy_balance():
    # Magnetic energy is i.lambda minus coenergy. Subtract shaft power and
    # copper loss from input and check against derivative at an arbitrary state.
    from scipy.integrate import quad

    x = np.array([0.4, -0.7, 0.13, -0.18, 1.23, -0.35])

    def stored(y):
        d, q, rd, rq, _, w = y
        linear = 0.5 * (
            LD * d * d + LQ * q * q + LM * ((d + rd) ** 2 + (q + rq) ** 2) + LR * (rd**2 + rq**2)
        )
        nonlin = K * (d * np.log(np.cosh(d)) - quad(lambda i: np.log(np.cosh(i)), 0, d)[0])
        return 1.5 * (linear + nonlin) + 0.5 * J * w * w

    va, vb, load = 45.0, -17.0, 0.21
    deriv = dynamics(x, va, vb, load, 1.0, 1.0, DRAG, J)
    delta = 1e-8
    change = (stored(x + delta * deriv) - stored(x - delta * deriv)) / (2 * delta)
    d, q, rd, rq, theta, w = x
    ia, ib = d * np.cos(theta) - q * np.sin(theta), d * np.sin(theta) + q * np.cos(theta)
    net = (
        1.5 * (va * ia + vb * ib - RS * (d * d + q * q) - RR * (rd * rd + rq * rq))
        - load * w
        - DRAG * w * w
    )
    assert change == pytest.approx(net, rel=2e-7)


def test_motion_deadband_and_wrap():
    t = np.arange(60) * 0.4
    a = (3.13 + t * 0.1 * 2 * np.pi / 60) % np.pi
    rpm = speed(a[None], 18)
    np.testing.assert_allclose(rpm, 0.1, atol=1e-12)
    assert np.all(confirmed_labels(rpm, 0.04)[:, 1:] == 1)
    assert not np.any(confirmed_labels(rpm, 0.11))


def test_burst_table_accepts_arbitrary_current_offsets():
    theta = np.arange(36) * np.pi / 18
    phase = np.arange(25) * 2 * np.pi / 25
    raw = np.stack((np.cos(theta[:, None] + phase), np.sin(theta[:, None] - phase)), axis=-1)
    cal = {"angle": theta, "current": raw}
    recovered = table_estimate(raw + np.array([0.45, -0.81]), cal)
    np.testing.assert_allclose(np.angle(np.exp(1j * (recovered - theta))), 0.0, atol=0.003)


def test_series_field_reproduces_single_section_at_zero_skew():
    from tools.run_air56_free_field import BandField, SkewBand

    class FlatBand(SkewBand):
        offsets = np.zeros(3)

    theta = 0.21
    single = BandField(0.0, 3)
    single.update(theta)
    series = FlatBand(theta, 3)
    a = single.equilibrium()
    b = series.equilibrium()
    for voltage in (np.array([82.0, 0.0]), np.array([-82.0, 0.0])):
        a = single.advance(a, voltage, 0.00025)
        b = series.advance(b, voltage, 0.00025)
        np.testing.assert_allclose(a[-2:], b[-2:], atol=2e-7)
        assert single.torque(a[:-3]) == pytest.approx(series.torque(b[:-3]), abs=1e-6)


def test_field_feedback_sensitivity_matches_independent_resolves():
    from tools.run_air56_field_observer import BandField, H, response

    field = BandField(0.0, 3)
    theta = 0.4
    voltage = np.array([30.0, 5.0])
    field.update(theta)
    initial = field.equilibrium()

    def solve(angle, command):
        field.update(angle)
        state = field.advance(initial, command, H)
        return state, field.torque(state[:-3])

    state, _ = solve(theta, voltage)
    tangent, torque_gradient = response(field, state, np.zeros((len(state), 3)))
    angle_step = 1e-5
    plus, _ = solve(theta + angle_step, voltage)
    minus, _ = solve(theta - angle_step, voltage)
    np.testing.assert_allclose(tangent[-2:, 0], (plus[-2:] - minus[-2:]) / (2 * angle_step), atol=1e-7)
    voltage_step = 0.001
    for axis, offset in enumerate(np.eye(2) * voltage_step):
        plus, forward = solve(theta, voltage + offset)
        minus, reverse = solve(theta, voltage - offset)
        np.testing.assert_allclose(
            tangent[-2:, axis + 1], (plus[-2:] - minus[-2:]) / (2 * voltage_step), atol=1e-8
        )
        assert torque_gradient[axis] == pytest.approx((forward - reverse) / (2 * voltage_step), abs=1e-9)

import numpy as np

from fem.air56.slope_tracking import FluxIncrement, causal_speed


def test_flux_increment_recovers_angle_with_rotor_and_saturation_terms():
    model = FluxIncrement(np.array([.03, .05, .002, .001, .002, -.001, -.002, .001, -.003]))
    rng = np.random.default_rng(20)
    angles = np.arange(24)*np.pi/24

    def observations(theta):
        current = rng.normal(0., 1e-2, (len(theta), 6, 2, 2))
        voltage = np.einsum("apij,apbj->apbi", model.inductance(theta), current)
        return np.concatenate((current, voltage, np.ones_like(current[..., :1])), axis=-1)

    fit = FluxIncrement.fit(angles, observations(angles))
    np.testing.assert_allclose(fit.coefficients, model.coefficients, atol=1e-14)
    trial = angles+np.deg2rad(1.)
    measurement = observations(trial)
    np.testing.assert_allclose(fit.acquire(measurement), trial, atol=np.deg2rad(.12))
    updated = fit.update(trial-np.deg2rad(1.5), measurement)
    np.testing.assert_allclose(updated, trial, atol=np.deg2rad(.6))
    np.testing.assert_allclose(fit.inductance(trial), fit.inductance(trial+np.pi), atol=1e-14)


def test_causal_speed_preserves_sign_across_the_half_turn_boundary():
    rpm = np.array([-.1, 0., .1])
    positions = (np.pi+rpm[:, None]*2*np.pi/60*np.arange(-10, 20)*.4) % np.pi
    estimate, sign = causal_speed(positions, 9, .4)
    np.testing.assert_allclose(estimate, np.broadcast_to(rpm[:, None], estimate.shape), atol=1e-12)
    np.testing.assert_array_equal(sign, np.broadcast_to(np.sign(rpm)[:, None], sign.shape))

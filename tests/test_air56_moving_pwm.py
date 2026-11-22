import numpy as np
import pytest

pytest.importorskip("skfem")
pytest.importorskip("numba")

from fem.air56.moving_pwm import MaterialModel, moving_circuit, moving_run
from fem.air56.near_zero_tracking import ramp_profile
from fem.air56.pwm_section import DEAD_TICKS, PWM_PERIOD_S, Projection, circuit, switched_run


def test_moving_integrator_reduces_to_fixed_transient_when_operators_are_equal():
    k = np.tile(np.array([[20., 2.], [2., 23.]]), (12, 1, 1))
    mass = np.array([[.008, .001], [.001, .01]])
    winding = np.array([[1., .2], [.1, .85]])
    linkage = np.array([.3+.03j, .03-.3j])
    local = Projection(k, mass, winding, linkage)
    fixed = circuit([local], [1.], .054)
    model = MaterialModel(np.stack((k, k, k)), mass, winding, np.stack((linkage, linkage, linkage)), 0., .1)
    moving = moving_circuit(model, .054, 0, 24)
    reference = switched_run(*fixed, 1.4, 0, 80, 6, DEAD_TICKS)
    result = moving_run(*moving, .07, 2.7, 1.4, 0, 80, 8, 6)
    np.testing.assert_allclose(result[0], reference[0], atol=1e-9, rtol=0)
    np.testing.assert_allclose(result[1][:, :5], reference[1], atol=1e-10, rtol=0)
    assert np.ptp(result[1][:, 5]) > 0
    assert result[2] > 0
    assert result[3] == 0

    reversal = moving_run(*moving, .07, .5, 1.4, 0, 80, 8, 80, .001, .002, -.5)
    times = (reversal[1][:, 0]+.5)*PWM_PERIOD_S
    _, displacement = ramp_profile(times, .001, .002, .5, -.5)
    np.testing.assert_allclose(reversal[1][:, 5], .07+displacement, atol=1e-12, rtol=0)
    assert reversal[1][-1, 5] < reversal[1][:, 5].max()

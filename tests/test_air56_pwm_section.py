import numpy as np
import pytest

pytest.importorskip("skfem")
pytest.importorskip("numba")

from fem.air56.pwm_section import (
    AB_TO_PHASE, BUS_V, DEAD_TICKS, PHASE_TO_AB, TIMER_TICKS, edges_for_period,
)


def test_pwm_edges_include_dead_time_and_recover_compensated_mean_voltage():
    demand = np.array([19., 13.])
    phase_demand = AB_TO_PHASE @ demand
    signs = np.array([1., -1., -1.])
    compensation = signs*BUS_V*DEAD_TICKS/TIMER_TICKS
    ideal_rise, ideal_fall = edges_for_period(phase_demand+compensation, signs, 0)
    rise, fall = edges_for_period(phase_demand+compensation, signs, DEAD_TICKS)
    np.testing.assert_array_equal(rise-ideal_rise, [DEAD_TICKS, 0, 0])
    np.testing.assert_array_equal(fall-ideal_fall, [0, DEAD_TICKS, DEAD_TICKS])
    assert np.all((0 < rise) & (rise < fall) & (fall < TIMER_TICKS))
    actual = PHASE_TO_AB @ ((fall-rise)*BUS_V/TIMER_TICKS)
    np.testing.assert_allclose(actual, demand, atol=2*BUS_V/TIMER_TICKS)

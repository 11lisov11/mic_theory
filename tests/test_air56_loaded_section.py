from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("skfem")

from fem.air56.loaded_section import loaded_probe, material, nonlinear_fundamental
from fem.air56.sliding_section import make_section, solve
from tools.run_air56_skew_audit import article_profile


def test_differential_reluctivity_matches_constitutive_finite_difference():
    field = np.array([1.2, .93])
    direction = np.array([.4, -.2])

    def force(values):
        nu, _ = material(*values, True)
        return nu*values

    difference = (force(field+1e-6*direction)-force(field-1e-6*direction))/2e-6
    _, (xx, xy, yy) = material(*field, True)
    np.testing.assert_allclose(np.array([[xx, xy], [xy, yy]]) @ direction, difference, rtol=1e-7)


def test_loaded_fundamental_has_cage_torque_and_the_probe_reduces_to_linear_model():
    from fem.air56.coupon import MU0

    geometry, closed = article_profile()
    section = make_section(geometry, .17, 4, replace(closed, bar_width_modulation=.60),
                           stator_opening_depth_m=.0007)
    operating = nonlinear_fundamental(geometry, section, 1.4, 5.)
    assert operating["residual"] < 5e-8
    assert operating["torque_nm"] > .4
    assert operating["max_steel_b_t"] > 1.45
    assert operating["steel_area_above_knee"] > .01
    loaded, residual, reciprocity = loaded_probe(geometry, section, operating, 100., 27.5/3)
    assert residual < 1e-9 and reciprocity < 1e-9

    nu = (1/(MU0*np.where(section.steel, geometry.mu_r, 1.)))[:, None]
    linear, _, _ = loaded_probe(geometry, section, {"tangent": (nu, np.zeros_like(nu), nu)}, 100., 27.5/3)
    np.testing.assert_allclose(linear, solve(geometry, 100., section, 27.5/3)["z"], atol=1e-9, rtol=1e-9)
    assert np.linalg.norm(loaded-linear) > .5


def test_scheduled_probe_reads_adjacent_field_phases_for_each_speed_and_angle():
    from fem.air56.current_tracking import CurrentMap
    from tools.run_air56_loaded_phase_tracking import SAMPLE_INTERVAL_S, ScheduledMap, scheduled_observations

    maps = tuple(CurrentMap(np.pi, np.tile(i*np.eye(2), (4, 1, 1))) for i in range(6))
    response = scheduled_observations(ScheduledMap(maps, maps[0]), np.zeros((1, 4)), np.ones((3, 1)),
                                      np.arange(6)*SAMPLE_INTERVAL_S, 100., 1.)
    assert response.shape == (3, 4, 6, 8)
    np.testing.assert_allclose(response[2, 3, :, 0], [0, 2, 4, 0, 2, 4])
    np.testing.assert_allclose(response[2, 3, :, 5], [1, 3, 5, 1, 3, 5])

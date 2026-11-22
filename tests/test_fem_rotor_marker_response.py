"""Regression tests for the FEM field transfer and measurement-only observer API."""

import json
from pathlib import Path

import numpy as np
import pytest

from control.active_belief_rotor_observer import ActiveBeliefRotorObserver
from control.info_optimal_coded_pwm import information_score
from models.fem_rotor_marker_response import FemRotorMarkerResponse

# The run has optional engineering dependencies. Legacy controller tests stay
# runnable from the root requirements alone.
pytest.importorskip("skfem")
from fem.air56.coupon import Coupon, contrast, make_problem, normalize_field, solve_coupon
from fem.air56.radial_mode import solve_mode
from fem.air56.stator import Geometry


def test_imposed_unit_induction():
    coupon = Coupon(mu_r=1.0, sigma_20=0.5)
    problem = make_problem(coupon, refinement=1)
    solution = solve_coupon(problem, 2500, cfg=coupon)
    assert np.max(abs(solution["cell_b_t"]-np.eye(3)[:2, None, :])) < 1e-6
    assert solution["residual"] < 1e-10


def test_contact_reciprocity_energy_and_closed_path():
    coupon = Coupon()
    problem = make_problem(coupon, refinement=1)
    control, opened, closed = [solve_coupon(problem, 2500, contacts=indices, cfg=coupon)
                               for indices in ((), (0,), (0, 1))]
    z = contrast(problem, control, closed)
    np.testing.assert_allclose(z, closed["z"]-control["z"], rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(z, z.T, rtol=1e-7)
    np.testing.assert_allclose(closed["z"].real.diagonal()/2,
                               closed["loss_w"]+closed["air_loss_w"], rtol=1e-8)
    assert closed["contact_loss_w"][0] > opened["contact_loss_w"][0] > 0
    assert np.linalg.norm(z) > np.linalg.norm(contrast(problem, control, opened))


@pytest.fixture
def fem_data(tmp_path):
    root = Path(__file__).resolve().parents[1]
    geometry = Geometry.from_config(json.loads((root/"research/air56_interlaminar_marker/geometry.json").read_text()))
    coupon = Coupon()
    problem = make_problem(coupon, refinement=1)
    maps = []
    z_stators = []
    for hz in (500, 2500):
        control = solve_coupon(problem, hz, cfg=coupon)
        closed = solve_coupon(problem, hz, (0, 1), cfg=coupon)
        norm = normalize_field(problem, control)
        matrix = norm.T @ contrast(problem, control, closed) @ norm
        stator = solve_mode(geometry, hz, sample_depths_m=(0.0004,))
        transfer = stator["transfer"]
        maps.append(np.einsum("nai,ab,nbj->nij", transfer, matrix, transfer))
        z_stators.append(stator["z"])
    path = tmp_path/"channel.npz"
    np.savez(path, theta_rad=np.arange(360)*2*np.pi/360, frequencies_hz=[500, 2500],
             delta_z_ohm=maps, mesh_error_ohm=np.abs(maps)*0.08, stator_z_ohm=z_stators, pole_pairs=1,
             profile="AIR56_PHYSICAL_CANDIDATE")
    return path


def test_fem_channel_is_periodic_and_interpolates(fem_data):
    response = FemRotorMarkerResponse.load(fem_data, 1000, 0.01, stator_resistance_ohm=5.5)
    control = FemRotorMarkerResponse.load(fem_data, 1000, 0.01, stator_resistance_ohm=5.5, marker_present=False)
    theta = np.array([-0.033, 0.0, 0.42, np.pi])
    np.testing.assert_allclose(response.impedance(theta), response.impedance(theta+np.pi), atol=1e-14)
    azimuth = response.impedance(theta) - response.impedance(0)
    assert np.max(abs(azimuth)) > 1e-9
    np.testing.assert_allclose(control.current_increment(1.0, 1.0, -1.0, 0.001), (0.1, -0.1))
    np.testing.assert_allclose(response.impedance(theta) @
                               response.phasor_current(theta, [1.0, 0.0])[..., None],
                               np.broadcast_to([[1], [0]], (4, 2, 1)), atol=1e-14)
    with pytest.raises(ValueError, match="outside"):
        FemRotorMarkerResponse.load(fem_data, 3000, 0.01)


def test_legacy_observer_uses_fem_prediction_matrix(fem_data):
    response = FemRotorMarkerResponse.load(fem_data, 500, 0.01, stator_resistance_ohm=5.5)
    observer = ActiveBeliefRotorObserver(response, noise_std_a=0.003)
    voltage = (10.0, -20.0)
    dwell = 10e-6
    predictions = observer.predict_response(voltage, dwell)
    nominal_gain = int(np.flatnonzero(observer.gains == 1.0)[0])
    np.testing.assert_allclose(predictions[:, 0, nominal_gain],
                               [response.current_increment(theta, *voltage, dwell) for theta in observer.angles])
    observer.update(tuple(predictions[3, 0, nominal_gain]), voltage, dwell, 1e-4)
    assert float(observer.weights.sum()) == pytest.approx(1.0)
    assert information_score(observer, voltage, dwell, "fisher") > 0

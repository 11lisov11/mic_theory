from pathlib import Path
import json

import numpy as np
import pytest

pytest.importorskip("skfem")

from fem.air56.current_tracking import CurrentMap, anisotropic_current, real_features, score_motion
from fem.air56.sliding_section import SlotDesign, make_section, solve
from fem.air56.stator import Geometry, solve_section


@pytest.fixture
def geometry():
    cfg = Path(__file__).resolve().parents[1]/"research/air56_interlaminar_marker/geometry.json"
    return Geometry.from_config(json.loads(cfg.read_text(encoding="utf8")))


def test_closed_mesh_energy_and_exact_stock_pitch_repeat(geometry):
    g = geometry
    theta = .037
    mesh = make_section(g, theta, 3)
    p = mesh.mesh.p
    t = mesh.mesh.t
    ab, ac = (p[:, t[i]]-p[:, t[0]] for i in (1, 2))
    basis_area = np.sum(np.abs(ab[0]*ac[1]-ab[1]*ac[0]))/2
    exterior = p[:, mesh.stator_boundary]
    nxt = np.roll(exterior, -1, axis=1)
    outer_area = np.sum(np.abs(exterior[0]*nxt[1]-exterior[1]*nxt[0]))/2
    assert basis_area == pytest.approx(outer_area, rel=1e-10)
    response = solve(g, 1000, mesh, 5.5, transfer=True)
    assert response["residual"] < 1e-8
    assert response["energy"] < 1e-8
    assert response["reciprocity"] < 1e-8
    repeat = solve(g, 1000, make_section(g, theta+2*np.pi/g.rotor_bars, 3), 5.5)
    np.testing.assert_allclose(response["z"], repeat["z"], rtol=1e-9, atol=1e-9)
    assert np.linalg.norm(response["transfer"]) > 0


def test_openings_change_port_and_keep_opposed_symmetry(geometry):
    g = geometry
    design = SlotDesign(modulation=.4)
    theta = .041
    test = lambda angle, rotor: solve(g, 1000, make_section(g, angle, 3, rotor), 5.5)["z"]
    reference = test(theta, SlotDesign())
    altered = test(theta, design)
    assert np.max(abs(reference-altered)) > .001
    np.testing.assert_allclose(altered, test(theta+np.pi, design), rtol=1e-9, atol=1e-9)


def test_legacy_section_without_cage_has_solution(geometry):
    result = solve_section(geometry, 1000, ntheta=168, slotted=False)
    assert np.linalg.norm(result["z"]) > 0


def test_current_tracker_orders_signed_speeds_and_abstains_without_signal():
    period = 2*np.pi/28
    phase = np.arange(40)*2*np.pi/40
    wave = np.zeros((40, 2, 2), dtype=complex)
    wave[:, 0, 0] = .05+.01*np.exp(1j*phase)
    wave[:, 1, 1] = .05+.01*np.exp(-1j*phase)
    test = CurrentMap(period, wave)
    flat = CurrentMap(period, np.broadcast_to(wave.mean(axis=0), wave.shape))
    rows = score_motion(test, test, 1000, 1.0, .00002, [8])
    assert all(row["declared_direction"] == np.sign(row["true_rpm"]) for row in rows)
    assert max(abs(row["est_rpm"]-row["true_rpm"]) for row in rows) <= .5
    rows = score_motion(flat, flat, 1000, 1.0, .00002, [8])
    assert all(row["declared_direction"] is None for row in rows)
    np.testing.assert_allclose([row["motion_probability"] for row in rows], .5, atol=1e-12)


def test_closed_eighteen_bar_cage_supports_a_scalar_port(geometry):
    from dataclasses import replace

    g = replace(geometry, rotor_bars=18)
    theta = .031
    design = SlotDesign(opening_fraction=0.)
    z = solve(g, 500, make_section(g, theta, 3, design), 5.5)["z"]
    np.testing.assert_allclose(z, np.trace(z)/2*np.eye(2), atol=1e-9)


def test_even_current_can_recover_speed_magnitude_without_direction():
    period = 2*np.pi/18
    angle = np.arange(40)*2*np.pi/40
    response = CurrentMap(period, (.05+.01*np.cos(angle))[:, None, None]*np.eye(2))
    rows = score_motion(response, response, 500, 1.0, .00005, [21],
                        scenario_rpms=(-9.8, 9.8))
    for row in rows:
        assert row["motion_probability"] >= .95
        assert row["declared_direction"] is None
        assert abs(row["est_abs_rpm"]-abs(row["true_rpm"])) <= .5


def test_modulated_eighteen_slots_keep_half_turn_but_lose_tooth_pitch_symmetry():
    from tools.run_air56_skew_audit import article_profile

    geometry, closed = article_profile()
    theta = .061
    design = SlotDesign(opening_fraction=.13, opening_depth_m=closed.opening_depth_m, modulation=.40)

    def impedance(angle):
        section = make_section(geometry, angle, 4, design, stator_opening_depth_m=.0007)
        return solve(geometry, 500, section, 27.5/3)["z"]

    reference = impedance(theta)
    np.testing.assert_allclose(reference, impedance(theta+np.pi), rtol=1e-9, atol=1e-9)
    assert np.linalg.norm(reference-impedance(theta+2*np.pi/18)) > 0.05


def test_one_tooth_axial_average_cancels_order_eighteen_and_retains_order_two():
    from tools.run_air56_slot_comparison import skewed

    theta = np.arange(144)*np.pi/144
    impedance = ((10 + .5*np.cos(2*theta) + .2*np.cos(18*theta))*1j)[:, None, None]*np.eye(2)
    original = CurrentMap.from_impedance(np.pi, impedance)
    averaged = skewed(original, 2*np.pi/18)
    before = np.fft.fft(impedance[:, 0, 0])/len(theta)
    after = np.fft.fft(np.linalg.inv(averaged.admittance)[:, 0, 0])/len(theta)
    assert abs(after[9]) < 1e-12
    assert abs(after[1])/abs(before[1]) == pytest.approx(np.sinc(1/9), abs=.001)


def test_anisotropy_channel_keeps_normalized_difference_and_symmetric_cross_current():
    background = np.array([[3+4j, 5+6j], [-5-6j, 3+4j]])
    anisotropy = np.array([[1+2j, 3+4j], [3+4j, -1-2j]])
    expected = np.sqrt(2)*np.array([1., 3., 2., 4.])
    np.testing.assert_allclose(anisotropic_current(real_features(background)), np.zeros(4))
    np.testing.assert_allclose(anisotropic_current(real_features(background+anisotropy)), expected)

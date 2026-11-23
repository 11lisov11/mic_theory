import ctypes as ct
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from fem.air56.servo_pwm import segments
from tools.run_air56_terminal_servo import Machine
from tools.score_air56_stock_release import beta_filter


def test_plain_cage_has_no_magnet_and_repeats_one_slot_pitch():
    m = Machine(3, 0, False, stock=True)
    assert not np.any(m.slices[0].pm)
    a = m.equilibrium(0.127, [0.30, 0.0])
    b = m.equilibrium(0.127 + 2 * np.pi / 18, [0.30, 0.0])
    assert abs(a[-6] - b[-6]) < 1e-8


def test_stock_sector_id_needs_half_turn_symmetry_breaking_in_the_stator():
    from tools.screen_air56_stock_tolerances import screen

    rotor_only, both = screen(2, 1, 96041, 83.0)
    assert rotor_only["max_pair_ua_per_v"] > 1.0
    assert rotor_only["max_antipodal_ua_per_v"] < 1e-6
    assert both["min_pair_ua_per_v"] > 0.01


def test_quadratic_filter_has_unit_displacement_gain_and_no_future_samples():
    w = beta_filter()
    assert np.array_equal(w, np.tril(w))
    t = np.arange(1, len(w) + 1)
    np.testing.assert_allclose(w @ (t**2), t**2)


@pytest.mark.parametrize("level", [3, 5])
def test_twenty_degree_alias_survives_switched_excitation_and_load(level):
    records = []
    for angle in np.deg2rad([7.32, 27.32]):
        plant = Machine(level, 0, False, stock=True)
        plant.trapezoid = True
        state = plant.equilibrium(angle, [0.3, 0.0])
        state[-1] = 0.02
        records.append([])
        for command in ([0, 0], [8, 12], [-12, 8], [4, -9], [-6, 0], [0, 0]):
            for voltage, dwell in segments(command, 0.001):
                for _ in range(2):
                    state, _, _ = plant.step(state, voltage, dwell / 2, load=0.02)
                    records[-1].append([*state[-5:-3], state[-3] - angle])
    np.testing.assert_allclose(records[0], records[1], rtol=0, atol=1e-9)


def test_short_pwm_dwell_retains_the_winding_circuit_increment():
    plant = Machine(3, 0, False, stock=True)
    x = plant.equilibrium(0.22, [0.30, 0.0])
    voltage = np.array([103.333333, -178.978583])
    normal = plant.step(x, voltage, 2.5e-4, load=0.02)[0]
    inc = plant.step(x, voltage, 2.5e-4, load=0.02, incremental=True)[0]
    np.testing.assert_allclose(inc[-5:], normal[-5:], atol=1e-8, rtol=1e-8)
    slow = plant.step(x, voltage, 4e-11, load=0.02, incremental=True)[0]
    fast = plant.step(x, voltage, 2e-11, load=0.02, incremental=True)[0]
    np.testing.assert_allclose(slow[-5:-3] - x[-5:-3], 2 * (fast[-5:-3] - x[-5:-3]), rtol=2e-4, atol=1e-12)


def test_portable_c_reader_matches_batch_rule_and_actuation_timing(tmp_path):
    compiler = shutil.which("gcc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    header = Path(__file__).resolve().parents[1] / "research" / "air56_rotor_fingerprinting"
    wrapper = tmp_path / "verify.c"
    wrapper.write_text(
        '#include "stock_rollback.h"\n'
        "void trace(const double *i,double *v,double sd,double floor){\n"
        "air56_release_reader r={0}; for(int k=0;k<28;k++) "
        "v[k]=air56_release_sample(&r,i[k],sd,floor);}",
        encoding="ascii",
    )
    library = tmp_path / ("verify.dll" if sys.platform == "win32" else "verify.so")
    subprocess.run(
        [compiler, "-shared", "-fPIC", "-O2", "-I", str(header), str(wrapper), "-o", str(library), "-lm"],
        check=True,
    )
    fn = ct.CDLL(str(library)).trace
    vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    fn.argtypes = [vector, vector, ct.c_double, ct.c_double]
    fn.restype = None
    rng = np.random.default_rng(163)
    weights = beta_filter()
    for noise in (0.0003, 0.003):
        for floor in (0.12, 0.40):
            for slope in (-0.009, 0.0, 0.009):
                for _ in range(20):
                    current = slope * (np.arange(1, 29) / 24) ** 2 + rng.normal(0, noise, 28)
                    value = -62.77561184070174 * (weights @ current[:24])
                    exceed = np.flatnonzero(
                        abs(value)
                        >= np.maximum(
                            floor, 3.5 * 62.77561184070174 * noise * np.linalg.norm(weights, axis=1)
                        )
                    )
                    expected = np.zeros(28)
                    if len(exceed):
                        k = exceed[0]
                        expected[k : min(k + 2, 23)] = -24 * np.sign(value[k])
                    out = np.zeros(28)
                    fn(current, out, noise, floor)
                    np.testing.assert_array_equal(out, expected)

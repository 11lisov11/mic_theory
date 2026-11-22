"""Compare both 24/18 designs under the same modeled cage torque and steel B-H curve."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import brentq

from fem.air56.current_tracking import CurrentMap, anisotropic_current, observations, real_features, score_motion
from fem.air56.loaded_section import STEEL_B_T, STEEL_H_A_M, loaded_probe_series, nonlinear_fundamental
from fem.air56.sliding_section import make_section
from tools.run_air56_18_slot_openings import LONG_WINDOW_RPM, file_sha256
from tools.run_air56_rotor_contrast import BAR_WIDTH_MODULATION, ERROR_LIMIT_RPM, OFFSET_A, metrics
from tools.run_air56_skew_audit import article_profile, write_csv
from tools.run_air56_slot_comparison import skewed
from tools.run_air56_speed_accuracy import result


FREQUENCY_HZ = 100.
SLIP_HZ = 5.
TORQUE_NM = .8
POINTS = 60
SECTOR_POINTS = POINTS//3
PHASES = tuple(range(0, 12, 2))
LEVELS = (5, 8)
NOISE = (.0003, .001, .003)
WINDOWS = (1.2, 3.6)
SAMPLE_INTERVAL_S = .02
SEEDS = range(745100, 745164)


def section_at(g, design, angle, level, gap_phase):
    return make_section(g, angle, level, design, gap_phase=gap_phase, stator_opening_depth_m=.0007)


def tuned_current(g, design):
    sections = [section_at(g, design, theta, LEVELS[0], 0) for theta in np.arange(4)*np.pi/4]
    states = [None]*len(sections)

    def torque_error(current_peak):
        torques = []
        for i, section in enumerate(sections):
            state = nonlinear_fundamental(g, section, current_peak, SLIP_HZ, initial=states[i])
            states[i] = state["state"]
            torques.append(state["torque_nm"])
        return np.mean(torques)-TORQUE_NM

    return float(brentq(torque_error, 1.20, 1.60, xtol=.001))


def rotate_sector(impedances, sector):
    theta = sector*np.pi/3
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_phase = np.roll(impedances, 2*sector, axis=-3)
    return rotation @ rotated_phase @ rotation.T


def phase_average_map(impedances, offset, axial_width=0.):
    maps = [CurrentMap.from_impedance(np.pi, phase_z, offset) for phase_z in impedances.swapaxes(0, 1)]
    if axial_width:
        maps = [skewed(table, axial_width) for table in maps]
    return CurrentMap(np.pi, np.mean([table.admittance for table in maps], axis=0), offset)


def fem_map(g, design, current_peak, level, mesh_index, name):
    offset = mesh_index*np.pi/(2*POINTS)
    angles = np.arange(SECTOR_POINTS)*np.pi/POINTS+offset
    impedances, rows = [], []
    initial, cycle_check = None, None
    for i, angle in enumerate(angles):
        section = section_at(g, design, angle, level, mesh_index)
        fundamental = nonlinear_fundamental(g, section, current_peak, SLIP_HZ, initial=initial)
        initial = fundamental["state"]
        diagnostics = mesh_index and i == 3*SECTOR_POINTS//4
        phase_list = (*range(12), None) if diagnostics else PHASES
        probes = loaded_probe_series(g, section, fundamental, FREQUENCY_HZ, 27.5/3, phase_list)
        tested = probes[:12:2] if diagnostics else probes
        impedances.append(np.array([z for z, _, _ in tested]))
        rows.append(dict(profile=name, level=level, angle_rad=float(angle),
                         probe_residual=max(r for _, r, _ in probes),
                         probe_reciprocity=max(r for _, _, r in probes),
                         **{key: val for key, val in fundamental.items() if key not in ("state", "tangent")}))
        if diagnostics:
            phase_y = anisotropic_current(real_features(np.linalg.inv(np.array([p[0] for p in probes[:12]]))))
            averaged_y = anisotropic_current(real_features(np.linalg.inv(probes[-1][0])))
            cycle_check = dict(angle_deg=float(np.rad2deg(angle)),
                peak_phase_variation_a_per_v=float(np.linalg.norm(phase_y-phase_y.mean(axis=0), axis=1).max()),
                mean_phase_vs_averaged_tangent_a_per_v=float(np.linalg.norm(phase_y.mean(axis=0)-averaged_y)),
                six_vs_twelve_phase_average_a_per_v=float(np.linalg.norm(phase_y[::2].mean(axis=0)-phase_y.mean(axis=0))))
            rotated = section_at(g, design, angle+np.pi/3, level, mesh_index)
            rotated_bias = nonlinear_fundamental(g, rotated, current_peak, SLIP_HZ)
            direct = np.array([z for z, _, _ in loaded_probe_series(
                g, rotated, rotated_bias, FREQUENCY_HZ, 27.5/3, PHASES)])
            discrepancy = np.linalg.norm(np.linalg.inv(direct)-np.linalg.inv(rotate_sector(impedances[-1], 1)))
            cycle_check["sector_rotation_error_a_per_v"] = float(discrepancy)
            if discrepancy > 1e-7:
                raise RuntimeError(f"60-degree rotational symmetry failed for {name}: {discrepancy:g}")
        if (i+1) % 5 == 0:
            print(f"  {name}, mesh {level}: {i+1}/{SECTOR_POINTS} sections", flush=True)
    full = np.concatenate([rotate_sector(np.array(impedances), sector) for sector in range(3)])
    return (full, offset), rows, cycle_check


def score_case(calibration, truth, case, raw, summary, *, sample_interval=SAMPLE_INTERVAL_S,
               observation_model=observations):
    noise, seconds = case["noise_a"], case["window_s"]
    rows = score_motion(calibration, truth, FREQUENCY_HZ, 1., noise, SEEDS,
                        steps=round(seconds/sample_interval), scenario_rpms=LONG_WINDOW_RPM,
                        anisotropic=True, offset_noise=OFFSET_A, offset_prior=OFFSET_A, refine_speed=True,
                        sample_interval_s=sample_interval, observation_model=observation_model)
    raw.extend(dict(**case, **r) for r in rows)
    for speed in (0., 1.2, 2.7):
        group = [r for r in rows if abs(r["true_rpm"]) == speed]
        within = sum(1 for r in group if r["true_rpm"] and r["est_refined_rpm"] is not None
                     and r["declared_direction"] == np.sign(r["true_rpm"])
                     and abs(r["est_refined_rpm"]-r["true_rpm"]) <= ERROR_LIMIT_RPM+1e-12)
        summary.append(dict(**case, **result(group, speed), refined_within_limit=within))


def load_linear_maps(root, profile, design, geometry):
    prior = json.loads((root/"summary.json").read_text(encoding="utf8"))
    path = root/f"{profile}_impedance.npz"
    if (prior["geometry"] != asdict(geometry) or prior["designs"][profile] != asdict(design)
            or prior["frequency_hz"] != FREQUENCY_HZ or prior["files_sha256"][path.name] != file_sha256(path)):
        raise ValueError("expected the verified linear maps for the same geometry and frequency")
    with np.load(path, allow_pickle=False) as data:
        period = float(data["period_rad"])
        offset = float(data["holdout_offset_rad"])
        return (CurrentMap.from_impedance(period, data["calibration_ohm"]),
                CurrentMap.from_impedance(period, data["holdout_ohm"], offset))


def run(out, geometry, designs, linear_root):
    pitch = 2*np.pi/geometry.rotor_bars
    raw, speed_summary, fem_rows, contrast = [], [], [], []
    working_points, phase_checks = {}, {}
    loaded_maps = {}
    for name, design in designs.items():
        current_peak = tuned_current(geometry, design)
        working_points[name] = current_peak
        print(f"FEM {name}: {current_peak:.3f} A peak, {SLIP_HZ:g} Hz slip, target {TORQUE_NM:g} N m", flush=True)
        maps = [fem_map(geometry, design, current_peak, level, mesh_index, name)
                for mesh_index, level in enumerate(LEVELS)]
        (first, a_rows, _), (second, b_rows, phase_check) = maps
        fem_rows.extend(a_rows+b_rows)
        phase_checks[name] = phase_check
        loaded_maps[name] = (first, second)
        np.savez_compressed(out/f"{name}_loaded_impedance.npz", period_rad=np.pi,
                            holdout_offset_rad=second[1], calibration_phase_ohm=first[0],
                            holdout_phase_ohm=second[0], phase_angles_rad=np.pi*np.asarray(PHASES)/12)

    for name, (first, second) in loaded_maps.items():
        linear, _ = load_linear_maps(linear_root, name, designs[name], geometry)
        for skew in (0., 1.):
            a, b = (phase_average_map(*m, axial_width=skew*pitch) for m in (first, second))
            lin = skewed(linear, pitch) if skew else linear
            print(f"Scoring {name}, skew {skew:g} pitch", flush=True)
            contrast.append(dict(profile=name, skew_pitches=skew, **metrics(a, b)))
            case = dict(profile=name, skew_pitches=skew, template="loaded")
            for seconds in WINDOWS:
                for noise in NOISE:
                    score_case(a, b, dict(**case, noise_a=noise, window_s=seconds), raw, speed_summary)
            score_case(lin, b, dict(profile=name, skew_pitches=skew, template="linear",
                                    noise_a=NOISE[0], window_s=WINDOWS[-1]), raw, speed_summary)

    write_csv(out/"trials.csv", raw)
    write_csv(out/"speed_results.csv", speed_summary)
    write_csv(out/"fem_operating_points.csv", fem_rows)
    loaded_points = {}
    for name in designs:
        fine_rows = [r for r in fem_rows if r["profile"] == name and r["level"] == LEVELS[-1]]
        stats = {}
        for field in ("torque_nm", "cage_loss_w", "max_steel_b_t", "steel_area_above_knee"):
            values = [r[field] for r in fine_rows]
            stats[field] = [float(np.min(values)), float(np.mean(values)), float(np.max(values))]
        loaded_points[name] = stats
    return dict(current_peak_a=working_points, working_point_min_mean_max=loaded_points,
                cycle_checks=phase_checks, contrast=contrast, speed_results=speed_summary, trial_count=len(raw),
                fem_sections=len(fem_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--linear-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out/"summary.json"
    if summary_path.exists():
        parser.error("choose a new output directory")

    start = perf_counter()
    geometry, closed = article_profile()
    designs = {"bar_width": replace(closed, bar_width_modulation=BAR_WIDTH_MODULATION),
               "modulated": replace(closed, opening_fraction=.13, modulation=.40)}
    payload = run(args.out, geometry, designs, args.linear_dir)
    files = ("trials.csv", "speed_results.csv", "fem_operating_points.csv", "bar_width_loaded_impedance.npz",
             "modulated_loaded_impedance.npz")
    payload.update(simulation_only=True, geometry=asdict(geometry), designs={n: asdict(d) for n, d in designs.items()},
                   frequency_hz=FREQUENCY_HZ, probe_v=1., target_torque_nm=TORQUE_NM, slip_hz=SLIP_HZ,
                   sample_interval_s=SAMPLE_INTERVAL_S, offset_a=OFFSET_A, pitch_deg=360/geometry.rotor_bars,
                   noise_a=NOISE, window_s=WINDOWS, signed_rpm=LONG_WINDOW_RPM, seeds=[SEEDS.start, SEEDS.stop],
                   mesh_levels=LEVELS, angles_per_mesh=POINTS, direct_angles_per_mesh=SECTOR_POINTS,
                   carrier_phases=len(PHASES), bh_tesla=STEEL_B_T.tolist(), bh_a_per_m=STEEL_H_A_M.tolist(),
                   elapsed_s=perf_counter()-start,
                   files_sha256={name: file_sha256(args.out/name) for name in files})
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()

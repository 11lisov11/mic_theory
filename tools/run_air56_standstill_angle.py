"""Compare rotor-angle estimators on two independent loaded AIR56 PWM FEM meshes."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fem.air56.current_tracking import anisotropic_current, real_features
from fem.air56.moving_pwm import common_model, moving_circuit, moving_run
from fem.air56.pwm_section import LOAD_HZ, PROBE_HZ, PWM_PERIOD_S, demodulate
from tools.run_air56_18_slot_openings import file_sha256
from tools.run_air56_loaded_contrast import NOISE
from tools.run_air56_moving_pwm import (
    AXIAL_MODES, LEVELS, MOTION_BLOCK, PERIODS, PHASE_NODES, RANK, SECTOR_NODES, TARGET_POINTS,
    build_sections, read_maps,
)
from tools.run_air56_skew_audit import article_profile, write_csv


PROFILES = ("bar_width", "modulated")
ACQUISITIONS = (1, 3, 6, 18)
ACQUISITION_S = 1/LOAD_HZ
BIAS_A = .0001
TRIALS = 512
GRID = np.arange(1800)*np.pi/1800


def build_square(args):
    geometry, closed = article_profile()
    design = replace(closed, bar_width_modulation=.60) if args.profile == "bar_width" else replace(
        closed, opening_fraction=.13, modulation=.40)
    source = args.loaded_dir/"summary.json"
    peak = json.loads(source.read_text(encoding="utf8"))["current_peak_a"][args.profile]
    for index, level in enumerate(LEVELS):
        print(f"{args.profile}: building mesh {level}", flush=True)
        sections, offset, _ = build_sections(geometry, design, peak, index)
        angles = (index/2+np.arange(TARGET_POINTS))*np.pi/TARGET_POINTS
        mapped = np.empty((2, len(angles), 6, 2, 2), complex)
        angular_step = np.pi/(3*SECTOR_NODES)
        for first in range(0, len(angles), 2):
            targets = angles[first:first+2]
            low = int(np.floor((targets[0]-offset)/angular_step))-3
            high = int(np.ceil((targets[-1]-offset)/angular_step))+3
            model = common_model(sections, geometry.rotor_radius, RANK, offset, tuple(range(low, high+1)))
            for skew in (0, 1):
                circuit = moving_circuit(model, geometry.stack_length, skew, PHASE_NODES, AXIAL_MODES)
                for i, theta in enumerate(targets, start=first):
                    traces = [moving_run(*circuit, theta, 0., peak, axis, PERIODS, MOTION_BLOCK, 0,
                                         square_probe=True)[0] for axis in (0, 1)]
                    mapped[skew, i] = demodulate(np.array(traces), PERIODS, subtract_background=False)
        np.savez_compressed(args.out/f"mesh_{level}.npz", angle_rad=angles, response_a=mapped)
    payload = dict(profile=args.profile, prior_sha256=file_sha256(source), square_hz=PROBE_HZ,
                   square_peak_v=np.pi/4, levels=LEVELS, pwm_hz=1/PWM_PERIOD_S,
                   current_peak_a=peak, phase_nodes=PHASE_NODES, rank=RANK,
                   files_sha256={p.name: file_sha256(p) for p in args.out.glob("mesh_*.npz")})
    (args.out/"summary.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf8")


def periodic_interpolation(angles, nodes, values):
    flat = values.reshape(len(nodes), -1)
    sampled = [np.interp(angles, nodes, channel, period=np.pi) for channel in flat.T]
    return np.array(sampled).T.reshape(len(angles), *values.shape[1:])


def angle_templates(nodes, response):
    # The twofold rotor harmonic is common to all phases of the working field.
    # The phase means estimate the contribution locked to that field.
    average = response.mean(axis=0)
    node_basis = np.column_stack((np.cos(2*nodes), np.sin(2*nodes)))
    grid_basis = np.column_stack((np.cos(2*GRID), np.sin(2*GRID)))
    rotor = grid_basis @ np.linalg.lstsq(node_basis, response.mean(axis=1), rcond=None)[0]
    yield "rotor_harmonic", rotor[:, None, :]+np.broadcast_to(average.mean(axis=0), average.shape)
    yield "saturation_compensated", rotor[:, None, :]+average
    yield "phase_resolved", periodic_interpolation(GRID, nodes, response)


def estimate_angle(measurements, templates):
    reference = templates.mean(axis=0)
    observed = (measurements-reference).reshape(len(measurements), -1)
    expected = (templates-reference).reshape(len(templates), -1)
    score = observed @ expected.T-.5*np.sum(expected**2, axis=1)
    return GRID[score.argmax(axis=1)]


def angle_error(estimate, truth):
    return np.rad2deg((estimate-truth+np.pi/2) % np.pi-np.pi/2)


def compare(calibration_nodes, calibration, test_nodes, test, case, independent_nodes):
    templates = dict(angle_templates(calibration_nodes, anisotropic_current(real_features(calibration))))
    test_features = anisotropic_current(real_features(test))
    direct = estimate_angle(test_features, templates["phase_resolved"])
    direct_error = abs(angle_error(direct, test_nodes))
    independent_nodes.append(dict(**case, angles=len(test_nodes), p95_deg=float(np.percentile(direct_error, 95)),
                                  max_deg=float(direct_error.max())))
    rng = np.random.default_rng(746203)
    angles = rng.uniform(0., np.pi, TRIALS)
    clean = periodic_interpolation(angles, test_nodes, test_features)
    bias = rng.normal(0, BIAS_A, clean.shape)
    noise = rng.standard_normal((max(ACQUISITIONS), *clean.shape))
    phases = rng.integers(clean.shape[1], size=TRIALS)
    for n in ACQUISITIONS:
        draws = noise[:n].mean(axis=0)
        for amplitude in NOISE:
            measures = clean+bias+amplitude*draws
            for method, grid in templates.items():
                estimates = estimate_angle(measures, grid)
                single = np.empty_like(estimates)
                for phase in range(clean.shape[1]):
                    selected = phases == phase
                    single[selected] = estimate_angle(measures[selected, phase], grid[:, phase])
                for count, result in ((1, single), (clean.shape[1], estimates)):
                    absolute = abs(angle_error(result, angles))
                    yield dict(**case, method=method, field_phases=count, seconds_per_axis=round(n*ACQUISITION_S, 2),
                               noise_a=amplitude, trials=TRIALS, median_deg=float(np.median(absolute)),
                               p95_deg=float(np.percentile(absolute, 95)),
                               within_5_deg=int((absolute <= 5).sum()), within_10_deg=int((absolute <= 10).sum()))


def score(args):
    results, independent_nodes, sources = [], [], {}
    for profile in PROFILES:
        sine_dir = args.sine_dir/profile
        sine_info = json.loads((sine_dir/"summary.json").read_text(encoding="utf8"))
        sources[profile] = dict(sine=file_sha256(sine_dir/"summary.json"))
        if args.square_dir:
            square_dir = args.square_dir/profile
            square_info = json.loads((square_dir/"summary.json").read_text(encoding="utf8"))
            if square_info["prior_sha256"] != sine_info["prior_sha256"]:
                raise ValueError("the two waveforms must use the same loaded operating point")
            sources[profile]["square"] = file_sha256(square_dir/"summary.json")
            square_maps = []
            for mesh in LEVELS:
                path = square_dir/f"mesh_{mesh}.npz"
                if file_sha256(path) != square_info["files_sha256"][path.name]:
                    raise ValueError(f"map checksum mismatch: {path}")
                with np.load(path, allow_pickle=False) as data:
                    square_maps.append((data["angle_rad"], data["response_a"]))
        for skew in (0, 1):
            sine_maps = []
            for mesh in LEVELS:
                phase_map = read_maps(sine_dir, sine_info, mesh, skew)[0.]
                nodes = phase_map.maps[0].offset+np.arange(TARGET_POINTS)*np.pi/TARGET_POINTS
                sine_maps.append((nodes, np.stack([p.admittance for p in phase_map.maps], axis=1)))
            case = dict(profile=profile, skew_pitches=skew)
            results.extend(compare(*sine_maps[0], *sine_maps[1], dict(**case, carrier="sine"), independent_nodes))
            if args.square_dir:
                (nodes, first), (test_nodes, second) = square_maps
                results.extend(compare(nodes, first[skew], test_nodes, second[skew], dict(**case, carrier="square"),
                                       independent_nodes))
    write_csv(args.out/"angle_results.csv", results)
    info = dict(simulation_only=True, sources_sha256=sources, calibration_mesh=LEVELS[0],
                holdout_mesh=LEVELS[1], angle_period_deg=180, trials_per_case=TRIALS,
                phase_noise_bias_a=BIAS_A, acquisition_seconds_per_axis=ACQUISITION_S, cases=len(results),
                no_added_noise_independent_fem_nodes=independent_nodes)
    (args.out/"summary.json").write_text(json.dumps(info, indent=2)+"\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    maps = commands.add_parser("square")
    maps.add_argument("--loaded-dir", type=Path, required=True)
    maps.add_argument("--profile", choices=PROFILES, required=True)
    maps.set_defaults(run=build_square)
    bench = commands.add_parser("score")
    bench.add_argument("--sine-dir", type=Path, required=True)
    bench.add_argument("--square-dir", type=Path)
    bench.set_defaults(run=score)
    for command in (maps, bench):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("choose an empty output directory")
    args.run(args)


if __name__ == "__main__":
    main()

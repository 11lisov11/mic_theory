"""Generate moving-mesh PWM records and benchmark low-speed readout."""

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fem.air56.loaded_section import nonlinear_fundamental
from fem.air56.moving_pwm import common_model, moving_circuit, moving_run
from fem.air56.pwm_section import (
    BUS_V, DEAD_TICKS, LOAD_HZ, PROBE_HZ, PWM_PERIOD_S, TIMER_HZ, demodulate,
    direct_nonlinear_trace, response_snapshots, section_matrices,
)
from tools.run_air56_18_slot_openings import LONG_WINDOW_RPM, file_sha256
from tools.run_air56_loaded_contrast import NOISE, score_case, section_at
from tools.run_air56_loaded_phase_tracking import SAMPLE_INTERVAL_S, scheduled_observations
from tools.run_air56_pwm_comparison import phase_current_map
from tools.run_air56_skew_audit import article_profile, write_csv


LEVELS = (4, 6)
RANK = 100
SECTOR_NODES = 12
TARGET_POINTS = 24
PHASE_NODES = 24
MOTION_BLOCK = 8
AXIAL_MODES = 3
PERIODS = round(.4/PWM_PERIOD_S)
RECORD_CENTER_S = .3
WINDOW_S = 3.6


@dataclass(frozen=True)
class TrajectoryBank:
    by_speed: dict

    @property
    def period(self):
        return self.by_speed[0.].period

    @property
    def admittance(self):
        return self.by_speed[0.].admittance


def moving_observations(table, starts, rpm, times, probe_hz, volts):
    if not isinstance(table, TrajectoryBank):
        return scheduled_observations(table, starts, rpm, times, probe_hz, volts)
    traces = np.empty((len(rpm), len(times), 8))
    for speed in np.unique(rpm):
        chosen = rpm == speed
        traces[chosen] = scheduled_observations(table.by_speed[float(speed)], starts[chosen], rpm[chosen],
                                                times, probe_hz, volts)
    return traces


def build_sections(g, design, peak, mesh_index):
    offset = mesh_index*np.pi/(12*SECTOR_NODES)
    rows, prepared, state = [], [], None
    for index, angle in enumerate(offset+np.arange(SECTOR_NODES)*np.pi/(3*SECTOR_NODES)):
        section = section_at(g, design, angle, LEVELS[mesh_index], mesh_index)
        load = nonlinear_fundamental(g, section, peak, LOAD_HZ, initial=state)
        state = load["state"]
        full, matrices, linkage = section_matrices(g, section, load)
        snapshots = response_snapshots(full, matrices)
        prepared.append((full, matrices, linkage, snapshots))
        rows.append(dict(level=LEVELS[mesh_index], angle_rad=float(angle), torque_nm=load["torque_nm"],
                         nonlinear_residual=float(load["residual"])))
    return prepared, offset, rows


def measured(system, peak, theta, rpm, *, block=MOTION_BLOCK):
    start = theta-rpm*2*np.pi/60*RECORD_CENTER_S
    runs = [moving_run(*system, start, rpm, peak, axis, PERIODS, block, 0) for axis in (-1, 0, 1)]
    y = demodulate(np.array([r[0] for r in runs]), PERIODS)
    return y, max(r[3] for r in runs)


def direct_check(g, design, system, peak, out, start):
    rpm, periods = 2.7, 12
    runs = [moving_run(*system, start, rpm, peak, axis, periods, MOTION_BLOCK, periods)
            for axis in (-1, 0, 1)]
    sample_angles = np.unique(runs[0][1][:, 5])
    full_sections, initial = [], None
    for angle in sample_angles:
        section = section_at(g, design, angle, LEVELS[-1], 1)
        f = nonlinear_fundamental(g, section, peak, LOAD_HZ, initial=initial)
        initial = f["state"]
        full_sections.append(section_matrices(g, section, f)[0])
    first = full_sections[0]
    for full in full_sections:
        np.testing.assert_array_equal(full.free, first.free)
        np.testing.assert_allclose(full.winding, first.winding, rtol=0, atol=1e-12)
        np.testing.assert_allclose(full.integral, first.integral, rtol=0, atol=1e-12)
    direct = [direct_nonlinear_trace(first, r[1][:, :5], g.stack_length, peak, 2,
                                     moving_sections=full_sections) for r in runs]
    simulated = np.array([r[0] for r in runs])
    checked = np.array([r[0] for r in direct])
    difference = (checked[1:]-checked[:1])-(simulated[1:]-simulated[:1])
    np.savez_compressed(out/"direct_moving_check.npz", angles_rad=sample_angles,
                        direct_current_a=checked, reduced_current_a=simulated,
                        switched_events=runs[1][1])
    return dict(remeshed_periods=len(sample_angles), angle_change_deg=float(np.rad2deg(np.ptp(sample_angles))),
                probe_difference_rms_a=float(np.sqrt(np.mean(difference**2))),
                max_newton_residual=max(r[1] for r in direct))


def maps_main(args):
    prior_path = args.prior_dir/"summary.json"
    prior = json.loads(prior_path.read_text(encoding="utf8"))
    g, closed = article_profile()
    design = (replace(closed, bar_width_modulation=.6) if args.profile == "bar_width" else
              replace(closed, opening_fraction=.13, modulation=.4))
    if prior["geometry"] != asdict(g) or prior["designs"][args.profile] != asdict(design):
        raise ValueError("geometry differs from the loaded reference")
    peak = prior["current_peak_a"][args.profile]
    start = perf_counter()
    op, audits = [], {}
    for mesh_index, level in enumerate(LEVELS):
        print(f"{args.profile}: assembling moving-band FEM grid, mesh {level}", flush=True)
        sections, physical_offset, rows = build_sections(g, design, peak, mesh_index)
        op.extend(rows)
        offset = mesh_index*np.pi/(2*TARGET_POINTS)
        angles = offset+np.arange(TARGET_POINTS)*np.pi/TARGET_POINTS
        speeds = LONG_WINDOW_RPM if mesh_index else (0.,)
        angular_step = np.pi/(3*SECTOR_NODES)
        for skew in (0, 1):
            data = np.empty((len(speeds), len(angles), 6, 2, 2), dtype=complex)
            motional, interval_error = 0., None
            for first in range(0, len(angles), 2):
                targets = angles[first:first+2]
                # Include all positions traversed during the preroll and
                # measurement window, then add the five axial offsets.
                start_node = int(np.floor((targets[0]-physical_offset)/angular_step))-1-2*skew
                end_node = int(np.ceil((targets[-1]-physical_offset)/angular_step))+1+2*skew
                model = common_model(sections, g.rotor_radius, RANK, physical_offset,
                                     tuple(range(start_node, end_node+1)))
                system = moving_circuit(model, g.stack_length, skew, PHASE_NODES, AXIAL_MODES)
                for s, speed in enumerate(speeds):
                    for i, theta in enumerate(targets, start=first):
                        data[s, i], voltage = measured(system, peak, theta, speed)
                        motional = max(motional, voltage)
                if mesh_index and first == 4:
                    if not skew:
                        audits["direct"] = direct_check(g, design, system, peak, args.out,
                                                        physical_offset+np.pi/6)
                    ref = data[speeds.index(2.7), first]
                    fine = measured(system, peak, targets[0], 2.7, block=MOTION_BLOCK//2)[0]
                    interval_error = float(np.sqrt(np.mean(abs(ref-fine)**2)))
                if (first+2) % 8 == 0:
                    print(f"  mesh {level}, skew {skew}: {first+2}/{len(angles)} angles over {len(speeds)} speeds", flush=True)
            if mesh_index:
                audits[f"skew_{skew}"] = dict(max_motional_voltage_v=motional,
                                              four_vs_eight_pwm_periods_rms_a_per_v=interval_error)
            np.savez_compressed(args.out/f"mesh_{level}_skew_{skew}.npz", admittance_a_per_v=data,
                                signed_rpm=speeds, offset_rad=offset, physical_mesh_offset_rad=physical_offset)
    write_csv(args.out/"operating_points.csv", op)
    payload = dict(simulation_only=True, profile=args.profile, geometry=asdict(g), prior_sha256=file_sha256(prior_path),
                   levels=LEVELS, rank=RANK, axial_modes=AXIAL_MODES, phase_nodes=PHASE_NODES,
                   angular_fem_nodes=6*SECTOR_NODES, observation_angles=TARGET_POINTS, motion_block=MOTION_BLOCK,
                   angles_per_local_basis=2,
                   bus_v=BUS_V, pwm_hz=1/PWM_PERIOD_S, deadtime_s=DEAD_TICKS/TIMER_HZ,
                   probe_hz=PROBE_HZ, load_hz=LOAD_HZ, current_peak_a=peak, duration_s=PERIODS*PWM_PERIOD_S,
                   record_center_s=RECORD_CENTER_S, elapsed_s=perf_counter()-start, checks=audits,
                   files_sha256={p.name: file_sha256(p) for p in args.out.iterdir() if p.suffix in (".csv", ".npz")})
    (args.out/"summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf8")


def read_maps(directory, info, level, skew):
    path = directory/f"mesh_{level}_skew_{skew}.npz"
    if file_sha256(path) != info["files_sha256"][path.name]:
        raise ValueError(f"map checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        offset = float(data["offset_rad"])
        return {float(rpm): phase_current_map(y, offset) for rpm, y in zip(data["signed_rpm"], data["admittance_a_per_v"])}


def score_main(args):
    start = perf_counter()
    raw, summary, sources = [], [], {}
    conditions = None
    for root in args.maps:
        source = root/"summary.json"
        info = json.loads(source.read_text())
        profile = info["profile"]
        if profile in sources:
            raise ValueError("provide one map for each rotor")
        sources[profile] = file_sha256(source)
        settings = tuple(info[k] for k in ("geometry", "prior_sha256", "rank", "levels", "pwm_hz", "bus_v",
                                           "angular_fem_nodes", "observation_angles", "axial_modes", "motion_block"))
        if conditions is not None and conditions != settings:
            raise ValueError("use identical FEM and inverter settings for the comparison")
        conditions = settings
        for skew in (0, 1):
            print(f"Scoring {profile} at skew {skew}", flush=True)
            calibration = read_maps(root, info, info["levels"][0], skew)[0.]
            holdout = read_maps(root, info, info["levels"][1], skew)
            for tag, truth in (("moving", TrajectoryBank(holdout)), ("fixed", holdout[0.])):
                for noise in NOISE:
                    case = dict(profile=profile, skew_pitches=skew, template=tag, noise_a=noise, window_s=WINDOW_S)
                    score_case(calibration, truth, case, raw, summary, sample_interval=SAMPLE_INTERVAL_S,
                               observation_model=moving_observations)
    write_csv(args.out/"trials.csv", raw)
    write_csv(args.out/"speed_results.csv", summary)
    payload = dict(simulation_only=True, sources_sha256=sources, trial_count=len(raw), speed_results=summary,
                   signed_rpm=LONG_WINDOW_RPM, noise_a=NOISE, elapsed_s=perf_counter()-start,
                   files_sha256={n: file_sha256(args.out/n) for n in ("trials.csv", "speed_results.csv")})
    (args.out/"summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    maps = commands.add_parser("maps")
    maps.add_argument("--profile", choices=("bar_width", "modulated"), required=True)
    maps.add_argument("--prior-dir", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--maps", nargs=2, type=Path, required=True)
    for cmd in (maps, score):
        cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"summary.json").exists():
        parser.error("choose a new output directory")
    (maps_main if args.command == "maps" else score_main)(args)


if __name__ == "__main__":
    main()

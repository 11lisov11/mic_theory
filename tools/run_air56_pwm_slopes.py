"""Compare intraperiod current-slope and phase-map observers for the AIR56."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fem.air56.moving_pwm import common_model, moving_circuit, moving_run
from fem.air56.near_zero_tracking import ramp_profile
from fem.air56.pwm_section import LOAD_HZ, PROBE_HZ, PWM_PERIOD_S, demodulate
from tools.run_air56_18_slot_openings import file_sha256
from tools.run_air56_moving_pwm import (
    AXIAL_MODES, LEVELS, MOTION_BLOCK, PERIODS, PHASE_NODES, RANK, SECTOR_NODES, TARGET_POINTS, build_sections,
)
from tools.run_air56_skew_audit import article_profile


BLOCK_S = .4
PHASES = np.arange(6)*np.pi/6
PROBE_V = 12.
SPEEDS = np.array([-.1, 0., .1])
SLOPE_PERIODS = round(.6/PWM_PERIOD_S/4)*4
DIRECT_ZERO_S = 6.4
DIRECT_RAMP_S = 2.4
DIRECT_BLOCKS = 36


def slope_block(trace, first, last):
    pairs = np.arange(first+(first % 4 > 0)*(4-first % 4), last-1, 2)
    assert len(pairs) and pairs[0] % 4 == 0
    plus, minus = trace[pairs], trace[pairs+1]
    phase = ((pairs+1)*PWM_PERIOD_S*LOAD_HZ*12+.5).astype(int) % 6
    axis = pairs//2 % 2
    weight = np.divide(1., plus[:, 4]+minus[:, 4], out=np.zeros(len(pairs)), where=(plus[:, 4]*minus[:, 4] > 0.))
    delta = plus[:, :4]-minus[:, :4]
    observed = np.empty((len(PHASES), 2, 5))
    for j in range(len(PHASES)):
        for k in range(2):
            use = (phase == j) & (axis == k)
            total = weight[use].sum()
            if not total:
                raise ValueError("no usable PWM slope pairs in an observation phase")
            observed[j, k, :4] = weight[use] @ delta[use]/total
            observed[j, k, 4] = 1/total
    observed[..., 2:4] *= PWM_PERIOD_S
    return observed, float(np.mean(weight > 0.))


def qd_measure(circuit, theta, speed, peak):
    center = (SLOPE_PERIODS*PWM_PERIOD_S+.2)/2
    trace = np.zeros((SLOPE_PERIODS, 5))
    moving_run(*circuit, theta-speed*2*np.pi/60*center, speed, peak, -1, SLOPE_PERIODS, MOTION_BLOCK, 0,
               slope_records=trace, alternating_probe_v=PROBE_V)
    return slope_block(trace, round(.2/PWM_PERIOD_S), SLOPE_PERIODS)


def phase_measure(circuit, theta, speed, peak):
    omega = speed*2*np.pi/60
    traces = [moving_run(*circuit, theta-omega*(.4-.2*axis), speed, peak, axis, PERIODS, MOTION_BLOCK, 0,
                         square_probe=True)[0] for axis in (0, 1)]
    return demodulate(np.array(traces), PERIODS, subtract_background=False)


def build_maps(args):
    geometry, closed = article_profile()
    design = replace(closed, bar_width_modulation=.60)
    source = args.loaded_dir/"summary.json"
    peak = json.loads(source.read_text(encoding="utf8"))["current_peak_a"]["bar_width"]
    index, level = args.mesh, LEVELS[args.mesh]
    sections, offset, load = build_sections(geometry, design, peak, index)
    angles = (index/2+np.arange(TARGET_POINTS))*np.pi/TARGET_POINTS
    speeds = SPEEDS if index else np.array([0.])
    slopes = np.empty((len(speeds), len(angles), 6, 2, 5))
    phasors = np.empty((len(speeds), len(angles), 6, 2, 2), complex)
    angular_step = np.pi/(3*SECTOR_NODES)
    coverage = []
    for first in range(0, len(angles), 2):
        targets = angles[first:first+2]
        low = int(np.floor((targets[0]-offset)/angular_step))-3
        high = int(np.ceil((targets[-1]-offset)/angular_step))+3
        model = common_model(sections, geometry.rotor_radius, RANK, offset, tuple(range(low, high+1)))
        circuit = moving_circuit(model, geometry.stack_length, args.skew, PHASE_NODES, AXIAL_MODES)
        for i, theta in enumerate(targets, start=first):
            for j, speed in enumerate(speeds):
                slopes[j, i], valid = qd_measure(circuit, theta, speed, peak)
                coverage.append(valid)
                phasors[j, i] = phase_measure(circuit, theta, speed, peak)
        if (first+2) % 8 == 0:
            print(f"mesh {level}, skew {args.skew}: {first+2}/{len(angles)} angles", flush=True)
    path = args.out/"observations.npz"
    np.savez_compressed(path, angle_rad=angles, rpm=speeds, slope=slopes, phasor=phasors)
    payload = dict(simulation_only=True, field_mesh_level=level, skew_pitches=args.skew, qd_probe_peak_v=PROBE_V,
                   qd_probe_hz=1/(2*PWM_PERIOD_S), phase_probe_hz=PROBE_HZ, phase_probe_peak_v=np.pi/4,
                   pwm_hz=1/PWM_PERIOD_S, mean_torque_nm=float(np.mean([r["torque_nm"] for r in load])),
                   min_slope_pair_coverage=min(coverage), mean_slope_pair_coverage=float(np.mean(coverage)),
                   prior_sha256=file_sha256(source), map_sha256=file_sha256(path))
    (args.out/"summary.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf8")


def build_direct(args):
    geometry, closed = article_profile()
    source = args.loaded_dir/"summary.json"
    peak = json.loads(source.read_text(encoding="utf8"))["current_peak_a"]["bar_width"]
    sections, offset, _ = build_sections(geometry, replace(closed, bar_width_modulation=.60), peak, 1)
    theta = np.deg2rad(34.)
    start_s = .2+np.arange(DIRECT_BLOCKS)*BLOCK_S
    time = start_s+BLOCK_S/2
    periods = round((start_s[-1]+BLOCK_S)/PWM_PERIOD_S)+1
    ramp_start = DIRECT_ZERO_S-DIRECT_RAMP_S/2
    step = np.pi/(3*SECTOR_NODES)
    node = int(np.floor((theta-offset)/step))
    model = common_model(sections, geometry.rotor_radius, RANK, offset, tuple(range(node-5, node+6)))
    circuit = moving_circuit(model, geometry.stack_length, args.skew, PHASE_NODES, AXIAL_MODES)
    for speed in (-.1, .1):
        trace = np.zeros((periods, 5))
        moving_run(*circuit, theta, speed, peak, -1, periods, MOTION_BLOCK, 0, ramp_start, DIRECT_RAMP_S, -speed,
                   slope_records=trace, alternating_probe_v=PROBE_V)
        currents = [moving_run(*circuit, theta, speed, peak, axis, periods, MOTION_BLOCK, 0,
                               ramp_start, DIRECT_RAMP_S, -speed, square_probe=True)[0] for axis in (0, 1)]
        slopes, phasors, coverage = [], [], []
        for start in start_s:
            record, valid = slope_block(trace, round(start/PWM_PERIOD_S), round((start+BLOCK_S)/PWM_PERIOD_S))
            slopes.append(record)
            coverage.append(valid)
            first = round((start-.2)/PWM_PERIOD_S)
            second = round(start/PWM_PERIOD_S)
            pair = np.stack((currents[0][first:first+PERIODS], currents[1][second:second+PERIODS]))
            phasors.append(demodulate(pair, PERIODS, subtract_background=False))
        velocity, displacement = ramp_profile(time, ramp_start, DIRECT_RAMP_S, speed, -speed)
        name = "negative_to_positive" if speed < 0 else "positive_to_negative"
        np.savez_compressed(args.out/f"{name}.npz", time_s=time, angle_rad=theta+displacement, rpm=velocity,
                            slope=slopes, phasor=phasors, pair_coverage=coverage)
        print(f"Finished continuous reversal {name}, skew {args.skew}", flush=True)
    summary = dict(simulation_only=True, zero_s=DIRECT_ZERO_S, ramp_s=DIRECT_RAMP_S, skew_pitches=args.skew,
                   field_mesh_level=LEVELS[1], files_sha256={p.name: file_sha256(p) for p in args.out.glob("*.npz")})
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loaded-dir", type=Path, required=True)
    parser.add_argument("--skew", type=int, choices=(0, 1), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mesh", type=int, choices=(0, 1))
    mode.add_argument("--direct", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("choose an empty output directory")
    (build_direct if args.direct else build_maps)(args)


if __name__ == "__main__":
    main()

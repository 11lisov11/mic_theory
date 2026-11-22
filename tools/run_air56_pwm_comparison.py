"""Build switched transient FEM maps and score both loaded AIR56 rotors."""

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

from fem.air56.current_tracking import CurrentMap
from fem.air56.loaded_section import nonlinear_fundamental
from fem.air56.pwm_section import (
    BUS_V, DEAD_TICKS, LOAD_HZ, PROBE_HZ, PWM_PERIOD_S, TIMER_HZ, TIMER_TICKS, circuit,
    demodulate, direct_nonlinear_trace, projected_section, rotate_projection, switched_run,
)
from tools.run_air56_18_slot_openings import file_sha256
from tools.run_air56_loaded_contrast import NOISE, score_case, section_at
from tools.run_air56_loaded_phase_tracking import (
    SAMPLE_INTERVAL_S, ScheduledMap, scheduled_observations,
)
from tools.run_air56_skew_audit import article_profile, write_csv


POINTS = 36
SECTOR_POINTS = POINTS//3
LEVELS = (4, 6)
RANK = 40
PERIODS = round(.4/PWM_PERIOD_S)
CAPTURE = 12
WINDOW_S = 3.6


def phase_current_map(admittance, offset):
    maps = tuple(CurrentMap(np.pi, y, offset) for y in admittance.swapaxes(0, 1))
    return ScheduledMap(maps, CurrentMap(np.pi, admittance.mean(axis=1), offset))


def measure(sections, weights, length, peak, capture=0):
    system = circuit(sections, weights, length)
    runs = [switched_run(*system, peak, axis, PERIODS, capture, DEAD_TICKS) for axis in (-1, 0, 1)]
    currents = np.array([run[0] for run in runs])
    return demodulate(currents, PERIODS), runs


def validate_direct(full, runs, length, peak):
    projected = np.array([r[0][:CAPTURE] for r in runs])
    direct, coarse, residuals, fluxes = [], [], [], []
    for run in runs:
        for subdivisions, trace in ((1, coarse), (3, direct)):
            exact, residual, flux = direct_nonlinear_trace(full, run[1], length, peak, subdivisions)
            trace.append(exact)
            residuals.append(residual)
            fluxes.append(flux)
    direct, coarse = np.array(direct), np.array(coarse)
    refine = lambda value: value[1:]-value[:1]
    rms = lambda values: float(np.sqrt(np.mean(values**2)))
    error = rms(refine(direct)-refine(projected))
    time_error = rms(refine(direct)-refine(coarse))
    return dict(periods=CAPTURE, interval_subdivisions=(1, 3), max_relative_newton_residual=max(residuals),
                max_increment_b_steel_t=max(fluxes), direct_probe_rms_a=rms(refine(direct)),
                paired_current_rms_difference_a=error, direct_step_rms_difference_a=time_error), direct


def build_map(g, design, current_peak, level, mesh_index, out):
    offset = mesh_index*np.pi/(2*POINTS)
    base, op, initial, representative, check_state = [], [], None, None, None
    check_index = SECTOR_POINTS//2
    for k, angle in enumerate(np.arange(SECTOR_POINTS)*np.pi/POINTS+offset):
        section = section_at(g, design, angle, level, mesh_index)
        load = nonlinear_fundamental(g, section, current_peak, LOAD_HZ, initial=initial)
        initial = load["state"]
        reduced, full = projected_section(g, section, load, RANK)
        base.append(reduced)
        if k == check_index:
            representative = full
            check_state = initial
        op.append(dict(level=level, angle_rad=float(angle), projection_rank=reduced.winding.shape[0],
                       **{name: value for name, value in load.items() if name not in ("state", "tangent")}))
    projections = [rotate_projection(p, j) for j in range(3) for p in base]
    admittances, stats = [], []
    checks = {}
    for skew in (0, 1):
        print(f"  mesh {level}, skew {skew}: {POINTS} switched runs per probe axis", flush=True)
        columns = []
        for k, p in enumerate(projections):
            if skew:
                sections = [projections[(k+j) % POINTS] for j in (-2, -1, 0, 1, 2)]
                weights = (1/12, 4/12, 2/12, 4/12, 1/12)
            else:
                sections, weights = (p,), (1.,)
            capture = CAPTURE if mesh_index and not skew and k == check_index else 0
            y, runs = measure(sections, weights, g.stack_length, current_peak, capture)
            columns.append(y)
            stats.append(dict(level=level, skew_pitches=skew, angle_deg=float(np.rad2deg(offset+k*np.pi/POINTS)),
                              max_period_current_increment_a=float(max(run[2] for run in runs)),
                              rms_baseline_current_increment_a=runs[0][3],
                              max_deadtime_sign_changes=int(max(run[4] for run in runs))))
            if capture:
                checks["direct"], direct = validate_direct(representative, runs, g.stack_length, current_peak)
                np.savez_compressed(out/"nonlinear_pwm_check.npz", direct_phase_current_a=direct,
                                    projected_phase_current_a=np.array([r[0][:CAPTURE] for r in runs]),
                                    baseline_events=runs[0][1], alpha_probe_events=runs[1][1], beta_probe_events=runs[2][1])
            if mesh_index and skew and k == check_index:
                middle_sections = []
                for increment in (-1.5, -.5, .5, 1.5):
                    angle = offset+(k+increment)*np.pi/POINTS
                    section = section_at(g, design, angle, level, mesh_index)
                    load = nonlinear_fundamental(g, section, current_peak, LOAD_HZ, initial=check_state)
                    middle_sections.append(projected_section(g, section, load, RANK)[0])
                nine = [sections[0]]
                for midpoint, endpoint in zip(middle_sections, sections[1:]):
                    nine.extend((midpoint, endpoint))
                fine, _ = measure(nine, np.array([1, 4, 2, 4, 2, 4, 2, 4, 1])/24,
                                  g.stack_length, current_peak)
                checks["axial_five_vs_nine_rms_a_per_v"] = float(np.sqrt(np.mean(abs(fine-y)**2)))
        admittances.append(columns)
    return np.array(admittances), op, stats, checks


def maps_main(args):
    prior_path = args.prior_dir/"summary.json"
    prior = json.loads(prior_path.read_text(encoding="utf8"))
    g, closed = article_profile()
    designs = {"bar_width": replace(closed, bar_width_modulation=.6),
               "modulated": replace(closed, opening_fraction=.13, modulation=.4)}
    name = args.profile
    design = designs[name]
    if prior["geometry"] != asdict(g) or prior["designs"][name] != asdict(design):
        raise ValueError("PWM model must use the previous loaded geometry")
    peak = prior["current_peak_a"][name]
    started = perf_counter()
    maps, op, stats = [], [], []
    for mesh, level in enumerate(LEVELS):
        print(f"{name}: nonlinear operating field and transient projections on mesh {level}", flush=True)
        data, points, switches, checks = build_map(g, design, peak, level, mesh, args.out)
        maps.append(data)
        op.extend(points)
        stats.extend(switches)
        np.savez_compressed(args.out/f"mesh_{level}_pwm_current.npz", admittance_a_per_v=data,
                            offset_rad=mesh*np.pi/(2*POINTS))
    write_csv(args.out/"operating_points.csv", op)
    write_csv(args.out/"switching_checks.csv", stats)
    files = tuple(path.name for path in args.out.iterdir() if path.suffix in (".npz", ".csv"))
    payload = dict(simulation_only=True, profile=name, design=asdict(design), geometry=asdict(g),
                   prior_sha256=file_sha256(prior_path), phase_offset_rad=np.pi/(2*POINTS),
                   pwm_hz=1/PWM_PERIOD_S, bus_v=BUS_V, deadtime_s=DEAD_TICKS/TIMER_HZ,
                   timer_ticks=TIMER_TICKS, probe_hz=PROBE_HZ, probe_v=1., load_hz=LOAD_HZ,
                   current_peak_a=peak, points_per_half_turn=POINTS, levels=LEVELS, rank=RANK,
                   periods_per_drive=PERIODS, elapsed_s=perf_counter()-started, checks=checks,
                   files_sha256={name: file_sha256(args.out/name) for name in files})
    (args.out/"summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf8")


def score_main(args):
    start = perf_counter()
    raw, summary, sources = [], [], {}
    reference = None
    for path in args.maps:
        source = path/"summary.json"
        info = json.loads(source.read_text(encoding="utf8"))
        profile = info["profile"]
        if profile not in ("bar_width", "modulated") or profile in sources:
            raise ValueError("provide one map for each design")
        conditions = tuple(info[k] for k in ("geometry", "prior_sha256", "pwm_hz", "bus_v", "deadtime_s",
                                             "probe_hz", "load_hz", "points_per_half_turn", "levels", "rank"))
        if reference is not None and conditions != reference:
            raise ValueError("compare maps computed with the same PWM, geometry and numerical settings")
        reference = conditions
        sources[profile] = file_sha256(source)
        maps = []
        for level in info["levels"]:
            file = path/f"mesh_{level}_pwm_current.npz"
            if file_sha256(file) != info["files_sha256"][file.name]:
                raise ValueError(f"bad map checksum: {file}")
            with np.load(file, allow_pickle=False) as data:
                maps.append((data["admittance_a_per_v"], float(data["offset_rad"])))
        for skew in (0, 1):
            print(f"Scoring {profile}, skew {skew} pitch", flush=True)
            a, b = (phase_current_map(admittance[skew], offset) for admittance, offset in maps)
            for noise in NOISE:
                case = dict(profile=profile, skew_pitches=skew, template="pwm", noise_a=noise, window_s=WINDOW_S)
                score_case(a, b, case, raw, summary, sample_interval=SAMPLE_INTERVAL_S,
                           observation_model=scheduled_observations)
    write_csv(args.out/"trials.csv", raw)
    write_csv(args.out/"speed_results.csv", summary)
    payload = dict(simulation_only=True, map_summaries_sha256=sources, window_s=WINDOW_S,
                   speeds_rpm=[-2.7, -1.2, 0., 1.2, 2.7], noise_a=NOISE, trial_count=len(raw),
                   elapsed_s=perf_counter()-start, speed_results=summary,
                   files_sha256={name: file_sha256(args.out/name) for name in ("trials.csv", "speed_results.csv")})
    (args.out/"summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    maps = commands.add_parser("maps")
    maps.add_argument("--profile", choices=("bar_width", "modulated"), required=True)
    maps.add_argument("--prior-dir", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--maps", nargs=2, type=Path, required=True)
    for command in (maps, score):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"summary.json").exists():
        parser.error("choose a new output directory")
    (maps_main if args.command == "maps" else score_main)(args)


if __name__ == "__main__":
    main()

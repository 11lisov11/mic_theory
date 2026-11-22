"""Separate speed-grid error, current noise and FEM map mismatch on the 18-slot rotor."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fem.air56.current_tracking import CurrentMap, score_motion
from tools.run_air56_18_slot_openings import FREQUENCY_HZ, file_sha256
from tools.run_air56_18_surface_bridge import opening_maps
from tools.run_air56_skew_audit import article_profile, write_csv
from tools.run_air56_slot_comparison import skewed


ABS_RPM = (.87, 1.17, 1.2, 1.61, 2.7, 2.73, 3.17)
RPM = (0., *(-v for v in reversed(ABS_RPM)), *ABS_RPM)
SEEDS = range(743100, 743164)
SECONDS = 3.6
VOLTAGE = 1.
CONDITIONS = (
    ("same_map_clean", False, 0., 0.),
    ("same_map_noise", False, .0003, .0001),
    ("holdout_clean", True, 0., 0.),
    ("holdout_noise", True, .0003, .0001),
    ("holdout_high_noise", True, .003, .0001),
    ("holdout_leading_harmonic", True, .0003, .0001),
)


def leading_harmonic(table):
    spectrum = np.fft.fft(table.admittance, axis=0)
    # The half-turn map's first harmonic has mechanical order two.
    spectrum[2:-1] = 0
    return CurrentMap(table.period, np.fft.ifft(spectrum, axis=0), table.offset)


def result(rows, speed):
    moving = [r for r in rows if r["motion_probability"] >= .95]
    correct = [r for r in moving if r["true_rpm"] and r["declared_direction"] == np.sign(r["true_rpm"])]
    wrong = sum(1 for r in moving if r["true_rpm"] and r["declared_direction"] == -np.sign(r["true_rpm"]))
    row = dict(abs_rpm=speed, total=len(rows), moving=len(moving), correct=len(correct), wrong_direction=wrong)
    for label, estimate in (("coarse", "est_rpm"), ("refined", "est_refined_rpm")):
        error = [abs(r[estimate]-r["true_rpm"]) for r in correct]
        row[f"{label}_p95_rpm"] = float(np.quantile(error, .95)) if error else None
        row[f"{label}_mae_rpm"] = float(np.mean(error)) if error else None
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-map", type=Path, required=True)
    parser.add_argument("--openings-map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=[row[0] for row in CONDITIONS])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out/"summary.json"
    if summary_path.exists():
        parser.error("choose a new output directory")

    start = perf_counter()
    geometry, _ = article_profile()
    pitch = 2*np.pi/geometry.rotor_bars
    closed_hash = file_sha256(args.closed_map)
    first, second = opening_maps(args.openings_map, geometry, closed_hash)
    selected = args.conditions or [row[0] for row in CONDITIONS]
    all_rows, summary = [], []
    for skew in (0., 1.):
        calibration, holdout = (skewed(first, pitch), skewed(second, pitch)) if skew else (first, second)
        for name, use_holdout, noise, offset in CONDITIONS:
            if name not in selected:
                continue
            print(f"Scoring {name}, {skew:g} pitch of skew", flush=True)
            truth = holdout if use_holdout else calibration
            template = leading_harmonic(calibration) if name == "holdout_leading_harmonic" else calibration
            rows = score_motion(template, truth, FREQUENCY_HZ, VOLTAGE, max(noise, .0003), SEEDS,
                                steps=round(100*SECONDS), scenario_rpms=RPM, anisotropic=True,
                                offset_noise=offset, offset_prior=.0001, observation_noise=noise, refine_speed=True)
            metadata = dict(condition=name, skew_pitches=skew, observation_noise_a=noise, offset_noise_a=offset)
            all_rows.extend(dict(**metadata, **r) for r in rows)
            groups = [[r for r in rows if abs(r["true_rpm"]) == s] for s in (0., *ABS_RPM)]
            for speed, group in zip((0., *ABS_RPM, "moving"), (*groups, [r for r in rows if r["true_rpm"]])):
                summary.append(dict(**metadata, **result(group, speed)))

    write_csv(args.out/"trials.csv", all_rows)
    write_csv(args.out/"speed_results.csv", summary)
    payload = dict(simulation_only=True, rotor_slots=geometry.rotor_bars, pitch_deg=float(np.rad2deg(pitch)),
                   window_s=SECONDS, probe_v=VOLTAGE, frequency_hz=FREQUENCY_HZ, signed_rpm=RPM,
                   coarse_grid_rpm=.5, fine_grid_rpm=.05, seeds=[SEEDS.start, SEEDS.stop], trial_count=len(all_rows),
                   closed_map_sha256=closed_hash, openings_map_sha256=file_sha256(args.openings_map),
                   elapsed_s=perf_counter()-start, results=summary,
                   files_sha256={name: file_sha256(args.out/name) for name in ("trials.csv", "speed_results.csv")})
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()

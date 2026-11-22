"""Compare closed-slot saliency designs with variable slot openings on the 24/18 rotor."""

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

from fem.air56.current_tracking import anisotropic_current, real_features, score_motion
from tools.run_air56_18_slot_openings import FREQUENCY_HZ, LONG_WINDOW_RPM, VOLTAGE_PEAK
from tools.run_air56_18_slot_openings import file_sha256, harmonic_current, prepared_maps
from tools.run_air56_skew_audit import NOISE, article_profile, write_csv
from tools.run_air56_slot_comparison import quality, skewed
from tools.run_air56_speed_accuracy import result


SEEDS = range(744100, 744164)
POINTS = 90
LEVELS = (7, 10)
RECESS_M = .00015
BAR_WIDTH_MODULATION = .60
WINDOWS = (1.2, 3.6)
OFFSET_A = .0001
ERROR_LIMIT_RPM = .1


def metrics(calibration, holdout):
    signal = anisotropic_current(real_features(holdout.admittance))
    derivative = (np.roll(signal, -1, axis=0)-np.roll(signal, 1, axis=0))/(2*holdout.period/len(signal))
    return dict(**quality(calibration, holdout),
                harmonic2_a_per_v=harmonic_current(holdout, 2),
                min_angular_slope_a_per_v_rad=float(np.linalg.norm(derivative, axis=1).min()))


def compare(out, geometry, designs, frequency_hz):
    pitch = 2*np.pi/geometry.rotor_bars
    sample_interval = max(.01, 2/frequency_hz)
    maps = {}
    for name, design in designs.items():
        print(f"FEM {name}: {POINTS} angles on each of two meshes", flush=True)
        first, second, checks = prepared_maps(geometry, design, np.pi, POINTS, LEVELS,
                                              frequency_hz=frequency_hz)
        maps[name] = (first, second, checks)
        np.savez_compressed(out/f"{name}_impedance.npz", period_rad=first.period,
                            holdout_offset_rad=second.offset,
                            calibration_ohm=np.linalg.inv(first.admittance),
                            holdout_ohm=np.linalg.inv(second.admittance))
        print(f"{name}: angular contrast {second.contrast()*1e6:.1f} uA/V", flush=True)

    all_rows, summary, contrast = [], [], []
    for name, (first, second, _) in maps.items():
        for skew in (0., 1.):
            print(f"Scoring {name}, skew {skew:g} pitch", flush=True)
            calibration, holdout = (skewed(first, pitch), skewed(second, pitch)) if skew else (first, second)
            contrast.append(dict(profile=name, skew_pitches=skew, **metrics(calibration, holdout)))
            for seconds in WINDOWS:
                for noise in NOISE:
                    trials = score_motion(calibration, holdout, frequency_hz, VOLTAGE_PEAK, noise, SEEDS,
                                          steps=round(seconds/sample_interval), scenario_rpms=LONG_WINDOW_RPM,
                                          anisotropic=True, offset_noise=OFFSET_A, offset_prior=OFFSET_A,
                                          refine_speed=True, sample_interval_s=sample_interval)
                    case = dict(profile=name, skew_pitches=skew, window_s=seconds, noise_a=noise)
                    all_rows.extend(dict(**case, **r) for r in trials)
                    for speed in (0., 1.2, 2.7):
                        group = [r for r in trials if abs(r["true_rpm"]) == speed]
                        row = result(group, speed)
                        row["refined_within_limit"] = sum(1 for r in group if speed
                            and r["est_refined_rpm"] is not None
                            and r["declared_direction"] == np.sign(r["true_rpm"])
                            and abs(r["est_refined_rpm"]-r["true_rpm"]) <= ERROR_LIMIT_RPM + 1e-12)
                        summary.append(dict(**case, **row))
    write_csv(out/"trials.csv", all_rows)
    write_csv(out/"speed_results.csv", summary)
    return dict(trial_count=len(all_rows), speed_results=summary, contrast=contrast, sample_interval_s=sample_interval,
                mesh_checks={name: m[2] for name, m in maps.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate", choices=("oval", "bar_width"), default="oval")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out/"summary.json"
    if summary_path.exists():
        parser.error("choose a new output directory")

    start = perf_counter()
    geometry, closed = article_profile()
    candidate, frequency_hz = {
        "oval": (replace(closed, surface_recess_m=RECESS_M), FREQUENCY_HZ),
        "bar_width": (replace(closed, bar_width_modulation=BAR_WIDTH_MODULATION), 100.),
    }[args.candidate]
    designs = {"modulated": replace(closed, opening_fraction=.13, modulation=.40), args.candidate: candidate}
    payload = compare(args.out, geometry, designs, frequency_hz)
    outputs = ("trials.csv", "speed_results.csv", f"{args.candidate}_impedance.npz", "modulated_impedance.npz")
    payload.update(geometry=asdict(geometry), designs={name: asdict(d) for name, d in designs.items()},
                   pitch_deg=360/geometry.rotor_bars, angular_samples=POINTS, mesh_levels=LEVELS,
                   frequency_hz=frequency_hz, probe_v=VOLTAGE_PEAK, signed_rpm=LONG_WINDOW_RPM,
                   noise_a=NOISE, offset_a=OFFSET_A, window_s=WINDOWS, simulation_only=True,
                   within_limit_rpm=ERROR_LIMIT_RPM, seeds=[SEEDS.start, SEEDS.stop],
                   elapsed_s=perf_counter()-start,
                   files_sha256={name: file_sha256(args.out/name) for name in outputs})
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()

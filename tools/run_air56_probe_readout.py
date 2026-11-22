"""Evaluate low-speed stator-current readout with a calibrated anisotropy channel."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fem.air56.current_tracking import score_motion
from tools.run_air56_18_slot_openings import FREQUENCY_HZ, LONG_WINDOW_RPM, article_baseline, file_sha256
from tools.run_air56_18_surface_bridge import opening_maps
from tools.run_air56_skew_audit import article_profile, group_summary, write_csv
from tools.run_air56_slot_comparison import skewed


SEEDS = range(742100, 742196)


@dataclass(frozen=True)
class Condition:
    window_s: float
    probe_v: float
    noise_a: float = .0003
    offset_a: float = .0001
    assumed_offset_a: float = .0001


CONDITIONS = (
    Condition(1.2, 1.), Condition(1.2, 2.), Condition(1.2, 3.),
    Condition(.6, 2.), Condition(.6, 3.),
    Condition(1.2, 2., offset_a=.0003, assumed_offset_a=.0003),
    Condition(1.2, 2., offset_a=.0015, assumed_offset_a=.0015),
    Condition(1.2, 2., offset_a=.0003),
    Condition(1.2, 3., noise_a=.003),
    Condition(1.8, 2.), Condition(1.8, 3.),
    Condition(2.4, 1.), Condition(2.4, 2.), Condition(2.4, 3.),
    Condition(2.4, 2., offset_a=.0003, assumed_offset_a=.0003),
    Condition(3.2, 2.), Condition(3.2, 3.),
    Condition(3.6, 1.), Condition(3.6, 2.),
    Condition(3.6, 2., offset_a=.0003, assumed_offset_a=.0003),
)
CONTROL = Condition(1.2, 3.)


def assess(profile, pair, skew, condition, method):
    c = condition
    using_mean = method == "anisotropic"
    trials = score_motion(*pair, FREQUENCY_HZ, c.probe_v, c.noise_a, SEEDS,
                          steps=round(100*c.window_s), scenario_rpms=LONG_WINDOW_RPM,
                          anisotropic=using_mean, offset_noise=c.offset_a,
                          offset_prior=c.assumed_offset_a if using_mean else None)
    metadata = dict(probe_v=c.probe_v, offset_a=c.offset_a, assumed_offset_a=c.assumed_offset_a,
                    method=method)
    summary = group_summary(trials, profile, skew, c.noise_a, c.window_s)
    for row in summary:
        speed = row["abs_rpm"]
        accepted = [r for r in trials if abs(r["true_rpm"]) == speed and r["motion_probability"] >= .95]
        correct = [r for r in accepted if speed and r["declared_direction"] == np.sign(r["true_rpm"])]
        signed_error = [abs(r["est_rpm"]-r["true_rpm"]) for r in correct]
        row.update(**metadata, moving_correct_direction=len(correct),
                   moving_correct_negative=sum(1 for r in correct if r["true_rpm"]<0),
                   moving_correct_positive=sum(1 for r in correct if r["true_rpm"]>0),
                   moving_wrong_direction=sum(1 for r in accepted if speed
                                              and r["declared_direction"] == -np.sign(r["true_rpm"])),
                   signed_p95_rpm=float(np.quantile(signed_error, .95)) if signed_error else None)
    rows = [dict(profile=profile, skew_pitches=skew, window_s=c.window_s, **metadata, **r) for r in trials]
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-map", type=Path, required=True)
    parser.add_argument("--openings-map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", nargs="+", type=float,
                        choices=sorted({c.window_s for c in CONDITIONS}),
                        help="run only the conditions at these window lengths")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out/"summary.json"
    if summary_path.exists():
        parser.error("choose a new output directory")

    start = perf_counter()
    geometry, _ = article_profile()
    pitch = 2*np.pi/geometry.rotor_bars
    closed_hash = file_sha256(args.closed_map)
    closed = article_baseline(args.closed_map)
    modulated = opening_maps(args.openings_map, geometry, closed_hash)
    windows = args.windows or {c.window_s for c in CONDITIONS}
    conditions = tuple(c for c in CONDITIONS if c.window_s in windows)
    profiles = [("modulated", modulated, conditions)]
    if CONTROL.window_s in windows:
        profiles.append(("closed", closed, (CONTROL,)))
    all_rows, groups = [], []
    for skew in (0., 1.):
        for name, maps, profile_conditions in profiles:
            pair = tuple(skewed(m, pitch) for m in maps) if skew else maps
            print(f"Scoring {name} with {skew:g} pitch of skew", flush=True)
            for condition in profile_conditions:
                for method in ("centered", "anisotropic"):
                    rows, summary = assess(name, pair, skew, condition, method)
                    all_rows.extend(rows)
                    groups.extend(summary)

    write_csv(args.out/"trials.csv", all_rows)
    write_csv(args.out/"speed_results.csv", groups)
    payload = dict(geometry=asdict(geometry), rotor_pitch_deg=float(np.rad2deg(pitch)),
                   frequency_hz=FREQUENCY_HZ, signed_rpm=LONG_WINDOW_RPM, seeds=[SEEDS.start, SEEDS.stop],
                   conditions=[asdict(c) for c in conditions], closed_map_sha256=closed_hash,
                   openings_map_sha256=file_sha256(args.openings_map), simulation_only=True,
                   trial_count=len(all_rows), speed_results=groups, elapsed_s=perf_counter()-start,
                   files_sha256={name: file_sha256(args.out/name) for name in ("trials.csv", "speed_results.csv")})
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()

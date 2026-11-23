"""Locked-parameter blind trials for stationary-frame direct torque control."""

from pathlib import Path
import argparse
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_air56_direct_polar_drive import K, simulate, summarize, write_csv


def run(out, group):
    cases = {
        "light": (0.02, 1.0, 1.0, False, False),
        "medium": (0.4, 1.0, 1.0, False, False),
        "rated": (0.8, 1.0, 1.0, False, False),
        "mismatch": (0.8, 1.02, 0.85, True, False),
        "turns": (0.4, 1.02, 0.85, True, True),
        "noisy1": (0.8, 1.0, 1.0, False, False),
        "noisy03": (0.8, 1.0, 1.0, False, False),
    }
    underlying = group.removesuffix("_skew")
    skew = int(group != underlying)
    rows = []
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(972811)
    for i in range(32):
        theta, base = rng.uniform(-np.pi, np.pi, 2)
        # Exogenous initial position and external command drawn independently.
        load, scale, mag, imperfect, turns = cases[underlying]
        sd = {"noisy1": 0.001, "noisy03": 0.0003}.get(underlying, 0.003)
        x = simulate(
            theta,
            base,
            sd,
            361805 + i,
            scale,
            mag,
            load,
            True,
            False,
            K,
            turns,
            True,
            imperfect,
            4.0,
            skew,
        )
        scores = summarize(x)
        good_angle = scores["max_observer_deg"] < 90 and scores["p95_observer_deg"] < 5
        good_speed = (
            abs(scores["forward_actual_rpm"] - 0.1) < 0.03
            and abs(scores["reverse_actual_rpm"] + 0.1) < 0.03
        )
        good_turns = (
            abs(scores["out_read_turns"] - scores["out_actual_turns"]) < 0.02
            and abs(scores["return_read_turns"] - scores["return_actual_turns"]) < 0.02
            and scores["out_actual_turns"] > 1
            and scores["return_actual_turns"] < -1
        )
        rows.append(
            dict(
                case=group,
                skew=skew,
                trial=i,
                theta0=theta,
                ref=base,
                load_nm=load,
                noise_a=sd,
                accepted=good_angle
                and (good_turns if turns else good_speed)
                and scores["p95_tracking_deg"] < 5,
                **scores,
            )
        )
        if i in (0, 31):
            np.savez_compressed(out / f"dtc_{group}_{i}.npz", trace=x)
    write_csv(out / f"dtc_{group}.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--group", required=True)
    a = p.parse_args()
    run(a.out, a.group)

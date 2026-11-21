"""Paired host study of likelihood tempering for uncertain current-sensor noise."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_info_optimal_coded_pwm_study import Case, exogenous, run_trial
from tools.study_coded_pwm_noise_miscalibration import _digest, _paired_effect


TEMPERATURES = (1.0, 2.0, 4.0, 8.0)


def run_study(*, seed_offset: int, seeds: int, steps: int,
              temperatures: tuple[float, ...] = TEMPERATURES,
              motors: tuple[str, ...] = ("AIR56", "AL31")) -> tuple[dict, list[dict]]:
    if seeds < 1 or steps < 6 or not motors or not temperatures or 1.0 not in temperatures:
        raise ValueError("seeds, steps, motors and temperatures must include the baseline")
    rows: list[dict] = []
    started = perf_counter()
    for motor in motors:
        for actual_ratio in (1.0, 4.0):
            for seed in range(seed_offset, seed_offset + seeds):
                case = Case(motor=motor, scenario="standstill", strength=0.15,
                            nonideal=True, mismatch="combined", seed=seed,
                            steps=steps, measurement_noise_beta_ratio=actual_ratio)
                stimulus = exogenous(case)
                digest = _digest(stimulus)
                for temperature in temperatures:
                    result = run_trial(
                        case, "B2_fixed_probe", stimulus=stimulus,
                        observer_profile="robust", fixed_probe_vector=6,
                        assumed_noise_beta_ratio=1.0,
                        noise_beta_ratio_candidates=(1.0, 4.0),
                        likelihood_temperature=temperature,
                    )
                    rows.append({
                        "motor": motor, "actual_ratio": actual_ratio, "seed": seed,
                        "temperature": temperature, "stimulus_digest": digest,
                        "angle_p95_deg": result["angle_p95_deg"],
                        "hpd95_coverage": result["hpd95_coverage"],
                        "hpd95_width_deg": result["hpd95_width_deg"],
                        "compute_mean_us": result["compute_mean_us"],
                        "gateway_rejections": result["gateway_rejections"],
                    })
    lookup = {(r["motor"], r["actual_ratio"], r["seed"], r["temperature"]): r for r in rows}
    cells: list[dict] = []
    for motor in motors:
        for actual_ratio in (1.0, 4.0):
            by_temperature = {
                temperature: [lookup[(motor, actual_ratio, seed, temperature)]
                              for seed in range(seed_offset, seed_offset + seeds)]
                for temperature in temperatures
            }
            baseline = by_temperature[1.0]
            cells.append({
                "motor": motor, "actual_ratio": actual_ratio, "n_seeds": seeds,
                "temperatures": {
                    str(temperature): {
                        "mean_p95_deg": float(np.mean([r["angle_p95_deg"] for r in items])),
                        "mean_hpd95_coverage": float(np.mean([r["hpd95_coverage"] for r in items])),
                        "mean_hpd95_width_deg": float(np.mean([r["hpd95_width_deg"] for r in items])),
                        "p95_vs_untempered": _paired_effect(
                            [r["angle_p95_deg"] for r in baseline],
                            [r["angle_p95_deg"] for r in items],
                        ),
                    } for temperature, items in by_temperature.items()
                },
            })
    gateway_rejections = sum(int(r["gateway_rejections"]) for r in rows)
    return ({
        "status": "synthetic_host_only", "seed_range": [seed_offset, seed_offset + seeds - 1],
        "steps_per_trial": steps, "trial_count": len(rows), "motors": list(motors),
        "actual_noise_ratios": [1.0, 4.0], "temperature_grid": list(temperatures),
        "noise_mixture_ratios": [1.0, 4.0], "probe_vector": 6,
        "selection_rule": "smallest T with mean coverage >=0.90 in all four cells and mean P95 <=1.20*T1 in all four cells; otherwise reject",
        "gateway_rejections": gateway_rejections, "cells": cells,
        "elapsed_wall_s": perf_counter() - started,
        "absolute_angle_claim": False, "hardware_claim": False,
    }, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-offset", type=int, default=19000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--temperatures", type=float, nargs="+", default=TEMPERATURES)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "info_optimal_coded_pwm" / "noise_temperature_development")
    args = parser.parse_args()
    payload, rows = run_study(seed_offset=args.seed_offset, seeds=args.seeds,
                              steps=args.steps, temperatures=tuple(args.temperatures))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    with (args.out / "trials.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: payload[key] for key in (
        "status", "seed_range", "trial_count", "gateway_rejections", "elapsed_wall_s"
    )}, indent=2))
    if payload["gateway_rejections"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

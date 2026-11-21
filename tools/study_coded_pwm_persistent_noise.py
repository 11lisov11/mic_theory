"""Paired synthetic test of persistent versus per-block current-noise uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_info_optimal_coded_pwm_study import Case, exogenous, run_trial
from tools.study_coded_pwm_noise_miscalibration import _digest, _paired_effect


ARMS = ("oracle_matched", "block_mixture_t1", "block_mixture_t2", "persistent_t2")


def run_study(*, seed_offset: int, seeds: int, steps: int,
              actual_noise_ratios: tuple[float, ...] = (1.0, 4.0),
              motors: tuple[str, ...] = ("AIR56", "AL31")) -> tuple[dict, list[dict]]:
    if (seeds < 1 or steps < 6 or not motors or not actual_noise_ratios
            or any(not math.isfinite(ratio) or ratio <= 0.0 for ratio in actual_noise_ratios)):
        raise ValueError("positive seeds, at least six steps, motors and noise ratios are required")
    rows: list[dict] = []
    started = perf_counter()
    for motor in motors:
        for actual_ratio in actual_noise_ratios:
            for seed in range(seed_offset, seed_offset + seeds):
                case = Case(motor=motor, scenario="standstill", strength=0.15,
                            nonideal=True, mismatch="combined", seed=seed,
                            steps=steps, measurement_noise_beta_ratio=actual_ratio)
                stimulus = exogenous(case)
                digest = _digest(stimulus)
                settings = (
                    ("oracle_matched", {"assumed_noise_beta_ratio": actual_ratio}),
                    ("block_mixture_t1", {"assumed_noise_beta_ratio": 1.0,
                                          "noise_beta_ratio_candidates": (1.0, 4.0)}),
                    ("block_mixture_t2", {"assumed_noise_beta_ratio": 1.0,
                                          "noise_beta_ratio_candidates": (1.0, 4.0),
                                          "likelihood_temperature": 2.0}),
                    ("persistent_t2", {"persistent_noise_ratios": (1.0, 4.0),
                                       "likelihood_temperature": 2.0}),
                )
                for arm, options in settings:
                    result = run_trial(case, "B2_fixed_probe", stimulus=stimulus,
                                       observer_profile="robust", fixed_probe_vector=6,
                                       **options)
                    mode_probs = result.get("noise_mode_probabilities")
                    rows.append({
                        "motor": motor, "actual_ratio": actual_ratio, "seed": seed,
                        "arm": arm, "stimulus_digest": digest,
                        "angle_p95_deg": result["angle_p95_deg"],
                        "hpd95_coverage": result["hpd95_coverage"],
                        "hpd95_width_deg": result["hpd95_width_deg"],
                        "true_mode_probability": (mode_probs.get(actual_ratio) if mode_probs else None),
                        "compute_mean_us": result["compute_mean_us"],
                        "gateway_rejections": result["gateway_rejections"],
                    })
    lookup = {(r["motor"], r["actual_ratio"], r["seed"], r["arm"]): r for r in rows}
    cells: list[dict] = []
    for motor in motors:
        for actual_ratio in actual_noise_ratios:
            sample = {arm: [lookup[(motor, actual_ratio, seed, arm)]
                            for seed in range(seed_offset, seed_offset + seeds)]
                      for arm in ARMS}
            persistent = sample["persistent_t2"]
            cells.append({
                "motor": motor, "actual_ratio": actual_ratio, "n_seeds": seeds,
                "arms": {arm: {
                    "mean_p95_deg": float(np.mean([r["angle_p95_deg"] for r in items])),
                    "mean_hpd95_coverage": float(np.mean([r["hpd95_coverage"] for r in items])),
                    "mean_hpd95_width_deg": float(np.mean([r["hpd95_width_deg"] for r in items])),
                    "mean_compute_us": float(np.mean([r["compute_mean_us"] for r in items])),
                } for arm, items in sample.items()},
                "persistent_mean_true_mode_probability": (float(np.mean(
                    [r["true_mode_probability"] for r in persistent]))
                    if actual_ratio in (1.0, 4.0) else None),
                "persistent_vs_block_t2": _paired_effect(
                    [r["angle_p95_deg"] for r in sample["block_mixture_t2"]],
                    [r["angle_p95_deg"] for r in persistent]),
                "persistent_vs_oracle": _paired_effect(
                    [r["angle_p95_deg"] for r in sample["oracle_matched"]],
                    [r["angle_p95_deg"] for r in persistent]),
            })
    return ({
        "status": "synthetic_host_only", "seed_range": [seed_offset, seed_offset + seeds - 1],
        "steps_per_trial": steps, "trial_count": len(rows), "motors": list(motors),
        "actual_noise_ratios": list(actual_noise_ratios), "noise_mode_scope": "persistent across trial",
        "probe_vector": 6, "arms": list(ARMS),
        "primary_comparison": "persistent_t2 versus block_mixture_t2 on paired modulo-angle P95",
        "gateway_rejections": sum(int(r["gateway_rejections"]) for r in rows),
        "cells": cells, "elapsed_wall_s": perf_counter() - started,
        "absolute_angle_claim": False, "hardware_claim": False,
    }, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-offset", type=int, default=22000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--actual-ratios", type=float, nargs="+", default=(1.0, 4.0))
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "info_optimal_coded_pwm" / "persistent_noise_holdout")
    args = parser.parse_args()
    payload, rows = run_study(seed_offset=args.seed_offset, seeds=args.seeds, steps=args.steps,
                              actual_noise_ratios=tuple(args.actual_ratios))
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

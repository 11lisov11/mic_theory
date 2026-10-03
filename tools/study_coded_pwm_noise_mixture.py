"""Paired host test of a two-component current-noise likelihood under degradation."""

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


ARMS = ("fixed6_matched", "fixed6_wrong", "fixed6_mixture", "cycle_matched")


def run_study(*, seed_offset: int, seeds: int, steps: int,
              motors: tuple[str, ...] = ("AIR56", "AL31")) -> tuple[dict, list[dict]]:
    if seeds < 1 or steps < 6 or not motors:
        raise ValueError("seeds, steps and motors must be nonempty")
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
                wrong_ratio = 4.0 if actual_ratio == 1.0 else 1.0
                settings = (
                    ("fixed6_matched", 6, actual_ratio, None),
                    ("fixed6_wrong", 6, wrong_ratio, None),
                    ("fixed6_mixture", 6, 1.0, (1.0, 4.0)),
                    ("cycle_matched", None, actual_ratio, None),
                )
                for arm, fixed, assumed_ratio, candidates in settings:
                    result = run_trial(
                        case, "B2_fixed_probe", stimulus=stimulus,
                        observer_profile="robust", fixed_probe_vector=fixed,
                        assumed_noise_beta_ratio=assumed_ratio,
                        noise_beta_ratio_candidates=candidates,
                    )
                    rows.append({
                        "motor": motor, "actual_ratio": actual_ratio, "seed": seed,
                        "arm": arm, "stimulus_digest": digest,
                        "angle_p95_deg": result["angle_p95_deg"],
                        "hpd95_coverage": result["hpd95_coverage"],
                        "hpd95_width_deg": result["hpd95_width_deg"],
                        "torque_ripple_rms_nm": result["torque_ripple_rms_nm"],
                        "switches_per_s": result["switches_per_s"],
                        "probe_copper_energy_j": result["probe_copper_energy_j"],
                        "compute_mean_us": result["compute_mean_us"],
                        "gateway_rejections": result["gateway_rejections"],
                    })
    lookup = {(r["motor"], r["actual_ratio"], r["seed"], r["arm"]): r for r in rows}
    cells: list[dict] = []
    for motor in motors:
        for actual_ratio in (1.0, 4.0):
            sample = {arm: [lookup[(motor, actual_ratio, seed, arm)]
                            for seed in range(seed_offset, seed_offset + seeds)]
                      for arm in ARMS}
            cell = {
                "motor": motor, "actual_ratio": actual_ratio, "n_seeds": seeds,
                "arms": {arm: {
                    "mean_p95_deg": float(np.mean([r["angle_p95_deg"] for r in items])),
                    "mean_hpd95_coverage": float(np.mean([r["hpd95_coverage"] for r in items])),
                    "mean_hpd95_width_deg": float(np.mean([r["hpd95_width_deg"] for r in items])),
                    "mean_compute_us": float(np.mean([r["compute_mean_us"] for r in items])),
                } for arm, items in sample.items()},
            }
            target = [r["angle_p95_deg"] for r in sample["fixed6_mixture"]]
            for baseline in ("fixed6_wrong", "fixed6_matched", "cycle_matched"):
                cell[f"mixture_vs_{baseline}"] = _paired_effect(
                    [r["angle_p95_deg"] for r in sample[baseline]], target,
                )
            cells.append(cell)
    gateway_rejections = sum(int(r["gateway_rejections"]) for r in rows)
    return ({
        "status": "synthetic_host_only", "seed_range": [seed_offset, seed_offset + seeds - 1],
        "steps_per_trial": steps, "trial_count": len(rows),
        "scenario": "standstill, nonideal inverter, combined Rs/Rr/Lm mismatch",
        "assumed_slot_harmonic": 28, "motors": list(motors),
        "actual_noise_ratios": [1.0, 4.0], "arms": list(ARMS),
        "noise_mixture": {"ratios": [1.0, 4.0], "weights": [0.5, 0.5],
                          "latent_scope": "one PWM observation block, not persistent across time"},
        "gateway_rejections": gateway_rejections, "cells": cells,
        "elapsed_wall_s": perf_counter() - started,
        "absolute_angle_claim": False, "hardware_claim": False,
    }, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-offset", type=int, default=20000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "info_optimal_coded_pwm" / "noise_mixture_holdout")
    args = parser.parse_args()
    payload, rows = run_study(seed_offset=args.seed_offset, seeds=args.seeds, steps=args.steps)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    with (args.out / "trials.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: payload[key] for key in (
        "status", "seed_range", "trial_count", "gateway_rejections", "elapsed_wall_s"
    )}, indent=2))
    if gateway_rejections := payload["gateway_rejections"]:
        raise SystemExit(f"gateway rejections: {gateway_rejections}")


if __name__ == "__main__":
    main()

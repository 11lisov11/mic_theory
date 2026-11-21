"""Held-out paired test of current-noise miscalibration in synthetic slot sensing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_info_optimal_coded_pwm_study import Case, exogenous, run_trial


ARMS = (("cycle", "B2_fixed_probe", None),
        ("fixed_6", "B2_fixed_probe", 6),
        ("fisher", "B3_info_fisher", None))


def _digest(stimulus: dict) -> str:
    raw = np.asarray([stimulus["theta0"], stimulus["bus_ripple"], stimulus["speed_bias_fraction"]]).tobytes()
    for name in ("base_noise", "probe_noise", "sample_jitter"):
        raw += np.asarray(stimulus[name]).tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def _paired_effect(a: list[float], b: list[float]) -> dict:
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    effects = (left - right) / np.maximum(left, 1e-12)
    rng = np.random.default_rng(20261003)
    bootstrap = np.mean(rng.choice(effects, size=(2000, len(effects)), replace=True), axis=1)
    return {"mean_relative_reduction": float(np.mean(effects)),
            "bootstrap95_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap95_high": float(np.quantile(bootstrap, 0.975))}


def run_study(*, seed_offset: int = 18000, seeds: int = 20, steps: int = 500,
              motors: tuple[str, ...] = ("AIR56", "AL31")) -> tuple[dict, list[dict]]:
    if seeds < 1 or steps < 6:
        raise ValueError("positive seed count and at least six PWM periods are required")
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
                for assumed_ratio in (1.0, 4.0):
                    for arm, method, fixed in ARMS:
                        result = run_trial(case, method, stimulus=stimulus,
                                           observer_profile="robust",
                                           fixed_probe_vector=fixed,
                                           assumed_noise_beta_ratio=assumed_ratio)
                        rows.append({
                            "motor": motor, "actual_ratio": actual_ratio,
                            "assumed_ratio": assumed_ratio, "seed": seed,
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
    lookup = {(r["motor"], r["actual_ratio"], r["assumed_ratio"], r["seed"], r["arm"]): r for r in rows}
    cells: list[dict] = []
    mismatch_effects: list[dict] = []
    for motor in motors:
        for actual_ratio in (1.0, 4.0):
            for assumed_ratio in (1.0, 4.0):
                sample = {arm: [lookup[(motor, actual_ratio, assumed_ratio, seed, arm)]
                                for seed in range(seed_offset, seed_offset + seeds)]
                          for arm, _, _ in ARMS}
                cell = {"motor": motor, "actual_ratio": actual_ratio,
                        "assumed_ratio": assumed_ratio, "n_seeds": seeds,
                        "arms": {arm: {
                            "mean_p95_deg": float(np.mean([r["angle_p95_deg"] for r in items])),
                            "mean_hpd95_coverage": float(np.mean([r["hpd95_coverage"] for r in items])),
                            "mean_hpd95_width_deg": float(np.mean([r["hpd95_width_deg"] for r in items])),
                            "mean_compute_us": float(np.mean([r["compute_mean_us"] for r in items])),
                        } for arm, items in sample.items()}}
                for baseline in ("cycle", "fixed_6"):
                    cell[f"fisher_vs_{baseline}"] = _paired_effect(
                        [r["angle_p95_deg"] for r in sample[baseline]],
                        [r["angle_p95_deg"] for r in sample["fisher"]],
                    )
                cells.append(cell)
            for arm, _, _ in ARMS:
                matched = [lookup[(motor, actual_ratio, actual_ratio, seed, arm)]["angle_p95_deg"]
                           for seed in range(seed_offset, seed_offset + seeds)]
                wrong_ratio = 4.0 if actual_ratio == 1.0 else 1.0
                wrong = [lookup[(motor, actual_ratio, wrong_ratio, seed, arm)]["angle_p95_deg"]
                         for seed in range(seed_offset, seed_offset + seeds)]
                mismatch_effects.append({"motor": motor, "actual_ratio": actual_ratio,
                                         "arm": arm, "matched_vs_miscalibrated": _paired_effect(wrong, matched)})
    gateway_faults = sum(int(r["gateway_rejections"]) for r in rows)
    return ({
        "status": "synthetic_host_holdout_only", "seed_range": [seed_offset, seed_offset + seeds - 1],
        "steps_per_trial": steps, "trial_count": len(rows),
        "physical_scenario": "standstill, nonideal inverter, combined Rs/Rr/Lm mismatch",
        "assumed_slot_harmonic": 28, "motors": list(motors),
        "true_noise_ratios": [1.0, 4.0], "observer_noise_ratios": [1.0, 4.0],
        "arms": [name for name, _, _ in ARMS], "gateway_rejections": gateway_faults,
        "cells": cells, "mismatch_effects": mismatch_effects,
        "elapsed_wall_s": perf_counter() - started,
        "absolute_angle_claim": False, "hardware_claim": False,
    }, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-offset", type=int, default=18000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "info_optimal_coded_pwm" / "noise_miscalibration_holdout")
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
    if payload["gateway_rejections"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

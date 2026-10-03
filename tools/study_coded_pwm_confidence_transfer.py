"""Test a frozen standstill confidence threshold under prescribed rotor motion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_info_optimal_coded_pwm_study import Case, exogenous
from tools.study_coded_pwm_confidence_gate import (
    BAD_ERROR_DEG, _bootstrap_seed_effect, _matched_posterior_baselines, _trial,
)
from tools.study_coded_pwm_noise_miscalibration import _digest


PROFILES = (("constant", 0.01), ("ramp", 0.02), ("reversal", 0.02))
RATIOS = (1.0, 3.0, 8.0)
MOTORS = ("AIR56", "AL31")


def run_study(*, frozen_threshold_deg: float, seed_offset: int = 33000,
              seeds: int = 20, steps: int = 500,
              profiles: tuple[tuple[str, float], ...] = PROFILES,
              ratios: tuple[float, ...] = RATIOS,
              motors: tuple[str, ...] = MOTORS) -> tuple[dict, list[dict]]:
    if (not math.isfinite(frozen_threshold_deg) or frozen_threshold_deg <= 0.0
            or seeds < 1 or steps < 6 or not profiles or not ratios or not motors):
        raise ValueError("a frozen threshold and nonempty experiment are required")
    rows: list[dict] = []
    for seed in range(seed_offset, seed_offset + seeds):
        for motor in motors:
            for scenario, speed_fraction in profiles:
                for ratio in ratios:
                    case = Case(motor=motor, scenario=scenario,
                                speed_fraction=speed_fraction, strength=0.15,
                                nonideal=True, mismatch="combined", seed=seed,
                                steps=steps, measurement_noise_beta_ratio=ratio)
                    result, widths = _trial(case)
                    period = math.radians(result["period_deg"])
                    estimates = np.asarray(result["trace_estimate_rad"])
                    truth = np.asarray(result["trace_theta_rad"])
                    errors = np.degrees(np.abs((estimates - truth + period / 2.0)
                                                % period - period / 2.0))
                    accepted = widths <= frozen_threshold_deg
                    count = int(np.count_nonzero(accepted))
                    entropy_selected, _ = _matched_posterior_baselines(
                        result["trace_angle_marginal"], count)
                    bad = errors > BAD_ERROR_DEG
                    rows.append({
                        "motor": motor, "scenario": scenario,
                        "speed_fraction": speed_fraction, "actual_ratio": ratio,
                        "seed": seed, "stimulus_digest": _digest(exogenous(case)),
                        "accepted_count": count, "availability": count / steps,
                        "bad_all_fraction": float(np.mean(bad)),
                        "bad_gate_count": int(np.count_nonzero(bad & accepted)),
                        "bad_entropy_count": int(np.count_nonzero(bad & entropy_selected)),
                        "random_expected_bad_count": float(count * np.mean(bad)),
                        "gateway_rejections": result["gateway_rejections"],
                    })
    cells = []
    for motor in motors:
        for scenario, speed_fraction in profiles:
            for ratio in ratios:
                group = [r for r in rows if r["motor"] == motor
                         and r["scenario"] == scenario
                         and r["speed_fraction"] == speed_fraction
                         and r["actual_ratio"] == ratio]
                accepted = sum(r["accepted_count"] for r in group)
                cells.append({
                    "motor": motor, "scenario": scenario,
                    "speed_fraction": speed_fraction, "actual_ratio": ratio,
                    "mean_availability": float(np.mean([r["availability"] for r in group])),
                    "zero_availability_trials": sum(r["accepted_count"] == 0 for r in group),
                    "bad_rate_all": float(np.mean([r["bad_all_fraction"] for r in group])),
                    "bad_rate_gate": (sum(r["bad_gate_count"] for r in group) / accepted
                                      if accepted else None),
                    "bad_rate_entropy": (sum(r["bad_entropy_count"] for r in group) / accepted
                                         if accepted else None),
                    "bad_rate_random": (sum(r["random_expected_bad_count"] for r in group)
                                        / accepted if accepted else None),
                })
    effects = {"random": [], "entropy": []}
    for seed in range(seed_offset, seed_offset + seeds):
        group = [r for r in rows if r["seed"] == seed]
        accepted = sum(r["accepted_count"] for r in group)
        if accepted:
            for baseline, key in (("random", "random_expected_bad_count"),
                                  ("entropy", "bad_entropy_count")):
                effects[baseline].append((sum(r[key] - r["bad_gate_count"] for r in group))
                                         / accepted)
    summary = {
        "status": "synthetic_host_only", "frozen_threshold_deg": frozen_threshold_deg,
        "seed_range": [seed_offset, seed_offset + seeds - 1],
        "steps_per_trial": steps, "trial_count": len(rows),
        "motors": list(motors), "profiles": [list(p) for p in profiles],
        "actual_noise_ratios": list(ratios), "bad_error_threshold_deg": BAD_ERROR_DEG,
        "mean_availability": float(np.mean([r["availability"] for r in rows])),
        "random_minus_gate": _bootstrap_seed_effect(effects["random"]),
        "entropy_minus_gate": _bootstrap_seed_effect(effects["entropy"]),
        "gateway_rejections": sum(int(r["gateway_rejections"]) for r in rows),
        "cells": cells, "absolute_angle_claim": False, "hardware_claim": False,
        "motion_is_prescribed": True,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-summary", type=Path, default=ROOT /
                        "experiments/info_optimal_coded_pwm/confidence_gate_holdout/summary.json")
    parser.add_argument("--seed-offset", type=int, default=33000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", type=Path, default=ROOT /
                        "experiments/info_optimal_coded_pwm/confidence_transfer_holdout")
    args = parser.parse_args()
    calibration = json.loads(args.calibration_summary.read_text(encoding="utf-8"))
    summary, rows = run_study(frozen_threshold_deg=float(calibration["width_threshold_deg"]),
                              seed_offset=args.seed_offset, seeds=args.seeds,
                              steps=args.steps)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False),
                                           encoding="utf-8")
    with (args.out / "trials.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: summary[key] for key in (
        "status", "frozen_threshold_deg", "seed_range", "trial_count",
        "mean_availability", "random_minus_gate", "entropy_minus_gate",
        "gateway_rejections")}, indent=2))
    if summary["gateway_rejections"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

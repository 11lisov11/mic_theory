"""Trial-group split calibration of modulo-angle posterior sets on synthetic data."""

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

from control.conformal_rotor_set import rank_mass
from tools.run_info_optimal_coded_pwm_study import Case, exogenous, run_trial
from tools.study_coded_pwm_noise_miscalibration import _digest


MOTORS = ("AIR56", "AL31")
RATIOS = (1.0, 2.0, 3.0, 8.0)
TRIAL_COVERAGE_TARGET = 0.95
GROUP_RELIABILITY_TARGET = 0.95


def _trial(case: Case) -> dict:
    stimulus = exogenous(case)
    result = run_trial(case, "B2_fixed_probe", stimulus=stimulus,
                       observer_profile="robust", fixed_probe_vector=6,
                       assumed_noise_beta_ratio=1.0,
                       noise_beta_ratio_candidates=(1.0, 4.0),
                       likelihood_temperature=2.0,
                       capture_posterior_trace=True)
    period_rad = math.radians(result["period_deg"])
    ranks = np.asarray([rank_mass(np.asarray(marginal))
                        for marginal in result["trace_angle_marginal"]])
    n_angle = ranks.shape[1]
    true_bins = np.asarray([int((theta % period_rad) / period_rad * n_angle) % n_angle
                            for theta in result["trace_theta_rad"]], dtype=int)
    scores = ranks[np.arange(len(ranks)), true_bins]
    return {
        "motor": case.motor, "actual_ratio": case.measurement_noise_beta_ratio,
        "seed": case.seed, "stimulus_digest": _digest(stimulus),
        "raw_hpd95_coverage": result["hpd95_coverage"],
        "raw_hpd95_width_deg": result["hpd95_width_deg"],
        "trial_score95": float(np.quantile(scores, TRIAL_COVERAGE_TARGET, method="higher")),
        "gateway_rejections": result["gateway_rejections"],
        "ranks": ranks, "scores": scores, "period_deg": result["period_deg"],
    }


def _threshold(group_scores: list[float], reliability: float = GROUP_RELIABILITY_TARGET) -> float:
    if (not group_scores or not 0.0 < reliability < 1.0
            or any(not math.isfinite(score) or not 0.0 <= score <= 1.0
                   for score in group_scores)):
        raise ValueError("scores must be in [0,1], with reliability in (0,1)")
    k = math.ceil((len(group_scores) + 1) * reliability)
    return float(sorted(group_scores)[k - 1]) if k <= len(group_scores) else 1.0


def run_study(*, calibration_offset: int = 28000, holdout_offset: int = 29000,
              calibration_seeds: int = 20, holdout_seeds: int = 20,
              steps: int = 500, motors: tuple[str, ...] = MOTORS,
              ratios: tuple[float, ...] = RATIOS) -> tuple[dict, list[dict], list[dict]]:
    if (calibration_seeds < 1 or holdout_seeds < 1 or steps < 6 or not motors or not ratios
            or set(range(calibration_offset, calibration_offset + calibration_seeds))
            & set(range(holdout_offset, holdout_offset + holdout_seeds))):
        raise ValueError("calibration and holdout must be nonempty and use disjoint seeds")
    started = perf_counter()
    calibration_rows: list[dict] = []
    group_scores: list[float] = []
    for seed in range(calibration_offset, calibration_offset + calibration_seeds):
        worst = 0.0
        for motor in motors:
            for ratio in ratios:
                record = _trial(Case(motor=motor, scenario="standstill", strength=0.15,
                                     nonideal=True, mismatch="combined", seed=seed,
                                     steps=steps, measurement_noise_beta_ratio=ratio))
                worst = max(worst, record["trial_score95"])
                calibration_rows.append({key: record[key] for key in (
                    "motor", "actual_ratio", "seed", "stimulus_digest", "trial_score95",
                    "raw_hpd95_coverage", "raw_hpd95_width_deg", "gateway_rejections")})
        group_scores.append(worst)
    threshold = _threshold(group_scores)
    holdout_rows: list[dict] = []
    for seed in range(holdout_offset, holdout_offset + holdout_seeds):
        for motor in motors:
            for ratio in ratios:
                record = _trial(Case(motor=motor, scenario="standstill", strength=0.15,
                                     nonideal=True, mismatch="combined", seed=seed,
                                     steps=steps, measurement_noise_beta_ratio=ratio))
                widths = np.sum(record["ranks"] <= threshold, axis=1)
                holdout_rows.append({
                    **{key: record[key] for key in (
                        "motor", "actual_ratio", "seed", "stimulus_digest", "trial_score95",
                        "raw_hpd95_coverage", "raw_hpd95_width_deg", "gateway_rejections")},
                    "calibrated_coverage": float(np.mean(record["scores"] <= threshold)),
                    "calibrated_width_deg": float(np.mean(widths) * record["period_deg"]
                                                   / record["ranks"].shape[1]),
                })
    cells = []
    for motor in motors:
        for ratio in ratios:
            subset = [r for r in holdout_rows if r["motor"] == motor and r["actual_ratio"] == ratio]
            cells.append({
                "motor": motor, "actual_ratio": ratio,
                "mean_raw_coverage": float(np.mean([r["raw_hpd95_coverage"] for r in subset])),
                "mean_calibrated_coverage": float(np.mean([r["calibrated_coverage"] for r in subset])),
                "mean_raw_width_deg": float(np.mean([r["raw_hpd95_width_deg"] for r in subset])),
                "mean_calibrated_width_deg": float(np.mean([r["calibrated_width_deg"] for r in subset])),
                "trial_pass_fraction": float(np.mean([
                    r["calibrated_coverage"] >= TRIAL_COVERAGE_TARGET for r in subset])),
            })
    group_passes = [all(r["calibrated_coverage"] >= TRIAL_COVERAGE_TARGET
                        for r in holdout_rows if r["seed"] == seed)
                    for seed in range(holdout_offset, holdout_offset + holdout_seeds)]
    return ({
        "status": "synthetic_host_only", "calibration_seeds": [calibration_offset,
            calibration_offset + calibration_seeds - 1],
        "holdout_seeds": [holdout_offset, holdout_offset + holdout_seeds - 1],
        "motors": list(motors), "actual_noise_ratios": list(ratios),
        "steps_per_trial": steps, "calibration_trial_count": len(calibration_rows),
        "holdout_trial_count": len(holdout_rows),
        "calibration_group_scores": group_scores,
        "global_rank_threshold": threshold,
        "trial_coverage_target": TRIAL_COVERAGE_TARGET,
        "group_reliability_target": GROUP_RELIABILITY_TARGET,
        "holdout_all_scenarios_pass_count": int(sum(group_passes)),
        "holdout_all_scenarios_pass_fraction": float(np.mean(group_passes)),
        "gateway_rejections": sum(int(r["gateway_rejections"])
                                  for r in calibration_rows + holdout_rows),
        "cells": cells, "elapsed_wall_s": perf_counter() - started,
        "absolute_angle_claim": False, "hardware_claim": False,
    }, calibration_rows, holdout_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-offset", type=int, default=28000)
    parser.add_argument("--holdout-offset", type=int, default=29000)
    parser.add_argument("--calibration-seeds", type=int, default=20)
    parser.add_argument("--holdout-seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "info_optimal_coded_pwm" / "conformal_set_holdout")
    args = parser.parse_args()
    payload, calibration_rows, holdout_rows = run_study(
        calibration_offset=args.calibration_offset, holdout_offset=args.holdout_offset,
        calibration_seeds=args.calibration_seeds, holdout_seeds=args.holdout_seeds,
        steps=args.steps)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    for name, rows in (("calibration_trials.csv", calibration_rows),
                       ("holdout_trials.csv", holdout_rows)):
        with (args.out / name).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({key: payload[key] for key in (
        "status", "calibration_seeds", "holdout_seeds", "global_rank_threshold",
        "holdout_all_scenarios_pass_count", "gateway_rejections", "elapsed_wall_s"
    )}, indent=2))
    if payload["gateway_rejections"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

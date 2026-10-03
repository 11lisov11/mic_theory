"""Paired host audit of posterior-width abstention under sensor degradation."""

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

from control.posterior_confidence_gate import hpd_width_deg
from tools.run_info_optimal_coded_pwm_study import Case, exogenous, run_trial
from tools.study_coded_pwm_noise_miscalibration import _digest


MOTORS = ("AIR56", "AL31")
RATIOS = (1.0, 2.0, 3.0, 8.0)
BAD_ERROR_DEG = 2.0


def _trial(case: Case) -> tuple[dict, np.ndarray]:
    stimulus = exogenous(case)
    result = run_trial(case, "B2_fixed_probe", stimulus=stimulus,
                       observer_profile="robust", fixed_probe_vector=6,
                       assumed_noise_beta_ratio=1.0,
                       noise_beta_ratio_candidates=(1.0, 4.0),
                       likelihood_temperature=2.0, capture_posterior_trace=True)
    widths = np.asarray([hpd_width_deg(np.asarray(marginal), result["period_deg"])
                         for marginal in result["trace_angle_marginal"]])
    return result, widths


def _case(motor: str, ratio: float, seed: int, steps: int) -> Case:
    return Case(motor=motor, scenario="standstill", strength=0.15,
                nonideal=True, mismatch="combined", seed=seed,
                steps=steps, measurement_noise_beta_ratio=ratio)


def _bootstrap_seed_effect(effects: list[float]) -> dict | None:
    if not effects:
        return None
    values = np.asarray(effects, dtype=float)
    rng = np.random.default_rng(20261003)
    bootstrap = np.mean(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    return {"mean_baseline_minus_gate": float(np.mean(values)),
            "bootstrap95_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap95_high": float(np.quantile(bootstrap, 0.975)),
            "n_seed_groups": len(effects)}


def _matched_posterior_baselines(trace: list, count: int) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(trace, dtype=float)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    log_probabilities = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0.0)
    entropy = -np.sum(probabilities * log_probabilities, axis=1)
    peak = np.max(probabilities, axis=1)
    entropy_selected = np.zeros(len(probabilities), dtype=bool)
    peak_selected = np.zeros(len(probabilities), dtype=bool)
    entropy_selected[np.argsort(entropy, kind="stable")[:count]] = True
    peak_selected[np.argsort(-peak, kind="stable")[:count]] = True
    return entropy_selected, peak_selected


def run_study(*, calibration_offset: int = 30000, holdout_offset: int = 31000,
              calibration_seeds: int = 20, holdout_seeds: int = 20,
              steps: int = 500, motors: tuple[str, ...] = MOTORS,
              ratios: tuple[float, ...] = RATIOS) -> tuple[dict, list[dict], list[dict]]:
    if (calibration_seeds < 1 or holdout_seeds < 1 or steps < 6 or not motors or not ratios
            or set(range(calibration_offset, calibration_offset + calibration_seeds))
            & set(range(holdout_offset, holdout_offset + holdout_seeds))):
        raise ValueError("calibration and holdout must be nonempty and use disjoint seeds")
    started = perf_counter()
    calibration_widths: list[np.ndarray] = []
    calibration_rows: list[dict] = []
    for seed in range(calibration_offset, calibration_offset + calibration_seeds):
        for motor in motors:
            for ratio in ratios:
                case = _case(motor, ratio, seed, steps)
                result, widths = _trial(case)
                calibration_widths.append(widths)
                calibration_rows.append({
                    "motor": motor, "actual_ratio": ratio, "seed": seed,
                    "stimulus_digest": _digest(exogenous(case)),
                    "median_width_deg": float(np.median(widths)),
                    "gateway_rejections": result["gateway_rejections"],
                })
    threshold = float(np.quantile(np.concatenate(calibration_widths), 0.5, method="higher"))
    holdout_rows: list[dict] = []
    for seed in range(holdout_offset, holdout_offset + holdout_seeds):
        for motor in motors:
            for ratio in ratios:
                case = _case(motor, ratio, seed, steps)
                result, widths = _trial(case)
                period = math.radians(result["period_deg"])
                estimates = np.asarray(result["trace_estimate_rad"])
                truth = np.asarray(result["trace_theta_rad"])
                errors = np.degrees(np.abs((estimates - truth + period / 2.0) % period - period / 2.0))
                accepted = widths <= threshold
                n_accepted = int(np.count_nonzero(accepted))
                entropy_selected, peak_selected = _matched_posterior_baselines(
                    result["trace_angle_marginal"], n_accepted)
                bad = errors > BAD_ERROR_DEG
                holdout_rows.append({
                    "motor": motor, "actual_ratio": ratio, "seed": seed,
                    "stimulus_digest": _digest(exogenous(case)),
                    "accepted_count": n_accepted, "availability": n_accepted / len(errors),
                    "p95_all_deg": float(np.quantile(errors, 0.95)),
                    "p95_accepted_deg": (float(np.quantile(errors[accepted], 0.95))
                                         if n_accepted else None),
                    "bad_all_fraction": float(np.mean(bad)),
                    "bad_accepted_count": int(np.count_nonzero(bad & accepted)),
                    "bad_entropy_count": int(np.count_nonzero(bad & entropy_selected)),
                    "bad_peak_count": int(np.count_nonzero(bad & peak_selected)),
                    "random_expected_bad_count": float(n_accepted * np.mean(bad)),
                    "gateway_rejections": result["gateway_rejections"],
                })
    cells = []
    for motor in motors:
        for ratio in ratios:
            subset = [r for r in holdout_rows if r["motor"] == motor and r["actual_ratio"] == ratio]
            total_accepted = sum(r["accepted_count"] for r in subset)
            cells.append({
                "motor": motor, "actual_ratio": ratio,
                "mean_availability": float(np.mean([r["availability"] for r in subset])),
                "zero_availability_trials": sum(r["accepted_count"] == 0 for r in subset),
                "mean_p95_all_deg": float(np.mean([r["p95_all_deg"] for r in subset])),
                "mean_p95_accepted_deg": (float(np.mean([
                    r["p95_accepted_deg"] for r in subset if r["p95_accepted_deg"] is not None]))
                    if total_accepted else None),
                "bad_rate_all": float(np.mean([r["bad_all_fraction"] for r in subset])),
                "bad_rate_accepted": (sum(r["bad_accepted_count"] for r in subset) / total_accepted
                                      if total_accepted else None),
                "bad_rate_entropy": (sum(r["bad_entropy_count"] for r in subset) / total_accepted
                                     if total_accepted else None),
                "bad_rate_peak": (sum(r["bad_peak_count"] for r in subset) / total_accepted
                                  if total_accepted else None),
                "bad_rate_matched_random": (sum(r["random_expected_bad_count"] for r in subset)
                                            / total_accepted if total_accepted else None),
            })
    seed_effects = {"random": [], "entropy": [], "peak": []}
    for seed in range(holdout_offset, holdout_offset + holdout_seeds):
        subset = [r for r in holdout_rows if r["seed"] == seed]
        total = sum(r["accepted_count"] for r in subset)
        if total:
            for baseline, key in (("random", "random_expected_bad_count"),
                                  ("entropy", "bad_entropy_count"),
                                  ("peak", "bad_peak_count")):
                seed_effects[baseline].append((sum(r[key] for r in subset)
                                               - sum(r["bad_accepted_count"] for r in subset)) / total)
    return ({
        "status": "synthetic_host_only", "calibration_seeds": [calibration_offset,
            calibration_offset + calibration_seeds - 1],
        "holdout_seeds": [holdout_offset, holdout_offset + holdout_seeds - 1],
        "motors": list(motors), "actual_noise_ratios": list(ratios),
        "steps_per_trial": steps, "calibration_trial_count": len(calibration_rows),
        "holdout_trial_count": len(holdout_rows),
        "width_threshold_deg": threshold, "bad_error_threshold_deg": BAD_ERROR_DEG,
        "target_calibration_availability": 0.5,
        "primary_comparison": "seed-paired bad-error risk, gate versus matched-count random selection",
        "primary_random_minus_gate": _bootstrap_seed_effect(seed_effects["random"]),
        "secondary_entropy_minus_gate": _bootstrap_seed_effect(seed_effects["entropy"]),
        "secondary_peak_minus_gate": _bootstrap_seed_effect(seed_effects["peak"]),
        "mean_holdout_availability": float(np.mean([r["availability"] for r in holdout_rows])),
        "gateway_rejections": sum(int(r["gateway_rejections"])
                                  for r in calibration_rows + holdout_rows),
        "cells": cells, "elapsed_wall_s": perf_counter() - started,
        "absolute_angle_claim": False, "hardware_claim": False,
    }, calibration_rows, holdout_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-offset", type=int, default=30000)
    parser.add_argument("--holdout-offset", type=int, default=31000)
    parser.add_argument("--calibration-seeds", type=int, default=20)
    parser.add_argument("--holdout-seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "info_optimal_coded_pwm" / "confidence_gate_holdout")
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
        "status", "calibration_seeds", "holdout_seeds", "width_threshold_deg",
        "mean_holdout_availability", "gateway_rejections", "elapsed_wall_s"
    )}, indent=2))
    if payload["gateway_rejections"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

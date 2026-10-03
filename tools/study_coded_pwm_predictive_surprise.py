"""Held-out test of predictive surprise for rejecting confident angle errors."""

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
from tools.study_coded_pwm_confidence_gate import BAD_ERROR_DEG, _trial
from tools.study_coded_pwm_noise_miscalibration import _digest
from tools.study_coded_pwm_speed_diffusion import (
    FROZEN_WIDTH_THRESHOLD_DEG, MOTORS, PROFILES, RATIOS,
)


def _select_lowest_within(eligible: np.ndarray, scores: np.ndarray,
                          count: int) -> np.ndarray:
    if len(eligible) != len(scores) or count < 0 or count > int(np.count_nonzero(eligible)):
        raise ValueError("invalid matched-count selection")
    selection = np.zeros(len(scores), dtype=bool)
    eligible_indices = np.flatnonzero(eligible)
    chosen = eligible_indices[np.argsort(scores[eligible], kind="stable")[:count]]
    selection[chosen] = True
    return selection


def _bootstrap_seed_effect(effects: list[float]) -> dict | None:
    if not effects:
        return None
    values = np.asarray(effects, dtype=float)
    rng = np.random.default_rng(20261003)
    samples = np.mean(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    return {"mean_baseline_minus_surprise": float(np.mean(values)),
            "bootstrap95_low": float(np.quantile(samples, 0.025)),
            "bootstrap95_high": float(np.quantile(samples, 0.975)),
            "n_seed_groups": len(effects)}


def run_study(*, seed_offset: int = 36100, seeds: int = 20, steps: int = 500,
              profiles: tuple[tuple[str, float], ...] = PROFILES,
              ratios: tuple[float, ...] = RATIOS,
              motors: tuple[str, ...] = MOTORS) -> tuple[dict, list[dict]]:
    if seeds < 1 or steps < 6 or not profiles or not ratios or not motors:
        raise ValueError("nonempty holdout required")
    rows: list[dict] = []
    for seed in range(seed_offset, seed_offset + seeds):
        for motor in motors:
            for scenario, speed_fraction in profiles:
                for ratio in ratios:
                    case = Case(motor=motor, scenario=scenario,
                                speed_fraction=speed_fraction, strength=0.15,
                                nonideal=True, mismatch="combined", seed=seed,
                                steps=steps, measurement_noise_beta_ratio=ratio)
                    result, widths = _trial(case, command_slew_speed_diffusion=True)
                    period = math.radians(result["period_deg"])
                    estimates = np.asarray(result["trace_estimate_rad"])
                    truth = np.asarray(result["trace_theta_rad"])
                    errors = np.degrees(np.abs((estimates - truth + period / 2.0)
                                                % period - period / 2.0))
                    bad = errors > BAD_ERROR_DEG
                    width_accepted = widths <= FROZEN_WIDTH_THRESHOLD_DEG
                    first_count = int(np.count_nonzero(width_accepted))
                    second_count = first_count // 2
                    posterior = np.asarray(result["trace_angle_marginal"])
                    log_posterior = np.zeros_like(posterior)
                    np.log(posterior, out=log_posterior, where=posterior > 0.0)
                    entropy = -np.sum(posterior * log_posterior, axis=1)
                    surprise = np.asarray(result["trace_predictive_surprise"])
                    if not np.all(np.isfinite(surprise)):
                        raise FloatingPointError("nonfinite predictive surprise")
                    surprise_selection = _select_lowest_within(
                        width_accepted, surprise, second_count)
                    entropy_selection = _select_lowest_within(
                        width_accepted, entropy, second_count)
                    rows.append({
                        "seed": seed, "motor": motor, "scenario": scenario,
                        "speed_fraction": speed_fraction, "actual_ratio": ratio,
                        "stimulus_digest": _digest(exogenous(case)),
                        "width_accepted_count": first_count,
                        "second_stage_count": second_count,
                        "bad_all_fraction": float(np.mean(bad)),
                        "bad_width_count": int(np.count_nonzero(bad & width_accepted)),
                        "bad_surprise_count": int(np.count_nonzero(bad & surprise_selection)),
                        "bad_entropy_count": int(np.count_nonzero(bad & entropy_selection)),
                        "random_expected_bad_count": (second_count
                                                      * np.mean(bad[width_accepted])
                                                      if first_count else 0.0),
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
                first = sum(r["width_accepted_count"] for r in group)
                second = sum(r["second_stage_count"] for r in group)
                cells.append({
                    "motor": motor, "scenario": scenario,
                    "speed_fraction": speed_fraction, "actual_ratio": ratio,
                    "width_availability": first / (len(group) * steps),
                    "second_stage_availability": second / (len(group) * steps),
                    "zero_second_stage_trials": sum(r["second_stage_count"] == 0 for r in group),
                    "bad_rate_width": (sum(r["bad_width_count"] for r in group) / first
                                       if first else None),
                    "bad_rate_surprise": (sum(r["bad_surprise_count"] for r in group) / second
                                          if second else None),
                    "bad_rate_entropy": (sum(r["bad_entropy_count"] for r in group) / second
                                         if second else None),
                    "bad_rate_random": (sum(r["random_expected_bad_count"] for r in group)
                                        / second if second else None),
                })
    effects = {"random": [], "entropy": []}
    for seed in range(seed_offset, seed_offset + seeds):
        group = [r for r in rows if r["seed"] == seed]
        count = sum(r["second_stage_count"] for r in group)
        if count:
            for baseline, column in (("random", "random_expected_bad_count"),
                                     ("entropy", "bad_entropy_count")):
                effects[baseline].append(
                    sum(r[column] - r["bad_surprise_count"] for r in group) / count)
    summary = {
        "status": "synthetic_host_only", "seed_range": [seed_offset, seed_offset + seeds - 1],
        "trial_count": len(rows), "steps_per_trial": steps,
        "observer": "command_slew_fixed_probe_per_block_mixture_T2",
        "first_stage_width_threshold_deg": FROZEN_WIDTH_THRESHOLD_DEG,
        "second_stage_fraction_of_width_accepted": 0.5,
        "bad_error_threshold_deg": BAD_ERROR_DEG,
        "mean_second_stage_availability": float(np.mean([
            r["second_stage_count"] / steps for r in rows])),
        "random_minus_surprise": _bootstrap_seed_effect(effects["random"]),
        "entropy_minus_surprise": _bootstrap_seed_effect(effects["entropy"]),
        "gateway_rejections": sum(int(r["gateway_rejections"]) for r in rows),
        "cells": cells, "motion_is_prescribed": True,
        "selection_uses_full_trial_ranking": True,
        "absolute_angle_claim": False, "hardware_claim": False,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-offset", type=int, default=36100)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", type=Path, default=ROOT /
                        "experiments/info_optimal_coded_pwm/predictive_surprise_holdout")
    args = parser.parse_args()
    summary, rows = run_study(seed_offset=args.seed_offset, seeds=args.seeds,
                              steps=args.steps)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False),
                                           encoding="utf-8")
    with (args.out / "trials.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: summary[key] for key in (
        "status", "seed_range", "trial_count", "mean_second_stage_availability",
        "random_minus_surprise", "entropy_minus_surprise", "gateway_rejections")}, indent=2))
    if summary["gateway_rejections"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

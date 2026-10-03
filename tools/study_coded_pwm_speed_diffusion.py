"""Paired synthetic test of speed-transition diffusion under rotor motion."""

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


PROFILES = (("constant", 0.01), ("ramp", 0.02), ("reversal", 0.02))
RATIOS = (3.0, 8.0)
MOTORS = ("AIR56", "AL31")
BASELINE_DIFFUSION = 0.001
CANDIDATE_DIFFUSION = 0.05
FROZEN_WIDTH_THRESHOLD_DEG = 4.821428571428572


def _bootstrap(effects: list[float]) -> dict | None:
    if not effects:
        return None
    values = np.asarray(effects, dtype=float)
    rng = np.random.default_rng(20261003)
    samples = np.mean(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    return {"mean_baseline_minus_candidate": float(np.mean(values)),
            "bootstrap95_low": float(np.quantile(samples, 0.025)),
            "bootstrap95_high": float(np.quantile(samples, 0.975)),
            "n_seed_groups": len(effects)}


def run_study(*, seed_offset: int = 34000, seeds: int = 20, steps: int = 500,
              profiles: tuple[tuple[str, float], ...] = PROFILES,
              ratios: tuple[float, ...] = RATIOS,
              motors: tuple[str, ...] = MOTORS) -> tuple[dict, list[dict]]:
    if seeds < 1 or steps < 6 or not profiles or not ratios or not motors:
        raise ValueError("nonempty paired experiment required")
    rows: list[dict] = []
    for seed in range(seed_offset, seed_offset + seeds):
        for motor in motors:
            for scenario, speed_fraction in profiles:
                for ratio in ratios:
                    case = Case(motor=motor, scenario=scenario,
                                speed_fraction=speed_fraction, strength=0.15,
                                nonideal=True, mismatch="combined", seed=seed,
                                steps=steps, measurement_noise_beta_ratio=ratio)
                    digest = _digest(exogenous(case))
                    for arm, diffusion in (("baseline", BASELINE_DIFFUSION),
                                           ("candidate", CANDIDATE_DIFFUSION)):
                        result, widths = _trial(case, speed_diffusion_override=diffusion)
                        period = math.radians(result["period_deg"])
                        estimates = np.asarray(result["trace_estimate_rad"])
                        truth = np.asarray(result["trace_theta_rad"])
                        errors = np.degrees(np.abs((estimates - truth + period / 2.0)
                                                    % period - period / 2.0))
                        accepted = widths <= FROZEN_WIDTH_THRESHOLD_DEG
                        rows.append({
                            "seed": seed, "motor": motor, "scenario": scenario,
                            "speed_fraction": speed_fraction, "actual_ratio": ratio,
                            "arm": arm, "speed_diffusion": diffusion,
                            "stimulus_digest": digest,
                            "bad_all_fraction": float(np.mean(errors > BAD_ERROR_DEG)),
                            "p95_all_deg": float(np.quantile(errors, 0.95)),
                            "accepted_count": int(np.count_nonzero(accepted)),
                            "bad_accepted_count": int(np.count_nonzero((errors > BAD_ERROR_DEG)
                                                                        & accepted)),
                            "gateway_rejections": result["gateway_rejections"],
                        })
    cells = []
    for motor in motors:
        for scenario, speed_fraction in profiles:
            for ratio in ratios:
                cell = {"motor": motor, "scenario": scenario,
                        "speed_fraction": speed_fraction, "actual_ratio": ratio}
                for arm in ("baseline", "candidate"):
                    group = [r for r in rows if r["motor"] == motor
                             and r["scenario"] == scenario
                             and r["speed_fraction"] == speed_fraction
                             and r["actual_ratio"] == ratio and r["arm"] == arm]
                    accepted = sum(r["accepted_count"] for r in group)
                    cell[f"{arm}_bad_all"] = float(np.mean([r["bad_all_fraction"] for r in group]))
                    cell[f"{arm}_mean_p95_deg"] = float(np.mean([r["p95_all_deg"] for r in group]))
                    cell[f"{arm}_availability"] = accepted / (len(group) * steps)
                    cell[f"{arm}_bad_accepted"] = (sum(r["bad_accepted_count"] for r in group)
                                                    / accepted if accepted else None)
                cells.append(cell)
    paired: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["seed"], row["motor"], row["scenario"], row["speed_fraction"],
               row["actual_ratio"])
        paired.setdefault(key, {})[row["arm"]] = row
    if any(set(arms) != {"baseline", "candidate"}
           or arms["baseline"]["stimulus_digest"] != arms["candidate"]["stimulus_digest"]
           for arms in paired.values()):
        raise AssertionError("paired arms must share their exogenous stimulus")
    effects = {"motion": [], "constant": []}
    for seed in range(seed_offset, seed_offset + seeds):
        for key, selected in (("motion", ("ramp", "reversal")),
                              ("constant", ("constant",))):
            differences = [arms["baseline"]["bad_all_fraction"]
                           - arms["candidate"]["bad_all_fraction"]
                           for identity, arms in paired.items()
                           if identity[0] == seed and identity[2] in selected]
            if differences:
                effects[key].append(float(np.mean(differences)))
    summary = {
        "status": "synthetic_host_only", "seed_range": [seed_offset, seed_offset + seeds - 1],
        "trial_count_per_arm": len(paired), "steps_per_trial": steps,
        "baseline_speed_diffusion": BASELINE_DIFFUSION,
        "candidate_speed_diffusion": CANDIDATE_DIFFUSION,
        "bad_error_threshold_deg": BAD_ERROR_DEG,
        "frozen_width_threshold_deg": FROZEN_WIDTH_THRESHOLD_DEG,
        "motion_bad_fraction_effect": _bootstrap(effects["motion"]),
        "constant_bad_fraction_effect": _bootstrap(effects["constant"]),
        "gateway_rejections": sum(int(r["gateway_rejections"]) for r in rows),
        "cells": cells, "motion_is_prescribed": True,
        "absolute_angle_claim": False, "hardware_claim": False,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-offset", type=int, default=34000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", type=Path, default=ROOT /
                        "experiments/info_optimal_coded_pwm/speed_diffusion_holdout")
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
        "status", "seed_range", "trial_count_per_arm", "motion_bad_fraction_effect",
        "constant_bad_fraction_effect", "gateway_rejections")}, indent=2))
    if summary["gateway_rejections"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

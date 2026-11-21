"""Fresh paired holdout for command-slew-conditioned speed diffusion."""

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

from control.command_slew_diffusion import HOLD_STEPS, SLEW_THRESHOLD_RAD_S_PER_STEP
from tools.run_info_optimal_coded_pwm_study import Case, exogenous
from tools.study_coded_pwm_confidence_gate import BAD_ERROR_DEG, _trial
from tools.study_coded_pwm_noise_miscalibration import _digest
from tools.study_coded_pwm_speed_diffusion import (
    BASELINE_DIFFUSION, CANDIDATE_DIFFUSION, FROZEN_WIDTH_THRESHOLD_DEG,
    MOTORS, PROFILES, RATIOS, _bootstrap,
)


ARMS = ("baseline", "fixed_high", "command_slew")


def run_study(*, seed_offset: int = 35000, seeds: int = 20, steps: int = 500,
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
                    for arm in ARMS:
                        result, widths = _trial(
                            case,
                            speed_diffusion_override=(CANDIDATE_DIFFUSION
                                                      if arm == "fixed_high" else None),
                            command_slew_speed_diffusion=(arm == "command_slew"),
                        )
                        period = math.radians(result["period_deg"])
                        estimate = np.asarray(result["trace_estimate_rad"])
                        truth = np.asarray(result["trace_theta_rad"])
                        errors = np.degrees(np.abs((estimate - truth + period / 2.0)
                                                    % period - period / 2.0))
                        accepted = widths <= FROZEN_WIDTH_THRESHOLD_DEG
                        rows.append({
                            "seed": seed, "motor": motor, "scenario": scenario,
                            "speed_fraction": speed_fraction, "actual_ratio": ratio,
                            "arm": arm, "stimulus_digest": digest,
                            "bad_all_fraction": float(np.mean(errors > BAD_ERROR_DEG)),
                            "p95_all_deg": float(np.quantile(errors, 0.95)),
                            "accepted_count": int(np.count_nonzero(accepted)),
                            "bad_accepted_count": int(np.count_nonzero((errors > BAD_ERROR_DEG)
                                                                        & accepted)),
                            "gateway_rejections": result["gateway_rejections"],
                        })
    paired: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["seed"], row["motor"], row["scenario"], row["speed_fraction"],
               row["actual_ratio"])
        paired.setdefault(key, {})[row["arm"]] = row
    if any(set(arms) != set(ARMS)
           or len({r["stimulus_digest"] for r in arms.values()}) != 1
           for arms in paired.values()):
        raise AssertionError("all arms must share the same exogenous stimulus")
    cells = []
    for motor in motors:
        for scenario, speed_fraction in profiles:
            for ratio in ratios:
                cell = {"motor": motor, "scenario": scenario,
                        "speed_fraction": speed_fraction, "actual_ratio": ratio}
                for arm in ARMS:
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
    effect_groups = {"motion_base_minus_schedule": [], "constant_base_minus_schedule": [],
                     "motion_fixed_minus_schedule": []}
    for seed in range(seed_offset, seed_offset + seeds):
        for name, scenarios, comparator in (
            ("motion_base_minus_schedule", ("ramp", "reversal"), "baseline"),
            ("constant_base_minus_schedule", ("constant",), "baseline"),
            ("motion_fixed_minus_schedule", ("ramp", "reversal"), "fixed_high"),
        ):
            differences = [arms[comparator]["bad_all_fraction"]
                           - arms["command_slew"]["bad_all_fraction"]
                           for identity, arms in paired.items()
                           if identity[0] == seed and identity[2] in scenarios]
            if differences:
                effect_groups[name].append(float(np.mean(differences)))
    summary = {
        "status": "synthetic_host_only", "seed_range": [seed_offset, seed_offset + seeds - 1],
        "trial_count_per_arm": len(paired), "steps_per_trial": steps,
        "arms": list(ARMS), "baseline_speed_diffusion": BASELINE_DIFFUSION,
        "fixed_high_speed_diffusion": CANDIDATE_DIFFUSION,
        "command_slew_threshold_rad_s_per_step": SLEW_THRESHOLD_RAD_S_PER_STEP,
        "command_slew_hold_steps": HOLD_STEPS,
        "bad_error_threshold_deg": BAD_ERROR_DEG,
        "frozen_width_threshold_deg": FROZEN_WIDTH_THRESHOLD_DEG,
        "effects": {name: _bootstrap(values) for name, values in effect_groups.items()},
        "gateway_rejections": sum(int(r["gateway_rejections"]) for r in rows),
        "cells": cells, "motion_is_prescribed": True,
        "absolute_angle_claim": False, "hardware_claim": False,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-offset", type=int, default=35000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", type=Path, default=ROOT /
                        "experiments/info_optimal_coded_pwm/command_slew_holdout")
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
        "status", "seed_range", "trial_count_per_arm", "effects",
        "gateway_rejections")}, indent=2))
    if summary["gateway_rejections"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

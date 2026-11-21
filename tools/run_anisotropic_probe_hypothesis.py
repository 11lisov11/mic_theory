"""Exploratory directional-noise test for coded PWM; synthetic sensor only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_info_optimal_coded_pwm_study import Case, exogenous, run_trial


ARMS = ("cycle", *(f"fixed_{vector}" for vector in range(1, 7)),
        "B3_info_separation", "B3_info_renyi2", "B3_info_fisher")
ORACLE_ARM = "B4_oracle_fisher"
RISK_ARM = "B3_info_bayes_risk"
METRICS = ("angle_p95_deg", "hpd95_coverage", "hpd95_width_deg",
           "torque_ripple_rms_nm", "switches_per_s", "compute_mean_us")


def run_experiment(*, motors: tuple[str, ...], ratio: float, scenario: str,
                   seeds: int, steps: int, seed_offset: int, out: Path,
                   arms: tuple[str, ...] = ARMS) -> dict:
    if ratio <= 0 or seeds < 1 or steps < 8:
        raise ValueError("ratio, seeds and steps must be positive")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for motor in motors:
        for seed in range(seed_offset, seed_offset + seeds):
            case = Case(motor=motor, scenario=scenario, speed_fraction=0.0 if scenario == "standstill" else 0.01,
                        strength=0.15, seed=seed, steps=steps, measurement_noise_beta_ratio=ratio)
            stimulus = exogenous(case)
            digest = hashlib.sha256(
                np.asarray((stimulus["theta0"], stimulus["speed_bias_fraction"])).tobytes()
                + stimulus["base_noise"].tobytes() + stimulus["probe_noise"].tobytes()
            ).hexdigest()[:16]
            for arm in arms:
                method = arm if arm.startswith(("B3", "B4")) else "B2_fixed_probe"
                vector = int(arm.split("_")[1]) if arm.startswith("fixed_") else None
                row = run_trial(case, method, stimulus=stimulus, observer_profile="robust",
                                fixed_probe_vector=vector)
                row.pop("trace_theta_rad")
                row.pop("trace_estimate_rad")
                row["arm"] = arm
                row["stimulus_digest"] = digest
                rows.append(row)
            print(f"{motor} seed={seed} completed", flush=True)

    summary = []
    winners = {}
    for motor in motors:
        grouped = {arm: [row for row in rows if row["motor"] == motor and row["arm"] == arm]
                   for arm in arms}
        for arm, trials in grouped.items():
            summary.append({"motor": motor, "arm": arm, "n": len(trials),
                            **{metric: float(np.mean([row[metric] for row in trials])) for metric in METRICS}})
        if set(ARMS).issubset(arms):
            winners[motor] = {
                "fixed": min((f"fixed_{i}" for i in range(1, 7)),
                             key=lambda arm: np.mean([r["angle_p95_deg"] for r in grouped[arm]])),
                "adaptive": min(("B3_info_separation", "B3_info_renyi2", "B3_info_fisher"),
                                key=lambda arm: np.mean([r["angle_p95_deg"] for r in grouped[arm]])),
            }
    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"study": "synthetic directional-noise coded-PWM hypothesis", "observer_profile": "robust",
              "motors": motors, "ratio": ratio, "scenario": scenario, "arms": arms, "seeds": seeds,
              "steps": steps, "seed_offset": seed_offset, "trial_count": len(rows),
              "winner_selection_is_exploratory": True, "winners": winners, "summary": summary,
              "limitation": "No measured AIR56/AL31 current-noise covariance or rotor saliency"}
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motors", nargs="+", choices=("AIR56", "AL31", "AO2"), default=("AIR56",))
    parser.add_argument("--ratio", type=float, default=4.0)
    parser.add_argument("--scenario", choices=("standstill", "constant"), default="standstill")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed-offset", type=int, default=12000)
    parser.add_argument("--arms", nargs="+", choices=(*ARMS, ORACLE_ARM, RISK_ARM), default=ARMS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_experiment(motors=tuple(args.motors), ratio=args.ratio, scenario=args.scenario,
                            seeds=args.seeds, steps=args.steps, seed_offset=args.seed_offset,
                            out=args.out, arms=tuple(args.arms))
    print(json.dumps({"trial_count": result["trial_count"], "winners": result["winners"]}, indent=2))


if __name__ == "__main__":
    main()

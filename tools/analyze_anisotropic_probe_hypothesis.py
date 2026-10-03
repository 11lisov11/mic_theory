"""Confirm development-selected coded-PWM arms on untouched synthetic seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyze(development: tuple[Path, ...], holdout: Path) -> dict:
    dev = [json.loads(path.read_text(encoding="utf-8")) for path in development]
    held_summary = json.loads((holdout.parent / "summary.json").read_text(encoding="utf-8"))
    if any(item["ratio"] != held_summary["ratio"] or item["scenario"] != held_summary["scenario"]
           or item["steps"] != held_summary["steps"] for item in dev):
        raise ValueError("development and holdout conditions differ")
    dev_seeds = {seed for item in dev for seed in range(item["seed_offset"], item["seed_offset"] + item["seeds"])}
    hold_seeds = set(range(held_summary["seed_offset"], held_summary["seed_offset"] + held_summary["seeds"]))
    if dev_seeds & hold_seeds:
        raise ValueError("development seeds overlap holdout")
    selection = {motor: choices for item in dev for motor, choices in item["winners"].items()}
    rows = _rows(holdout)
    output = []
    for motor in sorted(selection):
        arms = ("cycle", selection[motor]["fixed"], selection[motor]["adaptive"])
        by_seed: dict[int, dict[str, dict[str, str]]] = {}
        for row in rows:
            if row["motor"] == motor:
                seed = int(row["seed"])
                if row["arm"] in by_seed.setdefault(seed, {}):
                    raise ValueError(f"duplicate arm for {motor} seed {seed}")
                by_seed[seed][row["arm"]] = row
        if set(by_seed) != hold_seeds:
            raise ValueError(f"missing holdout seeds for {motor}")
        for seed, trial in by_seed.items():
            if any(arm not in trial for arm in arms):
                raise ValueError(f"missing selected arm for {motor} seed {seed}")
            if len({trial[arm]["stimulus_digest"] for arm in arms}) != 1:
                raise ValueError(f"stimuli differ for {motor} seed {seed}")
        for baseline in arms[:2]:
            method = arms[2]
            fixed = np.asarray([float(by_seed[s][baseline]["angle_p95_deg"]) for s in sorted(by_seed)])
            adaptive = np.asarray([float(by_seed[s][method]["angle_p95_deg"]) for s in sorted(by_seed)])
            effect = (fixed - adaptive) / np.maximum(fixed, 1e-12)
            rng = np.random.default_rng(20261004)
            boot = np.mean(rng.choice(effect, size=(5000, len(effect)), replace=True), axis=1)
            record = {"motor": motor, "baseline": baseline, "adaptive": method, "n": len(effect),
                      "paired_p95_reduction_mean": float(np.mean(effect)),
                      "bootstrap95_low": float(np.quantile(boot, 0.025)),
                      "bootstrap95_high": float(np.quantile(boot, 0.975))}
            for arm in (baseline, method):
                trials = [by_seed[s][arm] for s in sorted(by_seed)]
                for field in ("angle_p95_deg", "hpd95_coverage", "hpd95_width_deg",
                              "torque_ripple_rms_nm", "switches_per_s", "compute_mean_us",
                              "gateway_rejections"):
                    record[f"{arm}_{field}_mean"] = float(np.mean([float(t[field]) for t in trials]))
            output.append(record)
    return {"development": [str(path) for path in development], "holdout": str(holdout),
            "ratio": held_summary["ratio"], "scenario": held_summary["scenario"],
            "selection": selection, "comparisons": output,
            "scope": "synthetic measurement-channel result; no motor or sensor validation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(tuple(args.development), args.holdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"analyzed {len(result['comparisons'])} paired comparisons")


if __name__ == "__main__":
    main()

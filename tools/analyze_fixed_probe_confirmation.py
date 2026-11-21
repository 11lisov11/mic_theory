"""Paired confirmation of a preselected fixed probe against cyclic probing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> tuple[dict, list[dict[str, str]]]:
    meta = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if set(meta["arms"]) != {"cycle", "fixed_6"}:
        raise ValueError("expected preselected cycle and fixed_6 arms")
    return meta, rows


def analyze(paths: tuple[Path, ...]) -> dict:
    output = []
    for path in paths:
        meta, rows = _load(path)
        for motor in meta["motors"]:
            pairs: dict[int, dict[str, dict[str, str]]] = {}
            for row in rows:
                if row["motor"] != motor:
                    continue
                seed = int(row["seed"])
                if row["arm"] in pairs.setdefault(seed, {}):
                    raise ValueError(f"duplicate {motor} seed {seed} arm {row['arm']}")
                pairs[seed][row["arm"]] = row
            expected = set(range(meta["seed_offset"], meta["seed_offset"] + meta["seeds"]))
            if set(pairs) != expected:
                raise ValueError(f"missing seeds for {motor}")
            for seed, trial in pairs.items():
                if set(trial) != {"cycle", "fixed_6"} or trial["cycle"]["stimulus_digest"] != trial["fixed_6"]["stimulus_digest"]:
                    raise ValueError(f"unmatched stimulus for {motor} seed {seed}")
            cycle = np.asarray([float(pairs[s]["cycle"]["angle_p95_deg"]) for s in sorted(pairs)])
            fixed = np.asarray([float(pairs[s]["fixed_6"]["angle_p95_deg"]) for s in sorted(pairs)])
            effects = (cycle - fixed) / np.maximum(cycle, 1e-12)
            rng = np.random.default_rng(20261004)
            boot = np.mean(rng.choice(effects, size=(5000, len(effects)), replace=True), axis=1)
            item = {"ratio": meta["ratio"], "motor": motor, "n": len(effects),
                    "paired_p95_reduction_mean": float(np.mean(effects)),
                    "bootstrap95_low": float(np.quantile(boot, 0.025)),
                    "bootstrap95_high": float(np.quantile(boot, 0.975))}
            for arm in ("cycle", "fixed_6"):
                trials = [pairs[s][arm] for s in sorted(pairs)]
                for metric in ("angle_p95_deg", "hpd95_coverage", "hpd95_width_deg",
                               "torque_ripple_rms_nm", "switches_per_s", "compute_mean_us", "gateway_rejections"):
                    item[f"{arm}_{metric}_mean"] = float(np.mean([float(row[metric]) for row in trials]))
            output.append(item)
    return {"files": [str(path) for path in paths], "preselected_arm": "fixed_6",
            "paired_results": output, "physical_validation": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="+", required=True, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(tuple(args.trials))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"analyzed {len(result['paired_results'])} cells")


if __name__ == "__main__":
    main()

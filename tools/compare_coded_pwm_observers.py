"""Pair legacy and robust coded-PWM observer trials on identical stimuli."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


KEY_FIELDS = ("motor", "scenario", "speed_fraction", "strength", "nonideal", "mismatch", "seed", "method")
METRICS = ("angle_p95_deg", "hpd95_coverage", "hpd95_width_deg", "compute_mean_us")


def read_trials(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {tuple(row[field] for field in KEY_FIELDS): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate trial keys in {path}")
    return keyed


def compare(legacy_path: Path, robust_path: Path) -> dict:
    legacy = read_trials(legacy_path)
    robust = read_trials(robust_path)
    if legacy.keys() != robust.keys():
        raise ValueError("trial keys differ between observer profiles")
    groups: dict[tuple[str, str, str, str], list[tuple[dict, dict]]] = {}
    for key, before in legacy.items():
        after = robust[key]
        if before["stimulus_digest"] != after["stimulus_digest"]:
            raise ValueError(f"exogenous stimuli differ for {key}")
        group = (before["motor"], before["nonideal"], before["mismatch"], before["method"])
        groups.setdefault(group, []).append((before, after))
    result = []
    for (motor, nonideal, mismatch, method), pairs in sorted(groups.items()):
        entry = {"motor": motor, "nonideal": nonideal == "True", "mismatch": mismatch,
                 "method": method, "n": len(pairs)}
        for field in METRICS:
            old = np.asarray([float(before[field]) for before, _ in pairs])
            new = np.asarray([float(after[field]) for _, after in pairs])
            entry[f"{field}_legacy_mean"] = float(np.mean(old))
            entry[f"{field}_robust_mean"] = float(np.mean(new))
            entry[f"{field}_paired_delta_mean"] = float(np.mean(new - old))
            if field == "angle_p95_deg":
                effect = (old - new) / np.maximum(old, 1e-12)
                rng = np.random.default_rng(20261004)
                bootstrap = np.mean(rng.choice(effect, size=(2000, len(effect)), replace=True), axis=1)
                entry["paired_p95_reduction_mean"] = float(np.mean(effect))
                entry["bootstrap95_low"] = float(np.quantile(bootstrap, 0.025))
                entry["bootstrap95_high"] = float(np.quantile(bootstrap, 0.975))
        result.append(entry)
    return {"legacy": str(legacy_path), "robust": str(robust_path), "matched_trials": len(legacy),
            "groups": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--robust", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.legacy, args.robust)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"matched {result['matched_trials']} trials in {len(result['groups'])} groups")


if __name__ == "__main__":
    main()

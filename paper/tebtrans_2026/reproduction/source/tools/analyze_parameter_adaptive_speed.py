"""Validate paired speed-control evidence; exploratory cohorts are not confirmation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(result):
    keys = [(r["seed"], r["scenario"], r["method"]) for r in result["rows"]]
    expected = {(s, c, m) for s in result["seeds"] for c in result["cases"] for m in result["methods"]}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("missing or duplicate arm")
    if any(r["failure"] is not None or r["completed_steps"] != round(r["duration_s"]/result["settings"]["dt_s"])
           for r in result["rows"]):
        raise ValueError("failed/incomplete trial; cannot summarize as successful")
    metrics = ("tracking_mae_rad_s", "estimation_mae_rad_s", "peak_current_a", "copper_energy_j",
               "steady_torque_std_nm", "steady_current_std_a")
    if any(r[k] is None or not np.isfinite(r[k]) or r[k] < 0 for r in result["rows"] for k in metrics):
        raise ValueError("invalid score")
    aggregates, comparisons = [], []
    lookup = {key: row for key, row in zip(keys, result["rows"])}
    for seed in result["seeds"]:
        for case in result["cases"]:
            paired = [lookup[(seed, case, m)] for m in result["methods"]]
            signatures = [json.dumps({k: row[k] for k in (
                "actual_initial_parameters", "load_start_s", "load_level_nm")}, sort_keys=True) for row in paired]
            if len(set(signatures)) != 1:
                raise ValueError("unequal exogenous plant/load conditions")
    rng = np.random.default_rng(102314)
    draws = rng.integers(0, len(result["seeds"]), (10000, len(result["seeds"])))
    for case in result["cases"]:
        arms = {}
        for method in result["methods"]:
            group = [lookup[(seed, case, method)] for seed in result["seeds"]]
            aggregate = dict(scenario=case, method=method,
                **{k: float(np.mean([r[k] for r in group])) for k in metrics},
                current_limit_violations=sum(r["current_limit_violated"] for r in group),
                censored_recoveries=sum(r["recovery_s"] is None for r in group),
                bound_hits=sum(r["bound_hits"] for r in group),
                runtime_s=float(np.mean([r["runtime_s"] for r in group])))
            aggregates.append(aggregate)
            arms[method] = aggregate
        fixed = min((m for m in arms if m.startswith("fixed_")), key=lambda m: arms[m]["tracking_mae_rad_s"])
        for method in arms:
            if method.startswith("fixed_"):
                continue
            candidate = np.array([lookup[(s, case, method)]["tracking_mae_rad_s"] for s in result["seeds"]])
            baseline = np.array([lookup[(s, case, fixed)]["tracking_mae_rad_s"] for s in result["seeds"]])
            delta = candidate-baseline
            comparisons.append(dict(scenario=case, candidate=method, baseline=fixed,
                reduction_percent=100*(1-candidate.mean()/baseline.mean()),
                exploratory_difference_ci95=np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                peak_increase_percent=100*(arms[method]["peak_current_a"]/arms[fixed]["peak_current_a"]-1),
                copper_change_percent=100*(arms[method]["copper_energy_j"]/arms[fixed]["copper_energy_j"]-1),
                candidate_current_limit_violations=arms[method]["current_limit_violations"]))
    return dict(aggregates=aggregates, comparisons=comparisons, seed_clusters=len(result["seeds"]),
                verdict="EXPLORATORY_ONLY", baseline_selection="best fixed Q per scenario on THIS cohort; not independent confirmation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    path = args.directory/"results.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    analysis = analyze(result)
    analysis["provenance"] = dict(raw_results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_match=all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in result["source_sha256"].items()))
    (args.directory/"analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    with (args.directory/"aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis["aggregates"][0]))
        writer.writeheader()
        writer.writerows(analysis["aggregates"])
    lines = ["# Parameter-adaptive speed control", "", "EXPLORATORY ONLY: training/pilot, no confirmation claim.",
             f"Seed clusters: {analysis['seed_clusters']}. All arms complete with no numerical failures.", "",
             "| Scenario | Candidate | Strong fixed Q | MAE reduction % | Peak increase % | Copper change % | >2.8 A trials |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in analysis["comparisons"]:
        lines.append(f"| {r['scenario']} | {r['candidate']} | {r['baseline']} | {r['reduction_percent']:.2f} | "
                     f"{r['peak_increase_percent']:.2f} | {r['copper_change_percent']:.2f} | {r['candidate_current_limit_violations']} |")
    lines += ["", "Fixed selection and exploratory intervals are not a prespecified holdout superiority test.",
              "No novelty, hardware readiness, switching-level safety or parameter-identifiability proof.",
              "Mean peak and current-command limits are not a bound on every measured-current sample."]
    (args.directory/"results_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(analysis["comparisons"]))


if __name__ == "__main__":
    main()

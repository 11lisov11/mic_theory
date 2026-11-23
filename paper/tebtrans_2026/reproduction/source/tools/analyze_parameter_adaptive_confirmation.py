"""Prespecified paired-cluster analysis and monochrome closed-loop figures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.analyze_parameter_adaptive_speed import analyze as validate


def compare(rows, seeds, case, candidate, baseline, draws):
    lookup = {(r["seed"], r["scenario"], r["method"]): r for r in rows}
    values = {}
    for metric in ("tracking_mae_rad_s", "peak_current_a", "copper_energy_j", "steady_torque_std_nm", "steady_current_std_a"):
        a = np.array([lookup[(s, case, candidate)][metric] for s in seeds])
        b = np.array([lookup[(s, case, baseline)][metric] for s in seeds])
        boot = a[draws].mean(axis=1)/np.maximum(b[draws].mean(axis=1), 1e-12)
        values[metric] = dict(candidate_mean=float(a.mean()), baseline_mean=float(b.mean()),
            ratio=float(a.mean()/max(b.mean(), 1e-12)), ratio_ci975=np.quantile(boot, [.0125, .9875]).tolist())
    return dict(scenario=case, candidate=candidate, baseline=baseline, metrics=values)


def analyze(result):
    checked = validate(result)
    manifest = result["selection"]
    if result["seeds"] != manifest["holdout_seeds"] or result["cases"] != list(manifest["selection"]):
        raise ValueError("holdout contract changed")
    if manifest["confidence"] != .975 or manifest["bootstrap_draws"] != 10000:
        raise ValueError("statistical contract changed")
    for row in result["rows"]:
        expected = manifest["selection"][row["scenario"]][row["method"]]
        if row["base_method"] != expected or row["torque_limit_nm"] != .6:
            raise ValueError("controller configuration changed")
    rng = np.random.default_rng(manifest["bootstrap_seed"])
    draws = rng.integers(0, len(result["seeds"]), (10000, len(result["seeds"])))
    comparisons = [compare(result["rows"], result["seeds"], case, candidate, "fixed_reference", draws)
        for case in result["cases"] for candidate in ("joint_only", "direct_feedback", "confidence_feedback")]
    primary = []
    for row in comparisons:
        if row["scenario"] in manifest["primary_cases"] and row["candidate"] == manifest["primary_candidate"]:
            mae, peak = row["metrics"]["tracking_mae_rad_s"], row["metrics"]["peak_current_a"]
            group = [r for r in result["rows"] if r["scenario"] == row["scenario"] and r["method"] == row["candidate"]]
            primary.append(dict(**row, statistical_mae_pass=mae["ratio_ci975"][1] < 1,
                observed_practical_pass=mae["ratio"] <= .8 and mae["ratio_ci975"][1] < 1 and peak["ratio"] <= 1.05
                    and not any(r["current_limit_violated"] for r in group),
                peak_noninferiority_pass=peak["ratio_ci975"][1] <= 1.05))
    adaptive_secondary = [compare(result["rows"], result["seeds"], case, "confidence_feedback", "direct_feedback", draws)
                           for case in result["cases"]]
    return dict(primary=primary, all_vs_fixed=comparisons, adaptive_secondary=adaptive_secondary,
        aggregates=checked["aggregates"], seed_clusters=len(result["seeds"]), verdict="MIXED / NICHE",
        contract="two prespecified degraded-case endpoints; other comparisons exploratory; no novelty/hardware claim")


def plot(result, analysis, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(left=.075, right=.985, bottom=.2, top=.76, wspace=.28)
    cases, methods = result["cases"], result["methods"]
    labels = ["Fixed (tuned Q)", "Joint EKF", "Direct coupling", "Confidence coupling"]
    colors, hatches = ["white", ".7", ".4", ".15"], ["//", "", "..", ""]
    for ax, metric, ylabel in zip(axes, ["tracking_mae_rad_s", "peak_current_a"], ["Speed MAE (rad/s)", "Mean trial peak current (A)"]):
        for j, method in enumerate(methods):
            y = [next(r[metric] for r in analysis["aggregates"] if r["scenario"] == c and r["method"] == method) for c in cases]
            ax.bar(np.arange(len(cases))+(j-1.5)*.2, y, width=.19, color=colors[j], edgecolor="black", hatch=hatches[j], label=labels[j])
        ax.set_xticks(range(len(cases)), ["Nominal", "Parameter\ndrift", "Composite", "Slow\ncomposite"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .9), ncol=4, frameon=False)
    fig.suptitle("Current-only closed-loop control: common 0.6 Nm envelope", y=.98)
    fig.savefig(directory/"confirmation.png", dpi=180)
    fig.savefig(directory/"confirmation.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    path = args.directory/"results.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    analysis = analyze(result)
    analysis["provenance"] = dict(raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_match=all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in result["source_sha256"].items()),
        training_match=hashlib.sha256(Path(result["selection"]["training_path"]).read_bytes()).hexdigest() == result["selection"]["training_sha256"])
    (args.directory/"analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    with (args.directory/"aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis["aggregates"][0]))
        writer.writeheader()
        writer.writerows(analysis["aggregates"])
    lines = ["# Closed-loop parameter-adaptive confirmation", "", "MIXED / NICHE; no new-law or hardware claim.",
        f"{len(result['rows'])} complete trials; {analysis['seed_clusters']} paired seed clusters; no numerical failures.",
        "", "| Case | Candidate | MAE change % | Peak change % | Copper change % | Torque SD change % |",
        "|---|---|---:|---:|---:|---:|"]
    for r in analysis["all_vs_fixed"]:
        metrics = r["metrics"]
        changes = [100*(metrics[k]["ratio"]-1) for k in ("tracking_mae_rad_s", "peak_current_a", "copper_energy_j", "steady_torque_std_nm")]
        lines.append(f"| {r['scenario']} | {r['candidate']} | "+" | ".join(f"{v:.2f}" for v in changes)+" |")
    lines += ["", "Primary degraded-case comparisons (Bonferroni CI97.5):", "```json", json.dumps(analysis["primary"], indent=2), "```",
        "", "Full adaptive comparisons, peaks, recovery censoring and runtimes are in JSON/CSV.",
        "Improvement over a fixed model does not establish novelty. Contraction is not proof of identification.",
        "Averaged linear plant and F2 fitted prior; no absolute angle, switching-level or hardware validation."]
    (args.directory/"results_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    plot(result, analysis, args.directory)
    print(json.dumps(analysis["primary"]))


if __name__ == "__main__":
    main()

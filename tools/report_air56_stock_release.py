"""Export paired, geometry-stratified results of the plain-cage release experiments."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np


def write_report(root, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(exist_ok=True, parents=True)
    score = json.loads((root / "scores.json").read_text())
    result = {
        "initial_limits": [],
        "mechanical_gate": [],
        "longer_runs": [],
        "detector": [],
        "single_seed_pulse": [],
        "skew_check": {},
    }
    result["tolerance_screen"] = [
        {k: v for k, v in row.items() if k not in ("real_a_per_v", "imag_a_per_v", "half_turn_ua_per_v")}
        for path in sorted(root.glob("tolerance_L*.json"))
        for row in json.loads(path.read_text())
    ]
    for probe in sorted({r["probe"] for r in score["initial"]}):
        for noise in (0.3, 1.0, 3.0):
            rows = [r for r in score["initial"] if r["probe"] == probe and r["noise_ma"] == noise]
            ps = [r["optimal_binary_accuracy"] for r in rows]
            result["initial_limits"].append(
                dict(
                    probe=probe,
                    noise_ma=noise,
                    comparisons=len(rows),
                    min_p=min(ps),
                    max_p=max(ps),
                    min_window_ms=min(r["samples"] for r in rows),
                    max_window_ms=max(r["samples"] for r in rows),
                )
            )
    for source in ("brake_noise", "brake_fresh"):
        for gate in ("", "gate"):
            folder = root / source / gate
            records = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
            expected = 12 if source == "brake_noise" else 8
            if len(records) != expected:
                raise ValueError(f"{folder}: {len(records)}/{expected} completed")
            for slices in (1, 24):
                cases = [r for r in records if r["slices"] == slices]
                for noise in (0.3, 1.0, 3.0):
                    rows = [item for r in cases for item in r["results"] if item["noise_ma"] == noise]
                    result["mechanical_gate"].append(
                        dict(
                            sample=source,
                            gate_deg=0.4 if gate else 0.12,
                            slices=slices,
                            noise_ma=noise,
                            physical_cases=len(cases),
                            total=sum(r["trials"] for r in rows),
                            pass_count=sum(r["pass_count"] for r in rows),
                            passive_pass=sum(r["passive_deg"] <= 1 for r in cases),
                            worst_p95_deg=max(r["p95_peak_deg"] for r in rows),
                            worst_case_pass=min(r["pass_count"] / r["trials"] for r in rows),
                        )
                    )
    for locked in (False, True):
        for noise in (0.3, 1.0, 3.0):
            for slices in (1, 24):
                rows = [
                    r
                    for r in score["rollback"]
                    if r["level"] == 5
                    and r["locked"] == locked
                    and r["noise_ma"] == noise
                    and r["probe"] == "release30"
                    and (
                        (r["axial_slices"] == 24)
                        if slices == 24
                        else (r["axial_slices"] is None and "_s0_" in r["record"])
                    )
                ]
                active = [r for r in rows if r["exited"]]
                result["detector"].append(
                    dict(
                        locked=locked,
                        noise_ma=noise,
                        slices=slices,
                        cases=len(rows),
                        exit_cases=len(active),
                        correct_before_exit=np.mean([r["detection_before_exit"] for r in active])
                        if active
                        else None,
                        false_early=np.mean([r["false_early"] for r in rows]) if rows else None,
                        median_delay_range_ms=[
                            min(r["median_correct_ms"] for r in active),
                            max(r["median_correct_ms"] for r in active),
                        ]
                        if active
                        else None,
                    )
                )
    for folder in ("brake_holdout/unipolar", "brake_holdout/ms100/unipolar"):
        for p in sorted((root / folder).glob("*_v24_*")):
            baseline = p.with_name(p.name.replace("_v24_", "_v0_"))
            with np.load(p) as a, np.load(baseline) as b:
                x = a["trace"]
                y = b["trace"]
                result["single_seed_pulse"].append(
                    dict(
                        record=p.relative_to(root).as_posix(),
                        period_ms=len(x),
                        slices=int(a["slices"]),
                        load_nm=float(a["load_nm"]),
                        pulse_peak_deg=float(x[:, 5].max()),
                        passive_peak_deg=float(y[:, 5].max()),
                        event_ms=int(a["event_ms"]),
                    )
                )
    for folder in ("relative", "relative/witness", "relative/support/witness"):
        for p in sorted((root / folder).glob("*.npz")):
            with np.load(p) as a:
                x = a["trace"]
                result["longer_runs"].append(
                    dict(
                        record=p.relative_to(root).as_posix(),
                        path=folder,
                        time_ms=len(x),
                        peak_deg=float(x[:, 10].max()),
                        final_deg=float(x[-1, 1]),
                        max_read_error_deg=float(np.max(abs(x[:, 1] - x[:, 2]))),
                    )
                )
    result["skew_check"] = json.loads((root / "axial_L5_n288.json").read_text())["rows"]
    kinetic = [json.loads(p.read_text()) for p in sorted((root / "kinematic_holdout").glob("*.json"))]
    if len(kinetic) != 8:
        raise ValueError(f"Only {len(kinetic)}/8 final impulse-policy conditions")
    result["fresh_impulse_rules"] = []
    for n in (1, 24):
        group = [r for r in kinetic if r["slices"] == n]
        for noise in (0.3, 1.0, 3.0):
            for scheme in ("read", "distance", "speed"):
                scores = [
                    x for r in group for x in r["scores"] if x["noise_ma"] == noise and x["rule"] == scheme
                ]
                result["fresh_impulse_rules"].append(
                    dict(
                        slices=n,
                        noise_ma=noise,
                        rule=scheme,
                        cases=len(group),
                        correct=sum(x["passed"] for x in scores),
                        total=sum(x["replays"] for x in scores),
                        worst_case_correct=min(x["passed"] / x["replays"] for x in scores),
                        worst_p95_deg=max(x["p95_deg"] for x in scores),
                    )
                )
    for folder in ("relative", "relative/witness", "relative/support/witness"):
        assert sum(r["path"] == folder for r in result["longer_runs"]) == 8, folder
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for j, slices in enumerate((1, 24)):
        ax = axes[j]
        for gate, label in ((0.12, "Импульс по порогу чтения"), (0.4, "Порог действия 0,40°")):
            rows = [
                r
                for r in result["mechanical_gate"]
                if r["sample"] == "brake_fresh" and r["slices"] == slices and r["gate_deg"] == gate
            ]
            ax.plot(
                [r["noise_ma"] for r in rows],
                [100 * r["pass_count"] / r["total"] for r in rows],
                marker="o",
                label=label,
            )
        ax.set_title("Прямой ротор" if slices == 1 else "Скос на один шаг, 24 сечения")
        ax.set_ylim(-2, 103)
        ax.set_xlabel("Шум тока, мА RMS")
        ax.set_xticks([0.3, 1, 3])
    axes[0].set_ylabel("Доля в пределах 1° за 24 мс, %")
    axes[0].legend(fontsize=8, loc="lower right")
    for lvl in (3, 4, 5):
        rows = json.loads((root / f"axial_L{lvl}.json").read_text())["rows"]
        axes[2].semilogy(
            [r["slices"] for r in rows],
            [1000 * r["maximum_abs_nm"] for r in rows],
            "o-",
            label=f"Сетка {lvl}",
        )
    axes[2].axhline(20, ls="--", lw=0.8, color="black", label="Внешний момент")
    axes[2].set_title("Неподвижный вал, ток 0,60 А")
    axes[2].set_xlabel("Число осевых сечений")
    axes[2].set_ylabel("Максимум |M|, мН·м")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.savefig(out / "validation.png", dpi=160)
    plt.close(fig)
    archive = root / "stock_release_records.zip"
    paths = sorted([*root.rglob("*.npz"), *root.rglob("*.json")])
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.relative_to(root).as_posix())
    (out / "archive.json").write_text(
        json.dumps(
            dict(
                name=archive.name, files=len(paths), sha256=hashlib.sha256(archive.read_bytes()).hexdigest()
            ),
            indent=2,
        ),
        encoding="utf8",
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    write_report(a.root, a.out)

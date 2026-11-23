"""Summarize independently measured capture, servo trajectories, and start-map holdouts."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.score_air56_terminal_servo import score
from tools.report_air56_terminal_servo import write


def servo(out):
    captures, loops = [], []
    recipes = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True, sharey="row")
    for path in sorted(out.glob("*.npz")):
        if path.name.startswith(("cold_", "map_")) or path.name.endswith(".partial.npz"):
            continue
        with np.load(path) as archive:
            options = json.loads(str(archive["settings"]))
            # The two early substep trials reconstructed ADC currents along a
            # single chord. Their repeat with the final ADC model is timing1_0/1.
            if options.get("plant_substeps", 1) > 1 and "observer_substeps" not in options:
                continue
            recipes[path.stem] = {k: v for k, v in options.items() if k not in ("out", "name")}
            if recipes[path.stem].get("capture_bank"):
                recipes[path.stem]["capture_bank"] = (
                    "filtered" if Path(options["capture_bank"]).name == "filtered" else "."
                )
            burst, fit, trace = (archive[key] for key in ("startup_trace", "capture_estimate", "trace"))
            load = options["load"] if options["preloaded"] else 0.0
            assert np.all(burst[:, 3] == load)
            if options["preloaded"] and len(trace):
                assert np.all(trace[:, 7] == load)
            row = dict(
                case=path.stem,
                skew=int(options["skew"]),
                start_deg=options["initial"],
                load_nm=load,
                capture_ms=1000 * burst[-1, 0],
                capture_status=str(archive.get("capture_status", "unchecked")),
                post_capture_steps=len(trace),
                requested_post_s=options["seconds"],
                recorded_post_s=trace[-1, 0] if len(trace) else 0.0,
                post_complete=bool(
                    len(trace) and abs(trace[-1, 0] - options["seconds"]) <= options["h"] * 0.51
                ),
                plant_level=options["level"],
                selected_level=int(
                    archive.get(
                        "selected_observer_level", options.get("observer_level") or options["level"]
                    )
                ),
                move_deg=np.rad2deg(burst[-1, 1] - burst[0, 1]),
                max_abs_move_deg=np.rad2deg(max(abs(burst[:, 1] - burst[0, 1]))),
                actual_angle_deg=np.rad2deg(burst[-1, 1]),
                estimated_angle_deg=np.rad2deg(fit[0]),
                read_error_deg=np.rad2deg(np.angle(np.exp(1j * (fit[0] - burst[-1, 1])))),
                actual_rpm=burst[-1, 2] * 30 / np.pi,
                estimated_rpm=fit[1] * 30 / np.pi,
                load_estimate_nm=fit[2],
                peak_probe_a=float(archive["startup_peak_a"]),
                polarity_margin=float(archive["polarity_margin"]),
                reduced_cost=float(archive.get("capture_reduced_cost", np.nan)),
            )
            captures.append(row)
            if len(trace) and path.stem.startswith(("bank_", "time_servo", "timing")):
                settled_slice = trace[(trace[:, 0] >= 0.55) & (trace[:, 0] <= 0.8)]
                if len(settled_slice):
                    row["post_enable_max_track_deg"] = np.max(
                        abs(np.rad2deg(settled_slice[:, 1] - settled_slice[:, 3]))
                    )
                    angle_error = np.angle(np.exp(1j * (settled_slice[:, 2] - settled_slice[:, 1])))
                    row["post_enable_max_read_deg"] = np.max(np.abs(np.rad2deg(angle_error)))
            if path.stem not in (
                "preloaded_pwm0",
                "preloaded_pwm1",
                "bank_fine0",
                "bank_fine1",
            ) or not len(trace):
                continue
            m = score(trace, seconds=options["seconds"])
            if not m["complete"]:
                raise ValueError(f"incomplete trajectory {path.name}")
            j = row["skew"]
            t, shaft, read, ref = trace.T[:4]
            first = t < 1.0
            good = abs(np.rad2deg(shaft[first] - ref[first])) <= 1
            settled = np.logical_and.accumulate(good[::-1])[::-1]
            metrics = dict(
                case=path.stem,
                duration_s=t[-1],
                startup_ms=row["capture_ms"],
                initial_target_settle_s=t[first][settled][0] + burst[-1, 0] if settled.any() else None,
                worst_rollback_deg=max(
                    0.0, options["initial"] - np.rad2deg(min(burst[:, 1].min(), trace[first, 17].min()))
                ),
                **m,
            )
            loops.append(metrics)
            if path.stem.startswith("bank_fine"):
                continue
            axes[0, j].plot(burst[:, 0] * 1000, np.rad2deg(burst[:, 1] - burst[0, 1]), label="Вал FEM")
            axes[0, j].scatter(
                row["capture_ms"],
                row["move_deg"] + row["read_error_deg"],
                color="tab:orange",
                marker="x",
                label="Оценка по токам",
            )
            axes[0, j].set(
                title="Без скоса" if j == 0 else "Скос в один шаг",
                xlabel="От первого импульса, мс",
                ylabel="Смещение от пуска, °",
            )
            axes[1, j].plot(t, np.rad2deg(shaft), label="Вал FEM")
            axes[1, j].plot(t, np.rad2deg(read), lw=0.8, label="По токам")
            axes[1, j].plot(t, np.rad2deg(ref), "k--", lw=1, label="Задание")
            axes[1, j].set(xlabel="После захвата, с", ylabel="Угол, °")
    if not {"preloaded_pwm0", "preloaded_pwm1"}.issubset({row["case"] for row in loops}):
        raise ValueError("both free-shaft trajectories are required")
    for ax in axes.flat:
        ax.grid(alpha=0.24)
    axes[0, 0].legend()
    axes[1, 0].legend()
    fig.savefig(out / "preloaded_servo.png", dpi=150)
    plt.close(fig)
    write(out / "capture_cases.csv", captures)
    write(out / "preloaded_loops.csv", loops)
    (out / "cases.json").write_text(json.dumps(recipes, indent=2))


def replay(out, case):
    recipes = Path(__file__).with_name("reference") / "preloaded_start" / "cases.json"
    settings = json.loads(recipes.read_text())[case]
    if (out / (case + ".npz")).exists():
        raise FileExistsError("Choose a fresh output directory for this trial")
    if settings.get("capture_bank"):
        from research.air56_rotor_fingerprinting.preloaded_native import library

        bank_dir = out / settings["capture_bank"]
        bank_dir.mkdir(parents=True, exist_ok=True)
        stem = "capture_bank_skew" + str(int(settings["skew"]))
        for suffix in (".bin", ".json"):
            target = bank_dir / (stem + suffix)
            if not target.exists():
                shutil.copyfile(recipes.parent / settings["capture_bank"] / target.name, target)
        if not (bank_dir / ("capture_lookup.dll" if os.name == "nt" else "capture_lookup.so")).exists():
            library(bank_dir)
    flags = []
    for key, value in settings.items():
        if value is None or value is False:
            continue
        flags.append("--" + key.replace("_", "-"))
        if value is not True:
            flags.append(str((out / value).resolve() if key == "capture_bank" else value))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_air56_terminal_servo.py"),
            "--out",
            str(out.resolve()),
            "--name",
            case,
            *flags,
        ],
        check=True,
        cwd=ROOT,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
    )


def map_summary(out):
    rows = []
    for filename in ("map_scores.csv", "native_scores.csv", "filtered_scores.csv"):
        with (out / filename).open(encoding="utf8") as stream:
            data = list(csv.DictReader(stream))
        if filename != "map_scores.csv":
            families = [
                ("blind_test", "Сетка 3"),
                ("blind_transfer", "Сетка 4"),
                ("blind_time_holdout", "Сетка 4, два подшага"),
            ]
            if filename == "filtered_scores.csv":
                families.append(("deep_test", "Новые углы, четыре подшага"))
            fig, axes = plt.subplots(
                len(families),
                2,
                figsize=(10, 2.35 * len(families)),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            for i, (family, title) in enumerate(families):
                for skew in (0, 1):
                    panel = axes[i, skew]
                    subset = [
                        r
                        for r in data
                        if r["kind"] == family and int(r["skew"]) == skew and float(r["noise_ma"]) == 3.0
                    ]
                    for valid, label, color in (
                        ("True", "Захват принят", "tab:blue"),
                        ("False", "Захват отклонён", "tab:red"),
                    ):
                        group = [r for r in subset if r["accepted"] == valid]
                        panel.scatter(
                            [float(r["initial_deg"]) for r in group],
                            [max(0.001, abs(float(r["error_deg"]))) for r in group],
                            s=10,
                            alpha=0.55,
                            label=label,
                            color=color,
                        )
                    panel.axhline(1.0, color="black", ls="--", lw=0.8)
                    panel.set_yscale("log")
                    panel.set_title(title + (" · без скоса" if skew == 0 else " · один шаг"))
                    panel.grid(alpha=0.17)
            axes[0, 0].legend()
            fig.supxlabel("Механический угол перед пуском, °")
            fig.supylabel("Абсолютная ошибка угла в конце 11 мс, °")
            fig.savefig(
                out
                / ("filtered_capture.png" if filename == "filtered_scores.csv" else "native_capture.png"),
                dpi=150,
            )
            plt.close(fig)
        groups = sorted({(r["kind"], r["skew"], r["noise_ma"]) for r in data})
        for kind, skew, noise in groups:
            group = [r for r in data if (r["kind"], r["skew"], r["noise_ma"]) == (kind, skew, noise)]
            errors = np.array([abs(float(r["error_deg"])) for r in group])
            accepted = np.array([r.get("accepted") == "True" for r in group])
            elapsed = [float(r["compute_ms"]) for r in group]
            row = dict(
                reader=filename,
                kind=kind,
                skew=skew,
                noise_ma=noise,
                physical_starts=len({(r["initial_deg"], r["load_nm"]) for r in group}),
                noise_trials=len(group),
                p95_deg=np.percentile(errors, 95),
                max_deg=errors.max(),
                halfturn_errors=int(np.sum(errors >= 90)),
                within_1deg=int(np.sum(errors <= 1)),
                median_compute_ms=np.median(elapsed),
                p95_compute_ms=np.percentile(elapsed, 95),
                load_sign_correct=sum(
                    np.sign(float(r["estimated_load_nm"])) == np.sign(float(r["load_nm"])) for r in group
                ),
            )
            if filename != "map_scores.csv":
                row.update(
                    accepted=int(accepted.sum()),
                    accepted_over_1deg=int(np.sum(accepted & (errors > 1))),
                    accepted_halfturn_errors=int(np.sum(accepted & (errors >= 90))),
                    accepted_p95_deg=np.percentile(errors[accepted], 95) if accepted.any() else None,
                    accepted_max_deg=errors[accepted].max() if accepted.any() else None,
                )
            rows.append(row)
    write(out / "lookup_summary.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay", help="Repeat a named case from the reference configuration")
    options = parser.parse_args()
    out = options.out
    if options.replay:
        replay(out, options.replay)
        sys.exit(0)
    servo(out)
    map_summary(out)
    manifest = {
        p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "sha256.json" and not p.name.endswith(".partial.npz")
    }
    (out / "sha256.json").write_text(json.dumps(manifest, indent=2))

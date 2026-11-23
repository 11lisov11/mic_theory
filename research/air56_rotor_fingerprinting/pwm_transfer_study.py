"""Repeat or score the independent PWM torque-transfer experiments (simulation only)."""

# ruff: noqa: E402
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.score_air56_terminal_servo import score
from tools.report_air56_terminal_servo import write


def run(case, skew, out, seconds):
    from research.air56_rotor_fingerprinting.preloaded_native import library

    source = Path(__file__).with_name("reference") / "preloaded_start"
    settings = json.loads((source / "cases.json").read_text())["timing8_" + str(skew)]
    bank = out / "calibration" / f"{case}{skew}"
    bank.mkdir(parents=True, exist_ok=True)
    for path in (source / "filtered").glob(f"capture_bank_skew{skew}.*"):
        dest = bank / path.name
        if not dest.exists():
            shutil.copyfile(path, dest)
        if dest.read_bytes() != path.read_bytes():
            raise ValueError(f"Calibration differs from versioned data: {dest}")
    if not (bank / ("capture_lookup.dll" if os.name == "nt" else "capture_lookup.so")).exists():
        library(bank)
    duration = 0.8 if case in ("matched_reference", "unadjusted", "identified", "holdout6_") else 12.5
    settings.update(
        capture_bank=str(bank.resolve()),
        seconds=duration if seconds is None else seconds,
        observer_substeps=4 if case == "matched_reference" else 3,
        model_current_rms=0.002,
        load_process=0.0004,
        identify_dwell=case in ("identified", "tracked", "quiet", "parallel", "holdout6_"),
        track_dwell=case in ("tracked", "quiet", "parallel", "holdout6_"),
    )
    if case in ("quiet", "parallel", "holdout6_"):
        settings.update(probe_v=0.6, stiffness=0.70, seed=8271)
    if case == "parallel":
        settings.update(parallel_dwells=True, seed=19013)
    if case == "holdout6_":
        settings.update(plant_substeps=6, seed=10037)
    target = out / f"{case}{skew}.npz"
    if target.exists():
        raise FileExistsError(target)
    flags = []
    for key, value in settings.items():
        if value is False or value is None:
            continue
        flags.append("--" + key.replace("_", "-"))
        if value is not True:
            flags.append(str(value))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_air56_terminal_servo.py"),
            "--out",
            str(out),
            "--name",
            target.stem,
            *flags,
        ],
        cwd=ROOT,
        check=True,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
    )


def metrics(path):
    with np.load(path) as archive:
        opts = json.loads(str(archive["settings"]))
        x = archive["trace"]
        if not len(x):
            raise ValueError(f"Capture failed: {path.name}")
        assert not opts["oracle_diagnostic"] and opts["diagnostic_prior"] is None
        assert np.all(x[:, 7] == opts["load"])
        assert np.all(archive["startup_trace"][:, 3] == opts["load"])
        assert opts["pwm"] == 1 and opts["inside_pwm"] and opts["preloaded"]
        assert opts["plant_substeps"] != opts["observer_substeps"] or path.stem.startswith(
            "matched_reference"
        )
        t = x[:, 0]
        early = x[(t >= 0.55) & (t <= 0.8000001)]
        if not len(early):
            return None

        def max_hold(a):
            return np.rad2deg(np.max(np.abs(a[:, 17:19] - a[:, 3, None])))

        def rms(a):
            return float(np.sqrt(np.mean(a * a)))

        engage = x[t <= 0.8]
        in_band = np.max(np.abs(np.rad2deg(engage[:, 17:19] - engage[:, 3, None])), axis=1) <= 1.0
        remains_in_band = np.logical_and.accumulate(in_band[::-1])[::-1]
        q = archive.get("dwell_coordinate", [])
        row = dict(
            case=path.stem,
            skew=int(opts["skew"]),
            noise_ma=opts["noise"] * 1000,
            seed=opts["seed"],
            time_s=t[-1],
            response_from_start=archive["identified_inverse_steps"].item(),
            response_at_end=q[-1] if len(q) else np.nan,
            start_current_rms_ma=archive["start_fit_rms_a"].item() * 1000,
            early_hold_max_deg=max_hold(early),
            settle_s_after_first_pulse=engage[remains_in_band, 0][0] + archive["startup_trace"][-1, 0]
            if np.any(remains_in_band) and t[-1] >= 0.8
            else np.nan,
            early_torque_rms_mnm=rms(early[:, 9] - early[:, 10]) * 1000,
            early_load_mean_nm=np.mean(early[:, 8]),
        )
        m = score(x, seconds=12.5)
        if m["complete"]:
            held = x[t >= 9.95]
            row.update(
                m,
                final_hold_max_deg=max_hold(held),
                final_hold_end_speed_p95_rpm=np.percentile(np.abs(held[:, 4]) * 30 / np.pi, 95),
                forward_end_speed_p95_error_rpm=np.percentile(
                    np.abs(x[(t >= 1.65) & (t <= 5.18), 4] * 30 / np.pi - 0.1), 95
                ),
                reverse_end_speed_p95_error_rpm=np.percentile(
                    np.abs(x[(t >= 5.85) & (t <= 9.38), 4] * 30 / np.pi + 0.1), 95
                ),
                moving_max_track_deg=np.max(abs(np.rad2deg(x[t >= 1.65, 1] - x[t >= 1.65, 3]))),
                final_torque_rms_mnm=rms(held[:, 9] - held[:, 10]) * 1000,
            )
            row["strict_pass"] = (
                bool(m["passes"]) and row["final_hold_max_deg"] <= 1.0 and row["early_hold_max_deg"] <= 1.0
            )
        return row


def report(out, primary=None):
    rows = [metrics(p) for p in sorted(out.glob("*.npz")) if not p.name.endswith(".partial.npz")]
    write(out / "control.csv", [r for r in rows if r])
    rows = []
    for p in sorted((out / "replay").glob("*_diagnostic.csv")):
        with p.open() as stream:
            data = list(csv.DictReader(stream))
        records = [r for r in data if r["dwell"] == "-1"]
        for rate in sorted({r["rate"] for r in records}):
            selection = [r for r in records if r["rate"] == rate]
            errors = np.array(
                [[float(r[k]) for k in ("error_nm", "di_alpha_a", "di_beta_a")] for r in selection]
            )
            contributions = {}
            for r in data:
                if r["rate"] != rate or r["dwell"] == "-1":
                    continue
                key = (r["period"], int(r["active"]))
                contributions[key] = (
                    contributions.get(key, 0.0) + float(r["error_nm"]) * float(r["seconds"]) / 0.004
                )

            def portion(active):
                values = [v for (_, a), v in contributions.items() if a == active]
                return 1000 * np.sqrt(np.mean(np.square(values))) if values else None

            rows.append(
                dict(
                    source=p.stem,
                    zero_vector_contribution_rms_mnm=portion(0),
                    active_vector_contribution_rms_mnm=portion(1),
                    rate=rate,
                    periods=len(errors),
                    torque_rms_mnm=1000 * np.sqrt(np.mean(errors[:, 0] ** 2)),
                    current_vector_rms_ma=1000 * np.sqrt(np.mean(np.sum(errors[:, 1:] ** 2, axis=1))),
                )
            )
    if rows:
        write(out / "common_state.csv", rows)
    if primary:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
        for skew in (0, 1):
            p = out / f"{primary}{skew}.npz"
            if not metrics(p).get("complete", False):
                raise ValueError(f"Incomplete run: {p}")
            with np.load(p) as a:
                x = a["trace"]
                t = x[:, 0]
                axes[0, skew].plot(t, np.rad2deg(x[:, 1]), label="Вал FEM")
                axes[0, skew].plot(t, np.rad2deg(x[:, 2]), lw=0.7, label="По токам")
                axes[0, skew].plot(t, np.rad2deg(x[:, 3]), "k--", lw=0.8, label="Задание")
                axes[1, skew].plot(t, np.rad2deg(x[:, 1] - x[:, 3]), lw=0.8)
                axes[1, skew].axhspan(-1, 1, color="green", alpha=0.12)
                axes[1, skew].set_xlim(0, 12.5)
                axes[1, skew].set_ylim(-1.8, 1.8)
                axes[2, skew].plot(t, (x[:, 10] - x[:, 9]) * 1000, lw=0.6)
                axes[2, skew].set_xlabel("Время после подхвата, с")
                axes[0, skew].set_title("Без скоса" if not skew else "Скос в один пазовый шаг")
        axes[0, 0].legend()
        for row, label in enumerate(
            ("Механический угол, °", "Вал минус задание, °", "Ошибка момента, мН·м")
        ):
            axes[row, 0].set_ylabel(label)
        for ax in axes.flat:
            ax.grid(alpha=0.2)
        fig.savefig(out / "closed_loop.png", dpi=160)
        plt.close(fig)
        files = [
            *out.glob("*.npz"),
            *out.glob("*.csv"),
            *out.glob("*.png"),
            *(out / "replay").glob("*.csv"),
        ]
        hashes = {
            p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
            if not p.name.endswith(".partial.npz")
        }
        (out / "manifest.json").write_text(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=[
            "tracked",
            "identified",
            "unadjusted",
            "matched_reference",
            "quiet",
            "parallel",
            "holdout6_",
        ],
    )
    parser.add_argument("--skew", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--primary", choices=["tracked", "quiet", "parallel"])
    args = parser.parse_args()
    if args.case:
        run(args.case, args.skew, args.out.resolve(), args.seconds)
    else:
        report(args.out, args.primary)

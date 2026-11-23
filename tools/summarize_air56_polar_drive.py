"""Prepare the final per-condition table, plots and checksums from saved solver outputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_air56_skew_audit import write_csv


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(root):
    folder = root / "final_drive"
    summary = []
    for f in sorted(folder.glob("dtc_*.csv")):
        with f.open() as stream:
            rows = list(csv.DictReader(stream))
        scores = {
            k: np.array([float(v[k]) for v in rows]) for k in rows[0] if k not in ("accepted", "case")
        }
        out = dict(
            case=rows[0]["case"], passes=sum(r["accepted"] == "True" for r in rows), trials=len(rows)
        )
        for k in (
            "p95_observer_deg",
            "p95_tracking_deg",
            "rest_actual_rpm",
            "forward_actual_rpm",
            "reverse_actual_rpm",
            "p95_one_second_speed_error_rpm",
            "max_rest_excursion_deg",
            "mean_phase_rms_loaded_a",
            "peak_current_vector_a",
        ):
            out["min_" + k] = scores[k].min()
            out["max_" + k] = scores[k].max()
        out["max_turn_error"] = max(
            np.max(abs(scores[d + "_read_turns"] - scores[d + "_actual_turns"]))
            for d in ("out", "return")
        )
        summary.append(out)
    if len(summary) != 12 or any(r["trials"] != 32 for r in summary):
        raise ValueError("incomplete drive parameter grid")
    write_csv(folder / "summary.csv", summary)
    x = np.load(folder / "dtc_rated_0.npz")["trace"]
    t = x[:, 0]
    deg = lambda z: np.rad2deg(np.unwrap(z))
    ang = deg(x[:, 1])
    est = deg(x[:, 2])
    est -= round((est[0] - ang[0]) / 360) * 360
    tar = deg(x[:, 5])
    tar -= round((tar[0] - ang[0]) / 360) * 360
    fig, ax = plt.subplots(3, 1, figsize=(10, 7.4), constrained_layout=True)
    ax[0].plot(t, tar, color="black", ls="--", label="Задание")
    ax[0].plot(t, ang, lw=1.5, label="Вал модели")
    ax[0].plot(t, est, lw=0.8, alpha=0.85, label="Оценка по токам")
    ax[0].set(ylabel="Угол, °", title="Замкнутый цифровой привод, 0,8 Н·м, шум АЦП 3 мА")
    ax[0].legend(ncol=3)
    ax[1].plot(t, est - ang, lw=0.7, color="#ad451f")
    ax[1].set(ylabel="Ошибка оценки, °")
    turn = np.load(folder / "dtc_turns_skew_0.npz")["trace"]
    start = np.searchsorted(turn[:, 0], 5.0)
    for i, label in ((1, "Вал модели"), (2, "Оценка по токам")):
        ax[2].plot(turn[:, 0], (turn[:, i] - turn[start, i]) / (2 * np.pi), lw=1, label=label)
    ax[2].set(
        xlabel="Время, с",
        ylabel="Обороты от начала хода",
        title="Полный оборот и возврат, скос в один шаг, 0,4 Н·м",
    )
    for a in ax:
        a.grid(alpha=0.2)
    fig.savefig(folder / "digital_drive.png", dpi=150)
    plt.close(fig)
    if (root / "diode_rotating.csv").exists():
        angles = np.arange(7, 338, 30)
        currents = []
        for angle in angles:
            read = np.load(root / f"spin_{angle}_0.npz")["read_a"]
            currents.append(read[0] + 1j * read[1])
        currents = np.array(currents)
        tag = (currents - np.roll(currents, len(angles) // 2)) / 2
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(angles, 1e3 * tag.real, "o-", label="Синфазная составляющая")
        ax.plot(angles, 1e3 * tag.imag, "s-", label="Квадратура")
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(
            xlabel="Угол вала, °",
            ylabel="Ток второй гармоники, мА",
            title="Один диод на роторе, нелинейная FEM, парные пробы 24 В",
        )
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(root / "diode_signal.png", dpi=150)
        plt.close(fig)
    free = []
    for f in sorted(root.glob("free_*.npz")):
        if "smoke" in f.stem:
            continue
        x = np.load(f)["trace"]
        rad = lambda a: np.rad2deg(np.angle(np.exp(1j * a)))
        free.append(
            dict(
                case=f.stem,
                time_s=x[-1, 0],
                target_deg=np.rad2deg(x[-1, 5]),
                true_first_deg=np.rad2deg(x[0, 1]),
                true_final_deg=np.rad2deg(x[-1, 1]),
                read_final_deg=np.rad2deg(x[-1, 2]),
                final_target_error_deg=rad(x[-1, 5] - x[-1, 1]),
                max_speed_rpm=np.max(abs(x[:, 3])) * 30 / np.pi,
                peak_to_peak_final_60ms_deg=np.rad2deg(np.ptp(x[x[:, 0] >= x[-1, 0] - 0.06, 1])),
                p95_angle_error_deg=np.percentile(abs(rad(x[:, 2] - x[:, 1])), 95),
            )
        )
    if free:
        write_csv(root / "free_field.csv", free)
    verified = sorted(root.glob("free_resolved*.npz"))
    if verified:
        fig, axes = plt.subplots(
            1, len(verified), figsize=(5 * len(verified), 3.9), squeeze=False, constrained_layout=True
        )
        for ax, f in zip(axes[0], verified):
            x = np.load(f)["trace"]
            angle = np.rad2deg(x[:, 1])
            estimated = angle + np.rad2deg(np.angle(np.exp(1j * (x[:, 2] - x[:, 1]))))
            ax.plot(x[:, 0], angle, label="Вал FEM", lw=2)
            ax.plot(x[:, 0], estimated, label="По токам", lw=1.3)
            ax.axhline(np.rad2deg(x[0, 5]), color="black", ls="--", label="Задание")
            ax.set(xlabel="Время, с", ylabel="Угол, °")
            ax.grid(alpha=0.25)
        axes[0, 0].legend()
        fig.suptitle("Свободный вал, переходная FEM, шум токов 3 мА")
        fig.savefig(root / "free_field.png", dpi=150)
        plt.close(fig)
    files = [
        *folder.glob("*.csv"),
        *folder.glob("*.npz"),
        folder / "digital_drive.png",
        *root.glob("burst_*.npz"),
        *root.glob("free_*.npz"),
        *root.glob("L*.npz"),
        *root.glob("edm*.npz"),
        root / "harmonic_ports.npz",
    ]
    files.extend(
        root / f
        for f in (
            "motion.csv",
            "tuning.csv",
            "continuous.csv",
            "absolute_angle.csv",
            "burst_pairs.csv",
            "burst_scores.csv",
            "contrast.csv",
            "printed_loop.csv",
            "backscatter.csv",
            "harvest.csv",
            "free_field.csv",
            "band_readout.npz",
            "band_torque.npz",
            "dense_torque.npz",
            "torque_control.csv",
            "band_torque_check.csv",
            "cogging.csv",
        )
    )
    files.extend(p for p in (root / "cad_checked").glob("*") if p.suffix in (".step", ".FCStd", ".json"))
    files.extend(root.glob("band_*.npz"))
    for glob in (
        "diode*.npz",
        "diode*.csv",
        "rect_*.npz",
        "spin*.npz",
        "nonlinear_diode_angle.csv",
        "calibrated_diode.csv",
        "diode_signal.png",
    ):
        files.extend(root.glob(glob))
    if verified:
        files.append(root / "free_field.png")
    manifest = {
        p.relative_to(root).as_posix(): digest(p)
        for p in sorted(set(files))
        if p.exists() and "smoke" not in p.stem
    }
    (root / "polar_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    summarize(p.parse_args().out)

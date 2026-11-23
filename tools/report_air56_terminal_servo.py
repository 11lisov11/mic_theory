"""Archive reproducible figures and per-case scores of the free-shaft experiment."""

import argparse, csv, hashlib, json, sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.score_air56_terminal_servo import score
from research.air56_rotor_fingerprinting.terminal_servo import CASES


def write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        out = csv.DictWriter(stream, fieldnames=fields)
        out.writeheader()
        out.writerows(rows)


def diagnose(path, out):
    x = np.load(path)["trace"]
    t, p, e, w, we, g, instant, ux, uy, info = x.T
    dt = t[1] - t[0]
    load = 0.02 * np.clip((t - 0.03) / 0.15, 0, 1)
    requested = np.zeros(len(t))
    integrator = 0.0
    ref = e[0]
    wrap = lambda v: np.angle(np.exp(1j * v))
    for i in range(1, len(t)):
        step = np.clip(wrap(g[i] - ref), -0.6 * dt, 0.6 * dt)
        ref += step
        error = wrap(ref - e[i])
        integrator = np.clip(integrator + 0.05 * dt * error, -0.012, 0.012)
        requested[i] = np.clip(0.033 * error + 0.0035 * (step / dt - we[i]) + integrator, -0.028, 0.028)
    rows = [
        dict(
            time_s=t[i],
            load_nm=load[i],
            demand_next_nm=requested[i],
            actual_rpm=w[i] * 30 / np.pi,
            read_rpm=we[i] * 30 / np.pi,
            angle_error_deg=np.rad2deg(wrap(e[i] - p[i])),
            baseline_v=np.hypot(ux[i], uy[i]),
        )
        for i in range(len(t))
    ]
    write(out / "previous_failure.csv", rows)
    fig, ax = plt.subplots(3, 1, figsize=(8.3, 6.5), sharex=True, constrained_layout=True)
    ax[0].plot(t * 1000, 1000 * load, label="Нагрузка")
    ax[0].step(t * 1000, 1000 * requested, where="post", label="Запрос регулятора")
    ax[0].set_ylabel("Момент, мН·м")
    ax[1].plot(t * 1000, w * 30 / np.pi, label="Вал FEM")
    ax[1].plot(t * 1000, we * 30 / np.pi, label="Оценка по токам")
    ax[1].set_ylabel("Скорость, об/мин")
    ax[2].plot(t * 1000, np.rad2deg(wrap(e - p)))
    ax[2].set(xlabel="Время от включения, мс", ylabel="Ошибка угла, °")
    for q in ax:
        q.axvline(78, color="#ad451f", ls=":", lw=1.5)
        q.grid(alpha=0.2)
    ax[0].legend()
    ax[1].legend()
    fig.suptitle("Первый разворот против задания при нагрузке до 0,020 Н·м")
    fig.savefig(out / "previous_failure.png", dpi=150)
    plt.close(fig)


def render_pair(out, names, stem):
    series = [np.load(out / (n + ".npz"))["trace"] for n in names]
    fig, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(12, 8), constrained_layout=True)
    for j, x in enumerate(series):
        t, pose, read, ref = x.T[:4]
        axes[0, j].plot(t, np.rad2deg(pose), lw=1.4, label="Вал FEM")
        axes[0, j].plot(t, np.rad2deg(read), lw=0.8, label="По токам")
        axes[0, j].plot(t, np.rad2deg(ref), "k--", lw=1, label="Задание")
        axes[0, j].set_title(["Без скоса", "Скос в один шаг"][j])
        axes[0, j].set_ylabel("Угол, °")
        axes[1, j].plot(t, np.rad2deg(np.angle(np.exp(1j * (read - pose)))), lw=0.8)
        axes[1, j].set_ylabel("Ошибка чтения, °")
        sel = t >= 2.0
        for angle, style in [(pose, "C0-"), (read, "C1-"), (ref, "k--")]:
            speed = (angle[sel] - np.interp(t[sel] - 1.0, t, angle)) * 30 / np.pi
            axes[2, j].plot(t[sel], speed, style, lw=1)
        axes[2, j].set(xlabel="Время, с", ylabel="Скорость за 1 с, об/мин")
        for q in axes[:, j]:
            q.grid(alpha=0.2)
    axes[0, 0].legend(ncol=3)
    fig.savefig(out / (stem + ".png"), dpi=150)
    plt.close(fig)


def report(out, previous, pairs):
    diagnose(previous, out)
    rows = []
    files = []
    durations = {
        case + str(skew): float(args[args.index("--seconds") + 1])
        for case, args in CASES.items()
        for skew in (0, 1)
    }
    for f in sorted(out.glob("*.npz")):
        if f.stem.endswith(".partial"):
            continue
        with np.load(f) as d:
            if "trace" not in d:
                continue
            x = d["trace"]
            expected = durations.get(f.stem)
            metrics = score(x, seconds=expected) if expected is not None else {}
            protocol = "diagnostic"
            if expected is not None:
                protocol = "completed" if metrics["complete"] else "partial"
            records = dict(
                case=f.stem,
                protocol=protocol,
                seconds=x[-1, 0],
                points=len(x),
                target_final_deg=np.rad2deg(x[-1, 3]),
                shaft_final_deg=np.rad2deg(x[-1, 1]),
                cold_estimate_deg=np.rad2deg(float(d["cold_angle"])),
                **metrics,
            )
            if "polarity_margin" in d and np.isfinite(d["polarity_margin"]):
                records["polarity_delta_chi2"] = float(d["polarity_margin"])
            if "startup_peak_a" in d:
                records["startup_peak_a"] = float(d["startup_peak_a"])
            rows.append(records)
            files.append(f)
    write(out / "scores.csv", rows)
    for stem, a, b in pairs:
        render_pair(out, (a, b), stem)
    files += (
        list(out.glob("*.csv"))
        + list(out.glob("*.png"))
        + list(out.glob("cold*.npz"))
        + list(out.glob("locked*.npz"))
        + list(out.glob("torque_fit*.npz"))
    )
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (out / "sha256.json").write_text(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--previous",
        type=Path,
        default=Path("C:/mic_air56_marker_results/polar_motion/free_loaded_020.npz"),
    )
    p.add_argument("--pair", nargs=3, action="append", default=[])
    a = p.parse_args()
    report(a.out, a.previous, a.pair)

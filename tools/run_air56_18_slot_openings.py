"""Compare closed, uniform and modulated openings on the article-based AIR56 rotor."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skfem

from fem.air56.current_tracking import CurrentMap, score_motion
from tools.run_air56_skew_audit import NOISE, RPM, article_profile, group_summary, solve_maps, write_csv
from tools.run_air56_slot_comparison import quality, skewed


FREQUENCY_HZ = 500.0
VOLTAGE_PEAK = 1.0
SEEDS = range(741200, 741224)
LONG_WINDOW_RPM = (0., -2.7, -1.2, 1.2, 2.7)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def article_baseline(path):
    with np.load(path, allow_pickle=False) as data:
        period = float(data["period"])
        if float(data["frequency_hz"]) != FREQUENCY_HZ or not np.isclose(period, 2*np.pi/18):
            raise ValueError("expected the reference 18-slot map at 500 Hz")
        z_a, z_b = data["calibration_ohm"], data["holdout_ohm"]
        return tuple(CurrentMap.from_impedance(period, z, offset)
                     for z, offset in zip((z_a, z_b), data["offsets"]))


def prepared_maps(geometry, design, period, points, levels):
    offset = period/(2*points)
    z_a, a_metrics = solve_maps(geometry, design, FREQUENCY_HZ, points, levels[0],
                                mechanical_period_rad=period)
    z_b, b_metrics = solve_maps(geometry, design, FREQUENCY_HZ, points, levels[1],
                                offset=offset, gap_phase=1, mechanical_period_rad=period)
    first = CurrentMap.from_impedance(period, z_a)
    second = CurrentMap.from_impedance(period, z_b, offset)
    return first, second, [a_metrics, b_metrics]


def harmonic_current(table, mechanical_order):
    order = int(round(mechanical_order*table.period/(2*np.pi)))
    if not np.isclose(order*2*np.pi/table.period, mechanical_order):
        return 0.0
    response = np.fft.fft(table.centered(), axis=0)/len(table.admittance)
    return float(2*np.linalg.norm(response[order]))


def motion_results(name, calibration, truth, skew_pitches, seconds, noise, speeds):
    trials = score_motion(calibration, truth, FREQUENCY_HZ, VOLTAGE_PEAK, noise,
                          SEEDS, steps=int(round(100*seconds)), scenario_rpms=speeds)
    rows = [dict(profile=name, skew_pitches=skew_pitches, window_s=seconds, **trial)
            for trial in trials]
    groups = group_summary(trials, name, skew_pitches, noise, seconds)
    joint_counts = Counter(abs(r["true_rpm"]) for r in trials
                           if r["motion_probability"] >= .95
                           and r["declared_direction"] == np.sign(r["true_rpm"]))
    for group in groups:
        speed = group["abs_rpm"]
        group["moving_correct_direction"] = joint_counts[speed] if speed else None
    return rows, groups


def assess_design(name, a, b, pitch_rad):
    rows, summary, contrast_rows = [], [], []
    scenarios = [(seconds, noise, RPM) for seconds in (1.2, .24) for noise in NOISE]
    scenarios.append((4.8, NOISE[0], LONG_WINDOW_RPM))
    for skew_pitches in (0., 1.):
        if skew_pitches:
            calibration = skewed(a, pitch_rad)
            truth = skewed(b, pitch_rad)
        else:
            calibration, truth = a, b
        contrast_rows.append(dict(profile=name, skew_pitches=skew_pitches,
                                  **quality(calibration, truth),
                                  second_harmonic_a_per_v=harmonic_current(truth, 2),
                                  tooth_harmonic_a_per_v=harmonic_current(truth, 18)))
        for seconds, noise, speeds in scenarios:
            result, groups = motion_results(name, calibration, truth, skew_pitches, seconds, noise, speeds)
            rows.extend(result)
            summary.extend(groups)
    return rows, summary, contrast_rows


def figure(out, contrast, results):
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.65), layout="constrained")
    colors = {"closed": "#7c8996", "uniform": "#af7655", "modulated": "#29725d"}
    ax = axes[0]
    for i, name in enumerate(colors):
        values = [r for r in contrast if r["profile"] == name]
        for row, offset, marker in zip(values, (-.1, .1), ("o", "D")):
            value = row["contrast_a_per_v"]*1e6
            ax.scatter(i+offset, max(1e-3, value), color=colors[name], s=62,
                       marker=marker if value >= 1e-3 else "v")
    ax.scatter([], [], color="#64696c", marker="o", label="Без скоса")
    ax.scatter([], [], color="#64696c", marker="D", label="Скос 1 шаг")
    ax.scatter([], [], color="#64696c", marker="v", label="Ниже шкалы")
    ax.axhline(300, ls=":", color="#64696c", label="Шум 0,3 мА при 1 В")
    ax.set(yscale="log", ylabel="Пиковый контраст тока, мкА/В", title="Сравнение геометрий")
    ax.set_xticks(range(3), labels=["Закрытые", "Одинаковые", "Переменные"])
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=.22)

    ax = axes[1]
    for name, skew, style, label in (
        ("closed", 0., "o-", "Штатные · без скоса"),
        ("uniform", 0., "x--", "Одинаковые · без скоса"),
        ("modulated", 0., "o-", "Переменные · без скоса"),
        ("modulated", 1., "s--", "Переменные · скос 1 шаг"),
    ):
        rows = [r for r in results if r["profile"] == name and r["skew_pitches"] == skew
                and r["window_s"] == 1.2 and r["noise_a"] == .0003 and r["abs_rpm"]]
        ax.plot([r["abs_rpm"] for r in rows], [100*r["moving_correct_direction"]/r["n"] for r in rows],
                style, label=label, color=colors[name])
    ax.set(xscale="log", xlabel="Модуль истинной скорости, об/мин",
           ylabel="Движение с верным знаком, %", title="Окно 1,2 с · шум 0,3 мА", ylim=(-4, 104))
    ax.set_xticks([1.2, 2.7, 9.8, 27.3], labels=["1,2", "2,7", "9,8", "27,3"])
    ax.grid(alpha=.22)
    ax.legend(loc="best", fontsize=8.5)
    fig.suptitle("АИР56В2 · 18 пазов ротора · проба 500 Гц")
    fig.savefig(out/"air56_18_slot_openings.png", dpi=185)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-map", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out/"summary.json"
    if summary_path.exists():
        parser.error("choose a new output directory")
    start = perf_counter()
    g, closed = article_profile()
    rotor_pitch = 2*np.pi/g.rotor_bars
    designs = {"closed": closed, "uniform": replace(closed, opening_fraction=.13),
               "modulated": replace(closed, opening_fraction=.13, modulation=.40)}
    maps, checks = {"closed": article_baseline(args.closed_map)}, {}
    for name, period, points, levels in (("uniform", rotor_pitch, 32, (6, 8)),
                                         ("modulated", np.pi, 72, (5, 8))):
        print(f"FEM {name}: {points} angles at each of two mesh levels", flush=True)
        first, second, checks[name] = prepared_maps(g, designs[name], period, points, levels)
        maps[name] = first, second
        np.savez_compressed(out/f"{name}_impedance.npz", **{
            name+"_calibration_ohm": np.linalg.inv(first.admittance),
            name+"_holdout_ohm": np.linalg.inv(second.admittance),
            name+"_period_rad": period, name+"_holdout_offset_rad": second.offset,
        })
        print(f"{name}: {second.contrast()*1e6:.2f} uA/V", flush=True)
    all_rows, summary, contrast = [], [], []
    for name, (a, b) in maps.items():
        rows, groups, metrics = assess_design(name, a, b, rotor_pitch)
        all_rows.extend(rows)
        summary.extend(groups)
        contrast.extend(metrics)
        print(f"{name}: {len(rows)} motion trials", flush=True)
    write_csv(out/"trials.csv", all_rows)
    write_csv(out/"speed_results.csv", summary)
    figure(out, contrast, summary)
    outputs = {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.suffix in (".csv", ".npz", ".png")}
    payload = dict(geometry=asdict(g), designs={name: asdict(value) for name, value in designs.items()},
                   frequency_hz=FREQUENCY_HZ, peak_probe_v=VOLTAGE_PEAK,
                   rotor_pitch_deg=float(np.rad2deg(rotor_pitch)), noise_a=NOISE,
                   seeds=[SEEDS.start, SEEDS.stop], signed_rpm=RPM, elapsed_s=perf_counter()-start,
                   long_window_s=4.8, long_window_signed_rpm=LONG_WINDOW_RPM,
                   closed_map_sha256=file_sha256(args.closed_map), simulation_only=True,
                   versions=dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__, skfem=skfem.__version__),
                   trial_count=len(all_rows), contrast=contrast, mesh_checks=checks,
                   speed_results=summary, files_sha256=outputs)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()

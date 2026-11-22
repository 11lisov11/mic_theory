"""Plot the matched slot study with a common current and direction scale."""

import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(file):
    return json.loads(file.read_text(encoding="utf8"))


def direction_curve(table, noise):
    rows = table[str(noise)]
    speeds = sorted({abs(x["rpm"]) for x in rows if x["rpm"]})
    rates = []
    for speed in speeds:
        entries = [x for x in rows if abs(x["rpm"]) == speed]
        rates.append(100*sum(x["correct"] for x in entries)/sum(x["trials"] for x in entries))
    return speeds, rates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stock", type=Path, required=True)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    stock = load(args.stock/"stock_summary.json")
    compared = load(args.comparison/"compare_summary.json")
    if stock["frequency_hz"] != compared["frequency_hz"]:
        raise ValueError("diagrams require the same probe frequency")
    noise_a = 0.0003
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), layout="constrained")
    ax = axes[0]
    for label, name, style in (
        ("Штатный без скоса", "stock", "o-"),
        ("Штатный и контур · скос 1 шаг", "stock_one_pitch_skew", "v:"),
        ("Изменённые пазы · скос 1 шаг", "modulated_openings_one_pitch_skew", "s--"),
    ):
        x, y = direction_curve(compared["motion"][name], noise_a)
        ax.plot(x, y, style, label=label)
    ax.set(title="Направление при шуме 0,3 мА", ylabel="Правильно определён знак, % от всех опытов",
           xlabel="Модуль скорости, об/мин", ylim=(-5,105), xscale="log")
    ax.set_xticks([1,3,10,30], labels=[1,3,10,30])
    ax.grid(alpha=.20)
    ax.legend(loc="best")
    ax = axes[1]
    designs = compared["designs"]
    names = ["Штатный", "Контур у зубца", "Раскрытия пазов"]
    colors = ["#396d86", "#bd7757", "#357a65"]
    without = [stock["comparison"]["contrast_a_per_v"], designs["surface_bridge"]["contrast_a_per_v"],
               designs["modulated_openings"]["contrast_a_per_v"]]
    skewed = [stock["skew_contrast_a_per_v"]["1.0"], designs["surface_bridge"]["one_pitch_skew_contrast_a_per_v"],
              designs["modulated_openings"]["one_pitch_skew_contrast_a_per_v"]]
    ref = noise_a*1e6
    for i in range(len(names)):
        for z, x, symbol in [(without[i],i-.11,"o"),(skewed[i],i+.11,"D")]:
            y = 1e6*z
            if y >= 1e-5:
                ax.scatter(x, y, marker=symbol, color=colors[i], s=60)
            else:
                ax.scatter(x, 1e-5, marker="v", color=colors[i], s=60)
    ax.scatter([], [], marker="o", c="#777777", label="Без скоса")
    ax.scatter([], [], marker="D", c="#777777", label="Скос 1 шаг")
    ax.scatter([], [], marker="v", c="#777777", label="Ниже шкалы")
    ax.axhline(ref, color="#666666", linestyle=":", label="Шум 0,3 мА при 1 В")
    ax.set(title="Изменение углового сигнала", ylabel="Максимальный контраст, мкА/В",
           yscale="log", xlim=(-.41,2.42))
    ax.set_xticks(range(3), labels=names)
    ax.grid(axis="y", alpha=.20)
    ax.legend(loc="best")
    fig.suptitle(f"АИР56В2 · {stock['frequency_hz']:g} Гц · симуляция токов статора")
    fig.savefig(args.out, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()

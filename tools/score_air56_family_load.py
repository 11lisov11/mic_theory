"""Blind angular holdout for temperature and geometry candidates at matched torque."""
import argparse
import csv
from pathlib import Path
import numpy as np

from run_air56_family_screen import write_csv
from tools.run_air56_standstill_angle import GRID, estimate_angle, periodic_interpolation, angle_error
from fem.air56.current_tracking import real_features, anisotropic_current


def run(root):
    rows=[]
    rng=np.random.default_rng(11803)
    angles=rng.uniform(0,np.pi,512)
    noise=rng.standard_normal((512,6,4))
    for case in ("stock","bars30","bars45","bars60","barrier","combined"):
        for training,testing in ((20,20),(20,80),(80,80)):
            with np.load(root/"load"/f"{case}_{training}.npz") as a, np.load(root/"load_fine"/f"{case}_{testing}.npz") as b:
                features=lambda bank: anisotropic_current(real_features(bank["admittance"]))
                templates=periodic_interpolation(GRID,a["angle_rad"],features(a))
                clean=periodic_interpolation(angles,b["angle_rad"],features(b))
            for ma in (.3,1.,3.):
                estimated=estimate_angle(clean+ma*.001*noise,templates)
                error=abs(angle_error(estimated,angles))
                rows.append(dict(case=case,cal_c=training,test_c=testing,phasor_noise_ma=ma,
                                 p95_deg=float(np.percentile(error,95)),inside_5_deg=int((error<=5).sum()),n=len(error)))
    write_csv(root/"temperature_holdout.csv",rows)
    plot_tradeoff(root, rows)


def plot_tradeoff(root, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"stock":"Штатный", "bars30":"±30%", "bars45":"±45%",
              "bars60":"±60%", "barrier":"Барьер", "combined":"±60% + барьер"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, temperature in zip(axes, (20, 80)):
        costs = {}
        errors = {}
        for name in labels:
            path = root/"load_fine"/f"{name}_{temperature}.csv"
            with path.open() as stream:
                costs[name] = float(next(csv.DictReader(stream))["copper_w"])
            errors[name] = next(r["p95_deg"] for r in rows if r["case"] == name
                                and r["cal_c"] == r["test_c"] == temperature and r["phasor_noise_ma"] == 3)
        for name, label in labels.items():
            x, y = costs[name]-costs["stock"], errors[name]
            ax.scatter(x, y)
            ax.annotate(label, (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        ax.set(title=f"{temperature} °C", xlabel="Дополнительные потери меди статора, Вт", ylim=(0,100))
        ax.margins(x=.20)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Ошибка угла P95, градусы по модулю 180°")
    fig.suptitle("Нагрузка 0,8 Н·м, шум демодуляции 3 мА")
    fig.savefig(root/"loaded_tradeoff.png", dpi=160)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args().out)

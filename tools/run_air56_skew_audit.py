"""AIR56B2 rotor-skew sensitivity on rated-source and prior slot geometries."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from shutil import copyfile
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fem.air56.current_tracking import CurrentMap, score_motion
from fem.air56.sliding_section import SlotDesign, make_section, solve
from fem.air56.stator import Geometry
from tools.run_air56_marker_fem_study import CONFIG
from tools.run_air56_slot_comparison import quality, skewed


SKEW = (0., .125, .25, .375, .5, .625, .75, .8, .85, .9, .925, .95, .975,
        1., 1.025, 1.05, 1.075, 1.1, 1.125, 7./6, 1.25, 1.375, 1.5)
RPM = (0., -27.3, -9.8, -2.7, -1.2, 1.2, 2.7, 9.8, 27.3)
NOISE = (0.0003, 0.003)
SEEDS = range(740400, 740424)
ARTICLE = "https://pribor.ifmo.ru/file/article/20275.pdf"


def article_profile():
    prior = Geometry.from_config(json.loads(CONFIG.read_text(encoding="utf8")))
    bore = .02425
    pitch = 2*np.pi*bore/24
    measured = replace(prior, rotor_radius=.024, shaft_radius=.008, air_gap=.00025,
                       stator_radius=.0445, stack_length=.054, stator_slot_depth=.00875,
                       stator_slot_fraction=1-.003/pitch, rotor_bars=18, turns=540/np.sqrt(3),
                       rotor_bar_depth=.004, rotor_bar_fraction=.39)
    return measured, SlotDesign(opening_fraction=0., opening_depth_m=.0005)


def solve_maps(geometry, design, frequency, n, level, offset=0., gap_phase=0, *,
               mechanical_period_rad=None):
    period = (2*np.pi/geometry.rotor_bars
              if mechanical_period_rad is None else mechanical_period_rad)
    rows = []
    for theta in np.arange(n)*period/n+offset:
        section = make_section(geometry, theta, level, design, gap_phase=gap_phase,
                               stator_opening_fraction=.16, stator_opening_depth_m=.0007)
        rows.append(solve(geometry, frequency, section, 27.5/3))
    return np.array([r["z"] for r in rows]), {
        key: float(max(r[key] for r in rows)) for key in ("dofs", "energy", "residual", "reciprocity")}


def phi_even_error(table):
    angles = np.arange(len(table.admittance))*table.period/len(table.admittance) + table.offset
    # Use pairs exactly representable on both the node grid and its reflection.
    mirror = table.sample(-angles)
    y = table.admittance
    scalar = (.5*np.trace(y, axis1=1, axis2=2))[:, None, None]*np.eye(2)
    base = max(table.contrast(), 1e-30)
    return {"mirror_current_error_a_per_v": float(np.linalg.norm(y-mirror,axis=(1,2)).max()),
            "non_scalar_current_a_per_v": float(np.linalg.norm(y-scalar,axis=(1,2)).max()),
            "mirror_rel_to_contrast": float(np.linalg.norm(y-mirror,axis=(1,2)).max()/base)}


def make_article_maps(out):
    geom, design = article_profile()
    period = 2*np.pi/geom.rotor_bars
    starts = [0., period/64]
    cases = {}
    for frequency in (500, 1000, 2500):
        if frequency == 500:
            coarse, ca = solve_maps(geom, design, frequency, 32, 7)
            fine, cb = solve_maps(geom, design, frequency, 32, 10, starts[1], 1)
            first = CurrentMap.from_impedance(period, coarse)
            second = CurrentMap.from_impedance(period, fine, starts[1])
            np.savez_compressed(out/"article18-impedance.npz", frequency_hz=frequency, period=period,
                                calibration_ohm=coarse, holdout_ohm=fine, offsets=starts)
            row = {**quality(first, second), **phi_even_error(second), "mesh_checks":[ca,cb]}
        else:
            z, cb = solve_maps(geom, design, frequency, 16, 5)
            table = CurrentMap.from_impedance(period, z)
            row = {"contrast_a_per_v":table.contrast(), **phi_even_error(table), "mesh_checks":[cb]}
        cases[str(frequency)] = row
        print(f"ITMO geometry {frequency} Hz: contrast {row['contrast_a_per_v']*1e6:.4f} uA/V", flush=True)
    return first, second, cases


def load_article_maps(source, out):
    summary = json.loads((source/"skew_summary.json").read_text(encoding="utf8"))
    if summary["source_article"] != ARTICLE or summary["frequency_hz"] != 500:
        raise ValueError("the cached FEM section must match the article profile and 500 Hz")
    original = source/"article18-impedance.npz"
    with np.load(original) as data:
        period = float(data["period"])
        offsets = data["offsets"]
        if not np.isclose(period, 2*np.pi/18) or float(data["frequency_hz"]) != 500:
            raise ValueError("expected the article 18-slot FEM map at 500 Hz")
        a, b = [CurrentMap.from_impedance(period, data[key], offset)
                for key, offset in zip(("calibration_ohm", "holdout_ohm"), offsets)]
    copyfile(original, out/original.name)
    return a, b, summary["article18"]

def group_summary(rows, profile, pitch, noise, window):
    result = []
    for speed in sorted({abs(r["true_rpm"]) for r in rows}):
        group = [r for r in rows if abs(r["true_rpm"]) == speed]
        reference_sign = [np.sign(r["true_rpm"]) for r in group]
        sign = [r["declared_direction"] for r in group]
        moving_errors = [abs(r["est_abs_rpm"]-speed) for r in group if r["motion_probability"] >= .95]
        result.append(dict(profile=profile, skew_pitches=pitch, noise_a=noise,
            window_s=window, abs_rpm=speed, n=len(group),
            correct=int(sum(a == b for a,b in zip(sign,reference_sign))),
            incorrect=int(sum(a is not None and a != b for a,b in zip(sign,reference_sign))),
            abstained=int(sum(a is None for a in sign)),
            moving_confident=int(sum(r["motion_probability"]>=.95 for r in group)),
            magnitude_p95_rpm=float(np.quantile(moving_errors,.95)) if moving_errors else None,
            magnitude_mae_rpm=float(np.mean(moving_errors)) if moving_errors else None))
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def predefined_shapes(first, second, profile):
    for width in SKEW:
        yield width, "matched", skewed(first, width*first.period), skewed(second, width*second.period)
    # A separate run measures overconfidence if the estimated machine assumes
    # straight slots while the synthetic observations come from a skewed stack.
    if profile == "prior_28":
        for width in (.5, .9, 1., 7/6):
            yield width, "assumes_straight", first, skewed(second, width*second.period)


def plot(out, summary, sweeps):
    fig, ax = plt.subplots(1, 2, figsize=(11.1,4.5), layout="constrained")
    labels = {"prior_28":"24/28 · прежняя модель","article_18":"24/18 · геометрия статьи"}
    for name, rows in sweeps.items():
        x = [r["skew_pitches"] for r in rows]
        y = [max(r["contrast_a_per_v"]*1e6,1e-3) for r in rows]
        ax[0].plot(x,y,"o-",label=labels[name],ms=3.5)
        ax[0].scatter([k for k,v in zip(x,y) if v==1e-3],[1e-3]*sum(v==1e-3 for v in y),marker="v",s=29)
    ax[0].axhline(300,color="#64646a",ls=":",label="Шум 0,3 мА при 1 В")
    ax[0].set(yscale="log",xlabel="Скос, шагов ротора",ylabel="Пиковый контраст, мкА/В",
              title="Подавление углового сигнала")
    for model, speed, shape in (("prior_28",1.2,"o-"),("prior_28",2.7,"s--"),("article_18",2.7,"v:")):
        rows = [r for r in summary if r["profile"]==model and r["template"]=="matched"
                and r["noise_a"]==.0003 and r["window_s"]==1.2 and r["abs_rpm"]==speed]
        ax[1].plot([r["skew_pitches"] for r in rows],[100*r["correct"]/r["n"] for r in rows],
                   shape,ms=3.5,label=f"{model.split('_')[1]} пазов · ±{str(speed).replace('.', ',')} об/мин")
    ax[1].set(xlabel="Скос, шагов ротора",ylabel="Верный знак, % от всех прогонов",ylim=(-4,104),
              title="Определение направления")
    for a in ax:
        a.axvline(1.,color="#777777",lw=.85,alpha=.65)
        a.set_xlim(-.025,1.525)
        a.grid(alpha=.18)
        a.legend(loc="best",fontsize=9)
    fig.suptitle("АИР56В2 · проба 500 Гц · скос с равномерным поворотом вдоль пакета")
    fig.savefig(out/"skew_sweep.png",dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior-stock",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--article-maps-from", type=Path)
    args = p.parse_args()
    start = perf_counter()
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    dst = out/"skew_summary.json"
    if dst.exists():
        p.error("use a new output directory")
    with np.load(args.prior_stock) as data:
        if float(data["hz"]) != 500 or not np.isclose(float(data["period"]),2*np.pi/28):
            raise ValueError("expected 500 Hz prior with 28 rotor bars")
        p28 = [CurrentMap.from_impedance(float(data["period"]),data[key],offset)
               for key,offset in zip(("z_calibration","z_fine"),data["offsets"])]
    a18,b18,article = (load_article_maps(args.article_maps_from, out) if args.article_maps_from
                       else make_article_maps(out))
    motion_rows, results, sweeps = [], [], {}
    for profile,(first,second) in (("prior_28",p28),("article_18",(a18,b18))):
        sweeps[profile] = []
        for width,template,a,b in predefined_shapes(first,second,profile):
            if template=="matched":
                sweeps[profile].append({"skew_pitches":width,"mechanical_skew_deg":float(np.rad2deg(width*b.period)),
                                        "contrast_a_per_v":b.contrast()})
            for noise in NOISE:
                for steps in (120,24):
                    if steps==24 and (profile!="prior_28" or template!="matched" or width not in (0.,.5,.75,.9,.95,1.,7/6)):
                        continue
                    rows = score_motion(a,b,500,1.,noise,SEEDS,steps=steps,scenario_rpms=RPM)
                    motion_rows.extend(dict(profile=profile,template=template,skew_pitches=width,window_s=steps/100,**r)
                                       for r in rows)
                    results.extend(dict(template=template,**r) for r in group_summary(rows,profile,width,noise,steps/100))
        print(profile, "skew sweep finished",flush=True)
    write_csv(out/"skew_trials.csv",motion_rows)
    write_csv(out/"skew_by_speed.csv",results)
    plot(out,results,sweeps)
    geometry, rotor = article_profile()
    payload = dict(source_article=ARTICLE, frequency_hz=500, probe_peak_v=1.0,
                   article_geometry=asdict(geometry), article_rotor=asdict(rotor),
                   article_phase_resistance_star_ohm=27.5/3,
                   article_stator_opening_fraction=.16, article_stator_opening_depth_m=.0007,
                   prior_sha256=hashlib.sha256(args.prior_stock.read_bytes()).hexdigest(),
                   article18=article, seeds=[SEEDS.start,SEEDS.stop], scenarios_rpm=RPM,
                   contrast=sweeps, motion=results, elapsed_s=perf_counter()-start)
    dst.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


if __name__=="__main__":
    main()

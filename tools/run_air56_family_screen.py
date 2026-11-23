"""Small matched 24/18 FEM screen, passive resonator envelope and MIC AI export."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fem.air56.sliding_section import make_section, solve
from tools.run_air56_skew_audit import article_profile, write_csv
from tools.run_air56_18_slot_openings import file_sha256

FREQUENCIES = np.array([100., 500., 2500.])
ANGLES = np.arange(24)*np.pi/12
CASES = {"stock": (0., None), "bars30": (.3, None), "bars60": (.6, None),
         "barrier": (0., 1.), "insert30": (0., 30.), "combined": (.6, 1.)}


def field_map(name, path):
    g, closed = article_profile()
    width, permeability = CASES[name]
    design = replace(closed, bar_width_modulation=width)
    shape = (2, len(FREQUENCIES), len(ANGLES))
    z = np.empty((*shape, 2, 2), complex)
    loss = np.empty((*shape, 2))
    transfer = np.empty((*shape, 2, 2), complex)
    check = []
    for mesh, level in enumerate((4, 7)):
        for i, angle in enumerate(ANGLES):
            sec = make_section(g, angle, level, design, mesh)
            mu = np.where(sec.steel, g.mu_r, 1.)
            centers = sec.mesh.p[:,sec.mesh.t].mean(axis=1)
            relative = (np.arctan2(centers[1], centers[0])-angle+np.pi)%(2*np.pi)-np.pi
            radius = np.linalg.norm(centers, axis=0)
            if permeability is not None:
                patch = sec.steel & (radius > g.rotor_radius-.0005) & (radius < g.rotor_radius) & (abs(relative)<np.pi/18)
                mu[patch] = permeability
            for j, hz in enumerate(FREQUENCIES):
                v = solve(g, hz, sec, 27.5/3, transfer=True, element_mu=mu)
                z[mesh,j,i], loss[mesh,j,i], transfer[mesh,j,i] = v["z"], v["loss"], v["transfer"]
                check.append(max(v[k] for k in ("residual","energy","reciprocity")))
    if max(check) > 1e-6:
        raise RuntimeError("field consistency check failed")
    np.savez_compressed(path, z=z, cage_w_at_1apeak=loss, field_t_per_a=transfer, check=max(check))


def resonator(out, stock):
    rows = []
    # Compact 20-turn radial-normal coil, 4 x 4 mm; local field is a proxy
    # for aperture-average flux. The coupling factors bound environmental uncertainty.
    for j, hz in enumerate(FREQUENCIES):
        w = 2*np.pi*hz
        mutual = 20*16e-6*abs(stock["field_t_per_a"][1,j,:,0,0])
        z = stock["z"][1,j,:,0,0]
        for scale in (.1, .3, 1.):
            for resistance in (1., 4.):
                for detune in (.8, 1., 1.2):
                    inductance = 50e-6
                    cap = 1/((w*detune)**2*inductance)
                    secondary = resistance+1j*(w*inductance-1/(w*cap))
                    reflected = (w*scale*mutual)**2/secondary
                    current = 1/(z+reflected)
                    difference = current-1/z
                    rows.append(dict(hz=hz, coupling_scale=scale, resistance_ohm=resistance, resonance_ratio=detune,
                                     cap_uf=cap*1e6, q=w*detune*inductance/resistance,
                                     max_mutual_uh=max(mutual)*scale*1e6,
                                     max_centered_ma=1e3*max(abs(difference-difference.mean())),
                                     peak_tag_loss_w=max(.5*abs(current)**2*reflected.real)))
    write_csv(out/"lc.csv", rows)


def summarize(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
    for name in CASES:
        with np.load(out/f"{name}.npz") as data:
            z = data["z"]
            y = np.linalg.inv(z)
            for j, hz in enumerate(FREQUENCIES):
                response = y[:,j]-y[:,j].mean(axis=1,keepdims=True)
                contrast = np.linalg.norm(response,axis=(-2,-1)).max(axis=1)*1000
                rows.append(dict(case=name,hz=hz,contrast_ma_per_v=contrast[1],coarse_ma_per_v=contrast[0],
                                 cage_w_per_apeak=float(data["cage_w_at_1apeak"][1,j].mean()),
                                 half_turn_difference_ma=1e3*np.linalg.norm(y[1,j]-np.roll(y[1,j],12,axis=0),axis=(-2,-1)).max(),
                                 max_control_error=float(data["check"])))
            axes[0].plot(np.rad2deg(ANGLES),1e3*(y[1,0,:,0,0].imag-y[1,0,:,0,0].imag.mean()),label=name)
            axes[1].plot(np.rad2deg(ANGLES),data["cage_w_at_1apeak"][1,0,:,0],label=name)
            if name == "stock":
                resonator(out, data)
            if name == "bars60":
                centered = z-z.mean(axis=2,keepdims=True)
                symmetric = .5*(centered+np.roll(centered,12,axis=2))
                np.savez_compressed(out/"bars_channel.npz",theta_rad=ANGLES,frequencies_hz=FREQUENCIES,
                                    stator_z_ohm=z[1].mean(axis=1)-(27.5/3)*np.eye(2),
                                    delta_z_ohm=symmetric[1],mesh_error_ohm=abs(symmetric[1]-symmetric[0]),
                                    pole_pairs=1,profile="AIR56_PHYSICAL_CANDIDATE")
    axes[0].set_ylabel("Ток квадратуры, мА при 1 В")
    axes[1].set_ylabel("Потери клетки, Вт при 1 А пик")
    axes[1].set_xlabel("Механический угол, градусы")
    axes[0].legend(ncol=3)
    fig.savefig(out/"screen.png",dpi=150)
    write_csv(out/"families.csv",rows)
    outputs = [out/f"{name}.npz" for name in CASES]
    outputs.extend(out/name for name in ("lc.csv","bars_channel.npz","screen.png","families.csv"))
    (out/"manifest.json").write_text(json.dumps({p.name:file_sha256(p) for p in outputs}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.case:
        path = args.out/f"{args.case}.npz"
        if path.exists():
            parser.error("calculation already exists")
        field_map(args.case,path)
    else:
        summarize(args.out)

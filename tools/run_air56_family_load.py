"""Screen shortlisted designs at equal mean torque, two assumed temperatures."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fem.air56.loaded_section import nonlinear_fundamental, loaded_probe_series
from fem.air56.sliding_section import make_section
from tools.run_air56_family_screen import CASES
from tools.run_air56_skew_audit import article_profile, write_csv


def evaluate(name, temp, out, validation=False):
    g, closed = article_profile()
    width, insert = (.45, None) if name=="bars45" else CASES[name]
    g = replace(g, aluminium_sigma=g.aluminium_sigma/(1+.004*(temp-20)))
    resistance = 27.5/3*(1+.00393*(temp-20))
    sections = []
    angles = (np.arange(12)+.5*validation)*np.pi/12
    for angle in angles:
        sec = make_section(g, angle, 5+2*validation, replace(closed,bar_width_modulation=width), int(validation))
        if insert == 1.:
            x,y = sec.mesh.p[:,sec.mesh.t].mean(axis=1)
            mask = (abs((np.arctan2(y,x)-angle+np.pi)%(2*np.pi)-np.pi)<np.pi/18)
            sec.steel &= ~(mask & (np.hypot(x,y)>g.rotor_radius-.0005) & (np.hypot(x,y)<g.rotor_radius))
        sections.append(sec)
    states = [None]*len(sections)
    def torques(peak):
        for i in (0,3,6,9):
            prior = None if states[i] is None else states[i]["state"]
            states[i] = nonlinear_fundamental(g, sections[i],peak,5.,initial=prior)
        return np.mean([states[i]["torque_nm"] for i in (0,3,6,9)])-.8
    peak = brentq(torques, .8, 2.2, xtol=.0002)
    records, responses = [], []
    for i,sec in enumerate(sections):
        state = nonlinear_fundamental(g,sec,peak,5.,initial=None if states[i] is None else states[i]["state"])
        probes = loaded_probe_series(g,sec,state,100.,resistance,range(0,12,2))
        if max(max(r[1:]) for r in probes)>1e-6:
            raise ValueError("carrier field check failed")
        responses.append(np.linalg.inv(np.array([p[0] for p in probes])))
        records.append(dict(angle_rad=angles[i],torque_nm=state["torque_nm"],cage_w=state["cage_loss_w"],
                            copper_w=1.5*resistance*peak**2,b_max_t=state["max_steel_b_t"],peak_a=peak))
    y = np.array(responses)
    spread = y-y.mean(axis=0)
    contrast = 1e3*np.sqrt(np.mean(np.sum(abs(spread)**2,axis=(-2,-1))))
    for row in records:
        row["angular_rms_ma_per_v"]=contrast
    write_csv(out/f"{name}_{temp}.csv",records)
    np.savez_compressed(out/f"{name}_{temp}.npz",angle_rad=angles,admittance=y)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",choices=("stock","bars30","bars45","bars60","barrier","combined"),required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--validation",action="store_true")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    for temperature in (20,80):
        if not (args.out/f"{args.case}_{temperature}.npz").exists():
            evaluate(args.case,temperature,args.out,args.validation)

"""LC read-coil control and detuning sensitivity. Uses the saved terminal FEM maps."""
from pathlib import Path
import argparse

import numpy as np
from run_air56_family_screen import FREQUENCIES, write_csv


def run(root):
    rows = []
    with np.load(root/"stock.npz") as data:
        for j,f in enumerate(FREQUENCIES):
            w = 2*np.pi*f
            m = 20*16e-6*abs(data["field_t_per_a"][1,j,:,0,0])
            z = data["z"][1,j,:,0,0]
            # Conditional changes in coupling, effective L and series loss.
            for label, scale, lscale, r in (("near_metal",.1,.8,4.), ("spaced",.3,.95,2.),
                                            ("ferrite_low_loss",1.,1.3,1.), ("ferrite_lossy",.3,1.3,4.)):
                L = 50e-6*lscale
                for retune in (False,True):
                    cap = 1/(w*w*(L if retune else 50e-6))
                    reflection = (w*m*scale)**2/(r+1j*(w*L-1/(w*cap)))
                    change = 1/(z+reflection)-1/z
                    rows.append(dict(port="stator",case=label,retuned=retune,hz=f,
                                     contrast_ma=1000*max(abs(change-change.mean())),q=w*L/r))
    f = 100e3
    w = 2*np.pi*f
    angle = np.arange(360)*np.pi/180
    for k in (.01,.05,.2):
        reflection = (w*k*np.sqrt(100e-6*50e-6)*np.cos(angle))**2/2
        # Both series loops tuned; 1 V peak, primary series resistance 5 ohms.
        amps = 1/(5+reflection)
        rows.append(dict(port="reader",case=f"k={k}",retuned=True,hz=f,
                         contrast_ma=1000*max(abs(amps-amps.mean())),q=w*50e-6/2))
    write_csv(root/"tag_controls.csv",rows)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args().out)

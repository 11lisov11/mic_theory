"""18 physical bars and two finite-resistance end rings, low-slip loss localization."""
import argparse
from pathlib import Path

import numpy as np
from run_air56_family_screen import article_profile, write_csv


def evaluate(out):
    g,_=article_profile()
    count = 18
    angle = (np.arange(count)+.5)*2*np.pi/count
    pitch = 2*np.pi/count
    ring_length = .02125*pitch
    rows = []
    for modulation in (0.,.3,.6):
        area = .5*(.0235**2-.020**2)*pitch*.39*(1+modulation*np.cos(2*angle))
        for temperature in (20,80):
            rho = (1+.004*(temperature-20))/g.aluminium_sigma
            for ring_area in (5.4e-6,10.8e-6,21.6e-6):
                for skew in (0,1):
                    bar_length = np.hypot(.054,skew*.02175*pitch)
                    indices = np.arange(count)
                    starts = np.r_[indices,indices,indices+count]
                    ends = np.r_[indices+count,(indices+1)%count,(indices+1)%count+count]
                    incidence = np.zeros((2*count,len(starts)))
                    incidence[starts,np.arange(len(starts))] = 1
                    incidence[ends,np.arange(len(ends))] = -1
                    conductance = np.r_[area/(rho*bar_length),np.full(2*count,ring_area/(rho*ring_length))]
                    emf = np.r_[np.exp(1j*angle),np.zeros(2*count)]
                    node = incidence[1:]
                    potential = np.linalg.solve((node*conductance)@node.T,node@(conductance*emf))
                    current = conductance*(emf-node.T@potential)
                    if max(abs(incidence@current)) > 1e-8*max(abs(current)):
                        raise ValueError("unbalanced cage current")
                    loss = .5*abs(current)**2/conductance
                    loss *= (.8*2*np.pi*5)/sum(loss)
                    density = loss[:count]/(area*bar_length)
                    rows.append(dict(modulation=modulation,temp_c=temperature,skew=skew,ring_mm2=ring_area*1e6,
                                     total_cage_w=sum(loss),ring_w=sum(loss[count:]),max_bar_w=max(loss[:count]),
                                     max_bar_w_cm3=max(density)*1e-6))
    write_csv(out/"cage_network.csv",rows)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    evaluate(p.parse_args().out)

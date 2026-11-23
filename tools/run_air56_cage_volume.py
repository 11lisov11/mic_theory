"""Three-dimensional cage conduction with imposed slip EMF and both end rings."""
import argparse
from itertools import permutations
from pathlib import Path

import meshio
import numpy as np
from scipy.sparse.linalg import splu
from skfem import MeshTet, Basis, ElementTetP1, LinearForm
from skfem.models.poisson import laplace

from run_air56_family_screen import article_profile, write_csv


@LinearForm
def axial_emf(v,w):
    return w.source*v.grad[2]


@LinearForm
def volumetric_heat(v,w):
    return w.source*v


def cage_mesh(modulation, skew, level):
    pitch = 2*np.pi/18
    centers = (np.arange(18)+.5)*pitch
    width = .39*(1+modulation*np.cos(2*centers))
    start = np.sort(np.r_[centers-width*pitch/2,centers+width*pitch/2])
    azimuth = np.concatenate([np.linspace(a,b,level+1)[:-1] for a,b in
                              zip(start,np.r_[start[1:],start[0]+2*np.pi])])
    radii = np.r_[.019,np.linspace(.020,.0235,level+1)]
    axial = np.r_[np.linspace(-.0024,0,level+1)[:-1],np.linspace(0,.054,level*4+1),
                  np.linspace(.054,.0564,level+1)[1:]]
    r,a,z = np.meshgrid(radii,azimuth,axial,indexing="ij")
    theta = a+skew*pitch*(np.clip(z/.054,0,1)-.5)
    points = np.stack((r*np.cos(theta),r*np.sin(theta),z)).reshape(3,-1)
    i,j,k = np.indices((len(radii)-1,len(azimuth),len(axial)-1))
    bar = (j//level % 2==0) & (i>=1)
    inside = bar | (axial[k]<0) | (axial[k]>=.054)
    origin = np.array([i[inside],j[inside],k[inside]])
    cells = []
    for path in permutations(np.eye(3,dtype=int)):
        vertices = []
        for delta in np.vstack((np.zeros(3,dtype=int), np.cumsum(path,axis=0))):
            idx = origin+delta[:,None]
            idx[1] %= len(azimuth)
            vertices.append(np.ravel_multi_index(idx,r.shape))
        cells.append(vertices)
    cells = np.concatenate(cells,axis=1)
    used,inverse = np.unique(cells,return_inverse=True)
    return MeshTet(points[:,used],inverse.reshape(cells.shape))


def evaluate(out,modulation,skew,level):
    g,_=article_profile()
    mesh=cage_mesh(modulation,skew,level)
    basis=Basis(mesh,ElementTetP1())
    xyz=mesh.p[:,mesh.t].mean(axis=1)
    active=(xyz[2]>0)&(xyz[2]<.054)
    emf=active*np.exp(1j*np.arctan2(xyz[1],xyz[0]))
    source=axial_emf.assemble(basis,source=emf.real[:,None])+1j*axial_emf.assemble(basis,source=emf.imag[:,None])
    matrix=laplace.assemble(basis).tocsc()
    factor=splu(matrix[1:,1:].tocsc())
    v=np.r_[0.,factor.solve(source[1:].real)+1j*factor.solve(source[1:].imag)]
    field=-basis.interpolate(v).grad[:,:,0]
    field[2]+=emf
    sigma=g.aluminium_sigma
    density=.5*sigma*np.sum(abs(field)**2,axis=0)
    volume=basis.dx.sum(axis=1)
    loss=density@volume
    supply=.5*sigma*(np.real(field[2]*emf.conj())@volume)
    balance=abs(loss-supply)/loss
    if balance>1e-7:
        raise ValueError("cage 3D energy balance")
    scale=.8*2*np.pi*5/loss
    heat=volumetric_heat.assemble(basis,source=(density*scale)[:,None])
    end_plane = np.isclose(mesh.p[2],mesh.p[2].min()) | np.isclose(mesh.p[2],mesh.p[2].max())
    free = np.flatnonzero(~end_plane)
    rise = np.zeros(basis.N)
    rise[free] = splu(matrix[free][:,free].tocsc()).solve(heat[free])/190.
    if abs((heat-190.*(matrix@rise))[end_plane].sum()/(loss*scale)-1)>1e-7:
        raise ValueError("thermal power balance")
    if level==4:
        current=sigma*field.T*np.sqrt(scale)
        meshio.write_points_cells(out/f"cage_{modulation}_{skew}.vtu",mesh.p.T,[("tetra",mesh.t.T)],point_data={"temperature_rise_K":rise},
            cell_data={"J_real":[current.real],"J_imag":[current.imag],"W_per_m3":[density*scale]})
    return dict(modulation=modulation,skew=skew,level=level,tetrahedra=len(volume),power_w=loss*scale,
                ring_w=(density[~active]@volume[~active])*scale,
                max_w_cm3=density.max()*scale/1e6,max_rise_k=rise.max(),energy_rel=balance)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    out=p.parse_args().out
    out.mkdir(parents=True,exist_ok=True)
    rows=[evaluate(out,m,s,n) for m in (0.,.3,.6) for s in (0,1) for n in (2,4)]
    write_csv(out/"volume.csv",rows)

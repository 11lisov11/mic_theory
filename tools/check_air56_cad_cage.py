"""Mesh the aluminium cage from the candidate STEP and check loss using GetDP."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import gmsh
import numpy as np
from run_air56_family_screen import article_profile, file_sha256


def check(step, getdp, out):
    out.mkdir(parents=True,exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal",0)
        occ=gmsh.model.occ
        parts=occ.importShapes(str(step))
        core=max(parts,key=lambda entity:occ.getMass(*entity))
        cage=[entity for entity in parts if entity!=core]
        occ.remove([core],recursive=True)
        volume,_=occ.fuse(cage[:1],cage[1:])
        if len(volume)!=1:
            raise RuntimeError("disconnected STEP cage")
        occ.dilate(volume,0,0,0,.001,.001,.001)
        occ.synchronize()
        gmsh.model.addPhysicalGroup(3,[volume[0][1]],1)
        gmsh.model.addPhysicalGroup(0,[gmsh.model.getEntities(0)[0][1]],2)
        gmsh.option.setNumber("Mesh.MeshSizeMax",.0018)
        gmsh.option.setNumber("Mesh.MeshSizeMin",.0007)
        gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(out/"cage.msh"))
    finally:
        gmsh.finalize()
    pro=Path(__file__).resolve().parents[1]/"fem/air56/cage_conduction.pro"
    shutil.copyfile(pro,out/"cage.pro")
    result=[]
    for quadrature in (0,1):
        cmd=[str(getdp),"cage.pro","-msh","cage.msh","-setnumber","Quadrature",str(quadrature),
             "-setnumber","Sigma",str(article_profile()[0].aluminium_sigma),"-solve","SolveCage","-pos","Balance"]
        with (out/f"solver_{quadrature}.log").open("w") as log:
            subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
        loss,work=[np.loadtxt(out/name).ravel()[-1] for name in ("loss.txt","input.txt")]
        if loss<=0 or abs(loss-work)>loss*1e-5:
            raise ValueError("GetDP energy check failed")
        result.append(float(loss))
    (out/"check.json").write_text(json.dumps(dict(step_sha256=file_sha256(step),mesh_sha256=file_sha256(out/"cage.msh"),
        solver_sha256=file_sha256(getdp),power_w=sum(result),quadrature_w=result),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("step","getdp","out"):
        p.add_argument("--"+arg,type=Path,required=True)
    args=p.parse_args()
    check(args.step.resolve(),args.getdp.resolve(),args.out.resolve())

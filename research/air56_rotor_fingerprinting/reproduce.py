"""Reproduce the family screen in an empty output directory, or verify a saved run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.verify_air56_marker_results import paired_pwm


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def record(root):
    groups = ["families.csv","lc.csv","manifest.json","bars_channel.npz","*.npz","screen.png",
              "tag_controls.csv","cage_network.csv","temperature_holdout.csv","loaded_tradeoff.png","paired_pwm.json","load/*",
              "load_fine/*","cad_verified/*.step","cad_verified/*.FCStd","cage_thermal/*",
              "getdp_*/check.json","getdp_*/cage.msh","controllers/controller_*","coded_*/trials.csv",
              "coded_*/summary.json"]
    files = {f.relative_to(root).as_posix():digest(f) for group in groups for f in root.glob(group) if f.is_file()}
    required = ["temperature_holdout.csv","getdp_Bars60Skew/check.json","cad_verified/LC.step",
                "cage_thermal/cage_0.6_1.gif","controllers/controller_summary.json","paired_pwm.json"]
    if any(key not in files for key in required):
        raise ValueError("missing final deliverable")
    sources = [Path(__file__),*(ROOT/"tools").glob("*air56_family*.py"),*(ROOT/"tools").glob("*air56_cage*.py"),
               ROOT/"tools/run_air56_tag_controls.py",ROOT/"cad/air56/build_family_rotors.py",
               ROOT/"fem/air56/cage_conduction.pro",ROOT/"fem/air56/sliding_section.py",
               ROOT/"tools/check_air56_cad_cage.py"]
    return dict(files_sha256=files,sources_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
                versions={k:importlib.metadata.version(k) for k in ("numpy","scipy","scikit-fem","gmsh","meshio")})


def run(args):
    out = args.out.resolve()
    manifest = out/"study_manifest.json"
    if args.verify:
        prior=json.loads(manifest.read_text())
        current=record(out)
        if current!=prior:
            raise ValueError("file, source or environment differs from recorded study")
        print(f"Verified {len(prior['files_sha256'])} files")
        return
    if args.record:
        manifest.write_text(json.dumps(record(out),indent=2),encoding="utf8")
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("use a new output directory or --verify")
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",PYTHONPATH=os.pathsep.join(filter(None,(str(ROOT),os.environ.get("PYTHONPATH")))))
    def execute(name,*values,exe=sys.executable):
        subprocess.run([exe,str(ROOT/name),*map(str,values)],cwd=ROOT,env=env,check=True)
    cases=("stock","bars30","bars60","barrier","insert30","combined")
    for case in cases:
        execute("tools/run_air56_family_screen.py","--case",case,"--out",out)
    for tool in ("run_air56_family_screen","run_air56_tag_controls","run_air56_cage_network"):
        execute(f"tools/{tool}.py","--out",out)
    for case in ("stock","bars30","bars60","barrier","combined","bars45"):
        for folder,options in (("load",()),("load_fine",("--validation",))):
            execute("tools/run_air56_family_load.py","--case",case,"--out",out/folder,*options)
    execute("tools/score_air56_family_load.py","--out",out)
    execute("cad/air56/build_family_rotors.py","--out",out/"cad_verified",exe=args.cad_python)
    execute("tools/run_air56_cage_volume.py","--out",out/"cage_thermal")
    execute("tools/render_air56_cage_volume.py",out/"cage_thermal/cage_0.6_1.vtu")
    for name in ("Bars60","Bars60Skew"):
        execute("tools/check_air56_cad_cage.py","--step",out/f"cad_verified/{name}.step","--getdp",args.getdp,
                "--out",out/f"getdp_{name}")
    execute("tools/compare_air56_marker_controllers.py","--table",out/"bars_channel.npz","--policy-root",
            args.policy_root,"--out",out/"controllers")
    for name,options in (("marker",()),("control",("--fem-control",))):
        execute("tools/run_info_optimal_coded_pwm_study.py","--suite","smoke","--seeds",3,"--steps",3000,
                "--seed-offset",802100,"--measurement-channel","fem","--fem-response",out/"bars_channel.npz",
                *options,"--out",out/f"coded_{name}")
    contrast=paired_pwm(out/"coded_marker/trials.csv",out/"coded_control/trials.csv")
    (out/"paired_pwm.json").write_text(json.dumps(contrast,indent=2))
    manifest.write_text(json.dumps(record(out),indent=2),encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--record",action="store_true")
    mode.add_argument("--verify",action="store_true")
    p.add_argument("--cad-python",default="C:/Program Files/FreeCAD 1.1/bin/python.exe")
    p.add_argument("--policy-root",default="C:/mic_theory")
    p.add_argument("--getdp",default="C:/mic_air56_marker_results/family_screen/solvers/getdp/getdp-3.5.0-Windows64/getdp.exe")
    run(p.parse_args())

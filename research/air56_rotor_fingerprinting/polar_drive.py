"""Run or verify the coil-only AIR56 digital-drive research without hardware."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(out, archive, cad_python):
    if out.exists() and any(out.iterdir()):
        raise ValueError("fresh reproduction requires an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "PYTHONPATH": str(ROOT)}

    def call(name, *args):
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / (name + ".py")), *map(str, args)],
            env=env,
            cwd=ROOT,
            check=True,
        )

    call("run_air56_calibrated_motion", "--maps", archive / "pwm_slopes_18", "--out", out)
    for level, br, count, bg, name in (
        (4, 0.0, 8, 0.0, "L4B0"),
        (4, 0.4, 48, 0.0, "L4B0.4"),
        (6, 0.4, 48, 0.0, "L6B0.4"),
        (6, 0.2, 24, 0.0, "L6B0.2"),
        (7, 0.4, 32, 0.0, "L7bg0"),
        (6, 0.4, 32, 0.35, "L6bg0.35"),
    ):
        call(
            "run_air56_polarity",
            "--level",
            level,
            "--br",
            br,
            "--count",
            count,
            "--background",
            bg,
            "--out",
            out / (name + ".npz"),
        )
    # Historical calibration and coenergy exports used identical FEM settings.
    from shutil import copyfile

    copyfile(out / "L4B0.4.npz", out / "L4energy.npz")
    stock = archive / "family_screen/stock.npz"
    for level in (4, 7):
        call("run_air56_cut_loop", "--level", level, "--out", out)
    call("run_air56_cut_loop", "--stock", stock, "--out", out)
    call("run_air56_harvest", "--stock", stock, "--out", out)
    for level, count in ((4, 36), (6, 32)):
        call(
            "run_air56_distributed_diode",
            "--level",
            level,
            "--count",
            count,
            "--harmonics",
            24,
            "--out",
            out,
        )
    call("run_air56_distributed_diode", "--out", out)

    def rect(name, angle, bias=0, axis=2, level=4, flags=()):
        call(
            "check_air56_rectifier_field",
            "--out",
            out / (name + ".npz"),
            "--angle",
            angle,
            "--bias",
            bias,
            "--axis",
            axis,
            "--level",
            level,
            "--cycles",
            8,
            *flags,
        )

    for angle in (17, 197):
        for bias in (0, 0.35):
            rect(f"rect_{angle}_{bias}_0", angle, bias, 0)
    rect("rect_17_0_1", 17, 0, 0, flags=("--blank",))
    for angle in (0, *range(7, 338, 30), 45, 225):
        bias = 0.35 if angle in (45, 225) else 0
        rect(f"spin_{angle}_{bias}", angle, bias)
    for name, angle, flags in (
        ("calminus", 0, ("--invert",)),
        ("checkminus", 67, ("--invert",)),
        ("blank", 0, ("--blank",)),
        ("opposite", 180, ()),
    ):
        rect("spincheck_" + name, angle, flags=flags)
    for a in range(10, 60, 10):
        for angle in (a, a + 180, a + 2, a + 182):
            rect(f"spinmap_{angle}", angle, level=5 if angle % 10 == 2 else 4)
    call("score_air56_rectifier_field", "--out", out)
    call("score_air56_polarity", "--out", out)
    for case, level, count, steps, bias, br in (
        ("cal", 3, 36, 1, 0.0, 0.4),
        ("timecal", 3, 36, 2, 0.0, 0.4),
        ("hold", 5, 36, 2, 0.0, 0.4),
        ("load", 5, 24, 2, 0.35, 0.4),
        ("loadcal", 3, 36, 2, 0.35, 0.4),
        ("blank", 3, 12, 1, 0.0, 0.0),
        ("refined", 4, 32, 2, 0.0, 0.4),
        ("refinedload", 4, 32, 2, 0.35, 0.4),
    ):
        call(
            "run_air56_polar_burst",
            "--level",
            level,
            "--count",
            count,
            "--substeps",
            steps,
            "--bias",
            bias,
            "--br",
            br,
            "--out",
            out / f"burst_{case}.npz",
        )
    for case in ("hold", "load"):
        call(
            "run_air56_polar_burst",
            "--level",
            5,
            "--check-opposite",
            out / f"burst_{case}.npz",
            "--out",
            out / f"burst_symmetry_{case}.npz",
        )
    call("score_air56_polar_burst", "--out", out)
    for mode in (
        "rated",
        "rated_skew",
        "noisy1",
        "noisy1_skew",
        "noisy03",
        "noisy03_skew",
        "mismatch",
        "mismatch_skew",
        "turns",
        "turns_skew",
        "light",
        "medium",
    ):
        call("score_air56_direct_drive", "--out", out / "final_drive", "--group", mode)
    call("check_air56_polar_torque", "--out", out)
    subprocess.run(
        [
            cad_python,
            str(ROOT / "cad/air56/build_polar_rotor.py"),
            "--source",
            str(archive / "family_screen/cad_verified"),
            "--out",
            str(out / "cad_checked"),
        ],
        cwd=ROOT,
        check=True,
    )
    for flag in ("--calibrate-band", "--band-readout", "--dense-cal"):
        call("run_air56_free_field", "--out", out, "--theta", 0, "--target", 0, "--name", "cal", flag)
    for bias in ("xp", "xm", "yp", "ym"):
        call(
            "run_air56_free_field",
            "--out",
            out,
            "--theta",
            0,
            "--target",
            0,
            "--name",
            "cal",
            "--band-readout",
            "--bias-case",
            bias,
        )
    for initial, target, level, name in (
        (27, 35, 4, "positive"),
        (207, 199, 4, "negative"),
        (95, 105, 5, "cross"),
    ):
        for band in (False, True):
            call(
                "run_air56_free_field",
                "--out",
                out,
                "--theta",
                initial,
                "--target",
                target,
                "--level",
                level,
                "--seconds",
                0.30 if band else 0.36,
                "--name",
                ("band_" if band else "") + name,
                *(["--band"] if band else []),
            )
        call(
            "run_air56_free_field",
            "--out",
            out,
            "--theta",
            initial,
            "--target",
            target,
            "--level",
            level,
            "--seconds",
            0.36,
            "--name",
            "servo_" + name,
            "--conditioned",
        )
        call(
            "run_air56_free_field",
            "--out",
            out,
            "--theta",
            initial,
            "--target",
            target,
            "--level",
            level,
            "--seconds",
            0.36,
            "--name",
            "resolved_" + name,
            "--dense-servo",
        )
        if level == 4:
            call(
                "run_air56_free_field",
                "--out",
                out,
                "--theta",
                initial,
                "--target",
                target,
                "--level",
                level,
                "--seconds",
                0.36,
                "--name",
                "passive_" + name,
                "--conditioned",
                "--probes-only",
            )
            call(
                "run_air56_free_field",
                "--out",
                out,
                "--theta",
                initial,
                "--target",
                target,
                "--level",
                3,
                "--seconds",
                0.30,
                "--name",
                "axial_" + name,
                "--dense-servo",
                "--skew",
            )
    for name, initial, target, level in (
        ("positive", 27, 35, 3),
        ("negative", 207, 199, 3),
        ("independent", 95, 105, 4),
    ):
        for prefix, memory_ms in (("observer_", 0), ("memory_", 35)):
            call(
                "run_air56_field_observer",
                "--out",
                out,
                "--initial",
                initial,
                "--target",
                target,
                "--plant-level",
                level,
                "--blocks",
                120,
                "--memory-ms",
                memory_ms,
                "--name",
                prefix + name,
            )
    for name, load in (("005", 0.005), ("020", 0.02)):
        call(
            "run_air56_field_observer",
            "--out",
            out,
            "--initial",
            27,
            "--target",
            35,
            "--disturbance",
            load,
            "--load-ramp-s",
            0.15,
            "--blocks",
            120,
            "--memory-ms",
            35,
            "--name",
            "loaded_" + name,
        )
    call("summarize_air56_polar_drive", "--out", out)


def verify(root):
    manifest = json.loads((root / "polar_manifest.json").read_text())
    required = {
        "final_drive/summary.csv",
        "final_drive/digital_drive.png",
        "burst_pairs.csv",
        "absolute_angle.csv",
        "calibrated_diode.csv",
        "free_field.csv",
        "band_torque_check.csv",
        "cad_checked/Polar18.step",
        "cad_checked/Polar18Skew.step",
    }
    for mode in ("observer", "memory"):
        required.update(f"free_{mode}_{case}.npz" for case in ("positive", "negative", "independent"))
    required.update(("free_loaded_005.npz", "free_loaded_020.npz"))
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Missing research outputs in the manifest: {sorted(missing)}")
    for relative, expected in manifest.items():
        with (root / relative).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError(relative)
    print(f"Verified {len(manifest)} archived files")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--archive", type=Path, default=Path("C:/mic_air56_marker_results"))
    p.add_argument("--verify", action="store_true")
    p.add_argument("--cad-python", default="C:/Program Files/FreeCAD 1.1/bin/python.exe")
    a = p.parse_args()
    verify(a.out) if a.verify else run(a.out.resolve(), a.archive.resolve(), a.cad_python)

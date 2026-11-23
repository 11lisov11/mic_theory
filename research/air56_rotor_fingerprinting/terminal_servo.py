"""Reproduce a single loaded AIR56 FEM run in a separate local directory."""

import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PULSES = [
    "--resolve",
    "--adaptive",
    "--load-process",
    ".03",
    "--adc-samples",
    "32",
    "--adc-correlation",
    ".03",
]
BASE = ["--modes", "0", "--initial", "27", "--target", "35", "--leg", "4.2"]
CASES = {
    "impulse": [
        *PULSES,
        "--h",
        ".006",
        "--seconds",
        "12.5",
        "--seed",
        "6373",
        "--pwm",
        "1",
        "--inside-pwm",
        "--trapezoid",
        "--switched-start",
        "--start-volts",
        "35",
        "--shoot-pwm",
        "--probe-v",
        "1.2",
        "--stiffness",
        ".4",
        "--integral-gain",
        ".1",
    ],
    "original_pwm": [
        *PULSES,
        "--h",
        ".004",
        "--seconds",
        "12.5",
        "--seed",
        "9157",
        "--pwm",
        "1",
        "--inside-pwm",
        "--continuous-bias",
        "--probe-v",
        "2.5",
        "--stiffness",
        ".52",
        "--integral-gain",
        ".16",
    ],
    "switch_reversal": [
        *PULSES,
        "--ring",
        "--h",
        ".004",
        "--seconds",
        "12.5",
        "--seed",
        "6190",
        "--pwm",
        "1",
        "--stiffness",
        ".72",
        "--integral-gain",
        ".18",
    ],
    "burst_ring": [*PULSES, "--ring", "--h", ".006", "--seconds", "11.5", "--seed", "2119"],
    "burst_adc": [*PULSES, "--h", ".003", "--seconds", "11.5", "--seed", "2119"],
    "blind": [
        "--ring",
        "--h",
        ".004",
        "--seconds",
        "11.5",
        "--load-process",
        ".008",
        "--resolve",
        "--initial",
        "207",
        "--target",
        "215",
        "--seed",
        "5603",
    ],
    "independent": [
        *PULSES,
        "--ring",
        "--h",
        ".006",
        "--seconds",
        "12.5",
        "--level",
        "4",
        "--observer-level",
        "3",
        "--seed",
        "6201",
        "--stiffness",
        ".72",
        "--integral-gain",
        ".18",
    ],
    "crosscal": [
        *PULSES,
        "--ring",
        "--h",
        ".006",
        "--seconds",
        "12.5",
        "--level",
        "4",
        "--observer-level",
        "3",
        "--seed",
        "6201",
        "--stiffness",
        ".72",
        "--integral-gain",
        ".18",
        "--torque-calibration",
    ],
}
CASES["refit"] = CASES["independent"].copy()
CASES["refit"][CASES["refit"].index("--observer-level") + 1] = "4"
CASES["ring_low_noise"] = [*CASES["burst_ring"], "--noise", ".0003"]
CASES["ring_mid_noise"] = [*CASES["burst_ring"], "--noise", ".001"]
CASES["preloaded_pwm"] = [
    *CASES["original_pwm"],
    "--switched-start",
    "--start-volts",
    "35",
    "--preloaded",
    "--fit-start-load",
    "--check-capture",
]


def launch(out, case, skew):
    tag = case + str(skew)
    if (out / (tag + ".npz")).exists():
        raise FileExistsError("Choose a fresh output directory for this case.")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")

    def call(script, *args):
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / script), "--out", str(out), *args],
            cwd=ROOT,
            env=env,
            check=True,
        )

    if case == "crosscal":
        for j in range(4):
            if not (out / f"torque_fit_1_{j}.npz").is_file():
                call("calibrate_air56_terminal_torque.py", "--ring", "--part", str(j))
    call(
        "run_air56_terminal_servo.py", *BASE, *CASES[case], "--name", tag, *(("--skew",) if skew else ())
    )
    subprocess.run(
        [sys.executable, str(ROOT / "tools/score_air56_terminal_servo.py"), str(out / (tag + ".npz"))],
        cwd=ROOT,
        env=env,
        check=True,
    )


def verify(out):
    manifest = json.loads((out / "sha256.json").read_text())
    for name, h in manifest.items():
        with (out / name).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != h:
                raise ValueError(name)
    print("Verified", len(manifest), "archived files")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--case", choices=CASES, default="original_pwm")
    p.add_argument("--skew", type=int, choices=[0, 1], default=0)
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    verify(a.out) if a.verify else launch(a.out.resolve(), a.case, a.skew)

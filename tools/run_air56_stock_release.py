"""Free-shaft PWM currents before one degree of displacement. Simulation only."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fem.air56.servo_pwm import segments
from tools.run_air56_terminal_servo import Machine, R

PERIOD = 0.001
LOADS = (0.018, 0.020, 0.022)
SPEEDS = (-0.1, 0.0, 0.1)
# Bias is the prepared stator current; probe is peak voltage per counter-rotating sequence.
PROBES = {"cold": (0.0, 1.0), "pulse": (0.0, 12.0), "biased": (0.6, 1.0), "biased_pulse": (0.6, 12.0)}


PROBES.update(
    {
        "release10": (0.10, 0.0),
        "release20": (0.20, 0.0),
        "release30": (0.30, 0.0),
        "release60": (0.60, 0.0),
        "release90": (0.90, 0.0),
        "release120": (1.20, 0.0),
    }
)


def experiment(args):
    out, level, subdivisions, angle, skew, probe, rollback, slices, locked = args
    path = out / f"L{level}_n{subdivisions}_a{angle:g}_s{skew}_{probe}.npz"
    if path.exists():
        return
    machine = Machine(level, 0, skew, stock=True, axial_slices=slices)
    machine.trapezoid = True
    assert all(np.count_nonzero(f.pm) == 0 for f in machine.slices)
    bias, volts = PROBES[probe]
    start = np.deg2rad(angle)
    initial = machine.equilibrium(start, np.array([bias, 0.0]))
    loads = (-0.020, 0.0, 0.020) if rollback else (LOADS if level == 3 else (0.02,))
    speeds = (0.0,) if rollback else SPEEDS
    if locked:
        loads, speeds = (0.0,), (0.0,)
    recordings = []
    for external in loads:
        for rpm in speeds:
            x = initial.copy()
            x[-2:] = (rpm * np.pi / 30, external)
            samples = []
            for k in range(24):
                t = (k + 0.5) * PERIOD
                # Sum of equal clockwise and anticlockwise 3 Hz voltage vectors.
                u = np.array([bias * R + 2 * volts * np.cos(2 * np.pi * 3 * t), 0.0])
                if probe.startswith("release"):
                    u[:] = 0.0  # Both terminals in stationary axes see a shorted inverter vector.
                excursion = 0.0
                for voltage, duration in segments(u, PERIOD):
                    for _ in range(subdivisions):
                        x, _, _ = machine.step(x, voltage, duration / subdivisions, load=external)
                        if locked:
                            x[-3:-1] = (start, 0.0)
                        excursion = max(excursion, abs(np.rad2deg(x[-3] - start)))
                samples.append([*x[-5:-3], np.rad2deg(x[-3] - start), x[-2] * 30 / np.pi, excursion, x[-6]])
            recordings.append(samples)
    np.savez_compressed(
        path,
        traces=np.array(recordings).reshape(len(loads), len(speeds), 24, 6),
        loads=loads,
        speeds=speeds,
        period=PERIOD,
        angle_deg=angle,
        probe=probe,
        level=level,
        locked=locked,
        skew_pitch=skew,
        axial_slices=(slices or (3 if skew else 1)),
        subdivisions=subdivisions,
        bias_a=bias,
        sequence_peak_v=volts,
    )
    print(path.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--probes", nargs="+", choices=PROBES, default=["cold", "pulse", "biased", "biased_pulse"]
    )
    parser.add_argument("--angles", nargs="+", type=float, default=[7.3, 13.1])
    parser.add_argument(
        "--rollback", action="store_true", help="release from rest under either load sign or zero"
    )
    parser.add_argument("--slices", type=int, help="use this many axial midpoints for one pitch of skew")
    parser.add_argument("--levels", type=int, nargs="+", choices=(3, 4, 5), default=[3, 4])
    parser.add_argument("--locked", action="store_true", help="independent immobile null reference")
    opts = parser.parse_args()
    if opts.locked:
        opts.out = opts.out / "locked"
    if opts.rollback:
        opts.out = opts.out / "rollback"
    if opts.slices:
        opts.out = opts.out / f"axial{opts.slices}"
    opts.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (opts.out, lev, sub, a, skew, probe, opts.rollback, opts.slices, opts.locked)
        for lev in opts.levels
        for sub in [2 if lev == 3 else 4]
        for a in opts.angles
        for skew in ((1,) if opts.slices else (0, 1))
        for probe in opts.probes
    ]
    with ProcessPoolExecutor(opts.workers) as pool:
        list(pool.map(experiment, jobs))

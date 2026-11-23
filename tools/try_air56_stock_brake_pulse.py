"""A causal single braking pulse, paired with exactly the same passive FEM release."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_air56_terminal_servo import Machine
from tools.score_air56_stock_release import beta_filter
from fem.air56.servo_pwm import segments


def run(job):
    folder, level, slices, angle, load, seed, amp, sigma, pulse, periods = job
    skew = slices > 1
    dest = folder / f"L{level}_s{slices}_a{angle:g}_m{load:g}_rng{seed}_v{amp:g}_sd{sigma:g}.npz"
    if dest.exists():
        return
    m = Machine(level, 0, skew, stock=True, axial_slices=slices if skew else None)
    m.trapezoid = True
    x = m.equilibrium(np.deg2rad(angle), np.array([0.30, 0.0]))
    x[-1] = load
    rng = np.random.default_rng(seed)
    # Frozen using only level 3, 1.2 and 12.7 degrees, initial zero vector, 0.30 A.
    gain = -62.77561184070174
    filt = beta_filter()
    y = []
    event = None
    log = []
    for k in range(periods):
        cmd = 0.0
        if event is not None:
            delay = k - event[0]
            if 1 <= delay <= 2:
                cmd = -event[1] * amp
            if 3 <= delay <= 4 and pulse == "bipolar":
                cmd = event[1] * amp
        mx = 0.0
        for v, h in segments(np.array([0.0, cmd]), 0.001):
            for _ in range(4):
                x, tau, _ = m.step(x, v, h / 4, load=load)
                mx = max(mx, abs(np.rad2deg(x[-3]) - angle))
        reading = x[-4] + rng.normal(0, sigma)
        y.append(reading)
        if event is None and k < len(filt):
            weights = filt[k, : k + 1]
            estimate = gain * (weights @ y)
            if abs(estimate) >= max(0.12, 3.5 * abs(gain) * sigma * np.linalg.norm(weights)):
                event = (k, np.sign(estimate))
        log.append([k + 1, cmd, reading, np.rad2deg(x[-3]) - angle, x[-2] * 30 / np.pi, mx])
    np.savez_compressed(
        dest,
        trace=log,
        event_ms=event[0] + 1 if event else -1,
        load_nm=load,
        angle=angle,
        level=level,
        slices=slices,
        seed=seed,
        noise_a=sigma,
        voltage_v=amp,
    )
    print(dest.name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--level", type=int, default=4)
    p.add_argument("--slices", type=int, nargs="+", default=[1, 3])
    p.add_argument("--angles", type=float, nargs="+", default=[6.4, 17.3])
    p.add_argument("--amps", type=float, nargs="+", default=[0, 8, 16, 24])
    p.add_argument("--seeds", type=int, nargs="+", default=[581])
    p.add_argument("--noise", type=float, default=0.003)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pulse", choices=("bipolar", "unipolar"), default="bipolar")
    p.add_argument("--periods", type=int, default=24)
    a = p.parse_args()
    if a.periods != 24:
        a.folder = a.folder / f"ms{a.periods}"
    if a.pulse != "bipolar":
        a.folder = a.folder / a.pulse
    a.folder.mkdir(parents=True, exist_ok=True)
    jobs = [
        (a.folder, a.level, s, ang, ld, seed, v, a.noise, a.pulse, a.periods)
        for s in a.slices
        for ang in a.angles
        for ld in (-0.02, 0.02)
        for seed in a.seeds
        for v in a.amps
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(run, jobs))

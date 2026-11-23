"""Locked-shaft torque under stator rotating test voltages at fixed mechanical angles."""

import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_air56_terminal_servo import Machine, np


def sample(angle, level, modes, skew, freq, amplitude, sign, cycles=6, count=12):
    motor = Machine(level, modes, skew)
    a = np.deg2rad(angle)
    x = motor.equilibrium(a)
    values = []
    for k in range(cycles * count):
        phase = sign * 2 * np.pi * (k + 0.5) / count
        u = amplitude * np.array([np.cos(phase), np.sin(phase)])
        x, tau, _ = motor.step(x, u, 1 / (freq * count))
        x[-3:] = [a, 0.0, 0.0]
        values.append([tau, *x[-5:-3]])
    return np.array(values)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--angle", type=float, default=27.0)
    p.add_argument("--skew", action="store_true")
    p.add_argument("--modes", type=int, default=0)
    p.add_argument("--freq", type=float, default=50.0)
    p.add_argument("--amplitude", type=float, default=25.0)
    a = p.parse_args()
    for sign in (-1, 1):
        data = sample(a.angle, 3, a.modes, a.skew, a.freq, a.amplitude, sign)
        np.savez_compressed(
            a.out / f"locked_{a.angle}_{a.modes}_{int(a.skew)}_{a.amplitude}_{sign}.npz", data=data
        )
        print(
            sign,
            np.mean(data[-12:, 0]),
            np.mean(data[-24:-12, 0]),
            np.max(abs(data[-12:, 1:])),
            flush=True,
        )

"""Score the one-atan2 readout on raw nonlinear currents, including complementary probes."""

import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_air56_skew_audit import write_csv


def signal(data):
    y = data["read_a"]
    return y[0] + 1j * y[1]


def run(out):
    zero = np.load(out / "spin_0_0.npz")
    minus = np.load(out / "spincheck_calminus.npz")
    check = np.load(out / "spincheck_opposite.npz")
    sym = np.max(abs(minus["trace"][:, :2] + check["trace"][:, :2]))
    sym2 = np.max(
        abs(
            np.load(out / "spincheck_checkminus.npz")["trace"][:, :2]
            + np.load(out / "spin_247_0.npz")["trace"][:, :2]
        )
    )
    if max(sym, sym2) > 2e-5:
        raise RuntimeError("voltage sign symmetry mismatch")
    cal = (signal(zero) + signal(minus)) / 2
    rows = []
    for bias, angles in (("0", np.arange(7, 338, 30)), ("0.35", np.array([45, 225]))):
        src = [np.load(out / f"spin_{a}_{bias}.npz") for a in angles]
        unpaired = np.array([signal(d) for d in src])
        paired = (unpaired - np.roll(unpaired, len(angles) // 2)) / 2
        for mode, data, base in (("single", unpaired, signal(zero)), ("opposed", paired, cal)):
            for cycles in (1, 4, 16):
                for ma in (0.0, 0.3, 1.0, 3.0):
                    rng = np.random.default_rng(545321)
                    trial = np.repeat(data, 256)
                    # One complex spatial sequence combines two stator channels.
                    # A pair averages two independently sampled bursts.
                    sd = ma * 0.001 * np.sqrt(4 / (160 * cycles * (2 if mode == "opposed" else 1)))
                    trial += sd * (rng.normal(size=len(trial)) + 1j * rng.normal(size=len(trial)))
                    estimate = -np.angle(trial / base)
                    err = abs(
                        np.rad2deg(np.angle(np.exp(1j * (estimate - np.deg2rad(np.repeat(angles, 256))))))
                    )
                    rows.append(
                        dict(
                            bias_a=bias,
                            method=mode,
                            equivalent_cycles=cycles,
                            adc_noise_ma=ma,
                            n=len(err),
                            angles=len(angles),
                            min_signal_ma=1e3 * min(abs(data)),
                            max_signal_ma=1e3 * max(abs(data)),
                            p95_deg=np.percentile(err, 95),
                            wrong_half=int((err > 90).sum()),
                            opposed_check_a=max(sym, sym2),
                        )
                    )
    write_csv(out / "nonlinear_diode_angle.csv", rows)
    # A motor-specific six-point waveform catalogue. Its rotor poses are
    # distinct from both the full-turn test grid and the refined-mesh sector.
    ref = [cal]

    def map_signal(angle):
        return signal(np.load(out / f"spinmap_{angle}.npz"))

    for angle in range(10, 60, 10):
        ref.append((map_signal(angle) - map_signal(angle + 180)) / 2)
    ref = np.array(ref)
    nodes = np.arange(36) * np.pi / 18
    full = np.concatenate([ref * np.exp(-1j * k * np.pi / 3) for k in range(6)])
    grid = np.arange(1440) * np.pi / 720
    templates = np.interp(grid, nodes, full, period=2 * np.pi)
    table = []
    for name, angles in (
        ("full_turn", np.arange(7, 338, 30)),
        ("refined_mesh", np.array([12, 22, 32, 42, 52, 192, 202, 212, 222, 232])),
    ):
        inp = np.array(
            [
                signal(np.load(out / f"spin_{a}_0.npz")) if name == "full_turn" else map_signal(a)
                for a in angles
            ]
        )
        paired = (inp - np.roll(inp, len(inp) // 2)) / 2
        actual = np.repeat(np.deg2rad(angles), 256)
        for cycles in (4, 16, 64):
            for ma in (0.0, 0.3, 1.0, 3.0):
                rng = np.random.default_rng(92701)
                sd = ma * 0.001 * np.sqrt(2 / (160 * cycles))
                y = np.repeat(paired, 256) + sd * (
                    rng.normal(size=len(actual)) + 1j * rng.normal(size=len(actual))
                )
                pick = np.argmin(abs(y[:, None] - templates), axis=1)
                err = abs(np.rad2deg(np.angle(np.exp(1j * (grid[pick] - actual)))))
                table.append(
                    dict(
                        case=name,
                        n=len(err),
                        cycles=cycles,
                        raw_noise_ma=ma,
                        p95_deg=np.percentile(err, 95),
                        worst_deg=err.max(),
                        wrong_half=int((err > 90).sum()),
                    )
                )
    write_csv(out / "calibrated_diode.csv", table)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args().out)

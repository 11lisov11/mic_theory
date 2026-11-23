"""Evaluate a single current burst against independent nonlinear transient FEM records."""

import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.score_air56_polarity import interpolate
from tools.run_air56_skew_audit import write_csv
from tools.run_air56_direct_polar_drive import read_angle, K


def centered(x):
    return (x - x.mean(axis=-2, keepdims=True)).reshape(*x.shape[:-2], -1)


def table_estimate(data, cal):
    grid = np.deg2rad(np.arange(0, 360, 0.25))
    bank = centered(interpolate(cal["angle"], cal["current"], grid))
    flat = centered(data)
    return grid[np.argmax(flat @ bank.T - 0.5 * (bank**2).sum(axis=1), axis=-1)]


def record(data, bias):
    start, mid, end = data[:-2:2], data[1::2], data[2::2]
    n = len(start)
    p = np.arange(n)
    axes = p % 3 * np.pi / 3
    v = (
        82.0
        * np.where((p // 3) % 2 == 0, 1.0, -1.0)[:, None]
        * np.stack((np.cos(axes), np.sin(axes)), axis=-1)
    )
    dc = 27.5 / 3 * bias * np.array([0.8, 0.6])
    return np.column_stack(((start + mid) / 2, dc + v, (end + mid) / 2, dc - v, 2 * mid - start - end))


def evaluate(root):
    rows = []
    for name in ("blank", "hold", "load"):
        test = np.load(root / f"burst_{name}.npz")
        y, truth = test["current"], test["angle"]
        opposite = np.sqrt(np.mean((y - np.roll(y, len(y) // 2, axis=0))[:, 1:] ** 2, axis=(1, 2))) * 1000
        cal0 = np.load(root / "burst_cal.npz")
        for method, cal in (
            ("ROM", None),
            ("table_coarse_time", cal0),
            ("table_matched_time", np.load(root / "burst_timecal.npz")),
            (
                "table_measured_bias",
                np.load(root / ("burst_loadcal.npz" if name == "load" else "burst_timecal.npz")),
            ),
        ):
            for noise in (0.0, 0.3, 1.0, 3.0):
                # The analytic ROM fit is a transfer diagnostic; Monte Carlo uses
                # the same actual FEM angles, with more noise draws for the table.
                repeats = 8 if method == "ROM" else 128
                rng = np.random.default_rng(11619)
                measured = np.repeat(y, repeats, axis=0)
                measured += rng.normal(0, noise * 0.001, measured.shape)
                if method == "ROM":
                    estimate = np.array(
                        [read_angle(record(v, float(test["bias"])), 0.0, True, K) for v in measured]
                    )
                else:
                    estimate = table_estimate(measured, cal)
                err = abs(np.rad2deg(np.angle(np.exp(1j * (estimate - np.repeat(truth, repeats))))))
                rows.append(
                    dict(
                        case=name,
                        method=method,
                        raw_noise_ma=noise,
                        n=len(err),
                        opposed_rms_min_ma=opposite.min(),
                        opposed_rms_max_ma=opposite.max(),
                        p95_deg=np.percentile(err, 95),
                        wrong_half=int((err > 90).sum()),
                        worst_node_p95_deg=np.percentile(err.reshape(len(y), repeats), 95, axis=1).max(),
                    )
                )
    write_csv(root / "burst_scores.csv", rows)
    pairs = []
    for name in ("hold", "load"):
        check = np.load(root / f"burst_symmetry_{name}.npz")["max_diff_a"].max()
        if check > 1e-5:
            raise ValueError("opposite voltage trace symmetry check failed")
        test = np.load(root / f"burst_{name}.npz")
        families = ("loadcal", "refinedload") if name == "load" else ("timecal", "refined")
        for family in families:
            cal = np.load(root / f"burst_{family}.npz")
            theta = test["angle"]
            bank = cal["current"]
            grid = np.deg2rad(np.arange(0, 360, 0.25))
            even = centered(
                interpolate(cal["angle"], (bank + np.roll(bank, len(bank) // 2, axis=0)) / 2, grid)
            )
            odd = centered(
                interpolate(cal["angle"], (bank - np.roll(bank, len(bank) // 2, axis=0)) / 2, grid)
            )
            truth = np.repeat(theta, 128)
            plus = np.repeat(test["current"], 128, axis=0)
            minus = -np.repeat(np.roll(test["current"], len(theta) // 2, axis=0), 128, axis=0)
            for repeat in (1, 4):
                for ma in (0.3, 1.0, 3.0):
                    rng = np.random.default_rng(15067)
                    n1, n2 = (rng.normal(0, ma * 0.001 / np.sqrt(repeat), plus.shape) for _ in range(2))
                    # The two records start at the corresponding equilibria. Their
                    # sum isolates the sign-sensitive part, canceling even mesh bias.
                    common = centered((plus + n1 - minus - n2) / 2)
                    polar = centered((plus + n1 + minus + n2) / 2)
                    half = len(grid) // 2
                    axis = (common @ even[:half].T - 0.5 * (even[:half] ** 2).sum(axis=1)).argmax(axis=1)
                    branch = np.sum(polar * odd[axis], axis=1) < 0
                    estimate = grid[axis + half * branch]
                    err = abs(np.rad2deg(np.angle(np.exp(1j * (estimate - truth)))))
                    pairs.append(
                        dict(
                            case=name,
                            calibration=family,
                            pairs=repeat,
                            noise_ma=ma,
                            n=len(err),
                            p95_deg=np.percentile(err, 95),
                            wrong_half=int((err > 90).sum()),
                            symmetry_check_a=check,
                        )
                    )
    write_csv(root / "burst_pairs.csv", pairs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    evaluate(p.parse_args().out)

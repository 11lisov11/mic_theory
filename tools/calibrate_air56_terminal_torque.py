"""Calibrate an observer's static torque error against a higher-resolution FE mesh."""

import argparse, sys
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_air56_free_field import BandField
from fem.air56.p1_volume import P1Volume


def static(field, volume, a, amps):
    def residual(p):
        r, t = volume.field(p)
        return r + field.extra @ p - field.pm - field.W @ amps, t

    for _ in range(45):
        r, t = residual(a)
        norm = np.linalg.norm(r)
        if norm < 1e-6:
            return a
        d = splu((volume.matrix(t) + field.extra).tocsc(), permc_spec="MMD_AT_PLUS_A").solve(r)
        for damp in 2.0 ** -np.arange(15):
            candidate = a - damp * d
            if np.linalg.norm(residual(candidate)[0]) < norm:
                a = candidate
                break
        else:
            raise RuntimeError("DC continuation")
    raise RuntimeError("DC iteration limit")


def features(i):
    a, b = i.T
    return np.column_stack([np.ones(len(i)), a, b, a * a, a * b, b * b])


def calibrate(out, part, ring):
    # Periodic three-phase winding with identical 60 degree sectors.
    angle = np.arange(part, 60.0, 4.0)  # coarse chunks interleaved below
    angle = (angle[:, None] + np.arange(0.0, 1.0, 0.25)).ravel()
    currents = np.array(
        [
            [0, 0],
            [0.25, 0],
            [-0.25, 0],
            [0, 0.25],
            [0, -0.25],
            [0.22, 0.22],
            [-0.22, 0.22],
            [0.22, -0.22],
            [-0.22, -0.22],
        ]
    )
    X = features(currents)
    fit = []
    for level in (3, 4):
        field = BandField(0, level, **(dict(pole_half_width_deg=90.0, remanence_t=0.8) if ring else {}))
        volume = P1Volume(field)
        prior = [np.zeros(len(field.free)) for _ in currents]
        series = []
        for a in angle:
            field.update(np.deg2rad(a))
            values = []
            for j, i in enumerate(currents):
                prior[j] = static(field, volume, prior[j], i)
                values.append(field.torque(prior[j]))
            series.append(np.linalg.lstsq(X, values, rcond=None)[0])
        fit.append(series)
    np.savez_compressed(
        out / f"torque_fit_{int(ring)}_{part}.npz",
        angle=np.deg2rad(angle),
        coeff=np.subtract(fit[1], fit[0]),
    )


class TorqueCorrection:
    def __init__(self, out, ring):
        chunks = [np.load(out / f"torque_fit_{int(ring)}_{j}.npz") for j in range(4)]
        grid = np.concatenate([c["angle"] for c in chunks])
        order = np.argsort(grid)
        self.coeff = np.concatenate([c["coeff"] for c in chunks])[order]
        self.step = np.pi / 3 / len(grid)
        if not np.allclose(grid[order], self.step * np.arange(len(grid))):
            raise ValueError("incomplete calibration grid")

    def evaluate(self, theta, i):
        def polynomial(index, sector):
            phi = sector * np.pi / 3
            rotate = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
            a, b = rotate @ i
            c = self.coeff[index % len(self.coeff)]
            val = c @ np.array([1, a, b, a * a, a * b, b * b])
            grad = np.array([c[1] + 2 * c[3] * a + c[4] * b, c[2] + c[4] * a + 2 * c[5] * b]) @ rotate
            return val, grad

        sector = np.floor(theta / (np.pi / 3))
        loc = (theta - sector * np.pi / 3) / self.step
        k = min(int(loc), len(self.coeff) - 1)
        fraction = loc - k
        v0, g0 = polynomial(k, sector)
        v1, g1 = polynomial(k + 1, sector + (k + 1 == len(self.coeff)))
        return v0 + fraction * (v1 - v0), (v1 - v0) / self.step, g0 + fraction * (g1 - g0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--part", type=int, required=True, choices=range(4))
    p.add_argument("--ring", action="store_true")
    a = p.parse_args()
    calibrate(a.out, a.part, a.ring)

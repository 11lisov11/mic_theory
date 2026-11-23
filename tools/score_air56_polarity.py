"""Two-coordinate absolute-angle decoder and cross-mesh scoring of rotor polarity."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_air56_skew_audit import write_csv

TAU = 2 * np.pi


def interpolate(nodes, values, theta):
    flat = values.reshape(len(nodes), -1)
    return np.stack([np.interp(theta, nodes, v, period=TAU) for v in flat.T], axis=-1).reshape(
        theta.shape + values.shape[1:]
    )


def port(data, skew, angles):
    nodes, weights = np.polynomial.legendre.leggauss(5)
    impedance = sum(
        w / 2 * interpolate(data["angles"], data["z"], angles + n * skew * np.pi / 18)
        for n, w in zip(nodes, weights)
    )
    return np.linalg.inv(impedance)


def features(y):
    diff = np.stack((y[..., 0, :, :] - y[..., 1, :, :], y[..., 2, :, :] - y[..., 3, :, :]), axis=-3)
    flat = diff.reshape(*diff.shape[:-3], -1)
    return np.concatenate((flat.real, flat.imag), axis=-1)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-15)


def axis_features(y):
    mean = y.mean(axis=-3)
    vec = np.stack((mean[..., 0, 0] - mean[..., 1, 1], mean[..., 1, 0] + mean[..., 0, 1]), axis=-1)
    return np.concatenate((vec.real, vec.imag), axis=-1)


class PolarDecoder:
    def __init__(self, theta, admittance):
        # Fixed linear regression fitted at known calibration angles only.
        harmonics = np.stack((np.cos(theta), np.sin(theta)), axis=1)
        self.weights = np.linalg.lstsq(features(admittance), harmonics, rcond=1e-4)[0]
        self.grid = np.arange(720) * TAU / 720
        self.templates = unit(interpolate(theta, features(admittance), self.grid))
        second = np.stack((np.cos(2 * theta), np.sin(2 * theta)), axis=1)
        axial = axis_features(admittance)
        self.axis_offset = axial.mean(axis=0)
        self.axis_inverse = np.linalg.pinv(
            np.linalg.lstsq(second, axial - self.axis_offset, rcond=None)[0]
        )

    def read(self, y, method):
        values = features(y)
        if method == "harmonic":
            xy = values @ self.weights
            return np.arctan2(xy[..., 1], xy[..., 0])
        if method == "axis_and_sign":
            xy = (axis_features(y) - self.axis_offset) @ self.axis_inverse
            axis = (np.arctan2(xy[..., 1], xy[..., 0]) / 2) % np.pi
            idx = (np.rint(axis * 720 / TAU).astype(int)) % 720
            same = (values * self.templates[idx]).sum(axis=-1) >= 0
            return np.where(same, axis, axis + np.pi)
        return self.grid[np.argmax(values @ self.templates.T, axis=-1)]


def score(root):
    cal = np.load(root / "L4B0.4.npz")
    rows = []
    diagnostics = []
    for name in ("L4B0", "L4B0.4", "L6B0.2", "L6B0.4", "L7bg0", "L6bg0.35", "edm4", "edm7"):
        src = np.load(root / f"{name}.npz")
        for skew in (0, 1):
            theta = src["angles"]
            y = port(src, skew, theta)
            dy = (y - np.roll(y, len(y) // 2, axis=0)).reshape(len(y), -1)
            centered = (y - y.mean(axis=0)).reshape(len(y), -1)
            torque = np.nan
            if "coenergy_j" in src:
                coenergy = src["coenergy_j"]
                torque = np.max(
                    abs(
                        (np.roll(coenergy, -1, axis=0) - np.roll(coenergy, 1, axis=0))
                        / (2 * TAU / len(theta))
                    )
                )
            diagnostics.append(
                dict(
                    case=name,
                    skew=skew,
                    max_contrast_ma=1e3 * np.linalg.norm(centered, axis=1).max(),
                    min_opposed_ma=1e3 * np.linalg.norm(dy, axis=1).min(),
                    max_opposed_ma=1e3 * np.linalg.norm(dy, axis=1).max(),
                    mag_area_mm2=float(src["area_m2"].mean() * 1e6) if "area_m2" in src else "",
                    peak_bias_torque_nm_unskewed=torque if np.isfinite(torque) and not skew else "",
                )
            )
            if name in ("L4B0", "L4B0.4", "edm4", "edm7"):
                continue
            decoder = PolarDecoder(cal["angles"], port(cal, skew, cal["angles"]))
            # Score only actual independent FEM nodes, with 128 independent noise
            # draws at every node. Axial slice averaging uses periodic interpolation.
            clean = np.repeat(y, 128, axis=0)
            truth = np.repeat(theta, 128)
            rng = np.random.default_rng(17821)
            white = rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)
            for kind, count in (("phasor", 1), ("adc_white_1cycle", 160), ("adc_white_4cycles", 640)):
                for ma in (0.3, 1.0, 3.0):
                    sd = ma * 0.001 * (1 if kind == "phasor" else np.sqrt(2 / count))
                    for method in ("harmonic", "table", "axis_and_sign"):
                        estimated = decoder.read(clean + sd * white, method)
                        error = abs(np.rad2deg(np.angle(np.exp(1j * (estimated - truth)))))
                        rows.append(
                            dict(
                                case=name,
                                skew=skew,
                                noise_ma=ma,
                                domain=kind,
                                method=method,
                                n=len(truth),
                                p50_deg=np.median(error),
                                p95_deg=np.percentile(error, 95),
                                max_angle_p95_deg=np.percentile(
                                    error.reshape(len(y), 128), 95, axis=1
                                ).max(),
                                wrong_half_count=int((error > 90).sum()),
                                inside_5=int((error <= 5).sum()),
                            )
                        )
    write_csv(root / "absolute_angle.csv", rows)
    write_csv(root / "contrast.csv", diagnostics)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    score(p.parse_args().out)

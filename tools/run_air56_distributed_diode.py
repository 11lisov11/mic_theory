"""A full-pitch, field-powered passive rectifier winding inside the cage rotor."""

import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import circulant
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu
from skfem import Basis, ElementTriP2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fem.air56.sliding_section import make_section, winding_rhs
from fem.air56.stator import magnetic, current
from fem.air56.coupon import MU0
from tools.run_air56_skew_audit import article_profile, write_csv

TURNS = 8
WIRE_MM2 = 0.12
RCOIL = 1.72e-8 * TURNS * (0.108 + 0.094) / (WIRE_MM2 * 1e-6)
FREQ = 100.0


def ports(theta, level, harmonics):
    geom, design = article_profile()
    sec = make_section(geom, theta, level, design, int(level > 4))
    b = Basis(sec.mesh, ElementTriP2(), intorder=4)
    f = b.complement_dofs(b.get_dofs())
    x, y = sec.mesh.p[:, sec.mesh.t].mean(axis=1)
    r = np.hypot(x, y)
    phi = np.arctan2(y, x) - theta
    mask = (r > 0.0222) & (r < 0.02355) & sec.steel
    pockets = [
        mask & (abs((phi - j * np.pi + np.pi) % (2 * np.pi) - np.pi) < np.deg2rad(4)) for j in (0, 1)
    ]
    area = b.dx.sum(axis=1)
    source = sum((-1) ** j * TURNS * p / area[p].sum() for j, p in enumerate(pockets))
    if any(area[p].sum() * 1e6 < TURNS * WIRE_MM2 for p in pockets):
        raise ValueError("insufficient copper area")
    stator = winding_rhs(geom, sec, b)
    wind = np.column_stack((stator, current.assemble(b, source=source[:, None])))[f]
    nu = 1 / (MU0 * np.where(sec.steel & ~(pockets[0] | pockets[1]), geom.mu_r, 1.0))
    sigma = sec.cage * geom.aluminium_sigma
    integral = current.assemble(b, source=sigma[:, None])[f]
    total = sigma @ area
    output = []
    for k in range(1, harmonics + 1):
        w = 2 * np.pi * k * FREQ
        op = magnetic.assemble(b, nu=nu[:, None], electric=(1j * w * sigma)[:, None])[f][:, f].tocsc()
        float_rhs = 1j * w * integral
        v = splu(op).solve(np.column_stack((wind, float_rhs)).astype(complex))
        vals = v[:, :3] + v[:, 3, None] * (
            (float_rhs @ v[:, :3]) / (1j * w * total - float_rhs @ v[:, 3])
        )
        z = 1j * w * geom.stack_length * (wind.T @ vals)
        if np.linalg.norm(z - z.T) / np.linalg.norm(z) > 1e-7:
            raise ValueError("three-port reciprocity")
        if np.linalg.eigvalsh(z.real).min() < -1e-9:
            raise ValueError("port passivity")
        z[:2] *= 2 / 3
        z += np.diag([27.5 / 3, 27.5 / 3, RCOIL + 1j * w * 4e-6])
        output.append(z)
    return np.array(output), [area[p].sum() * 1e6 for p in pockets]


def diode(z, drive, voltage, threshold, enabled=True):
    # Passive secondary terminated by a one-way element. Stator voltages are
    # clamped at harmonics. Positive j is delivered to the nonlinear load.
    zss = z[:, :2, :2]
    row = z[:, 2, :2]
    col = z[:, :2, 2]
    reflected = np.linalg.solve(zss, col[..., None])[..., 0]
    impedance = z[:, 2, 2] - np.sum(row * reflected, axis=-1)
    n = 2 * len(z)
    voc = row[0] @ np.linalg.solve(zss[0], voltage * drive)
    t = np.arange(n) * 2 * np.pi / n
    driving = (voc * np.exp(1j * t)).real
    spectrum = np.r_[RCOIL, impedance]
    kernel = np.fft.irfft(spectrum, n=n)
    matrix = circulant(kernel)
    j = np.zeros(n)
    if enabled and driving.max() > threshold:
        active = driving > threshold
        for _ in range(8 * n):
            j[:] = 0.0
            j[active] = np.linalg.solve(matrix[np.ix_(active, active)], (driving - threshold)[active])
            if j.min() < -1e-9:
                active[j.argmin()] = False
                continue
            slack = threshold - driving + matrix @ j
            if slack.min() < -1e-9:
                active[slack.argmin()] = True
                continue
            break
        else:
            scale = max(RCOIL, float(abs(impedance[0])))
            eye = np.eye(n)

            def fb(v, jac=False):
                slack = (threshold - driving + matrix @ v) / scale
                norm = np.sqrt(v * v + slack * slack + 1e-24)
                if jac:
                    return (v / norm - 1)[:, None] * eye + (slack / norm - 1)[:, None] * matrix / scale
                return norm - v - slack

            sol = least_squares(
                fb,
                np.maximum(0, j),
                jac=lambda v: fb(v, True),
                gtol=1e-12,
                xtol=1e-13,
                ftol=1e-13,
                max_nfev=500,
            )
            j = sol.x
            if np.max(abs(fb(j))) > 2e-7:
                raise RuntimeError("diode complementarity failed")
    ph = 2 * np.fft.rfft(j) / n
    secondary = reflected[1] * ph[2]
    return secondary, abs(voc), j.max(), np.mean(j * j) * RCOIL, np.mean(j) * threshold


def run(out, level, count, harmonics):
    nodes = (np.arange(count) + 0.5 * (level > 4)) * 2 * np.pi / count
    z = []
    areas = []
    for a in nodes:
        port, area = ports(a, level, harmonics)
        z.append(port)
        areas.append(area)
    np.savez_compressed(out / f"diode_mesh{level}.npz", z=z, angle=nodes, area_mm2=areas)


def extract(data, skew, voltage, threshold, harmonics):
    from tools.score_air56_polarity import interpolate

    angles = data["angle"]
    # Co-moving through winding follows skew across the stack. Its current is
    # shared just like stator currents. Average impedances in series first.
    grid, weights = np.polynomial.legendre.leggauss(5)
    z = sum(
        w / 2 * interpolate(angles, data["z"][:, :harmonics], angles + skew * g * np.pi / 18)
        for g, w in zip(grid, weights)
    )
    y = []
    extras = []
    for point in z:
        observations = []
        for drive in np.eye(2):
            current2, *info = diode(point, drive, voltage, threshold)
            observations.append(current2)
            extras.append(info)
        y.append(np.ravel(observations))
    y = np.array(y)
    return np.column_stack((y.real, y.imag)), np.array(extras)


def score(out):
    from tools.score_air56_polarity import interpolate

    cal = np.load(out / "diode_mesh4.npz")
    test = np.load(out / "diode_mesh6.npz")
    rows = []
    mesh_checks = []
    for skew in (0, 1):
        for v in (1.0, 5.0, 12.0, 24.0):
            for threshold in (0.15, 0.35):
                a, extra = extract(cal, skew, v, threshold, 24)
                b, validation = extract(test, skew, v, threshold, 24)
                lower, _ = extract(test, skew, v, threshold, 16)
                mesh_checks.append(
                    dict(
                        skew=skew, v=v, vf=threshold, max_harmonic_change_ma=1e3 * np.max(abs(b - lower))
                    )
                )
                half = b - np.roll(b, len(b) // 2, axis=0)
                grid = np.arange(720) * np.pi / 360
                templates = interpolate(cal["angle"], a, grid)
                truth = np.repeat(test["angle"], 128)
                clean = np.repeat(b, 128, axis=0)
                for cycles in (1, 4, 16):
                    for ma in (0.3, 1.0, 3.0):
                        rng = np.random.default_rng(141912)
                        # Two sequential excitations, 1 cycle each, 160 ADC samples per axis.
                        # Exact synchronous demodulation in an ideal white-noise channel.
                        noisy = clean + ma * 0.001 * np.sqrt(2 / (160 * cycles)) * rng.normal(
                            size=clean.shape
                        )
                        estimated = grid[
                            (noisy @ templates.T - 0.5 * (templates**2).sum(axis=1)).argmax(axis=1)
                        ]
                        err = abs(np.rad2deg(np.angle(np.exp(1j * (estimated - truth)))))
                        rows.append(
                            dict(
                                skew=skew,
                                cycles_per_axis=cycles,
                                drive_v=v,
                                threshold_v=threshold,
                                raw_noise_ma=ma,
                                n=len(truth),
                                min_opposed_ma=1000 * np.linalg.norm(half, axis=1).min(),
                                max_opposed_ma=1000 * np.linalg.norm(half, axis=1).max(),
                                max_induced_v=validation[:, 0].max(),
                                peak_secondary_a=validation[:, 1].max(),
                                max_coil_w=validation[:, 2].max(),
                                max_diode_w=validation[:, 3].max(),
                                p95_deg=np.percentile(err, 95),
                                wrong_half=int((err > 90).sum()),
                            )
                        )
    write_csv(out / "diode_angle.csv", rows)
    write_csv(out / "diode_harmonics.csv", mesh_checks)
    rotating = []
    for skew in (0, 1):

        def sequence(model, v, vf):
            angles = model["angle"]
            nodes, weights = np.polynomial.legendre.leggauss(5)
            z = sum(
                w / 2 * interpolate(angles, model["z"], angles + skew * d * np.pi / 18)
                for d, w in zip(nodes, weights)
            )
            y = np.array([diode(p, np.array([1.0, -1j]), v, vf)[0] for p in z])
            return y[:, 0] + 1j * y[:, 1]

        for v in (12.0, 24.0, 36.0):
            for vf in (0.15, 0.35):
                u = sequence(cal, v, vf)
                s = sequence(test, v, vf)
                hc = [np.mean(u * np.exp(-1j * m * cal["angle"])) for m in (-1, 1)]
                idx = int(np.argmax(abs(np.array(hc))))
                order = (-1, 1)[idx]
                for cycles in (1, 4, 16):
                    rng = np.random.default_rng(18491)
                    sigma = 0.003 * np.sqrt(4 / (160 * cycles))
                    y = np.repeat(s, 128) + sigma * (
                        rng.normal(size=len(s) * 128) + 1j * rng.normal(size=len(s) * 128)
                    )
                    angle = order * (np.angle(y) - np.angle(hc[idx]))
                    err = abs(np.rad2deg(np.angle(np.exp(1j * (angle - np.repeat(test["angle"], 128))))))
                    rotating.append(
                        dict(
                            skew=skew,
                            v=v,
                            vf=vf,
                            cycles=cycles,
                            raw_noise_ma=3.0,
                            order=order,
                            calibration_ma=abs(hc[idx]) * 1000,
                            min_signal_ma=min(abs(s)) * 1000,
                            p95_deg=np.percentile(err, 95),
                            wrong_half=int((err > 90).sum()),
                            n=len(err),
                        )
                    )
    write_csv(out / "diode_rotating.csv", rotating)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", type=int)
    p.add_argument("--count", type=int, default=24)
    p.add_argument("--harmonics", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.level:
        run(a.out, a.level, a.count, a.harmonics)
    else:
        score(a.out)

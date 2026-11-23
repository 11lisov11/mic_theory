"""Independent nonlinear time-domain FEM of the polarity burst at locked shaft."""

import argparse
from pathlib import Path
import sys
from dataclasses import replace

import numpy as np
from scipy.sparse import bmat, csc_matrix, eye
from scipy.sparse.linalg import splu
from skfem import Basis, ElementTriP1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fem.air56.coupon import MU0
from fem.air56.loaded_section import cage_matrices, magnetic_force, material, tensor_matrix
from fem.air56.sliding_section import make_section, winding_rhs
from tools.run_air56_skew_audit import article_profile

H = 0.00025
R = 27.5 / 3


def burst(theta, level, bias, steps_per_pulse, cycles, br, invert=False):
    geom, design = article_profile()
    s = make_section(geom, theta, level, replace(design, bar_width_modulation=0.6), int(level > 3))
    b = Basis(s.mesh, ElementTriP1(), intorder=2)
    free = b.complement_dofs(b.get_dofs())
    x, y = s.mesh.p[:, s.mesh.t].mean(axis=1)
    r = np.hypot(x, y)
    rel = (np.arctan2(y, x) - theta + np.pi / 2) % np.pi - np.pi / 2
    cut = (
        s.steel & (abs(rel) < np.deg2rad(8)) & (r > geom.rotor_radius - 0.0005) & (r < geom.rotor_radius)
    )
    s.steel &= ~cut
    m0 = cut[:, None] * (br / MU0)
    pm = magnetic_force.assemble(b, hx=-np.sin(theta) * m0, hy=np.cos(theta) * m0)[free]
    W = winding_rhs(geom, s, b)[free]
    weights, g, total = cage_matrices(b, np.where(s.cage, geom.aluminium_sigma, 0), free)
    h = H / steps_per_pulse
    coil = csc_matrix(W)
    floating = csc_matrix(g[:, None])
    lam = (2 / 3) * geom.stack_length
    zero = np.zeros(len(free))
    polarity = -1.0 if invert else 1.0
    current0 = polarity * bias * np.array([0.8, 0.6])
    old = zero.copy()

    def steel(a):
        values = np.zeros(b.N)
        values[free] = a
        gx, gy = b.interpolate(values).grad[..., 0]
        nu, t = material(gx, gy, s.steel)
        force = magnetic_force.assemble(b, hx=(gx * nu)[:, None], hy=(gy * nu)[:, None])[free] - pm
        return force, t

    # Initialize the equilibrium permanent magnet and any specified holding current.
    for _ in range(40):
        f, t = steel(old)
        f -= W @ current0
        e = np.linalg.norm(f)
        if e < 1e-7 * max(1.0, np.linalg.norm(pm), np.linalg.norm(W @ current0)):
            break
        step = splu(tensor_matrix(b, t)[free][:, free].tocsc()).solve(f)
        for d in 2.0 ** -np.arange(15):
            candidate = old - d * step
            if np.linalg.norm(steel(candidate)[0] - W @ current0) < e:
                old = candidate
                break
        else:
            raise RuntimeError("DC line search")
    else:
        raise RuntimeError("DC initialization")
    drive = np.r_[old, 0.0, current0]
    history = [current0]
    max_residual = 0.0
    for k in range(12 * cycles):
        pair = k // 2
        axis = pair % 3 * np.pi / 3
        v = R * current0 + polarity * 82.0 * ((-1) ** (k % 2 + (pair // 3) % 2)) * np.array(
            [np.cos(axis), np.sin(axis)]
        )
        for _ in range(steps_per_pulse):
            old = drive[:-3].copy()

            def residual(state):
                rate = (state[:-3] - old) / h
                force, tangent = steel(state[:-3])
                return np.r_[
                    force + weights @ rate - g * state[-3] - W @ state[-2:],
                    g @ rate - total * state[-3],
                    lam * (W.T @ rate) + R * state[-2:] - v,
                ], tangent

            for iteration in range(20):
                res, t = residual(drive)
                scaled = np.max(abs(res)) / max(1.0, np.max(abs(pm)), np.max(abs(v)))
                if scaled < 2e-8:
                    max_residual = max(max_residual, scaled)
                    break
                mat = bmat(
                    [
                        [tensor_matrix(b, t)[free][:, free] + weights / h, -floating, -coil],
                        [floating.T / h, csc_matrix([[-total]]), None],
                        [lam * coil.T / h, None, R * eye(2)],
                    ],
                    format="csc",
                )
                step = splu(mat).solve(res)
                for d in 2.0 ** -np.arange(13):
                    trial = drive - d * step
                    if np.linalg.norm(residual(trial)[0]) < np.linalg.norm(res):
                        drive = trial
                        break
                else:
                    raise RuntimeError("transient line search")
            else:
                raise RuntimeError("transient convergence")
        history.append(drive[-2:].copy())
    return np.array(history), max_residual


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--count", type=int, default=24)
    p.add_argument("--bias", type=float, default=0.0)
    p.add_argument("--substeps", type=int, default=1)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--br", type=float, default=0.4)
    p.add_argument("--invert", action="store_true")
    p.add_argument(
        "--check-opposite", type=Path, help="verify symmetry against three nodes of an existing burst"
    )
    args = p.parse_args()
    if args.check_opposite:
        source = np.load(args.check_opposite)
        angles = source["angle"]
        errors = []
        for j in (0, 3, 7):
            raw, _ = burst(
                angles[j],
                int(source["level"]),
                float(source["bias"]),
                int(source["substeps"]),
                (source["current"].shape[1] - 1) // 12,
                float(source["br"]),
                True,
            )
            errors.append(np.max(abs(raw + source["current"][(j + len(angles) // 2) % len(angles)])))
        np.savez(args.out, max_diff_a=errors)
        sys.exit(0)
    angles = (np.arange(args.count) + 0.5 * (args.level > 3)) * 2 * np.pi / args.count
    currents = []
    errors = []
    for theta in angles:
        i, e = burst(theta, args.level, args.bias, args.substeps, args.cycles, args.br, args.invert)
        currents.append(i)
        errors.append(e)
    np.savez_compressed(
        args.out,
        angle=angles,
        current=currents,
        max_residual=max(errors),
        bias=args.bias,
        br=args.br,
        substeps=args.substeps,
        level=args.level,
        h=H,
    )

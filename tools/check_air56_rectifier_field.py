"""Nonlinear transient cross-check of one field-powered rotor rectifier using only coil terminals."""

import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import splu
from skfem import Basis, ElementTriP1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_air56_distributed_diode import TURNS, RCOIL, FREQ
from tools.run_air56_free_field import Field, R
from fem.air56.loaded_section import cage_matrices
from fem.air56.sliding_section import make_section, winding_rhs
from fem.air56.stator import current
from tools.run_air56_skew_audit import article_profile


class RectifiedField(Field):
    def __init__(self, theta, level):
        geom, design = article_profile()
        sec = make_section(geom, theta, level, design, int(level > 4))
        b = self.b = Basis(sec.mesh, ElementTriP1(), intorder=2)
        f = self.free = b.complement_dofs(b.get_dofs())
        x, y = sec.mesh.p[:, sec.mesh.t].mean(axis=1)
        r = np.hypot(x, y)
        relative = np.arctan2(y, x) - theta
        pockets = [
            sec.steel
            & (r > 0.0222)
            & (r < 0.02355)
            & (abs((relative - k * np.pi + np.pi) % (2 * np.pi) - np.pi) < np.deg2rad(4))
            for k in (0, 1)
        ]
        self.steel = sec.steel & ~(pockets[0] | pockets[1])
        area = b.dx.sum(axis=1)
        source = sum((-1) ** k * TURNS * p / area[p].sum() for k, p in enumerate(pockets))
        wind = winding_rhs(geom, sec, b)
        self.W = wind[f]
        self.allwind = np.column_stack((wind, current.assemble(b, source=source[:, None])))[f]
        self.pm = np.zeros(len(f))
        self.mass, self.g, self.total = cage_matrices(b, np.where(sec.cage, geom.aluminium_sigma, 0.0), f)
        self.coil = csc_matrix(self.allwind)
        self.floating = csc_matrix(self.g[:, None])
        self.scale = geom.stack_length * np.array([2 / 3, 2 / 3, 1.0])

    def update(self, state, u, h, vf, diode_on):
        old = state[:-4].copy()
        old_ir = state[-1]
        for branch in range(2):
            closed = diode_on if branch == 0 else not diode_on
            z = np.diag([R, R, RCOIL + 4e-6 / h] if closed else [R, R, 1.0])
            linkage = self.scale[:, None] * self.allwind.T / h
            right = np.r_[u, (vf + 4e-6 / h * old_ir) if closed else 0.0]
            if not closed:
                linkage[-1] = 0.0

            def residual(s):
                rate = (s[:-4] - old) / h
                force, t = self.field(s[:-4])
                r = np.r_[
                    force + self.mass @ rate - self.g * s[-4] - self.allwind @ s[-3:],
                    self.g @ rate - self.total * s[-4],
                    linkage @ (s[:-4] - old) + z @ s[-3:] - right,
                ]
                return r, t

            s = state.copy()
            for _ in range(30):
                res, t = residual(s)
                if np.max(abs(res)) < 2e-6:
                    break
                mat = bmat(
                    [
                        [self.matrix(t) + self.mass / h, -self.floating, -self.coil],
                        [self.floating.T / h, csc_matrix([[-self.total]]), None],
                        [csc_matrix(linkage), None, csc_matrix(z)],
                    ],
                    format="csc",
                )
                step = splu(mat).solve(res)
                for damp in 2.0 ** -np.arange(16):
                    trial = s - damp * step
                    if np.linalg.norm(residual(trial)[0]) < np.linalg.norm(res):
                        s = trial
                        break
                else:
                    raise RuntimeError("nonlinear rotor loop line search")
            else:
                raise RuntimeError("rotor loop Newton convergence")
            rotor_voltage = (
                self.scale[-1] * (self.allwind[:, -1] @ (s[:-4] - old)) / h
                + 4e-6 / h * (s[-1] - old_ir)
                + RCOIL * s[-1]
            )
            if (closed and s[-1] <= 1e-8) or (not closed and rotor_voltage <= vf + 1e-8):
                return s, closed
        raise RuntimeError("diode complementarity")


def measure(out, angle, bias, cycles, level, voltage, axis, blank, invert=False):
    model = RectifiedField(np.deg2rad(angle), level)
    if invert:
        voltage = -voltage
        bias = -bias
    state = np.r_[model.equilibrium(bias)[:-2], bias, 0.0]
    trace = []
    on = False
    n = 40
    h = 1 / (FREQ * n)
    for k in range(cycles * n):
        phase = 2 * np.pi * (k + 0.5) / n
        test = np.array([np.cos(phase), np.sin(phase)]) if axis == 2 else np.eye(2)[axis] * np.cos(phase)
        volts = R * bias + voltage * test
        state, on = model.update(state, volts, h, 1e6 if blank else 0.15, on)
        trace.append([*state[-3:], on])
    values = np.array(trace)
    carrier = np.exp(-4j * np.pi * (np.arange(n) + 1) / n)
    y = 2 / n * (carrier @ values[-n:, :2])
    prior = 2 / n * (carrier @ values[-2 * n : -n, :2])
    np.savez_compressed(
        out,
        angle_deg=angle,
        bias=bias,
        read_a=y,
        last_cycle_change_a=abs(y - prior),
        trace=values,
        h=h,
        volts=voltage,
        axis=axis,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--angle", type=float, required=True)
    p.add_argument("--bias", type=float, default=0.0)
    p.add_argument("--volts", type=float, default=24.0)
    p.add_argument("--axis", type=int, default=0)
    p.add_argument("--blank", action="store_true")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--cycles", type=int, default=5)
    a = p.parse_args()
    measure(
        a.out,
        a.angle,
        a.bias * np.array([0.8, 0.6]),
        a.cycles,
        a.level,
        a.volts,
        a.axis,
        a.blank,
        a.invert,
    )

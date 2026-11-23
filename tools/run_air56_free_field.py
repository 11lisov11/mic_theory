"""Spatially resolved free-shaft check. Field, cage and winding states persist between control bursts."""

from pathlib import Path
import sys
import argparse
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
from tools.score_air56_polar_burst import centered
from tools.score_air56_polarity import interpolate
from tools.run_air56_polar_burst import H, R


class Field:
    def __init__(self, theta, level, *, pole_half_width_deg=8., remanence_t=.4):
        g, design = article_profile()
        s = make_section(g, theta, level, replace(design, bar_width_modulation=0.6), int(level > 3))
        self.b = b = Basis(s.mesh, ElementTriP1(), intorder=2)
        self.free = f = b.complement_dofs(b.get_dofs())
        x, y = s.mesh.p[:, s.mesh.t].mean(axis=1)
        r = np.hypot(x, y)
        rel = (np.arctan2(y, x) - theta + np.pi / 2) % np.pi - np.pi / 2
        cut = s.steel & (abs(rel) <= np.deg2rad(pole_half_width_deg)) & (r > g.rotor_radius - 0.0005) & (r < g.rotor_radius)
        s.steel &= ~cut
        self.steel = s.steel
        m = cut[:, None] * remanence_t / MU0
        self.pm = magnetic_force.assemble(b, hx=-np.sin(theta) * m, hy=np.cos(theta) * m)[f]
        self.W = winding_rhs(g, s, b)[f]
        self.mass, self.g, self.total = cage_matrices(b, np.where(s.cage, g.aluminium_sigma, 0), f)
        self.lam = 2 * g.stack_length / 3
        self.coil = csc_matrix(self.W)
        self.floating = csc_matrix(self.g[:, None])
        # Identify elements spanning the air gap by their material vertices.
        rr = np.hypot(*s.mesh.p)[s.mesh.t]
        self.gap = (rr.min(axis=0) >= g.rotor_radius - 1e-10) & (
            rr.max(axis=0) <= g.rotor_radius + g.air_gap + 1e-10
        )
        self.cos = x[self.gap] / r[self.gap]
        self.sin = y[self.gap] / r[self.gap]
        self.tw = g.stack_length / MU0 / g.air_gap * (b.dx.sum(axis=1) * r)[self.gap]

    def grad(self, a):
        values = np.zeros(self.b.N)
        values[self.free] = a
        return self.b.interpolate(values).grad[..., 0]

    def field(self, a):
        gx, gy = self.grad(a)
        nu, t = material(gx, gy, self.steel)
        return magnetic_force.assemble(self.b, hx=(gx * nu)[:, None], hy=(gy * nu)[:, None])[
            self.free
        ] - self.pm, t

    def torque(self, a):
        gx, gy = self.grad(a)[:, self.gap]
        br = gy * self.cos - gx * self.sin
        bt = -gy * self.sin - gx * self.cos
        return self.tw @ (br * bt)

    def matrix(self, t):
        return tensor_matrix(self.b, t)[self.free][:, self.free]

    def equilibrium(self, bias=None):
        current = np.zeros(2) if bias is None else np.asarray(bias)
        a = np.zeros(len(self.free))
        for _ in range(40):
            f, t = self.field(a)
            f -= self.W @ current
            if np.linalg.norm(f) < max(1.0, np.linalg.norm(self.pm)) * 1e-8:
                return np.r_[a, 0.0, current]
            step = splu(self.matrix(t).tocsc()).solve(f)
            for d in 2.0 ** -np.arange(13):
                candidate = a - d * step
                if np.linalg.norm(self.field(candidate)[0] - self.W @ current) < np.linalg.norm(f):
                    a = candidate
                    break
            else:
                raise RuntimeError("DC convergence")
        raise RuntimeError("DC iterations")

    def advance(self, state, voltage, h, depth=0):
        try:
            return self._advance(state, voltage, h)
        except RuntimeError:
            if depth >= 5:
                raise
            midpoint = self.advance(state, voltage, h / 2, depth + 1)
            return self.advance(midpoint, voltage, h / 2, depth + 1)

    def _advance(self, state, voltage, h):
        old = state[:-3].copy()

        def residual(v):
            rate = (v[:-3] - old) / h
            f, t = self.field(v[:-3])
            return np.r_[
                f + self.mass @ rate - self.g * v[-3] - self.W @ v[-2:],
                self.g @ rate - self.total * v[-3],
                self.lam * self.W.T @ rate + R * v[-2:] - voltage,
            ], t

        for _ in range(25):
            f, t = residual(state)
            if np.max(abs(f)) < 1e-6:
                return state
            matrix = bmat(
                [
                    [self.matrix(t) + self.mass / h, -self.floating, -self.coil],
                    [self.floating.T / h, csc_matrix([[-self.total]]), None],
                    [self.lam * self.coil.T / h, None, R * eye(2)],
                ],
                format="csc",
            )
            step = splu(matrix).solve(f)
            for d in 2.0 ** -np.arange(13):
                candidate = state - d * step
                if np.linalg.norm(residual(candidate)[0]) < np.linalg.norm(f):
                    state = candidate
                    break
            else:
                raise RuntimeError("transient line search")
        raise RuntimeError("transient iteration limit")


class BandField(Field):
    """Fourier Dirichlet-to-Neumann coupling across the circular air gap."""

    def __init__(self, theta, level, **pole_options):
        super().__init__(theta, level, **pole_options)
        g, _ = article_profile()
        self.length = g.stack_length
        self.origin = theta
        points = self.b.mesh.p[:, self.free]
        radii = np.hypot(*points)
        inner = np.flatnonzero(np.isclose(radii, g.rotor_radius, atol=1e-10, rtol=0))
        outer = np.flatnonzero(np.isclose(radii, g.rotor_radius + g.air_gap, atol=1e-10, rtol=0))
        self.nodes = np.r_[inner, outer]
        self.count = len(inner)
        log = np.log1p(g.air_gap / g.rotor_radius)
        harmonics = np.arange(1, max(len(inner), len(outer)) + 1)
        self.orders = harmonics
        k = harmonics * log
        self.coupling = np.pi / MU0 * harmonics / np.sinh(k)
        diagonal = np.pi / MU0 * harmonics / np.tanh(k)

        def boundary(indices):
            # Integrate the Fourier coefficients of every periodic piecewise
            # linear boundary basis by parts. High modes have 1/n^2 decay.
            order = np.argsort(np.arctan2(*points[::-1, indices]) % (2 * np.pi))
            pos = np.arctan2(*points[::-1, indices[order]]) % (2 * np.pi)
            widths = (np.roll(pos, -1) - pos) % (2 * np.pi)
            e = np.exp(-1j * harmonics[:, None] * pos)
            slopes = (np.roll(e, -1, axis=1) - e) / widths
            coeff = (np.roll(slopes, 1, axis=1) - slopes) / (np.pi * harmonics[:, None] ** 2)
            # Common sign of all nonconstant basis coefficients cancels in K.
            return coeff[:, np.argsort(order)], ((widths + np.roll(widths, 1)) / (4 * np.pi))[
                np.argsort(order)
            ]

        self.inner, mi = boundary(inner)
        self.outer, mo = boundary(outer)
        self.fixed = np.zeros((len(self.nodes), len(self.nodes)))
        for sl, coeff in ((slice(None, self.count), self.inner), (slice(self.count, None), self.outer)):
            self.fixed[sl, sl] = np.real(coeff.conj().T @ (diagonal[:, None] * coeff))
        mean = np.r_[mi, -mo]
        self.fixed += 2 * np.pi / (MU0 * log) * np.outer(mean, mean)
        # Remove the mesh zipper elements; all other volume operators are rigid
        # in material coordinates and are retained during physical rotation.
        gap_nu = self.gap.astype(float) / MU0
        self.zip = self.matrix_volume((gap_nu, np.zeros_like(gap_nu), gap_nu))
        self.update(theta)

    def matrix_volume(self, t):
        return tensor_matrix(self.b, t)[self.free][:, self.free]

    def update(self, theta):
        self.rot = np.exp(-1j * self.orders * (theta - self.origin))
        cross = -self.inner.conj().T @ ((self.rot.conj() * self.coupling)[:, None] * self.outer)
        band = self.fixed.copy()
        band[: self.count, self.count :] += cross.real
        band[self.count :, : self.count] += cross.real.T
        n = len(self.free)
        self.extra = (
            csc_matrix(
                (
                    band.ravel(),
                    (np.repeat(self.nodes, len(self.nodes)), np.tile(self.nodes, len(self.nodes))),
                ),
                shape=(n, n),
            )
            - self.zip
        )

    def field(self, a):
        f, t = super().field(a)
        return f + self.extra @ a, t

    def matrix(self, t):
        return self.matrix_volume(t) + self.extra

    def torque(self, a):
        values = a[self.nodes]
        ar = self.rot * (self.inner @ values[: self.count])
        ao = self.outer @ values[self.count :]
        return self.length * np.sum(self.coupling * self.orders * np.real(1j * ar.conj() * ao))


class SkewBand:
    """Three material sections in series, sharing the two stator currents."""

    offsets = np.deg2rad(np.array([-20 / 3, 0.0, 20 / 3]))

    def __init__(self, theta, level):
        self.sections = [BandField(0.0, level) for _ in self.offsets]
        self.n = len(self.sections[0].free)
        self.update(theta)

    def update(self, theta):
        for f, offset in zip(self.sections, self.offsets):
            f.update(theta + offset)

    def equilibrium(self):
        states = [f.equilibrium() for f in self.sections]
        return np.r_[np.concatenate([s[:-2] for s in states]), 0.0, 0.0]

    def torque(self, portion):
        return np.mean(
            [
                f.torque(portion[j * (self.n + 1) : j * (self.n + 1) + self.n])
                for j, f in enumerate(self.sections)
            ]
        )

    def advance(self, state, voltage, h, depth=0):
        try:
            return self._advance(state, voltage, h)
        except RuntimeError:
            if depth >= 5:
                raise
            mid = self.advance(state, voltage, h / 2, depth + 1)
            return self.advance(mid, voltage, h / 2, depth + 1)

    def _advance(self, state, voltage, h):
        old = state[:-2].reshape(3, -1)[:, :-1].copy()

        def residual(v):
            local = []
            tangents = []
            circuit = R * v[-2:] - voltage
            for i, (s, f) in enumerate(zip(v[:-2].reshape(3, -1), self.sections)):
                rate = (s[:-1] - old[i]) / h
                force, tan = f.field(s[:-1])
                local.append(
                    np.r_[
                        force + f.mass @ rate - f.g * s[-1] - f.W @ v[-2:], f.g @ rate - f.total * s[-1]
                    ]
                )
                circuit = circuit + (f.lam / 3) * (f.W.T @ rate)
                tangents.append(tan)
            return np.concatenate([*local, circuit]), tangents

        for _ in range(35):
            res, tan = residual(state)
            if np.max(abs(res)) < 1e-6:
                return state
            matrix = R * np.eye(2)
            right = -res[-2:].copy()
            maps = []
            for f, t, r in zip(self.sections, tan, res[:-2].reshape(3, -1)):
                k = bmat(
                    [
                        [f.matrix(t) + f.mass / h, -f.floating],
                        [f.floating.T / h, csc_matrix([[-f.total]])],
                    ],
                    format="csc",
                )
                coupled = np.zeros((len(r), 3))
                coupled[:, 0] = -r
                coupled[:-1, 1:] = f.W
                solved = splu(k).solve(coupled)
                induced = (f.lam / (3 * h)) * (f.W.T @ solved[:-1])
                right -= induced[:, 0]
                matrix += induced[:, 1:]
                maps.append(solved)
            di = np.linalg.solve(matrix, right)
            step = np.concatenate([*[m[:, 0] + m[:, 1:] @ di for m in maps], di])
            for damp in 2.0 ** -np.arange(14):
                trial = state + damp * step
                if np.linalg.norm(residual(trial)[0]) < np.linalg.norm(res):
                    state = trial
                    break
            else:
                raise RuntimeError("series field line search")
        raise RuntimeError("series field iteration limit")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def calibrate_band(root):
    field = BandField(0.0, 3)
    angles = np.arange(48) * 2 * np.pi / 48
    values = []
    for angle in angles:
        field.update(angle)
        values.append(
            [
                field.torque(field.equilibrium(i)[:-3])
                for i in ([0.0, 0.0], [0.15, 0.0], [-0.15, 0.0], [0.0, 0.15], [0.0, -0.15])
            ]
        )
    np.savez_compressed(root / "band_torque.npz", angles=angles, torque=values)


def band_readout(root, bias_case="zero"):
    field = BandField(0.0, 3)
    bias = np.array(
        {
            "zero": [0.0, 0.0],
            "xp": [0.20, 0.0],
            "xm": [-0.20, 0.0],
            "yp": [0.0, 0.20],
            "ym": [0.0, -0.20],
        }[bias_case]
    )
    angles = np.arange(36) * 2 * np.pi / 36
    records, torques = [], []
    for theta in angles:
        field.update(theta)
        state = field.equilibrium(bias)
        raw = [state[-2:].copy()]
        force = []
        for k in range(24):
            p = k // 2
            axis = p % 3 * np.pi / 3
            v = 82 * ((-1) ** (k % 2 + (p // 3) % 2)) * np.array([np.cos(axis), np.sin(axis)])
            state = field.advance(state, v + R * bias, H)
            raw.append(state[-2:].copy())
            force.append(field.torque(state[:-3]))
        records.append(raw)
        torques.append(force)
    name = "band_readout" if bias_case == "zero" else "band_" + bias_case
    np.savez_compressed(root / (name + ".npz"), angle=angles, current=records, torque=torques, bias=bias)


def dense_torque(root):
    mag = 0.4
    field = BandField(0.0, 3)
    angles = np.deg2rad(np.arange(0, 60, 0.75))
    directions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0], [-1.0, -1.0]]
    )
    currents = mag * directions
    rows = []
    for theta in angles:
        field.update(theta)
        rows.append([field.torque(field.equilibrium(i)[:-3]) for i in currents])
    stator = np.column_stack(
        (np.ones(len(currents)), currents, currents**2, currents[:, 0] * currents[:, 1])
    )
    coeff = np.linalg.lstsq(stator, np.array(rows).T, rcond=None)[0].T
    np.savez_compressed(root / "dense_torque.npz", angle=angles, coeff=coeff)


def torque_current(estimate, demand, last, dense, skew=False):
    # Torque is continuous across the sector endpoint; include its rotated
    # coefficients by evaluating the neighbouring sector rather than wrapping
    # the fixed stator-axis coefficients directly.
    grid = dense["angle"]
    step = grid[1] - grid[0]
    angles = np.arange(18) * 2 * np.pi / 18
    candidates = (
        np.arange(0.0, 0.801, 0.08)[:, None, None] * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    ).reshape(-1, 2)

    def predict(k, base):
        c, s = np.cos(base), np.sin(base)
        i = candidates @ np.array([[c, -s], [s, c]])
        design = np.column_stack((np.ones(len(i)), i, i**2, i[:, 0] * i[:, 1]))
        return design @ dense["coeff"][k]

    def torque_at(theta):
        sector_angle = np.floor(theta / (np.pi / 3)) * np.pi / 3
        pos = (theta - sector_angle) / step
        j = min(int(pos), len(grid) - 1)
        frac = pos - j
        return (1 - frac) * predict(j, sector_angle) + frac * predict(
            (j + 1) % len(grid), sector_angle + (np.pi / 3 if j + 1 == len(grid) else 0.0)
        )

    torques = (
        np.mean([torque_at(estimate + d) for d in SkewBand.offsets], axis=0)
        if skew
        else torque_at(estimate)
    )
    err = torques - demand
    score = (
        err**2
        + (0.0018**2) * (candidates**2).sum(axis=1)
        + (0.002**2) * ((candidates - last) ** 2).sum(axis=1)
    )
    return candidates[score.argmin()]


def run(
    root,
    theta,
    level,
    seconds,
    noise,
    target_deg,
    seed,
    name,
    band=False,
    conditioned=False,
    probes_only=False,
    dense_mode=False,
    skew=False,
):
    rng = np.random.default_rng(seed)
    cal = np.load(root / ("band_readout.npz" if band else "burst_timecal.npz"))
    grid = np.deg2rad(np.arange(0, 360, 0.25))

    # Angle-dependent fingerprint of the most recent 12-pulse sequence.
    def template(nodes, waves):
        offsets = SkewBand.offsets if skew else (0.0,)
        return sum(interpolate(nodes, waves, grid + d) for d in offsets) / len(offsets)

    lib = centered(template(cal["angle"], cal["current"][:, -13:]))
    first_lib = centered(template(cal["angle"], cal["current"][:, :13]))
    energy = np.load(root / "L4energy.npz")
    angles = energy["angles"]
    derivative = (
        np.roll(energy["coenergy_j"], -1, axis=0) - np.roll(energy["coenergy_j"], 1, axis=0)
    ) / (4 * np.pi / len(angles))
    kt = (
        np.stack((derivative[:, 0] - derivative[:, 1], derivative[:, 2] - derivative[:, 3]), axis=-1)
        / 1.6
    )
    if band:
        tab = np.load(root / "band_torque.npz")
        angles = tab["angles"]
        torques = tab["torque"]
        kt = np.stack((torques[:, 1] - torques[:, 2], torques[:, 3] - torques[:, 4]), axis=-1) / 0.30
    if conditioned:
        samples = [np.load(root / f"band_{case}.npz") for case in ("xp", "xm", "yp", "ym")]
        raw = np.stack([template(t["angle"], t["current"][:, -13:]) for t in samples])
        jac = np.stack(((raw[0] - raw[1]) / 0.4, (raw[2] - raw[3]) / 0.4), axis=-1)
        correction = np.linalg.inv(jac.mean(axis=1))
        base_wave = template(cal["angle"], cal["current"][:, -13:])
        base_mean = base_wave.mean(axis=1)
        slope_x = centered(jac[..., 0])
        slope_y = centered(jac[..., 1])
        angles = cal["angle"]
        forces = np.stack([t["torque"][:, -12:].mean(axis=1) for t in samples])
        kt = np.stack(((forces[0] - forces[1]) / 0.4, (forces[2] - forces[3]) / 0.4), axis=-1)
        offset_torque = forces.mean(axis=0)
    field = SkewBand(theta, level) if skew else (BandField(theta, level) if band else Field(theta, level))
    state = field.equilibrium()
    velocity = 0.0
    estimate = None
    read_speed = 0.0
    ref = np.deg2rad(target_deg)
    baseline = np.zeros(2)
    traj = []
    measured = state[-2:] + rng.normal(0, noise, 2)
    integral = 0.0
    if dense_mode:
        with np.load(root / "dense_torque.npz") as d:
            dense = {k: d[k] for k in d.files}
    previous_ref = np.zeros(2)
    for block in range(round(seconds / (12 * H))):
        data = [measured]
        for k in range(12):
            if band:
                field.update(theta)
            elif k % 4 == 0:
                field = Field(theta, level)
            pair = k // 2
            axis = pair % 3 * np.pi / 3
            v = baseline + 82 * ((-1) ** (k % 2 + pair // 3)) * np.array([np.cos(axis), np.sin(axis)])
            state = field.advance(state, v, H)
            tau = field.torque(state[:-3])
            velocity += H * (tau - 0.000179 * velocity) / 0.0001662
            theta += H * velocity
            measured = state[-2:] + rng.normal(0, noise, 2)
            data.append(measured)
        data = np.array(data)
        # All information across this boundary consists of two noisy stator currents.
        templates = first_lib if estimate is None and band else lib
        if conditioned and estimate is not None:
            holding = np.einsum("nij,nj->ni", correction, data.mean(axis=0) - base_mean)
            holding = np.clip(holding, -0.36, 0.36)
            templates = lib + holding[:, 0, None] * slope_x + holding[:, 1, None] * slope_y
        costs = np.sum((templates - centered(data)) ** 2, axis=-1)
        if estimate is None:
            new = grid[costs.argmin()]
            estimate = new
        else:
            candidates = abs(wrap(grid - estimate - read_speed * 12 * H)) < np.deg2rad(10)
            new = grid[np.argmin(np.where(candidates, costs, np.inf))]
            if conditioned:
                predicted = estimate + read_speed * 12 * H
                delta = wrap(new - predicted)
                estimate = predicted + 0.12 * delta
                read_speed += 0.003 * delta / (12 * H)
            else:
                delta = wrap(new - estimate)
                estimate += 0.45 * delta
                read_speed = 0.88 * read_speed + 0.12 * 0.45 * delta / (12 * H)
        err = wrap(ref - estimate)
        # Brief initial acquisition with no commanded torque.
        if block >= 4 and not probes_only:
            integral = np.clip(integral + 12 * H * 0.005 * err, -0.01, 0.01)
            demand = np.clip(0.009 * err - 0.0025 * read_speed + integral, -0.012, 0.012)
            gain = interpolate(angles, kt, np.array([estimate]))[0]
            if band:
                base_force = offset_torque if conditioned else torques[:, 0]
                cog = interpolate(angles, base_force, np.array([estimate]))[0]
                demand = np.clip(0.023 * err - 0.0035 * read_speed + integral - cog, -0.028, 0.028)
            if conditioned:
                demand = np.clip(0.048 * err - 0.0055 * read_speed + integral - cog, -0.024, 0.024)
            iref = demand * gain / max(0.001, float(gain @ gain))
            if conditioned:
                iref *= min(1.0, 0.34 / max(1e-8, np.linalg.norm(iref)))
            if dense_mode:
                integral = np.clip(integral + 12 * H * 0.115 * err, -0.025, 0.025)
                torque = np.clip(0.052 * err - 0.0055 * read_speed + integral, -0.045, 0.045)
                iref = torque_current(estimate, torque, previous_ref, dense, skew)
                previous_ref = iref
            mean_current = np.trapezoid(data, axis=0) / 12
            limit = 11.0 if dense_mode else 5.0
            baseline = np.clip(R * iref + 10 * (iref - mean_current), -limit, limit)
        traj.append(((block + 1) * 12 * H, theta, estimate, velocity, read_speed, ref, tau, *baseline))
    np.savez_compressed(root / f"free_{name}.npz", trace=traj, level=level, noise=noise, seed=seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--theta", type=float, required=True, help="degrees in plant initialization only")
    p.add_argument("--level", type=int, default=4)
    p.add_argument("--seconds", type=float, default=0.36)
    p.add_argument("--noise", type=float, default=0.003)
    p.add_argument("--target", type=float, required=True)
    p.add_argument("--seed", type=int, default=1471)
    p.add_argument("--name", required=True)
    p.add_argument("--band", action="store_true")
    p.add_argument("--conditioned", action="store_true")
    p.add_argument("--probes-only", action="store_true")
    p.add_argument("--dense-servo", action="store_true")
    p.add_argument("--dense-cal", action="store_true")
    p.add_argument("--skew", action="store_true")
    p.add_argument("--calibrate-band", action="store_true")
    p.add_argument("--band-readout", action="store_true")
    p.add_argument("--bias-case", choices=("zero", "xp", "xm", "yp", "ym"), default="zero")
    a = p.parse_args()
    if a.dense_cal:
        dense_torque(a.out)
    elif a.calibrate_band:
        calibrate_band(a.out)
    elif a.band_readout:
        band_readout(a.out, a.bias_case)
    else:
        if a.skew and not a.dense_servo:
            p.error("--skew uses the dense torque controller")
        run(
            a.out,
            np.deg2rad(a.theta),
            a.level,
            a.seconds,
            a.noise,
            a.target,
            a.seed,
            a.name,
            a.band or a.conditioned or a.dense_servo,
            a.conditioned or a.dense_servo,
            a.probes_only,
            a.dense_servo,
            a.skew,
        )

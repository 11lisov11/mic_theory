"""Short current-only relative-position servo, calibrated on an immobilised plain cage.

No rotor angle input. The calibration measures incremental transverse impedance around
a prepared DC field; mechanical state is recorded by the independent moving FEM.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_air56_terminal_servo import Machine, R
from fem.air56.servo_pwm import segments
from tools.score_air56_stock_release import beta_filter

H = 0.001
BIAS = 0.30


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def calibrate():
    m = Machine(4, 0, False, stock=True)
    m.trapezoid = True
    theta = 0.101
    state = m.equilibrium(theta, [BIAS, 0])
    f = m.slices[0]
    initial_link = f.lam * (f.W.T @ m.parts(state)[0, :-1, 0])
    samples = []
    for k in range(140):
        v = 1.0 if 2 <= k < 12 else -0.5 if 26 <= k < 41 else 0.35 if 57 <= k < 109 else 0.0
        for _ in range(4):
            state, _, _ = m.step(state, np.array([R * BIAS, v]), H / 4, load=0.0)
            state[-3:-1] = (theta, 0.0)
        samples.append((v, state[-4]))
    samples = np.array(samples)

    def residual(z):
        leakage, mag, tr = np.exp(z)
        a = np.array(
            [
                [-(R + mag / tr) / leakage, 1 / (tr * leakage), 1 / leakage],
                [mag / tr, -1 / tr, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        step = expm(a * H)
        x = np.zeros(3)
        error = []
        for v, i in samples:
            x[2] = v
            x = step @ x
            error.append(x[0] - i)
        return error

    fit = least_squares(
        residual, np.log([0.06, 0.85, 0.05]), bounds=(np.log([0.001, 0.01, 0.001]), np.log([1, 4, 2]))
    )
    ls, mag, tr = np.exp(fit.x)
    return dict(
        leakage_h=ls,
        mag_h=mag,
        tau_s=tr,
        initial_wb=float(initial_link[0]),
        calibration_rms_a=float(np.sqrt(np.mean(np.square(fit.fun)))),
    )


def advance_checked(machine, state, voltage, dt, load, reference_deg, fixed=False, depth=0):
    try:
        state, _, _ = machine.step(state, voltage, dt, load=load, incremental=True)
    except RuntimeError as exc:
        if str(exc) not in ("field line search", "field iterations"):
            raise
        if depth == 4:
            raise RuntimeError(
                f"{exc} after refinement, dt={dt:.12g}, fixed={fixed}, "
                f"shaft={np.rad2deg(state[-3]):.9g}, v={voltage}"
            ) from exc
        machine.factor_cache.clear()
        mid, p1, n1 = advance_checked(machine, state, voltage, dt / 2, load, reference_deg, fixed, depth + 1)
        end, p2, n2 = advance_checked(machine, mid, voltage, dt / 2, load, reference_deg, fixed, depth + 1)
        return end, max(p1, p2), n1 + n2 + 1
    if fixed:
        state[-3:-1] = np.deg2rad(reference_deg), 0.0
    return state, abs(np.rad2deg(state[-3]) - reference_deg), 0


def run(job):
    path, par, level, slices, angle, load, seed, noise, periods, witness, support_drive = job
    f = path / f"L{level}_s{slices}_a{angle:g}_m{load:g}_rng{seed}_n{periods}.npz"
    if f.exists():
        return
    m = Machine(level, 0, slices > 1, stock=True, axial_slices=slices if slices > 1 else None)
    m.trapezoid = True
    x = m.equilibrium(np.deg2rad(angle), [BIAS, 0.0])
    x[-1] = load
    ls, mag, tr = (par[k] for k in ("leakage_h", "mag_h", "tau_s"))
    psi = np.array([par["initial_wb"], 0.0])
    i = np.array([BIAS, 0.0])
    q_last = psi - ls * i
    rotor = velocity = integral = 0.0
    support = filtered_position = 0.0
    rng = np.random.default_rng(seed)
    if witness:
        locked = Machine(4, 0, slices > 1, stock=True)
        locked.trapezoid = True
        origin = 0.101
        fixed = locked.equilibrium(origin, [BIAS, 0.0])
        fixed_i = i.copy()
        fixed_psi = psi.copy()
        fixed_q = q_last.copy()
    started = not witness
    passive = []
    weight = beta_filter()
    refinements = 0
    trace = []
    for k in range(periods):
        unit = psi / max(np.linalg.norm(psi), 0.02)
        quadr = np.array([-unit[1], unit[0]])
        err = -rotor
        integral = np.clip(integral + 4.0 * H * err, -0.06, 0.06)
        torque = np.clip(3.0 * err - 0.045 * velocity + integral, -0.12, 0.12)
        if support_drive:
            integral = np.clip(integral + H * 16.0 * (-filtered_position), -0.06, 0.06)
            torque = np.clip(support - 3.6 * filtered_position - 0.024 * velocity + integral, -0.12, 0.12)
        current = BIAS * unit + torque / (1.5 * max(np.linalg.norm(psi), 0.05)) * quadr
        u = R * current + 45.0 * (current - i)
        umax = abs(u).max()
        if umax > 55.0:
            u *= 55.0 / umax
        if not started:
            u[:] = 0.0
        excursion = 0.0
        for v, dt in segments(u, H):
            for _ in range(4):
                x, peak, splits = advance_checked(m, x, v, dt / 4, load, angle)
                refinements += splits
                if witness:
                    fixed, _, splits = advance_checked(
                        locked, fixed, v, dt / 4, 0.0, np.rad2deg(origin), True
                    )
                    refinements += splits
                excursion = max(excursion, peak)
        observed = x[-5:-3] + rng.normal(0.0, noise, 2)
        psi += H * (u - R * (i + observed) * 0.5)
        q = psi - ls * observed
        slip = (
            (mag / tr)
            * 0.5
            * (cross(q_last, i) / max(q_last @ q_last, 1e-5) + cross(q, observed) / max(q @ q, 1e-5))
        )
        turn = np.arctan2(cross(q_last, q), q @ q_last) - H * slip
        if witness:
            new_i = fixed[-5:-3].copy()
            fixed_psi += H * (u - R * (fixed_i + new_i) * 0.5)
            new_q = fixed_psi - ls * new_i
            fixed_slip = (
                (mag / tr)
                * 0.5
                * (
                    cross(fixed_q, fixed_i) / max(fixed_q @ fixed_q, 1e-5)
                    + cross(new_q, new_i) / max(new_q @ new_q, 1e-5)
                )
            )
            turn -= np.arctan2(cross(fixed_q, new_q), new_q @ fixed_q) - H * fixed_slip
            fixed_q = new_q
            fixed_i = new_i
        rotor += turn
        velocity += (0.055 if support_drive else 0.18) * (turn / H - velocity)
        filtered_position += 0.25 * (rotor - filtered_position)
        if not started and k < 24:
            passive.append(observed[1])
            w = weight[k, : k + 1]
            estimate = -62.77561184070174 * (w @ passive)
            rotor = np.deg2rad(estimate)
            velocity = 2 * rotor / ((k + 1) * H)
            if abs(estimate) >= max(0.12, 3.5 * 62.77561184070174 * noise * np.linalg.norm(w)):
                started = True
                filtered_position = rotor
                support = float(np.clip(-2 * 0.0001662 * rotor / (((k + 1) * H) ** 2), -0.045, 0.045))
            elif k == 23:
                started = True
        q_last = q
        i = observed
        trace.append(
            [
                (k + 1) * H,
                np.rad2deg(x[-3]) - angle,
                np.rad2deg(rotor),
                x[-2] * 30 / np.pi,
                velocity * 30 / np.pi,
                *u,
                *i,
                torque,
                excursion,
            ]
        )
    np.savez_compressed(
        f,
        trace=trace,
        field_step_splits=refinements,
        incremental_newton=True,
        parameters=json.dumps(par),
        noise_a=noise,
        seed=seed,
        load=load,
        slices=slices,
        start_angle_deg=angle,
        level=level,
    )
    print(f.name, round(max(t[-1] for t in trace), 3), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--angles", type=float, nargs="+", default=[3.7, 9.9])
    p.add_argument("--slices", type=int, nargs="+", default=[1, 3])
    p.add_argument("--periods", type=int, default=120)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--level", type=int, default=5)
    p.add_argument("--witness", action="store_true")
    p.add_argument("--support", action="store_true")
    a = p.parse_args()
    if a.support:
        if not a.witness:
            p.error("--support requires --witness")
        a.path = a.path / "support"
    if a.witness:
        a.path = a.path / "witness"
    a.path.mkdir(parents=True, exist_ok=True)
    par = calibrate()
    (a.path / "calibration.json").write_text(json.dumps(par, indent=2), encoding="utf8")
    jobs = [
        (a.path, par, a.level, s, theta, load, 13079, 0.003, a.periods, a.witness, a.support)
        for s in a.slices
        for theta in a.angles
        for load in (-0.02, 0.02)
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(run, jobs))

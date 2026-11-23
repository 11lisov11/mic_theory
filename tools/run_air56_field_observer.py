"""Full-field current observer and direct torque feedback for the free polar cage."""

import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import bmat, csc_matrix, eye
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_air56_free_field import BandField, H, R, wrap, centered, interpolate

J = 0.0001662
DRAG = 0.000179


def response(model, state, memory):
    # Differentiate the discretized winding/field equations. Columns represent
    # a constant angular offset and two constant stationary-axis voltage offsets.
    a = state[:-3]
    n = model.count
    g = 1j * model.orders * model.rot.conj() * model.coupling
    vals = a[model.nodes]
    dri = -np.real(model.inner.conj().T @ (g * (model.outer @ vals[n:])))
    dro = -np.real(model.outer.conj().T @ (g.conj() * (model.inner @ vals[:n])))
    dk = np.zeros_like(a)
    dk[model.nodes] = np.r_[dri, dro]
    t = model.field(a)[1]
    jac = bmat(
        [
            [model.matrix(t) + model.mass / H, -model.floating, -model.coil],
            [model.floating.T / H, csc_matrix([[-model.total]]), None],
            [model.lam * model.coil.T / H, None, R * eye(2)],
        ],
        format="csc",
    )
    history = memory[:-3]
    field = model.mass @ history / H
    field[:, 0] -= dk
    rhs = np.vstack((field, model.g @ history / H, model.lam * (model.W.T @ history) / H))
    rhs[-2:, 1:] += np.eye(2)
    out = splu(jac).solve(rhs)
    return out, -model.length * (dk @ out[:-3, 1:])


def run(
    root,
    name,
    initial,
    target,
    plant_level,
    observer_level,
    steps,
    seed,
    disturbance,
    memory_ms,
    load_ramp_s=0.0,
):
    rng = np.random.default_rng(seed)
    pose = np.deg2rad(initial)
    goal = np.deg2rad(target)
    plant = BandField(0.0, plant_level)
    plant.update(pose)
    real = plant.equilibrium()
    cal = np.load(root / "band_readout.npz")
    grid = np.arange(720) * np.pi / 360
    initlib = centered(interpolate(cal["angle"], cal["current"][:, :13], grid))
    estimate = 0.0
    omega = 0.0
    est_w = 0.0
    torque_i = 0.0
    baseline = np.zeros(2)
    obs = None
    replica = None
    memory = None
    trace = []
    prev_tau = 0.0
    ref = 0.0
    # The conducting cage carries voltage perturbations across probe bursts.
    retention = np.exp(-12 * H / (0.001 * memory_ms)) if memory_ms > 0 else 0.0
    for b in range(steps):
        raw = [real[-2:] + rng.normal(0, 0.003, 2)]
        gradient = np.zeros(2)
        torque = 0.0
        info = innovation = 0.0
        if obs is not None:
            memory[:, 1:] *= retention
        for k in range(12):
            p = k // 2
            direction = p % 3 * np.pi / 3
            voltage = baseline + 82 * ((-1) ** (k % 2 + p // 3)) * np.array(
                [np.cos(direction), np.sin(direction)]
            )
            plant.update(pose)
            real = plant.advance(real, voltage, H)
            moment = plant.torque(real[:-3])
            elapsed = (b * 12 + k + 1) * H
            load_fraction = np.clip((elapsed - 0.03) / load_ramp_s, 0.0, 1.0) if load_ramp_s else 1.0
            omega += H * (moment - DRAG * omega - disturbance * load_fraction) / J
            pose += H * omega
            meas = real[-2:] + rng.normal(0, 0.003, 2)
            raw.append(meas)
            if obs is not None:
                obs.update(estimate)
                replica = obs.advance(replica, voltage, H)
                memory, dtau = response(obs, replica, memory)
                current_grad = memory[-2:, 0]
                innovation += current_grad @ (meas - replica[-2:])
                info += current_grad @ current_grad
                predicted = obs.torque(replica[:-3])
                est_w += H * (predicted - DRAG * est_w) / J
                estimate += H * est_w
                gradient += dtau / 12
                torque += predicted / 12
        if obs is None:
            estimate = grid[np.argmin(np.sum((initlib - centered(np.array(raw))) ** 2, axis=1))]
            # Replay the first burst from independently initialized field states.
            obs = BandField(0.0, observer_level)
            obs.update(estimate)
            replica = obs.equilibrium()
            memory = np.zeros((len(replica), 3))
            for k in range(12):
                p = k // 2
                direction = p % 3 * np.pi / 3
                u = 82 * ((-1) ** (k % 2 + p // 3)) * np.array([np.cos(direction), np.sin(direction)])
                replica = obs.advance(replica, u, H)
                memory, _ = response(obs, replica, memory)
            ref = estimate
        else:
            correction = np.clip(innovation / (info + 1e-5), -0.045, 0.045)
            estimate += 0.45 * correction
            est_w += 0.013 * correction / (12 * H)
            replica += memory[:, 0] * (0.45 * correction)
            # Bound the commanded trajectory speed by applying a position
            # slew limit and use a damped, local voltage-to-torque update.
            refstep = np.clip(wrap(goal - ref), -0.6 * 12 * H, 0.6 * 12 * H)
            ref += refstep
            err = wrap(ref - estimate)
            torque_i = np.clip(torque_i + 12 * H * 0.05 * err, -0.012, 0.012)
            demand = np.clip(
                0.033 * err + 0.0035 * (refstep / (12 * H) - est_w) + torque_i, -0.028, 0.028
            )
            correction_t = demand - torque - 0.65 * (torque - prev_tau)
            prev_tau = torque
            delta = 0.32 * gradient * correction_t / (gradient @ gradient + 0.00008**2)
            baseline += delta * min(1.0, 0.75 / max(1e-8, np.linalg.norm(delta)))
            baseline *= min(1.0, 10 / max(1e-8, np.linalg.norm(baseline)))
        trace.append(((b + 1) * 12 * H, pose, estimate, omega, est_w, goal, moment, *baseline, info))
    np.savez_compressed(
        root / f"free_{name}.npz",
        trace=trace,
        plant_level=plant_level,
        observer_level=observer_level,
        disturbance_nm=disturbance,
        load_ramp_s=load_ramp_s,
        voltage_memory_ms=memory_ms,
        noise_a=0.003,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--initial", type=float, required=True)
    p.add_argument("--target", type=float, required=True)
    p.add_argument("--plant-level", type=int, default=3)
    p.add_argument("--observer-level", type=int, default=3)
    p.add_argument("--blocks", type=int, default=120)
    p.add_argument("--seed", type=int, default=2881)
    p.add_argument("--disturbance", type=float, default=0.0)
    p.add_argument("--load-ramp-s", type=float, default=0.0)
    p.add_argument("--memory-ms", type=float, default=0.0)
    a = p.parse_args()
    run(
        a.out,
        a.name,
        a.initial,
        a.target,
        a.plant_level,
        a.observer_level,
        a.blocks,
        a.seed,
        a.disturbance,
        a.memory_ms,
        a.load_ramp_s,
    )

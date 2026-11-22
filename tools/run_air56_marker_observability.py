"""Paired phasor observability, similarity maps and design screening for AIR56."""

import argparse
import csv
import hashlib
import math
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config.env_air56_physical_candidate import MOTOR, HF_INDUCTANCE_H
from models.fem_rotor_marker_response import FemRotorMarkerResponse
from tools.run_air56_marker_fem_study import save_json


DIRECTIONS = np.array([(math.cos(a), math.sin(a)) for a in np.arange(6)*np.pi/6])
HOLDOUT = 560027
DEVELOPMENT = 559026


def full_admittance(response, theta, voltage):
    return np.column_stack([response.phasor_current(theta, v) for v in voltage]).reshape(len(theta), len(voltage), 2)


def wrap_error(angle, truth, period):
    return abs((angle-truth+period/2) % period-period/2)


def alias_and_fisher(response, voltage_peak, noise_a):
    theta = np.arange(360)*np.pi/180
    currents = full_admittance(response, theta, voltage_peak*DIRECTIONS)
    contrasts = np.stack([response.phasor_delta_current(theta, v) for v in voltage_peak*DIRECTIONS], axis=1)
    centered = contrasts - contrasts.mean(axis=0)
    variations = np.linalg.norm(centered, axis=-1)
    derivatives = (np.roll(centered, -1, axis=0)-np.roll(centered, 1, axis=0))/(2*np.pi/180)
    information = np.sum(abs(derivatives)**2, axis=-1)/noise_a**2
    antipodes = len(theta)//response.harmonic
    delta = centered[:, 0]-centered[0, 0]
    row = dict(frequency_hz=response.frequency_hz,
               max_peak_current_contrast_a=float(np.max(variations)),
               peak_alias_difference_a=float(np.max(abs(centered-np.roll(centered, antipodes, axis=0)))),
               worst_fixed_crlb_deg_at_one_probe=float(np.rad2deg(1/np.sqrt(max(information[:, 0].min(), 1e-300)))),
               max_noise_separation_sigma=float(np.max(np.linalg.norm(delta, axis=-1))/noise_a),
               max_impedance_mesh_difference_ohm=float(response.mesh_error_ohm.max()))
    return row, theta, currents, information


def sequential_trial(response, seed, motion, steps, noise_a, peak_v, method, control):
    # All arms generate the same phase and standard normals before choosing probes.
    rng = np.random.default_rng(seed)
    dt = 1/response.frequency_hz
    theta = rng.uniform(0, 2*np.pi) + np.arange(steps)*dt*motion
    noise = noise_a*(rng.normal(size=(steps, 2))+1j*rng.normal(size=(steps, 2)))
    # 1 degree and 0.25 rad/s spacing with a finite train-time speed prior.
    grid = np.arange(180)*response.period_rad/180
    speed = np.r_[np.arange(-4., 4.1, 0.25), -10., 10., -20., 20.]
    prediction = full_admittance(response, grid, peak_v*DIRECTIONS)
    weight = np.full((len(grid), len(speed)), 1/(len(grid)*len(speed)))
    shifts = speed*dt/(grid[1]-grid[0])
    bins = np.arange(len(grid))[:, None]-np.floor(shifts).astype(int)
    fractions = shifts-np.floor(shifts)
    columns = np.arange(len(speed))[None, :]
    slopes = np.sum(abs(np.roll(prediction, -1, axis=0)-np.roll(prediction, 1, axis=0))**2, axis=-1)
    observations = []
    start = perf_counter()
    z0 = np.linalg.inv(response.z_stator_ohm)
    sum_energy = 0.0
    for k in range(steps):
        weight = (1-fractions)*weight[bins % len(grid), columns]+fractions*weight[(bins-1) % len(grid), columns]
        direction = 0
        if method == "fixed_alternating":
            direction = 3*(k % 2)
        elif method == "adaptive_fisher":
            direction = int(np.argmax(weight.sum(axis=1) @ slopes))
        voltage = peak_v*DIRECTIONS[direction]
        i = z0 @ voltage if control else response.phasor_current(theta[k], voltage)
        if method != "no_probe":
            pred = (z0 @ voltage)[None, :] if control else prediction[:, direction]
            y = i+noise[k]
            distance = np.sum(abs(pred-y)**2, axis=-1)/noise_a**2
            likelihood = np.exp(-0.5*(distance-distance.min()))
            weight *= likelihood[:, None]
            weight /= weight.sum()
            sum_energy += 0.75*np.vdot(voltage, i).real*dt
        angle_density = weight.sum(axis=1)
        index_order = np.argsort(-angle_density, kind="stable")
        width = 1+np.searchsorted(np.cumsum(angle_density[index_order]), 0.95)
        observations.append((grid[np.argmax(angle_density)], weight.sum(axis=0) @ speed, width/len(grid)))
    estimates = np.array(observations)
    period_error = np.rad2deg(wrap_error(estimates[:, 0], theta, response.period_rad))
    full_error = np.rad2deg(wrap_error(estimates[:, 0], theta, 2*np.pi))
    row = dict(seed=seed, true_speed_rad_s=motion, method=method, control=control, noise_std_a=noise_a,
               period_p95_deg=float(np.quantile(period_error, 0.95)),
               full_p95_deg=float(np.quantile(full_error, 0.95)),
               speed_rmse_rad_s=float(np.sqrt(np.mean((estimates[:, 1]-motion)**2))),
               median_credible_fraction=float(np.median(estimates[:, 2])),
               median_compute_us=(perf_counter()-start)*1e6/steps,
               sign_hit_rate=float(np.mean(np.sign(estimates[:, 1]) == np.sign(motion))) if motion else None,
               delivered_carrier_j=float(sum_energy), probe_count=0 if method == "no_probe" else steps)
    return row, (theta, estimates[:, 0], estimates[:, 1])


def pareto(rows):
    for item in rows:
        objectives = np.array([-item["contrast_a"], item["loss_or_power"], item["contacts"]])
        item["pareto"] = not any(np.all(other <= objectives) and np.any(other < objectives)
                                 for other in (np.array([-x["contrast_a"], x["loss_or_power"], x["contacts"]])
                                               for x in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fem", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seeds", type=int, default=24)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.seeds < 4 or args.steps < 16 or (args.out/"observability.json").exists():
        parser.error("use at least 4 seeds, 16 samples and a new output directory")
    source = args.fem/"channel.npz"
    peak_v = 1.0
    noise_a = 0.003
    all_frequency = []
    with np.load(source) as archive:
        frequencies = archive["frequencies_hz"]
    for hz in frequencies:
        response = FemRotorMarkerResponse.load(source, hz, HF_INDUCTANCE_H,
                                              stator_resistance_ohm=MOTOR.Rs)
        row, _, _, _ = alias_and_fisher(response, peak_v, noise_a)
        all_frequency.append(row)
    chosen = max(all_frequency, key=lambda row: row["max_peak_current_contrast_a"])
    response = FemRotorMarkerResponse.load(source, chosen["frequency_hz"], HF_INDUCTANCE_H,
                                          stator_resistance_ohm=MOTOR.Rs)
    _, theta, currents, _ = alias_and_fisher(response, peak_v, noise_a)
    z = response._at(response.delta_z_ohm, theta)
    maps = np.column_stack((np.rad2deg(theta), z[:, 0, 0].real, z[:, 0, 0].imag, z[:, 0, 1].real,
                            z[:, 0, 1].imag, currents[:, 0, 0].real, currents[:, 0, 0].imag,
                            currents[:, 0, 1].real, currents[:, 0, 1].imag))
    np.savetxt(args.out/"angular_map.csv", maps, delimiter=",", comments="",
               header="theta_deg,delta_z_aa_real,delta_z_aa_imag,delta_z_ab_real,delta_z_ab_imag,i_alpha_real,i_alpha_imag,i_beta_real,i_beta_imag")
    modes = (("no_probe", True), ("fixed", True), ("fixed", False),
             ("fixed_alternating", False), ("adaptive_fisher", False))
    conditions = (("development", DEVELOPMENT, 8, 0.003), ("holdout", HOLDOUT, args.seeds, 0.003),
                  ("holdout_noise_floor", HOLDOUT+1_000_000, args.seeds, 0.0003))
    seeds_used = [seed for _, base, count, _ in conditions for seed in range(base, base+count)]
    if len(seeds_used) != len(set(seeds_used)):
        raise ValueError("development and holdout seeds overlap")
    rows = []
    trace = None
    for split, base, count, sigma in conditions:
        for index in range(count):
            for speed in (0.0, -0.75, 0.75, -3.2, 3.2):
                for mode, control in modes:
                    row, state = sequential_trial(response, base+index, speed, args.steps, sigma, peak_v, mode, control)
                    rows.append(dict(split=split, **row))
                    if split == "holdout" and index == 0 and mode == "fixed" and not control and speed == 3.2:
                        trace = state
    with (args.out/"phasor_trials.csv").open("w", newline="", encoding="utf8") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
    axes[0].plot(maps[:, 0], maps[:, 1]*1e6, label="Re ΔZαα")
    axes[0].plot(maps[:, 0], maps[:, 2]*1e6, label="Im ΔZαα")
    axes[0].set(ylabel="ΔZ, µΩ", title=f"FEM phasor map, {response.frequency_hz:g} Hz")
    axes[0].legend()
    ri = currents[:, 0, :]-currents[:, 0, :].mean(axis=0)
    axes[1].plot(maps[:, 0], ri[:, 0].real*1e9, label="Re ΔIα")
    axes[1].plot(maps[:, 0], ri[:, 1].real*1e9, label="Re ΔIβ")
    axes[1].set(ylabel="Relative current, nA", xlabel="Mechanical angle, deg")
    axes[1].legend()
    axes[2].plot(np.rad2deg(trace[0] % response.period_rad), color="#247ca8", label="Reference for scoring")
    axes[2].plot(np.rad2deg(trace[1]), color="#d26933", label="Posterior MAP")
    axes[2].set(ylabel="Angle modulo 180°, deg", xlabel="Probe index")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.savefig(args.out/"angular_map.png", dpi=180)
    plt.close(fig)
    summaries = []
    rng = np.random.default_rng(49156)
    for split, _, _, _ in conditions:
        fixed = [r for r in rows if r["split"] == split and r["method"] == "fixed" and not r["control"]]
        for method, control in modes:
            subset = [r for r in rows if r["split"] == split and r["method"] == method and r["control"] == control]
            scores = np.array([r["period_p95_deg"] for r in subset])
            paired = scores-np.array([r["period_p95_deg"] for r in fixed])
            groups = paired.reshape(-1, 5).mean(axis=1)
            boot = groups[rng.integers(0, len(groups), (4000, len(groups)))].mean(axis=1)
            summaries.append(dict(split=split, method=method, control=control,
                                  mean_trial_p95_deg=float(scores.mean()),
                                  mean_full_p95_deg=float(np.mean([r["full_p95_deg"] for r in subset])),
                                  avg_speed_rmse_rad_s=float(np.mean([r["speed_rmse_rad_s"] for r in subset])),
                                  mean_credible_fraction=float(np.mean([r["median_credible_fraction"] for r in subset])),
                                  mean_p95_delta_vs_marker_fixed=float(groups.mean()),
                                  bootstrap_95_deg=np.quantile(boot, [0.025, 0.975]).tolist(),
                                  energy_j=float(np.mean([r["delivered_carrier_j"] for r in subset]))))
    np.savez_compressed(args.out/"observer_trace.npz", truth_rad=trace[0], map_rad=trace[1],
                        speed_hat_rad_s=trace[2], frequency_hz=response.frequency_hz)
    from json import loads
    fem = loads((args.fem/"fem_summary.json").read_text(encoding="utf8"))
    design_hz = 2500
    base_z = FemRotorMarkerResponse.load(source, design_hz, HF_INDUCTANCE_H,
                                        stator_resistance_ohm=MOTOR.Rs).z_stator_ohm
    inverse_norm = np.linalg.norm(np.linalg.inv(base_z), ord=2)
    designs = [dict(name=d["design"], frequency_hz=design_hz, contacts=len(d["contacts"]),
                    contrast_a=peak_v*d["delta_z_signal_ohm"]*inverse_norm**2,
                    loss_or_power=max(d["max_delta_total_loss_w_per_t2"], 0))
               for d in fem["designs"]]
    pareto(designs)
    save_json(args.out/"observability.json", dict(
        data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        summary=summaries, pareto_contact_screen=designs, frequencies=all_frequency,
        selected_frequency_hz=response.frequency_hz, peak_voltage_v=peak_v,
        holdout_seed=HOLDOUT, development_seed=DEVELOPMENT,
        note="The contrast ranking uses the FEM map at 1 V. Contact screen loss is normalized per local T² and is not motor efficiency."))
    print(chosen)


if __name__ == "__main__":
    main()

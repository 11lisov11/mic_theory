"""Frozen current-only controller stress test with timestamped ADC samples."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
from random import Random
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control.current_only_induction_ekf import CurrentOnlyInductionEKF
from control.current_only_speed_foc import CurrentOnlySpeedFOC, SpeedFocConfig
from control.innovation_load_gate import InnovationLoadGate
from control.timestamped_current_observer import TimestampedCurrentObserver
from tools.study_current_only_speed_loop import (
    DEFAULT_BUNDLE, DT, calibrated_covariance, load_plant_module, nominal_motor,
    physical_covariance, source_hashes, speed_reference,
)


METHODS = ("naive_fixed", "naive_capped", "aligned_fixed", "aligned_capped",
           "robust_aligned_fixed", "robust_aligned_capped",
           "persistent_aligned_fixed", "persistent_aligned_capped")
CASES = ("nominal", "delay", "impulses", "parameter_drift", "composite", "slow_composite")


def prepare(seed, case, steps, nominal):
    rng = Random(seed)
    scales = [rng.uniform(.95, 1.05) for _ in range(4)]
    actual = replace(nominal, rs=nominal.rs*scales[0], rr=nominal.rr*scales[1],
                     inertia=nominal.inertia*scales[2], friction=nominal.friction*scales[3])
    if case in ("parameter_drift", "composite", "slow_composite"):
        lm = nominal.lm*.85
        actual = replace(actual, lm=lm, ls=lm+1.2*(nominal.ls-nominal.lm),
                         lr=lm+1.2*(nominal.lr-nominal.lm))
    start, load = rng.uniform(.95, 1.25), rng.uniform(.25, .45)
    noise_rng = np.random.default_rng(seed+190000)
    samples = noise_rng.multivariate_normal(np.zeros(2), physical_covariance("anisotropic_noise"), size=steps)
    impulses = np.zeros_like(samples)
    dropouts = np.zeros(steps, dtype=bool)
    if case in ("impulses", "composite", "slow_composite"):
        active = noise_rng.random(steps) < .001
        impulses[active] = noise_rng.normal(0., .5, size=(int(active.sum()), 2))
    if case in ("composite", "slow_composite"):
        dropouts = noise_rng.random(steps) < .005
    delay = 3 if case in ("delay", "composite", "slow_composite") else 0
    return actual, start, load, samples, impulses, dropouts, delay


def run_trial(nominal, seed, case, method, duration=2.5, save_trace=False):
    if method not in METHODS or case not in CASES:
        raise ValueError("unknown arm")
    steps = int(round(duration/DT))
    actual, load_start, load_level, noise, impulses, dropouts, delay = prepare(seed, case, steps, nominal)
    module = load_plant_module()
    params = module.CurrentFluxMotorParams(actual.rs, actual.rr, actual.ls, actual.lr,
        actual.lm, actual.inertia, actual.friction, actual.pole_pairs, current_limit_a=20.)
    plant = module.CurrentFluxInductionMotorRk4(params)
    observer = CurrentOnlyInductionEKF(nominal, calibrated_covariance("anisotropic_noise", "full_covariance_ekf"))
    capped = method.endswith("capped")
    observer.q_spectral[5] = .0001 if capped else .01
    aligned = "aligned" in method
    robust = method.startswith(("robust", "persistent"))
    replay = TimestampedCurrentObserver(observer, history_steps=16, robust_nis_limit=25. if robust else None,
        persistent_samples=4 if method.startswith("persistent") else None)
    controller = CurrentOnlySpeedFOC(nominal, SpeedFocConfig(speed_bandwidth=80., load_feedforward=True))
    gate = InnovationLoadGate(mode="energy", threshold=10., high=.01) if capped else None
    measured_history = []
    errors, estimation, steady_torques, steady_currents = [], [], [], []
    trace, peak, failure, complete, received = [], 0., None, 0, 0
    scenario = "very_low_speed" if case == "slow_composite" else "startup_load"
    for k in range(steps):
        t = k*DT
        state = plant.state
        y = np.array([state.i_s_alpha_a, state.i_s_beta_a]) + noise[k] + impulses[k]
        if case in ("composite", "slow_composite"):
            y = .002*np.round(y/.002)
        measured_history.append(y)
        try:
            sample = k-delay
            corrected = sample >= 0 and not dropouts[sample]
            if corrected:
                if aligned:
                    replay.correct(tuple(measured_history[sample]), sample)
                else:
                    observer.correct(tuple(measured_history[sample]))
                received += 1
            if gate is not None:
                # No new sample means no repeated detector evidence from old NIS.
                if corrected:
                    observer.q_spectral[5] = gate.update(observer.innovation,
                        observer.innovation_covariance, observer.x[2:4], enabled=t >= .8)
                else:
                    observer.q_spectral[5] = .0001
            command = controller.command(observer.x, speed_reference=speed_reference(t, scenario),
                                         vdc=325., time_s=t, dt=DT)
            def drop(current):
                norm = math.hypot(*current)
                return command if norm < 1e-12 else tuple(command[j]-(1.+.05*norm)*current[j]/norm for j in range(2))
            predicted_voltage = drop(observer.x[:2])
            applied_voltage = drop((state.i_s_alpha_a, state.i_s_beta_a))
            if case in ("parameter_drift", "composite", "slow_composite"):
                drift = .3*min(1., max(0., (t-.8)/1.2))
                plant.params = replace(params, rs_ohm=actual.rs*(1.+drift))
            load = load_level if t >= load_start else 0.
            output = plant.step(*applied_voltage, load, DT)
            if aligned:
                replay.predict(predicted_voltage, DT)
            else:
                observer.predict(predicted_voltage, DT)
            complete += 1
            peak = max(peak, output.current_abs_a)
            if t >= .8:
                errors.append(abs(output.state.omega_m_rad_s-speed_reference(t+DT, scenario)))
                estimation.append(abs(observer.x[4]-output.state.omega_m_rad_s))
            if t >= 1.7:
                steady_torques.append(output.torque_nm)
                steady_currents.append(output.current_abs_a)
            if save_trace and k % 20 == 0:
                trace.append(dict(time_s=t+DT, speed_true=output.state.omega_m_rad_s,
                    speed_estimated=float(observer.x[4]), speed_reference=speed_reference(t+DT, scenario),
                    current_a=output.current_abs_a, load_true=load, load_estimated=float(observer.x[5]),
                    q_load=float(observer.q_spectral[5]), torque_nm=output.torque_nm,
                    sample_index=sample if corrected else -1))
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            failure = str(error)
            break
    row = dict(seed=seed, scenario=case, method=method, duration_s=duration,
        completed_steps=complete, failure=failure,
        tracking_mae_rad_s=float(np.mean(errors)) if errors else None,
        estimation_mae_rad_s=float(np.mean(estimation)) if estimation else None,
        peak_current_a=peak, steady_torque_std_nm=float(np.std(steady_torques)) if steady_torques else None,
        steady_current_std_a=float(np.std(steady_currents)) if steady_currents else None,
        downweighted_samples=replay.downweighted, received_samples=received, replay_steps=replay.replay_steps,
        delay_steps=delay, impulse_count=int(np.count_nonzero(np.any(impulses != 0, axis=1))),
        dropped_samples=int(dropouts.sum()), actual_initial_parameters=asdict(actual),
        load_start_s=load_start, load_level_nm=load_level,
        gate_active_fraction=gate.active_steps/steps if gate else 0.)
    return row, trace


def execute(job):
    return run_trial(**job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-offset", type=int, default=77200)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.jobs < 1 or args.duration <= 0:
        parser.error("positive seeds, jobs and duration required")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bundle_hash = hashlib.sha256(DEFAULT_BUNDLE.read_bytes()).hexdigest()
    nominal = nominal_motor(json.loads(DEFAULT_BUNDLE.read_text(encoding="utf-8")))
    hashes = source_hashes()
    for path in (Path(__file__).resolve(), ROOT/"control/timestamped_current_observer.py"):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    jobs = [dict(nominal=nominal, seed=args.seed_offset+i, case=case, method=method,
                 duration=args.duration, save_trace=i == 0)
            for case in args.cases for i in range(args.seeds) for method in args.methods]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for index, (row, trace) in enumerate(pool.map(execute, jobs)):
            rows.append(row)
            if trace:
                with (args.out/f"{row['scenario']}_{row['method']}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
                    writer.writeheader()
                    writer.writerows(trace)
            if (index+1) % 24 == 0:
                print(f"{index+1}/{len(jobs)} completed, failures {sum(r['failure'] is not None for r in rows)}", flush=True)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != value for path, value in hashes.items()):
        raise AssertionError("source changed during study")
    if hashlib.sha256(DEFAULT_BUNDLE.read_bytes()).hexdigest() != bundle_hash:
        raise AssertionError("input changed during study")
    result = dict(hardware_claim=False, motor_parameters_identified=False, rows=rows,
        runtime_s=time.perf_counter()-started, source_sha256=hashes, bundle_sha256=bundle_hash,
        contract="timestamped current and command voltage only; chronological known delay; no plant state passed to policy",
        settings=dict(dt_s=DT, speed_bandwidth=80., robust_nis_limit=25., persistent_samples=4, delay_steps=3,
                      quantum_a=.002, impulse_probability=.001, impulse_std_a=.5,
                      dropout_probability=.005, lm_factor=.85, leakage_factor=1.2, rs_final_factor=1.3))
    (args.out/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(trials=len(rows), failures=sum(r["failure"] is not None for r in rows), runtime_s=result["runtime_s"])))


if __name__ == "__main__":
    main()

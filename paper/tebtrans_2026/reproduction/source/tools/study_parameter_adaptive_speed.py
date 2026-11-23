"""Causal closed-loop exploration: fixed, joint-estimator and controller coupling."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control.current_only_induction_ekf import CurrentOnlyInductionEKF
from control.current_only_speed_foc import CurrentOnlySpeedFOC, SpeedFocConfig
from control.parameter_adaptive_induction_ekf import ParameterAdaptiveInductionEKF
from control.timestamped_current_observer import TimestampedCurrentObserver
from tools.study_current_observer_stress import prepare
from tools.study_current_only_speed_loop import (
    DEFAULT_BUNDLE, DT, calibrated_covariance, load_plant_module, nominal_motor,
    source_hashes, speed_reference,
)

METHODS = ("fixed_q0001", "fixed_q001", "fixed_q01", "fixed_q1", "rs_only",
           "joint_estimator", "joint_coupled", "joint_confidence")
CASES = ("nominal", "parameter_drift", "composite", "slow_composite")
Q = dict(zip(METHODS[:4], (.0001, .001, .01, .1)))


def run_trial(nominal, seed, case, method, duration=2.5, save_trace=False):
    if case not in CASES or method not in METHODS or duration <= 0:
        raise ValueError("invalid experiment")
    started = time.perf_counter()
    steps = int(round(duration/DT))
    actual, load_start, load_level, noise, impulses, dropouts, delay = prepare(seed, case, steps, nominal)
    module = load_plant_module()
    params = module.CurrentFluxMotorParams(actual.rs, actual.rr, actual.ls, actual.lr,
        actual.lm, actual.inertia, actual.friction, actual.pole_pairs, current_limit_a=20.)
    plant = module.CurrentFluxInductionMotorRk4(params)
    r = calibrated_covariance("anisotropic_noise", "full_covariance_ekf")
    if method in Q:
        observer = CurrentOnlyInductionEKF(nominal, r)
        observer.q_spectral[5] = Q[method]
    else:
        observer = ParameterAdaptiveInductionEKF(nominal, r,
            active=(True, False, False) if method == "rs_only" else (True, True, True))
    replay = TimestampedCurrentObserver(observer, history_steps=16, robust_nis_limit=25.)
    controller = CurrentOnlySpeedFOC(nominal, SpeedFocConfig(speed_bandwidth=80., load_feedforward=True))
    scenario = "very_low_speed" if case == "slow_composite" else "startup_load"
    measured_history, errors, estimation, currents, torques, trace = [], [], [], [], [], []
    peak, copper, complete, failure, recovery_s = 0., 0., 0, None, None
    recovery_run = 0
    for k in range(steps):
        t = k*DT
        state = plant.state
        y = np.array([state.i_s_alpha_a, state.i_s_beta_a])+noise[k]+impulses[k]
        if case in ("composite", "slow_composite"):
            y = .002*np.round(y/.002)
        measured_history.append(y)
        try:
            sample = k-delay
            if sample >= 0 and not dropouts[sample]:
                replay.correct(tuple(measured_history[sample]), sample)
            if method in ("joint_coupled", "joint_confidence"):
                controller.motor = observer.controller_motor(confidence=method == "joint_confidence")
            command = controller.command(observer.x[:6], speed_reference=speed_reference(t, scenario),
                vdc=325., time_s=t, dt=DT)
            def drop(current):
                norm = math.hypot(*current)
                return command if norm < 1e-12 else tuple(command[j]-(1.+.05*norm)*current[j]/norm for j in range(2))
            predicted_voltage = drop(observer.x[:2])
            applied_voltage = drop((state.i_s_alpha_a, state.i_s_beta_a))
            drift = .3*min(1., max(0., (t-.8)/1.2)) if case != "nominal" else 0.
            plant.params = replace(params, rs_ohm=actual.rs*(1.+drift))
            load = load_level if t >= load_start else 0.
            output = plant.step(*applied_voltage, load, DT)
            replay.predict(predicted_voltage, DT)
            complete += 1
            peak = max(peak, output.current_abs_a)
            copper += 1.5*plant.params.rs_ohm*output.current_abs_a**2*DT
            error = abs(output.state.omega_m_rad_s-speed_reference(t+DT, scenario))
            if t >= .8:
                errors.append(error)
                estimation.append(abs(observer.x[4]-output.state.omega_m_rad_s))
            if t >= 1.7:
                torques.append(output.torque_nm)
                currents.append(output.current_abs_a)
            if t >= load_start:
                tolerance = .3 if case == "slow_composite" else 3.
                recovery_run = recovery_run+1 if error < tolerance else 0
                if recovery_s is None and recovery_run >= 1000:
                    recovery_s = max(0., t+DT-load_start-.1)
            if save_trace and k % 50 == 0:
                trace.append(dict(time_s=t+DT, speed_true=output.state.omega_m_rad_s,
                    speed_estimated=float(observer.x[4]), speed_reference=speed_reference(t+DT, scenario),
                    current_a=output.current_abs_a, torque_nm=output.torque_nm,
                    parameter_logs=observer.x[6:].tolist(),
                    actual_rs=plant.params.rs_ohm, actual_rr=actual.rr, actual_lm=actual.lm))
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError, ValueError) as error:
            failure = str(error)
            break
    row = dict(seed=seed, scenario=case, method=method, duration_s=duration,
        completed_steps=complete, failure=failure,
        tracking_mae_rad_s=float(np.mean(errors)) if errors else None,
        estimation_mae_rad_s=float(np.mean(estimation)) if estimation else None,
        peak_current_a=peak, current_limit_violated=peak > 2.8,
        copper_energy_j=copper, recovery_s=recovery_s,
        steady_torque_std_nm=float(np.std(torques)) if torques else None,
        steady_current_std_a=float(np.std(currents)) if currents else None,
        parameter_logs=observer.x[6:].tolist(), bound_hits=getattr(observer, "bound_hits", 0),
        actual_initial_parameters=asdict(actual), load_start_s=load_start, load_level_nm=load_level,
        downweighted_samples=replay.downweighted, replay_steps=replay.replay_steps,
        runtime_s=time.perf_counter()-started)
    return row, trace


def execute(job):
    return run_trial(**job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-offset", type=int, default=96200)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.jobs < 1 or args.duration <= 0:
        parser.error("positive counts required")
    if (args.out/"results.json").exists():
        parser.error("do not overwrite previous evidence")
    args.out.mkdir(parents=True, exist_ok=True)
    nominal = nominal_motor(json.loads(DEFAULT_BUNDLE.read_text(encoding="utf-8")))
    hashes = source_hashes()
    for relative in ("control/parameter_adaptive_induction_ekf.py", "tools/study_parameter_adaptive_speed.py",
                     "tools/study_current_observer_stress.py", "control/timestamped_current_observer.py",
                     "docs/parameter_adaptive_speed_protocol.md"):
        path = ROOT/relative
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    bundle_hash = hashlib.sha256(DEFAULT_BUNDLE.read_bytes()).hexdigest()
    jobs = [dict(nominal=nominal, seed=args.seed_offset+i, case=case, method=method,
                 duration=args.duration, save_trace=i == 0)
            for case in args.cases for i in range(args.seeds) for method in args.methods]
    rows, traces = [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for row, trace in pool.map(execute, jobs):
            rows.append(row)
            if trace:
                traces.append(dict(scenario=row["scenario"], method=row["method"], trace=trace))
            print(f"{len(rows)}/{len(jobs)} completed; failures {sum(r['failure'] is not None for r in rows)}", flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    assert hashlib.sha256(DEFAULT_BUNDLE.read_bytes()).hexdigest() == bundle_hash
    result = dict(rows=rows, traces=traces, runtime_s=time.perf_counter()-started,
        source_sha256=hashes, bundle_sha256=bundle_hash, seeds=list(range(args.seed_offset, args.seed_offset+args.seeds)),
        cases=args.cases, methods=args.methods, hardware_claim=False, novelty_claim=False,
        settings=dict(dt_s=DT, parameter_density=1e-5, parameter_prior_sd=.2, log_bound=.7,
            duration_s=args.duration, confidence="posterior variance contraction heuristic"),
        contract="same seeded plant disturbances; chronological current/voltage only; truth used solely in plant/metrics")
    (args.out/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(trials=len(rows), failures=sum(r["failure"] is not None for r in rows), runtime_s=result["runtime_s"])))


if __name__ == "__main__":
    main()

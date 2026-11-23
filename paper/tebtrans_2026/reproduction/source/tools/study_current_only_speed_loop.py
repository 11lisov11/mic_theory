"""AIR56 host speed loop with current-only EKF and an independent RK4 plant."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import importlib.util
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.current_only_induction_ekf import CurrentOnlyInductionEKF, ObserverMotor
from control.current_only_speed_foc import CurrentOnlySpeedFOC, SpeedFocConfig
from control.innovation_load_gate import InnovationLoadGate


PRACTICE = Path("C:/mic_practice")
SOURCE = PRACTICE / "research/mic_ai_theory/snh_pwm/source"
PLANT_SOURCE = SOURCE / "models/induction_motor_current_flux_rk4.py"
DEFAULT_BUNDLE = PRACTICE / "artifacts/air56b2_fidelity_bundle.json"
METHODS = ("isotropic_ekf", "diagonal_ekf", "full_covariance_ekf", "full_covariance_load_ff")
ADAPTIVE_METHODS = ("fixed_load_q_001", "fixed_load_q_01", "coherence_load_gate", "energy_load_gate",
                    "coherence_bounded_gate", "energy_bounded_gate", "fixed_load_q_0001")
METHODS += ADAPTIVE_METHODS
SCENARIOS = ("startup_load", "anisotropic_noise", "resistance_error")
DT = 1.0e-4


def load_plant_module():
    name = "mic_speed_study_independent_plant"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PLANT_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def source_hashes() -> dict[str, str]:
    paths = [ROOT / name for name in (
        "control/current_only_induction_ekf.py", "control/current_only_speed_foc.py",
        "control/innovation_load_gate.py",
        "tools/study_current_only_speed_loop.py", "config/env.py")]
    paths.append(PLANT_SOURCE)
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def nominal_motor(bundle: dict) -> ObserverMotor:
    values = bundle["fidelity"]["f2_samples"][128]["transformed_motor"]
    return ObserverMotor(values["Rs"], values["Rr"], values["Lm"] + values["Ls_sigma"],
                          values["Lm"] + values["Lr_sigma"], values["Lm"],
                          values["J"], values["B"], values["p"])


def physical_covariance(scenario: str) -> np.ndarray:
    if scenario == "anisotropic_noise":
        return np.array([[0.01**2, 0.4 * 0.01 * 0.03],
                         [0.4 * 0.01 * 0.03, 0.03**2]])
    return np.eye(2) * 0.01**2


def calibrated_covariance(scenario: str, method: str) -> np.ndarray:
    # Independent standstill noise calibration; no moving-plant truth is used.
    z = np.random.default_rng(85000).multivariate_normal(
        np.zeros(2), physical_covariance(scenario), size=4000)
    covariance = np.cov(z, rowvar=False)
    if method == "isotropic_ekf":
        return np.eye(2) * np.trace(covariance) / 2
    if method == "diagonal_ekf":
        return np.diag(np.diag(covariance))
    if method in ("full_covariance_ekf", "full_covariance_load_ff", *ADAPTIVE_METHODS):
        return covariance
    raise ValueError(method)


def speed_reference(time_s: float, scenario: str) -> float:
    level = {"low_speed": 30.0, "very_low_speed": 3.0}.get(scenario, 150.0)
    value = level * min(1.0, max(0.0, (time_s - 0.25) / 0.5))
    if scenario == "reversal" and time_s >= 1.5:
        value = 150.0 - 300.0 * min(1.0, (time_s - 1.5) / 0.5)
    return value


def run_trial(*, nominal: ObserverMotor, seed: int, method: str, scenario: str,
              duration_s: float, parameter_span: float = 0.0,
              config: SpeedFocConfig | None = None,
              save_trace: bool = False) -> tuple[dict, list[dict]]:
    if method not in METHODS or scenario not in (*SCENARIOS, "low_speed", "very_low_speed", "reversal"):
        raise ValueError("invalid trial")
    if duration_s <= 0 or not 0 <= parameter_span < 1:
        raise ValueError("invalid interval or parameter span")
    rng = Random(seed)
    def scale():
        return 1.0 + rng.uniform(-parameter_span, parameter_span)
    actual = replace(nominal, rs=nominal.rs * scale(), rr=nominal.rr * scale(),
                     inertia=nominal.inertia * scale(), friction=nominal.friction * scale())
    if scenario == "resistance_error":
        actual = replace(actual, rs=actual.rs * 1.3)
    load_start = rng.uniform(0.95, 1.25)
    load_level = rng.uniform(0.25, 0.45)
    module = load_plant_module()
    plant = module.CurrentFluxInductionMotorRk4(module.CurrentFluxMotorParams(
        rs_ohm=actual.rs, rr_ohm=actual.rr, ls_h=actual.ls,
        lr_h=actual.lr, lm_h=actual.lm, inertia_kgm2=actual.inertia,
        viscous_b_nms=actual.friction, pole_pairs=actual.pole_pairs,
        current_limit_a=20.0))
    observer = CurrentOnlyInductionEKF(nominal, calibrated_covariance(scenario, method))
    selected_config = config or SpeedFocConfig(load_feedforward=method in ("full_covariance_load_ff", *ADAPTIVE_METHODS))
    if method == "fixed_load_q_0001":
        observer.q_spectral[5] = 0.001
    if method == "fixed_load_q_001":
        observer.q_spectral[5] = 0.01
    if method == "fixed_load_q_01":
        observer.q_spectral[5] = 0.1
    gate = None
    if method in ("coherence_load_gate", "energy_load_gate", "coherence_bounded_gate", "energy_bounded_gate"):
        coherent = method.startswith("coherence")
        gate = InnovationLoadGate(mode="coherence" if coherent else "energy",
                                  threshold=20.0 if coherent else 10.0,
                                  high=0.01 if "bounded" in method else 0.1)
    controller = CurrentOnlySpeedFOC(nominal, selected_config)
    steps = int(round(duration_s / DT))
    noise = np.random.default_rng(seed + 100000).multivariate_normal(
        np.zeros(2), physical_covariance(scenario), size=steps)
    errors, estimates, currents, nis, steady_errors, transient_speeds = [], [], [], [], [], []
    trace, steady_torques, steady_currents = [], [], []
    failure = None
    for k in range(steps):
        time_s = k * DT
        state = plant.state
        measured = (state.i_s_alpha_a + float(noise[k, 0]),
                    state.i_s_beta_a + float(noise[k, 1]))
        try:
            observer.correct(measured)
            if gate is not None:
                observer.q_spectral[5] = gate.update(observer.innovation,
                    observer.innovation_covariance, observer.x[2:4], enabled=time_s >= 0.8)
            reference = speed_reference(time_s, scenario)
            command = controller.command(observer.x, speed_reference=reference,
                                         vdc=325.0, time_s=time_s, dt=DT)
            # Nominal voltage reconstruction uses measured current, not plant current.
            def with_drop(current):
                magnitude = math.hypot(*current)
                if magnitude < 1e-12:
                    return command
                drop = 1.0 + 0.05 * magnitude
                return tuple(command[j] - drop * current[j] / magnitude for j in range(2))
            predicted_voltage = with_drop(measured)
            actual_voltage = with_drop((state.i_s_alpha_a, state.i_s_beta_a))
            load = load_level if time_s >= load_start else 0.0
            plant.step(*actual_voltage, load, DT)
            observer.predict(predicted_voltage, DT)
            next_time = time_s + DT
            next_reference = speed_reference(next_time, scenario)
            if time_s >= 0.8:
                errors.append(abs(plant.state.omega_m_rad_s - next_reference))
                estimates.append(abs(observer.x[4] - plant.state.omega_m_rad_s))
                nis.append(observer.normalized_innovation)
            if time_s >= (2.2 if scenario == "reversal" else 1.7):
                steady_errors.append(abs(plant.state.omega_m_rad_s - next_reference))
                steady_torques.append(1.5 * actual.pole_pairs * actual.lm / actual.lr *
                    (plant.state.psi_r_alpha_wb * plant.state.i_s_beta_a -
                     plant.state.psi_r_beta_wb * plant.state.i_s_alpha_a))
                steady_currents.append(math.hypot(plant.state.i_s_alpha_a, plant.state.i_s_beta_a))
            if load_start <= time_s <= load_start + 0.3:
                transient_speeds.append(plant.state.omega_m_rad_s)
            currents.append(math.hypot(plant.state.i_s_alpha_a, plant.state.i_s_beta_a))
            if save_trace and k % 20 == 0:
                trace.append({"time_s": next_time, "speed_reference": next_reference,
                              "speed_true": plant.state.omega_m_rad_s,
                              "speed_estimated": observer.x[4],
                              "current_a": currents[-1],
                              "load_true": load, "load_estimated": observer.x[5],
                              "q_load": float(observer.q_spectral[5]),
                              "gate_statistic": gate.statistic if gate else 0.0})
        except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as error:
            failure = str(error)
            break
    row = {"seed": seed, "method": method, "scenario": scenario,
           "duration_s": duration_s, "parameter_span": parameter_span,
           "completed_steps": len(currents), "failure": failure,
           "tracking_mae_rad_s": float(np.mean(errors)) if errors else None,
           "estimation_mae_rad_s": float(np.mean(estimates)) if estimates else None,
           "peak_current_a": max(currents, default=0.0),
           "final_speed_rad_s": plant.state.omega_m_rad_s,
           "final_estimated_speed_rad_s": float(observer.x[4]),
           "mean_nis": float(np.mean(nis)) if nis else None,
           "steady_tracking_mae_rad_s": float(np.mean(steady_errors)) if steady_errors else None,
           "minimum_speed_after_load_rad_s": min(transient_speeds) if transient_speeds else None,
           "steady_torque_std_nm": float(np.std(steady_torques)) if steady_torques else None,
           "steady_current_std_a": float(np.std(steady_currents)) if steady_currents else None,
           "gate_active_fraction": gate.active_steps / steps if gate else 0.0,
           "load_start_s": load_start, "load_level_nm": load_level,
           "controller": asdict(selected_config),
           "actual_parameters": asdict(actual),
           "measurement_covariance": observer.r.tolist()}
    return row, trace


def execute_job(job):
    return run_trial(**job)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--seed-offset", type=int, default=73000)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--parameter-span", type=float, default=0.0)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/current_only_speed_loop/pilot")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1 or args.seeds < 1:
        parser.error("jobs and seeds must be positive")
    started = time.perf_counter()
    bundle_hash = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    if bundle.get("hardware_claim") is not False:
        raise ValueError("expected a simulation-prior bundle")
    motor = nominal_motor(bundle)
    hashes = source_hashes()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    jobs = [dict(nominal=motor, seed=args.seed_offset+index, method=method,
                 scenario=scenario, duration_s=args.duration,
                 parameter_span=args.parameter_span, save_trace=index == 0)
            for scenario in args.scenarios for index in range(args.seeds) for method in args.methods]
    executor = ProcessPoolExecutor(max_workers=args.jobs) if args.jobs > 1 else None
    try:
        results = executor.map(execute_job, jobs) if executor else map(execute_job, jobs)
        for row, trace in results:
                rows.append(row)
                if trace:
                    with (args.out / f"{row['scenario']}_{row['method']}.csv").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
                        writer.writeheader()
                        writer.writerows(trace)
                print(json.dumps({key: row[key] for key in (
                    "scenario", "method", "seed", "tracking_mae_rad_s",
                    "estimation_mae_rad_s", "peak_current_a", "failure")}), flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True)
    if source_hashes() != hashes or hashlib.sha256(args.bundle.read_bytes()).hexdigest() != bundle_hash:
        raise AssertionError("source or input changed during study")
    result = {"status": "host_current_only_speed_loop", "hardware_claim": False,
              "measurement_contract": "sampled_stator_current_and_reconstructed_command_voltage",
              "plant_state_passed_to_policy": False, "plant": "independent_current_flux_rk4",
              "inverter": "averaged_limited_voltage_with_drop_no_gate_waveforms",
              "motor_parameters_identified": False,
              "bundle": {"path": str(args.bundle), "sha256": bundle_hash},
              "nominal_parameters": asdict(motor), "controller": asdict(SpeedFocConfig()),
              "dt_s": DT, "runtime_s": time.perf_counter()-started,
              "jobs": args.jobs, "rows": rows, "source_sha256": hashes}
    (args.out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

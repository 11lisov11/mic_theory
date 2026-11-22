"""Closed-loop A/B on the catalog-prior AIR56, entirely in software."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from config.env_air56_physical_candidate import ENV, PROFILE, HF_INDUCTANCE_H
from config import env_research_air56_025kw as historical
from control.active_belief_rotor_observer import ActiveBeliefRotorObserver
from control.safe_neural_horizon_pwm import NeuralHorizonConfig, SafeNeuralHorizonPwmController
from control.sensorless_adaptive_foc_baseline import SensorlessAdaptiveFocBaselineController, SensorlessAdaptiveFocBaselineConfig
from control.vector_foc import FocController
from models.fem_rotor_marker_response import FemRotorMarkerResponse
from models.induction_motor_alpha_beta import AlphaBetaInductionMotorModel, AlphaBetaMotorParams, AlphaBetaMotorState
from models.transformations import alpha_beta_to_abc, alpha_beta_to_dq, dq_to_alpha_beta
from models.two_level_inverter import TwoLevelInverterParams, alpha_beta_voltage, estimate_inverter_losses
from safety.ai_pwm_gateway import AIPwmSafetyGateway
from tools.air56_unoq_bridge import Air56BridgeRuntimeState, _build_bundle, _process_telemetry_step
from tools.run_air56_marker_fem_study import save_json
from tools.run_info_optimal_coded_pwm_study import _limits
from tools.uno_q_protocol import Telemetry


RATED = PROFILE["speed_rpm"] * 2*np.pi/60
TORQUE = PROFILE["power_w"]/RATED
CURRENT_STD_A = 0.003
SEEDS = (57201, 57202, 57203, 57204, 57401, 57402, 57403, 57404)
ARMS = {
    "A_ideal_speed_FOC": "vector",
    "B_existing_sensorless_FOC": "adaptive",
    "B_matched_sensorless_FOC": "vector",
    "C_sensorless_FOC_marker": "vector",
    "D_MIC_AI_control": "vector",
    "E_MIC_AI_marker": "vector",
    "G_SNH_control": "neural",
    "G_SNH_marker": "neural",
}


class ElectricalEstimator:
    """Leaky voltage-model flux, with speed from the rotor circuit identity."""

    def __init__(self, p, dt):
        self.p = p
        self.dt = dt
        self.l_r = p.Llr+p.Lm
        self.l_transient = p.Lls+p.Lm-p.Lm**2/self.l_r
        self.stator_flux = 0j
        self.rotor_flux = 0j
        self.speed = 0.0
        self.derivative = 0j
        self.quality = 0.0

    def update(self, voltage, measured_current):
        i = complex(*measured_current)
        u = complex(*voltage)
        current_model = self.rotor_flux*self.p.Lm/self.l_r + self.l_transient*i
        self.stator_flux += self.dt*(u-self.p.Rs*i-5.0*(self.stator_flux-current_model))
        inferred = (self.stator_flux-self.l_transient*i)*self.l_r/self.p.Lm
        new_flux = self.rotor_flux + 0.22*(inferred-self.rotor_flux)
        self.derivative += 0.10*((new_flux-self.rotor_flux)/self.dt-self.derivative)
        self.rotor_flux = new_flux
        flux_sq = abs(new_flux)**2
        if flux_sq > 0.025**2:
            emf = self.derivative-(self.p.Rr/self.l_r)*(self.p.Lm*i-new_flux)
            sample = (new_flux.conjugate()*emf).imag/(self.p.p*flux_sq)
            self.speed += 0.08*(np.clip(sample, -1.5*RATED, 1.5*RATED)-self.speed)
        self.quality = min(1.0, abs(new_flux)/0.15)

    def state(self, estimated_speed, measured_current):
        flux_s = self.p.Lm/self.l_r*self.rotor_flux+self.l_transient*complex(*measured_current)
        return AlphaBetaMotorState(psi_s_alpha=flux_s.real, psi_s_beta=flux_s.imag,
                                   psi_r_alpha=self.rotor_flux.real, psi_r_beta=self.rotor_flux.imag,
                                   omega_m=estimated_speed)

    def torque_hat(self, current):
        return 1.5*self.p.p*self.p.Lm/self.l_r*(self.rotor_flux.conjugate()*complex(*current)).imag


class MarkerEstimator:
    def __init__(self, table):
        self.table = table
        self.belief = ActiveBeliefRotorObserver(table, noise_std_a=CURRENT_STD_A, n_angle=90,
                                               speed_hypotheses=tuple(np.linspace(-RATED, RATED, 41)),
                                               gain_hypotheses=(1.0,))
        self.speed_hat = 0.0
        self.angle = 0.0
        self.relative_width = 1.0
        self.usable = False
        self.samples = 0
        self.energy_j = 0.0
        self.delta_energy_j = 0.0
        self.dwell = 8e-6
        # Full variation bounds the available information before any posterior update.
        prediction = table.response_matrix(self.belief.angles, self.dwell)
        self.max_slope_per_v = float(np.max(np.linalg.norm(prediction-prediction.mean(axis=0), axis=(1, 2))))

    def update(self, measured_increment, voltage, dt):
        self.belief.update(measured_increment, voltage, self.dwell, dt)
        _, width = self.belief.credible_set(0.95)
        self.relative_width = width/self.belief.period_rad
        self.angle = self.belief.map_angle()
        weights = self.belief.weights.sum(axis=(0, 2))
        self.speed_hat = float(weights @ self.belief.speeds)
        spread = np.sqrt(weights @ (self.belief.speeds-self.speed_hat)**2)
        amplitude = np.linalg.norm(voltage)
        snr_bound = self.max_slope_per_v*amplitude*np.sqrt(len(self.belief.history))/CURRENT_STD_A
        self.usable = self.relative_width <= 0.35 and spread < 12 and snr_bound >= 3


def measure_marker(plant, table, voltage, dwell, noise):
    # Hidden mechanical angle is accessible solely through this plant sensor.
    theta = plant.state.theta_m
    current_increment = np.asarray(table.current_increment(theta, *voltage, dwell))+noise
    incremental_power = table.marker_power_delta_w(theta, voltage)
    baseline_power = 0.75*np.vdot(voltage, np.linalg.solve(table.z_stator_ohm, voltage)).real
    return current_increment, baseline_power, incremental_power


def policy_bundles(root):
    paths = (root/historical.ai_eval_checkpoint_path, root/historical.ai_eval_hybrid_secondary_checkpoint_path)
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError(f"frozen AIR56 primary and secondary checkpoints required under {root}")
    # Preserve the frozen actors and supervisory gains. Rebase the bounded
    # ampere command to the separate two-pole catalog-prior FOC.
    from types import SimpleNamespace
    cfg = SimpleNamespace(**{name: getattr(historical, name) for name in dir(historical)
                             if name.startswith("ai_eval_")})
    cfg.foc = ENV.foc
    bundle_params = (("primary", "ai_eval_", "ai_eval_supervisor_enabled", "ai_eval_sup_"),
                     ("secondary", "ai_eval_hybrid_secondary_", "ai_eval_hybrid_secondary_supervisor_enabled",
                      "ai_eval_hybrid_secondary_sup_"))
    bundles = tuple(_build_bundle(name=name, checkpoint=path, env_cfg=cfg, prefix=prefix, enabled_attr=enabled,
                               supervisor_prefix=sup, id_ref_min=0.45, id_ref_max=1.0, omega_nominal=RATED)
                 for path, (name, prefix, enabled, sup) in zip(paths, bundle_params))
    return bundles


def reference(t, mode):
    if t < 0.06:
        return 0.0
    direction = -1 if mode == "reverse" and t > 0.25 else 1
    magnitude = 9.0 if mode == "crawl" else 30.0
    return direction*magnitude*min(1.0, (t-0.06)/0.12)


def trial(arm, seed, mode, table, no_marker_table, policy_root, steps):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, CURRENT_STD_A, (steps, 2))
    hf_noise = rng.normal(0, CURRENT_STD_A, (steps, 2))
    theta0 = rng.uniform(0, 2*np.pi)
    perturbed = seed >= 57400
    p = AlphaBetaMotorParams.from_motor_params(ENV.motor)
    plant_params = replace(p, Rs=p.Rs*(1.05 if perturbed else 1), Rr=p.Rr*(1.10 if perturbed else 1),
                           Lm=p.Lm*(0.9 if seed % 2 and perturbed else 1))
    temp = 75.0 if perturbed else 20.0
    plant = AlphaBetaInductionMotorModel(plant_params, AlphaBetaMotorState(theta_m=theta0, temp_s_c=temp, temp_r_c=temp))
    inverter = TwoLevelInverterParams(Vdc=ENV.inverter.Vdc)
    dt = inverter.t_pwm_s
    gateway = AIPwmSafetyGateway(_limits(p, inverter, physical_candidate=True))
    estimator = ElectricalEstimator(p, dt)
    marker = MarkerEstimator(table if arm.endswith("_marker") else no_marker_table)
    vector_foc = FocController(replace(ENV.foc, kp_speed=0.026, ki_speed=0.50, kp_id=28, kp_iq=28,
                                       ki_id=550, ki_iq=550, iq_limit=1.25, v_limit=105.0), ENV.motor, dt)
    adaptive = SensorlessAdaptiveFocBaselineController(p, inverter, gateway,
        SensorlessAdaptiveFocBaselineConfig(dt_s=dt, torque_limit_nm=TORQUE,
                                            flux_ref_wb=ENV.foc.id_ref*p.Lm))
    neural = SafeNeuralHorizonPwmController(p, inverter, gateway,
        NeuralHorizonConfig(dt_s=dt, torque_limit_nm=TORQUE, flux_ref_wb=ENV.foc.id_ref*p.Lm,
                            horizon=2, max_branching=3))
    policy = policy_bundles(policy_root) if "MIC_AI" in arm else None
    policy_state = Air56BridgeRuntimeState(last_id_ref=ENV.foc.id_ref)
    voltage = (0.0, 0.0)
    previous_power = 0.0
    filtered_power = 0.0
    ai_input_energy = 0.0
    ai_shaft_energy = 0.0
    trace = []
    marker_used = 0
    ai_updates = 0
    ai_enabled = 0
    faults = 0
    transitions = 0
    switch_j = 0.0
    copper_j = 0.0
    input_j = 0.0
    mechanical_j = 0.0
    start = perf_counter()
    for k in range(steps):
        t = k*dt
        ref = reference(t, mode)
        load = TORQUE*(0.13 if t < 0.24 else (0.19 if mode == "crawl" else 0.28))
        # A smoothed opposing load acts only on the simulated plant.
        load = load*math.tanh(plant.state.omega_m/1.5)
        current = plant.currents()
        measured = np.asarray((current.i_s_alpha, current.i_s_beta))+noise[k]
        measured_abs = float(np.linalg.norm(measured))
        estimator.update(voltage, measured)
        if arm.endswith("_marker") or arm in ("B_matched_sensorless_FOC", "D_MIC_AI_control", "G_SNH_control"):
            if k % 4 == 0:
                probe = 18*np.array((1.0, 0.0) if k % 8 == 0 else (0.0, 1.0))
                sample, base_power, delta_power = measure_marker(plant, marker.table, probe, marker.dwell, hf_noise[k])
                marker.update(sample, tuple(probe), 4*dt)
                marker.samples += 1
                marker.energy_j += base_power*marker.dwell
                marker.delta_energy_j += delta_power*marker.dwell
        trust = marker.usable and arm.endswith("_marker")
        marker_used += trust
        speed = float(marker.speed_hat if trust else estimator.speed)
        estimated = estimator.state(speed, measured)
        torque_hat = float(np.clip(estimator.torque_hat(measured), -TORQUE, TORQUE))
        if policy is not None:
            power = max(0.0, previous_power)
            shaft_est = max(0.0, torque_hat*speed)
            filtered_power += -math.expm1(-dt/0.03)*(power-filtered_power)
            ai_input_energy += power*dt
            ai_shaft_energy += shaft_est*dt
        if policy is not None and k % 100 == 0:
            id_meas, iq_meas = alpha_beta_to_dq(*measured, vector_foc.theta_e)
            telem = Telemetry(round(t*1000), speed, ref, id_meas, iq_meas,
                              inverter.Vdc, math.hypot(*measured)/math.sqrt(2), previous_power,
                              int(measured_abs > p.i_limit))
            power_obs = {"p_in_norm": min(10.0, filtered_power/PROFILE["power_w"]),
                         "p_el_filt": filtered_power,
                         "p_shaft_norm": min(10.0, shaft_est/PROFILE["power_w"]),
                         "eta_norm": min(1.0, shaft_est/max(power, 1e-9)),
                         "eta_episode_norm": min(1.0, ai_shaft_energy/max(ai_input_energy, 1e-9))}
            mic = _process_telemetry_step(telem=telem, primary=policy[0], secondary=policy[1], state=policy_state,
                omega_nominal=RATED, i_base=PROFILE["line_current_a"], pole_pairs=p.p, rr=p.Rr,
                lr_total=p.Llr+p.Lm, load_est_gain=TORQUE/1.25, load_base_nm=TORQUE, load_delta_threshold=0.05,
                positive_only=True, latch_steps=1, speed_err_threshold_rel=0.0, speed_err_threshold_abs=0.0,
                fault_mask=0, disable_on_fault=True, disable_on_guard=True, cmd_rate_limit_a_per_s=3.0,
                id_min=0.45, id_max=1.0, energy_observations=power_obs)
            vector_foc.params = replace(vector_foc.params, id_ref=mic.id_ref)
            ai_updates += 1
            ai_enabled += mic.enable_ai
        prev_vector = gateway.current_vector_id
        if ARMS[arm] == "vector":
            control_speed = plant.state.omega_m if arm == "A_ideal_speed_FOC" else speed
            v_d, v_q, theta_e, _, _ = vector_foc.step(t, ref, control_speed, alpha_beta_to_abc(*measured),
                                                       torque_hat, 0.0)
            command = dq_to_alpha_beta(v_d, v_q, theta_e)
            if measured_abs >= p.i_limit:
                command = (0.0, 0.0)
                faults += 1
            voltage = command
        else:
            controlled = adaptive if ARMS[arm] == "adaptive" else neural
            decision = controlled.step(omega_ref=ref, load_torque_nm=0.0, measured_state=estimated,
                                       measured_i_abs=measured_abs)
            if controlled is adaptive:
                speed = controlled.omega_hat
            voltage = (alpha_beta_voltage(decision.vector_id, inverter, i_alpha_beta=tuple(measured))
                       if decision.decision.pwm_enabled else (0.0, 0.0))
            faults += int(not decision.decision.accepted)
            loss = estimate_inverter_losses(prev_vector_id=prev_vector, next_vector_id=decision.vector_id,
                                            params=inverter, i_alpha_beta=tuple(measured))
            transitions += loss.switch_events
            switch_j += loss.switching_w*dt
        # Truth from the plant is used by the integrator and evaluation below.
        previous_power = 1.5*float(np.dot(voltage, measured))
        true_power = 1.5*(voltage[0]*current.i_s_alpha+voltage[1]*current.i_s_beta)
        input_j += max(0.0, true_power)*dt
        copper_j += 1.5*(plant.params.Rs*current.stator_abs**2 +
                         plant.params.Rr*(current.i_r_alpha**2+current.i_r_beta**2))*dt
        result = plant.step(*voltage, load, dt)
        mechanical_j += max(0.0, load*result.state.omega_m)*dt
        flux_truth = complex(result.state.psi_r_alpha, result.state.psi_r_beta)
        trace.append((t, ref, result.state.omega_m, speed, result.torque_nm, result.currents.stator_abs,
                      abs(estimator.rotor_flux-flux_truth), true_power, marker.relative_width))
    a = np.array(trace)
    transient = a[(a[:, 0] >= 0.06) & (a[:, 0] < 0.16)]
    after = a[a[:, 0] >= 0.24]
    steady_torque = after[:, 4]
    total_input = input_j+marker.energy_j+marker.delta_energy_j+switch_j
    row = dict(arm=arm, seed=seed, scenario=mode, temp_c=temp,
               speed_tracking_rmse_rad_s=float(np.sqrt(np.mean((a[:, 2]-a[:, 1])**2))),
               speed_est_rmse_rad_s=0.0 if arm == "A_ideal_speed_FOC" else float(np.sqrt(np.mean((a[:, 2]-a[:, 3])**2))),
               flux_error_rms_wb=float(np.sqrt(np.mean(a[:, 6]**2))),
               starting_torque_peak_nm=float(np.max(transient[:, 4])),
               post_load_speed_error_rmse_rad_s=float(np.sqrt(np.mean((after[:, 2]-after[:, 1])**2))),
               torque_ripple_rms_nm=float(np.std(steady_torque)), peak_current_a=float(np.max(a[:, 5])),
               input_mean_w=total_input/(steps*dt), copper_loss_j=copper_j, load_work_j=mechanical_j,
               energy_ratio=mechanical_j/max(total_input, 1e-12), probe_energy_j=marker.energy_j,
               marker_delta_terminal_energy_j=marker.delta_energy_j, probe_count=marker.samples,
               marker_speed_used_fraction=marker_used/steps, ai_updates=ai_updates, ai_enabled_updates=ai_enabled,
               switch_count=transitions, switch_loss_j=switch_j, faults_or_limit_steps=faults,
               host_loop_us=(perf_counter()-start)*1e6/steps)
    return row, a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--policy-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.steps < 2600 or (args.out/"controller_summary.json").exists():
        parser.error("use at least 2600 steps and a new output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    p = AlphaBetaMotorParams.from_motor_params(ENV.motor)
    checkpoint_paths = (historical.ai_eval_checkpoint_path, historical.ai_eval_hybrid_secondary_checkpoint_path)
    import torch
    bundles = policy_bundles(args.policy_root)
    for bundle, checkpoint in zip(bundles, checkpoint_paths):
        frozen = torch.load(args.policy_root/checkpoint, map_location="cpu", weights_only=True)
        loaded = bundle.agent.net.state_dict()
        if frozen.keys() != loaded.keys() or any(not torch.equal(frozen[key], loaded[key]) for key in frozen):
            raise ValueError(f"frozen actor was changed during loading: {checkpoint}")

    def load(marker):
        return FemRotorMarkerResponse.load(args.table, 2500.0, HF_INDUCTANCE_H,
                                           stator_resistance_ohm=p.Rs, marker_present=marker)
    table, blank = load(True), load(False)
    rows, plots = [], {}
    seeds = SEEDS[:1] if args.smoke else SEEDS
    scenarios = ("crawl",) if args.smoke else ("crawl", "reverse", "load_step")
    for seed in seeds:
        for scenario in scenarios:
            for arm in ARMS:
                row, trace = trial(arm, seed, scenario, table, blank, args.policy_root, args.steps)
                rows.append(row)
                if seed == seeds[0] and scenario == "crawl":
                    plots[arm] = trace
        print(f"completed paired controller seed {seed}", flush=True)
    with (args.out/"controller_trials.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    means = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        means[arm] = {key: float(np.mean([row[key] for row in group])) for key in group[0]
                      if key not in ("seed", "arm", "scenario")}
    rng = np.random.default_rng(5907)
    contrasts = []
    for control, marker in (("B_matched_sensorless_FOC", "C_sensorless_FOC_marker"),
                            ("D_MIC_AI_control", "E_MIC_AI_marker"), ("G_SNH_control", "G_SNH_marker")):
        for metric in ("speed_tracking_rmse_rad_s", "input_mean_w", "energy_ratio"):
            values = np.array([np.mean([next(r[metric] for r in rows if r["arm"] == marker and
                                            r["seed"] == seed and r["scenario"] == s) -
                                        next(r[metric] for r in rows if r["arm"] == control and
                                             r["seed"] == seed and r["scenario"] == s)
                                        for s in scenarios]) for seed in seeds])
            boot = values[rng.integers(0, len(values), (4000, len(values)))].mean(axis=1)
            contrasts.append(dict(control=control, marker=marker, metric=metric, paired_mean=float(values.mean()),
                                  bootstrap_95=np.quantile(boot, (0.025, 0.975)).tolist()))
    np.savez_compressed(args.out/"controller_traces.npz", **plots)
    save_json(args.out/"controller_summary.json", dict(trials=len(rows), means=means, contrasts=contrasts,
        checkpoints={str(path): hashlib.sha256((args.policy_root/path).read_bytes()).hexdigest()
                     for path in checkpoint_paths},
        frozen_features={bundle.name: list(bundle.agent.feature_keys) for bundle in bundles},
        channel_sha256=hashlib.sha256(args.table.read_bytes()).hexdigest(),
        current_noise_std_a=CURRENT_STD_A, p=asdict(p),
        notes=["A uses ideal mechanical speed in the FOC only. Other arms use measured current and applied voltage.",
               "MIC uses frozen four-pole policy weights transferred to the two-pole candidate with rebased id_ref limits.",
               "Vector FOC uses averaged voltages. SNH and existing adaptive FOC use protected switching vectors.",
               "HF probes use the FEM-derived one-frequency correction and a separate energy ledger.",
               "Energy ratio is simulated positive load work / positive input, not a measured nominal efficiency."]))


if __name__ == "__main__":
    main()

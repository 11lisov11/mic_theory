"""Reproducible host-only coded-PWM observability experiment (issue #1)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env_research_air56_025kw import ENV as AIR56_ENV
from config.env_research_al31_4_06kw import ENV as AL31_ENV
from config.env_research_ao2_32_4_3kw import ENV as AO2_ENV
from control.active_belief_rotor_observer import ActiveBeliefRotorObserver
from control.persistent_noise_rotor_observer import PersistentNoiseRotorObserver
from control.info_optimal_coded_pwm import CodedPwmConfig, choose_sequence, legal_sequence, make_sequence
from control.sensorless_adaptive_foc_baseline import SensorlessAdaptiveFocBaselineController
from models.induction_motor_alpha_beta import AlphaBetaInductionMotorModel, AlphaBetaMotorParams, AlphaBetaMotorState
from models.rotor_slot_saliency import SlotSaliency
from models.two_level_inverter import TwoLevelInverterParams, alpha_beta_voltage, switch_events
from safety.ai_pwm_gateway import AIPwmRequest, AIPwmSafetyGateway, GatewayLimits


MOTORS = {"AIR56": AIR56_ENV, "AL31": AL31_ENV, "AO2": AO2_ENV}
METHODS = ("B0_model_only", "B1_belief", "B2_fixed_probe", "B3_info_separation", "B3_info_renyi2", "B3_info_fisher", "B3_info_bayes_risk", "B4_oracle_fisher",
           "B3_lag1", "B3_lag2", "B3_lag4", "B3_lag8", "B3_sparse")
MAIN_METHODS = ("B0_model_only", "B1_belief", "B2_fixed_probe", "B3_info_separation", "B3_info_renyi2", "B3_lag4", "B3_sparse")
MASSES = (0.50, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class Case:
    motor: str = "AIR56"
    scenario: str = "standstill"
    speed_fraction: float = 0.0
    strength: float = 0.15
    nonideal: bool = False
    mismatch: str = "nominal"
    seed: int = 0
    steps: int = 160
    vdc: float = 80.0
    noise_std_a: float = 0.002
    measurement_noise_beta_ratio: float = 1.0


def phase_zero_check(motor: str = "AIR56") -> dict:
    """Identical legal vectors with only mechanical angle changed."""
    params = AlphaBetaMotorParams.from_motor_params(MOTORS[motor].motor)
    inverter = TwoLevelInverterParams(Vdc=80.0)
    gateway = AIPwmSafetyGateway(_limits(params, inverter))
    a = AlphaBetaInductionMotorModel(params, AlphaBetaMotorState(omega_m=0.7, theta_m=0.0))
    b = AlphaBetaInductionMotorModel(params, replace(a.state, theta_m=1.23456789))
    max_delta = 0.0
    for k in range(80):
        vector = (1, 0, 3, 0, 5, 0)[k % 6]
        decision = gateway.evaluate(AIPwmRequest(
            vector_id=vector, dwell_s=10e-6, confidence=0.99,
            predicted_i_abs=a.currents().stator_abs, measured_i_abs=a.currents().stator_abs,
            vdc=inverter.Vdc, tj_c=inverter.ambient_c,
        ))
        if not decision.accepted:
            raise AssertionError(f"phase-zero switching sequence rejected at step {k}: {decision.fault_flags}")
        va, vb = alpha_beta_voltage(vector, inverter)
        ra = a.step(va, vb, 0.0, 10e-6)
        rb = b.step(va, vb, 0.0, 10e-6)
        values_a = (ra.currents.i_s_alpha, ra.currents.i_s_beta, ra.state.psi_s_alpha,
                    ra.state.psi_s_beta, ra.state.psi_r_alpha, ra.state.psi_r_beta, ra.torque_nm)
        values_b = (rb.currents.i_s_alpha, rb.currents.i_s_beta, rb.state.psi_s_alpha,
                    rb.state.psi_s_beta, rb.state.psi_r_alpha, rb.state.psi_r_beta, rb.torque_nm)
        max_delta = max(max_delta, *(abs(x - y) for x, y in zip(values_a, values_b)))
    return {"motor": motor, "tested_steps": 80, "angle_offset_rad": 1.23456789,
            "max_electrical_output_delta": max_delta, "classification": "not represented" if max_delta == 0.0 else "angle dependent"}


def _limits(params: AlphaBetaMotorParams, inverter: TwoLevelInverterParams) -> GatewayLimits:
    return GatewayLimits(
        t_pwm_s=inverter.t_pwm_s, dead_time_s=inverter.dead_time_s, min_pulse_s=inverter.min_pulse_s,
        i_soft_a=max(2.5 * params.i_limit, 1.0), i_trip_a=max(3.2 * params.i_limit, 1.5),
        vdc_min_v=40.0, vdc_max_v=110.0, max_switch_events_per_window=48, switch_window_steps=8,
        confidence_min=0.35,
    )


def _speed(case: Case, k: int, rated_rad_s: float) -> float:
    frac = k / max(case.steps - 1, 1)
    target = case.speed_fraction * rated_rad_s
    if case.scenario == "reversal":
        return target if frac < 0.5 else -target
    if case.scenario == "ramp":
        return target * (2.0 * frac - 1.0)
    if case.scenario == "startup":
        return target * min(frac * 4.0, 1.0)
    if case.scenario == "locked" or case.scenario == "standstill":
        return 0.0
    return target


def _multipliers(case: Case, k: int) -> tuple[float, float, float]:
    table = {
        "nominal": (1.0, 1.0, 1.0), "rs_minus30": (0.7, 1.0, 1.0),
        "rs_plus30": (1.3, 1.0, 1.0), "rs_plus60": (1.6, 1.0, 1.0),
        "rr_minus30": (1.0, 0.7, 1.0), "rr_plus30": (1.0, 1.3, 1.0),
        "rr_plus60": (1.0, 1.6, 1.0), "lm_minus20": (1.0, 1.0, 0.8),
        "lm_plus20": (1.0, 1.0, 1.2), "combined": (1.3, 0.7, 0.8),
        "drift": (1.0 + 0.3 * k / max(case.steps - 1, 1), 1.0 + 0.3 * k / max(case.steps - 1, 1), 1.0 - 0.2 * k / max(case.steps - 1, 1)),
    }
    return table[case.mismatch]


def exogenous(case: Case) -> dict:
    """Single immutable seed stream reused by every method in a case."""
    rng = np.random.default_rng(case.seed)
    return {
        "theta0": float(rng.uniform(0.0, 2.0 * math.pi)),
        "base_noise": rng.normal(0.0, case.noise_std_a, size=(case.steps, 2)),
        "probe_noise": rng.normal(0.0, case.noise_std_a, size=(case.steps, 2)),
        "bus_ripple": rng.uniform(0.0, 2.0 * math.pi),
        "sample_jitter": rng.normal(0.0, 0.02, size=case.steps),
        "speed_bias_fraction": float(rng.uniform(-0.001, 0.001)),
    }


def _carrier(case: Case, params: AlphaBetaMotorParams, inverter: TwoLevelInverterParams, rated_rad_s: float) -> tuple[list[int], list[float]]:
    # The existing MRAS-like controller requires flux state for its usual
    # feedback. Here it runs its no-feedback fallback; no true state enters.
    controller = SensorlessAdaptiveFocBaselineController(params, inverter, AIPwmSafetyGateway(_limits(params, inverter)))
    vectors: list[int] = []
    angle_estimates: list[float] = []
    theta_hat = 0.0
    for k in range(case.steps):
        step = controller.step(omega_ref=_speed(case, k, rated_rad_s), load_torque_nm=0.0, measured_state=None, vdc=inverter.Vdc)
        vectors.append(step.vector_id)
        theta_hat += float(step.metrics["omega_hat"]) * inverter.t_pwm_s
        angle_estimates.append(theta_hat)
    return vectors, angle_estimates


def _error_deg(estimate: float, truth: float, period: float) -> float:
    difference = (estimate - truth + 0.5 * period) % period - 0.5 * period
    return abs(math.degrees(difference))


def _p95(values: list[float]) -> float:
    return float(np.quantile(values, 0.95)) if values else math.nan


def _speed_hypotheses(case: Case, rated_rad_s: float) -> tuple[float, ...]:
    # This grid depends on the commanded speed range, never on the true speed.
    commanded = abs(case.speed_fraction) * rated_rad_s
    if commanded <= 20.0:
        return (-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0)
    return tuple(sorted({-1.5 * commanded, -commanded, -0.5 * commanded,
                         -5.0, -1.0, 0.0, 1.0, 5.0,
                         0.5 * commanded, commanded, 1.5 * commanded}))


def run_trial(case: Case, method: str, *, stimulus: dict | None = None,
              carrier: tuple[list[int], list[float]] | None = None,
              observer_profile: str = "legacy", fixed_probe_vector: int | None = None,
              assumed_noise_beta_ratio: float | None = None,
              noise_beta_ratio_candidates: tuple[float, ...] | None = None,
              likelihood_temperature: float = 1.0,
              persistent_noise_ratios: tuple[float, ...] | None = None) -> dict:
    if method not in METHODS:
        raise ValueError(method)
    if observer_profile not in ("legacy", "dense", "robust"):
        raise ValueError(observer_profile)
    if not math.isfinite(case.noise_std_a) or case.noise_std_a <= 0:
        raise ValueError("measurement noise standard deviation must be finite and positive")
    if not math.isfinite(case.measurement_noise_beta_ratio) or case.measurement_noise_beta_ratio <= 0:
        raise ValueError("measurement noise ratio must be finite and positive")
    if assumed_noise_beta_ratio is not None and (not math.isfinite(assumed_noise_beta_ratio) or assumed_noise_beta_ratio <= 0):
        raise ValueError("assumed noise ratio must be finite and positive")
    if fixed_probe_vector is not None and (method != "B2_fixed_probe" or fixed_probe_vector not in range(1, 7)):
        raise ValueError("fixed probe vector must be 1..6 for B2")
    if persistent_noise_ratios is not None and (method != "B2_fixed_probe" or noise_beta_ratio_candidates is not None):
        raise ValueError("persistent noise mode is exclusive to the B2 fixed-probe experiment")
    env = MOTORS[case.motor]
    params = AlphaBetaMotorParams.from_motor_params(env.motor)
    inverter = TwoLevelInverterParams(Vdc=case.vdc, f_pwm=10_000.0, dead_time_s=1e-6,
                                      min_pulse_s=2e-6, r_on_ohm=0.08, v_drop_v=0.8)
    rated_rad_s = {"AIR56": 1380.0, "AL31": 1390.0, "AO2": 1430.0}[case.motor] * 2.0 * math.pi / 60.0
    stimulus = stimulus if stimulus is not None else exogenous(case)
    carrier = carrier if carrier is not None else _carrier(case, params, inverter, rated_rad_s)
    true_saliency = SlotSaliency(harmonic=28, strength=case.strength, differential_inductance_h=max(params.Lls, 0.045))
    speed_grid = _speed_hypotheses(case, rated_rad_s)
    observer_kwargs = {}
    if observer_profile in ("dense", "robust"):
        if abs(case.speed_fraction) * rated_rad_s <= 20.0:
            speed_grid = tuple(sorted({-20.0, -5.0, *np.arange(-3.0, 3.01, 0.25), 5.0, 20.0}))
    if observer_profile == "robust":
        observer_kwargs = {"angle_diffusion": 0.003, "speed_diffusion": 0.001,
                           "gain_diffusion": 0.001}
    if method == "B0_model_only":
        observer = None
    elif persistent_noise_ratios is not None:
        observer = PersistentNoiseRotorObserver(
            true_saliency, noise_std_a=case.noise_std_a,
            noise_ratios=persistent_noise_ratios, speed_hypotheses=speed_grid,
            likelihood_temperature=likelihood_temperature, **observer_kwargs,
        )
    else:
        observer = ActiveBeliefRotorObserver(
            true_saliency, noise_std_a=case.noise_std_a,
            noise_std_axes_a=(case.noise_std_a, case.noise_std_a * (
                case.measurement_noise_beta_ratio if assumed_noise_beta_ratio is None else assumed_noise_beta_ratio)),
            speed_hypotheses=speed_grid, noise_beta_ratio_candidates=noise_beta_ratio_candidates,
            likelihood_temperature=likelihood_temperature, **observer_kwargs,
        )
    cfg = CodedPwmConfig(information_method={"B3_info_renyi2": "renyi2", "B3_info_fisher": "fisher",
                                               "B3_info_bayes_risk": "bayes_risk"}.get(method, "separation"),
                         reuse_opposite_probe_score=method == "B3_info_bayes_risk")
    noise_scale = np.asarray((1.0, case.measurement_noise_beta_ratio))
    gateway = AIPwmSafetyGateway(_limits(params, inverter))
    speed_bias = 0.0 if case.scenario in ("standstill", "locked") else float(stimulus["speed_bias_fraction"]) * rated_rad_s
    initial_speed = _speed(case, 0, rated_rad_s) + speed_bias
    motor = AlphaBetaInductionMotorModel(params, AlphaBetaMotorState(theta_m=stimulus["theta0"], omega_m=initial_speed))
    errors: list[float] = []
    full_errors: list[float] = []
    torques: list[float] = []
    currents: list[float] = []
    speed_errors: list[float] = []
    compute_us: list[float] = []
    switching = 0
    estimated_switch_loss_j = 0.0
    probe_energy_j = 0.0
    positive_electrical_input_j = 0.0
    flux_magnitudes: list[float] = []
    mean_voltage_deviations: list[float] = []
    probes = 0
    faults = 0
    coverage = {mass: [] for mass in MASSES}
    widths = {mass: [] for mass in MASSES}
    true_theta_trace: list[float] = []
    estimate_trace: list[float] = []
    acquisition_step: int | None = None
    good_run = 0
    for k in range(case.steps):
        tic = perf_counter()
        target_speed = _speed(case, k, rated_rad_s) + speed_bias
        base = carrier[0][k]
        measured_i = motor.currents().stator_abs
        reference = make_sequence(base, None, inverter, cfg)
        if method == "B2_fixed_probe":
            probe = fixed_probe_vector if fixed_probe_vector is not None else (1, 2, 3, 4, 5, 6)[k % 6]
            choice = make_sequence(base, probe, inverter, cfg)
            vprobe = alpha_beta_voltage(probe, inverter)
            predicted = measured_i + cfg.probe_dwell_s * math.hypot(*vprobe) * (1.0 + case.strength) / true_saliency.differential_inductance_h
            if predicted >= gateway.limits.i_soft_a or not legal_sequence(choice, reference=reference, inverter=inverter,
                    gateway=gateway, measured_i_abs=measured_i, predicted_i_abs=predicted):
                choice = reference
        elif method == "B4_oracle_fisher":
            # Evaluation-only ceiling: true angle never enters observer or B3.
            best = -math.inf
            choice = reference
            phase = true_saliency.harmonic * motor.state.theta_m
            factor = cfg.probe_dwell_s * true_saliency.strength * true_saliency.harmonic / true_saliency.differential_inductance_h
            for candidate_vector in range(1, 7):
                candidate = make_sequence(base, candidate_vector, inverter, cfg)
                va, vb = alpha_beta_voltage(candidate_vector, inverter)
                predicted = measured_i + cfg.probe_dwell_s * math.hypot(va, vb) * (1.0 + case.strength) / true_saliency.differential_inductance_h
                if predicted >= gateway.limits.i_soft_a or not legal_sequence(
                    candidate, reference=reference, inverter=inverter, gateway=gateway,
                    measured_i_abs=measured_i, predicted_i_abs=predicted,
                ):
                    continue
                da = factor * (-math.sin(phase) * va + math.cos(phase) * vb)
                db = factor * (math.cos(phase) * va + math.sin(phase) * vb)
                score = (da / case.noise_std_a) ** 2 + (db / (case.noise_std_a * case.measurement_noise_beta_ratio)) ** 2
                if score > best:
                    best = score
                    choice = candidate
        elif method.startswith("B3"):
            assert observer is not None
            choice = choose_sequence(base_vector=base, observer=observer, inverter=inverter,
                                     gateway=gateway, measured_i_abs=measured_i, cfg=cfg,
                                     force_probe=method != "B3_sparse")
        else:
            choice = reference
        compute_us.append((perf_counter() - tic) * 1e6)
        probe = choice.probe_vector
        probes += int(probe is not None)
        rs, rr, lm = _multipliers(case, k)
        motor.params = replace(params, Rs=params.Rs * rs, Rr=params.Rr * rr, Lm=params.Lm * lm)
        gain = 1.0 / (1.0 + 0.2 * (rs - 1.0) + 0.2 * (rr - 1.0) - 0.5 * (lm - 1.0))
        bus = case.vdc * (1.0 + (0.04 * math.sin(2.0 * math.pi * k / 71.0 + stimulus["bus_ripple"]) if case.nonideal else 0.0))
        actual_inverter = replace(inverter, Vdc=bus)
        base_theta = motor.state.theta_m
        base_voltage = alpha_beta_voltage(base, inverter)
        probe_voltage = alpha_beta_voltage(probe, inverter) if probe is not None else None
        current_for_sensing = motor.currents()
        nominal_mean = np.asarray([sum(dwell * alpha_beta_voltage(vector, inverter)[axis]
                                       for vector, dwell in choice.slots) / inverter.t_pwm_s for axis in (0, 1)])
        actual_mean = np.asarray([sum(dwell * alpha_beta_voltage(vector, actual_inverter,
            i_alpha_beta=(current_for_sensing.i_s_alpha, current_for_sensing.i_s_beta) if case.nonideal else None)[axis]
            for vector, dwell in choice.slots) / inverter.t_pwm_s for axis in (0, 1)])
        mean_voltage_deviations.append(float(np.linalg.norm(actual_mean - nominal_mean)))
        actual_base_voltage = alpha_beta_voltage(base, actual_inverter,
            i_alpha_beta=(current_for_sensing.i_s_alpha, current_for_sensing.i_s_beta) if case.nonideal else None)
        aperture_scale = 1.0 + (float(np.clip(stimulus["sample_jitter"][k], -0.05, 0.05)) if case.nonideal else 0.0)
        base_obs = true_saliency.current_increment(base_theta, *actual_base_voltage, cfg.base_dwell_s * aperture_scale, gain=gain) + np.asarray(stimulus["base_noise"][k]) * noise_scale
        probe_obs = None
        if probe is not None:
            actual_probe_voltage = alpha_beta_voltage(probe, actual_inverter,
                i_alpha_beta=(current_for_sensing.i_s_alpha, current_for_sensing.i_s_beta) if case.nonideal else None)
            probe_obs = true_saliency.current_increment(base_theta, *actual_probe_voltage, cfg.probe_dwell_s * aperture_scale, gain=gain) + np.asarray(stimulus["probe_noise"][k]) * noise_scale
        if case.nonideal:
            q = 0.001
            base_obs = np.round(base_obs / q) * q
            if probe_obs is not None:
                probe_obs = np.round(probe_obs / q) * q
        for vector, dwell in choice.slots:
            current = motor.currents()
            previous = gateway.current_vector_id
            decision = gateway.evaluate(AIPwmRequest(
                vector_id=vector, dwell_s=dwell, confidence=0.99,
                predicted_i_abs=current.stator_abs, measured_i_abs=current.stator_abs,
                vdc=bus, tj_c=actual_inverter.ambient_c,
            ))
            if not decision.accepted:
                faults += 1
            applied = decision.vector_id if decision.pwm_enabled else 0
            switching += switch_events(previous, applied)
            estimated_switch_loss_j += switch_events(previous, applied) * actual_inverter.e_sw_j_per_a * max(current.stator_abs, 0.0)
            va, vb = alpha_beta_voltage(applied, actual_inverter,
                i_alpha_beta=(current.i_s_alpha, current.i_s_beta) if case.nonideal else None)
            load = (0.2 * float(env.motor.I_n) * 2.0 if k >= case.steps // 2 else 0.0) if case.scenario == "torque_step" else 0.0
            positive_electrical_input_j += max(0.0, 1.5 * (va * current.i_s_alpha + vb * current.i_s_beta)) * dwell
            step = motor.step(va, vb, load, dwell)
            if case.scenario != "torque_step":
                # Prescribed mechanical trajectory isolates sensing from a speed
                # controller's behavior. The same trajectory is used for all arms.
                motor.state = replace(step.state, omega_m=target_speed, theta_m=motor.state.theta_m + target_speed * dwell)
            torques.append(step.torque_nm)
            currents.append(step.currents.stator_abs)
            flux_magnitudes.append(math.hypot(step.state.psi_s_alpha, step.state.psi_s_beta))
        if probe is not None:
            probe_energy_j += params.Rs * (cfg.probe_dwell_s / inverter.t_pwm_s) * sum(x * x for x in true_saliency.current_increment(base_theta, *probe_voltage, cfg.probe_dwell_s, gain=gain)) * inverter.t_pwm_s
        tic = perf_counter()
        if observer is not None:
            observations = [(tuple(base_obs), base_voltage, cfg.base_dwell_s)]
            if probe is not None:
                observations.append((tuple(probe_obs), probe_voltage, cfg.probe_dwell_s))
            observer.update_observations(observations, inverter.t_pwm_s)
            lag = min(int(method.split("lag")[-1]), len(observer.history) - 1) if method.startswith("B3_lag") else 0
            estimate = observer.map_angle(lag)
            speed_hat = float(np.sum(observer.weights * observer.speeds[None, :, None]))
            target_theta = true_theta_trace[k - lag] if lag > 0 else base_theta
            for mass in MASSES:
                indices, width = observer.credible_set(mass, lag)
                true_bin = int((target_theta % observer.period_rad) / observer.period_rad * len(observer.angles)) % len(observer.angles)
                coverage[mass].append(int(true_bin in indices))
                widths[mass].append(math.degrees(width))
        else:
            estimate = carrier[1][k]
            speed_hat = 0.0
            target_theta = base_theta
        compute_us[-1] += (perf_counter() - tic) * 1e6
        err = _error_deg(estimate, target_theta, true_saliency.period_rad)
        errors.append(err)
        full_errors.append(_error_deg(estimate, target_theta, 2.0 * math.pi))
        speed_errors.append(speed_hat - motor.state.omega_m)
        true_theta_trace.append(base_theta)
        estimate_trace.append(estimate)
        good_run = good_run + 1 if err < math.degrees(true_saliency.period_rad) / 8.0 else 0
        if acquisition_step is None and good_run >= 5:
            acquisition_step = k - 4
    steady = errors[len(errors) // 2:]
    trace = np.asarray(torques[len(torques) // 2:])
    ripple = trace - float(np.mean(trace))
    drift = np.polyfit(np.arange(len(steady)) * inverter.t_pwm_s, steady, 1)[0] if len(steady) > 1 else math.nan
    result = {
        **asdict(case), "method": method, "fixed_probe_vector": fixed_probe_vector,
        "period_deg": math.degrees(true_saliency.period_rad),
        "angle_mae_deg": float(np.mean(errors)), "angle_rmse_deg": float(np.sqrt(np.mean(np.square(errors)))),
        "angle_p95_deg": _p95(errors), "angle_p99_deg": float(np.quantile(errors, 0.99)),
        "angle_max_deg": max(errors), "full_angle_mae_deg": float(np.mean(full_errors)),
        "full_angle_modes": true_saliency.harmonic, "steady_error_slope_deg_s": float(drift),
        "mode_loss_fraction": float(np.mean(np.asarray(errors) > math.degrees(true_saliency.period_rad) / 4.0)),
        "acquisition_time_s": math.nan if acquisition_step is None else acquisition_step * inverter.t_pwm_s,
        "torque_ripple_rms_nm": float(np.sqrt(np.mean(ripple * ripple))),
        "torque_ripple_pp_nm": float(np.ptp(trace)), "torque_rms_nm": float(np.sqrt(np.mean(np.square(torques)))),
        "speed_rmse_rad_s": float(np.sqrt(np.mean(np.square(speed_errors)))),
        "current_rms_a": float(np.sqrt(np.mean(np.square(currents)))),
        "current_peak_a": max(currents),
        "current_ripple_rms_a": float(np.std(currents[len(currents) // 2:])),
        "flux_rms_wb": float(np.sqrt(np.mean(np.square(flux_magnitudes)))),
        "mean_voltage_deviation_v": float(np.mean(mean_voltage_deviations)),
        "current_thd": None,
        "switches_per_s": switching / (case.steps * inverter.t_pwm_s), "switch_loss_proxy_j": estimated_switch_loss_j,
        "probe_copper_energy_j": probe_energy_j, "probe_fraction": probes / case.steps,
        "positive_electrical_input_j": positive_electrical_input_j,
        "probe_loss_fraction_of_input": probe_energy_j / positive_electrical_input_j if positive_electrical_input_j > 0 else math.nan,
        "compute_mean_us": float(np.mean(compute_us)), "compute_max_us": max(compute_us),
        "hypotheses": 0 if observer is None else int(observer.weights.size),
        "candidate_sequences": 7 if method == "B3_sparse" else 6 if method.startswith(("B3", "B4")) else 1,
        "gateway_rejections": faults, "trace_theta_rad": true_theta_trace, "trace_estimate_rad": estimate_trace,
    }
    for mass in MASSES:
        key = f"hpd{int(mass * 100)}"
        result[f"{key}_coverage"] = float(np.mean(coverage[mass])) if observer is not None else math.nan
        result[f"{key}_width_deg"] = float(np.mean(widths[mass])) if observer is not None else math.nan
        result[f"{key}_calibration_error"] = abs(result[f"{key}_coverage"] - mass) if observer is not None else math.nan
    if isinstance(observer, PersistentNoiseRotorObserver):
        result["noise_mode_probabilities"] = dict(zip(observer.noise_ratios,
                                                       map(float, observer.noise_mode_probabilities())))
    return result


def cases_for_suite(suite: str, *, seeds: int | None = None, steps: int | None = None, seed_offset: int = 0) -> list[Case]:
    cases: list[Case] = []
    if suite == "quick":
        points = [("AIR56", "standstill", 0.0, 0.0, False, "nominal"),
                  ("AIR56", "standstill", 0.0, 0.15, False, "nominal"),
                  ("AIR56", "constant", 0.01, 0.15, False, "nominal")]
        n, length = 3, 80
    elif suite == "smoke":
        points = [(motor, scenario, speed, strength, False, "nominal")
                  for motor in MOTORS for scenario, speed in (("standstill", 0.0), ("constant", 0.01), ("constant", -0.01))
                  for strength in (0.0, 0.15)]
        n, length = 20, 160
    elif suite == "stress":
        points = [(motor, scenario, speed, strength, nonideal, mismatch)
                  for motor in MOTORS
                  for scenario, speed in (("standstill", 0.0), ("constant", 0.001), ("constant", 0.005),
                                          ("constant", 0.02), ("constant", 0.05), ("constant", 0.10),
                                          ("constant", 0.50), ("reversal", 0.02), ("ramp", 0.02),
                                          ("startup", 0.01), ("locked", 0.0), ("torque_step", 0.01))
                  for strength, nonideal, mismatch in ((0.0, False, "nominal"), (0.05, False, "nominal"),
                                                       (0.15, False, "nominal"), (0.15, True, "nominal"),
                                                       (0.15, True, "combined"), (0.15, True, "drift"))]
        n, length = 20, 300
    elif suite == "research":
        points = [(motor, scenario, speed, 0.15, nonideal, mismatch)
                  for motor in ("AIR56", "AL31") for scenario, speed in (("standstill", 0.0), ("constant", 0.01))
                  for nonideal, mismatch in ((False, "nominal"), (True, "nominal"), (True, "combined"), (True, "drift"))]
        n, length = 100, 300
    elif suite == "focused":
        points = [(motor, "constant", 0.01, 0.15, nonideal, mismatch)
                  for motor in ("AIR56", "AL31")
                  for nonideal, mismatch in ((False, "nominal"), (True, "combined"))]
        n, length = 100, 160
    elif suite == "lag":
        points = [(motor, scenario, speed, 0.15, False, "nominal")
                  for motor in ("AIR56", "AL31")
                  for scenario, speed in (("standstill", 0.0), ("constant", 0.01))]
        n, length = 20, 160
    elif suite == "long":
        points = [(motor, "constant", 0.01, 0.15, nonideal, mismatch)
                  for motor in ("AIR56", "AL31")
                  for nonideal, mismatch in ((False, "nominal"), (True, "combined"))]
        n, length = 20, 1000
    else:
        raise ValueError(suite)
    for motor, scenario, speed, strength, nonideal, mismatch in points:
        for seed in range(seed_offset, seed_offset + (n if seeds is None else seeds)):
            cases.append(Case(motor=motor, scenario=scenario, speed_fraction=speed,
                              strength=strength, nonideal=nonideal, mismatch=mismatch,
                              seed=seed, steps=length if steps is None else steps))
    return cases


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[x] for x in ("motor", "scenario", "speed_fraction", "strength", "nonideal", "mismatch", "method"))
        groups.setdefault(key, []).append(row)
    out = []
    fields = ("angle_mae_deg", "angle_p95_deg", "angle_p99_deg", "full_angle_mae_deg",
              "torque_ripple_rms_nm", "current_rms_a", "switches_per_s", "switch_loss_proxy_j",
              "probe_copper_energy_j", "probe_loss_fraction_of_input", "current_peak_a",
              "mean_voltage_deviation_v", "current_ripple_rms_a", "flux_rms_wb",
              "compute_mean_us", "compute_max_us", "gateway_rejections",
              "hpd50_coverage", "hpd90_coverage", "hpd95_coverage", "hpd99_coverage",
              "hpd95_width_deg")
    for key, items in groups.items():
        entry = dict(zip(("motor", "scenario", "speed_fraction", "strength", "nonideal", "mismatch", "method"), key))
        entry["n"] = len(items)
        for field in fields:
            values = np.asarray([r[field] for r in items], dtype=float)
            entry[field + "_mean"] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else None
            entry[field + "_p95"] = float(np.nanquantile(values, 0.95)) if np.any(np.isfinite(values)) else None
            finite = values[np.isfinite(values)]
            entry[field + "_ci95_half"] = float(1.96 * np.std(finite, ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else None
        out.append(entry)
    return out


def _paired(rows: list[dict]) -> list[dict]:
    lookup = {(r["motor"], r["scenario"], r["speed_fraction"], r["strength"], r["nonideal"],
               r["mismatch"], r["seed"], r["method"]): r for r in rows}
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        if row["method"] not in ("B3_info_separation", "B3_info_renyi2", "B3_info_fisher", "B3_info_bayes_risk", "B3_sparse", "B3_lag1", "B3_lag2", "B3_lag4", "B3_lag8"):
            continue
        key = (row["motor"], row["scenario"], row["speed_fraction"], row["strength"], row["nonideal"], row["mismatch"], row["seed"])
        fixed = lookup.get(key + ("B2_fixed_probe",))
        if fixed is None:
            continue
        effect = (fixed["angle_p95_deg"] - row["angle_p95_deg"]) / max(fixed["angle_p95_deg"], 1e-12)
        grouped.setdefault(key[:-1] + (row["method"],), []).append(effect)
    output = []
    for key, effects in grouped.items():
        values = np.asarray(effects)
        rng = np.random.default_rng(20261003)
        boot = np.mean(rng.choice(values, size=(1000, len(values)), replace=True), axis=1)
        output.append({**dict(zip(("motor", "scenario", "speed_fraction", "strength", "nonideal", "mismatch", "method"), key)),
                       "n": len(effects), "paired_p95_reduction_mean": float(np.mean(values)),
                       "bootstrap95_low": float(np.quantile(boot, 0.025)),
                       "bootstrap95_high": float(np.quantile(boot, 0.975))})
    return output


def _plots(summary: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 220})
    scenario = "standstill" if any(r["motor"] == "AIR56" and r["scenario"] == "standstill" for r in summary) else "constant"
    chosen = [r for r in summary if r["motor"] == "AIR56" and r["scenario"] == scenario
              and r["speed_fraction"] == (0.0 if scenario == "standstill" else 0.01)
              and not r["nonideal"] and r["mismatch"] == "nominal"]
    if not chosen:
        return
    methods = tuple(method for method in METHODS if any(r["method"] == method for r in chosen))
    symbols = ("o", "s", "^", "D", "x", "P", "v", "<", ">", "h")[:len(methods)]
    styles = ("-", "--", ":", "-.", (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (1, 1)), "--", ":", "-.")[:len(methods)]
    strength = max(r["strength"] for r in chosen)

    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    for index, (method, marker) in enumerate(zip(methods, symbols)):
        item = next((r for r in chosen if r["method"] == method and r["strength"] == strength), None)
        if item:
            ax.errorbar(index, item["angle_p95_deg_mean"], yerr=item["angle_p95_deg_ci95_half"] or 0.0,
                        marker=marker, markersize=6, color="black", linewidth=1.0, capsize=3)
    ax.set_xticks(range(len(methods)), [m.replace("_", " ") for m in methods], rotation=25, ha="right")
    ax.set(ylabel="Mean trial P95 modulo-slot error (mechanical deg)",
           title=f"AIR56; {scenario}; assumed saliency={strength:g}; 95% normal CI")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "angle_p95_ablation_ci.png")
    fig.savefig(out / "angle_p95_ablation_ci.svg")
    plt.close(fig)

    sensitivity_methods = tuple(m for m in ("B0_model_only", "B2_fixed_probe", "B3_info_separation") if m in methods)
    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    for method, marker, style in zip(sensitivity_methods, ("o", "s", "^"), ("-", "--", ":")):
        points = sorted((r for r in chosen if r["method"] == method), key=lambda r: r["strength"])
        if points:
            ax.errorbar([p["strength"] for p in points], [p["angle_p95_deg_mean"] for p in points],
                        yerr=[p["angle_p95_deg_ci95_half"] or 0.0 for p in points],
                        marker=marker, linestyle=style, color="black", linewidth=1.1,
                        capsize=2.0, markersize=5, label=method.replace("_", " "))
    ax.set(xlabel="Assumed slot-saliency strength", ylabel="Mean trial P95 modulo-slot error (mechanical deg)",
           title=f"AIR56, {scenario}; paired seeds")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=1, frameon=False, loc="best")
    fig.savefig(out / "angle_p95_vs_saliency.png")
    fig.savefig(out / "angle_p95_vs_saliency.svg")
    plt.close(fig)

    bars = [next((r for r in chosen if r["method"] == method and r["strength"] == strength), None) for method in methods]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    labels = [m.replace("B3_info_", "B3 ").replace("_", " ") for m in methods]
    ax.bar(np.arange(len(methods)), [b["angle_p95_deg_mean"] if b else math.nan for b in bars],
           color="white", edgecolor="black", hatch=("", "//", "..", "xx", "\\\\", "--", "++", "oo", "||", "**")[:len(methods)])
    ax.set_xticks(np.arange(len(methods)), labels, rotation=25, ha="right")
    ax.set(ylabel="Mean trial P95 error (mechanical deg)", title=f"AIR56; {scenario}; saliency={strength:g}; assumed period=12.86 deg")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "ablation_angle_p95.png")
    fig.savefig(out / "ablation_angle_p95.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), layout="constrained")
    labels = {"B0_model_only": "B0", "B1_belief": "B1", "B2_fixed_probe": "B2",
              "B3_info_separation": "B3 sep", "B3_info_renyi2": "B3 R2", "B3_lag4": "B3 lag4", "B3_sparse": "B3 sparse"}
    offsets = {"B0_model_only": (6, 2), "B1_belief": (6, 2), "B2_fixed_probe": (6, 3),
               "B3_info_separation": (-48, 10), "B3_info_renyi2": (10, 15),
               "B3_lag4": (10, -18), "B3_sparse": (6, 2)}
    pareto_methods = {"B0_model_only", "B1_belief", "B2_fixed_probe", "B3_info_separation",
                      "B3_info_renyi2", "B3_lag4", "B3_sparse"}
    plotted_y = []
    for method, marker in zip(methods, symbols):
        if method not in pareto_methods:
            continue
        item = next((r for r in chosen if r["method"] == method and r["strength"] == strength), None)
        if item:
            plotted_y.append(item["angle_p95_deg_mean"])
            ax.scatter(item["switches_per_s_mean"], item["angle_p95_deg_mean"],
                       marker=marker, s=55, color="black")
            ax.annotate(labels.get(method, method.replace("_", " ")),
                        (item["switches_per_s_mean"], item["angle_p95_deg_mean"]), xytext=offsets.get(method, (6, 2)),
                        textcoords="offset points", fontsize=8)
    ax.set(xlabel="Leg transitions per second", ylabel="Mean trial P95 modulo-slot error (mechanical deg)",
           title=f"Sensing accuracy versus switching; AIR56 {scenario}")
    if plotted_y:
        ax.set_ylim(bottom=max(0.0, min(plotted_y) - 0.35))
    ax.grid(alpha=0.2)
    fig.savefig(out / "pareto_switching.png")
    fig.savefig(out / "pareto_switching.svg")
    plt.close(fig)

    lag_items = [r for r in chosen if r["strength"] == strength and
                 r["method"] in ("B3_info_separation", "B3_lag1", "B3_lag2", "B3_lag4", "B3_lag8")]
    if len(lag_items) >= 3:
        lag_items.sort(key=lambda r: 0 if r["method"] == "B3_info_separation" else int(r["method"].split("lag")[-1]))
        fig, ax = plt.subplots(figsize=(6.6, 4.0), layout="constrained")
        lag = [0 if r["method"] == "B3_info_separation" else int(r["method"].split("lag")[-1]) for r in lag_items]
        ax.errorbar(lag, [r["angle_p95_deg_mean"] for r in lag_items],
                    yerr=[r["angle_p95_deg_ci95_half"] or 0 for r in lag_items],
                    color="black", marker="o", capsize=3)
        ax.set(xlabel="Fixed-lag delay (PWM periods)", ylabel="Mean trial P95 modulo error (deg)",
               title="Delayed smoothing versus forward filter")
        ax.set_xticks(lag)
        ax.grid(alpha=0.2)
        fig.savefig(out / "lag_ablation.png")
        fig.savefig(out / "lag_ablation.svg")
        plt.close(fig)

    for svg in out.glob("*.svg"):
        data = svg.read_text(encoding="utf-8")
        svg.write_text("\n".join(line.rstrip() for line in data.splitlines()) + "\n", encoding="utf-8")


def run_suite(suite: str, out: Path, *, seeds: int | None = None, steps: int | None = None,
              seed_offset: int = 0, observer_profile: str = "legacy") -> dict:
    if observer_profile not in ("legacy", "dense", "robust"):
        raise ValueError(observer_profile)
    out.mkdir(parents=True, exist_ok=True)
    cases = cases_for_suite(suite, seeds=seeds, steps=steps, seed_offset=seed_offset)
    methods = (("B2_fixed_probe", "B3_info_separation", "B3_lag1", "B3_lag2", "B3_lag4", "B3_lag8") if suite == "lag"
               else ("B2_fixed_probe", "B3_info_separation", "B3_info_renyi2") if suite == "long" else MAIN_METHODS)
    rows: list[dict] = []
    representative = None
    start = perf_counter()
    for index, case in enumerate(cases):
        stimulus = exogenous(case)
        params = AlphaBetaMotorParams.from_motor_params(MOTORS[case.motor].motor)
        inverter = TwoLevelInverterParams(Vdc=case.vdc)
        rated = {"AIR56": 1380.0, "AL31": 1390.0, "AO2": 1430.0}[case.motor] * 2.0 * math.pi / 60.0
        carrier = _carrier(case, params, inverter, rated)
        digest = hashlib.sha256(
            np.asarray([stimulus["theta0"], stimulus["bus_ripple"], stimulus["speed_bias_fraction"]]).tobytes()
            + np.asarray(stimulus["base_noise"]).tobytes()
            + np.asarray(stimulus["probe_noise"]).tobytes()
            + np.asarray(stimulus["sample_jitter"]).tobytes()
        ).hexdigest()[:16]
        for method in methods:
            result = run_trial(case, method, stimulus=stimulus, carrier=carrier,
                               observer_profile=observer_profile)
            if representative is None and method == "B3_info_renyi2":
                representative = {"case": asdict(case), "true_theta_rad": result["trace_theta_rad"],
                                  "estimate_modulo_rad": result["trace_estimate_rad"]}
            result.pop("trace_theta_rad")
            result.pop("trace_estimate_rad")
            result["stimulus_digest"] = digest
            rows.append(result)
        if (index + 1) % 20 == 0:
            print(f"{index + 1}/{len(cases)} cases", flush=True)
    summary = _aggregate(rows)
    paired = _paired(rows)
    phase_zero = [phase_zero_check(motor) for motor in MOTORS]
    verdict = "MODEL DOES NOT SUPPORT THE CLAIM" if all(x["max_electrical_output_delta"] == 0.0 for x in phase_zero) else "UNRESOLVED"
    payload = {
        "study": "Information-Optimal Coded PWM, host simulation only",
        "issue": "https://github.com/11lisov11/mic_theory/issues/1",
        "suite": suite, "seed_offset": seed_offset, "observer_profile": observer_profile,
        "case_count": len(cases), "trial_count": len(rows),
        "elapsed_s": perf_counter() - start, "phase_zero": phase_zero,
        "plant_extension": {"type": "unvalidated rotor-slot differential inductance sensor channel",
                            "harmonic": 28, "period_mechanical_deg": 360.0 / 28.0,
                            "absolute_angle_observable": False},
        "verdict_on_absolute_angle": verdict,
        "summary": summary, "paired_against_fixed_probe": paired,
        "notes": ["Mechanical speed is externally prescribed; this is a sensing experiment, not closed-loop FOC validation.",
                  "Existing adaptive FOC supplies the common no-feedback carrier; its full-state feedback API is not used.",
                  "B0 has no mechanical-angle observer and is a model-only control reference.",
                  "HF saliency measurement is synthetic and not hardware-validated.",
                  "Credible sets are modulo slot pitch, not absolute-angle bounds."],
    }
    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out / "representative_trace.json").write_text(json.dumps(representative, indent=2), encoding="utf-8")
    _plots(summary, out)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quick", "smoke", "stress", "research", "focused", "lag", "long"), default="quick")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--observer-profile", choices=("legacy", "dense", "robust"), default="legacy")
    parser.add_argument("--plot-only", action="store_true", help="regenerate figures from an existing summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "info_optimal_coded_pwm" / "quick")
    args = parser.parse_args()
    if args.seeds is not None and args.seeds < 1 or args.steps is not None and args.steps < 8 or args.seed_offset < 0:
        parser.error("seeds must be >=1, steps >=8 and seed offset >=0")
    if args.plot_only:
        payload = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
        _plots(payload["summary"], args.out)
        print(f"plots updated: {args.out}")
        return
    payload = run_suite(args.suite, args.out, seeds=args.seeds, steps=args.steps,
                        seed_offset=args.seed_offset, observer_profile=args.observer_profile)
    print(json.dumps({"suite": payload["suite"], "trial_count": payload["trial_count"],
                      "elapsed_s": payload["elapsed_s"], "verdict": payload["verdict_on_absolute_angle"]}, indent=2))


if __name__ == "__main__":
    main()

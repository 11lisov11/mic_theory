"""Torque-equivalent finite coded-PWM search through the existing gatekeeper."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import numpy as np

from control.active_belief_rotor_observer import ActiveBeliefRotorObserver
from models.two_level_inverter import TwoLevelInverterParams, alpha_beta_voltage, switch_events
from safety.ai_pwm_gateway import (
    AIPwmRequest,
    AIPwmSafetyGateway,
    has_direct_leg_transition,
    has_shoot_through,
    transition_waveform,
)


@dataclass(frozen=True)
class CodedPwmConfig:
    base_dwell_s: float = 10e-6
    probe_dwell_s: float = 5e-6
    information_method: str = "separation"
    lambda_switch: float = 0.01
    lambda_current: float = 0.01
    lambda_probe_energy: float = 0.01
    lambda_torque: float = 0.01
    mean_tolerance_v: float = 1e-10


@dataclass(frozen=True)
class CodedSequence:
    slots: tuple[tuple[int, float], ...]
    probe_vector: int | None
    information_score: float = 0.0


def average_voltage(sequence: CodedSequence, inverter: TwoLevelInverterParams) -> tuple[float, float]:
    duration = sum(dwell for _, dwell in sequence.slots)
    if duration <= 0:
        raise ValueError("empty PWM interval")
    va = vb = 0.0
    for vector, dwell in sequence.slots:
        a, b = alpha_beta_voltage(vector, inverter)
        va += dwell * a
        vb += dwell * b
    return va / duration, vb / duration


def make_sequence(base_vector: int, probe_vector: int | None, inverter: TwoLevelInverterParams, cfg: CodedPwmConfig) -> CodedSequence:
    period = inverter.t_pwm_s
    remaining = period - cfg.base_dwell_s - 2.0 * cfg.probe_dwell_s
    if remaining < inverter.min_pulse_s or cfg.probe_dwell_s < inverter.min_pulse_s or cfg.base_dwell_s < inverter.min_pulse_s:
        raise ValueError("requested slots violate minimum pulse")
    if probe_vector is None:
        return CodedSequence(((base_vector, cfg.base_dwell_s), (0, remaining + 2.0 * cfg.probe_dwell_s)), None)
    if probe_vector not in (1, 2, 3, 4, 5, 6):
        raise ValueError("probe must be an active vector")
    return CodedSequence(
        ((base_vector, cfg.base_dwell_s), (0, remaining), (probe_vector, cfg.probe_dwell_s), (7 - probe_vector, cfg.probe_dwell_s)),
        probe_vector,
    )


def legal_sequence(
    sequence: CodedSequence,
    *,
    reference: CodedSequence,
    inverter: TwoLevelInverterParams,
    gateway: AIPwmSafetyGateway,
    measured_i_abs: float,
    predicted_i_abs: float,
    tolerance_v: float = 1e-10,
) -> bool:
    if not math.isclose(sum(t for _, t in sequence.slots), inverter.t_pwm_s, abs_tol=1e-12):
        return False
    a, b = average_voltage(sequence, inverter)
    ra, rb = average_voltage(reference, inverter)
    if math.hypot(a - ra, b - rb) > tolerance_v:
        return False
    trial = copy.deepcopy(gateway)
    previous = trial.current_vector_id
    for vector, dwell in sequence.slots:
        if dwell < inverter.min_pulse_s or dwell <= inverter.dead_time_s:
            return False
        waveform = transition_waveform(previous, vector, dead_time_ticks=1)
        if has_shoot_through(waveform) or has_direct_leg_transition(waveform):
            return False
        decision = trial.evaluate(AIPwmRequest(
            vector_id=vector, dwell_s=dwell, confidence=0.99,
            predicted_i_abs=predicted_i_abs, measured_i_abs=measured_i_abs,
            vdc=inverter.Vdc, tj_c=inverter.ambient_c,
        ))
        if not decision.accepted or decision.vector_id != vector:
            return False
        previous = vector
    return True


def information_score(observer: ActiveBeliefRotorObserver, voltage: tuple[float, float], dwell_s: float, method: str) -> float:
    if observer.saliency.strength == 0:
        return 0.0
    predicted = observer.predict_response(voltage, dwell_s)
    weights = observer.weights
    angle_mass = np.sum(weights, axis=(1, 2))
    angle_mean = np.sum(weights[..., None] * predicted, axis=(1, 2)) / np.maximum(angle_mass[:, None], 1e-300)
    if method == "separation":
        mean = np.sum(angle_mass[:, None] * angle_mean, axis=0)
        variance = np.sum(angle_mass[:, None] * (angle_mean - mean) ** 2)
        return float(variance / (2.0 * observer.noise_std_a**2))
    if method == "renyi2":
        # Pairwise Gaussian overlap approximates the reduction in collision
        # entropy of the angle marginal. Nuisance gain/speed are marginalized.
        delta = angle_mean[:, None, :] - angle_mean[None, :, :]
        overlap = np.exp(-np.sum(delta * delta, axis=-1) / (4.0 * observer.noise_std_a**2))
        collision = float(np.sum(angle_mass[:, None] * angle_mass[None, :] * overlap))
        return -math.log(max(collision, 1e-300))
    if method == "fisher":
        # Local Fisher information about mechanical angle, not a global
        # identifiability certificate; the slot-pitch ambiguity remains.
        va, vb = voltage
        phase = observer.saliency.harmonic * observer.angles
        c, s = np.cos(phase), np.sin(phase)
        factor = dwell_s * observer.saliency.strength * observer.saliency.harmonic / observer.saliency.differential_inductance_h
        deriv_a = factor * (-s * va + c * vb)
        deriv_b = factor * (c * va + s * vb)
        squared = (deriv_a**2 + deriv_b**2)[:, None, None] * observer.gains[None, None, :] ** 2
        return float(np.sum(weights * squared) / observer.noise_std_a**2) * observer.period_rad**2 / 12.0
    raise ValueError(f"unknown information method: {method}")


def choose_sequence(
    *,
    base_vector: int,
    observer: ActiveBeliefRotorObserver,
    inverter: TwoLevelInverterParams,
    gateway: AIPwmSafetyGateway,
    measured_i_abs: float,
    cfg: CodedPwmConfig,
    force_probe: bool = False,
) -> CodedSequence:
    reference = make_sequence(base_vector, None, inverter, cfg)
    limit = gateway.limits.i_soft_a
    baseline_score = -math.inf if force_probe else 0.0
    best = CodedSequence(reference.slots, None, baseline_score)
    for probe in range(1, 7):
        candidate = make_sequence(base_vector, probe, inverter, cfg)
        voltage = alpha_beta_voltage(probe, inverter)
        peak_increment = cfg.probe_dwell_s * math.hypot(*voltage) / observer.saliency.differential_inductance_h
        predicted_i_abs = measured_i_abs + peak_increment * (1.0 + observer.saliency.strength)
        if predicted_i_abs >= limit:
            continue
        if not legal_sequence(candidate, reference=reference, inverter=inverter, gateway=gateway,
                              measured_i_abs=measured_i_abs, predicted_i_abs=predicted_i_abs,
                              tolerance_v=cfg.mean_tolerance_v):
            continue
        info = information_score(observer, voltage, cfg.probe_dwell_s, cfg.information_method)
        previous = gateway.current_vector_id
        transitions = sum(switch_events(a, b) for a, b in zip((previous,) + tuple(v for v, _ in candidate.slots[:-1]), (v for v, _ in candidate.slots)))
        current_cost = (peak_increment / max(limit, 1e-12)) ** 2
        energy_cost = (peak_increment**2 * observer.saliency.differential_inductance_h) / max(inverter.Vdc * max(measured_i_abs, 0.1) * inverter.t_pwm_s, 1e-12)
        # The first-order mean-voltage torque action is identical for every
        # candidate; this term penalizes additional subcycle ripple only.
        torque_cost = current_cost
        score = info - cfg.lambda_switch * transitions - cfg.lambda_current * current_cost - cfg.lambda_probe_energy * energy_cost - cfg.lambda_torque * torque_cost
        if score > best.information_score:
            best = CodedSequence(candidate.slots, probe, score)
    if best.probe_vector is None and force_probe:
        return reference
    return best


__all__ = ["CodedPwmConfig", "CodedSequence", "average_voltage", "make_sequence", "legal_sequence", "information_score", "choose_sequence"]

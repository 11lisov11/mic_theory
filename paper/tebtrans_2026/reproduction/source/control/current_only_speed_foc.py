"""Averaged-voltage FOC driven by current-only state estimates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from control.current_only_induction_ekf import ObserverMotor


@dataclass(frozen=True)
class SpeedFocConfig:
    flux_reference: float = 0.35
    current_bandwidth: float = 400.0
    speed_bandwidth: float = 20.0
    torque_limit: float = 1.1
    current_limit: float = 2.8
    preflux_s: float = 0.25
    load_feedforward: bool = False


class CurrentOnlySpeedFOC:
    def __init__(self, motor: ObserverMotor, config: SpeedFocConfig | None = None) -> None:
        self.motor = motor
        self.config = config or SpeedFocConfig()
        self.current_integral = np.zeros(2)
        self.speed_integral = 0.0

    def command(self, estimate: np.ndarray, *, speed_reference: float,
                vdc: float, time_s: float, dt: float) -> tuple[float, float]:
        cfg, motor = self.config, self.motor
        ia, ib, pa, pb, omega, load_estimate = estimate
        flux = math.hypot(pa, pb)
        theta = math.atan2(pb, pa) if flux > 1e-5 else 0.0
        c, s = math.cos(theta), math.sin(theta)
        current = np.array([c * ia + s * ib, -s * ia + c * ib])
        speed_error = speed_reference - omega
        kp_speed = 2.0 * motor.inertia * cfg.speed_bandwidth
        ki_speed = motor.inertia * cfg.speed_bandwidth**2
        torque_raw = kp_speed * speed_error + self.speed_integral
        if cfg.load_feedforward:
            torque_raw += float(np.clip(load_estimate, -cfg.torque_limit, cfg.torque_limit))
        torque = float(np.clip(torque_raw, -cfg.torque_limit, cfg.torque_limit))
        if time_s < cfg.preflux_s:
            torque = 0.0
            self.speed_integral = 0.0
        elif abs(torque_raw) < cfg.torque_limit or speed_error * torque_raw < 0:
            self.speed_integral += ki_speed * speed_error * dt
        kr, ar = motor.lm / motor.lr, motor.rr / motor.lr
        id_ref = cfg.flux_reference / motor.lm * min(1.0, time_s / 0.05)
        iq_limit = math.sqrt(max(0.0, cfg.current_limit**2 - id_ref**2))
        iq_ref = float(np.clip(torque / (1.5 * motor.pole_pairs * kr * max(flux, 0.08)),
                              -iq_limit, iq_limit))
        error = np.array([id_ref, iq_ref]) - current
        omega_sync = motor.pole_pairs * omega + ar * motor.lm * current[1] / max(flux, 0.025)
        omega_sync = float(np.clip(omega_sync, -600.0, 600.0))
        resistance = motor.rs + kr * ar * motor.lm
        feedforward = np.array([
            resistance * current[0] - kr * ar * flux - omega_sync * motor.leakage * current[1],
            resistance * current[1] + kr * motor.pole_pairs * omega * flux
            + omega_sync * motor.leakage * current[0],
        ])
        dq_voltage = (feedforward + motor.leakage * cfg.current_bandwidth * error
                      + self.current_integral)
        voltage_limit = 0.95 * vdc / math.sqrt(3.0)
        magnitude = float(np.linalg.norm(dq_voltage))
        limited = dq_voltage * min(1.0, voltage_limit / max(magnitude, 1e-12))
        # Back-calculation prevents voltage saturation from accumulating PI error.
        self.current_integral += dt * (motor.rs * cfg.current_bandwidth * error
                                      + cfg.current_bandwidth * (limited - dq_voltage))
        return float(c * limited[0] - s * limited[1]), float(s * limited[0] + c * limited[1])

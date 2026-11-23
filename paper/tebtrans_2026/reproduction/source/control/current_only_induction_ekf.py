"""Current/voltage-only induction-machine EKF for host experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ObserverMotor:
    rs: float
    rr: float
    ls: float
    lr: float
    lm: float
    inertia: float
    friction: float
    pole_pairs: int

    def __post_init__(self) -> None:
        values = (self.rs, self.rr, self.ls, self.lr, self.lm, self.inertia)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("motor parameters must be finite and positive")
        if self.ls * self.lr <= self.lm**2 or self.pole_pairs < 1:
            raise ValueError("invalid inductance matrix or pole pairs")
        if not math.isfinite(self.friction) or self.friction < 0:
            raise ValueError("invalid friction")

    @property
    def leakage(self) -> float:
        return self.ls - self.lm**2 / self.lr


def derivative_and_jacobian(x: np.ndarray, voltage: np.ndarray,
                            motor: ObserverMotor) -> tuple[np.ndarray, np.ndarray]:
    """State: i_alpha, i_beta, rotor flux alpha/beta, speed, load torque."""
    ia, ib, pa, pb, omega, load = x
    ar, kr = motor.rr / motor.lr, motor.lm / motor.lr
    p, leakage = motor.pole_pairs, motor.leakage
    da = ar * (motor.lm * ia - pa) - p * omega * pb
    db = ar * (motor.lm * ib - pb) + p * omega * pa
    torque_gain = 1.5 * p * kr
    torque = torque_gain * (pa * ib - pb * ia)
    f = np.array([
        (voltage[0] - motor.rs * ia - kr * da) / leakage,
        (voltage[1] - motor.rs * ib - kr * db) / leakage,
        da, db, (torque - load - motor.friction * omega) / motor.inertia, 0.0,
    ])
    jac = np.zeros((6, 6))
    jac[2] = (ar * motor.lm, 0, -ar, -p * omega, -p * pb, 0)
    jac[3] = (0, ar * motor.lm, p * omega, -ar, p * pa, 0)
    jac[0] = -kr * jac[2] / leakage
    jac[1] = -kr * jac[3] / leakage
    jac[0, 0] -= motor.rs / leakage
    jac[1, 1] -= motor.rs / leakage
    jac[4] = np.array([-torque_gain * pb, torque_gain * pa,
                       torque_gain * ib, -torque_gain * ia,
                       -motor.friction, -1.0]) / motor.inertia
    return f, jac


def rk4_transition(x: np.ndarray, voltage: np.ndarray, dt: float,
                    motor: ObserverMotor) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the RK4 transition itself, including its intermediate states."""
    identity = np.eye(6)
    k1, a1 = derivative_and_jacobian(x, voltage, motor)
    k2, a2 = derivative_and_jacobian(x + 0.5 * dt * k1, voltage, motor)
    g2 = a2 @ (identity + 0.5 * dt * a1)
    k3, a3 = derivative_and_jacobian(x + 0.5 * dt * k2, voltage, motor)
    g3 = a3 @ (identity + 0.5 * dt * g2)
    k4, a4 = derivative_and_jacobian(x + dt * k3, voltage, motor)
    g4 = a4 @ (identity + dt * g3)
    next_x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    transition = identity + dt * (a1 + 2 * g2 + 2 * g3 + g4) / 6
    return next_x, transition


class CurrentOnlyInductionEKF:
    def __init__(self, motor: ObserverMotor, measurement_covariance: np.ndarray,
                 *, process_spectral_density: tuple[float, ...] =
                 (0.02, 0.02, 0.00001, 0.00001, 4.0, 0.0001)) -> None:
        self.motor = motor
        covariance = np.asarray(measurement_covariance, dtype=float)
        if (covariance.shape != (2, 2) or not np.all(np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T)
                or np.min(np.linalg.eigvalsh(covariance)) <= 0):
            raise ValueError("measurement covariance must be positive definite")
        spectral = np.asarray(process_spectral_density, dtype=float)
        if spectral.shape != (6,) or np.any(spectral < 0) or not np.all(np.isfinite(spectral)):
            raise ValueError("invalid process spectral density")
        self.r = covariance.copy()
        self.q_spectral = spectral
        self.x = np.zeros(6)
        self.p = np.diag([0.05**2, 0.05**2, 0.05**2, 0.05**2, 30.0**2, 0.05**2])
        self.innovation = np.zeros(2)
        self.innovation_covariance = covariance.copy()
        self.normalized_innovation = 0.0

    def correct(self, current: tuple[float, float]) -> None:
        measurement = np.asarray(current, dtype=float)
        if measurement.shape != (2,) or not np.all(np.isfinite(measurement)):
            raise ValueError("invalid measured current")
        self.innovation = measurement - self.x[:2]
        s = self.p[:2, :2] + self.r
        self.innovation_covariance = s.copy()
        gain = np.linalg.solve(s, self.p[:2, :]).T
        self.x += gain @ self.innovation
        residual = np.eye(6)
        residual[:, :2] -= gain
        self.p = residual @ self.p @ residual.T + gain @ self.r @ gain.T
        self.p = 0.5 * (self.p + self.p.T)
        self.normalized_innovation = float(self.innovation @ np.linalg.solve(s, self.innovation))
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.p)):
            raise FloatingPointError("EKF correction diverged")

    def predict(self, voltage: tuple[float, float], dt: float) -> None:
        value = np.asarray(voltage, dtype=float)
        if value.shape != (2,) or not np.all(np.isfinite(value)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("invalid voltage or interval")
        self.x, transition = rk4_transition(self.x, value, dt, self.motor)
        self.p = transition @ self.p @ transition.T + np.diag(self.q_spectral * dt)
        self.p = 0.5 * (self.p + self.p.T)
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.p)):
            raise FloatingPointError("EKF prediction diverged")

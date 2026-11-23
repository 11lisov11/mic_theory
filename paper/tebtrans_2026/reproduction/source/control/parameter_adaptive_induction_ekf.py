"""Isolated current-only joint state/log-parameter EKF research reference."""

from dataclasses import replace
import math

import numpy as np

from control.current_only_induction_ekf import (
    CurrentOnlyInductionEKF, derivative_and_jacobian,
)


def motor_from_logs(nominal, logs):
    rs, rr, lm = np.array([nominal.rs, nominal.rr, nominal.lm])*np.exp(logs)
    return replace(nominal, rs=float(rs), rr=float(rr), lm=float(lm),
                   ls=float(lm+nominal.ls-nominal.lm),
                   lr=float(lm+nominal.lr-nominal.lm))


def augmented_derivative(x, voltage, nominal):
    motor = motor_from_logs(nominal, x[6:])
    f, a = derivative_and_jacobian(x[:6], voltage, motor)
    ia, ib, pa, pb, omega, _ = x[:6]
    current, flux = np.array([ia, ib]), np.array([pa, pb])
    kr, ar, leakage = motor.lm/motor.lr, motor.rr/motor.lr, motor.leakage
    columns = np.zeros((6, 3))
    columns[:2, 0] = -motor.rs*current/leakage
    dflux_rr = ar*(motor.lm*current-flux)
    columns[2:4, 1] = dflux_rr
    columns[:2, 1] = -kr*dflux_rr/leakage
    dk = (motor.lr-motor.lm)/motor.lr**2
    da = -motor.rr/motor.lr**2
    dflux_lm = da*(motor.lm*current-flux)+ar*current
    columns[2:4, 2] = motor.lm*dflux_lm
    columns[:2, 2] = motor.lm*(-dk*f[2:4]-kr*dflux_lm-f[:2]*(1-kr)**2)/leakage
    columns[4, 2] = motor.lm*1.5*motor.pole_pairs*dk*(pa*ib-pb*ia)/motor.inertia
    jac = np.zeros((9, 9))
    jac[:6, :6], jac[:6, 6:] = a, columns
    return np.r_[f, np.zeros(3)], jac


def augmented_transition(x, voltage, dt, nominal):
    identity = np.eye(9)
    k1, a1 = augmented_derivative(x, voltage, nominal)
    k2, a2 = augmented_derivative(x+.5*dt*k1, voltage, nominal)
    g2 = a2@(identity+.5*dt*a1)
    k3, a3 = augmented_derivative(x+.5*dt*k2, voltage, nominal)
    g3 = a3@(identity+.5*dt*g2)
    k4, a4 = augmented_derivative(x+dt*k3, voltage, nominal)
    g4 = a4@(identity+dt*g3)
    return x+dt*(k1+2*k2+2*k3+k4)/6, identity+dt*(a1+2*g2+2*g3+g4)/6


class ParameterAdaptiveInductionEKF:
    def __init__(self, nominal, measurement_covariance, *, active=(True, True, True),
                 parameter_density=1e-5, load_density=.01):
        if len(active) != 3 or not all(isinstance(v, bool) for v in active):
            raise ValueError("three boolean parameter switches required")
        if not all(math.isfinite(v) and v >= 0 for v in (parameter_density, load_density)):
            raise ValueError("invalid process density")
        reference = CurrentOnlyInductionEKF(nominal, measurement_covariance)
        self.nominal, self.r = nominal, reference.r
        self.x = np.zeros(9)
        self.p = np.zeros((9, 9))
        self.p[:6, :6] = reference.p
        self.prior_parameter_variance = .2**2
        self.p[6:, 6:] = np.diag(np.array(active)*self.prior_parameter_variance)
        self.q_spectral = np.r_[reference.q_spectral, np.array(active)*parameter_density]
        self.q_spectral[5] = load_density
        self.innovation, self.innovation_covariance = reference.innovation, reference.innovation_covariance
        self.normalized_innovation = 0.
        self.bound_hits = 0

    def controller_motor(self, *, confidence=False):
        logs = self.x[6:].copy()
        if confidence:
            # Posterior contraction is a heuristic, not an identifiability proof.
            contraction = np.clip(1-np.diag(self.p)[6:]/self.prior_parameter_variance, 0., 1.)
            logs *= contraction
        return motor_from_logs(self.nominal, logs)

    def correct(self, current):
        y = np.asarray(current, dtype=float)
        if y.shape != (2,) or not np.all(np.isfinite(y)):
            raise ValueError("invalid measured current")
        self.innovation = y-self.x[:2]
        s = self.p[:2, :2]+self.r
        self.innovation_covariance = s.copy()
        gain = np.linalg.solve(s, self.p[:2, :]).T
        self.x += gain@self.innovation
        residual = np.eye(9)
        residual[:, :2] -= gain
        self.p = residual@self.p@residual.T+gain@self.r@gain.T
        self.p = .5*(self.p+self.p.T)
        self.normalized_innovation = float(self.innovation@np.linalg.solve(s, self.innovation))
        bounded = np.clip(self.x[6:], -.7, .7)
        self.bound_hits += int(np.any(bounded != self.x[6:]))
        self.x[6:] = bounded
        self._check()

    def predict(self, voltage, dt):
        value = np.asarray(voltage, dtype=float)
        if value.shape != (2,) or not np.all(np.isfinite(value)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("invalid voltage or interval")
        self.x, transition = augmented_transition(self.x, value, dt, self.nominal)
        self.p = transition@self.p@transition.T+np.diag(self.q_spectral*dt)
        self.p = .5*(self.p+self.p.T)
        self._check()

    def _check(self):
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.p)):
            raise FloatingPointError("joint parameter EKF diverged")

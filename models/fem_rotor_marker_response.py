"""Periodic terminal impedance from the reduced AIR56 FEM study."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Protocol

import numpy as np


class RotorPulseResponse(Protocol):
    @property
    def harmonic(self) -> int: ...

    @property
    def period_rad(self) -> float: ...

    @property
    def strength(self) -> float: ...

    @property
    def differential_inductance_h(self) -> float: ...

    def current_increment(self, theta_m: float, v_alpha: float, v_beta: float,
                          dwell_s: float, *, gain: float = 1.0) -> tuple[float, float]: ...


@dataclass(frozen=True, eq=False)
class FemRotorMarkerResponse:
    """A frozen operating point, with a separate correction for model uncertainty.

    The coded-PWM interface uses a one-frequency quadrature approximation for
    the pulse correction. Use phasor_current for an actual sinusoidal probe.
    """
    frequency_hz: float
    differential_inductance_h: float
    harmonic: int
    z_stator_ohm: np.ndarray
    delta_z_ohm: np.ndarray
    mesh_error_ohm: np.ndarray
    profile: str
    _pulse_matrix: np.ndarray
    strength: float
    marker_present: bool
    data_sha256: str

    @classmethod
    def load(cls, path: Path, frequency_hz: float, differential_inductance_h: float, *,
             stator_resistance_ohm: float = 0.0, marker_present: bool = True):
        if (not math.isfinite(differential_inductance_h) or differential_inductance_h <= 0
                or not math.isfinite(stator_resistance_ohm) or stator_resistance_ohm < 0
                or not math.isfinite(frequency_hz)):
            raise ValueError("invalid operating frequency, resistance or differential inductance")
        with np.load(path, allow_pickle=False) as data:
            theta, frequencies = data["theta_rad"], data["frequencies_hz"]
            delta, error, z_stator = data["delta_z_ohm"], data["mesh_error_ohm"], data["stator_z_ohm"]
            poles = int(data["pole_pairs"].item())
            profile = str(data["profile"].item())
        count = len(theta)
        n_freq = len(frequencies)
        if (poles < 1 or count < 16 or count % (2*poles) or n_freq < 2 or
                not np.allclose(theta, np.arange(count)*2*np.pi/count) or
                not np.isfinite(frequencies).all() or frequencies[0] <= 0 or
                np.any(np.diff(frequencies) <= 0) or delta.shape != (n_freq, count, 2, 2) or
                error.shape != delta.shape or z_stator.shape != (n_freq, 2, 2) or
                not all(np.isfinite(x).all() for x in (delta, error, z_stator)) or np.any(error < 0)):
            raise ValueError("invalid FEM response table")
        if not frequencies[0] <= frequency_hz <= frequencies[-1]:
            raise ValueError("frequency is outside the computed FEM band")
        slots = count//(2*poles)
        if not np.allclose(delta, np.roll(delta, slots, axis=1), rtol=1e-6, atol=1e-15):
            raise ValueError("FEM table does not satisfy the declared half-electrical-turn symmetry")
        lo = min(np.searchsorted(frequencies, frequency_hz, side="right")-1, n_freq-2)
        weight = math.log(frequency_hz/frequencies[lo]) / math.log(frequencies[lo+1]/frequencies[lo])

        def interp(values):
            return (1-weight)*values[lo]+weight*values[lo+1]

        base = interp(z_stator)+stator_resistance_ohm*np.eye(2)
        correction = interp(delta)[:slots] if marker_present else np.zeros((slots, 2, 2), dtype=complex)
        uncertainty = interp(error)[:slots] if marker_present else np.zeros((slots, 2, 2))
        # Algebraic inverse contrast avoids subtracting two close admittances.
        y = np.linalg.inv(base+correction)
        dy = -(y @ correction @ np.linalg.inv(base))
        pulse = np.eye(2)/differential_inductance_h-2*np.pi*frequency_hz*dy.imag
        strength = float(np.max(np.linalg.norm(pulse*differential_inductance_h-np.eye(2), ord=2, axis=(1, 2))))
        return cls(float(frequency_hz), differential_inductance_h, 2*poles, base, correction,
                   uncertainty, profile, pulse, strength, marker_present,
                   hashlib.sha256(Path(path).read_bytes()).hexdigest())

    @property
    def period_rad(self) -> float:
        return 2*np.pi/self.harmonic

    def _at(self, values: np.ndarray, theta_m):
        position = np.mod(np.asarray(theta_m, dtype=float), self.period_rad)*len(values)/self.period_rad
        lower = np.floor(position).astype(int)
        fraction = (position-lower)[..., None, None]
        return (1-fraction)*values[lower % len(values)]+fraction*values[(lower+1) % len(values)]

    def impedance(self, theta_m):
        return self.z_stator_ohm+self._at(self.delta_z_ohm, theta_m)

    def uncertainty_ohm(self, theta_m):
        return self._at(self.mesh_error_ohm, theta_m)

    def phasor_current(self, theta_m, voltage):
        voltage = np.broadcast_to(np.asarray(voltage), (*np.shape(theta_m), 2))
        return np.linalg.solve(self.impedance(theta_m), voltage[..., None])[..., 0]

    def phasor_delta_current(self, theta_m, voltage):
        z_delta = self._at(self.delta_z_ohm, theta_m)
        y0v = np.linalg.solve(self.z_stator_ohm, voltage)
        # Resolvent identity, avoiding cancellation against the base current.
        rhs = -(z_delta @ y0v)
        return np.linalg.solve(self.z_stator_ohm+z_delta, rhs[..., None])[..., 0]

    def marker_power_delta_w(self, theta_m, voltage):
        voltage = np.asarray(voltage)
        di = self.phasor_delta_current(theta_m, voltage)
        # Peak alpha-beta convention, same 3/2 power normalization as the plant.
        return float(0.75*np.vdot(voltage, di).real)
    def response_matrix(self, theta_m, dwell_s):
        return max(float(dwell_s), 0)*self._at(self._pulse_matrix, theta_m)

    def response_derivative(self, theta_m, dwell_s):
        h = self.period_rad/len(self._pulse_matrix)/2
        return (self.response_matrix(theta_m+h, dwell_s)-self.response_matrix(theta_m-h, dwell_s))/(2*h)

    def current_increment(self, theta_m, v_alpha, v_beta, dwell_s, *, gain=1.0):
        result = float(gain)*(self.response_matrix(theta_m, dwell_s) @ (v_alpha, v_beta))
        return float(result[0]), float(result[1])

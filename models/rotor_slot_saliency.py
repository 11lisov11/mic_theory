"""Optional small-signal rotor-slot sensing channel for host experiments.

This is an explicit, unvalidated research extension. It does not alter the
accepted symmetric alpha-beta plant or assert that a particular motor has a
measurable slot harmonic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SlotSaliency:
    harmonic: int = 28
    strength: float = 0.0
    differential_inductance_h: float = 0.05

    def __post_init__(self) -> None:
        if self.harmonic < 1 or not (0.0 <= self.strength < 1.0):
            raise ValueError("harmonic must be positive and strength in [0, 1)")
        if self.differential_inductance_h <= 0.0:
            raise ValueError("differential inductance must be positive")

    @property
    def period_rad(self) -> float:
        return 2.0 * math.pi / self.harmonic

    def current_increment(
        self,
        theta_m: float,
        v_alpha: float,
        v_beta: float,
        dwell_s: float,
        *,
        gain: float = 1.0,
    ) -> tuple[float, float]:
        """Linearized HF response: di=(dt/L)*g*(I+s*Q(N*theta))*v.

        Q is a traceless reflection matrix. The signal repeats every rotor
        slot pitch. `gain` represents uncertain HF inductance/calibration.
        """
        phase = self.harmonic * theta_m
        c, s = math.cos(phase), math.sin(phase)
        scale = max(float(dwell_s), 0.0) * float(gain) / self.differential_inductance_h
        return (
            scale * (v_alpha + self.strength * (c * v_alpha + s * v_beta)),
            scale * (v_beta + self.strength * (s * v_alpha - c * v_beta)),
        )


__all__ = ["SlotSaliency"]

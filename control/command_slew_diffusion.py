"""Command-only speed-hypothesis diffusion for transient rotor observation."""

from __future__ import annotations

import math


STEADY_PROBABILITY = 0.001
TRANSIENT_PROBABILITY = 0.05
SLEW_THRESHOLD_RAD_S_PER_STEP = 0.005
HOLD_STEPS = 50


class CommandSlewDiffusion:
    def __init__(self, *, steady_probability: float = STEADY_PROBABILITY,
                 transient_probability: float = TRANSIENT_PROBABILITY,
                 slew_threshold_rad_s_per_step: float = SLEW_THRESHOLD_RAD_S_PER_STEP,
                 hold_steps: int = HOLD_STEPS) -> None:
        if (not 0.0 <= steady_probability <= transient_probability <= 0.5
                or not math.isfinite(slew_threshold_rad_s_per_step)
                or slew_threshold_rad_s_per_step <= 0.0 or hold_steps < 1):
            raise ValueError("invalid command-slew diffusion configuration")
        self.steady_probability = float(steady_probability)
        self.transient_probability = float(transient_probability)
        self.slew_threshold_rad_s_per_step = float(slew_threshold_rad_s_per_step)
        self.hold_steps = int(hold_steps)
        self._previous_command: float | None = None
        self._remaining = 0

    def update(self, commanded_speed_rad_s: float) -> float:
        if not math.isfinite(commanded_speed_rad_s):
            raise ValueError("commanded speed must be finite")
        if (self._previous_command is not None
                and abs(commanded_speed_rad_s - self._previous_command)
                >= self.slew_threshold_rad_s_per_step):
            self._remaining = self.hold_steps
        self._previous_command = commanded_speed_rad_s
        if self._remaining:
            self._remaining -= 1
            return self.transient_probability
        return self.steady_probability

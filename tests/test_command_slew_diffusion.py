from __future__ import annotations

import math

import pytest

from control.command_slew_diffusion import CommandSlewDiffusion
from tools.run_info_optimal_coded_pwm_study import Case, run_trial
from tools.study_coded_pwm_confidence_gate import _trial


def test_command_slew_transient_and_hold_have_no_truth_input() -> None:
    schedule = CommandSlewDiffusion(hold_steps=3)
    assert schedule.update(1.0) == 0.001
    assert schedule.update(1.0) == 0.001
    assert schedule.update(1.01) == 0.05
    assert schedule.update(1.01) == 0.05
    assert schedule.update(1.01) == 0.05
    assert schedule.update(1.01) == 0.001
    with pytest.raises(ValueError):
        schedule.update(math.nan)


def test_constant_command_matches_original_observer_exactly() -> None:
    case = Case(motor="AIR56", scenario="constant", speed_fraction=0.01,
                strength=0.15, nonideal=True, mismatch="combined",
                seed=35000, steps=30, measurement_noise_beta_ratio=3.0)
    baseline, widths = _trial(case)
    scheduled, scheduled_widths = _trial(case, command_slew_speed_diffusion=True)
    assert baseline["trace_estimate_rad"] == scheduled["trace_estimate_rad"]
    assert widths.tolist() == scheduled_widths.tolist()
    with pytest.raises(ValueError):
        run_trial(case, "B2_fixed_probe", observer_profile="legacy",
                  command_slew_speed_diffusion=True)

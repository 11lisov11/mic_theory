from __future__ import annotations

import numpy as np
import pytest

from tools.run_info_optimal_coded_pwm_study import Case, run_trial
from tools.study_coded_pwm_confidence_gate import _trial
from tools.study_coded_pwm_speed_diffusion import run_study


def test_explicit_baseline_diffusion_matches_default() -> None:
    case = Case(motor="AIR56", scenario="ramp", speed_fraction=0.02,
                strength=0.15, nonideal=True, mismatch="combined",
                seed=34000, steps=8, measurement_noise_beta_ratio=3.0)
    default, default_width = _trial(case)
    explicit, explicit_width = _trial(case, speed_diffusion_override=0.001)
    assert np.array_equal(default_width, explicit_width)
    assert default["trace_estimate_rad"] == explicit["trace_estimate_rad"]
    with pytest.raises(ValueError):
        run_trial(case, "B2_fixed_probe", observer_profile="robust",
                  speed_diffusion_override=0.6)


def test_speed_diffusion_study_is_paired() -> None:
    summary, rows = run_study(seed_offset=34000, seeds=1, steps=6,
                              profiles=(("ramp", 0.02),), ratios=(3.0,),
                              motors=("AIR56",))
    assert len(rows) == 2
    assert rows[0]["stimulus_digest"] == rows[1]["stimulus_digest"]
    assert summary["trial_count_per_arm"] == 1
    assert summary["gateway_rejections"] == 0

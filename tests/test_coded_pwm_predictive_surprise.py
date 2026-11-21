from __future__ import annotations

import numpy as np

from tools.study_coded_pwm_predictive_surprise import _select_lowest_within, run_study


def test_matched_selection_stays_inside_first_stage() -> None:
    selected = _select_lowest_within(
        np.asarray([True, False, True, True]),
        np.asarray([3.0, -10.0, 1.0, 2.0]), 2)
    assert selected.tolist() == [False, False, True, True]


def test_predictive_surprise_holdout_smoke() -> None:
    summary, rows = run_study(seed_offset=36100, seeds=1, steps=6,
                              profiles=(("ramp", 0.02),), ratios=(8.0,),
                              motors=("AIR56",))
    assert len(rows) == summary["trial_count"] == 1
    assert 0 <= rows[0]["second_stage_count"] <= rows[0]["width_accepted_count"] <= 6
    assert summary["gateway_rejections"] == 0
    assert summary["absolute_angle_claim"] is False

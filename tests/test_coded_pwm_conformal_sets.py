from __future__ import annotations

import pytest

from tools.study_coded_pwm_conformal_sets import _threshold, run_study


def test_group_order_statistic_has_explicit_small_sample_fallback() -> None:
    assert _threshold([0.2, 0.4], 0.95) == 1.0
    assert _threshold([index / 20 for index in range(20)], 0.95) == 0.95
    with pytest.raises(ValueError):
        _threshold([], 0.95)
    with pytest.raises(ValueError):
        _threshold([1.2], 0.95)


def test_conformal_study_smoke_keeps_calibration_and_holdout_disjoint() -> None:
    payload, calibration, holdout = run_study(
        calibration_offset=28000, holdout_offset=29000,
        calibration_seeds=1, holdout_seeds=1, steps=6,
        motors=("AIR56",), ratios=(2.0,))
    assert payload["global_rank_threshold"] == 1.0
    assert payload["holdout_all_scenarios_pass_count"] == 1
    assert len(calibration) == len(holdout) == 1
    assert holdout[0]["calibrated_coverage"] == 1.0
    assert payload["gateway_rejections"] == 0
    with pytest.raises(ValueError):
        run_study(calibration_offset=28000, holdout_offset=28000,
                  calibration_seeds=1, holdout_seeds=1, steps=6,
                  motors=("AIR56",), ratios=(2.0,))

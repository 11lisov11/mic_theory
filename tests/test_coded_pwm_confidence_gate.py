from __future__ import annotations

import numpy as np

from tools.study_coded_pwm_confidence_gate import _matched_posterior_baselines, run_study


def test_posterior_baselines_accept_identical_counts_without_labels() -> None:
    trace = [[0.9, 0.1, 0.0], [0.6, 0.4, 0.0], [0.4, 0.3, 0.3]]
    entropy, peak = _matched_posterior_baselines(trace, 2)
    assert np.array_equal(entropy, [True, True, False])
    assert np.array_equal(peak, [True, True, False])


def test_gate_study_smoke_has_disjoint_seed_groups_and_finite_metrics() -> None:
    payload, calibration, holdout = run_study(
        calibration_offset=30000, holdout_offset=31000,
        calibration_seeds=1, holdout_seeds=1, steps=6,
        motors=("AIR56",), ratios=(2.0,))
    assert len(calibration) == len(holdout) == 1
    assert payload["gateway_rejections"] == 0
    assert payload["absolute_angle_claim"] is False
    assert payload["hardware_claim"] is False
    assert 0.0 <= holdout[0]["availability"] <= 1.0
    assert 0.0 <= holdout[0]["bad_all_fraction"] <= 1.0
    assert 0 <= holdout[0]["bad_entropy_count"] <= holdout[0]["accepted_count"]
    assert 0 <= holdout[0]["bad_peak_count"] <= holdout[0]["accepted_count"]
    assert "mean_baseline_minus_gate" in payload["primary_random_minus_gate"]
    assert "mean_baseline_minus_gate" in payload["secondary_entropy_minus_gate"]
    assert "mean_baseline_minus_gate" in payload["secondary_peak_minus_gate"]

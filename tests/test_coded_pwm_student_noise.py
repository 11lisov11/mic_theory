from __future__ import annotations

from tools.study_coded_pwm_student_noise import run_study


def test_student_study_smoke_is_paired_and_macro_metric_is_finite() -> None:
    payload, rows = run_study(seed_offset=26000, seeds=1, steps=6,
                              actual_ratios=(2.0,), motors=("AIR56",))
    assert payload["trial_count"] == 3
    assert payload["gateway_rejections"] == 0
    assert payload["absolute_angle_claim"] is False
    assert payload["hardware_claim"] is False
    assert len({r["stimulus_digest"] for r in rows}) == 1
    assert -1e3 < payload["macro_student_vs_block"]["mean_relative_reduction"] < 1.0

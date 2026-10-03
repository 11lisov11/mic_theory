from __future__ import annotations

from tools.study_coded_pwm_persistent_noise import run_study


def test_persistent_noise_study_smoke_is_paired() -> None:
    payload, rows = run_study(seed_offset=21500, seeds=1, steps=6, motors=("AIR56",))
    assert payload["trial_count"] == 8
    assert payload["gateway_rejections"] == 0
    assert payload["absolute_angle_claim"] is False
    assert payload["hardware_claim"] is False
    for ratio in (1.0, 4.0):
        matched = [r for r in rows if r["actual_ratio"] == ratio]
        assert len(matched) == 4
        assert len({r["stimulus_digest"] for r in matched}) == 1
        assert 0.0 <= next(r["true_mode_probability"] for r in matched
                           if r["arm"] == "persistent_t2") <= 1.0


def test_out_of_grid_noise_has_no_false_true_mode_label() -> None:
    payload, rows = run_study(seed_offset=23000, seeds=1, steps=6,
                              actual_noise_ratios=(2.0,), motors=("AIR56",))
    assert payload["actual_noise_ratios"] == [2.0]
    assert payload["cells"][0]["persistent_mean_true_mode_probability"] is None
    assert next(r for r in rows if r["arm"] == "persistent_t2")["true_mode_probability"] is None

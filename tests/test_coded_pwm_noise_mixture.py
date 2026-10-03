from __future__ import annotations

from tools.study_coded_pwm_noise_mixture import run_study


def test_noise_mixture_study_smoke_is_paired_and_host_only() -> None:
    payload, rows = run_study(seed_offset=19000, seeds=1, steps=6, motors=("AIR56",))
    assert payload["trial_count"] == 8
    assert payload["gateway_rejections"] == 0
    assert payload["absolute_angle_claim"] is False
    assert payload["hardware_claim"] is False
    for ratio in (1.0, 4.0):
        matched = [row for row in rows if row["actual_ratio"] == ratio]
        assert len(matched) == 4
        assert len({row["stimulus_digest"] for row in matched}) == 1
    assert len(payload["cells"]) == 2

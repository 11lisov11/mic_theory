from __future__ import annotations

import pytest

from tools.study_coded_pwm_noise_miscalibration import _paired_effect, run_study


def test_paired_effect_keeps_seed_order_and_sign() -> None:
    result = _paired_effect([2.0, 4.0], [1.0, 2.0])
    assert result["mean_relative_reduction"] == pytest.approx(0.5)
    assert result["bootstrap95_low"] == pytest.approx(0.5)
    assert result["bootstrap95_high"] == pytest.approx(0.5)


def test_miscalibration_study_quick_smoke_has_shared_stimuli() -> None:
    payload, rows = run_study(seed_offset=17800, seeds=1, steps=6, motors=("AIR56",))
    assert payload["trial_count"] == 12
    assert payload["gateway_rejections"] == 0
    assert payload["absolute_angle_claim"] is False
    assert len({r["stimulus_digest"] for r in rows if r["actual_ratio"] == 1.0}) == 1
    assert len({r["stimulus_digest"] for r in rows if r["actual_ratio"] == 4.0}) == 1
    assert len(payload["cells"]) == 4

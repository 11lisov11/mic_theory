from __future__ import annotations

from tools.study_coded_pwm_confidence_transfer import run_study


def test_motion_transfer_smoke_uses_frozen_threshold() -> None:
    summary, rows = run_study(
        frozen_threshold_deg=4.821428571428572, seed_offset=33000,
        seeds=1, steps=6, profiles=(("constant", 0.01),),
        ratios=(3.0,), motors=("AIR56",))
    assert len(rows) == summary["trial_count"] == 1
    assert summary["frozen_threshold_deg"] == 4.821428571428572
    assert summary["motion_is_prescribed"] is True
    assert summary["gateway_rejections"] == 0
    assert 0 <= rows[0]["bad_gate_count"] <= rows[0]["accepted_count"] <= 6

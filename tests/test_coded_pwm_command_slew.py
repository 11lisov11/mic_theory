from __future__ import annotations

from tools.study_coded_pwm_command_slew import run_study


def test_command_slew_study_is_three_way_paired() -> None:
    summary, rows = run_study(seed_offset=35000, seeds=1, steps=6,
                              profiles=(("ramp", 0.02),), ratios=(3.0,),
                              motors=("AIR56",))
    assert len(rows) == 3
    assert {row["arm"] for row in rows} == {"baseline", "fixed_high", "command_slew"}
    assert len({row["stimulus_digest"] for row in rows}) == 1
    assert summary["trial_count_per_arm"] == 1
    assert summary["gateway_rejections"] == 0

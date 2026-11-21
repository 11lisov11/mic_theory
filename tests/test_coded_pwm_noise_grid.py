from __future__ import annotations

import numpy as np

from tools.study_coded_pwm_noise_grid import DENSE_RATIOS, run_study


def test_log_grid_is_fixed_and_monotone() -> None:
    np.testing.assert_allclose(DENSE_RATIOS, (1.0, 2**0.5, 2.0, 2**1.5, 4.0, 2**2.5, 8.0))


def test_noise_grid_study_smoke_is_paired_and_host_only() -> None:
    payload, rows = run_study(seed_offset=24000, seeds=1, steps=6,
                              actual_ratios=(2.0,), motors=("AIR56",))
    assert payload["trial_count"] == 4
    assert payload["gateway_rejections"] == 0
    assert payload["absolute_angle_claim"] is False
    assert payload["hardware_claim"] is False
    assert len({r["stimulus_digest"] for r in rows}) == 1
    assert {r["arm"] for r in rows} == set(payload["arms"])

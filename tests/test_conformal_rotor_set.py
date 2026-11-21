from __future__ import annotations

import math

import numpy as np
import pytest

from control.conformal_rotor_set import rank_mass, ranked_set
from tools.run_info_optimal_coded_pwm_study import Case, exogenous, run_trial


def test_ranked_set_uses_posterior_only_and_is_nested() -> None:
    probabilities = np.asarray([0.5, 0.3, 0.2])
    np.testing.assert_allclose(rank_mass(probabilities), (0.0, 0.5, 0.8))
    first, width_first = ranked_set(probabilities, 0.5, 1.0)
    second, width_second = ranked_set(probabilities, 0.8, 1.0)
    np.testing.assert_array_equal(first, (0, 1))
    np.testing.assert_array_equal(second, (0, 1, 2))
    assert width_first < width_second


@pytest.mark.parametrize("marginal", [np.asarray([0.0, 0.0]),
                                      np.asarray([0.5, math.nan]),
                                      np.asarray([-0.5, 1.5])])
def test_invalid_posterior_rejected(marginal: np.ndarray) -> None:
    with pytest.raises(ValueError):
        rank_mass(marginal)


def test_trial_posterior_capture_is_optional_and_matches_trace_length() -> None:
    case = Case(motor="AIR56", scenario="standstill", strength=0.15, nonideal=True,
                mismatch="combined", seed=28000, steps=6, measurement_noise_beta_ratio=2.0)
    stimulus = exogenous(case)
    options = {"stimulus": stimulus, "observer_profile": "robust", "fixed_probe_vector": 6,
               "assumed_noise_beta_ratio": 1.0,
               "noise_beta_ratio_candidates": (1.0, 4.0), "likelihood_temperature": 2.0}
    captured = run_trial(case, "B2_fixed_probe", capture_posterior_trace=True, **options)
    ordinary = run_trial(case, "B2_fixed_probe", **options)
    assert "trace_angle_marginal" not in ordinary
    assert len(captured["trace_angle_marginal"]) == len(captured["trace_theta_rad"]) == case.steps
    assert captured["trace_theta_rad"] == ordinary["trace_theta_rad"]
    assert all(sum(marginal) == pytest.approx(1.0) for marginal in captured["trace_angle_marginal"])

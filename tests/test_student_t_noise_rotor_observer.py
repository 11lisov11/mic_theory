from __future__ import annotations

import math

import numpy as np
import pytest

from control.student_t_noise_rotor_observer import StudentTNoiseRotorObserver
from models.rotor_slot_saliency import SlotSaliency
from tools.run_info_optimal_coded_pwm_study import Case, exogenous, run_trial


def test_student_t_update_is_normalized_and_has_finite_credible_set() -> None:
    observer = StudentTNoiseRotorObserver(SlotSaliency(strength=0.15), noise_std_a=0.002,
                                          beta_df=3.0, beta_scale_ratio=2.0)
    observer.update_observations([((0.004, -0.005), (80.0, 0.0), 10e-6)], 100e-6)
    assert np.sum(observer.weights) == pytest.approx(1.0)
    indices, width = observer.credible_set(0.95)
    assert len(indices) > 0
    assert 0.0 < width <= observer.period_rad


@pytest.mark.parametrize("value", [0.0, 2.0, math.nan, math.inf])
def test_invalid_student_degrees_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        StudentTNoiseRotorObserver(SlotSaliency(strength=0.15), noise_std_a=0.002,
                                   beta_df=value)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_invalid_student_scales_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        StudentTNoiseRotorObserver(SlotSaliency(strength=0.15), noise_std_a=0.002,
                                   beta_scale_ratio=value)


def test_student_trial_uses_same_exogenous_data_as_fixed_baseline() -> None:
    case = Case(motor="AIR56", scenario="standstill", strength=0.15, nonideal=True,
                mismatch="combined", seed=26000, steps=6, measurement_noise_beta_ratio=3.0)
    stimulus = exogenous(case)
    student = run_trial(case, "B2_fixed_probe", stimulus=stimulus,
                        observer_profile="robust", fixed_probe_vector=6,
                        student_t_beta_df=3.0, student_t_beta_scale_ratio=2.0,
                        likelihood_temperature=2.0)
    fixed = run_trial(case, "B2_fixed_probe", stimulus=stimulus,
                      observer_profile="robust", fixed_probe_vector=6,
                      assumed_noise_beta_ratio=1.0,
                      noise_beta_ratio_candidates=(1.0, 4.0), likelihood_temperature=2.0)
    assert student["gateway_rejections"] == fixed["gateway_rejections"] == 0
    assert student["trace_theta_rad"] == fixed["trace_theta_rad"]

from __future__ import annotations

import math

import numpy as np
import pytest

from control.active_belief_rotor_observer import ActiveBeliefRotorObserver
from control.persistent_noise_rotor_observer import PersistentNoiseRotorObserver
from models.rotor_slot_saliency import SlotSaliency
from tools.run_info_optimal_coded_pwm_study import Case, exogenous, run_trial


def test_first_update_matches_equal_block_mixture_at_unit_temperature() -> None:
    saliency = SlotSaliency(strength=0.15)
    kwargs = {"noise_std_a": 0.002, "speed_hypotheses": (-1.0, 0.0, 1.0)}
    block = ActiveBeliefRotorObserver(saliency, noise_beta_ratio_candidates=(1.0, 4.0), **kwargs)
    persistent = PersistentNoiseRotorObserver(saliency, noise_ratios=(1.0, 4.0), **kwargs)
    observations = [((0.004, -0.005), (80.0, 0.0), 10e-6),
                    ((0.003, 0.002), (40.0, 60.0), 5e-6)]
    block.update_observations(observations, 100e-6)
    persistent.update_observations(observations, 100e-6)
    np.testing.assert_allclose(persistent.weights, block.weights, atol=1e-14)
    assert np.sum(persistent.noise_mode_probabilities()) == pytest.approx(1.0)


def test_persistent_mode_learns_from_current_residuals_without_angle() -> None:
    observer = PersistentNoiseRotorObserver(SlotSaliency(strength=0.15), noise_std_a=0.002,
                                            noise_ratios=(1.0, 4.0))
    for _ in range(4):
        observer.update_observations([((0.0, 0.0), (0.0, 0.0), 10e-6)], 100e-6)
    assert observer.noise_mode_probabilities()[0] > 0.99
    assert np.sum(observer.weights) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        observer.angle_marginal(lag=1)


@pytest.mark.parametrize("ratios", [(), (1.0, 1.0), (0.0, 4.0), (math.nan,), (math.inf,)])
def test_invalid_persistent_modes_are_rejected(ratios: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        PersistentNoiseRotorObserver(SlotSaliency(strength=0.15), noise_std_a=0.002,
                                     noise_ratios=ratios)


@pytest.mark.parametrize("conflict", [
    {"noise_beta_ratio_candidates": (1.0, 4.0)},
    {"noise_std_axes_a": (0.002, 0.008)},
])
def test_persistent_model_rejects_ignored_noise_settings(conflict: dict) -> None:
    with pytest.raises(ValueError):
        PersistentNoiseRotorObserver(SlotSaliency(strength=0.15), noise_std_a=0.002,
                                     noise_ratios=(1.0, 4.0), **conflict)


def test_persistent_trial_smoke_uses_same_exogenous_data_as_block_mixture() -> None:
    case = Case(motor="AIR56", scenario="standstill", strength=0.15, nonideal=True,
                mismatch="combined", seed=21500, steps=6, measurement_noise_beta_ratio=4.0)
    stimulus = exogenous(case)
    persistent = run_trial(case, "B2_fixed_probe", stimulus=stimulus,
                           observer_profile="robust", fixed_probe_vector=6,
                           persistent_noise_ratios=(1.0, 4.0), likelihood_temperature=2.0)
    block = run_trial(case, "B2_fixed_probe", stimulus=stimulus,
                      observer_profile="robust", fixed_probe_vector=6,
                      assumed_noise_beta_ratio=1.0, noise_beta_ratio_candidates=(1.0, 4.0),
                      likelihood_temperature=2.0)
    assert persistent["gateway_rejections"] == block["gateway_rejections"] == 0
    assert persistent["trace_theta_rad"] == block["trace_theta_rad"]
    assert sum(persistent["noise_mode_probabilities"].values()) == pytest.approx(1.0)

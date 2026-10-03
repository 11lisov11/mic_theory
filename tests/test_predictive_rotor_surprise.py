from __future__ import annotations

import math

import numpy as np
import pytest

from control.active_belief_rotor_observer import ActiveBeliefRotorObserver
from models.rotor_slot_saliency import SlotSaliency
from tools.run_info_optimal_coded_pwm_study import Case
from tools.study_coded_pwm_confidence_gate import _trial


def test_predictive_evidence_is_prior_weighted_likelihood() -> None:
    observer = ActiveBeliefRotorObserver(
        SlotSaliency(harmonic=28, strength=0.15, differential_inductance_h=0.045),
        noise_std_a=0.002, n_angle=8, speed_hypotheses=(0.0,),
        gain_hypotheses=(1.0,))
    measurement = (0.01, -0.02)
    voltage = (4.0, 3.0)
    dwell_s = 10e-6
    prediction = observer.predict_response(voltage, dwell_s)
    residual = (prediction - np.asarray(measurement)) / 0.002
    likelihood = np.exp(-0.5 * np.sum(residual * residual, axis=-1))
    expected = math.log(float(np.sum(observer.weights * likelihood)))
    observer.update_observations([(measurement, voltage, dwell_s)], dt_s=0.0)
    assert observer.last_predictive_log_evidence == pytest.approx(expected)
    assert float(np.sum(observer.weights)) == pytest.approx(1.0)


def test_research_trace_exposes_finite_surprise_without_truth_input() -> None:
    result, widths = _trial(Case(motor="AIR56", scenario="ramp",
                                 speed_fraction=0.02, strength=0.15,
                                 nonideal=True, mismatch="combined",
                                 seed=36000, steps=8,
                                 measurement_noise_beta_ratio=8.0))
    scores = np.asarray(result["trace_predictive_surprise"])
    assert len(scores) == len(widths) == 8
    assert np.all(np.isfinite(scores))

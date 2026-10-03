from __future__ import annotations

import csv
import inspect
import math

import numpy as np
import pytest

from control.active_belief_rotor_observer import ActiveBeliefRotorObserver
from control.info_optimal_coded_pwm import (
    CodedPwmConfig, CodedSequence, average_voltage, choose_sequence, legal_sequence, make_sequence,
)
from models.rotor_slot_saliency import SlotSaliency
from models.two_level_inverter import TwoLevelInverterParams
from safety.ai_pwm_gateway import AIPwmSafetyGateway, GatewayLimits, has_shoot_through, transition_waveform
from tools.run_info_optimal_coded_pwm_study import Case, METHODS, exogenous, phase_zero_check, run_suite, run_trial


def _gateway() -> AIPwmSafetyGateway:
    return AIPwmSafetyGateway(GatewayLimits(
        vdc_min_v=40, vdc_max_v=110, i_soft_a=3, i_trip_a=4,
        max_switch_events_per_window=48, switch_window_steps=8,
    ))


def test_symmetric_plant_contains_no_rotor_angle_information() -> None:
    for motor in ("AIR56", "AL31", "AO2"):
        result = phase_zero_check(motor)
        assert result["max_electrical_output_delta"] == 0.0
        assert result["classification"] == "not represented"


def test_slot_channel_is_switchable_and_only_periodic() -> None:
    off = SlotSaliency(strength=0.0)
    on = SlotSaliency(strength=0.25)
    assert off.current_increment(0.2, 80, 20, 5e-6) == pytest.approx(off.current_increment(1.1, 80, 20, 5e-6))
    assert on.current_increment(0.2, 80, 20, 5e-6) == pytest.approx(on.current_increment(0.2 + on.period_rad, 80, 20, 5e-6))
    assert on.current_increment(0.2, 80, 20, 5e-6) != pytest.approx(on.current_increment(0.3, 80, 20, 5e-6))


def test_observer_api_cannot_receive_truth_and_identical_measurements_give_identical_posteriors() -> None:
    signature = inspect.signature(ActiveBeliefRotorObserver.update_observations)
    assert not ({"theta_m", "omega_m", "true_angle", "true_speed", "measured_state"} & set(signature.parameters))
    saliency = SlotSaliency(strength=0.2)
    a = ActiveBeliefRotorObserver(saliency, noise_std_a=0.002)
    b = ActiveBeliefRotorObserver(saliency, noise_std_a=0.002)
    observation = (0.006, 0.002)
    a.update(observation, (80, 0), 5e-6, 100e-6)
    b.update(observation, (80, 0), 5e-6, 100e-6)
    np.testing.assert_array_equal(a.weights, b.weights)


def test_posterior_normalization_zero_information_and_noise_uncertainty() -> None:
    off = ActiveBeliefRotorObserver(SlotSaliency(strength=0.0), noise_std_a=0.002)
    prior = off.angle_marginal().copy()
    off.update((0.007, 0.001), (80, 0), 5e-6, 100e-6)
    assert np.sum(off.weights) == pytest.approx(1.0)
    np.testing.assert_allclose(off.angle_marginal(), prior, atol=1e-14)

    plant = SlotSaliency(strength=0.25)
    measurement = plant.current_increment(0.03, 80, 0, 5e-6)
    precise = ActiveBeliefRotorObserver(plant, noise_std_a=0.0005)
    noisy = ActiveBeliefRotorObserver(plant, noise_std_a=0.01)
    precise.update(measurement, (80, 0), 5e-6, 100e-6)
    noisy.update(measurement, (80, 0), 5e-6, 100e-6)
    assert precise.entropy() < noisy.entropy()
    assert precise.credible_set(0.95)[1] <= noisy.credible_set(0.95)[1]


def test_candidate_equivalence_legality_and_gateway_current_limit() -> None:
    inv = TwoLevelInverterParams(Vdc=80)
    cfg = CodedPwmConfig()
    ref = make_sequence(4, None, inv, cfg)
    for probe in range(1, 7):
        candidate = make_sequence(4, probe, inv, cfg)
        assert average_voltage(candidate, inv) == pytest.approx(average_voltage(ref, inv), abs=1e-10)
        assert legal_sequence(candidate, reference=ref, inverter=inv, gateway=_gateway(),
                              measured_i_abs=0.1, predicted_i_abs=0.2)
        previous = 0
        for vector, dwell in candidate.slots:
            assert dwell >= inv.min_pulse_s
            assert not has_shoot_through(transition_waveform(previous, vector))
            previous = vector
    assert not legal_sequence(make_sequence(4, 1, inv, cfg), reference=ref, inverter=inv,
                              gateway=_gateway(), measured_i_abs=0.1, predicted_i_abs=5.0)
    unequal = CodedSequence(((4, 10e-6), (0, 80e-6), (1, 10e-6)), 1)
    assert not legal_sequence(unequal, reference=ref, inverter=inv, gateway=_gateway(),
                              measured_i_abs=0.1, predicted_i_abs=0.2)
    overvoltage = TwoLevelInverterParams(Vdc=120)
    assert not legal_sequence(make_sequence(4, 1, overvoltage, cfg),
                              reference=make_sequence(4, None, overvoltage, cfg),
                              inverter=overvoltage, gateway=_gateway(),
                              measured_i_abs=0.1, predicted_i_abs=0.2)


def test_no_saliency_cannot_induce_active_probe_choice_or_angle_gain() -> None:
    inv = TwoLevelInverterParams(Vdc=80)
    observer = ActiveBeliefRotorObserver(SlotSaliency(strength=0.0), noise_std_a=0.002)
    selected = choose_sequence(base_vector=4, observer=observer, inverter=inv, gateway=_gateway(),
                               measured_i_abs=0.1, cfg=CodedPwmConfig())
    assert selected.probe_vector is None
    stimulus = exogenous(Case(steps=20, strength=0.0))
    errors = [run_trial(Case(steps=20, strength=0.0), method, stimulus=stimulus)["angle_p95_deg"] for method in METHODS]
    assert max(errors) - min(errors) < 1e-12


def test_reproducibility_and_common_exogenous_stream() -> None:
    case = Case(seed=77, steps=30)
    stream = exogenous(case)
    assert exogenous(case)["theta0"] == stream["theta0"]
    np.testing.assert_array_equal(exogenous(case)["probe_noise"], stream["probe_noise"])
    first = run_trial(case, "B2_fixed_probe", stimulus=stream)
    second = run_trial(case, "B2_fixed_probe", stimulus=stream)
    for field in ("angle_p95_deg", "torque_ripple_rms_nm", "switches_per_s", "probe_copper_energy_j"):
        assert first[field] == second[field]


def test_credible_set_nominal_coverage_under_uniform_uninformative_prior() -> None:
    observer = ActiveBeliefRotorObserver(SlotSaliency(strength=0.0), noise_std_a=0.002)
    chosen, width = observer.credible_set(0.95)
    coverage = sum(int(k in chosen) for k in range(len(observer.angles))) / len(observer.angles)
    assert coverage >= 0.95
    assert width / observer.period_rad == pytest.approx(coverage)


def test_lag_smoother_normalizes_and_only_revises_past() -> None:
    observer = ActiveBeliefRotorObserver(SlotSaliency(strength=0.2), noise_std_a=0.002)
    for k in range(10):
        observer.update((0.005 + k * 1e-5, 0.001), (80, 0), 5e-6, 100e-6)
    assert np.sum(observer.angle_marginal(4)) == pytest.approx(1.0)
    assert 0 <= observer.map_angle(4) < observer.period_rad


def test_trial_never_exceeds_gateway_limits_in_nominal_smoke() -> None:
    result = run_trial(Case(steps=40), "B3_info_renyi2")
    assert result["gateway_rejections"] == 0
    assert result["current_peak_a"] < _gateway().limits.i_soft_a
    assert math.isfinite(result["angle_p95_deg"])


def test_all_arms_share_exogenous_digest(tmp_path) -> None:
    run_suite("quick", tmp_path, seeds=1, steps=8, seed_offset=9191)
    with (tmp_path / "trials.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_case: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        key = (row["motor"], row["scenario"], row["strength"])
        by_case.setdefault(key, set()).add(row["stimulus_digest"])
    assert all(len(digests) == 1 for digests in by_case.values())

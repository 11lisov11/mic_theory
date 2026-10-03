from __future__ import annotations

import csv
import inspect
import math

import numpy as np
import pytest

from control.active_belief_rotor_observer import ActiveBeliefRotorObserver
from control.info_optimal_coded_pwm import (
    CodedPwmConfig, CodedSequence, average_voltage, choose_sequence, information_score,
    legal_sequence, make_sequence,
)
from models.rotor_slot_saliency import SlotSaliency
from models.two_level_inverter import TwoLevelInverterParams, alpha_beta_voltage
from safety.ai_pwm_gateway import AIPwmSafetyGateway, GatewayLimits, has_shoot_through, transition_waveform
from tools.run_info_optimal_coded_pwm_study import Case, METHODS, exogenous, phase_zero_check, run_suite, run_trial
from tools.compare_coded_pwm_observers import compare


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


@pytest.mark.parametrize("noise_ratio", [1.0, 4.0])
def test_fisher_quadratic_matches_direct_state_sum(noise_ratio: float) -> None:
    saliency = SlotSaliency(strength=0.15)
    observer = ActiveBeliefRotorObserver(
        saliency, noise_std_a=0.002, noise_std_axes_a=(0.002, 0.002 * noise_ratio),
        speed_hypotheses=(-1.0, 0.0, 1.0),
    )
    rng = np.random.default_rng(41)
    posterior = rng.random(observer.weights.shape)
    observer.weights = posterior / posterior.sum()
    dwell = 5e-6
    for voltage in ((0.0, 0.0), (80.0, 0.0), (0.0, 80.0), (47.0, -21.0), (-47.0, 21.0)):
        va, vb = voltage
        phase = saliency.harmonic * observer.angles
        factor = dwell * saliency.strength * saliency.harmonic / saliency.differential_inductance_h
        deriv_a = factor * (-np.sin(phase) * va + np.cos(phase) * vb)
        deriv_b = factor * (np.cos(phase) * va + np.sin(phase) * vb)
        direct = np.sum(observer.weights * observer.gains[None, None, :] ** 2 *
                        ((deriv_a / 0.002)[:, None, None] ** 2 +
                         (deriv_b / (0.002 * noise_ratio))[:, None, None] ** 2))
        direct *= observer.period_rad**2 / 12.0
        actual = information_score(observer, voltage, dwell, "fisher")
        assert actual == pytest.approx(float(direct), rel=1e-12, abs=1e-12)
        opposite = information_score(observer, (-va, -vb), dwell, "fisher")
        assert opposite == pytest.approx(actual, rel=1e-12, abs=1e-12)


def test_noise_miscalibration_is_separate_from_actual_stimulus() -> None:
    case = Case(seed=17777, steps=8, strength=0.15, measurement_noise_beta_ratio=4.0)
    stimulus = exogenous(case)
    matched = run_trial(case, "B2_fixed_probe", stimulus=stimulus,
                        observer_profile="robust", assumed_noise_beta_ratio=4.0)
    default = run_trial(case, "B2_fixed_probe", stimulus=stimulus, observer_profile="robust")
    mismatched = run_trial(case, "B2_fixed_probe", stimulus=stimulus,
                           observer_profile="robust", assumed_noise_beta_ratio=1.0)
    assert matched["angle_p95_deg"] == pytest.approx(default["angle_p95_deg"])
    assert matched["torque_ripple_rms_nm"] == pytest.approx(mismatched["torque_ripple_rms_nm"])
    assert matched["switches_per_s"] == mismatched["switches_per_s"]
    with pytest.raises(ValueError):
        run_trial(case, "B2_fixed_probe", assumed_noise_beta_ratio=0.0)


def test_noise_likelihood_mixture_normalizes_and_singleton_matches_existing_filter() -> None:
    saliency = SlotSaliency(strength=0.15)
    kwargs = {"noise_std_a": 0.002, "speed_hypotheses": (-1.0, 0.0, 1.0)}
    standard = ActiveBeliefRotorObserver(saliency, noise_std_axes_a=(0.002, 0.008), **kwargs)
    singleton = ActiveBeliefRotorObserver(saliency, noise_beta_ratio_candidates=(4.0,), **kwargs)
    forward = ActiveBeliefRotorObserver(saliency, noise_beta_ratio_candidates=(1.0, 4.0), **kwargs)
    reversed_order = ActiveBeliefRotorObserver(saliency, noise_beta_ratio_candidates=(4.0, 1.0), **kwargs)
    observations = [((0.004, -0.005), (80.0, 0.0), 10e-6),
                    ((0.003, 0.002), (40.0, 60.0), 5e-6)]
    for observer in (standard, singleton, forward, reversed_order):
        observer.update_observations(observations, 100e-6)
        assert np.sum(observer.weights) == pytest.approx(1.0)
    np.testing.assert_allclose(standard.weights, singleton.weights, atol=1e-14)
    np.testing.assert_allclose(forward.weights, reversed_order.weights, atol=1e-14)
    assert not np.allclose(forward.weights, standard.weights)


@pytest.mark.parametrize("candidates", [(), (0.0,), (math.nan,), (math.inf,)])
def test_invalid_noise_mixture_hypotheses_are_rejected(candidates: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        ActiveBeliefRotorObserver(SlotSaliency(strength=0.15), noise_std_a=0.002,
                                  noise_beta_ratio_candidates=candidates)


def test_likelihood_tempering_reduces_one_update_concentration() -> None:
    saliency = SlotSaliency(strength=0.15)
    standard = ActiveBeliefRotorObserver(saliency, noise_std_a=0.002,
                                         noise_beta_ratio_candidates=(1.0, 4.0))
    tempered = ActiveBeliefRotorObserver(saliency, noise_std_a=0.002,
                                         noise_beta_ratio_candidates=(1.0, 4.0),
                                         likelihood_temperature=4.0)
    observations = [((0.004, -0.005), (80.0, 0.0), 10e-6)]
    standard.update_observations(observations, 100e-6)
    tempered.update_observations(observations, 100e-6)
    assert np.max(tempered.weights) < np.max(standard.weights)
    assert np.sum(tempered.weights) == pytest.approx(1.0)


@pytest.mark.parametrize("temperature", [0.0, 0.9, math.nan, math.inf])
def test_invalid_likelihood_temperatures_are_rejected(temperature: float) -> None:
    with pytest.raises(ValueError):
        ActiveBeliefRotorObserver(SlotSaliency(strength=0.15), noise_std_a=0.002,
                                  likelihood_temperature=temperature)


@pytest.mark.parametrize("invalid", [0.0, -1.0, math.nan, math.inf])
def test_nonfinite_or_nonpositive_noise_parameters_are_rejected(invalid: float) -> None:
    with pytest.raises(ValueError):
        run_trial(Case(steps=6, noise_std_a=invalid), "B2_fixed_probe")
    with pytest.raises(ValueError):
        run_trial(Case(steps=6, measurement_noise_beta_ratio=invalid), "B2_fixed_probe")


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


def test_robust_transport_preserves_mass_and_has_exact_adjoint() -> None:
    observer = ActiveBeliefRotorObserver(
        SlotSaliency(strength=0.2), noise_std_a=0.003,
        speed_hypotheses=(-2.0, -1.0, 0.0, 1.0, 2.0),
        angle_diffusion=0.003, speed_diffusion=0.001, gain_diffusion=0.001,
    )
    rng = np.random.default_rng(57)
    a = rng.random(observer.weights.shape)
    b = rng.random(observer.weights.shape)
    forward = observer._transport(a, 100e-6)
    backward = observer._transport(b, 100e-6, backward=True)
    assert np.sum(forward) == pytest.approx(np.sum(a))
    assert np.vdot(forward, b) == pytest.approx(np.vdot(a, backward), rel=1e-13)
    for k in range(12):
        observer.update((0.005 + k * 1e-5, 0.001), (80, 0), 5e-6, 100e-6)
    assert np.sum(observer.angle_marginal(4)) == pytest.approx(1.0)


def test_vectorized_transport_matches_per_speed_reference() -> None:
    observer = ActiveBeliefRotorObserver(
        SlotSaliency(strength=0.2), noise_std_a=0.003,
        speed_hypotheses=(-20.0, -1.5, 0.0, 0.25, 5.0),
    )
    values = np.random.default_rng(73).random(observer.weights.shape)
    for backward in (False, True):
        reference = np.empty_like(values)
        for j, speed in enumerate(observer.speeds):
            shift = speed * 100e-6 / (observer.period_rad / len(observer.angles))
            whole = math.floor(shift)
            frac = shift - whole
            direction = -1 if backward else 1
            reference[:, j, :] = ((1.0 - frac) * np.roll(values[:, j, :], direction * whole, axis=0)
                                  + frac * np.roll(values[:, j, :], direction * (whole + 1), axis=0))
        np.testing.assert_allclose(observer._transport(values, 100e-6, backward=backward), reference,
                                   rtol=0.0, atol=1e-15)


def test_robust_profile_does_not_change_legacy_trial() -> None:
    case = Case(seed=101, steps=40, speed_fraction=0.01)
    legacy = run_trial(case, "B2_fixed_probe")
    explicit_legacy = run_trial(case, "B2_fixed_probe", observer_profile="legacy")
    dense = run_trial(case, "B2_fixed_probe", observer_profile="dense")
    robust = run_trial(case, "B2_fixed_probe", observer_profile="robust")
    assert legacy["angle_p95_deg"] == explicit_legacy["angle_p95_deg"]
    assert dense["hypotheses"] == robust["hypotheses"]
    assert robust["hypotheses"] > legacy["hypotheses"]
    assert math.isfinite(robust["hpd95_coverage"])


def test_robust_observer_cannot_create_angle_information_without_saliency() -> None:
    observer = ActiveBeliefRotorObserver(
        SlotSaliency(strength=0.0), noise_std_a=0.002,
        speed_hypotheses=(-2.0, -1.0, 0.0, 1.0, 2.0),
        angle_diffusion=0.003, speed_diffusion=0.001, gain_diffusion=0.001,
    )
    for _ in range(20):
        observer.update((0.007, 0.001), (80, 0), 5e-6, 100e-6)
    np.testing.assert_allclose(observer.angle_marginal(), np.full(48, 1.0 / 48.0), atol=1e-12)


def test_probe_orientation_is_symmetric_only_with_isotropic_noise() -> None:
    saliency = SlotSaliency(strength=0.15)
    weights = np.random.default_rng(123).dirichlet(np.ones(48 * 7)).reshape(48, 7, 1)
    inverter = TwoLevelInverterParams(Vdc=80)
    for noise_axes, expected_equal in (((0.002, 0.002), True), ((0.002, 0.008), False)):
        observer = ActiveBeliefRotorObserver(
            saliency, noise_std_a=0.002, noise_std_axes_a=noise_axes,
            gain_hypotheses=(1.0,),
        )
        observer.weights = weights.copy()
        for method in ("separation", "renyi2", "fisher"):
            scores = [information_score(observer, alpha_beta_voltage(vector, inverter), 5e-6, method)
                      for vector in range(1, 7)]
            if expected_equal:
                assert max(scores) - min(scores) < 1e-12
            else:
                assert max(scores) - min(scores) > 1e-5


def test_bayesian_risk_score_is_finite_and_sign_symmetric() -> None:
    observer = ActiveBeliefRotorObserver(
        SlotSaliency(strength=0.15), noise_std_a=0.002,
        noise_std_axes_a=(0.002, 0.008), gain_hypotheses=(1.0,),
    )
    observer.weights = np.random.default_rng(321).dirichlet(np.ones(observer.weights.size)).reshape(observer.weights.shape)
    inverter = TwoLevelInverterParams(Vdc=80)
    scores = [information_score(observer, alpha_beta_voltage(vector, inverter), 5e-6, "bayes_risk")
              for vector in range(1, 7)]
    assert all(math.isfinite(score) for score in scores)
    assert scores[0] == pytest.approx(scores[5], abs=1e-12)
    assert scores[1] == pytest.approx(scores[4], abs=1e-12)
    assert scores[2] == pytest.approx(scores[3], abs=1e-12)


def test_anisotropic_trial_accepts_each_fixed_probe_without_truth_leak() -> None:
    case = Case(seed=99, steps=20, measurement_noise_beta_ratio=4.0)
    for vector in range(1, 7):
        result = run_trial(case, "B2_fixed_probe", observer_profile="robust",
                           fixed_probe_vector=vector)
        assert math.isfinite(result["angle_p95_deg"])
        assert result["fixed_probe_vector"] == vector
    with pytest.raises(ValueError, match="fixed probe"):
        run_trial(case, "B3_info_separation", fixed_probe_vector=1)
    oracle = run_trial(case, "B4_oracle_fisher", observer_profile="robust")
    assert oracle["candidate_sequences"] == 6
    assert oracle["gateway_rejections"] == 0


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


def test_profile_comparison_rejects_unmatched_stimuli(tmp_path) -> None:
    case = Case(seed=42, steps=8)
    row = run_trial(case, "B2_fixed_probe")
    row.pop("trace_theta_rad")
    row.pop("trace_estimate_rad")
    row["stimulus_digest"] = "original"
    legacy = tmp_path / "legacy.csv"
    robust = tmp_path / "robust.csv"
    for path, digest in ((legacy, "original"), (robust, "different")):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow({**row, "stimulus_digest": digest})
    with pytest.raises(ValueError, match="exogenous stimuli differ"):
        compare(legacy, robust)

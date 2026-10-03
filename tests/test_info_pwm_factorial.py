from __future__ import annotations

import pytest

from tools.audit_info_pwm_factorial import (
    METHODS, MISMATCHES, NONIDEAL, NOISE_RATIOS, OPERATING_POINTS,
    STRENGTHS, cases, run_audit, run_static_audit,
)
from tools.run_info_optimal_coded_pwm_study import MOTORS


def test_full_matrix_is_cartesian_without_duplicate_cases() -> None:
    matrix = cases(seed=15000, steps=6)
    expected = (len(MOTORS) * len(OPERATING_POINTS) * len(STRENGTHS)
                * len(NONIDEAL) * len(MISMATCHES) * len(NOISE_RATIOS))
    assert len(matrix) == expected == 5940
    keys = {(case.motor, case.scenario, case.speed_fraction, case.strength,
             case.nonideal, case.mismatch, case.measurement_noise_beta_ratio) for case in matrix}
    assert len(keys) == expected
    assert len(METHODS) == 3


def test_quick_audit_uses_common_seed_and_has_no_gateway_faults() -> None:
    report = run_audit(seed=16000, steps=6, quick=True)
    assert report["unique_cases"] == 32
    assert report["processed_trials"] == 96
    assert report["failure_count"] == 0
    assert report["gate_rejections"] == 0
    assert report["accuracy_claim"] is False
    assert report["hardware_claim"] is False


def test_shorter_than_fixed_probe_cycle_is_rejected() -> None:
    with pytest.raises(ValueError):
        cases(seed=1, steps=5)


def test_static_exhaustion_covers_vectors_and_rotor_phase_grid() -> None:
    report = run_static_audit()
    assert report["checked_sequences"] == 7 * 8 * 6 * 2
    assert report["checked_phase_responses"] == 3 * 3 * 48 * 28 * 6
    assert report["checked_fisher_scores"] == 3 * 2 * 48 * 6
    assert report["failure_count"] == 0

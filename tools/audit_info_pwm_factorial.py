"""Exhaustive finite-grid software audit for the coded-PWM host experiment.

The short trials detect exceptions, invalid outputs and gate failures. They do
not measure steady-state accuracy or establish hardware readiness.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_info_optimal_coded_pwm_study import Case, MOTORS, exogenous, run_trial
from control.active_belief_rotor_observer import ActiveBeliefRotorObserver
from control.info_optimal_coded_pwm import CodedPwmConfig, average_voltage, information_score, legal_sequence, make_sequence
from models.rotor_slot_saliency import SlotSaliency
from models.two_level_inverter import TwoLevelInverterParams, alpha_beta_voltage
from safety.ai_pwm_gateway import AIPwmSafetyGateway, GatewayLimits


OPERATING_POINTS = (
    ("standstill", 0.0), ("constant", -0.50), ("constant", -0.01),
    ("constant", 0.001), ("constant", 0.005), ("constant", 0.01),
    ("constant", 0.02), ("constant", 0.05), ("constant", 0.10),
    ("constant", 0.50), ("reversal", 0.02), ("ramp", 0.02),
    ("startup", 0.01), ("locked", 0.0), ("torque_step", 0.01),
)
STRENGTHS = (0.0, 0.05, 0.15)
NONIDEAL = (False, True)
MISMATCHES = (
    "nominal", "rs_minus30", "rs_plus30", "rs_plus60",
    "rr_minus30", "rr_plus30", "rr_plus60", "lm_minus20",
    "lm_plus20", "combined", "drift",
)
NOISE_RATIOS = (1.0, 4.0)
METHODS = ("B2_fixed_probe", "B3_info_separation", "B3_info_fisher")


def run_static_audit() -> dict:
    failures: list[dict] = []
    failure_count = 0
    checked_sequences = 0
    checked_phase_responses = 0
    checked_fisher_scores = 0
    cfg = CodedPwmConfig()
    for vdc, base, probe, predicted_current in itertools.product(
        (39.9, 40.0, 40.1, 80.0, 109.9, 110.0, 110.1), range(8), range(1, 7), (0.2, 3.5)
    ):
        inverter = TwoLevelInverterParams(Vdc=vdc)
        reference = make_sequence(base, None, inverter, cfg)
        candidate = make_sequence(base, probe, inverter, cfg)
        gateway = AIPwmSafetyGateway(GatewayLimits(
            vdc_min_v=40.0, vdc_max_v=110.0, i_soft_a=3.0, i_trip_a=4.0,
            max_switch_events_per_window=48, switch_window_steps=8,
        ))
        ra, rb = average_voltage(reference, inverter)
        ca, cb = average_voltage(candidate, inverter)
        equal_mean = math.hypot(ra - ca, rb - cb) <= cfg.mean_tolerance_v
        legal = legal_sequence(
            candidate, reference=reference, inverter=inverter, gateway=gateway,
            measured_i_abs=0.1, predicted_i_abs=predicted_current,
        )
        expected_legal = 40.0 < vdc < 110.0 and predicted_current < 3.0
        checked_sequences += 1
        if not equal_mean or legal != expected_legal:
            failure_count += 1
            if len(failures) < 100:
                failures.append({"kind": "sequence", "vdc": vdc, "base": base, "probe": probe,
                                 "predicted_current": predicted_current, "equal_mean": equal_mean,
                                 "legal": legal, "expected_legal": expected_legal})

    for strength, gain, phase_bin, period_index, probe in itertools.product(
        STRENGTHS, (0.8, 1.0, 1.2), range(48), range(28), range(1, 7)
    ):
        channel = SlotSaliency(harmonic=28, strength=strength)
        theta = (period_index + phase_bin / 48.0) * channel.period_rad
        voltage = alpha_beta_voltage(probe, TwoLevelInverterParams(Vdc=80.0))
        response = channel.current_increment(theta, *voltage, 5e-6, gain=gain)
        phase_zero = channel.current_increment(phase_bin * channel.period_rad / 48.0,
                                               *voltage, 5e-6, gain=gain)
        opposite = channel.current_increment(theta, -voltage[0], -voltage[1], 5e-6, gain=gain)
        checked_phase_responses += 1
        if (any(abs(a - b) > 1e-12 for a, b in zip(response, phase_zero))
                or any(abs(a + b) > 1e-12 for a, b in zip(response, opposite))):
            failure_count += 1
            if len(failures) < 100:
                failures.append({"kind": "slot_periodicity", "strength": strength, "gain": gain,
                                 "phase_bin": phase_bin, "period_index": period_index, "probe": probe})

    for strength, noise_ratio in itertools.product(STRENGTHS, NOISE_RATIOS):
        channel = SlotSaliency(harmonic=28, strength=strength)
        observer = ActiveBeliefRotorObserver(channel, noise_std_a=0.002,
                                             noise_std_axes_a=(0.002, 0.002 * noise_ratio))
        for angle_bin, probe in itertools.product(range(48), range(1, 7)):
            observer.weights.fill(0.0)
            observer.weights[angle_bin, 3, 1] = 1.0
            voltage = alpha_beta_voltage(probe, TwoLevelInverterParams(Vdc=80.0))
            measured_score = information_score(observer, voltage, 5e-6, "fisher")
            phase = channel.harmonic * observer.angles[angle_bin]
            factor = 5e-6 * strength * channel.harmonic / channel.differential_inductance_h
            da = factor * (-math.sin(phase) * voltage[0] + math.cos(phase) * voltage[1])
            db = factor * (math.cos(phase) * voltage[0] + math.sin(phase) * voltage[1])
            direct_score = (da**2 / 0.002**2 + db**2 / (0.002 * noise_ratio)**2) * observer.period_rad**2 / 12.0
            checked_fisher_scores += 1
            if not math.isclose(measured_score, direct_score, rel_tol=1e-12, abs_tol=1e-12):
                failure_count += 1
                if len(failures) < 100:
                    failures.append({"kind": "fisher", "strength": strength,
                                     "noise_ratio": noise_ratio, "angle_bin": angle_bin,
                                     "probe": probe, "actual": measured_score, "direct": direct_score})

    return {"status": "finite_grid_static_audit", "checked_sequences": checked_sequences,
            "checked_phase_responses": checked_phase_responses,
            "checked_fisher_scores": checked_fisher_scores,
            "failure_count": failure_count, "failures_first_100": failures,
            "hardware_claim": False}


def cases(*, seed: int, steps: int, quick: bool = False) -> list[Case]:
    if steps < 6:
        raise ValueError("at least six PWM periods are needed to cycle every fixed probe")
    if quick:
        # CI smoke: every method and both sensor-noise levels, not full coverage.
        points = itertools.product(("AIR56",), OPERATING_POINTS[:2], (0.0, 0.15),
                                   NONIDEAL, ("nominal", "combined"), NOISE_RATIOS)
    else:
        points = itertools.product(MOTORS, OPERATING_POINTS, STRENGTHS,
                                   NONIDEAL, MISMATCHES, NOISE_RATIOS)
    return [Case(motor=motor, scenario=scenario, speed_fraction=speed,
                 strength=strength, nonideal=nonideal, mismatch=mismatch,
                 measurement_noise_beta_ratio=noise_ratio, seed=seed, steps=steps)
            for motor, (scenario, speed), strength, nonideal, mismatch, noise_ratio in points]


def run_audit(*, seed: int = 15000, steps: int = 6, quick: bool = False) -> dict:
    matrix = cases(seed=seed, steps=steps, quick=quick)
    failures: list[dict] = []
    failure_count = 0
    gate_rejections = 0
    no_probe_trials = 0
    processed = 0
    started = perf_counter()
    for case in matrix:
        stimulus = exogenous(case)
        for method in METHODS:
            processed += 1
            try:
                row = run_trial(case, method, stimulus=stimulus, observer_profile="robust")
                gate_rejections += int(row["gateway_rejections"])
                no_probe_trials += int(float(row["probe_fraction"]) == 0.0)
                period_half = float(row["period_deg"]) / 2.0
                if (not math.isfinite(float(row["angle_p95_deg"]))
                        or not 0.0 <= float(row["angle_p95_deg"]) <= period_half + 1e-9
                        or int(row["gateway_rejections"]) != 0):
                    raise AssertionError("nonfinite/invalid angle P95 or rejected gateway request")
            except Exception as exc:
                failure_count += 1
                if len(failures) < 100:
                    failures.append({"case": asdict(case), "method": method,
                                     "error": f"{type(exc).__name__}: {exc}"})
    return {
        "status": "host_software_audit_only",
        "seed": seed,
        "steps_per_trial": steps,
        "quick": quick,
        "factor_levels": {
            "motor": list(MOTORS), "operating_point": list(OPERATING_POINTS),
            "strength": list(STRENGTHS), "nonideal": list(NONIDEAL),
            "mismatch": list(MISMATCHES), "noise_ratio": list(NOISE_RATIOS),
            "method": list(METHODS),
        },
        "unique_cases": len(matrix),
        "expected_trials": len(matrix) * len(METHODS),
        "processed_trials": processed,
        "failure_count": failure_count,
        "gate_rejections": gate_rejections,
        "no_probe_trials": no_probe_trials,
        "failures_first_100": failures,
        "elapsed_wall_s": perf_counter() - started,
        "accuracy_claim": False,
        "hardware_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run_static_audit() if args.static_only else run_audit(seed=args.seed, steps=args.steps, quick=args.quick)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = (("status", "checked_sequences", "checked_phase_responses", "checked_fisher_scores", "failure_count")
              if args.static_only else
              ("status", "unique_cases", "processed_trials", "failure_count",
               "gate_rejections", "no_probe_trials", "elapsed_wall_s"))
    print(json.dumps({key: result[key] for key in fields}, indent=2))
    if result["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

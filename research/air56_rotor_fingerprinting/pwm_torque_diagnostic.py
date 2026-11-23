"""Replay recorded commands and isolate the one-period field and quadrature errors.

This is an offline common-state diagnostic. Its state resets are never used in control.
"""

# ruff: noqa: E402
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fem.air56.dwell_response import RATES, weights
from fem.air56.polar_start import resolving_pulses
from fem.air56.servo_pwm import segments
from tools.run_air56_terminal_servo import Machine, advance_plant, cold_voltage


def locked_period(model, initial, stages, interval):
    """Split the torque difference while both field solves see identical fixed geometry."""
    angle = initial[-3]
    states = {n: initial.copy() for n in (4, 8)}
    means = np.zeros(3)
    for voltage, dt in stages:
        samples = {}
        for n in states:
            x = states[n]
            endpoints = [x[-6]]
            for _ in range(n):
                x, _, _ = model.step(x, voltage, dt / n)
                x[-3:-1] = (angle, 0.0)
                endpoints.append(x[-6])
            states[n] = x
            samples[n] = np.asarray(endpoints)
        for j, values in enumerate((samples[4], samples[8][::2], samples[8])):
            means[j] += dt / interval * np.mean((values[:-1] + values[1:]) / 2)
    coarse, refined_at_coarse_times, refined = means
    return dict(
        coarse_nm=coarse,
        refined_nm=refined,
        total_nm=refined - coarse,
        field_nm=refined_at_coarse_times - coarse,
        quadrature_nm=refined - refined_at_coarse_times,
        current_diff_a=np.linalg.norm(states[8][-5:-3] - states[4][-5:-3]),
    )


def replay(path, out, locked=False):
    with np.load(path) as record:
        args = json.loads(str(record["settings"]))
        trace = record["trace"]
    machine = Machine(args["level"], args["modes"], args["skew"])
    comparison = Machine(args["level"], args["modes"], args["skew"])
    comparison.trapezoid = machine.trapezoid = True
    state = machine.equilibrium(np.deg2rad(args["initial"]))
    load = args["load"]
    pulses = [(cold_voltage(k), 0.00025) for k in range(12)]
    pulses.extend((u * (args["start_volts"] / 115), h) for u, h in resolving_pulses())
    for u, h in pulses:
        for v, dt in segments(u, h):
            state, _ = advance_plant(machine, state, v, dt, load, args["plant_substeps"])
    out.mkdir(parents=True, exist_ok=True)
    if locked:
        rows = []
        for k, row in enumerate(trace):
            stages = segments(row[11:13], args["h"])
            if k % 11 == 0 or k == len(trace) - 1:
                rows.append(dict(period=k, **locked_period(comparison, state, stages, args["h"])))
            for v, dt in stages:
                state, _ = advance_plant(machine, state, v, dt, load, args["plant_substeps"])
            if abs(state[-3] - row[1]) > 1e-6:
                raise RuntimeError(f"Plant replay diverged at period {k}")
        with (out / (path.stem + "_locked.csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, rows[0])
            writer.writeheader()
            writer.writerows(rows)
        print(path.stem, "verified locked-pose decomposition on", len(rows), "periods", flush=True)
        return
    fields = [
        "period",
        "dwell",
        "seconds",
        "active",
        "rate",
        "torque_nm",
        "reference_nm",
        "error_nm",
        "di_alpha_a",
        "di_beta_a",
    ]
    with (out / (path.stem + "_diagnostic.csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for k, row in enumerate(trace):
            states = {}
            torques = {}
            selected = k % 11 == 0 or k == len(trace) - 1
            stages = segments(row[11:13], args["h"])
            if selected:
                rates = (1, 2, 3, 5, 7, 8)
                states = {rate: state.copy() for rate in rates}
                torques = {rate: 0.0 for rate in rates}
                period_ref = 0.0
                averaged, _, _ = comparison.step(state, row[11:13], args["h"], load=load)
            for j, (v, dt) in enumerate(stages):
                state, reference = advance_plant(machine, state, v, dt, load, args["plant_substeps"])
                if not selected:
                    continue
                period_ref += reference * dt / args["h"]
                values = {}

                def emit(label, torque, current, reference, truth, dwell, interval, active):
                    error = current - truth
                    writer.writerow(
                        dict(
                            period=k,
                            dwell=dwell,
                            seconds=interval,
                            active=active,
                            rate=label,
                            torque_nm=torque,
                            reference_nm=reference,
                            error_nm=torque - reference,
                            di_alpha_a=error[0],
                            di_beta_a=error[1],
                        )
                    )

                for rate in rates:
                    states[rate], torque = advance_plant(comparison, states[rate], v, dt, load, rate)
                    values[rate] = torque
                    torques[rate] += torque * dt / args["h"]
                    emit(rate, torque, states[rate][-5:-3], reference, state[-5:-3], j, dt, int(v @ v > 1))
                # Offline best-case interpolation using the known reference step, for diagnosis only.
                blend = weights(1 / args["plant_substeps"])

                def mix(data):
                    return sum(w * data[rate] for w, rate in zip(blend, RATES))

                current = mix({rate: states[rate][-5:-3] for rate in RATES})
                emit("interpolated", mix(values), current, reference, state[-5:-3], j, dt, int(v @ v > 1))
            if selected:
                for rate in rates:
                    emit(
                        rate,
                        torques[rate],
                        states[rate][-5:-3],
                        period_ref,
                        state[-5:-3],
                        -1,
                        args["h"],
                        -1,
                    )
                emit("interpolated", mix(torques), current, period_ref, state[-5:-3], -1, args["h"], -1)
                emit(
                    "mean_voltage_endpoint",
                    averaged[-6],
                    averaged[-5:-3],
                    period_ref,
                    state[-5:-3],
                    -1,
                    args["h"],
                    -1,
                )
            if abs(state[-3] - row[1]) > 1e-6:
                raise RuntimeError(f"Plant replay diverged at period {k}")
    print(path.stem, "verified", len(trace), "original plant periods", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--locked", action="store_true")
    args = parser.parse_args()
    replay(args.trace, args.out, args.locked)

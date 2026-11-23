"""Build and blind-test a terminal-current lookup for an initially loaded free shaft."""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.run_air56_terminal_servo import Machine, cold_voltage, resolving_pulses, voltage_stages


def burst(model, angle, load, subdivisions=1):
    state = model.equilibrium(np.deg2rad(angle))
    currents = []
    probes = [(cold_voltage(k), 0.00025) for k in range(12)]
    probes += [(v * (35 / 115), h) for v, h in resolving_pulses()]
    for command, interval in probes:
        for voltage, h in voltage_stages(command, interval, True):
            for _ in range(subdivisions):
                state, _, _ = model.step(state, voltage, h / subdivisions, load=load)
            currents.extend(state[-5:-3])
    return np.r_[currents, np.rad2deg(state[-3]) - angle, state[-2]]


def angles_for(kind):
    if kind in ("mid", "mid_fine"):
        return 1.25 + np.arange(144) * 2.5
    if kind in ("cal", "cal_fine"):
        return np.arange(144) * 2.5
    # One independent, non-grid angle in each 15-degree interval.
    seed = 38623 if kind == "deep_test" else (91478 if kind.startswith("blind_") else 27111)
    random = np.random.default_rng(seed)
    return 15 * np.arange(24) + random.uniform(1, 14, 24)


def build_part(task):
    out, kind, skew, part = task
    path = out / f"map_{kind}_{skew}_{part}.npz"
    angles = angles_for(kind)[part::8]
    if path.is_file():
        with np.load(path) as archive:
            if np.array_equal(archive["angles"], angles):
                return
    family = kind.removeprefix("blind_")
    fine = family in ("transfer", "cal_fine", "mid_fine", "time_holdout", "deep_test")
    model = Machine(4 if fine else 3, 0, bool(skew))
    model.trapezoid = family in ("time_holdout", "deep_test")
    substeps = 4 if family == "deep_test" else (2 if model.trapezoid else 1)
    magnitude = 0.03 if family in ("cal", "cal_fine", "mid", "mid_fine") else 0.02
    values = [
        [burst(model, angle, load, substeps) for load in (-magnitude, magnitude)] for angle in angles
    ]
    temp = path.with_suffix(".partial.npz")
    np.savez_compressed(temp, angles=angles, values=values)
    temp.replace(path)


def read_records(out, kind, skew):
    angles, values = [], []
    for part in range(8):
        with np.load(out / f"map_{kind}_{skew}_{part}.npz") as data:
            angles.extend(data["angles"])
            values.extend(data["values"])
    order = np.argsort(angles)
    return np.asarray(angles)[order], np.asarray(values)[order]


class CaptureMap:
    def __init__(self, angles, values):
        coefficients = np.stack((values.mean(axis=1), (values[:, 1] - values[:, 0]) / 0.06), axis=1)
        self.angles = angles
        self.coarse = coefficients
        self.interpolate = CubicSpline(
            np.r_[angles, 360.0],
            np.concatenate((coefficients, coefficients[:1])),
            bc_type="periodic",
            axis=0,
        )

    @staticmethod
    def match(coefficients, reading, noise):
        residual = reading - coefficients[:, 0, :-2]
        slope = coefficients[:, 1, :-2]
        load = np.sum(residual * slope, axis=1) / (np.sum(slope * slope, axis=1) + (noise / 0.05) ** 2)
        load = np.clip(load, -0.05, 0.05)
        cost = np.sum((residual - load[:, None] * slope) ** 2, axis=1) + (noise * load / 0.05) ** 2
        return cost, load

    def estimate(self, reading, noise):
        cost, _ = self.match(self.coarse, reading, noise)
        # Refine eight local minima distributed around the full turn.
        minima = np.flatnonzero((cost <= np.roll(cost, 1)) & (cost <= np.roll(cost, -1)))
        candidates = self.angles[minima[np.argsort(cost[minima])[:8]]]
        grid = (candidates[:, None] + np.arange(-2.5, 2.501, 0.1)).ravel() % 360
        coefficients = self.interpolate(grid)
        cost, loads = self.match(coefficients, reading, noise)
        j = np.argmin(cost)
        load = loads[j]
        movement, speed = coefficients[j, 0, -2:] + load * coefficients[j, 1, -2:]
        return (grid[j] + movement) % 360, speed, load, cost[j]


def measure(out, bank=False):
    rows = []
    for skew in (0, 1):
        models = [CaptureMap(*read_records(out, "cal", skew))]
        if bank:
            models.append(CaptureMap(*read_records(out, "cal_fine", skew)))
        for kind in ("test", "transfer"):
            angles, values = read_records(out, kind, skew)
            for i, angle in enumerate(angles):
                for sign, response in zip((-1, 1), values[i]):
                    for seed in range(4):
                        rng = np.random.default_rng([3107, i, sign + 1, skew, seed])
                        perturbation = rng.standard_normal(len(response) - 2)
                        for noise in (0.0003, 0.001, 0.003):
                            reading = response[:-2] + noise * perturbation
                            begin = perf_counter_ns()
                            fits = [model.estimate(reading, noise) for model in models]
                            chosen = int(np.argmin([fit[-1] for fit in fits]))
                            estimate, speed, load, cost = fits[chosen]
                            duration = (perf_counter_ns() - begin) / 1e6
                            expected = angle + response[-2]
                            rows.append(
                                dict(
                                    kind=kind,
                                    skew=skew,
                                    initial_deg=angle,
                                    noise_ma=1000 * noise,
                                    draw=seed,
                                    load_nm=sign * 0.02,
                                    capture_deg=expected,
                                    estimate_deg=estimate,
                                    error_deg=(estimate - expected + 180) % 360 - 180,
                                    true_rpm=response[-1] * 30 / np.pi,
                                    estimated_rpm=speed * 30 / np.pi,
                                    estimated_load_nm=load,
                                    compute_ms=duration,
                                    chosen_level=3 + chosen,
                                    reduced_cost=cost / (noise**2 * (len(reading) - 1)),
                                )
                            )
    with (out / ("bank_scores.csv" if bank else "map_scores.csv")).open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Reproduce calibration, development, blind trials and native reader",
    )
    parser.add_argument("--measure-only", action="store_true")
    parser.add_argument("--build-fine-only", action="store_true")
    parser.add_argument("--build-time-only", action="store_true")
    parser.add_argument(
        "--extra",
        nargs="+",
        choices=["mid", "mid_fine", "blind_test", "blind_transfer", "blind_time_holdout", "deep_test"],
    )
    parser.add_argument("--bank", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.measure_only:
        kinds = ("cal_fine",) if args.build_fine_only else ("cal", "test", "transfer")
        if args.build_time_only:
            kinds = ("time_holdout",)
        if args.extra:
            kinds = args.extra
        if args.full:
            kinds = (
                "cal",
                "cal_fine",
                "mid",
                "mid_fine",
                "test",
                "transfer",
                "time_holdout",
                "blind_test",
                "blind_transfer",
                "blind_time_holdout",
                "deep_test",
            )
        tasks = [(args.out, kind, skew, part) for kind in kinds for part in range(8) for skew in (0, 1)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for _ in pool.map(build_part, tasks):
                pass
    if not (args.build_fine_only or args.build_time_only or args.extra):
        measure(args.out, args.bank)
    if args.full:
        from research.air56_rotor_fingerprinting.preloaded_native import validate

        validate(args.out)
        validate(args.out, filtered=True)

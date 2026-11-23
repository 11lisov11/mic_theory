"""Fresh-angle validation. Freeze three pulse policies before generating the held-out FEM traces."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_air56_terminal_servo import Machine
from tools.score_air56_stock_release import beta_filter
from fem.air56.servo_pwm import segments


def machine(n):
    plant = Machine(5, 0, n > 1, stock=True, axial_slices=n if n > 1 else None)
    plant.compact_lu = n >= 24
    plant.trapezoid = True
    return plant


def period(plant, state, voltage, load, angle):
    peak = 0.0
    for u, h in segments([0.0, voltage], 0.001):
        for _ in range(4):
            state, _, _ = plant.step(state, u, h / 4, load=load)
            peak = max(peak, abs(np.rad2deg(state[-3]) - angle))
    return state, peak


def tails(plant, angle, load, branches):
    outcomes = {}
    for k, v, start, peak in branches:
        x = start.copy()
        for j in range(k, 24):
            x, p = period(plant, x, v if j < k + 2 else 0.0, load, angle)
            peak = max(peak, p)
        outcomes[k, v] = peak
    return outcomes


def independent_block(job):
    n, angle, load, branches = job
    return tails(machine(n), angle, load, branches)


def result_path(job):
    folder, n, angle, load = job
    return folder / f"s{n}_a{angle:g}_m{load:g}.json"


def run(job, branch_workers=1):
    folder, n, angle, load = job
    out = result_path(job)
    if out.exists():
        return
    plant = machine(n)
    x = plant.equilibrium(np.deg2rad(angle), [0.3, 0.0])
    x[-1] = load

    history = [x]
    peaks = [0.0]
    for k in range(24):
        x, p = period(plant, x, 0.0, load, angle)
        history.append(x)
        peaks.append(max(peaks[-1], p))
    t = np.arange(1, 25)
    currents = np.array([x[-4] for x in history[1:]])
    weights = beta_filter()
    counts = {}
    for sd in (0.0003, 0.001, 0.003):
        rng = np.random.default_rng(71209)
        estimates = -62.77561184070174 * ((currents + rng.normal(0, sd, (512, 24))) @ weights.T)
        for rule in ("read", "distance", "speed"):
            # For constant acceleration from rest, speed ~= 2*displacement/time.
            floor = {"read": 0.12, "distance": 0.4, "speed": np.maximum(0.12, 0.030 * t)}[rule]
            edges = abs(estimates) >= np.maximum(
                floor, 3.5 * 62.77561184070174 * sd * np.linalg.norm(weights, axis=1)
            )
            first = np.where(edges.any(axis=1), edges.argmax(axis=1) + 1, 24)
            pos = estimates[np.arange(len(first)), first - 1]
            voltage = np.full(len(pos), 24.0)
            if rule == "speed":
                voltage = 8 * np.clip(np.rint(3 * abs(pos) / (0.030 * first)), 1, 5)
            counts[(sd, rule)] = Counter(
                (int(k), int(-np.sign(e) * v) if k < 24 else 0) for k, e, v in zip(first, pos, voltage)
            )
    branches = [(k, v, history[k], peaks[k]) for k, v in sorted(set().union(*counts.values()))]
    if branch_workers == 1:
        outcomes = tails(plant, angle, load, branches)
    else:
        parts = [(n, angle, load, branches[j::branch_workers]) for j in range(branch_workers)]
        outcomes = {}
        with ProcessPoolExecutor(branch_workers) as pool:
            for partial in pool.map(independent_block, parts):
                outcomes.update(partial)
    scores = []
    for (sd, rule), bins in counts.items():
        maxima = np.concatenate([np.repeat(outcomes[key], count) for key, count in bins.items()])
        scores.append(
            dict(
                noise_ma=1000 * sd,
                rule=rule,
                passed=int(np.sum(maxima <= 1)),
                replays=len(maxima),
                p95_deg=float(np.percentile(maxima, 95)),
            )
        )
    out.write_text(
        json.dumps(
            dict(
                slices=n,
                angle=angle,
                load=load,
                passive_deg=peaks[-1],
                compact_newton_factors=plant.compact_lu,
                scores=scores,
                branches=[dict(t_ms=k, volts=v, peak_deg=p) for (k, v), p in sorted(outcomes.items())],
            ),
            indent=2,
        ),
        encoding="utf8",
    )
    print(out.name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    jobs = [(a.folder, n, angle, load) for n in (1, 24) for angle in (4.6, 14.2) for load in (-0.02, 0.02)]
    pending = [job for job in jobs if not result_path(job).exists()]
    if len(pending) < a.workers:
        for job in pending:
            run(job, branch_workers=a.workers)
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            list(pool.map(run, pending))

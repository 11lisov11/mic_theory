"""Enumerate actual FEM responses to every distinct trigger produced by ADC-noise trials."""

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


def check(job):
    out, n, a, load, gate = job
    path = out / f"ax{n}_a{a:g}_load{load:g}.json"
    reference = path
    if gate:
        path = out / "gate" / path.name
        path.parent.mkdir(exist_ok=True)
    if path.exists():
        return
    m = Machine(5, 0, n > 1, stock=True, axial_slices=n if n > 1 else None)
    m.compact_lu = n >= 24
    m.trapezoid = True
    state = m.equilibrium(np.deg2rad(a), [0.30, 0.0])
    state[-1] = load

    def step(x, volts):
        peak = 0.0
        for v, h in segments(np.array([0.0, volts]), 0.001):
            for _ in range(4):
                x, _, _ = m.step(x, v, h / 4, load=load)
                peak = max(peak, abs(np.rad2deg(x[-3]) - a))
        return x, peak

    states = [state]
    maxima = [0.0]
    for k in range(24):
        state, peak = step(state, 0.0)
        states.append(state)
        maxima.append(max(peak, maxima[-1]))
    filt = beta_filter()
    gain = -62.77561184070174
    currents = np.array([s[-4] for s in states[1:]])
    bins = {}
    for noise in (0.0003, 0.001, 0.003):
        rng = np.random.default_rng(34419)
        observed = currents + rng.normal(0.0, noise, (512, 24))
        estimates = gain * (observed @ filt.T)
        hits = abs(estimates) >= np.maximum(
            0.40 if gate else 0.12, 3.5 * abs(gain) * noise * np.linalg.norm(filt, axis=1)
        )
        times = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 24)
        signs = np.sign(estimates[np.arange(len(times)), times - 1]).astype(int)
        bins[noise] = Counter((int(t), int(s) if t < 24 else 0) for t, s in zip(times, signs))
    cached = {}
    other = reference if gate else out / "gate" / reference.name
    if other.exists():
        previous = json.loads(other.read_text())
        cached = {(r["t_ms"], r["sign"]): r for r in previous["branches"]}
    responses = []
    for t, sgn in sorted(set().union(*bins.values())):
        if (t, sgn) in cached:
            responses.append(cached[t, sgn])
            continue
        state = states[t].copy()
        maximum = maxima[t]
        for k in range(t, 24):
            volts = -sgn * 24.0 if k < t + 2 else 0.0
            state, peak = step(state, volts)
            maximum = max(maximum, peak)
        responses.append(dict(t_ms=t, sign=sgn, max_deg=maximum))
    results = []
    for noise, hist in bins.items():
        values = np.concatenate([np.repeat(r["max_deg"], hist[(r["t_ms"], r["sign"])]) for r in responses])
        assert len(values) == 512
        results.append(
            dict(
                noise_ma=noise * 1000,
                pass_count=int(np.sum(values <= 1.0)),
                trials=len(values),
                median_peak_deg=float(np.median(values)),
                p95_peak_deg=float(np.percentile(values, 95)),
                worst_deg=float(values.max()),
            )
        )
    path.write_text(
        json.dumps(
            dict(
                angle=a,
                slices=n,
                load_nm=load,
                seed=34419,
                gate_deg=0.40 if gate else 0.12,
                passive_deg=maxima[-1],
                results=results,
                branches=responses,
                compact_newton_factors=m.compact_lu,
            ),
            indent=2,
        ),
        encoding="utf8",
    )
    print(path.name, flush=True)


def check_pair(job):
    for gate in (False, True):
        check((*job, gate))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--angles", nargs="+", type=float, default=[3.7, 9.9, 15.4])
    p.add_argument("--slices", nargs="+", type=int, default=[1, 24])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gate", action="store_true")
    p.add_argument("--both", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cases = [(a.out, n, angle, load) for n in a.slices for angle in a.angles for load in (-0.02, 0.02)]
    with ProcessPoolExecutor(a.workers) as pool:
        if a.both:
            list(pool.map(check_pair, cases))
        else:
            list(pool.map(check, [(*case, a.gate) for case in cases]))

"""Calibrate a causal motion deadband on separate noisy FEM-map trajectories."""

import argparse
from pathlib import Path
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.score_air56_pwm_slopes import (
    BLOCK_S,
    PhaseMap,
    phase_features,
    load_bank,
    phase_noise_factor,
    sample_bank,
    ramp_profile,
)
from tools.run_air56_skew_audit import write_csv

WINDOWS = (10, 18, 24, 32)
T = (np.arange(90) + 0.5) * BLOCK_S
ZERO = 18.0
SCENES = ("rest", "forward", "backward", "reverse_plus", "reverse_minus", "start_plus", "start_minus")


def speed(angles, window):
    unwrapped = np.unwrap(2 * angles, axis=-1) / 2
    t = (np.arange(window) - (window - 1) / 2) * BLOCK_S
    return sliding_window_view(unwrapped, window, axis=-1) @ (t / (t @ t) * 60 / (2 * np.pi))


def trajectory(scene):
    if scene.startswith("reverse"):
        final = 0.1 if scene.endswith("plus") else -0.1
        return ramp_profile(T, ZERO - 1.2, 2.4, -final, final)
    if scene.startswith("start"):
        return ramp_profile(T, ZERO, 2.4, 0.0, 0.1 if scene.endswith("plus") else -0.1)
    v = {"rest": 0.0, "forward": 0.1, "backward": -0.1}[scene]
    return np.full_like(T, v), T * v * 2 * np.pi / 60


def observe(scene, seed, count, noise, rho, bank, model, factor):
    rng = np.random.default_rng(seed)
    velocity, delta = trajectory(scene)
    initial = rng.uniform(0, np.pi, (count, 1))
    observations = sample_bank(
        phase_features(bank["phasor"]), bank["angle_rad"], initial + delta, velocity
    )
    white = rng.standard_normal(observations.shape)
    if rho:
        white[:, 1:] *= np.sqrt(1 - rho * rho)
        white = lfilter([1], [1, -rho], white, axis=1)
    return model.acquire(observations + noise * 0.001 * (white @ factor.T))


def confirmed_labels(v, threshold):
    sign = np.where(abs(v) > threshold, np.sign(v), 0)
    # No latch: return unknown immediately if a confirmation expires.
    labels = np.zeros_like(sign, dtype=int)
    labels[:, 1:] = np.where(sign[:, 1:] == sign[:, :-1], sign[:, 1:], 0)
    return labels


def peak_confirmed(v):
    pair = np.where(v[:, 1:] * v[:, :-1] > 0, np.minimum(abs(v[:, 1:]), abs(v[:, :-1])), 0)
    return pair.max(axis=1)


def summarize(angles, window, threshold, scene):
    v = speed(angles, window)
    signs = confirmed_labels(v, threshold)
    clock = T[window - 1 :] + BLOCK_S / 2
    target = np.sign(trajectory(scene)[0][-1])
    row = dict(
        scene=scene,
        n=len(v),
        window_s=window * BLOCK_S,
        deadband_rpm=threshold,
        rest_false_records="",
        correct_final="",
        wrong_final="",
        unknown_final="",
        both_or_started="",
        median_delay_s="",
        p95_delay_s="",
        p95_speed_error_rpm=float(np.percentile(abs(v[:, -1] - 0.1 * target), 95)),
    )
    if scene == "rest":
        row["rest_false_records"] = int(np.any(signs, axis=1).sum())
    else:
        row.update(
            correct_final=int((signs[:, -1] == target).sum()),
            wrong_final=int((signs[:, -1] == -target).sum()),
            unknown_final=int((signs[:, -1] == 0).sum()),
        )
    if scene.startswith(("start", "reverse")):
        prior = (
            np.any(signs[:, clock <= ZERO - 1.2] == -target, axis=1)
            if scene.startswith("reverse")
            else ~np.any(signs[:, clock <= ZERO], axis=1)
        )
        after = (signs == target) & (clock >= ZERO)
        good = prior & after.any(axis=1)
        row["both_or_started"] = int(good.sum())
        if good.any():
            delay = clock[np.argmax(after[good], axis=1)] - ZERO
            row.update(
                median_delay_s=float(np.median(delay)), p95_delay_s=float(np.percentile(delay, 95))
            )
    return row


def run(maps, out):
    factor = phase_noise_factor()
    rows, tuning, direct = [], [], []
    for skew in (0, 1):
        cal, _, _ = load_bank(maps / f"skew_{skew}/mesh_0")
        bank, _, _ = load_bank(maps / f"skew_{skew}/mesh_1")
        model = PhaseMap(cal["angle_rad"], cal["phasor"][0], factor)
        for noise in (0.3, 1.0, 3.0):
            cases = [
                observe(scene, 41000 + j, 256, noise, 0, bank, model, factor)
                for j, scene in enumerate(SCENES[:3])
            ]
            options = []
            for window in WINDOWS:
                peak = peak_confirmed(speed(cases[0], window))
                threshold = float(np.quantile(peak, 0.995, method="higher"))
                fractions = [
                    summarize(a, window, threshold, s)["correct_final"] / len(a)
                    for s, a in zip(SCENES[1:3], cases[1:])
                ]
                options.append((window, threshold, min(fractions)))
                tuning.append(
                    dict(
                        skew=skew,
                        noise_ma=noise,
                        window=window,
                        threshold=threshold,
                        weakest_sign_fraction=min(fractions),
                    )
                )
            chosen = next((c for c in options if c[2] >= 0.95), max(options, key=lambda c: c[2]))
            for rho in [0, 0.6] if noise == 3.0 else [0]:
                for j, scene in enumerate(SCENES):
                    angles = observe(scene, 52000 + j, 384, noise, rho, bank, model, factor)
                    rows.append(
                        dict(
                            skew=skew,
                            noise_ma=noise,
                            block_noise_correlation=rho,
                            **summarize(angles, *chosen[:2], scene),
                        )
                    )
            # Only 14.4 s of independent direct FEM data. Use no prehistory.
            for f in (maps / "direct" / f"skew_{skew}").glob("*.npz"):
                with np.load(f) as data:
                    clean = phase_features(data["phasor"])
                    rng = np.random.default_rng(61199)
                    obs = clean + noise * 0.001 * (rng.standard_normal((384, *clean.shape)) @ factor.T)
                    v = speed(model.acquire(obs), chosen[0])
                    signs = confirmed_labels(v, chosen[1])
                    target = np.sign(data["rpm"][-1])
                    direct.append(
                        dict(
                            skew=skew,
                            noise_ma=noise,
                            path=f.stem,
                            window_s=chosen[0] * BLOCK_S,
                            n=len(obs),
                            correct=int((signs[:, -1] == target).sum()),
                            wrong=int((signs[:, -1] == -target).sum()),
                            unknown=int((signs[:, -1] == 0).sum()),
                            median_rpm=float(np.median(v[:, -1])),
                        )
                    )
            print(skew, noise, chosen, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, records in (("tuning", tuning), ("motion", rows), ("continuous", direct)):
        write_csv(out / f"{name}.csv", records)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.maps, a.out)

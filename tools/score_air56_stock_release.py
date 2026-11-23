"""Distinguish initial micro-speed from subsequent rollback using terminal currents."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

SIGMAS = (0.0003, 0.001, 0.003)
SEED = 20261124


def initial_limits(root):
    result = []
    for f in sorted([*root.glob("L4_*.npz"), *root.glob("L5_*.npz"), *(root / "axial24").glob("L5_*.npz")]):
        with np.load(f) as z:
            y = z["traces"][0]
            bad = np.flatnonzero(y[:, :, 4].max(axis=0) > 1.0)
            n = int(bad[0]) if len(bad) else len(y[0])
            for j in (0, 2):
                distance = float(np.linalg.norm(y[j, :n, :2] - y[1, :n, :2]))
                for sigma in SIGMAS:
                    # Equal priors, two known deterministic histories, white Gaussian ADC errors.
                    result.append(
                        dict(
                            record=f.relative_to(root).as_posix(),
                            skew=int(z["skew_pitch"]),
                            probe=str(z["probe"]),
                            alternative=float(z["speeds"][j]),
                            samples=n,
                            noise_ma=sigma * 1000,
                            current_distance_ua=distance * 1e6,
                            optimal_binary_accuracy=float(ndtr(distance / (2 * sigma))),
                        )
                    )
    return result


def beta_filter(n=24):
    # Matched to early constant acceleration. Unit gain for a t^2-shaped displacement.
    f = np.zeros((n, n))
    for k in range(1, n + 1):
        shape = (np.arange(1, k + 1) / k) ** 2
        f[k - 1, :k] = shape / (shape @ shape)
    return f


def records(path, level, probe, angles):
    found = []
    for p in sorted(path.glob(f"L{level}_*_{probe}.npz")):
        with np.load(p) as z:
            if z["angle_deg"].item() in angles:
                for load, y in zip(z["loads"], z["traces"][:, 0]):
                    found.append((p.stem, float(load), y.copy()))
    return found


def controller_readout(root, probe, axial=None, level=4, locked=False):
    folder = root / "rollback"
    train = records(folder, 3, probe, (1.2, 12.7))
    test_dir = root / "locked" if locked else folder
    if axial is not None:
        test_dir = test_dir / f"axial{axial}"
    angles = (6.4, 17.3) if level == 4 else (3.7, 9.9, 15.4)
    all_test = records(test_dir, level, probe, angles)
    if not train or not all_test:
        return []
    transform = beta_filter()
    fit_x, fit_y = [], []
    for _, _, y in train:
        mask = (np.maximum.accumulate(y[:, 4]) <= 1.0) & (np.arange(24) >= 3)
        fit_x.extend((transform @ y[:, 1])[mask])
        fit_y.extend(y[mask, 2])
    fit_x = np.array(fit_x)
    gain = float(fit_x @ fit_y / (fit_x @ fit_x))
    rows = []
    for sigma in SIGMAS:
        rng = np.random.default_rng(SEED)
        for name, load, y in all_test:
            noisy = y[:, 1] + rng.normal(0, sigma, (512, 24))
            prediction = gain * (noisy @ transform.T)
            threshold = np.maximum(0.12, 3.5 * abs(gain) * sigma * np.linalg.norm(transform, axis=1))
            fired = abs(prediction) >= threshold
            first = np.where(fired.any(axis=1), fired.argmax(axis=1), 24)
            exits = np.flatnonzero(y[:, 4] > 1.0)
            exit_index = int(exits[0]) if len(exits) else 24
            selected = np.minimum(first, 23)
            correct = (
                (first < exit_index)
                & (abs(y[selected, 2]) >= 0.1)
                & (np.sign(prediction[np.arange(len(noisy)), selected]) == np.sign(y[selected, 2]))
            )
            wrong = (first < exit_index) & ~correct
            rows.append(
                dict(
                    probe=probe,
                    axial_slices=axial,
                    record=name,
                    load_nm=load,
                    noise_ma=sigma * 1000,
                    n_replays=len(noisy),
                    level=level,
                    locked=locked,
                    gain_deg_a=gain,
                    exited=bool(len(exits)),
                    exit_by_ms=int(exit_index + 1) if len(exits) else None,
                    detection_before_exit=float(correct.mean()),
                    false_early=float(wrong.mean()),
                    median_correct_ms=float(np.median(first[correct] + 1)) if correct.any() else None,
                    median_true_displacement_at_correct_deg=float(np.median(abs(y[selected[correct], 2])))
                    if correct.any()
                    else None,
                    peak_displacement_deg=float(y[:, 4].max()),
                    p95_error_at_8ms_deg=float(np.percentile(abs(prediction[:, 7] - y[7, 2]), 95)),
                )
            )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    rows = [
        r
        for probe in ("cold", "biased_pulse", "release10", "release20", "release30")
        for r in controller_readout(a.root, probe)
    ]
    rows += controller_readout(a.root, "release20", 24)
    for axial in (None, 24):
        for locked in (False, True):
            rows += controller_readout(a.root, "release30", axial, level=5, locked=locked)
    result = dict(initial=initial_limits(a.root), rollback=rows)
    (a.root / "scores.json").write_text(json.dumps(result, indent=2), encoding="utf8")

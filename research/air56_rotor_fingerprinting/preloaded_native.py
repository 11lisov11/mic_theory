"""Export two FEM signatures and test the freestanding C reader on held-out starts."""

import argparse
import csv
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.air56_rotor_fingerprinting.preloaded_map import read_records


class Capture(ct.Structure):
    _fields_ = [
        (key, ct.c_double) for key in ("angle", "speed", "load", "redcost", "margin", "initial_deg")
    ] + [("profile", ct.c_int)]


def temporal_modes(out, skew):
    """Fit only on paired development currents. Hold out entire angles for the error budget."""
    _, coarse = read_records(out, "transfer", skew)
    _, fine = read_records(out, "time_holdout", skew)
    difference = (fine - coarse)[..., :-2]
    samples = difference.shape[-1]

    def directions(waves):
        return np.linalg.svd(waves.reshape(-1, samples), full_matrices=False)[2][:8]

    worst = 0.0
    for fold in range(4):
        holdout = np.arange(len(difference)) % 4 == fold
        axes = directions(difference[~holdout])
        delta = difference[holdout].reshape(-1, samples)
        residual = delta - (delta @ axes.T) @ axes
        worst = max(worst, np.linalg.norm(residual, axis=1).max() / np.sqrt(samples - 9))
    return directions(difference), float(worst)


def export(out, skew, filtered=False):
    banks = []
    for family, mid in (("cal", "mid"), ("cal_fine", "mid_fine")):
        angles, first = read_records(out, family, skew)
        mid_angles, second = read_records(out, mid, skew)
        order = np.argsort(np.r_[angles, mid_angles])
        banks.append(np.concatenate((first, second))[order])
    raw = np.stack(banks)
    samples = raw.shape[-1] - 2
    current = raw[..., :samples].reshape(-1, samples)
    center = current.mean(axis=0)
    modes, refinement_rms = temporal_modes(out, skew) if filtered else (np.empty((0, samples)), 0.0)
    centered = current - center
    if filtered:
        centered = centered - (centered @ modes.T) @ modes
    dof = samples - len(modes) - 1
    u, singular, rows = np.linalg.svd(centered, full_matrices=False)
    # Stop when the worst calibration waveform loses less than 0.1 mA RMS.
    energy = (u * singular) ** 2
    tail = np.cumsum(energy[:, ::-1], axis=1)[:, ::-1]
    choices = np.flatnonzero(np.sqrt(tail.max(axis=0) / samples) <= 0.0001)
    if not len(choices) or choices[0] > 96:
        raise ValueError("calibration requires more than 96 components to meet its 0.1 mA RMS tolerance")
    rank = int(choices[0])
    basis = rows[:rank].T
    residual_ma = float(np.sqrt(tail[:, rank].max() / samples) * 1000)
    features = np.concatenate(((raw[..., :samples] - center) @ basis, raw[..., -2:]), axis=-1)
    table = np.stack((features.mean(axis=2), (features[:, :, 1] - features[:, :, 0]) / 0.06), axis=2)
    # Calibrate a waveform error allowance at known angle and torque. These
    # development angles are disjoint from all subsequent blind trials.
    discrepancy = []
    for profile, dev in enumerate(("test", "transfer")):
        poses, records = read_records(out, dev, skew)
        norms = []
        for angle, pair in zip(poses, records):
            coordinate = angle * table.shape[1] / 360
            node = int(coordinate)
            t = coordinate - node
            a, b, c, d = table[profile, (node + np.arange(-1, 3)) % table.shape[1]]
            coef = 0.5 * (
                2 * b
                + (-a + c) * t
                + (2 * a - 5 * b + 4 * c - d) * t**2
                + (-a + 3 * b - 3 * c + d) * t**3
            )
            for load, measured in zip((-0.02, 0.02), pair):
                latent = coef[0, :rank] + load * coef[1, :rank]
                residual = center + basis @ latent - measured[:samples]
                if filtered:
                    residual -= (residual @ modes.T) @ modes
                norms.append(np.linalg.norm(residual) / np.sqrt(dof))
        discrepancy.append(float(max(norms)))
    discrepancy[1] += 2 * refinement_rms
    arrays = tuple(np.ascontiguousarray(a, dtype="<f4") for a in (center, basis, table))
    stem = f"capture_bank_skew{skew}"
    dest = out / "filtered" if filtered else out
    dest.mkdir(exist_ok=True)
    with (dest / (stem + ".bin")).open("wb") as stream:
        for array in arrays:
            stream.write(array.tobytes())
        stream.write(np.asarray(modes, dtype="<f4").tobytes())
    config = dict(
        schema="air56-capture-v1",
        samples=samples,
        dimensions=rank,
        angles=table.shape[1],
        profiles=table.shape[0],
        order=["center", "basis", "table"] + (["removed_basis"] if filtered else []),
        removed=len(modes),
        development_refinement_rms_a=refinement_rms,
        dtype="float32-le",
        skew_steps=skew,
        calibration_max_residual_ma=residual_ma,
        threshold_reduced_cost=chi2.ppf(0.999, dof) / dof,
        threshold_polarity_margin=float(2 * np.log(1000)),
        profile_discrepancy_a=discrepancy,
        calibration_loads_nm=[-0.03, 0.03],
        calibration_levels=[3, 4],
        calibration_step_deg=360 / table.shape[1],
        signature="12 x 0.25 ms at 82 V + 8 x 1 ms at 35 V, 310 V SVM, alpha/beta at dwell ends",
        sha256=hashlib.sha256((dest / (stem + ".bin")).read_bytes()).hexdigest(),
    )
    (dest / (stem + ".json")).write_text(json.dumps(config, indent=2))
    return arrays, config


def library(out, build=True, filtered=False):
    out.mkdir(parents=True, exist_ok=True)
    binary = out / ("capture_lookup.dll" if os.name == "nt" else "capture_lookup.so")
    if build:
        subprocess.run(
            [
                "gcc",
                "-std=c99",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-shared",
                "-fPIC",
                str(Path(__file__).with_name("capture_lookup.c")),
                "-o",
                str(binary),
                "-lm",
            ],
            check=True,
        )
    binary_library = ct.CDLL(str(binary.resolve()))
    fn = binary_library.air56_capture_filtered if filtered else binary_library.air56_capture
    vector = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    fn.argtypes = (
        [vector] * 4
        + [ct.c_int] * 4
        + [ct.c_double]
        + ([vector, ct.c_int] if filtered else [])
        + [ct.POINTER(Capture)]
    )
    fn.restype = ct.c_int
    return fn


def compatible(result, config, noise):
    defect = config["profile_discrepancy_a"][result.profile]
    residual_ok = np.sqrt(result.redcost) <= np.sqrt(config["threshold_reduced_cost"]) + defect / noise
    separation_ok = result.margin >= config["threshold_polarity_margin"] * (1 + (defect / noise) ** 2)
    return bool(residual_ok and separation_ok)


def read_bank(out, skew, currents, noise):
    stem = f"capture_bank_skew{int(skew)}"
    config = json.loads((out / (stem + ".json")).read_text())
    n, dim, angles, profiles = (config[k] for k in ("samples", "dimensions", "angles", "profiles"))
    payload = (out / (stem + ".bin")).read_bytes()
    if config["schema"] != "air56-capture-v1" or hashlib.sha256(payload).hexdigest() != config["sha256"]:
        raise ValueError("the capture binary does not match its calibration metadata")
    data = np.frombuffer(payload, dtype="<f4")
    removed = config.get("removed", 0)
    table_end = n + n * dim + profiles * angles * 2 * (dim + 2)
    if len(currents) != n or data.size != table_end + removed * n:
        raise ValueError("capture table or sample count does not match the signature")
    center, basis, table, modes = np.split(data, [n, n + n * dim, table_end])
    reader = library(out, build=False, filtered=bool(removed))
    result = Capture()
    if not reader(
        np.asarray(currents, dtype="float32"),
        center,
        basis,
        table,
        n,
        dim,
        angles,
        profiles,
        noise,
        *((modes, removed) if removed else ()),
        ct.byref(result),
    ):
        raise ValueError("invalid capture table inputs")
    valid = compatible(result, config, noise)
    return result, valid


def validate(out, build=True, filtered=False):
    reader = library(out / "filtered" if filtered else out, build=build, filtered=filtered)
    records = []
    for skew in (0, 1):
        arrays, config = export(out, skew, filtered=filtered)
        extra = ()
        if filtered:
            modes, _ = temporal_modes(out, skew)
            extra = (np.ascontiguousarray(modes, dtype="float32"), len(modes))
        shape = [config[k] for k in ("samples", "dimensions", "angles", "profiles")]
        for kind in ("blind_test", "blind_transfer", "blind_time_holdout") + (
            ("deep_test",) if filtered else ()
        ):
            angles, responses = read_records(out, kind, skew)
            for i, a in enumerate(angles):
                for sign, response in zip((-1, 1), responses[i]):
                    for seed in range(4):
                        rng = np.random.default_rng([3107, i, sign + 1, skew, seed])
                        perturbation = rng.standard_normal(shape[0])
                        for noise in (0.0003, 0.001, 0.003):
                            reading = np.asarray(response[:-2] + noise * perturbation, dtype=np.float32)
                            result = Capture()
                            begin = perf_counter_ns()
                            ok = reader(reading, *arrays, *shape, noise, *extra, ct.byref(result))
                            elapsed = (perf_counter_ns() - begin) / 1e6
                            if not ok:
                                raise ValueError("invalid reader inputs")
                            records.append(
                                dict(
                                    kind=kind,
                                    skew=skew,
                                    initial_deg=a,
                                    draw=seed,
                                    noise_ma=1000 * noise,
                                    load_nm=sign * 0.02,
                                    capture_deg=a + response[-2],
                                    estimate_deg=result.angle,
                                    error_deg=(result.angle - a - response[-2] + 180) % 360 - 180,
                                    true_rpm=response[-1] * 30 / np.pi,
                                    estimated_rpm=result.speed * 30 / np.pi,
                                    estimated_load_nm=result.load,
                                    chosen_level=3 + result.profile,
                                    compute_ms=elapsed,
                                    reduced_cost=result.redcost,
                                    margin=result.margin,
                                    accepted=compatible(result, config, noise),
                                )
                            )
    with (out / ("filtered_scores.csv" if filtered else "native_scores.csv")).open(
        "w", encoding="utf8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--prepare", action="store_true", help="Compile and export banks before blind validation"
    )
    parser.add_argument("--existing-library", action="store_true")
    parser.add_argument("--filtered", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        library(args.out / "filtered" if args.filtered else args.out, filtered=args.filtered)
        for skew in (0, 1):
            export(args.out, skew, filtered=args.filtered)
    else:
        validate(args.out, build=not args.existing_library, filtered=args.filtered)

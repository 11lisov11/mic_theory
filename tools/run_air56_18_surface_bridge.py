"""Compare the localized interlaminar bridge with modulated slots on the 24/18 profile."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fem.air56.current_tracking import CurrentMap
from fem.air56.sliding_section import make_section, solve
from tools.run_air56_18_slot_openings import (
    FREQUENCY_HZ, LONG_WINDOW_RPM, SEEDS, VOLTAGE_PEAK, article_baseline,
    file_sha256, harmonic_current, motion_results,
)
from tools.run_air56_marker_fem_study import complex_json
from tools.run_air56_skew_audit import NOISE, RPM, article_profile, write_csv
from tools.run_air56_slot_comparison import BRIDGE_OHM, PATCH, coupon, quality, skewed


PERIOD = np.pi
POINTS = 112
TRANSFER_LEVELS = (7, 10)


def opening_maps(path, geometry, closed_hash):
    summary = json.loads((path.parent/"summary.json").read_text(encoding="utf8"))
    if (summary["geometry"] != asdict(geometry) or summary["closed_map_sha256"] != closed_hash
            or summary["frequency_hz"] != FREQUENCY_HZ
            or summary["files_sha256"][path.name] != file_sha256(path)):
        raise ValueError("expected the verified modulated-opening maps for the same rotor profile")
    with np.load(path, allow_pickle=False) as data:
        period = float(data["modulated_period_rad"])
        offset = float(data["modulated_holdout_offset_rad"])
        return tuple(CurrentMap.from_impedance(period, data[f"modulated_{label}_ohm"], shift)
                     for label, shift in (("calibration", 0.), ("holdout", offset)))


def transfer_map(geometry, design, angles, level, gap_phase):
    def at(angle):
        section = make_section(geometry, angle, level, design, gap_phase=gap_phase,
                               stator_opening_fraction=.16, stator_opening_depth_m=.0007)
        return solve(geometry, FREQUENCY_HZ, section, 27.5/3, transfer=True)

    rows = [at(angle) for angle in angles]
    earlier = rows[len(angles)//3]["transfer"]
    opposite = at(angles[len(angles)//3]+PERIOD)["transfer"]
    checks = {field: float(max(row[field] for row in rows))
              for field in ("dofs", "energy", "residual", "reciprocity")}
    checks["half_turn_transfer_rel_error"] = float(np.linalg.norm(earlier+opposite)/np.linalg.norm(earlier))
    return np.array([row["transfer"] for row in rows]), checks


def apply_bridge(background, offset, delta_impedance):
    angles = np.arange(len(delta_impedance))*PERIOD/len(delta_impedance)+offset
    plain = background.sample(angles)
    # Keep the localized impedance term at the axial midpoint. The caller
    # averages the closed-slot background before adding it for a skewed stack.
    combined = CurrentMap.from_impedance(PERIOD, np.linalg.inv(plain)+delta_impedance, offset)
    increment = CurrentMap(PERIOD, combined.admittance-plain, offset)
    return combined, increment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-map", required=True, type=Path)
    parser.add_argument("--openings-map", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out/"summary.json"
    if summary_path.exists():
        parser.error("choose a new output directory")

    start = perf_counter()
    geometry, design = article_profile()
    pitch = 2*np.pi/geometry.rotor_bars
    closed_hash = file_sha256(args.closed_map)
    closed = article_baseline(args.closed_map)
    openings = opening_maps(args.openings_map, geometry, closed_hash)

    print("Solving the isolated and bridged 3D coupons at two mesh levels", flush=True)
    local = coupon(FREQUENCY_HZ)
    offsets = (0., PERIOD/(2*POINTS))
    nominal = np.arange(POINTS)*PERIOD/POINTS
    correction, transfer_checks = [], []
    arrays = dict(period_rad=PERIOD, offsets_rad=offsets)
    for i, (label, level, offset, response) in enumerate(zip(
        ("calibration", "holdout"), TRANSFER_LEVELS, offsets, local,
    )):
        print(f"Solving closed-rotor transfer: {label}, {POINTS} angles, mesh level {level}", flush=True)
        fields, checks = transfer_map(geometry, design, nominal+offset, level, i)
        # Reciprocal first-order transfer from the localized 3D coupon to the port.
        delta_z = (2/3)*np.einsum("nai,ab,nbj->nij", fields, response["tensor"], fields)
        correction.append(delta_z)
        transfer_checks.append(checks)
        arrays[f"{label}_transfer_t_per_a"] = fields
        arrays[f"{label}_delta_ohm"] = delta_z

    contrast, rows, groups = [], [], []
    for skew_pitches in (0., 1.):
        width = skew_pitches*pitch
        background = tuple(skewed(m, width) for m in closed) if width else closed
        opening_pair = tuple(skewed(m, width) for m in openings) if width else openings
        bridge_a, bridge_b = [apply_bridge(m, offset, delta)
                              for m, offset, delta in zip(background, offsets, correction)]
        bridge_pair = (bridge_a[0], bridge_b[0])
        increment_pair = (bridge_a[1], bridge_b[1])
        for label, pair in (("bridge", bridge_pair), ("modulated", opening_pair),
                            ("bridge_increment", increment_pair)):
            contrast.append(dict(profile=label, skew_pitches=skew_pitches, **quality(*pair),
                                 second_harmonic_a_per_v=harmonic_current(pair[1], 2),
                                 tooth_harmonic_a_per_v=harmonic_current(pair[1], geometry.rotor_bars)))
        for label, m in zip(("calibration", "holdout"), bridge_pair):
            arrays[f"{label}_skew_{int(skew_pitches)}_ohm"] = np.linalg.inv(m.admittance)

        for label, pair in (("bridge", bridge_pair), ("modulated", opening_pair)):
            print(f"Scoring {label}, skew {skew_pitches:g} pitch", flush=True)
            for seconds, speeds in ((1.2, RPM), (4.8, LONG_WINDOW_RPM)):
                for noise in NOISE:
                    trials, stats = motion_results(label, *pair, skew_pitches, seconds, noise, speeds)
                    # Retain the original 1.2 s scenario draw to reproduce the previous noise/angle pairs.
                    rows.extend(r for r in trials if r["true_rpm"] in LONG_WINDOW_RPM)
                    groups.extend(r for r in stats if r["abs_rpm"] in LONG_WINDOW_RPM)

    np.savez_compressed(args.out/"bridge_impedance.npz", **arrays)
    write_csv(args.out/"trials.csv", rows)
    write_csv(args.out/"speed_results.csv", groups)
    outputs = ("bridge_impedance.npz", "trials.csv", "speed_results.csv")
    payload = dict(
        geometry=asdict(geometry), frequency_hz=FREQUENCY_HZ, peak_probe_v=VOLTAGE_PEAK,
        rotor_pitch_deg=float(np.rad2deg(pitch)), noise_a=NOISE, window_s=(1.2, 4.8),
        signed_rpm=LONG_WINDOW_RPM, seeds=[SEEDS.start, SEEDS.stop], trial_count=len(rows),
        coupon_geometry=asdict(PATCH), contact_ohm=BRIDGE_OHM,
        coupon=[dict(tensor=complex_json(r["tensor"]), edges=int(r["edges"]),
                     contact_loss=r["contact_loss"].tolist(), control_loss=r["control_loss"].tolist(),
                     marked_loss=r["marked_loss"].tolist()) for r in local],
        coupon_tensor_relative_difference=float(np.linalg.norm(local[0]["tensor"]-local[1]["tensor"])/
                                                np.linalg.norm(local[1]["tensor"])),
        transfer_mesh_levels=TRANSFER_LEVELS, transfer_checks=transfer_checks,
        closed_map_sha256=closed_hash, openings_map_sha256=file_sha256(args.openings_map),
        contrast=contrast, speed_results=groups, elapsed_s=perf_counter()-start, simulation_only=True,
        files_sha256={name: file_sha256(args.out/name) for name in outputs},
    )
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()

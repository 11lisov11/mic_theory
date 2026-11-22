"""Simulate current-based tracking of three AIR56 rotor designs without hardware."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
import skfem

from config.env_air56_physical_candidate import MOTOR
from fem.air56.coupon import Coupon, contrast, make_problem, normalize_field, solve_coupon
from fem.air56.current_tracking import CurrentMap, score_motion, summarize_motion
from fem.air56.sliding_section import SlotDesign, make_section, solve
from fem.air56.stator import Geometry
from tools.run_air56_marker_fem_study import CONFIG, complex_json


STOCK_SEEDS = (730100, 730124)
NOISE_A = (0.0003, 0.003)
VOLTAGE_PEAK = 1.0
STOCK = SlotDesign()
OPENINGS = SlotDesign(modulation=.40)
PATCH = Coupon(width_mm=.28, contact_positions_mm=(-.78, .78),
               radial_half_extent_mm=.32, tangential_half_extent_mm=1.13)
BRIDGE_OHM = 1e-4
STATOR_OPENING = .16


def run_map(g, frequency, theta, level, design=STOCK, *, transfer=False, source="belts", gap_phase=0, stator_opening=STATOR_OPENING):
    zs, fields, checks = [], [], []
    for angle in theta:
        section = make_section(g, angle, level, design, gap_phase=gap_phase, stator_opening_fraction=stator_opening)
        out = solve(g, frequency, section, MOTOR.Rs, transfer=transfer, current_source=source)
        zs.append(out["z"])
        if transfer:
            fields.append(out["transfer"])
        checks.append([out[name] for name in ("dofs", "residual", "energy", "reciprocity")])
    return np.array(zs), np.array(fields), np.array(checks)


def quality(first, second):
    # Difference of only the angle-dependent signal. The constant impedance
    # offset does not enter the motion estimator either.
    mesh = first.sample(second.offset+np.arange(len(second.admittance))*second.period/len(second.admittance))
    mesh -= mesh.mean(axis=0)
    error = np.linalg.norm((mesh-second.centered()).reshape(len(mesh), -1), axis=1).max()
    return dict(contrast_a_per_v=second.contrast(), angular_rank2_to_rank1=second.rank(),
                angular_map_error_a_per_v=float(error),
                angular_map_error_rel=float(error/max(second.contrast(), 1e-20)))


def benchmark(models, frequency, root, prefix="", *, independent_rpms=None, steps=120):
    all_rows, result = [], {}
    for label, calibration, truth in models:
        result[label] = {}
        for noise in NOISE_A:
            if independent_rpms is None:
                rows = score_motion(calibration, truth, frequency, VOLTAGE_PEAK, noise, range(*STOCK_SEEDS), steps=steps)
            else:
                rows = score_motion(calibration, truth, frequency, VOLTAGE_PEAK, noise,
                                    range(730300, 730324), steps=steps, scenario_rpms=independent_rpms)
            all_rows.extend([dict(model=label, **row) for row in rows])
            result[label][str(noise)] = summarize_motion(rows)
    with (root/f"{prefix}motion_trials.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0])
        writer.writeheader()
        writer.writerows(all_rows)
    return result


def skewed(table, width_rad):
    theta = table.offset+np.arange(len(table.admittance))*table.period/len(table.admittance)
    slices = (np.arange(64)+.5)/64-.5
    z = np.linalg.inv(table.sample(theta[:, None]+slices*width_rad)).mean(axis=1)
    return CurrentMap.from_impedance(table.period, z, table.offset)


def stock(g, root):
    period = 2*np.pi/g.rotor_bars
    angles = np.arange(16)*period/16
    candidates = []
    for frequency in (500, 1000, 2500):
        coarse, _, _ = run_map(g, frequency, angles, 4)
        middle, _, stats = run_map(g, frequency, angles, 7, gap_phase=1)
        a, b = [CurrentMap.from_impedance(period, data) for data in (coarse, middle)]
        row = dict(frequency=frequency, **quality(a, b),
                   max_fem_residual=stats[:, 1].max(), max_energy_error=stats[:, 2].max())
        candidates.append(row)
        print(f"Stock {frequency} Hz, {row['contrast_a_per_v']*1e6:.1f} uA/V", flush=True)
    selected = max(candidates, key=lambda row: row["contrast_a_per_v"]*row["angular_rank2_to_rank1"])
    frequency = selected["frequency"]
    # Mark independently shifted meshes and angle samples. Holdout current
    # interpolation is based on newly solved FEM fields, not the template.
    n = 32
    nominal = np.arange(n)*period/n
    delta = period/(2*n)
    shifted = nominal+delta
    calibration, _, cal_checks = run_map(g, frequency, nominal, 7)
    fine, _, fine_checks = run_map(g, frequency, shifted, 10, gap_phase=1)
    wider, _, wider_checks = run_map(replace(g, air_gap=g.air_gap+.00005),
                                     frequency, shifted, 7, gap_phase=1)
    continuous, _, cont_checks = run_map(g, frequency, shifted, 7, source="continuous")
    a, b, c, d = [CurrentMap.from_impedance(period, arr, offset)
                  for arr, offset in zip((calibration, fine, wider, continuous), (0, delta, delta, delta))]
    zero = CurrentMap(period, np.broadcast_to(b.admittance.mean(axis=0), b.admittance.shape))
    repeat_angle = 0.037
    zs, _, symmetry = run_map(g, frequency, [repeat_angle, repeat_angle+period, repeat_angle+np.pi], 7)
    skewed_models = []
    skew_contrast = {}
    for width in (0.5, 0.9, 1.0):
        skew_a, skew_b = skewed(a, width*period), skewed(b, width*period)
        skew_contrast[str(width)] = skew_b.contrast()
        skewed_models.append((f"stock_skew_{width:g}_pitch", skew_a, skew_b))
    trials = benchmark((("stock", a, b), ("stock_wider_gap", a, c), ("stock_continuous_matched", d, d),
                        ("angle_independent_control", a, zero), *skewed_models), frequency, root, "stock_")
    np.savez_compressed(root/"stock_impedance.npz", hz=frequency, period=period, offsets=[0, delta],
                        z_calibration=calibration, z_fine=fine, z_wider_gap=wider, z_continuous=continuous)
    result = dict(frequency_hz=frequency, comparison=quality(a, b), frequency_screen=candidates,
                  wider_gap=quality(a, c), continuous_winding_contrast_a_per_v=d.contrast(),
                  skew_contrast_a_per_v=skew_contrast,
                  continuous_winding_rank=d.rank(), mechanical_period_deg=np.rad2deg(period),
                  repeats_by_pitch_ohm=np.max(abs(zs[0]-zs[1])), repeat_by_half_turn_ohm=np.max(abs(zs[0]-zs[2])),
                  checks={"calibration": cal_checks.max(axis=0), "fine": fine_checks.max(axis=0),
                          "wider_gap": wider_checks.max(axis=0), "continuous": cont_checks.max(axis=0),
                          "symmetry": symmetry.max(axis=0)}, motion=trials)
    print("Stock FEM and independent tracking finished", flush=True)
    return result


def coupon(frequency):
    local = []
    for level in (6, 9):
        problem = make_problem(PATCH, level, padding_mm=1.5)
        a, b = [solve_coupon(problem, frequency, ids, BRIDGE_OHM, PATCH) for ids in ((), (0, 1))]
        n = normalize_field(problem, a)
        tensor = contrast(problem, a, b)
        np.testing.assert_allclose(tensor, b["z"]-a["z"], rtol=1e-5, atol=1e-11)
        local.append(dict(tensor=n.T @ tensor @ n, edges=problem[0].N,
                          contact_loss=b["contact_loss_w"], control_loss=a["loss_w"],
                          marked_loss=b["loss_w"]))
    return local


def compare(g, stock_root, root):
    stock_result = json.loads((stock_root/"stock_summary.json").read_text(encoding="utf8"))
    if stock_result["config"].get("stator_opening_fraction") != STATOR_OPENING:
        raise ValueError("stock table uses a different stator opening")
    with np.load(stock_root/"stock_impedance.npz") as data:
        frequency, period = float(data["hz"]), float(data["period"])
        a_stock = CurrentMap.from_impedance(period, data["z_calibration"], data["offsets"][0])
        b_stock = CurrentMap.from_impedance(period, data["z_fine"], data["offsets"][1])
    local_a, local_b = coupon(frequency)
    n = 112
    full = np.pi
    theta = np.arange(n)*full/n
    offset = full/(2*n)
    stock_on_grid = CurrentMap(full, a_stock.sample(theta))
    stock_holdout_on_grid = CurrentMap(full, b_stock.sample(theta+offset), offset)
    skew_stock = skewed(a_stock, period), skewed(b_stock, period)
    models = [("stock", a_stock, b_stock), ("stock_on_design_grid", stock_on_grid, stock_holdout_on_grid),
              ("stock_one_pitch_skew", *skew_stock)]
    design_rows = {"stock_on_design_grid": quality(stock_on_grid, stock_holdout_on_grid)}
    impedances = {}
    for label in ("surface_bridge", "modulated_openings"):
        print(f"Solving {label}", flush=True)
        if label == "modulated_openings":
            base, _, qa = run_map(g, frequency, theta, 4, OPENINGS)
            fine, _, qb = run_map(g, frequency, theta+offset, 7, OPENINGS, gap_phase=1)
            variant_only = None
        else:
            # The admissible narrow coupon is recalculated in 3D. Transfer is
            # computed on the tooth and superposed on the stock port solution.
            _, fields_a, qa = run_map(g, frequency, theta, 4, transfer=True)
            _, fields_b, qb = run_map(g, frequency, theta+offset, 7, transfer=True, gap_phase=1)
            def added(transfer, response):
                return (2/3)*np.einsum("nai,ab,nbj->nij", transfer, response["tensor"], transfer)
            addition_a, addition_b = added(fields_a, local_a), added(fields_b, local_b)
            base = np.linalg.inv(a_stock.sample(theta))+addition_a
            fine = np.linalg.inv(b_stock.sample(theta+offset))+addition_b
            delta = np.linalg.inv(fine)-b_stock.sample(theta+offset)
            variant_only = CurrentMap(full, delta, offset).contrast()
        first = CurrentMap.from_impedance(full, base)
        second = CurrentMap.from_impedance(full, fine, offset)
        models.append((label, first, second))
        if label == "surface_bridge":
            # The contact patch is localized in the middle of the stack.
            # Average the unmarked background and keep the midplane correction.
            skew_a = CurrentMap.from_impedance(full, np.linalg.inv(skew_stock[0].sample(theta))+addition_a)
            skew_b = CurrentMap.from_impedance(full, np.linalg.inv(skew_stock[1].sample(theta+offset))+addition_b, offset)
        else:
            skew_a, skew_b = skewed(first, period), skewed(second, period)
        models.append((f"{label}_one_pitch_skew", skew_a, skew_b))
        impedances[f"{label}_calibration_ohm"] = base
        impedances[f"{label}_holdout_ohm"] = fine
        harmonic = 2*np.abs(np.fft.fft(second.centered(), axis=0))/len(second.admittance)
        design_rows[label] = dict(**quality(first, second), bridge_only_angular_current_a_per_v=variant_only,
                                  one_pitch_skew_contrast_a_per_v=skew_b.contrast(),
                                  second_mechanical_order_a_per_v=float(np.linalg.norm(harmonic[1])),
                                  tooth_mechanical_order_a_per_v=float(np.linalg.norm(harmonic[g.rotor_bars//2])),
                                  fem_checks=[qa.max(axis=0),qb.max(axis=0)])
    np.savez_compressed(root/"comparison_impedance.npz", hz=frequency, angle_rad=theta,
                        holdout_offset_rad=offset, **impedances)
    draws = benchmark(models, frequency, root, "comparison_")
    return dict(frequency_hz=frequency, shared_probe_peak_v=VOLTAGE_PEAK, motion=draws, designs=design_rows,
                stock_input_sha256=hashlib.sha256((stock_root/"stock_impedance.npz").read_bytes()).hexdigest(),
                compared_openings=asdict(OPENINGS), contact_parameters=asdict(PATCH),
                coupon=[dict(row, tensor=complex_json(row["tensor"])) for row in (local_a, local_b)],
                coupon_tensor_relative_difference=float(np.linalg.norm(local_a["tensor"]-local_b["tensor"])/
                                                       np.linalg.norm(local_b["tensor"])))


def offgrid(stock_root, comparison_root, output):
    models = []
    with np.load(stock_root/"stock_impedance.npz") as data:
        frequency, period = float(data["hz"]), float(data["period"])
        a, b = [CurrentMap.from_impedance(period, data[key], offset)
                for key, offset in zip(("z_calibration", "z_fine"), data["offsets"])]
        models.extend((("stock", a, b), ("stock_one_pitch_skew", skewed(a, period), skewed(b, period))))
    with np.load(comparison_root/"comparison_impedance.npz") as data:
        if float(data["hz"]) != frequency:
            raise ValueError("off-grid comparison requires a shared probe frequency")
        offset = float(data["holdout_offset_rad"])
        for label in ("surface_bridge", "modulated_openings"):
            first = CurrentMap.from_impedance(np.pi, data[f"{label}_calibration_ohm"])
            second = CurrentMap.from_impedance(np.pi, data[f"{label}_holdout_ohm"], offset)
            models.append((label, first, second))
    speeds = [0., -27.3, -9.8, -2.7, -1.2, 1.2, 2.7, 9.8, 27.3]
    result = benchmark(models, frequency, output, "offgrid_", independent_rpms=speeds)
    short = benchmark(models[:1], frequency, output, "stock_024s_", independent_rpms=speeds, steps=24)
    return dict(frequency_hz=frequency, speeds_rpm=speeds, seed_range=[730300, 730324], motion=result,
                stock_024s=short["stock"])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stock", "compare", "offgrid"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stock", type=Path)
    parser.add_argument("--comparison", type=Path)
    args = parser.parse_args()
    dst = args.out/f"{args.stage}_summary.json"
    if dst.exists() or (args.stage != "stock" and args.stock is None):
        parser.error("provide an unused output and the completed stock directory")
    if args.stage == "offgrid" and args.comparison is None:
        parser.error("offgrid requires the completed comparison directory")
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf8"))
    geometry = Geometry.from_config(config)
    start = perf_counter()
    if args.stage == "stock":
        result = stock(geometry, args.out)
    elif args.stage == "compare":
        result = compare(geometry, args.stock, args.out)
    else:
        result = offgrid(args.stock, args.comparison, args.out)
    result["elapsed_s"] = perf_counter()-start
    result["solver_versions"] = dict(python=platform.python_version(), numpy=np.__version__,
                                     scipy=scipy.__version__, skfem=skfem.__version__)
    result["geometry_sha256"] = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    result["config"] = dict(stator_resistance=MOTOR.Rs, stock_slots=asdict(STOCK),
                            stator_opening_fraction=STATOR_OPENING,
                            peak_voltage=VOLTAGE_PEAK, noise_a=NOISE_A)
    result["config"]["run_seeds"] = (730300, 730324) if args.stage == "offgrid" else STOCK_SEEDS
    dst.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda number: number.tolist())+"\n", encoding="utf8")


if __name__ == "__main__":
    main()

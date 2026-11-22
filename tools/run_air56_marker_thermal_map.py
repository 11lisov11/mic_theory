"""Postprocess solved coupon fields into angular heat and temperature maps."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fem.air56.coupon import Coupon, MU0, contrast, make_problem, normalize_field, solve_coupon
from fem.air56.radial_mode import solve_mode
from fem.air56.stator import Geometry
from fem.air56.thermal import steady_temperature
from tools.run_air56_marker_fem_study import CONFIG, REF_HZ, save_json


def local_angular_fields(problem, solution, drives):
    basis = problem[0]
    material = solution["materials"]
    conductivity = solution["sigma"]
    electric_field = solution["field_at_q"]
    q = 0.5*conductivity[None, None, :, None]*np.einsum(
        "aijk,bijk->abjk", electric_field, electric_field.conj())
    q = (q*basis.dx[None, None, ...]).sum(axis=-1)/basis.dx.sum(axis=1)
    profiles = []
    # Four positive heat cases recover the Hermitian basis for every node.
    for drive in (np.array([1, 0]), np.array([0, 1]), np.array([1, 1]), np.array([1, 1j])):
        heat = np.einsum("a,b,abm->m", drive, drive.conj(), q).real
        temperature, balance = steady_temperature(basis.mesh, material, heat, ambient_c=0)
        if abs(balance-1) > 1e-6:
            raise ValueError("thermal heat-sink balance did not close")
        profiles.append(temperature)
    t0, t1, t01, t0i = profiles
    drives = drives[:, :, 0]
    cross = drives[:, 0]*drives[:, 1].conj()
    temperature = (abs(drives[:, 0, None])**2*t0 + abs(drives[:, 1, None])**2*t1 +
                   cross.real[:, None]*(t01-t0-t1) - cross.imag[:, None]*(t0i-t0-t1))
    heat = np.einsum("na,nb,abm->nm", drives, drives.conj(), q).real
    volumes = basis.dx.sum(axis=1)*1e-9
    metal = material > 0
    magnetic = np.einsum("na,amj->nmj", drives, solution["cell_b_t"][:, ~metal])
    peak_b = np.linalg.norm(magnetic, axis=-1).max(axis=1)
    # Air Maxwell stress envelope for the diagnostic phasor alone.
    pressure_bound = peak_b**2/(2*MU0)
    return {"loss_w": heat[:, metal] @ volumes[metal],
            "contact_loss_w": heat[:, material == 2] @ volumes[material == 2],
            "max_temperature_rise_k": np.nanmax(temperature, axis=1), "pressure_peak_bound_pa": pressure_bound}


def refinement_check(directory, summary, coupon):
    rows = []
    reference_level = max(row["refinement"] for row in summary["mesh_3d"])
    problem = make_problem(coupon, 3)
    for hz in (50, REF_HZ, 10000):
        blank = solve_coupon(problem, hz, cfg=coupon)
        marker = solve_coupon(problem, hz, (0, 1), cfg=coupon)
        norm = normalize_field(problem, blank)
        value = norm.T @ contrast(problem, blank, marker) @ norm
        prior = next(row["normalized_delta_z_per_t2"] for row in summary["mesh_3d"]
                     if row["frequency_hz"] == hz and row["refinement"] == reference_level)
        old = np.asarray(prior["real"])+1j*np.asarray(prior["imag"])
        rows.append(dict(frequency_hz=hz, dofs=problem[0].N,
                         norm_rel_change=float(np.linalg.norm(value-old)/np.linalg.norm(value)),
                         loss_w_control=blank["loss_w"].tolist(), loss_w_contact=marker["loss_w"].tolist(),
                         real=value.real.tolist(), imag=value.imag.tolist()))
    save_json(directory/"refinement_check.json", rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fem", required=True, type=Path)
    parser.add_argument("--refine-reference", action="store_true")
    args = parser.parse_args()
    directory = args.fem
    summary = json.loads((directory/"fem_summary.json").read_text(encoding="utf8"))
    config = json.loads(CONFIG.read_text(encoding="utf8"))
    d = config["dimensions_mm"]
    geom = Geometry.from_config(config)
    coupon = Coupon(thickness_mm=d["sheet_thickness"][1], gap_mm=d["insulation_gap"][1],
                    width_mm=d["contact_width"][1], mu_r=geom.mu_r,
                    sigma_20=config["steel_sigma_s_m_20c"][1],
                    reference_temperature_coefficient=config["steel_temperature_coefficient_per_k"],
                    contact_positions_mm=tuple(config["contact_locations_local_mm"]))
    level = max(row["refinement"] for row in summary["mesh_3d"] if row["frequency_hz"] == REF_HZ)
    problem = make_problem(coupon, level)
    blank = solve_coupon(problem, REF_HZ, cfg=coupon)
    marker = solve_coupon(problem, REF_HZ, (0, 1), cfg=coupon)
    normalizer = normalize_field(problem, blank)
    mapped = normalizer.T @ contrast(problem, blank, marker) @ normalizer
    expected = next(r["normalized_delta_z_per_t2"] for r in summary["mesh_3d"]
                    if r["frequency_hz"] == REF_HZ and r["refinement"] == level)
    np.testing.assert_allclose(mapped, np.asarray(expected["real"])+1j*np.asarray(expected["imag"]),
                               rtol=1e-6, atol=1e-11)
    transfer = solve_mode(geom, REF_HZ, skin_refinement=12, sample_depths_m=(0.002,))["transfer"]
    drive = normalizer @ transfer
    a = local_angular_fields(problem, blank, drive)
    b = local_angular_fields(problem, marker, drive)
    np.testing.assert_allclose(b["max_temperature_rise_k"][0], summary["marker_fields"]["delta_t_max_k"],
                               rtol=3e-6, atol=3e-14)
    dst = directory/"angular_thermal_map_2500hz.csv"
    with dst.open("w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta_deg", *[f"control_{k}" for k in a], *[f"marker_{k}" for k in b],
                    "marker_thermal_expansion_bound_m"])
        for angle in range(360):
            w.writerow([angle, *[v[angle] for v in a.values()], *[v[angle] for v in b.values()],
                        b["max_temperature_rise_k"][angle]*12e-6*0.012])
    save_json(directory/"angular_thermal_summary.json",
              {"basis": "1 A peak alpha current, diagnostic only, fixed 20 C radial faces",
               "frequency_hz": REF_HZ, "peak_control": {k: v.max() for k, v in a.items()},
               "peak_marker": {k: v.max() for k, v in b.items()},
               "maximum_thermal_expansion_bound_m": b["max_temperature_rise_k"].max()*12e-6*0.012})
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), constrained_layout=True)
    theta = np.arange(360)
    axes[0].plot(theta, a["loss_w"]*1e6, label="Isolated sheets")
    axes[0].plot(theta, b["loss_w"]*1e6, label="Two bridges")
    axes[0].plot(theta, b["contact_loss_w"]*1e6, label="Contact material only")
    axes[0].set(ylabel="Joule loss, µW", title="Local FEM at 2.5 kHz · 1 A peak α")
    axes[0].legend()
    axes[1].plot(theta, a["max_temperature_rise_k"]*1e9, label="Isolated sheets")
    axes[1].plot(theta, b["max_temperature_rise_k"]*1e9, label="Two bridges")
    axes[1].set(xlabel="Mechanical angle, deg", ylabel="Peak temperature rise, nK")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 360)
    fig.savefig(directory/"angular_thermal_map.png", dpi=170)
    plt.close(fig)
    if args.refine_reference:
        refinement_check(directory, summary, coupon)


if __name__ == "__main__":
    main()

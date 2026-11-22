"""Run the AIR56 laminate coupon and reduced machine experiment, host only."""

import argparse
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

import meshio
import numpy as np
import scipy
import skfem

from fem.air56.coupon import Coupon, contrast, make_problem, normalize_field, solve_coupon
from fem.air56.radial_mode import solve_mode
from fem.air56.stator import Geometry, solve_section
from fem.air56.thermal import steady_temperature


CONFIG = ROOT / "research/air56_interlaminar_marker/geometry.json"
PROFILE = "AIR56_PHYSICAL_CANDIDATE"
FREQUENCIES = (50, 500, 1000, 2500, 5000, 10000)
REF_HZ = 2500


def relative(a, b):
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-20))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=lambda scalar: scalar.item()) + "\n", encoding="utf8")


def complex_json(value):
    return {"real": value.real.tolist(), "imag": value.imag.tolist()}


def fields(problem, response, drive, temp, path):
    basis = problem[0]
    cell_b = np.einsum("i,ijk->jk", drive, response["cell_b_t"])
    cell_j = np.einsum("i,ijk->jk", drive, response["cell_j_a_m2"])
    field_q = np.einsum("i,ijkl->jkl", drive, response["field_at_q"])
    density = 0.5*response["sigma"]*np.average(abs(field_q)**2, axis=-1, weights=basis.W).sum(axis=0)
    temperature, balance = steady_temperature(basis.mesh, response["materials"], density, temp)
    meshio.write(path, meshio.Mesh(
        basis.mesh.p.T*1e-3, [("tetra", basis.mesh.t.T)],
        point_data={"temperature_C": temperature},
        cell_data={key: [value] for key, value in {
            "B_real_T": cell_b.real, "B_imag_T": cell_b.imag,
            "J_real_A_m2": cell_j.real, "J_imag_A_m2": cell_j.imag,
            "heat_W_m3": density, "material": response["materials"],
        }.items()}))
    return {"delta_t_max_k": float(np.nanmax(temperature)-temp),
            "thermal_energy_balance": balance,
            "peak_contact_heat_w_m3": float(density[response["materials"] == 2].max(initial=0)),
            "peak_b_t": float(np.max(np.linalg.norm(cell_b, axis=1))),
            "peak_j_a_m2": float(np.max(np.linalg.norm(cell_j, axis=1))),
            "rms_a_per_drive": float(np.linalg.norm(drive))}


def angular_transfer(section, normalized_response):
    t = section["transfer"]
    return np.einsum("nai,ab,nbj->nij", t, normalized_response, t)


def local(config, root, fast=False):
    g = Geometry.from_config(config)
    dimensions = config["dimensions_mm"]
    c = Coupon(thickness_mm=dimensions["sheet_thickness"][1], gap_mm=dimensions["insulation_gap"][1],
               width_mm=dimensions["contact_width"][1], mu_r=g.mu_r,
               sigma_20=config["steel_sigma_s_m_20c"][1],
               reference_temperature_coefficient=config["steel_temperature_coefficient_per_k"],
               contact_positions_mm=tuple(config["contact_locations_local_mm"]))
    levels = (1, 1.5, 2) if fast else (1.5, 2, 2.5)
    angle = np.arange(360)*2*np.pi/360
    depths = (0.0004, 0.0012, 0.002)
    coarse_sections = {hz: solve_mode(g, hz, skin_refinement=6, sample_depths_m=depths)
                       for hz in FREQUENCIES}
    sections = {hz: solve_mode(g, hz, skin_refinement=12, sample_depths_m=depths)
                for hz in FREQUENCIES}
    section_error = {hz: relative(coarse_sections[hz]["transfer"], sections[hz]["transfer"])
                     for hz in FREQUENCIES}
    depth_sensitivity = []
    for hz, data in sections.items():
        depth_sensitivity.append(dict(frequency_hz=hz, skin_depth_mm=data["skin_depth_mm"],
                                     radial_elements=data["radial_elements"],
                                     field_t_per_a_peak_by_depth_mm={f"{depth*1000:g}": float(abs(value).max())
                                                                      for depth, value in data["curves"].items()}))
        motion_profile = json.loads((CONFIG.parent / "profiles.json").read_text())[PROFILE]
    slip = 1-motion_profile["speed_rpm"]/3000
    nominal = []
    for angular_grid in (336, 672):
        res = solve_section(g, 50, ntheta=angular_grid, radial_refinement=angular_grid//336,
                            slotted=True, slip=slip)
        nominal.append({"ntheta": angular_grid, "z_ohm_without_copper": complex_json(res["z"]),
                        "cage_loss_w_at_1a_peak": res["cage_loss_w"].tolist(),
                        "electrical_balance_rel": relative(0.5*res["z"].real.diagonal(),
                                                           res["cage_loss_w"]/slip),
                        "torque_nm_at_1a_peak": res["torque_nm_per_a_peak_squared"]})
    meshio.write(root / "machine_2d_50hz.vtu", meshio.Mesh(
        np.c_[res["mesh"].p.T, np.zeros(res["mesh"].nvertices)],
        [("triangle", res["mesh"].t.T)],
        point_data={"Az_real_Wb_m": res["values"][:, 0].real, "Az_imag_Wb_m": res["values"][:, 0].imag},
        cell_data={"cage": [res["cage"].astype(np.uint8)]}))
    # Static slot-opening comparator at one fixed rotor registration. An angle
    # sweep is performed independently of the reduced smooth-airgap transfer.
    slot_modes = []
    for openings in (g.rotor_bar_fraction, g.rotor_bar_fraction*0.65, g.rotor_bar_fraction*1.35):
        slot_geometry = replace(g, rotor_bar_fraction=openings)
        for degrees in np.linspace(0, 360/g.rotor_bars, 7, endpoint=False):
            q = solve_section(slot_geometry, REF_HZ, slotted=True, rotor_angle_rad=np.deg2rad(degrees))
            slot_modes.append({"opening_fraction": openings, "angle_deg": degrees,
                               "raw_terminal_z_ohm": complex_json(q["z"])})
    mark, raw, trends, depth_maps = [], [], [], []
    reference_3d = None
    for hz in FREQUENCIES:
        contact = (0, 1)
        study_levels = levels if hz in (50, REF_HZ, 10000) else levels[-2:]
        entries = []
        for level in study_levels:
            problem = make_problem(c, level)
            origin = solve_coupon(problem, hz, cfg=c)
            active = solve_coupon(problem, hz, contact, cfg=c)
            matrix = contrast(problem, origin, active)
            normalizer = normalize_field(problem, origin)
            mapped = angular_transfer(sections[hz], normalizer.T @ matrix @ normalizer)
            entries.append(mapped)
            cell_balance = relative(active["z"].real.diagonal()*0.5,
                                    active["loss_w"]+active["air_loss_w"])
            raw.append(dict(frequency_hz=hz, refinement=level, edges=problem[0].N,
                            max_residual=max(origin["residual"], active["residual"]),
                            normalized_delta_z_per_t2=complex_json(normalizer.T @ matrix @ normalizer),
                            energy_balance_rel=cell_balance,
                            reciprocity_rel=relative(matrix.T, matrix),
                            contrast_boundary_rel=relative(matrix, active["z"]-origin["z"]),
                            loss_w_control=origin["loss_w"].tolist(), loss_w_active=active["loss_w"].tolist(),
                            gauge_loss_w=active["air_loss_w"].tolist()))
            if hz == REF_HZ and level == levels[-1]:
                drive = normalizer @ sections[hz]["transfer"][0, :, 0]
                baseline_fields = fields(problem, origin, drive, 20, root/"coupon_control_2500hz.vtu")
                marker_fields = fields(problem, active, drive, 20, root/"coupon_marker_2500hz.vtu")
                reference_3d = normalizer.T @ matrix @ normalizer
        mark.append(entries[-1])
        coarse_transfer = angular_transfer(coarse_sections[hz], normalizer.T @ matrix @ normalizer)
        trends.append(np.abs(entries[-1]-entries[-2]) + np.abs(entries[-1]-coarse_transfer))
        depth_maps.append(np.stack([angular_transfer({"transfer": value}, normalizer.T @ matrix @ normalizer)
                                    for value in sections[hz]["curves"].values()]))
        print(f"{hz} Hz: {problem[0].N:,} edge DOFs, max dZ {abs(entries[-1]).max():.3g} ohm", flush=True)
    # One-variable and topology screens. Reuse controls for resistance sweeps.
    designs = [
        ("isolated", c, (), 1e-4, 0),
        ("one_bridge", c, (0,), 1e-4, 0),
        ("two_bridges", c, (0, 1), 1e-4, 0),
        ("three_sheets_two_bridges", replace(c, sheets=3), (0, 3), 1e-4, 0),
        ("three_sheets_two_loops", replace(c, sheets=3), (0, 1, 2, 3), 1e-4, 0),
        ("four_sheets", replace(c, sheets=4), tuple(range(6)), 1e-4, 0),
        ("five_sheets", replace(c, sheets=5), tuple(range(8)), 1e-4, 0),
        ("pole_face", replace(c, radial_contact_mm=2.5), (0, 1), 1e-4, 0),
        ("towards_yoke", replace(c, radial_contact_mm=-2.5), (0, 1), 1e-4, 45),
        ("compact", replace(c, contact_positions_mm=(-1.6, 1.6)), (0, 1), 1e-4, 20),
        ("narrow_contact", replace(c, width_mm=0.35), (0, 1), 1e-4, 0),
        ("wide_contact", replace(c, width_mm=1.35), (0, 1), 1e-4, 0),
        ("steel_low_mu", replace(c, mu_r=80), (0, 1), 1e-4, 0),
        ("steel_high_mu", replace(c, mu_r=500), (0, 1), 1e-4, 0),
        ("hot_steel", replace(c, temperature_c=100), (0, 1), 1e-4, 0),
        ("one_bridge_cage_return", replace(c, cage_return=True), (0,), 1e-4, 0),
        ("cage_only", replace(c, cage_return=True), (), 1e-4, 0),
    ]
    designs += [(f"r_{value:g}_ohm", c, (0, 1), value, 0)
                for value in (1e-5, 3e-5, 3e-4, 1e-3, 1e-2)]
    design_rows = []
    cached = {}
    for label, cfg, contacts, resistance, offset_deg in designs:
        if cfg not in cached:
            p = make_problem(cfg, levels[0])
            cached[cfg] = p, solve_coupon(p, REF_HZ, cfg=cfg)
        p, baseline = cached[cfg]
        candidate = solve_coupon(p, REF_HZ, contacts, resistance, cfg) if contacts else baseline
        matrix = contrast(p, baseline, candidate)
        norm = normalize_field(p, baseline)
        extra = norm.T @ matrix @ norm
        mapped = np.roll(angular_transfer(sections[REF_HZ], extra), offset_deg, axis=0)
        centered = mapped - mapped.mean(axis=0)
        max_signal_per_amp = np.max(np.linalg.norm(centered, axis=(1, 2)))
        local_loss = candidate["loss_w"]-baseline["loss_w"]
        contact_mass_kg = 7800*len(contacts)*cfg.gap_mm*cfg.width_mm**2*1e-9
        design_rows.append(dict(design=label, parameters=asdict(cfg), contacts=list(contacts),
                                r_ohm=resistance, angle_offset_deg=offset_deg,
                                delta_z_signal_ohm=float(max_signal_per_amp),
                                delta_z_intrinsic_real_peak_ohm=float(np.max(abs(mapped.real))),
                                max_contact_loss_w_per_t2=float(max(candidate["contact_loss_w"])),
                                max_delta_total_loss_w_per_t2=float(max(local_loss)),
                                mass_upper_kg=contact_mass_kg,
                                centrifugal_force_upper_n=contact_mass_kg*g.rotor_radius*(3000*2*np.pi/60)**2,
                                normalized_response=complex_json(extra)))
    # Two sufficiently separated coupons superposed in first-order perturbation.
    base = next(row for row in design_rows if row["design"] == "two_bridges")
    base_matrix = np.array(base["normalized_response"]["real"])+1j*np.array(base["normalized_response"]["imag"])
    for offset in (90, 180):
        m = angular_transfer(sections[REF_HZ], base_matrix)
        total = m+np.roll(m, offset, axis=0)
        grouped = dict(base, design=f"dual_marker_{offset}_deg", contacts=[0, 1, 0, 1], angle_offset_deg=offset,
                       delta_z_signal_ohm=float(np.linalg.norm(total-total.mean(axis=0), axis=(1, 2)).max()),
                       max_delta_total_loss_w_per_t2=2*base["max_delta_total_loss_w_per_t2"],
                       delta_z_intrinsic_real_peak_ohm=float(np.max(abs(total.real))),
                       mass_upper_kg=2*base["mass_upper_kg"],
                       centrifugal_force_upper_n=2*base["centrifugal_force_upper_n"],
                       mechanically_opposed=offset == 180)
        grouped.pop("normalized_response")
        design_rows.append(grouped)
    # Estimate transfer at a marked tooth in the slotted 2D model. The bar
    # shifts with the rotor. This sensitivity is kept separate from the
    # symmetry-projected, fine radial mode at the primary operating point.
    slotted_transfer = []
    tooth_offset = 180/g.rotor_bars
    for angle_deg in np.arange(0, 360, 30 if fast else 15):
        angle_rad = np.deg2rad(angle_deg)
        resolutions = (336, 672) if angle_deg % 45 == 0 else (336,)
        for ntheta in resolutions:
            sol = solve_section(g, REF_HZ, ntheta=ntheta, radial_refinement=4 if ntheta == 336 else 6,
                                slotted=True, rotor_angle_rad=angle_rad)
            index = int(np.argmin(abs((sol["sample_angle"]-np.deg2rad(angle_deg+tooth_offset)+np.pi) %
                                      (2*np.pi)-np.pi)))
            transfer = sol["transfer"][index]
            mapped = transfer.T @ reference_3d @ transfer
            slotted_transfer.append(dict(angle_deg=int(angle_deg), ntheta=ntheta,
                                         delta_z_ohm=complex_json(mapped),
                                         base_z_ohm=complex_json(sol["z"]),
                                         max_field_t_per_a=float(abs(transfer).max())))
    sensitivity = []

    for padding, gauge in ((5, 0.5), (3, 0.05), (3, 5.0)):
        p = make_problem(c, levels[1], padding, gauge)
        a = solve_coupon(p, REF_HZ, cfg=c)
        b = solve_coupon(p, REF_HZ, (0, 1), cfg=c)
        check = contrast(p, a, b)
        norm = normalize_field(p, a)
        reference_row = next(x for x in raw if x["frequency_hz"] == REF_HZ and x["refinement"] == levels[1])
        reference = np.array(reference_row["normalized_delta_z_per_t2"]["real"]) + 1j*np.array(reference_row["normalized_delta_z_per_t2"]["imag"])
        sensitivity.append(dict(padding_mm=padding, gauge_s_m=gauge, edges=p[0].N,
                                relative_change=relative(norm.T @ check @ norm, reference)))
    np.savez_compressed(root / "channel.npz", theta_rad=angle, frequencies_hz=FREQUENCIES,
                        delta_z_ohm=np.array(mark), mesh_error_ohm=np.array(trends),
                        stator_z_ohm=np.array([sections[hz]["z"] for hz in FREQUENCIES]), pole_pairs=1,
                        profile=PROFILE, sample_depths_m=depths,
                        depth_delta_z_ohm=np.array(depth_maps))
    return dict(profile=PROFILE, mesh_3d=raw, mesh_2d_relative_change=section_error,
                skin_and_depth=depth_sensitivity, slotted_transfer=slotted_transfer,
                boundary_sensitivity=sensitivity, nominal_2d=nominal,
                marker_fields=marker_fields, control_fields=baseline_fields, slot_screen=slot_modes,
                designs=design_rows, ambient_boundary="two radial coupon faces held at 20 C",
                circuit_scope="smooth 2D field projected onto a local homogeneous-B 3D correction",
                rms_input_note="all local W/T2 values refer to a 1 T peak phasor before field normalization")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"fem_summary.json").exists():
        parser.error("choose a new output directory")
    start = perf_counter()
    result = local(json.loads(CONFIG.read_text(encoding="utf8")), args.out, args.fast)
    result["inputs_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (CONFIG, CONFIG.parent/"profiles.json")}
    result["elapsed_s"] = perf_counter()-start
    result["versions"] = dict(python=platform.python_version(), numpy=np.__version__,
                              scipy=scipy.__version__, skfem=skfem.__version__, meshio=meshio.__version__)
    result["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in args.out.iterdir() if p.suffix in (".npz", ".vtu")}
    save_json(args.out/"fem_summary.json", result)


if __name__ == "__main__":
    main()

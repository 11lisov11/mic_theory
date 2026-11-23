"""Locked-shaft polarity interrogation using paired DC current biases and a 100 Hz probe."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.linalg import splu
from skfem import Basis, ElementTriP1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fem.air56.coupon import MU0
from fem.air56.loaded_section import (
    H_OF_B,
    STEEL_B_T,
    loaded_probe,
    material,
    magnetic_force,
    tensor_matrix,
)
from fem.air56.sliding_section import make_section, winding_rhs
from tools.run_air56_skew_audit import article_profile


def solve_bias(geometry, section, magnet, br, angle, bias):
    basis = Basis(section.mesh, ElementTriP1(), intorder=2)
    free = basis.complement_dofs(basis.get_dofs())
    # B = (dA/dy, -dA/dx). Both opposed pockets have the same rotor-frame
    # magnetization vector, giving opposite outward radial polarities.
    hm = br / MU0 * magnet[:, None]
    source = magnetic_force.assemble(basis, hx=-np.sin(angle) * hm, hy=np.cos(angle) * hm)
    rhs = (source + winding_rhs(geometry, section, basis) @ bias)[free]
    norm = np.linalg.norm(rhs)
    values = np.zeros(basis.N)

    def state(potential):
        gx, gy = basis.interpolate(potential).grad[..., 0]
        nu, tangent = material(gx, gy, section.steel)
        force = magnetic_force.assemble(basis, hx=(gx * nu)[:, None], hy=(gy * nu)[:, None])
        return force[free] - rhs, tangent

    for _ in range(40):
        residual, tangent = state(values)
        error = np.linalg.norm(residual)
        if error < norm * 1e-8:
            break
        change = splu(tensor_matrix(basis, tangent)[free][:, free].tocsc()).solve(-residual)
        for damping in 2.0 ** -np.arange(15):
            trial = values.copy()
            trial[free] += damping * change
            if np.linalg.norm(state(trial)[0]) < error:
                values = trial
                break
        else:
            raise RuntimeError("nonlinear line search")
    else:
        raise RuntimeError("nonlinear convergence")
    z, res, reciprocity = loaded_probe(
        geometry, section, {"tangent": [t[:, None] for t in tangent]}, 100.0, 27.5 / 3
    )
    gradient = basis.interpolate(values).grad[..., 0]
    magnitude = np.hypot(*gradient)
    limit = STEEL_B_T[-1]
    excess = np.maximum(magnitude - limit, 0.0)
    steel_energy = (
        H_OF_B.antiderivative()(np.minimum(magnitude, limit))
        + H_OF_B(limit) * excess
        + excess**2 / (2 * MU0)
    )
    energy = np.where(section.steel, steel_energy, magnitude**2 / (2 * MU0))
    coenergy = geometry.stack_length * (rhs @ values[free] - energy @ basis.dx.sum(axis=1))
    return z, max(error / norm, res, reciprocity), coenergy


def run(level, br, out, count, background):
    geometry, profile = article_profile()
    shift = 0.5 if level > 4 else 0.0
    angles = (np.arange(count) + shift) * 2 * np.pi / count
    responses, errors, areas, energies = [], [], [], []
    for angle in angles:
        section = make_section(
            geometry, angle, level, replace(profile, bar_width_modulation=0.6), int(shift > 0)
        )
        x, y = section.mesh.p[:, section.mesh.t].mean(axis=1)
        r = np.hypot(x, y)
        local = (np.arctan2(y, x) - angle + np.pi / 2) % np.pi - np.pi / 2
        mask = (
            section.steel
            & (abs(local) < np.deg2rad(8))
            & (r > geometry.rotor_radius - 0.0005)
            & (r < geometry.rotor_radius)
        )
        section.steel &= ~mask
        areas.append(Basis(section.mesh, ElementTriP1()).dx[mask].sum())
        measurements = []
        for bias in (0.8 * np.eye(2)[0], -0.8 * np.eye(2)[0], 0.8 * np.eye(2)[1], -0.8 * np.eye(2)[1]):
            z, error, energy = solve_bias(
                geometry, section, mask, br, angle, bias + background * np.array([0.8, 0.6])
            )
            measurements.append(z)
            errors.append(error)
            energies.append(energy)
        responses.append(measurements)
    if max(errors) > 1e-7 or min(areas) <= 0:
        raise RuntimeError("FEM controls failed")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        angles=angles,
        z=responses,
        area_m2=areas,
        max_error=max(errors),
        br=br,
        level=level,
        coenergy_j=np.reshape(energies, (-1, 4)),
        background_a=background,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--br", type=float, required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--background", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.level, args.br, args.out, args.count, args.background)

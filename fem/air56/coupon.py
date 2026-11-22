"""Time-harmonic local eddy currents in laminated steel, e^(i omega t)."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, ElementTetN0, MeshTet
from skfem.helpers import curl, dot


MU0 = 4e-7 * np.pi
LENGTH = 1e-3  # mesh coordinates in millimetres
AIR_SIGMA = 0.5  # divergence gauge for otherwise insulating cells


@dataclass(frozen=True)
class Coupon:
    sheets: int = 2
    thickness_mm: float = 0.35
    gap_mm: float = 0.04
    width_mm: float = 0.70
    mu_r: float = 200
    sigma_20: float = 2e6
    temperature_c: float = 20
    reference_temperature_coefficient: float = 0.003
    contact_positions_mm: tuple = (-3.55, 3.55)
    radial_contact_mm: float = 0.0
    cage_return: bool = False
    radial_half_extent_mm: float = 3.0
    tangential_half_extent_mm: float = 6.0


@BilinearForm(dtype=np.complex128)
def eddy(e, v, w):
    return w.nu * dot(curl(e), curl(v)) + w.electric * dot(e, v)


def _axis(splits, spacing):
    return np.concatenate([
        np.linspace(lo, hi, max(1, int(np.ceil((hi - lo) / spacing))) + 1)[:-1]
        for lo, hi in zip(splits[:-1], splits[1:])
    ] + [np.array(splits[-1:])])


def make_problem(cfg, refinement, padding_mm=3.0, air_sigma=AIR_SIGMA):
    """A conforming voxel partition tetrahedralized across every sheet/contact."""
    starts = (np.arange(cfg.sheets) - (cfg.sheets - 1) / 2) * (cfg.thickness_mm + cfg.gap_mm)
    faces = sorted(np.r_[starts - cfg.thickness_mm / 2, starts + cfg.thickness_mm / 2])
    half = cfg.width_mm / 2
    r = cfg.radial_contact_mm
    rx, ty = cfg.radial_half_extent_mm, cfg.tangential_half_extent_mm
    if abs(r)+half >= rx or any(abs(pos)+half >= ty for pos in cfg.contact_positions_mm):
        raise ValueError("contact footprint must fit inside the coupon")
    x = _axis(sorted({-rx-padding_mm, -rx, r-half, r+half, rx, rx+padding_mm}), 3/refinement)
    y = _axis(sorted({-ty-padding_mm, -ty, ty, ty+padding_mm} |
                     {p + sign*half for p, sign in product(cfg.contact_positions_mm, (-1, 1))}), 3/refinement)
    z = np.r_[_axis([faces[0]-padding_mm, faces[0]], 3/refinement)[:-1],
              _axis(faces, cfg.thickness_mm/refinement),
              _axis([faces[-1], faces[-1]+padding_mm], 3/refinement)[1:]]
    mesh = MeshTet.init_tensor(x, y, z)
    basis = Basis(mesh, ElementTetN0(), intorder=2)
    centers = mesh.p[:, mesh.t].mean(axis=1)
    inside = (abs(centers[0]) < rx) & (abs(centers[1]) < ty)
    steel = inside & np.any(abs(centers[2, :, None] - starts) < cfg.thickness_mm/2, axis=1)
    cage = (~steel & (centers[0] > r-half) & (centers[0] < r+half)
            & (centers[1] > ty) & (centers[1] < ty+padding_mm)
            & (abs(centers[2]) < faces[-1])) if cfg.cage_return else np.zeros_like(steel)
    masks = tuple((abs(centers[0]-r) < half) & (abs(centers[1]-y_pos) < half)
                  & (abs(centers[2]-z_pos) < cfg.gap_mm/2)
                  for z_pos, y_pos in product((starts[:-1]+starts[1:])/2, cfg.contact_positions_mm))
    steel_sigma = cfg.sigma_20 / (1 + cfg.reference_temperature_coefficient*(cfg.temperature_c-20))
    conductivity = np.where(steel, steel_sigma, np.where(cage, 28e6, air_sigma))
    reluctivity = np.where(steel, 1/cfg.mu_r, 1.0)
    boundary = basis.get_dofs().all()
    free = np.setdiff1d(np.arange(basis.N), boundary)
    edge_start, edge_end = mesh.p[:, mesh.edges[0]], mesh.p[:, mesh.edges[1]]
    midpoint, delta = ((edge_start + edge_end) / 2).T, (edge_end - edge_start).T
    imposed = (0.5 * np.cross(midpoint, delta))[:, :2]
    return basis, masks, steel, cage, conductivity, reluctivity, boundary, free, imposed


def solve_coupon(problem, frequency_hz, contacts=(), resistance_ohm=1e-4, cfg=Coupon()):
    basis, masks, steel, cage, conductivity, nu, boundary, free, imposed = problem
    sigma = conductivity.copy()
    contact = np.zeros(steel.shape, dtype=bool)
    for idx in contacts:
        contact |= masks[idx]
    sigma[contact] = cfg.gap_mm*LENGTH / (resistance_ohm * (cfg.width_mm*LENGTH)**2)
    omega = 2*np.pi*frequency_hz
    matrix = eddy.assemble(basis, nu=nu[:, None],
                           electric=(1j*omega*MU0*LENGTH**2*sigma)[:, None]).tocsc()
    values = np.zeros_like(imposed, dtype=complex)
    values[boundary] = imposed[boundary]
    inner = matrix[free][:, free].tocsc()
    rhs = -matrix[free][:, boundary] @ values[boundary]
    values[free] = splu(inner).solve(rhs)
    residual = np.linalg.norm(inner @ values[free] - rhs) / np.linalg.norm(rhs)
    fields = [basis.interpolate(values[:, idx]) for idx in range(2)]
    electric_v_m = -1j*omega*LENGTH * np.array(fields)
    flux_t = np.array([field.curl for field in fields])
    density = 0.5*sigma[None, :, None]*np.sum(abs(electric_v_m)**2, axis=1)
    dv = basis.dx * LENGTH**3
    loss = lambda mask: np.sum(density[:, mask] * dv[mask], axis=(1, 2))
    complex_energy = values.T @ (matrix @ values)
    impedance = -1j*omega*LENGTH**3 / MU0 * complex_energy
    materials = steel.astype(np.int32) + 2 * contact + 3 * cage
    return dict(z=impedance, residual=residual, loss_w=loss(steel | contact | cage),
                air_loss_w=loss(~(steel | contact | cage)), steel_loss_w=loss(steel),
                contact_loss_w=loss(contact), cage_loss_w=loss(cage),
                cell_b_t=flux_t.mean(axis=-1).transpose(0, 2, 1),
                cell_j_a_m2=electric_v_m.mean(axis=-1).transpose(0, 2, 1)*sigma[None, :, None],
                cell_loss_w_m3=(density*dv).sum(axis=-1)/dv.sum(axis=-1),
                dofs=values, materials=materials, sigma=sigma, field_at_q=electric_v_m)


def contrast(problem, control, modified):
    """Reciprocity contrast integral without subtracting the air energy."""
    basis = problem[0]
    w = (modified["sigma"] - control["sigma"])[:, None] * basis.dx * LENGTH**3
    # E = -i omega A. Both fields are divided by -i here, without conjugation.
    return -np.einsum("aijk,bijk,jk->ab", control["field_at_q"], modified["field_at_q"], w)


def normalize_field(problem, control):
    """Map prescribed mean in-sheet B to the imposed uniform-field basis."""
    steel = problem[2]
    volumes = problem[0].dx.sum(axis=1)[steel]
    mean = np.average(control["cell_b_t"][:, steel, :2], axis=1, weights=volumes)
    return np.linalg.inv(mean.T)

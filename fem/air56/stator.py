"""2-D magnetoquasistatic machine cross-section and reciprocity transfer."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, ElementTriP1, LinearForm, MeshTri
from skfem.helpers import dot, grad

from fem.air56.coupon import MU0


@dataclass(frozen=True)
class Geometry:
    shaft_radius: float
    rotor_radius: float
    air_gap: float
    stator_radius: float
    stack_length: float
    stator_slot_depth: float
    rotor_bar_depth: float
    stator_slots: int
    rotor_bars: int
    turns: int
    mu_r: float
    aluminium_sigma: float
    stator_slot_fraction: float
    rotor_bar_fraction: float

    @classmethod
    def from_config(cls, config):
        dimensions = config["dimensions_mm"]
        return cls(
            **{key: dimensions[key][1] / 1000 for key in (
                "shaft_radius", "rotor_radius", "air_gap", "stator_radius", "stack_length",
                "stator_slot_depth", "rotor_bar_depth")},
            stator_slots=config["stator_slots"][1], rotor_bars=config["rotor_bars"][1],
            turns=config["equivalent_turns"][1], mu_r=config["steel_mu_r_incremental"][1],
            aluminium_sigma=config["aluminium_sigma_s_m"],
            stator_slot_fraction=config["stator_slot_fraction"],
            rotor_bar_fraction=config["rotor_bar_fraction"],
        )


@BilinearForm(dtype=np.complex128)
def magnetic(a, v, w):
    return w.nu*dot(grad(a), grad(v)) + w.electric*a*v


@LinearForm
def current(v, w):
    return w.source*v


def polar_mesh(g, ntheta, radial_refinement):
    bore = g.rotor_radius + g.air_gap
    breaks = (0, g.shaft_radius, g.rotor_radius-g.rotor_bar_depth, g.rotor_radius,
              bore, bore+g.stator_slot_depth, g.stator_radius, 1.16*g.stator_radius)
    levels = np.concatenate([np.linspace(a, b, n*radial_refinement+1)[:-1]
                             for a, b, n in zip(breaks, breaks[1:], (3, 12, 4, 3, 8, 8, 5))]
                            + [np.array(breaks[-1:])])[1:]
    theta = np.arange(ntheta)*2*np.pi/ntheta
    points = np.c_[np.zeros(2), np.outer(levels, np.array([np.cos(theta), np.sin(theta)])).reshape(-1, 2*ntheta)
                   .reshape(-1, 2, ntheta).transpose(1, 0, 2).reshape(2, -1)]
    rings = 1 + np.arange(len(levels)*ntheta).reshape(-1, ntheta)
    nxt = np.roll(rings, -1, axis=1)
    triangles = np.column_stack((
        np.vstack((np.zeros(ntheta, dtype=int), rings[0], nxt[0])),
        np.vstack((rings[:-1].ravel(), rings[1:].ravel(), nxt[1:].ravel())),
        np.vstack((rings[:-1].ravel(), nxt[1:].ravel(), nxt[:-1].ravel())),
    ))
    radii_by_layer = np.r_[levels[0]/2, (levels[:-1]+levels[1:])/2]
    cell_radius = np.r_[np.repeat(radii_by_layer[0], ntheta),
                        np.tile(np.repeat(radii_by_layer[1:], ntheta), 2)]
    return MeshTri(np.ascontiguousarray(points), np.ascontiguousarray(triangles)), cell_radius


def solve_section(g, frequency_hz, pole_pairs=1, *, radial_refinement=1, ntheta=336,
                  slotted=False, slip=1.0, rotor_angle_rad=0.0, sample_depth_m=0.002):
    mesh, radial = polar_mesh(g, ntheta, radial_refinement)
    basis = Basis(mesh, ElementTriP1(), intorder=3)
    x, y = mesh.p[:, mesh.t].mean(axis=1)
    theta = np.arctan2(y, x)
    bore = g.rotor_radius + g.air_gap
    winding_ring = (radial > bore) & (radial < bore+g.stator_slot_depth)
    rotor_shell = (radial < g.rotor_radius) & (radial > g.rotor_radius-g.rotor_bar_depth)
    rotor = (radial < g.rotor_radius) & (radial > g.shaft_radius)
    stator = (radial > bore) & (radial < g.stator_radius)
    in_slot = lambda count, angle, fraction: abs((count*angle/(2*np.pi)+0.5) % 1-0.5) < fraction/2
    slots = winding_ring & in_slot(g.stator_slots, theta, g.stator_slot_fraction)
    bars = rotor_shell & in_slot(g.rotor_bars, theta-rotor_angle_rad, g.rotor_bar_fraction)
    if slotted:
        mu = np.where((rotor & ~bars) | (stator & ~slots), g.mu_r, 1.0)
        sigma = np.where(bars, g.aluminium_sigma, 0.0)
        winding = slots
    else:
        mu = np.select([winding_ring, rotor | stator], [4.0, g.mu_r], default=1.0)
        sigma = np.where(rotor_shell, g.aluminium_sigma*g.rotor_bar_fraction, 0.0)
        bars = rotor_shell
        winding = winding_ring
    omega = 2*np.pi*frequency_hz
    areas = basis.dx.sum(axis=1)
    qx, qy = basis.global_coordinates()
    phase = pole_pairs*np.arctan2(qy, qx)
    strength = 2*g.turns / (areas[winding].sum()*2/np.pi)
    density = strength*winding[:, None]*np.array([np.cos(phase), np.sin(phase)])
    rhs = np.column_stack([current.assemble(basis, source=source) for source in density])
    K = magnetic.assemble(basis, nu=(1/(MU0*mu))[:, None], electric=(1j*omega*slip*sigma)[:, None])
    free = basis.complement_dofs(basis.get_dofs())
    solver = splu(K[free][:, free].tocsc())
    values_free = solver.solve(rhs[free].astype(complex))
    offsets = np.zeros(2, dtype=complex)
    if slotted and slip > 0:
        # Floating, ideal end rings enforce zero net axial current. A scalar
        # Schur complement avoids a dense graph edge spanning all cage bars.
        conductance = current.assemble(basis, source=np.broadcast_to(sigma[:, None], basis.dx.shape))
        b = 1j*omega*slip*conductance[free]
        s = 1j*omega*slip*np.dot(sigma, areas)
        response = solver.solve(b)
        offsets = (b @ values_free)/(s-b @ response)
        values_free += response[:, None]*offsets
        values = np.zeros((basis.N, 2), dtype=complex)
    values[free] = values_free
    fields = [basis.interpolate(values[:, idx]) for idx in range(2)]
    cage_loss = 0.5*g.stack_length*(omega*slip)**2*np.array([
        np.sum(sigma[:, None]*abs(field-offset)**2*basis.dx)
        for field, offset in zip(fields, offsets)])
    z = 1j*omega*g.stack_length*(rhs.T @ values)
    sample_angle = (np.arange(360)+0.39)*2*np.pi/360
    radial_dir = np.array([np.cos(sample_angle), np.sin(sample_angle)])
    tangential = np.array([-radial_dir[1], radial_dir[0]])
    radius = g.rotor_radius-sample_depth_m
    # Evaluate the piecewise-constant grad(A) on the sample circle.
    elements = mesh.element_finder()(radius*radial_dir[0], radius*radial_dir[1])
    flux = np.array([[f.grad[1, elements, 0], -f.grad[0, elements, 0]] for f in fields])
    transfer = np.stack((np.einsum("acn,cn->na", flux, radial_dir),
                         np.einsum("acn,cn->na", flux, tangential)), axis=1)
    raw_transfer = transfer.copy()
    if not slotted:
        # The smooth continuum has exactly p pole pairs. Remove mesh angular
        # imprint rather than misreporting it as absolute rotor-angle evidence.
        harmonics = np.column_stack((np.cos(pole_pairs*sample_angle), np.sin(pole_pairs*sample_angle)))
        coefficient = np.linalg.lstsq(harmonics, transfer.reshape(len(transfer), 4), rcond=None)[0]
        theta_out = np.arange(360)*2*np.pi/360
        modes = np.column_stack((np.cos(pole_pairs*theta_out), np.sin(pole_pairs*theta_out)))
        transfer = (modes @ coefficient).reshape(-1, 2, 2)
    result = dict(z=z, cage_loss_w=cage_loss, transfer=transfer, field_t=flux,
                  mesh=mesh, values=values, cage=bars, sample_angle=sample_angle,
                  mesh_angular_error=float(np.linalg.norm(raw_transfer - (
                      (harmonics @ coefficient).reshape(-1, 2, 2) if not slotted else raw_transfer))
                      / np.linalg.norm(raw_transfer)))
    if 0 < slip < 1:
        result["torque_nm_per_a_peak_squared"] = pole_pairs*cage_loss[0] / (slip*omega)
    return result

"""Conforming slotted AIR56 section with independently rotating meshes."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import splu
from skfem import Basis, ElementTriP2, MeshTri

from fem.air56.coupon import MU0
from fem.air56.stator import current, magnetic


TAU = 2*np.pi


@dataclass(frozen=True)
class SlotDesign:
    opening_fraction: float = 0.13
    opening_depth_m: float = 0.0005
    modulation: float = 0.0
    # The opening profile is mechanically opposed. The bar bodies are fixed.
    modulation_order: int = 2
    surface_recess_m: float = 0.0
    bar_width_modulation: float = 0.0


@dataclass
class Section:
    mesh: MeshTri
    cage: np.ndarray
    steel: np.ndarray
    winding_slot: np.ndarray
    stator_boundary: np.ndarray
    transfer_points: np.ndarray
    rotor_angle: float


def angular_grid(count, fractions, level):
    """Split each slot pitch at all material interfaces."""
    center = (np.arange(count)+0.5)*TAU/count
    pitch = TAU/count
    theta = []
    for i, c in enumerate(center):
        splits = sorted({0., 1., *[v for f in fractions for v in
                                   ((1-f[i])/2, (1+f[i])/2)]})
        for left, right in zip(splits[:-1], splits[1:]):
            n = max(1, int(np.ceil((right-left)*level)))
            theta.extend(c+pitch*(np.linspace(left, right, n+1)[:-1]-0.5))
    return np.array(theta)


def radial_grid(breaks, subdivisions):
    bands = []
    labels = []
    for i, (lo, hi, count) in enumerate(zip(breaks, breaks[1:], subdivisions)):
        bands.extend(np.linspace(lo, hi, count+1)[:-1])
        labels.extend([i]*count)
    return np.r_[bands, breaks[-1]], np.array(labels)


def make_section(g, angle, level, design=SlotDesign(), gap_phase=0,
                 stator_opening_fraction=.16, stator_opening_depth_m=.00065):
    ns, nr = g.stator_slots, g.rotor_bars
    if ns % 6:
        raise ValueError("the double-sided phase-belt winding requires a multiple of six slots")
    if not 0 <= design.modulation < 1 or not 0 < design.opening_depth_m < g.rotor_bar_depth:
        raise ValueError("invalid rotor slot design")
    if not 0 <= design.surface_recess_m < design.opening_depth_m:
        raise ValueError("surface recess must stay outside the bar region")
    bore = g.rotor_radius+g.air_gap
    centers = (np.arange(nr)+0.5)*TAU/nr
    opening = design.opening_fraction*(1+design.modulation*np.cos(
        design.modulation_order*centers))
    bar_width = g.rotor_bar_fraction*(1+design.bar_width_modulation*np.cos(2*centers))
    if np.any(bar_width <= 0) or np.any(bar_width >= 1):
        raise ValueError("bar width must lie within one rotor pitch")
    if np.any(opening < 0) or np.any(opening >= bar_width):
        raise ValueError("slot opening must be nonnegative and smaller than its bar body")
    rt = angular_grid(nr, (opening, bar_width), level)
    if not 0 < stator_opening_fraction <= g.stator_slot_fraction:
        raise ValueError("stator opening must fit inside its winding slot")
    st = angular_grid(ns, (np.full(ns, g.stator_slot_fraction),
                          np.full(ns, stator_opening_fraction)), level)
    rr, rb = radial_grid((g.shaft_radius, g.rotor_radius-g.rotor_bar_depth,
                          g.rotor_radius-design.opening_depth_m, g.rotor_radius),
                         (max(2, level//2), level, max(2, level//3)))
    if not 0 < stator_opening_depth_m < g.stator_slot_depth:
        raise ValueError("stator neck depth must fit inside its winding slot")
    rs, sb = radial_grid((bore, bore+stator_opening_depth_m, bore+g.stator_slot_depth,
                          g.stator_radius, 1.15*g.stator_radius),
                         (max(2, level//3), level, max(2, level//2), max(2, level//2)))
    vertices, triangles, cage, steel, winding_slots = [[0., 0.]], [], [], [], []

    # The alpha-beta port is a distributed single-layer, full-pitched phase-belt prior.
    def add_tri(cells, c, m, s):
        triangles.extend(cells.T)
        cage.extend(c)
        steel.extend(m)
        winding_slots.extend(s)

    def add_block(radii, bands, theta, rotation, rotor):
        ntheta = len(theta)
        ids = len(vertices)+np.arange(len(radii)*ntheta).reshape(-1, ntheta)
        unit = np.column_stack((np.cos(theta+rotation), np.sin(theta+rotation)))
        radial_nodes = radii[:, None]
        if rotor and design.surface_recess_m:
            # Move only the crown between the bar roof and the rotor surface.
            crown = np.clip((radii-g.rotor_radius+design.opening_depth_m)/design.opening_depth_m, 0., 1.)
            recess = design.surface_recess_m*np.sin(theta)**2
            radial_nodes = radial_nodes-crown[:, None]*recess
        vertices.extend((radial_nodes[..., None]*unit[None]).reshape(-1, 2))
        next_ids = np.roll(ids, -1, axis=1)
        middle = (theta+np.r_[theta[1:], theta[0]+TAU])/2
        count = nr if rotor else ns
        q = middle/TAU*count
        slot = q.astype(int).clip(0, count-1)
        distance = abs(q-slot-0.5)
        if rotor:
            bar = distance < bar_width[slot]/2
            neck = distance < opening[slot]/2
            conducting = (bands[:, None] == 1) & bar
            material = ~(conducting | ((bands[:, None] == 2) & neck))
            winding = np.full_like(material, -1, dtype=int)
            add_tri(np.vstack((np.zeros(ntheta, dtype=int), ids[0], next_ids[0])),
                    np.zeros(ntheta, bool), np.zeros(ntheta, bool), np.full(ntheta, -1))
        else:
            coil = (bands[:, None] == 1) & (distance < g.stator_slot_fraction/2)
            neck = (bands[:, None] == 0) & (distance < stator_opening_fraction/2)
            material = (bands[:, None] != 3) & ~(coil | neck)
            winding = np.where(coil, slot, -1)
            conducting = np.zeros_like(material)
        add_tri(np.column_stack((
                    np.vstack((ids[:-1].ravel(), ids[1:].ravel(), next_ids[1:].ravel())),
                    np.vstack((ids[:-1].ravel(), next_ids[1:].ravel(), next_ids[:-1].ravel())))),
                np.tile(conducting.ravel(), 2), np.tile(material.ravel(), 2), np.tile(winding.ravel(), 2))
        return ids

    ir = add_block(rr, rb, rt, angle, True)
    iss = add_block(rs, sb, st, 0., False)
    inner = (rt+angle) % TAU
    outer = st % TAU
    # Zipper triangulation of the air gap. Reversing the tie break is a
    # second remeshing pattern at geometrically coincident boundary rays.
    events = [(theta, 0, int(node)) for theta, node in zip(inner, ir[-1])]
    events += [(theta, 1, int(node)) for theta, node in zip(outer, iss[0])]
    events.sort(key=lambda e: (round(e[0], 13), e[1] ^ gap_phase))
    last = [int(ir[-1][np.argmax(inner)]), int(iss[0][np.argmax(outer)])]
    gap = []
    for _, side, node in events:
        gap.append([last[0], last[1], node])
        last[side] = node
    size = len(gap)
    add_tri(np.array(gap).T, np.zeros(size, bool), np.zeros(size, bool), np.full(size, -1))
    points = np.array(vertices).T
    mesh = MeshTri(np.ascontiguousarray(points), np.ascontiguousarray(np.array(triangles).T))
    # A tooth lies between slot centers. The bridge has its own local, narrow footprint.
    offset = np.array([-0.78, 0, 0.78])*1e-3/g.rotor_radius
    transfer_points = (g.rotor_radius-0.00043)*np.vstack((np.cos(angle+offset), np.sin(angle+offset)))
    return Section(mesh, np.array(cage), np.array(steel), np.array(winding_slots), iss[-1], transfer_points, angle)


def solve(g, frequency, section, stator_resistance, *, transfer=False, current_source="belts"):
    mesh = section.mesh
    basis = Basis(mesh, ElementTriP2(), intorder=4)
    sigma = np.where(section.cage, g.aluminium_sigma, 0.0)
    nu = 1/(MU0*np.where(section.steel, g.mu_r, 1.0))
    w = TAU*frequency
    areas = basis.dx.sum(axis=1)
    slot = section.winding_slot
    valid = slot >= 0
    phase_belts = np.arange(g.stator_slots)*6//g.stator_slots
    if current_source == "continuous":
        x, y = basis.global_coordinates()
        phase = np.arctan2(y, x)
        density = (np.pi*g.turns/areas[valid].sum())*valid[:, None]*np.array([np.cos(phase), np.sin(phase)])
    elif current_source == "belts":
        direction = np.column_stack((np.cos(phase_belts*TAU/6), np.sin(phase_belts*TAU/6)))
        n_per_belt = g.stator_slots//6
        amp = g.turns/n_per_belt/np.bincount(slot[valid], weights=areas[valid], minlength=g.stator_slots)
        density = np.where(valid[:, None], direction[slot]*amp[slot, None], 0.).T[:, :, None]
    else:
        raise ValueError(f"unknown winding {current_source}")
    rhs = np.column_stack([current.assemble(basis, source=u) for u in density])
    matrix = magnetic.assemble(basis, nu=nu[:, None], electric=(1j*w*sigma)[:, None])
    free = basis.complement_dofs(basis.get_dofs())
    inner = matrix[free][:, free].tocsc()
    b = 1j*w*current.assemble(basis, source=sigma[:, None])[free]
    s = 1j*w*np.dot(sigma, areas)
    inv = splu(inner).solve(np.column_stack((rhs[free], b)).astype(complex))
    potential, u = inv[:, :2].copy(), inv[:, 2]
    floating = (b @ potential)/(s-b @ u)
    potential += u[:, None]*floating
    a = np.zeros((basis.N, 2), dtype=complex)
    a[free] = potential
    z_field = (2./3)*1j*w*g.stack_length*(rhs.T @ a)
    loss = 0.5*g.stack_length*w*w*np.array([
        np.sum(sigma[:, None]*abs(basis.interpolate(a[:, i])-floating[i])**2*basis.dx) for i in range(2)])
    res = np.linalg.norm(inner @ potential-b[:, None]*floating-rhs[free])/np.linalg.norm(rhs[free])
    out = {"z": z_field+stator_resistance*np.eye(2), "loss": loss, "residual": float(res),
           "energy": float(np.linalg.norm(0.75*z_field.real.diagonal()-loss)/np.linalg.norm(loss)),
           "reciprocity": float(np.linalg.norm(z_field-z_field.T)/np.linalg.norm(z_field)), "dofs": basis.N}
    if transfer:
        # Differentiate the P2 interpolant at three sample points, and average
        # in the fixed coordinate frame of the narrow rotor coupon.
        q = section.transfer_points
        h = 2e-7
        gradients = [(basis.probes(q+d[:, None]*h) @ a
                     -basis.probes(q-d[:, None]*h) @ a)/(2*h)
                     for d in np.eye(2)]
        bx, by = gradients[1].mean(axis=0), -gradients[0].mean(axis=0)
        c, sn = np.cos(section.rotor_angle), np.sin(section.rotor_angle)
        out["transfer"] = np.array([c*bx+sn*by, -sn*bx+c*by])
    return out

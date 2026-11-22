"""Radial P2 FEM of the exact 2D Fourier mode of the smooth reference motor."""

import numpy as np
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, ElementLineP2, LinearForm, MeshLine
from skfem.helpers import grad

from fem.air56.coupon import MU0


@BilinearForm(dtype=np.complex128)
def modal(a, v, w):
    r = w.x[0]
    return (r*w.nu*grad(a)[0]*grad(v)[0] + (w.nu*w.poles**2/r + w.mass*r)*a*v)


@LinearForm
def forcing(v, w):
    return w.x[0]*w.current*v


def solve_mode(g, frequency_hz, pole_pairs=1, skin_refinement=5, sample_depths_m=(0.002,)):
    omega = 2*np.pi*frequency_hz
    bore = g.rotor_radius+g.air_gap
    shell = g.rotor_radius-g.rotor_bar_depth
    cond = g.aluminium_sigma*g.rotor_bar_fraction
    skin = np.sqrt(2/(omega*MU0*g.mu_r*cond))
    ring_limit = bore+g.stator_slot_depth
    intervals = [(0, g.shaft_radius, 1, 0), (g.shaft_radius, shell, g.mu_r, 0),
                 (shell, g.rotor_radius, g.mu_r, cond), (g.rotor_radius, bore, 1, 0),
                 (bore, ring_limit, 4, 0), (ring_limit, g.stator_radius, g.mu_r, 0),
                 (g.stator_radius, g.stator_radius*1.16, 1, 0)]
    edges, mu, sigma, charge = [], [], [], []
    for idx, (left, right, material_mu, material_sigma) in enumerate(intervals):
        h = skin/skin_refinement if material_sigma else 1e-3/np.sqrt(skin_refinement)
        n = max(2, int(np.ceil((right-left)/h)))
        edges.extend(np.linspace(left, right, n+1)[:-1])
        mu.extend([material_mu]*n)
        sigma.extend([material_sigma]*n)
        charge.extend([int(idx == 4)]*n)
    edges.append(intervals[-1][1])
    mesh = MeshLine(np.array(edges))
    basis = Basis(mesh, ElementLineP2(), intorder=5)
    rhs = forcing.assemble(basis, current=np.array(charge)[:, None]
                          * g.turns/(ring_limit**2-bore**2))
    K = modal.assemble(basis, nu=(1/(MU0*np.array(mu)))[:, None],
                       mass=(1j*omega*np.array(sigma))[:, None], poles=pole_pairs)
    free = basis.complement_dofs(basis.get_dofs())
    result = np.zeros(basis.N, dtype=complex)
    result[free] = splu(K[free][:, free].tocsc()).solve(rhs[free].astype(complex))
    impedance = 1j*omega*g.stack_length*np.pi*np.dot(rhs, result)
    theta = np.arange(360)*2*np.pi/360
    sin, cos = np.sin(pole_pairs*theta), np.cos(pole_pairs*theta)
    curves = {}
    for depth in sample_depths_m:
        r = g.rotor_radius-depth
        h = min(1e-7, skin/100)
        samples = basis.probes(np.array([[r-h, r, r+h]])) @ result
        value, derivative = samples[1], (samples[2]-samples[0])/(2*h)
        curves[depth] = np.array([[-sin*pole_pairs*value/r, cos*pole_pairs*value/r],
                                  [-cos*derivative, -sin*derivative]]).transpose(2, 0, 1)
    return {"z": impedance*np.eye(2), "curves": curves, "transfer": curves[sample_depths_m[-1]],
            "skin_depth_mm": 1e3*skin, "radial_elements": mesh.nelements}

"""Steady coupon temperature for the explicitly prescribed heat-sink scenario."""

import numpy as np
from skfem import Basis, BilinearForm, ElementTetP1, LinearForm, condense, solve
from skfem.helpers import dot, grad


@BilinearForm
def conductivity(u, v, w):
    return w.k*dot(grad(u), grad(v))


@LinearForm
def source(v, w):
    return w.q*v


def steady_temperature(mesh, material, loss_w_m3, ambient_c=20, contact_k=15):
    centers = mesh.p[:, mesh.t].mean(axis=1)
    # Stay inside the coupon plus any cage return. Outer surfaces are insulating.
    internal = (abs(centers[0]) < 3) & (abs(centers[1]) < 6)
    indices = np.flatnonzero(internal | (material == 3))
    submesh, vertices = mesh.restrict(indices, return_mapping=True)
    basis = Basis(submesh, ElementTetP1())
    k = np.choose(material[indices], [0.2, 22.0, contact_k, 160.0])
    matrix = conductivity.assemble(basis, k=k[:, None])
    heat = source.assemble(basis, q=loss_w_m3[indices, None]*1e-6)
    ends = basis.get_dofs(lambda x: np.isclose(abs(x[0]), 3)).all()
    delta = solve(*condense(matrix, heat, D=ends))
    balance = float(-np.sum((matrix @ delta - heat)[ends]) / np.sum(heat))
    temperature = np.full(mesh.nvertices, np.nan)
    temperature[vertices] = ambient_c+delta
    return temperature, balance

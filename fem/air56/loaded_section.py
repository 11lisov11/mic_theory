"""Nonlinear fundamental field and cycle-averaged incremental carrier response."""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, LinearForm
from skfem.helpers import grad

from fem.air56.coupon import MU0
from fem.air56.sliding_section import TAU, winding_rhs
from fem.air56.stator import current


# M400-50A, table 3 in the published motor model. Above 1.8 T the
# analyst-specified extension gradually approaches the air reluctivity.
STEEL_B_T = np.r_[np.arange(19)/10, 1.9, 2.0, 2.1, 2.3]
STEEL_H_A_M = np.array([0, 32.6, 43.5, 50.8, 57.2, 63.4, 69.9, 77.3, 86.0, 97.2,
                       113.2, 137.8, 180.2, 269.5, 516.8, 1307, 3180, 6361, 10890,
                       17000, 31000, 90000, 249000])
H_OF_B = PchipInterpolator(STEEL_B_T, STEEL_H_A_M)
DH_DB = H_OF_B.derivative()


@BilinearForm
def stiffness(a, v, w):
    ax, ay = grad(a)
    vx, vy = grad(v)
    return w.xx*ax*vx + w.xy*(ax*vy+ay*vx) + w.yy*ay*vy


@BilinearForm
def mass(a, v, w):
    return w.sigma*a*v


@LinearForm
def magnetic_force(v, w):
    vx, vy = grad(v)
    return w.hx*vx+w.hy*vy


def material(gx, gy, steel):
    """Secant H/B and differential coefficients in the gradient frame."""
    magnitude = np.hypot(gx, gy)
    clip = np.minimum(magnitude, STEEL_B_T[-1])
    h = H_OF_B(clip)+(magnitude-clip)/MU0
    parallel = np.where(magnitude > clip, 1/MU0, DH_DB(clip))
    secant = np.divide(h, magnitude, out=np.full_like(h, float(DH_DB(0.))), where=magnitude > 1e-12)
    secant = np.where(steel, secant, 1/MU0)
    delta = np.divide(parallel-secant, magnitude**2, out=np.zeros_like(magnitude),
                      where=steel & (magnitude > 1e-12))
    return secant, (secant+delta*gx*gx, delta*gx*gy, secant+delta*gy*gy)


def tensor_matrix(basis, coefficients):
    return stiffness.assemble(basis, **{k: v[:, None] for k, v in zip(("xx", "xy", "yy"), coefficients)})


def cage_matrices(basis, sigma, free):
    weights = mass.assemble(basis, sigma=sigma[:, None])[free][:, free].tocsc()
    integral = current.assemble(basis, source=sigma[:, None])[free]
    total = float(sigma @ basis.dx.sum(axis=1))
    return weights, integral, total


def nonlinear_fundamental(g, section, current_peak, slip_hz, *, phases=12, initial=None):
    basis = Basis(section.mesh, ElementTriP1(), intorder=2)
    free = basis.complement_dofs(basis.get_dofs())
    rhs = current_peak*(winding_rhs(g, section, basis) @ np.array([1, -1j]))[free]
    sigma = np.where(section.cage, g.aluminium_sigma, 0.)
    weights, integral, total = cage_matrices(basis, sigma, free)
    omega = TAU*slip_hz
    coupling = csc_matrix((omega*integral)[:, None])
    total_matrix = csc_matrix([[omega*total]])
    coefficients = np.stack([np.cos(np.pi*np.arange(phases)/phases), -np.sin(np.pi*np.arange(phases)/phases)])
    projections = 2*coefficients/phases
    steel = section.steel[:, None]
    dofs = len(free)

    def evaluate(values, jacobian=False):
        potential = np.zeros((2, basis.N))
        potential[:, free] = values[:2*dofs].reshape(2, dofs)
        gradients = np.stack([basis.interpolate(v).grad[..., 0] for v in potential], axis=0)
        gx, gy = gradients[:, 0].T @ coefficients, gradients[:, 1].T @ coefficients
        nu, tangent = material(gx, gy, steel)
        hx, hy = (nu*gx) @ projections.T, (nu*gy) @ projections.T
        force = [magnetic_force.assemble(basis, hx=hx[:, i, None], hy=hy[:, i, None])[free] for i in range(2)]
        x, y = potential[:, free]
        xr, yi = values[-2:]
        residual = np.r_[force[0]-omega*(weights @ y-integral*yi)-rhs.real,
                         force[1]+omega*(weights @ x-integral*xr)-rhs.imag,
                         omega*(integral @ y-total*yi), -omega*(integral @ x-total*xr)]
        if not jacobian:
            return residual, gx, gy, tangent
        aa, ab, bb = (tensor_matrix(basis, [t @ (projections[i]*coefficients[j]) for t in tangent])[free][:, free]
                      for i, j in ((0, 0), (0, 1), (1, 1)))
        jac = bmat([[aa, ab-omega*weights, None, coupling],
                    [ab+omega*weights, bb, -coupling, None],
                    [None, coupling.T, None, -total_matrix],
                    [-coupling.T, None, total_matrix, None]], format="csc")
        return residual, jac

    state = np.zeros(2*dofs+2) if initial is None else initial.copy()
    if state.shape != (2*dofs+2,):
        raise ValueError("initial state must match the mesh")
    rhs_norm = np.linalg.norm(rhs)
    for iteration in range(1, 35):
        residual, jac = evaluate(state, jacobian=True)
        error = np.linalg.norm(residual)
        if error <= 5e-8*rhs_norm:
            break
        step = splu(jac).solve(-residual)
        for damping in 2.**-np.arange(12):
            trial = state+damping*step
            trial_error = np.linalg.norm(evaluate(trial)[0])
            if trial_error < error:
                state = trial
                break
        else:
            raise RuntimeError(f"nonlinear line search stalled at relative residual {error/rhs_norm:g}")
    else:
        raise RuntimeError(f"nonlinear fundamental failed at relative residual {error/rhs_norm:g}")

    _, gx, gy, tangent = evaluate(state)
    potential = np.zeros(basis.N, dtype=complex)
    potential[free] = state[:dofs]+1j*state[dofs:2*dofs]
    floating = state[-2]+1j*state[-1]
    cage_loss = .5*g.stack_length*omega**2*np.sum(
        sigma[:, None]*abs(basis.interpolate(potential)-floating)**2*basis.dx)
    peak = np.hypot(gx, gy).max(axis=1)
    steel_area = basis.dx.sum(axis=1)[section.steel]
    return dict(state=state, tangent=tangent, iterations=iteration, residual=error/rhs_norm,
                torque_nm=float(cage_loss/omega), cage_loss_w=float(cage_loss),
                max_steel_b_t=float(peak[section.steel].max()),
                steel_area_above_knee=float(steel_area[peak[section.steel] >= 1.45].sum()/steel_area.sum()))


def loaded_probe_series(g, section, fundamental, frequency_hz, stator_resistance, phases):
    basis = Basis(section.mesh, ElementTriP2(), intorder=4)
    free = basis.complement_dofs(basis.get_dofs())
    sigma = np.where(section.cage, g.aluminium_sigma, 0.)
    weights, integral, total = cage_matrices(basis, sigma, free)
    omega = TAU*frequency_hz
    floating_rhs = 1j*omega*integral
    rhs = winding_rhs(g, section, basis)[free]
    forcing = np.column_stack((rhs, floating_rhs)).astype(complex)
    answers = []
    for phase in phases:
        tangent = tuple(t.mean(axis=1) if phase is None else t[:, phase] for t in fundamental["tangent"])
        field = tensor_matrix(basis, tangent)[free][:, free]+1j*omega*weights
        response = splu(field.tocsc()).solve(forcing)
        drive, coupling = response[:, :2], response[:, 2]
        floating = (floating_rhs @ drive)/(1j*omega*total-floating_rhs @ coupling)
        values = drive+coupling[:, None]*floating
        z = (2/3)*1j*omega*g.stack_length*(rhs.T @ values)+stator_resistance*np.eye(2)
        residual = np.linalg.norm(field @ values-floating_rhs[:, None]*floating-rhs)/np.linalg.norm(rhs)
        reciprocity = np.linalg.norm(z-z.T)/np.linalg.norm(z)
        answers.append((z, float(residual), float(reciprocity)))
    return answers


def loaded_probe(g, section, fundamental, frequency_hz, stator_resistance, *, phase=None):
    return loaded_probe_series(g, section, fundamental, frequency_hz, stator_resistance, (phase,))[0]

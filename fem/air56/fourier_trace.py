"""Fourier trace space for nonmatching P1 submeshes in a circular machine gap."""

import numpy as np
from tools.run_air56_free_field import BandField, MU0, article_profile


def trace_basis(points, modes):
    """Fit resolved trigonometric modes, retaining independent high-mode self energy."""
    angle = np.arctan2(points[1], points[0])
    phi = angle[:, None] * modes
    basis = np.column_stack((np.ones(len(angle)), np.cos(phi), np.sin(phi)))
    order = np.argsort(angle)
    widths = np.diff(np.r_[angle[order], angle[order[0]] + 2 * np.pi])
    width = (0.5 * (widths + np.roll(widths, 1)))[np.argsort(order)]
    weighted = basis * np.sqrt(width[:, None])
    fit = np.linalg.pinv(weighted) * np.sqrt(width)
    unresolved = np.eye(len(angle)) - basis @ fit
    penalty = ((modes[-1] + 1) / MU0) * (unresolved.T @ (width[:, None] * unresolved))
    coeff = fit[1 : 1 + len(modes)] - 1j * fit[1 + len(modes) :]
    return coeff, fit[0], penalty


class TraceField(BandField):
    def __init__(self, level=3, modes=24, **pole_options):
        super().__init__(0.0, level, **pole_options)
        g, _ = article_profile()
        self.orders = np.arange(1, modes + 1)
        log = np.log1p(g.air_gap / g.rotor_radius)
        self.coupling = np.pi / MU0 * self.orders / np.sinh(log * self.orders)
        diagonal = np.pi / MU0 * self.orders / np.tanh(log * self.orders)
        points = self.b.mesh.p[:, self.free[self.nodes]]
        self.inner, mi, pi = trace_basis(points[:, : self.count], self.orders)
        self.outer, mo, po = trace_basis(points[:, self.count :], self.orders)
        self.fixed[:] = 0.0
        for sl, c, penalty in (
            (slice(None, self.count), self.inner, pi),
            (slice(self.count, None), self.outer, po),
        ):
            self.fixed[sl, sl] = np.real(c.conj().T @ (diagonal[:, None] * c)) + penalty
        mean = np.r_[mi, -mo]
        self.fixed += 2 * np.pi / (MU0 * log) * np.outer(mean, mean)
        self.update(0.0)
